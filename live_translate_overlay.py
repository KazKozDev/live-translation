#!/usr/bin/env python3
"""
live_translate_overlay.py

Потоково слушает системный звук macOS через BlackHole, транскрибирует короткими
чанками через MLX Whisper, переводит локальной LLM и выводит результат в
полупрозрачное macOS-окно с frosted-glass эффектом.

Быстрый старт:

    brew install blackhole-2ch ffmpeg
    pip install mlx-whisper mlx-lm sounddevice soundfile numpy pyobjc-framework-Cocoa

    ./live_translate_overlay.py --target ru

По умолчанию перевод идёт через Ollama Gemma 4:

    brew install ollama
    ollama pull gemma4:26b-mlx
    ./live_translate_overlay.py --ollama-model gemma4:26b-mlx

Звук настраивается так же, как в record_and_transcribe.py:
Multi-Output Device = твои наушники/колонки + BlackHole 2ch.
"""

import argparse
import os
import queue
import re
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import numpy as np
import sounddevice as sd
import soundfile as sf

from live_translation.text_pipeline import (
    LANG_MENU,
    WAITING_ORIGINAL,
    WAITING_TRANSLATION,
    language_label,
    last_word_end_seconds,
    merge_overlap_text,
    merge_partial_buffer,
    punctuated_context,
    sentence_case_text,
    strip_hallucinations,
    take_blocks_for_translation,
    take_confirmed_blocks_for_translation,
    take_endpoint_blocks,
)
from live_translation.translators import LanguageSettings, OllamaTranslator

WHISPER_MODELS = {
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}

WHISPER_MENU = [
    ("small", "Whisper Small"),
    ("medium", "Whisper Medium"),
    ("large", "Whisper Large"),
]

GEMMA_MODELS = {
    "gemma4:26b-mlx": "Gemma 4 26B",
    "gemma4:e4b-mlx": "Gemma 4 E4B",
    "gemma4:12b-mlx": "Gemma 4 12B",
}

GEMMA_MENU = list(GEMMA_MODELS.items())

DEFAULT_AUDIO_QUEUE_BLOCKS = 120
STREAM_CATCHUP_HIGH_WATER = 40
STREAM_CATCHUP_KEEP_BLOCKS = 8
NO_SPEECH_KEEP_AUDIO_BLOCKS = 2
CHUNK_CATCHUP_KEEP_CHUNKS = 2
CHUNK_PRODUCER_KEEP_CHUNKS = 1
TRANSLATION_QUEUE_KEEP_TASKS = 1
SLOW_STEP_LOG_SECONDS = 6.0
STATUS_UPDATE_INTERVAL_SECONDS = 0.25
TRANSLATION_TAIL_REGROUP_SECONDS = 8.0
TRANSLATION_TAIL_MAX_MERGED_CHARS = 420
TRANSLATION_TAIL_MAX_SOURCE_MERGED_CHARS = 1400
TRANSLATION_MAX_DYNAMIC_TOKENS = 900

ELLIPSIS_END_RE = re.compile(r"(?:\.{2,}|…)\s*$")
SOFT_END_RE = re.compile(r"(?:[!?]+|(?<!\.)\.(?!\.))\s*$")


def starts_like_continuation(text):
    text = str(text or "").lstrip()
    if not text:
        return False
    if text[0] in ",;:)-—–":
        return True
    for char in text:
        if char.isalpha():
            return char.islower()
        if char.isdigit():
            return False
    return False


def translation_token_budget(text, base_tokens):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    base_tokens = int(base_tokens or 180)
    # Translation into Russian often expands versus Spanish/English. The default
    # live limit is intentionally small for latency, but polished merged paragraphs
    # need enough room to finish instead of stopping mid-word.
    estimated = int(len(text) * 0.9) + 96
    return max(base_tokens, min(TRANSLATION_MAX_DYNAMIC_TOKENS, estimated))


def find_blackhole():
    for idx, dev in enumerate(sd.query_devices()):
        dev = cast(dict, dev)  # sounddevice yields dict-like entries
        if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
            return idx, dev["name"]
    return None, None


def list_devices():
    print(sd.query_devices())
    print("\nИщи входное устройство BlackHole. Его номер можно передать через --device N.")


class ConsoleOverlay:
    def __init__(self, stop_event, show_partial=False):
        self.stop_event = stop_event
        self.show_partial = show_partial

    def post_pair(
        self,
        source,
        translated,
        pause_ms=0,
        replace_translation_tail=False,
        combined_source=None,
    ):
        print("\n--- transcript ---")
        print(source)
        print("--- translation" + (" (revised) ---" if replace_translation_tail else " ---"))
        print(translated, flush=True)

    def post_status(self, lag_chunks):
        if lag_chunks:
            print(f"lag: {lag_chunks} chunks", flush=True)

    def post_partial(self, source):
        if self.show_partial and source:
            print("\n--- partial ---")
            print(source, flush=True)

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(0.2)

    def stop(self):
        self.stop_event.set()


_GLASS_PDF_VIEW_CLASS = None


def _glass_pdf_view_class():
    """Lazily build & cache a flipped NSView subclass that vector-draws the
    'liquid glass' PDF: a soft diagonal gradient backdrop, frosted rounded cards
    with a hairline rim and a soft drop shadow, and the transcript text on top.

    Cached because an Obj-C class can only be registered once per process; and
    built lazily so importing this module (e.g. for --help) doesn't pull AppKit."""
    global _GLASS_PDF_VIEW_CLASS
    if _GLASS_PDF_VIEW_CLASS is not None:
        return _GLASS_PDF_VIEW_CLASS

    import objc
    from AppKit import (
        NSBezierPath,
        NSColor,
        NSGraphicsContext,
        NSShadow,
        NSStringDrawingUsesLineFragmentOrigin,
        NSView,
    )

    class _GlassPDFView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(_GlassPDFView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._gradient = None
            self._cards = []   # list of NSRect (card backgrounds)
            self._texts = []   # list of (NSAttributedString, NSRect)
            return self

        def isFlipped(self):  # top-down layout
            return True

        def drawRect_(self, _dirty):
            bounds = self.bounds()
            if self._gradient is not None:
                self._gradient.drawInRect_angle_(bounds, -60.0)

            fill = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.62)
            rim = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.9)
            card_radius = float(getattr(self, "_card_radius", 20.0))
            shadow_blur = float(getattr(self, "_shadow_blur", 16.0))
            for rect in self._cards:
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, card_radius, card_radius
                )
                NSGraphicsContext.saveGraphicsState()
                shadow = NSShadow.alloc().init()
                shadow.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.13))
                shadow.setShadowBlurRadius_(shadow_blur)
                shadow.setShadowOffset_((0.0, 3.0))  # flipped view -> downward
                shadow.set()
                fill.set()
                path.fill()
                NSGraphicsContext.restoreGraphicsState()
                rim.set()
                path.setLineWidth_(1.0)
                path.stroke()

            for attr, rect in self._texts:
                attr.drawWithRect_options_(rect, NSStringDrawingUsesLineFragmentOrigin)

    _GLASS_PDF_VIEW_CLASS = _GlassPDFView
    return _GLASS_PDF_VIEW_CLASS


class GlassOverlay:
    def __init__(self, stop_event, settings, title, width, height, opacity, show_partial=False):
        try:
            from Cocoa import (
                NSApp,
                NSApplication,
                NSApplicationActivationPolicyRegular,
                NSBackingStoreBuffered,
                NSButton,
                NSColor,
                NSFloatingWindowLevel,
                NSFont,
                NSFontAttributeName,
                NSForegroundColorAttributeName,
                NSMakeRange,
                NSMakeRect,
                NSMakeSize,
                NSMomentaryPushInButton,
                NSMutableAttributedString,
                NSNormalWindowLevel,
                NSObject,
                NSOffState,
                NSOnState,
                NSPopUpButton,
                NSScrollView,
                NSSlider,
                NSSwitchButton,
                NSTextField,
                NSTextView,
                NSView,
                NSViewHeightSizable,
                NSViewWidthSizable,
                NSVisualEffectBlendingModeBehindWindow,
                NSVisualEffectMaterialHUDWindow,
                NSVisualEffectStateActive,
                NSVisualEffectView,
                NSWindow,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSWindowStyleMaskClosable,
                NSWindowStyleMaskFullSizeContentView,
                NSWindowStyleMaskResizable,
                NSWindowStyleMaskTitled,
            )
            from PyObjCTools import AppHelper
        except ImportError as exc:
            raise RuntimeError(
                "Для стеклянного окна установи `pip install pyobjc-framework-Cocoa`."
            ) from exc

        self.AppHelper = AppHelper
        self.NSMakeRange = NSMakeRange
        self.NSMakeRect = NSMakeRect
        self.NSMakeSize = NSMakeSize
        self.NSButton = NSButton
        self.NSColor = NSColor
        self.NSFont = NSFont
        self.NSFontAttributeName = NSFontAttributeName
        self.NSForegroundColorAttributeName = NSForegroundColorAttributeName
        self.NSNormalWindowLevel = NSNormalWindowLevel
        self.NSFloatingWindowLevel = NSFloatingWindowLevel
        self.NSOffState = NSOffState
        self.NSOnState = NSOnState
        self.NSSlider = NSSlider
        self.NSMutableAttributedString = NSMutableAttributedString
        self.NSMomentaryPushInButton = NSMomentaryPushInButton
        self.NSSwitchButton = NSSwitchButton
        self.NSView = NSView
        self.NSViewHeightSizable = NSViewHeightSizable
        self.NSViewWidthSizable = NSViewWidthSizable
        self.stop_event = stop_event
        self.settings = settings
        self.original_text = ""
        self.partial_text = ""
        self.show_partial = show_partial
        self.translated_text = ""
        self.original_blocks = []
        self.translation_blocks = []
        self.history_pairs = []  # full, uncapped transcript+translation for export
        self.session_started_at = time.monotonic()
        self._last_status_posted_at = 0.0
        self._last_status_value = 0
        self.font_size = 20
        self.compact_mode = False
        self.pin_enabled = True
        self.settings_visible = False
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        class MenuTarget(NSObject):
            def compactChanged_(menu_self, sender):
                self._compact_changed(sender)

            def pinChanged_(menu_self, sender):
                self._pin_changed(sender)

            def opacityChanged_(menu_self, sender):
                self._opacity_changed(sender)

            def smallerText_(menu_self, sender):
                self._adjust_text_size(-1)

            def largerText_(menu_self, sender):
                self._adjust_text_size(1)

            def sourceChanged_(menu_self, sender):
                self._source_changed(sender)

            def targetChanged_(menu_self, sender):
                self._target_changed(sender)

            def clearAll_(menu_self, sender):
                self._clear_all(sender)

            def savePDF_(menu_self, sender):
                self._save_pdf(sender)

            def settingsToggled_(menu_self, sender):
                self._settings_toggled(sender)

            def whisperChanged_(menu_self, sender):
                self._whisper_changed(sender)

            def ollamaModelChanged_(menu_self, sender):
                self._ollama_model_changed(sender)

        class WindowDelegate(NSObject):
            def windowDidResize_(delegate_self, notification):
                self._relayout()

        self.menu_target = MenuTarget.alloc().init()
        self.window_delegate = WindowDelegate.alloc().init()

        frame = NSMakeRect(120, 700, width, height)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
            | NSWindowStyleMaskFullSizeContentView
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            NSBackingStoreBuffered,
            False,
        )
        self._layout_ready = False
        self.window.setDelegate_(self.window_delegate)
        self.window.setContentMinSize_(NSMakeSize(420, 220))
        self.window.setTitle_(title)
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setMovableByWindowBackground_(True)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setAlphaValue_(opacity)
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = self.window.contentView()
        visual = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        visual.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        visual.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        visual.setMaterial_(NSVisualEffectMaterialHUDWindow)
        visual.setState_(NSVisualEffectStateActive)
        content.addSubview_(visual)

        inset = 18
        menu_height = 70
        column_label_height = 20
        gap = 32
        column_width = int((width - inset * 2 - gap) / 2)
        scroll_height = height - inset * 2 - menu_height - column_label_height
        scroll_y = inset
        label_y = scroll_y + scroll_height + 6
        menu_y = label_y + column_label_height + 18
        status_width = 120
        target_width = 140
        source_width = 126
        control_gap = 18
        status_x = width - inset - status_width
        target_label_x = status_x - control_gap - target_width - 44
        target_popup_x = target_label_x + 44
        source_label_x = target_label_x - control_gap - source_width - 48
        source_popup_x = source_label_x + 48

        controls_x = inset
        self.compact_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 78, 28),
            "Compact",
            "compactChanged:",
            momentary=True,
        )
        controls_x += 86
        self.pin_button = self._make_button(
            NSMakeRect(controls_x, menu_y + 2, 48, 24),
            "Pin",
            "pinChanged:",
        )
        self.pin_button.setState_(NSOnState)
        controls_x += 56
        self.opacity_label = self._make_label(
            NSTextField,
            "Opacity",
            NSMakeRect(controls_x, menu_y + 5, 46, 18),
            size=12,
            alpha=0.62,
        )
        visual.addSubview_(self.opacity_label)
        controls_x += 50
        self.opacity_slider = NSSlider.alloc().initWithFrame_(
            NSMakeRect(controls_x, menu_y + 2, 86, 24)
        )
        self.opacity_slider.setMinValue_(0.35)
        self.opacity_slider.setMaxValue_(1.0)
        self.opacity_slider.setDoubleValue_(opacity)
        self.opacity_slider.setTarget_(self.menu_target)
        self.opacity_slider.setAction_("opacityChanged:")
        controls_x += 96
        self.smaller_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 32, 28),
            "A-",
            "smallerText:",
            momentary=True,
        )
        controls_x += 36
        self.larger_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 32, 28),
            "A+",
            "largerText:",
            momentary=True,
        )
        self.clear_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 40, 28),
            "Clear",
            "clearAll:",
            momentary=True,
        )
        self._set_trash_icon(self.clear_button)
        self.save_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 50, 28),
            "PDF",
            "savePDF:",
            momentary=True,
        )
        self.settings_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 32, 28),
            "Settings",
            "settingsToggled:",
            momentary=True,
        )
        self._set_gear_icon(self.settings_button)
        for control in (
            self.compact_button,
            self.pin_button,
            self.opacity_slider,
            self.smaller_button,
            self.larger_button,
            self.clear_button,
            self.save_button,
            self.settings_button,
        ):
            visual.addSubview_(control)

        self.source_popup = self._make_popup(
            NSPopUpButton,
            NSMakeRect(source_popup_x, menu_y, source_width, 28),
            LANG_MENU,
            self.settings.get()[0],
            "sourceChanged:",
        )
        self.target_popup = self._make_popup(
            NSPopUpButton,
            NSMakeRect(target_popup_x, menu_y, target_width, 28),
            [item for item in LANG_MENU if item[0] != "auto"],
            self.settings.get()[1],
            "targetChanged:",
        )

        self.source_label = self._make_label(
            NSTextField,
            "Source",
            NSMakeRect(source_label_x, menu_y + 5, 44, 18),
            size=12,
            alpha=0.72,
        )
        visual.addSubview_(self.source_label)
        visual.addSubview_(self.source_popup)
        self.target_label = self._make_label(
            NSTextField,
            "Target",
            NSMakeRect(target_label_x, menu_y + 5, 40, 18),
            size=12,
            alpha=0.72,
        )
        visual.addSubview_(self.target_label)
        visual.addSubview_(self.target_popup)
        self.status_label = self._make_label(
            NSTextField,
            "lag: 0 chunks",
            NSMakeRect(status_x, menu_y + 5, status_width, 18),
            size=12,
            alpha=0.62,
        )
        visual.addSubview_(self.status_label)

        self.settings_panel = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(width - inset - 456, menu_y - 80, 456, 70)
        )
        self.settings_panel.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self.settings_panel.setMaterial_(NSVisualEffectMaterialHUDWindow)
        self.settings_panel.setState_(NSVisualEffectStateActive)
        self.settings_panel.setHidden_(True)
        self.settings_panel.setWantsLayer_(True)
        try:
            self.settings_panel.layer().setCornerRadius_(14.0)
            self.settings_panel.layer().setMasksToBounds_(True)
        except Exception:
            pass
        visual.addSubview_(self.settings_panel)

        self.whisper_settings_label = self._make_label(
            NSTextField,
            "Whisper",
            NSMakeRect(14, 42, 64, 18),
            size=12,
            alpha=0.72,
        )
        self.whisper_popup = self._make_popup(
            NSPopUpButton,
            NSMakeRect(82, 36, 142, 28),
            WHISPER_MENU,
            self.settings.get_whisper_size(),
            "whisperChanged:",
        )
        self.model_settings_label = self._make_label(
            NSTextField,
            "Gemma",
            NSMakeRect(244, 42, 54, 18),
            size=12,
            alpha=0.72,
        )
        self.ollama_model_popup = self._make_popup(
            NSPopUpButton,
            NSMakeRect(302, 36, 138, 28),
            GEMMA_MENU,
            self.settings.get_ollama_model(),
            "ollamaModelChanged:",
        )
        self.settings_hint_label = self._make_label(
            NSTextField,
            "Applies to the next audio block",
            NSMakeRect(14, 12, 426, 18),
            size=11,
            alpha=0.52,
        )
        for control in (
            self.whisper_settings_label,
            self.whisper_popup,
            self.model_settings_label,
            self.ollama_model_popup,
            self.settings_hint_label,
        ):
            self.settings_panel.addSubview_(control)

        left_x = inset
        right_x = inset + column_width + gap
        divider_x = inset + column_width + int(gap / 2)
        divider_height = scroll_height + column_label_height + 2
        self.expanded_frames = {
            "left_scroll": NSMakeRect(left_x, scroll_y, divider_x - left_x - 2, scroll_height),
            "right_scroll": NSMakeRect(right_x, scroll_y, width - right_x, scroll_height),
            "original_label": NSMakeRect(left_x, label_y, column_width, 18),
            "translation_label": NSMakeRect(right_x, label_y, column_width, 18),
        }
        self.compact_frames = {
            # Right edge flush to the window border (x == width) so the overlay
            # scrollbar tucks against the edge exactly like in expanded mode; keep the
            # left inset for text padding.
            "right_scroll": NSMakeRect(inset, scroll_y, width - inset, scroll_height),
            "translation_label": NSMakeRect(inset, label_y, width - inset * 2, 18),
        }
        self.divider_shadow = self._make_line(
            NSMakeRect(divider_x - 1, scroll_y, 1, divider_height),
            NSColor.blackColor().colorWithAlphaComponent_(0.36),
        )
        self.divider_highlight = self._make_line(
            NSMakeRect(divider_x, scroll_y, 1, divider_height),
            NSColor.whiteColor().colorWithAlphaComponent_(0.12),
        )
        visual.addSubview_(self.divider_shadow)
        visual.addSubview_(self.divider_highlight)
        self.original_label = self._make_label(
            NSTextField,
            "Original",
            self.expanded_frames["original_label"],
            size=12,
            alpha=0.62,
        )
        self.translation_label = self._make_label(
            NSTextField,
            "Translation",
            self.expanded_frames["translation_label"],
            size=12,
            alpha=0.62,
        )
        visual.addSubview_(self.original_label)
        visual.addSubview_(self.translation_label)

        self.left_scroll = NSScrollView.alloc().initWithFrame_(self.expanded_frames["left_scroll"])
        self.right_scroll = NSScrollView.alloc().initWithFrame_(self.expanded_frames["right_scroll"])
        self.left_scroll.setAutoresizingMask_(0)
        self.right_scroll.setAutoresizingMask_(0)
        self.left_scroll.setDrawsBackground_(False)
        self.right_scroll.setDrawsBackground_(False)
        self.left_scroll.setHasVerticalScroller_(True)
        self.right_scroll.setHasVerticalScroller_(True)
        # Thin, modern overlay scrollbars with a light knob (for the dark glass), that
        # float at the very edge and auto-hide when idle.
        try:
            from AppKit import NSScrollerKnobStyleLight, NSScrollerStyleOverlay

            for scroll in (self.left_scroll, self.right_scroll):
                scroll.setScrollerStyle_(NSScrollerStyleOverlay)
                scroll.setAutohidesScrollers_(True)
                scroller = scroll.verticalScroller()
                if scroller is not None:
                    scroller.setKnobStyle_(NSScrollerKnobStyleLight)
        except Exception:
            pass
        visual.addSubview_(self.left_scroll)
        visual.addSubview_(self.right_scroll)

        self.original_view = self._make_text_view(NSTextView, self.left_scroll)
        self.translated_view = self._make_text_view(NSTextView, self.right_scroll)
        self.original_view.setString_(WAITING_ORIGINAL)
        self.translated_view.setString_(self._waiting_translation())
        self.left_scroll.setDocumentView_(self.original_view)
        self.right_scroll.setDocumentView_(self.translated_view)
        self.settings_panel.removeFromSuperview()
        visual.addSubview_(self.settings_panel)

        self.NSApp = NSApp
        self._layout_ready = True
        self._relayout()

    def _relayout(self):
        if not getattr(self, "_layout_ready", False):
            return
        size = self.window.contentView().bounds().size
        width = size.width
        height = size.height
        inset = 18
        menu_height = 70
        column_label_height = 20
        gap = 32
        column_width = int((width - inset * 2 - gap) / 2)
        scroll_height = height - inset * 2 - menu_height - column_label_height
        if scroll_height < 1:
            scroll_height = 1
        scroll_y = inset
        label_y = scroll_y + scroll_height + 6
        menu_y = label_y + column_label_height + 18
        status_width = 120
        target_width = 140
        source_width = 126
        control_gap = 18
        save_x = width - inset - 50
        settings_x = save_x - 10 - 32
        clear_x = settings_x - 10 - 40
        status_x = clear_x - control_gap - status_width
        target_label_x = status_x - control_gap - target_width - 44
        target_popup_x = target_label_x + 44
        source_label_x = target_label_x - control_gap - source_width - 48
        source_popup_x = source_label_x + 48

        controls_x = inset
        self.compact_button.setFrame_(self.NSMakeRect(controls_x, menu_y, 78, 28))
        controls_x += 86
        self.pin_button.setFrame_(self.NSMakeRect(controls_x, menu_y + 2, 48, 24))
        controls_x += 56
        self.opacity_label.setFrame_(self.NSMakeRect(controls_x, menu_y + 5, 46, 18))
        controls_x += 50
        self.opacity_slider.setFrame_(self.NSMakeRect(controls_x, menu_y + 2, 86, 24))
        controls_x += 96
        self.smaller_button.setFrame_(self.NSMakeRect(controls_x, menu_y, 32, 28))
        controls_x += 36
        self.larger_button.setFrame_(self.NSMakeRect(controls_x, menu_y, 32, 28))

        self.save_button.setFrame_(self.NSMakeRect(save_x, menu_y, 50, 28))
        self.settings_button.setFrame_(self.NSMakeRect(settings_x, menu_y, 32, 28))
        self.clear_button.setFrame_(self.NSMakeRect(clear_x, menu_y, 40, 28))
        self.source_label.setFrame_(self.NSMakeRect(source_label_x, menu_y + 5, 44, 18))
        self.source_popup.setFrame_(self.NSMakeRect(source_popup_x, menu_y, source_width, 28))
        self.target_label.setFrame_(self.NSMakeRect(target_label_x, menu_y + 5, 40, 18))
        self.target_popup.setFrame_(self.NSMakeRect(target_popup_x, menu_y, target_width, 28))
        self.status_label.setFrame_(self.NSMakeRect(status_x, menu_y + 5, status_width, 18))

        # Right-side controls hide progressively when the window is too narrow, so they
        # never overlap the left cluster (Compact/Pin/Opacity/A-/A+). Clear/Save are
        # right-anchored and the last to go; Source is the first.
        left_end = inset + 86 + 56 + 50 + 96 + 36 + 32  # right edge of left cluster
        edge = left_end + 12
        show_save = save_x >= edge
        show_settings = settings_x >= edge
        show_clear = clear_x >= edge
        show_status = status_x >= edge
        show_target = target_label_x >= edge
        show_source = source_label_x >= edge
        self.save_button.setHidden_(not show_save)
        self.settings_button.setHidden_(not show_settings)
        self.clear_button.setHidden_(not show_clear)
        self.settings_panel.setHidden_((not self.settings_visible) or (not show_settings))
        self.status_label.setHidden_(not show_status)
        self.target_label.setHidden_(not show_target)
        self.target_popup.setHidden_(not show_target)
        self.source_label.setHidden_(not show_source)
        self.source_popup.setHidden_(not show_source)

        left_x = inset
        right_x = inset + column_width + gap
        divider_x = inset + column_width + int(gap / 2)
        divider_height = scroll_height + column_label_height + 2
        self.expanded_frames = {
            "left_scroll": self.NSMakeRect(left_x, scroll_y, divider_x - left_x - 2, scroll_height),
            "right_scroll": self.NSMakeRect(right_x, scroll_y, width - right_x, scroll_height),
            "original_label": self.NSMakeRect(left_x, label_y, column_width, 18),
            "translation_label": self.NSMakeRect(right_x, label_y, column_width, 18),
        }
        self.compact_frames = {
            # Right edge flush to the window border (x == width) so the overlay
            # scrollbar tucks against the edge exactly like in expanded mode; keep the
            # left inset for text padding.
            "right_scroll": self.NSMakeRect(inset, scroll_y, width - inset, scroll_height),
            "translation_label": self.NSMakeRect(inset, label_y, width - inset * 2, 18),
        }
        self.divider_shadow.setFrame_(self.NSMakeRect(divider_x - 1, scroll_y, 1, divider_height))
        self.divider_highlight.setFrame_(self.NSMakeRect(divider_x, scroll_y, 1, divider_height))
        self.original_label.setFrame_(self.expanded_frames["original_label"])
        self.left_scroll.setFrame_(self.expanded_frames["left_scroll"])
        if self.compact_mode:
            self.right_scroll.setFrame_(self.compact_frames["right_scroll"])
            self.translation_label.setFrame_(self.compact_frames["translation_label"])
        else:
            self.right_scroll.setFrame_(self.expanded_frames["right_scroll"])
            self.translation_label.setFrame_(self.expanded_frames["translation_label"])
        self.original_view.setFrame_(self.left_scroll.contentView().bounds())
        self.translated_view.setFrame_(self.right_scroll.contentView().bounds())
        panel_w = min(456, max(300, width - inset * 2))
        panel_x = max(inset, width - inset - panel_w)
        panel_y = max(scroll_y + 8, menu_y - 80)
        self.settings_panel.setFrame_(self.NSMakeRect(panel_x, panel_y, panel_w, 70))
        self.whisper_popup.setFrame_(self.NSMakeRect(82, 36, min(142, panel_w - 96), 28))
        model_x = min(244, max(14, panel_w - 212))
        self.model_settings_label.setFrame_(self.NSMakeRect(model_x, 42, 54, 18))
        self.ollama_model_popup.setFrame_(self.NSMakeRect(model_x + 58, 36, max(120, panel_w - model_x - 72), 28))
        self.settings_hint_label.setFrame_(self.NSMakeRect(14, 12, panel_w - 28, 18))
        # Re-flow text into the new frames; without this the resized text views go blank.
        self._render_original()
        self._render_translation()

    def _make_label(self, text_field_cls, value, frame, size, alpha):
        label = text_field_cls.alloc().initWithFrame_(frame)
        label.setStringValue_(value)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(self.NSColor.whiteColor().colorWithAlphaComponent_(alpha))
        label.setFont_(self.NSFont.systemFontOfSize_weight_(size, 0.18))
        return label

    def _make_button(self, frame, title, action, momentary=False):
        button = self.NSButton.alloc().initWithFrame_(frame)
        button.setTitle_(title)
        button.setButtonType_(self.NSMomentaryPushInButton if momentary else self.NSSwitchButton)
        button.setBezelStyle_(1)
        button.setTarget_(self.menu_target)
        button.setAction_(action)
        button.setFont_(self.NSFont.systemFontOfSize_weight_(12, 0.18))
        return button

    def _set_trash_icon(self, button):
        # Native vector trash glyph (SF Symbol), tinted white to match the toolbar.
        try:
            from AppKit import NSImage, NSImageOnly

            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_("trash", "Clear")
            if img is not None:
                button.setImage_(img)
                button.setImagePosition_(NSImageOnly)
                button.setTitle_("")
                try:
                    button.setContentTintColor_(self.NSColor.whiteColor())
                except Exception:
                    pass
        except Exception:
            pass  # fall back to the "Clear" text title

    def _set_gear_icon(self, button):
        try:
            from AppKit import NSImage, NSImageOnly

            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_("gearshape", "Settings")
            if img is not None:
                button.setImage_(img)
                button.setImagePosition_(NSImageOnly)
                button.setTitle_("")
                try:
                    button.setContentTintColor_(self.NSColor.whiteColor())
                except Exception:
                    pass
        except Exception:
            pass

    def _make_line(self, frame, color):
        line = self.NSView.alloc().initWithFrame_(frame)
        line.setWantsLayer_(True)
        line.layer().setBackgroundColor_(color.CGColor())
        return line

    def _make_popup(self, popup_cls, frame, items, selected_code, action):
        popup = popup_cls.alloc().initWithFrame_pullsDown_(frame, False)
        selected_label = None
        for code, label in items:
            popup.addItemWithTitle_(label)
            popup.itemWithTitle_(label).setRepresentedObject_(code)
            if code == selected_code:
                selected_label = label
        if selected_label is None:
            selected_label = language_label(selected_code)
        popup.selectItemWithTitle_(selected_label)
        popup.setTarget_(self.menu_target)
        popup.setAction_(action)
        return popup

    def _make_text_view(self, text_view_cls, scroll):
        view = text_view_cls.alloc().initWithFrame_(scroll.contentView().bounds())
        view.setAutoresizingMask_(self.NSViewWidthSizable | self.NSViewHeightSizable)
        view.setEditable_(False)
        view.setSelectable_(True)
        view.setDrawsBackground_(False)
        view.setTextColor_(self.NSColor.whiteColor())
        view.setFont_(self.NSFont.systemFontOfSize_weight_(20, 0.22))
        # Padding so text doesn't butt against the divider / window edge.
        view.setTextContainerInset_(self.NSMakeSize(10, 6))
        return view

    def _selected_code(self, popup):
        item = popup.selectedItem()
        code = item.representedObject() if item else None
        return str(code) if code else "auto"

    def _source_changed(self, sender):
        self.settings.set_source(self._selected_code(sender))

    def _waiting_translation(self):
        try:
            code = self.settings.get()[1]
        except Exception:
            code = "en"
        return WAITING_TRANSLATION.get(code, WAITING_TRANSLATION["en"])

    def _target_changed(self, sender):
        code = self._selected_code(sender)
        if code != "auto":
            self.settings.set_target(code)
        self._render_translation()  # refresh placeholder language if empty

    def _settings_toggled(self, sender):
        self.settings_visible = not self.settings_visible
        self.settings_panel.setHidden_(not self.settings_visible)
        self._relayout()

    def _whisper_changed(self, sender):
        code = self._selected_code(sender)
        if code in WHISPER_MODELS:
            self.settings.set_whisper_size(code)

    def _ollama_model_changed(self, sender):
        code = self._selected_code(sender)
        if code not in GEMMA_MODELS:
            return
        previous = self.settings.get_ollama_model()
        self.settings.set_ollama_model(code)
        translator = getattr(self, "translator", None)
        if previous and previous != code and translator is not None:
            # Free the old model from VRAM right now, on a background thread so the menu
            # doesn't hang on the Ollama unload request. The new model loads on the next
            # translation. (The translation worker would also unload it on its next phrase;
            # a double unload is harmless.)
            threading.Thread(
                target=translator._unload, args=(previous,), daemon=True
            ).start()

    def _compact_changed(self, sender):
        self.compact_mode = not self.compact_mode
        self.left_scroll.setHidden_(self.compact_mode)
        self.original_label.setHidden_(self.compact_mode)
        self.divider_shadow.setHidden_(self.compact_mode)
        self.divider_highlight.setHidden_(self.compact_mode)
        if self.compact_mode:
            self.right_scroll.setFrame_(self.compact_frames["right_scroll"])
            self.translation_label.setFrame_(self.compact_frames["translation_label"])
            self.compact_button.setTitle_("Expanded")
        else:
            self.right_scroll.setFrame_(self.expanded_frames["right_scroll"])
            self.translation_label.setFrame_(self.expanded_frames["translation_label"])
            self.compact_button.setTitle_("Compact")
        self.translated_view.setFrame_(self.right_scroll.contentView().bounds())
        self._render_translation()

    def _pin_changed(self, sender):
        self.pin_enabled = sender.state() == self.NSOnState
        self.window.setLevel_(self.NSFloatingWindowLevel if self.pin_enabled else self.NSNormalWindowLevel)

    def _opacity_changed(self, sender):
        self.window.setAlphaValue_(sender.doubleValue())

    def _adjust_text_size(self, delta):
        self.font_size = max(14, min(32, self.font_size + delta))
        self._render_original()
        self._render_translation()

    def _clear_all(self, sender=None):
        self.original_blocks = []
        self.translation_blocks = []
        self.history_pairs = []
        self.session_started_at = time.monotonic()
        self.partial_text = ""
        # Also wipe both workers' rolling state (audio buffer, tail, sentence buffer,
        # Whisper context) and pending queues so it restarts from a clean slate.
        reset_gen = getattr(self, "reset_gen", None)
        if reset_gen is not None:
            reset_gen[0] += 1
        self._render_original()
        self._render_translation()

    def _notify(self, title, message):
        try:
            from AppKit import NSAlert

            alert = NSAlert.alloc().init()
            alert.setMessageText_(str(title))
            alert.setInformativeText_(str(message))
            alert.runModal()
        except Exception:
            print(f"[{title}] {message}", file=sys.stderr)

    def _save_pdf(self, sender=None):
        import datetime
        import os
        import re

        from AppKit import NSSavePanel

        if not self.history_pairs:
            self._notify("Нечего сохранять", "Пока нет переведённого текста.")
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        panel = NSSavePanel.savePanel()
        panel.setNameFieldStringValue_(f"LiveTranslate-{ts}.pdf")
        try:
            panel.setAllowedFileTypes_(["pdf"])
        except Exception:
            pass
        self.app.activateIgnoringOtherApps_(True)
        if panel.runModal() != 1:  # 1 == NSModalResponseOK
            return
        path = panel.URL().path()

        from AppKit import (
            NSGradient,
            NSGraphicsContext,
            NSKernAttributeName,
            NSMutableParagraphStyle,
            NSParagraphStyleAttributeName,
        )
        from Foundation import NSURL
        from Quartz import (
            CGRectMake,
            CGPDFContextBeginPage,
            CGPDFContextClose,
            CGPDFContextCreateWithURL,
            CGPDFContextEndPage,
        )

        NSColor = self.NSColor
        NSFont = self.NSFont
        NSFontAttr = self.NSFontAttributeName
        NSFgAttr = self.NSForegroundColorAttributeName

        page_width = 595.28
        page_height = 841.89
        outer = 30.0
        card_gap = 8.0
        pad_x, pad_top, pad_bottom = 16.0, 10.0, 12.0
        footer_h = 24.0
        inner_w = page_width - outer * 2 - pad_x * 2

        c_title = NSColor.colorWithCalibratedWhite_alpha_(0.13, 1.0)
        c_subtitle = NSColor.colorWithCalibratedWhite_alpha_(0.45, 1.0)
        c_label = NSColor.colorWithCalibratedWhite_alpha_(0.50, 1.0)
        c_src = NSColor.colorWithCalibratedWhite_alpha_(0.32, 1.0)
        c_tr = NSColor.colorWithCalibratedWhite_alpha_(0.13, 1.0)
        c_accent = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.20, 0.36, 0.92, 1.0)

        gradient = NSGradient.alloc().initWithColors_(
            [
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.88, 0.92, 0.99, 1.0),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.93, 0.91, 0.99, 1.0),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.99, 0.94, 0.96, 1.0),
            ]
        )

        title_font = NSFont.systemFontOfSize_weight_(20.0, 0.32)
        subtitle_font = NSFont.systemFontOfSize_(9.5)
        label_font = NSFont.systemFontOfSize_weight_(7.8, 0.34)
        body_font = NSFont.systemFontOfSize_(10.4)
        footer_font = NSFont.systemFontOfSize_(7.8)
        try:
            time_font = NSFont.monospacedDigitSystemFontOfSize_weight_(8.2, 0.34)
        except Exception:
            time_font = label_font

        def para(line_spacing=0.0, after=0.0, before=0.0):
            p = NSMutableParagraphStyle.alloc().init()
            p.setLineSpacing_(line_spacing)
            p.setParagraphSpacing_(after)
            p.setParagraphSpacingBefore_(before)
            return p

        ps_body = para(line_spacing=1.6)
        ps_label = para(after=2.0)
        ps_label_before = para(after=2.0, before=8.0)
        ps_time = para(after=3.0)
        ps_footer = para(line_spacing=1.2)
        ps_title = para(after=2.0)

        def piece(text, font, color, ps, kern=None):
            attrs = {NSFgAttr: color, NSFontAttr: font, NSParagraphStyleAttributeName: ps}
            if kern is not None:
                attrs[NSKernAttributeName] = kern
            return self.NSMutableAttributedString.alloc().initWithString_attributes_(text, attrs)

        from AppKit import NSTextView

        # TextKit measurement matches final NSTextView rendering and avoids clipped last lines.
        def make_tv(attr, width):
            tv = NSTextView.alloc().initWithFrame_(((0, 0), (width, 1.0e6)))
            tv.setDrawsBackground_(False)
            tv.setEditable_(False)
            tv.setSelectable_(False)
            tv.textContainer().setLineFragmentPadding_(0.0)
            tv.setTextContainerInset_((0.0, 0.0))
            tv.textStorage().setAttributedString_(attr)
            lm, tc = tv.layoutManager(), tv.textContainer()
            lm.ensureLayoutForTextContainer_(tc)
            h = float(lm.usedRectForTextContainer_(tc).size.height)
            return tv, h

        def format_timecode(pair):
            elapsed = pair.get("elapsed")
            if elapsed is None:
                absolute = pair.get("time")
                if isinstance(absolute, (int, float)):
                    elapsed = absolute - getattr(self, "session_started_at", absolute)
            if elapsed is None:
                return "--:--:--"
            total = max(0, int(round(float(elapsed))))
            hours, rem = divmod(total, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        human = datetime.datetime.now().strftime("%d %B %Y - %H:%M")
        header = self.NSMutableAttributedString.alloc().init()
        header.appendAttributedString_(piece("Live Translation\n", title_font, c_title, ps_title))
        header.appendAttributedString_(piece(human, subtitle_font, c_subtitle, para()))
        header_w = page_width - outer * 2
        _, header_h = make_tv(header, header_w)
        header_gap = 14.0
        content_bottom = page_height - outer - footer_h
        first_page_max_card_h = content_bottom - (outer + header_h + header_gap)
        max_card_h = max(120.0, first_page_max_card_h)

        def make_card(src, tr, timecode, continued=False):
            src = (src or "").strip()
            tr = (tr or "").strip()
            first_label = "ORIGINAL" if src else "TRANSLATION"
            first_label_color = c_label if src else c_accent
            suffix = " CONT." if continued else ""
            content = self.NSMutableAttributedString.alloc().init()
            content.appendAttributedString_(piece(f"{timecode}  ", time_font, c_label, ps_time, kern=0.2))
            content.appendAttributedString_(piece(f"{first_label}{suffix}\n", label_font, first_label_color, ps_time, kern=1.2))
            if src:
                content.appendAttributedString_(
                    piece(src + ("\n" if tr else ""), body_font, c_src, ps_body)
                )
            if tr:
                if src:
                    content.appendAttributedString_(piece("TRANSLATION\n", label_font, c_accent, ps_label_before, kern=1.2))
                content.appendAttributedString_(piece(tr, body_font, c_tr, ps_body))
            _, content_h = make_tv(content, inner_w)
            return {"attr": content, "content_h": content_h, "card_h": pad_top + content_h + pad_bottom}

        def tokens_for(text):
            tokens = re.findall(r"\S+\s*", text)
            if tokens:
                return tokens
            return [text[i : i + 120] for i in range(0, len(text), 120)]

        def split_single(label, text, timecode):
            tokens = tokens_for(text.strip())
            cards = []
            pos = 0
            while pos < len(tokens):
                lo, hi = 1, len(tokens) - pos
                best = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    chunk = "".join(tokens[pos : pos + mid]).strip()
                    src = chunk if label == "ORIGINAL" else ""
                    tr = chunk if label == "TRANSLATION" else ""
                    card = make_card(src, tr, timecode, continued=bool(cards))
                    if card["card_h"] <= max_card_h:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                chunk = "".join(tokens[pos : pos + best]).strip()
                src = chunk if label == "ORIGINAL" else ""
                tr = chunk if label == "TRANSLATION" else ""
                cards.append(make_card(src, tr, timecode, continued=bool(cards)))
                pos += best
            return cards

        cards = []
        for pair in self.history_pairs:
            src = (pair.get("source") or "").strip()
            tr = (pair.get("translated") or "").strip()
            if not src and not tr:
                continue
            timecode = format_timecode(pair)
            card = make_card(src, tr, timecode)
            if card["card_h"] <= max_card_h:
                cards.append(card)
            else:
                if src:
                    cards.extend(split_single("ORIGINAL", src, timecode))
                if tr:
                    cards.extend(split_single("TRANSLATION", tr, timecode))

        if not cards:
            self._notify("Нечего сохранять", "Пока нет переведённого текста.")
            return

        pages = []
        current = []
        include_header = True
        y = outer + header_h + header_gap
        for card in cards:
            gap = card_gap if current else 0.0
            if current and y + gap + card["card_h"] > content_bottom:
                pages.append((include_header, current))
                current = []
                include_header = False
                y = outer
                gap = 0.0
            current.append(card)
            y += gap + card["card_h"]
        if current:
            pages.append((include_header, current))

        ViewClass = _glass_pdf_view_class()

        def add_text_view(view, attr, x, y_pos, width):
            tv, tv_h = make_tv(attr, width)
            tv.setFrame_(((x, y_pos), (width, tv_h)))
            view.addSubview_(tv)
            return tv_h

        def make_page_view(page_index, total_pages, has_header, page_cards):
            view = ViewClass.alloc().initWithFrame_(((0, 0), (page_width, page_height)))
            view._gradient = gradient
            view._texts = []
            view._cards = []
            view._card_radius = 12.0
            view._shadow_blur = 8.0

            y_pos = outer
            if has_header:
                add_text_view(view, header, outer, y_pos, header_w)
                y_pos += header_h + header_gap

            for idx, card in enumerate(page_cards):
                if idx:
                    y_pos += card_gap
                view._cards.append(((outer, y_pos), (page_width - outer * 2, card["card_h"])))
                tv, _ = make_tv(card["attr"], inner_w)
                tv.setFrame_(((outer + pad_x, y_pos + pad_top), (inner_w, card["content_h"])))
                view.addSubview_(tv)
                y_pos += card["card_h"]

            footer = self.NSMutableAttributedString.alloc().init()
            footer_text = f"Live Translation - {page_index + 1}/{total_pages}"
            if page_index == 0:
                footer_text += (
                    "\nTimecode is measured from app launch; launch time is treated as 00:00:00."
                )
            footer.appendAttributedString_(
                piece(footer_text, footer_font, c_subtitle, ps_footer)
            )
            add_text_view(view, footer, outer, page_height - outer - footer_h, header_w)
            return view

        media_box = CGRectMake(0.0, 0.0, page_width, page_height)
        ctx = CGPDFContextCreateWithURL(NSURL.fileURLWithPath_(path), media_box, None)
        ok = bool(ctx)
        if ok:
            total_pages = len(pages)
            for page_index, (has_header, page_cards) in enumerate(pages):
                page_view = make_page_view(page_index, total_pages, has_header, page_cards)
                CGPDFContextBeginPage(ctx, {})
                graphics = NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, False)
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.setCurrentContext_(graphics)
                page_view.displayRectIgnoringOpacity_inContext_(page_view.bounds(), graphics)
                NSGraphicsContext.restoreGraphicsState()
                CGPDFContextEndPage(ctx)
            CGPDFContextClose(ctx)
            ok = os.path.exists(path) and os.path.getsize(path) > 0
        self._notify(
            "PDF сохранён" if ok else "Ошибка",
            path if ok else "Не удалось записать файл.",
        )

    def post_pair(
        self,
        source,
        translated,
        pause_ms=0,
        replace_translation_tail=False,
        combined_source=None,
    ):
        self.AppHelper.callAfter(
            self._append_pair,
            source,
            translated,
            pause_ms,
            replace_translation_tail,
            combined_source,
        )

    def post_partial(self, source):
        if not self.show_partial:
            return
        self.AppHelper.callAfter(self._set_partial, source)

    def post_status(self, lag_chunks):
        lag_chunks = int(lag_chunks)
        now = time.monotonic()
        if (
            lag_chunks != 0
            and now - self._last_status_posted_at < STATUS_UPDATE_INTERVAL_SECONDS
        ):
            return
        if lag_chunks == self._last_status_value and now - self._last_status_posted_at < 1.0:
            return
        self._last_status_posted_at = now
        self._last_status_value = lag_chunks
        self.AppHelper.callAfter(self._set_status, lag_chunks)

    def _set_status(self, lag_chunks):
        self.status_label.setStringValue_(f"lag: {lag_chunks} chunks")
        alpha = 0.9 if lag_chunks else 0.62
        self.status_label.setTextColor_(self.NSColor.whiteColor().colorWithAlphaComponent_(alpha))

    def _append_pair(
        self,
        source,
        translated,
        pause_ms=0,
        replace_translation_tail=False,
        combined_source=None,
    ):
        # Guard the main-thread render path: an exception here can stop the run loop from
        # delivering further UI updates, freezing the display while workers keep running.
        try:
            now = time.monotonic()
            speaker_gap = pause_ms >= 1400
            if source and not str(translated).startswith("Ошибка:"):
                if replace_translation_tail and self.history_pairs:
                    previous = self.history_pairs[-1]
                    previous["source"] = str(combined_source or "").strip() or (
                        f"{str(previous.get('source') or '').rstrip()} {source}".strip()
                    )
                    previous["translated"] = translated or ""
                    previous["time"] = now
                    previous["elapsed"] = now - self.session_started_at
                else:
                    self.history_pairs.append(
                        {
                            "source": source,
                            "translated": translated or "",
                            "time": now,
                            "elapsed": now - self.session_started_at,
                        }
                    )
                if len(self.history_pairs) > 5000:
                    self.history_pairs = self.history_pairs[-5000:]
            if source:
                self.original_blocks.append({"text": source, "time": now, "gap": speaker_gap})
                self.original_blocks = self.original_blocks[-16:]
                self.partial_text = ""
                self._render_original()
            if translated:
                if replace_translation_tail:
                    self._replace_translation_tail(
                        translated,
                        now,
                        bool(speaker_gap),
                        combined_source or source,
                    )
                else:
                    self._append_translation_block(translated, now, speaker_gap, source)
                self._render_translation()
        except Exception as exc:
            print(f"[ui] _append_pair: {exc!r}", file=sys.stderr)

    def _should_merge_translation_tail(self, previous, incoming, now, speaker_gap, incoming_source=None):
        previous_text = (previous.get("text") or "").strip()
        incoming_text = (incoming or "").strip()
        previous_source = (previous.get("source") or "").strip()
        incoming_source = (incoming_source or "").strip()
        if not previous_text or not incoming_text:
            return False
        boundary_text = previous_source or previous_text
        source_aware = bool(previous_source and incoming_source)
        if (
            not source_aware
            and now - float(previous.get("time") or now) > TRANSLATION_TAIL_REGROUP_SECONDS
        ):
            return False
        max_merged = (
            TRANSLATION_TAIL_MAX_SOURCE_MERGED_CHARS
            if source_aware
            else TRANSLATION_TAIL_MAX_MERGED_CHARS
        )
        if len(previous_text) + len(incoming_text) > max_merged:
            return False
        if boundary_text.endswith((",", ";", ":", "-", "—")):
            return True
        if ELLIPSIS_END_RE.search(boundary_text):
            return True
        if SOFT_END_RE.search(boundary_text):
            if source_aware and starts_like_continuation(incoming_source):
                return True
            return False
        if speaker_gap and not source_aware:
            return False
        return source_aware

    def _append_translation_block(self, translated, now, speaker_gap, source=None):
        text = str(translated).strip()
        if not text:
            return
        source_text = str(source or "").strip()
        if self.translation_blocks and self._should_merge_translation_tail(
            self.translation_blocks[-1],
            text,
            now,
            speaker_gap,
            source_text,
        ):
            previous = self.translation_blocks[-1]
            previous["text"] = f"{str(previous.get('text') or '').rstrip()} {text}".strip()
            if source_text:
                previous["source"] = f"{str(previous.get('source') or '').rstrip()} {source_text}".strip()
            previous["time"] = now
            previous["gap"] = bool(previous.get("gap")) or speaker_gap
        else:
            self.translation_blocks.append(
                {"text": text, "source": source_text, "time": now, "gap": speaker_gap}
            )
        self.translation_blocks = self.translation_blocks[-16:]

    def _replace_translation_tail(self, translated, now, speaker_gap, source=None):
        text = str(translated).strip()
        if not text:
            return
        source_text = str(source or "").strip()
        if not self.translation_blocks:
            self.translation_blocks.append(
                {"text": text, "source": source_text, "time": now, "gap": speaker_gap}
            )
            return
        previous = self.translation_blocks[-1]
        previous["text"] = text
        previous["source"] = source_text
        previous["time"] = now
        previous["gap"] = bool(previous.get("gap"))

    def _set_partial(self, source):
        try:
            self.partial_text = source.strip()
            self._render_original()
        except Exception as exc:
            print(f"[ui] _set_partial: {exc!r}", file=sys.stderr)

    def _render_original(self):
        full_text, spans = self._compose_blocks(self.original_blocks)
        partial = self.partial_text.strip()
        if partial:
            if full_text and not full_text.endswith("\n\n"):
                full_text += "\n\n"
            full_text += partial

        if not full_text:
            full_text = WAITING_ORIGINAL

        font = self.NSFont.systemFontOfSize_weight_(self.font_size, 0.22)
        committed_attrs = {
            self.NSForegroundColorAttributeName: self.NSColor.whiteColor(),
            self.NSFontAttributeName: font,
        }
        attributed = self.NSMutableAttributedString.alloc().initWithString_attributes_(
            full_text,
            committed_attrs,
        )
        self._apply_block_fade(attributed, spans, font)
        if partial:
            partial_start = len(full_text) - len(partial)
            partial_attrs = {
                self.NSForegroundColorAttributeName: self.NSColor.whiteColor().colorWithAlphaComponent_(0.32),
                self.NSFontAttributeName: font,
            }
            attributed.addAttributes_range_(
                partial_attrs,
                self.NSMakeRange(partial_start, len(partial)),
            )
        self.original_view.textStorage().setAttributedString_(attributed)
        self.original_view.scrollRangeToVisible_(self.NSMakeRange(len(full_text), 0))

    def _render_translation(self):
        full_text, spans = self._compose_blocks(self.translation_blocks)
        if not full_text:
            full_text = self._waiting_translation()
        font = self.NSFont.systemFontOfSize_weight_(self.font_size, 0.22)
        attrs = {
            self.NSForegroundColorAttributeName: self.NSColor.whiteColor(),
            self.NSFontAttributeName: font,
        }
        attributed = self.NSMutableAttributedString.alloc().initWithString_attributes_(
            full_text,
            attrs,
        )
        self._apply_block_fade(attributed, spans, font)
        self.translated_view.textStorage().setAttributedString_(attributed)
        self.translated_view.scrollRangeToVisible_(self.NSMakeRange(len(full_text), 0))

    def _compose_blocks(self, blocks):
        chunks = []
        spans = []
        pos = 0
        for idx, block in enumerate(blocks):
            sep = "\n\n\n" if block.get("gap") and chunks else ("\n\n" if chunks else "")
            chunks.append(sep)
            pos += len(sep)
            text = block["text"].strip()
            chunks.append(text)
            spans.append((pos, len(text), idx))
            pos += len(text)
        return "".join(chunks), spans

    def _apply_block_fade(self, attributed, spans, font):
        total = len(spans)
        for start, length, idx in spans:
            age_from_end = total - idx - 1
            alpha = 1.0 if age_from_end < 3 else max(0.48, 0.82 - age_from_end * 0.06)
            attrs = {
                self.NSForegroundColorAttributeName: self.NSColor.whiteColor().colorWithAlphaComponent_(alpha),
                self.NSFontAttributeName: font,
            }
            attributed.addAttributes_range_(attrs, self.NSMakeRange(start, length))

    def run(self):
        self.window.makeKeyAndOrderFront_(None)
        self.app.activateIgnoringOtherApps_(True)
        self.AppHelper.runEventLoop()

    def stop(self):
        self.stop_event.set()
        self.AppHelper.callAfter(self.AppHelper.stopEventLoop)


def put_drop_oldest(q, item):
    dropped = False
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
            dropped = True
        except queue.Empty:
            pass
        q.put_nowait(item)
    return dropped


def drain_queue_keep_latest(q, keep=0):
    drained = []
    while True:
        try:
            drained.append(q.get_nowait())
        except queue.Empty:
            break
    if keep > 0 and drained:
        for item in drained[-keep:]:
            try:
                q.put_nowait(item)
            except queue.Full:
                break
    return max(0, len(drained) - keep)


def enqueue_translation(translation_q, source_text, pause_ms=0.0, source_language=None):
    if not source_text:
        return
    if translation_q.maxsize and translation_q.qsize() >= translation_q.maxsize:
        dropped = drain_queue_keep_latest(translation_q, TRANSLATION_QUEUE_KEEP_TASKS)
        if dropped:
            print(
                f"[translate] очередь перевода переполнена — сброшено {dropped}, оставляю свежие задачи",
                file=sys.stderr,
            )
    task = {
        "source": source_text,
        "pause_ms": float(pause_ms or 0.0),
    }
    if source_language:
        # Resolved source language (e.g. Whisper's detected language in auto mode), so the
        # translator doesn't have to fall back to "auto" — matters for TranslateGemma.
        task["source_language"] = source_language
    dropped = put_drop_oldest(translation_q, task)
    if dropped:
        print("[translate] очередь перевода забита — пропущена старая задача", file=sys.stderr)


def should_retranslate_merged_source(previous_source, incoming_source, pause_ms=0.0):
    previous_source = str(previous_source or "").strip()
    incoming_source = str(incoming_source or "").strip()
    if not previous_source or not incoming_source:
        return False
    if len(previous_source) + len(incoming_source) > TRANSLATION_TAIL_MAX_SOURCE_MERGED_CHARS:
        return False
    if previous_source.endswith((",", ";", ":", "-", "—")):
        return True
    if ELLIPSIS_END_RE.search(previous_source):
        return True
    if SOFT_END_RE.search(previous_source):
        return starts_like_continuation(incoming_source)
    return True


def load_vad():
    """Lazily load Silero VAD. Returns (model, get_speech_timestamps) or (None, None)
    so the pipeline degrades gracefully to fixed-size chunking if VAD is unavailable."""
    try:
        import torch  # noqa: F401
        from silero_vad import get_speech_timestamps, load_silero_vad

        return load_silero_vad(), get_speech_timestamps
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[VAD] недоступен, фиксированная нарезка ({exc})", file=sys.stderr)
        return None, None


def _to_mono_16k(audio, samplerate):
    import torch
    import torchaudio

    a = audio.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    tensor = torch.from_numpy(np.ascontiguousarray(a))
    if int(samplerate) != 16000:
        tensor = torchaudio.functional.resample(tensor, int(samplerate), 16000)
    return tensor


def vad_analyze(audio, samplerate, vad_model, get_speech_timestamps, min_keep_frames, min_speech_ms=250.0):
    """Analyze a chunk with Silero VAD. Returns (has_speech, cut):
    - has_speech: whether enough *real* speech was detected. We require the total speech
      duration to reach min_speech_ms — a stray blip/noise that Silero marks as a tiny
      speech island won't pass, which keeps short bursts of noise out of Whisper.
    - cut: a sample index (original samplerate) to cut at, in the silence right after the
      last speech (never mid-word), or None to keep the fixed-size cut.
    On any error returns (True, None) so the caller falls back to the RMS gate."""
    try:
        wav = _to_mono_16k(audio, samplerate)
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    except Exception:
        return True, None
    if not timestamps:
        return False, None
    speech_16k = sum(int(t["end"]) - int(t["start"]) for t in timestamps)
    if speech_16k < int(min_speech_ms / 1000.0 * 16000):
        return False, None
    total_16k = int(wav.shape[-1])
    last_end_16k = int(timestamps[-1]["end"])
    # Require real trailing silence after the last speech, otherwise we'd risk a mid-word cut.
    if total_16k - last_end_16k < int(0.10 * 16000):
        return True, None
    cut = int(last_end_16k * samplerate / 16000)
    if cut < min_keep_frames:
        return True, None
    return True, cut


# ---------------------------------------------------------------------------
# LocalAgreement-2 streaming (Whisper-Streaming style). Instead of cutting audio
# into independent chunks, we keep a rolling buffer, re-transcribe it on each
# update, and only *commit* the longest common prefix that two consecutive
# transcriptions agree on. Committed text never changes; the unconfirmed tail is
# shown as a live draft. This removes word breaks, dupes and false periods at
# chunk boundaries that plain chunking produced.
# ---------------------------------------------------------------------------

def _norm_word(text):
    return re.sub(r"[^\w'’-]", "", text).casefold()


class HypothesisBuffer:
    def __init__(self):
        self.committed_in_buffer = []  # (start, end, text) absolute-time words
        self.buffer = []               # previous hypothesis tail (unconfirmed)
        self.new = []
        self.last_committed_time = 0.0

    def insert(self, words, offset):
        words = [(s + offset, e + offset, t) for (s, e, t) in words]
        self.new = [w for w in words if w[0] > self.last_committed_time - 0.1]

    def flush(self):
        # Commit the longest common prefix between the new and the previous hypothesis.
        commit = []
        while self.new:
            ns, ne, nt = self.new[0]
            if not self.buffer:
                break
            if _norm_word(nt) and _norm_word(nt) == _norm_word(self.buffer[0][2]):
                commit.append((ns, ne, nt))
                self.last_committed_time = ne
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit


class OnlineProcessor:
    def __init__(self, samplerate, max_active_seconds=14.0):
        self.sr = samplerate
        self.audio = None
        self.buffer_offset = 0.0  # seconds of audio already trimmed away
        self.hyp = HypothesisBuffer()
        self.max_active = max_active_seconds

    def active_seconds(self):
        return 0.0 if self.audio is None else len(self.audio) / self.sr

    def insert_audio(self, block):
        self.audio = block if self.audio is None else np.concatenate((self.audio, block), axis=0)

    def reset(self):
        self.audio = None
        self.buffer_offset = 0.0
        self.hyp = HypothesisBuffer()

    def finalize(self):
        # Utterance ended (silence): there won't be another pass to agree with, so accept
        # the current unconfirmed draft as final instead of dropping it. Returns its text.
        tail = " ".join(t for (_, _, t) in self.hyp.buffer).strip()
        self.hyp.committed_in_buffer.extend(self.hyp.buffer)
        self.hyp.buffer = []
        self.hyp.new = []
        return tail

    def drop_active(self):
        # Discard buffered (silent) audio without committing anything.
        if self.audio is not None:
            self.buffer_offset += len(self.audio) / self.sr
            self.audio = None
        self.hyp.buffer = []
        self.hyp.new = []

    def _trim_to(self, t):
        cut_idx = int((t - self.buffer_offset) * self.sr)
        if self.audio is not None and 0 < cut_idx < len(self.audio):
            self.audio = self.audio[cut_idx:]
            self.buffer_offset = t
        self.hyp.committed_in_buffer = [
            w for w in self.hyp.committed_in_buffer if w[1] >= t - 30
        ]

    def process(self, transcribe_words_fn):
        """Returns (committed_words, draft_text)."""
        words = transcribe_words_fn(self.audio)
        self.hyp.insert(words, self.buffer_offset)
        committed = self.hyp.flush()
        if committed:
            self._trim_to(self.hyp.last_committed_time)
        elif self.active_seconds() > self.max_active:
            # Unstable for too long — force the buffer down so we never blow past
            # Whisper's window or transcribe an ever-growing clip.
            self._trim_to(self.buffer_offset + self.active_seconds() - self.max_active * 0.6)
        draft = " ".join(t for (_, _, t) in self.hyp.buffer)
        return committed, draft


def audio_capture_worker(audio_q, stop_event, device, samplerate, channels, block_seconds):
    blocksize = max(256, int(samplerate * block_seconds))
    last_audio = {"t": time.monotonic()}

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        last_audio["t"] = time.monotonic()
        put_drop_oldest(audio_q, indata.copy())

    def open_stream():
        stream = sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            blocksize=blocksize,
            callback=callback,
        )
        stream.start()
        return stream

    # The audio device keeps delivering frames (even silence) on its own clock, so a
    # gap with no callback means the stream stalled — e.g. CoreAudio re-inits the route
    # when a YouTube video switches. Watch for that and transparently reopen the stream.
    stream = None
    while not stop_event.is_set():
        try:
            if stream is None:
                stream = open_stream()
                last_audio["t"] = time.monotonic()
            stalled = (time.monotonic() - last_audio["t"]) > 3.0
            inactive = not stream.active
            if stalled or inactive:
                print("[audio] поток залип — перезапуск", file=sys.stderr)
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                stream = None
                time.sleep(0.3)
                continue
        except Exception as exc:
            print(f"[audio] ошибка потока: {exc!r} — перезапуск", file=sys.stderr)
            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            stream = None
            time.sleep(0.5)
            continue
        time.sleep(0.2)

    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def chunk_worker(
    audio_q,
    chunk_q,
    overlay,
    stop_event,
    samplerate,
    chunk_seconds,
    overlap_seconds,
    endpointing_ms,
    silence_rms,
    use_vad=True,
    min_chunk_seconds=1.2,
    pause_emit_ms=350.0,
    reset_gen=None,
    vad_min_speech_ms=250.0,
):
    # chunk_seconds is the MAX audio window; we emit earlier whenever the speaker
    # makes a natural pause (>= pause_emit_ms) after at least min_chunk_seconds. This
    # keeps latency low while giving Whisper longer, sentence-complete context on
    # run-on speech (short fixed chunks made Whisper end every piece with a period).
    frames_needed = int(samplerate * chunk_seconds)
    min_chunk_frames = int(samplerate * min_chunk_seconds)
    overlap_frames = max(0, min(int(samplerate * overlap_seconds), frames_needed - 1))
    vad_model, get_speech_timestamps = load_vad() if use_vad else (None, None)
    min_keep_frames = int(samplerate * min(1.0, min_chunk_seconds * 0.5))
    buffer = np.empty((0, 1), dtype=np.float32)
    channel_count = None
    trailing_silence_ms = 0.0
    endpoint_sent = False
    frames = 0
    seen_gen = reset_gen[0] if reset_gen is not None else 0

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            # Clear button: drop buffered audio so old speech can't leak through.
            seen_gen = reset_gen[0]
            buffer = np.empty((0, channel_count or 1), dtype=np.float32)
            trailing_silence_ms = 0.0
            endpoint_sent = False
            while True:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break
        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            block_rms = float(np.sqrt(np.mean(np.square(block.astype(np.float32)))))
            block_ms = len(block) / samplerate * 1000.0
            if block_rms >= silence_rms:
                trailing_silence_ms = 0.0
                endpoint_sent = False
            else:
                trailing_silence_ms += block_ms
                if trailing_silence_ms >= endpointing_ms and not endpoint_sent:
                    endpoint_sent = True
                    dropped = put_drop_oldest(
                        chunk_q,
                        {
                            "audio": None,
                            "endpoint": True,
                            "trailing_silence_ms": trailing_silence_ms,
                        },
                    )
                    if dropped:
                        print("[chunk] очередь забита — пропущен старый чанк, догоняю", file=sys.stderr)
                    overlay.post_status(chunk_q.qsize())
                if len(buffer) == 0:
                    if endpoint_sent and audio_q.qsize() > NO_SPEECH_KEEP_AUDIO_BLOCKS:
                        drained = drain_queue_keep_latest(audio_q, NO_SPEECH_KEEP_AUDIO_BLOCKS)
                        if drained:
                            print(
                                f"[chunk] нет речи — сброшено ~{drained * block_ms / 1000.0:.1f}s тишины",
                                file=sys.stderr,
                            )
                            overlay.post_status(chunk_q.qsize())
                    continue

            if channel_count is None:
                channel_count = block.shape[1] if block.ndim > 1 else 1
                buffer = np.empty((0, channel_count), dtype=block.dtype)
            buffer = np.concatenate((buffer, block), axis=0)
            frames = len(buffer)
            pause_now = frames >= min_chunk_frames and trailing_silence_ms >= pause_emit_ms
            if frames < frames_needed and not pause_now:
                continue

            cut = min(frames, frames_needed)
            has_speech = True
            if vad_model is not None:
                has_speech, vad_cut = vad_analyze(
                    buffer[:cut],
                    samplerate,
                    vad_model,
                    get_speech_timestamps,
                    min_keep_frames,
                    vad_min_speech_ms,
                )
                if vad_cut is not None:
                    cut = vad_cut

            audio = buffer[:cut].copy()
            tail_start = max(0, cut - overlap_frames) if overlap_frames else cut
            buffer = buffer[tail_start:]
            frames = len(buffer)

            # Don't send silence/music to Whisper: VAD says no speech (or RMS too low when
            # VAD is off). Feeding long pauses to Whisper is what stalled the pipeline.
            if not has_speech:
                continue
            rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
            if rms < silence_rms:
                continue
            # Recovery-first: under normal load every chunk is processed once; if the
            # transcribe/translate worker falls badly behind, keep the freshest chunks
            # instead of blocking forever on stale audio.
            if chunk_q.maxsize and chunk_q.qsize() >= chunk_q.maxsize:
                drained = drain_queue_keep_latest(chunk_q, CHUNK_PRODUCER_KEEP_CHUNKS)
                if drained:
                    print(
                        f"[chunk] очередь переполнена — сброшено {drained}, оставляю свежий хвост",
                        file=sys.stderr,
                    )
            dropped = put_drop_oldest(
                chunk_q,
                {
                    "audio": audio,
                    "endpoint": trailing_silence_ms >= endpointing_ms,
                    "trailing_silence_ms": trailing_silence_ms,
                },
            )
            if dropped:
                print("[chunk] очередь забита — пропущен старый чанк, догоняю", file=sys.stderr)
            overlay.post_status(chunk_q.qsize())
        except Exception as exc:
            # Never let the chunk thread die silently — that permanently stops transcription.
            print(f"[chunk] ошибка: {exc!r} — продолжаю", file=sys.stderr)
            buffer = np.empty((0, channel_count or 1), dtype=np.float32)
            trailing_silence_ms = 0.0
            time.sleep(0.1)
            continue


def transcribe_translate_worker(
    chunk_q,
    translation_q,
    overlay,
    stop_event,
    samplerate,
    settings,
    sentence_mode,
    flush_after,
    min_block_chars,
    max_block_chars,
    max_sentences,
    mutable_tail_seconds,
    endpointing_ms,
    endpoint_min_words,
    max_partial_seconds,
    word_timestamps,
    reset_gen=None,
):
    import mlx_whisper

    last_text = ""
    tail_history = []
    sentence_buffer = ""
    buffer_started_at = None
    recent_context = ""  # last committed/emitted text, fed to Whisper as initial_prompt
    seen_gen = reset_gen[0] if reset_gen is not None else 0

    def maybe_log_slow(label, started_at, extra=""):
        elapsed = time.monotonic() - started_at
        if elapsed >= SLOW_STEP_LOG_SECONDS:
            suffix = f" {extra}" if extra else ""
            print(f"[{label}] медленно: {elapsed:.1f}s{suffix}", file=sys.stderr)

    def note_context(text):
        # Keep the most recent ~240 chars of finalized text to condition the next chunk.
        nonlocal recent_context
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            recent_context = f"{recent_context} {text}".strip()[-240:]

    def refresh_tail(now):
        nonlocal tail_history
        tail_history = [
            (stamp, value)
            for stamp, value in tail_history
            if now - stamp <= mutable_tail_seconds
        ]
        return " ".join(value for _, value in tail_history)[-600:]

    def emit_final_blocks(force=False, pause_ms=0.0):
        nonlocal sentence_buffer, buffer_started_at
        previous_buffer = sentence_buffer
        if force:
            blocks, sentence_buffer = take_endpoint_blocks(
                sentence_buffer,
                min_chars=min_block_chars,
                max_chars=max_block_chars,
                min_words=endpoint_min_words,
            )
        else:
            blocks, sentence_buffer = take_confirmed_blocks_for_translation(
                sentence_buffer,
                min_chars=min_block_chars,
                max_chars=max_block_chars,
                max_sentences=max_sentences,
            )
        for source_text in blocks:
            source_text = sentence_case_text(source_text)
            enqueue_translation(translation_q, source_text, pause_ms=pause_ms)
            note_context(source_text)
        overlay.post_partial(sentence_case_text(sentence_buffer))
        if not sentence_buffer:
            buffer_started_at = None
        elif blocks or sentence_buffer != previous_buffer or buffer_started_at is None:
            buffer_started_at = time.monotonic()

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            # Clear button pressed: wipe all rolling state and pending chunks so
            # transcription/translation truly start from a clean slate.
            seen_gen = reset_gen[0]
            last_text = ""
            tail_history.clear()
            sentence_buffer = ""
            buffer_started_at = None
            recent_context = ""
            while True:
                try:
                    chunk_q.get_nowait()
                except queue.Empty:
                    break
            while True:
                try:
                    translation_q.get_nowait()
                except queue.Empty:
                    break

        if chunk_q.maxsize and chunk_q.qsize() >= chunk_q.maxsize:
            dropped = drain_queue_keep_latest(chunk_q, CHUNK_CATCHUP_KEEP_CHUNKS)
            if dropped:
                last_text = ""
                if sentence_buffer.strip():
                    emit_final_blocks(force=True, pause_ms=0.0)
                overlay.post_status(chunk_q.qsize())
                print(
                    f"[transcribe] очередь чанков полная — сброшено {dropped}, беру свежие; буфер фразы сохранён",
                    file=sys.stderr,
                )

        try:
            item = chunk_q.get(timeout=0.2)
        except queue.Empty:
            continue
        overlay.post_status(chunk_q.qsize())

        if isinstance(item, dict):
            audio = item.get("audio")
            is_endpoint = bool(item.get("endpoint"))
            trailing_silence_ms = float(item.get("trailing_silence_ms") or 0.0)
        else:
            audio = item
            is_endpoint = False
            trailing_silence_ms = 0.0

        if audio is None:
            if is_endpoint and sentence_mode:
                emit_final_blocks(force=True, pause_ms=trailing_silence_ms)
            continue

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="live_translate_")
            os.close(fd)
            sf.write(temp_path, audio, int(samplerate), subtype="PCM_16")
            source_language, target_language = settings.get()
            # Feed the recently finalized text as initial_prompt: gives cross-chunk
            # context (punctuation, casing, word continuity at boundaries) without the
            # runaway-hallucination risk of condition_on_previous_text=True. Skip it
            # when that context has no punctuation, or it reinforces Whisper's drift
            # into never punctuating (see punctuated_context).
            context_prompt = punctuated_context(f"{recent_context} {sentence_buffer}")
            whisper_size = settings.get_whisper_size()
            gen_at_start = reset_gen[0] if reset_gen is not None else seen_gen
            transcribe_started = time.monotonic()
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=WHISPER_MODELS[whisper_size],
                language=None if source_language == "auto" else source_language,
                condition_on_previous_text=False,
                initial_prompt=context_prompt,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                temperature=(0.0, 0.2, 0.4),
                word_timestamps=word_timestamps,
                verbose=False,
            )
            maybe_log_slow("whisper", transcribe_started, f"size={whisper_size}")
            if reset_gen is not None and reset_gen[0] != gen_at_start:
                # Clear was pressed while this chunk was inside Whisper — drop its result.
                continue
            if word_timestamps:
                last_word_end = last_word_end_seconds(result)
                if last_word_end is not None:
                    audio_duration = len(audio) / samplerate
                    word_gap_ms = max(0.0, (audio_duration - last_word_end) * 1000.0)
                    trailing_silence_ms = max(trailing_silence_ms, word_gap_ms)
                    is_endpoint = is_endpoint or trailing_silence_ms >= endpointing_ms

            text = re.sub(r"\s+", " ", result.get("text", "")).strip()
            text = strip_hallucinations(text)
            if len(text) < 2:
                continue
            if text == last_text:
                continue
            last_text = text

            now = time.monotonic()
            last_tail = refresh_tail(now)
            text = merge_overlap_text(last_tail, text)
            if not text:
                continue
            tail_history.append((now, text))

            if sentence_mode:
                sentence_buffer = merge_partial_buffer(sentence_buffer, text)
                if sentence_buffer and buffer_started_at is None:
                    buffer_started_at = now
                overlay.post_partial(sentence_case_text(sentence_buffer))
                buffer_age = (now - buffer_started_at) if buffer_started_at else 0.0
                stable_timeout = max_partial_seconds > 0 and buffer_age >= max_partial_seconds
                too_long = len(sentence_buffer) >= max_block_chars
                if is_endpoint or stable_timeout or too_long:
                    emit_final_blocks(force=True, pause_ms=trailing_silence_ms)
                else:
                    emit_final_blocks(force=False)
            else:
                enqueue_translation(translation_q, sentence_case_text(text))
                note_context(text)
        except Exception as exc:
            overlay.post_pair("", f"Ошибка: {exc}")
            time.sleep(1.0)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)


def translation_worker(translation_q, overlay, translator, stop_event, settings, reset_gen=None):
    seen_gen = reset_gen[0] if reset_gen is not None else 0
    last_source = ""

    def maybe_log_slow(label, started_at, extra=""):
        elapsed = time.monotonic() - started_at
        if elapsed >= SLOW_STEP_LOG_SECONDS:
            suffix = f" {extra}" if extra else ""
            print(f"[{label}] медленно: {elapsed:.1f}s{suffix}", file=sys.stderr)

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            seen_gen = reset_gen[0]
            last_source = ""
            while True:
                try:
                    translation_q.get_nowait()
                except queue.Empty:
                    break

        if translation_q.maxsize and translation_q.qsize() >= translation_q.maxsize:
            dropped = drain_queue_keep_latest(translation_q, TRANSLATION_QUEUE_KEEP_TASKS)
            if dropped:
                print(
                    f"[translate] очередь перевода полная — сброшено {dropped}, беру свежие",
                    file=sys.stderr,
                )

        try:
            item = translation_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if not isinstance(item, dict):
            continue
        source_text = str(item.get("source") or "").strip()
        pause_ms = float(item.get("pause_ms") or 0.0)
        task_source_language = str(item.get("source_language") or "").strip()
        if not source_text:
            continue

        should_replace_tail = should_retranslate_merged_source(
            last_source,
            source_text,
            pause_ms=pause_ms,
        )
        combined_source = (
            f"{last_source.rstrip()} {source_text}".strip()
            if should_replace_tail
            else source_text
        )

        gen_at_start = reset_gen[0] if reset_gen is not None else seen_gen
        source_language, target_language = settings.get()
        translator.set_target(target_language)
        # Prefer the source language resolved by the producer (Whisper detection); fall back
        # to the configured one (possibly "auto") when the task didn't carry it.
        translator.set_source(task_source_language or source_language)
        translator.set_model(settings.get_ollama_model())
        try:
            translate_started = time.monotonic()
            max_tokens = translation_token_budget(
                combined_source,
                getattr(translator, "max_tokens", 180),
            )
            translated = translator.translate(combined_source, max_tokens=max_tokens)
            maybe_log_slow(
                "translate",
                translate_started,
                f"model={settings.get_ollama_model()} chars={len(combined_source)} max_tokens={max_tokens}",
            )
            if reset_gen is not None and reset_gen[0] != gen_at_start:
                continue
            if translated:
                overlay.post_pair(
                    source_text,
                    translated,
                    pause_ms=pause_ms,
                    replace_translation_tail=should_replace_tail,
                    combined_source=combined_source,
                )
                last_source = combined_source
        except Exception as exc:
            overlay.post_pair("", f"Ошибка: {exc}")
            time.sleep(1.0)


def streaming_worker(
    audio_q,
    translation_q,
    overlay,
    translator,
    stop_event,
    samplerate,
    settings,
    min_block_chars,
    max_block_chars,
    max_sentences,
    endpoint_min_words,
    endpointing_ms,
    silence_rms=0.006,
    reset_gen=None,
    use_vad=True,
    vad_min_speech_ms=250.0,
    update_seconds=1.0,
    block_seconds=0.2,
):
    """LocalAgreement-2 streaming transcription + translation."""
    import mlx_whisper

    proc = OnlineProcessor(samplerate)
    committed_text = ""   # confirmed words not yet emitted as a full sentence
    recent_context = ""   # fed to Whisper as initial_prompt
    detected_lang = "auto"  # Whisper's detected source language (for TranslateGemma)
    pending_seconds = 0.0
    trailing_silence_ms = 0.0
    endpoint_sent = False
    seen_gen = reset_gen[0] if reset_gen is not None else 0
    vad_model, get_speech_timestamps = load_vad() if use_vad else (None, None)
    fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="live_stream_")
    os.close(fd)

    def note_context(text):
        nonlocal recent_context
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            recent_context = f"{recent_context} {text}".strip()[-240:]

    def transcribe_words(audio):
        nonlocal detected_lang
        source_language, _ = settings.get()
        sf.write(temp_path, audio, int(samplerate), subtype="PCM_16")
        context = punctuated_context(f"{recent_context} {committed_text}")
        result = mlx_whisper.transcribe(
            temp_path,
            path_or_hf_repo=WHISPER_MODELS[settings.get_whisper_size()],
            language=None if source_language == "auto" else source_language,
            condition_on_previous_text=False,
            initial_prompt=context,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            temperature=0.0,  # no fallback retries — re-transcription self-corrects next pass
            word_timestamps=True,
            verbose=False,
        )
        detected_lang = result.get("language") or detected_lang
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                t = (w.get("word") or "").strip()
                if t:
                    words.append((float(w.get("start", 0.0)), float(w.get("end", 0.0)), t))
        return words

    def emit_sentences(force=False, pause_ms=0.0):
        nonlocal committed_text
        committed_text = strip_hallucinations(committed_text)
        blocks, committed_text = take_blocks_for_translation(
            committed_text, min_block_chars, max_block_chars, max_sentences, force=force
        )
        if not blocks and (force or len(committed_text) >= max_block_chars):
            extra, committed_text = take_endpoint_blocks(
                committed_text, min_block_chars, max_block_chars, endpoint_min_words
            )
            blocks += extra
        for block in blocks:
            source_language, _ = settings.get()
            resolved_source = detected_lang if source_language == "auto" else source_language
            src = sentence_case_text(block)
            # Hand the block to the translation worker instead of blocking this audio-reader
            # thread on Ollama; otherwise translation latency stalls transcription and the
            # audio queue overflows.
            enqueue_translation(
                translation_q, src, pause_ms=pause_ms, source_language=resolved_source
            )
            note_context(src)

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            seen_gen = reset_gen[0]
            proc.reset()
            committed_text = ""
            recent_context = ""
            pending_seconds = 0.0
            trailing_silence_ms = 0.0
            endpoint_sent = False
            while True:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break

        # Catch-up: if transcription can't keep pace and the audio queue is piling up,
        # skip the stale backlog and jump back to live. Reset the rolling Whisper state
        # too; otherwise it can keep re-transcribing stale buffered audio and never commit
        # new translation blocks after overload.
        if audio_q.qsize() > STREAM_CATCHUP_HIGH_WATER:
            dropped = 0
            while audio_q.qsize() > STREAM_CATCHUP_KEEP_BLOCKS:
                try:
                    audio_q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                tail = proc.finalize()
                if tail:
                    committed_text = f"{committed_text} {tail}".strip()
                    emit_sentences(force=True, pause_ms=dropped * block_seconds * 1000.0)
                proc.reset()
                pending_seconds = 0.0
                print(
                    f"[stream] отстаём — пропущено ~{dropped * block_seconds:.1f}s аудио, догоняю",
                    file=sys.stderr,
                )
                overlay.post_status(audio_q.qsize())

        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            force_endpoint_process = False
            block_ms = len(block) / samplerate * 1000.0
            block_rms = float(np.sqrt(np.mean(np.square(block.astype(np.float32)))))
            if block_rms < silence_rms:
                trailing_silence_ms += block_ms
                if trailing_silence_ms >= endpointing_ms and not endpoint_sent:
                    if proc.audio is not None and proc.active_seconds() > 0:
                        force_endpoint_process = True
                        pending_seconds = update_seconds
                    else:
                        tail = proc.finalize()
                        if tail:
                            committed_text = f"{committed_text} {tail}".strip()
                        proc.drop_active()
                        pending_seconds = 0.0
                        endpoint_sent = True
                        if committed_text.strip():
                            emit_sentences(force=True, pause_ms=trailing_silence_ms)
                        overlay.post_partial("")
                if endpoint_sent and audio_q.qsize() > NO_SPEECH_KEEP_AUDIO_BLOCKS:
                    drained = drain_queue_keep_latest(audio_q, NO_SPEECH_KEEP_AUDIO_BLOCKS)
                    if drained:
                        print(
                            f"[stream] нет речи — сброшено ~{drained * block_ms / 1000.0:.1f}s тишины",
                            file=sys.stderr,
                        )
                        overlay.post_status(audio_q.qsize())
                if not force_endpoint_process:
                    continue

            if block_rms >= silence_rms:
                trailing_silence_ms = 0.0
                endpoint_sent = False
                proc.insert_audio(block)
                pending_seconds += len(block) / samplerate
            if pending_seconds < update_seconds:
                continue
            pending_seconds = 0.0

            # Skip pure silence/non-speech: don't re-transcribe it (hallucination source).
            if vad_model is not None and proc.audio is not None:
                has_speech, _ = vad_analyze(
                    proc.audio, samplerate, vad_model, get_speech_timestamps, 0, vad_min_speech_ms
                )
                if not has_speech:
                    # Commit the still-unconfirmed tail before dropping the audio, otherwise
                    # the last words of every utterance are lost at the pause.
                    tail = proc.finalize()
                    if tail:
                        committed_text = f"{committed_text} {tail}".strip()
                    proc.drop_active()
                    pending_seconds = 0.0
                    drained = drain_queue_keep_latest(audio_q, NO_SPEECH_KEEP_AUDIO_BLOCKS)
                    if drained:
                        print(
                            f"[stream] нет речи — сброшено ~{drained * block_seconds:.1f}s аудио",
                            file=sys.stderr,
                        )
                        overlay.post_status(audio_q.qsize())
                    if committed_text.strip():
                        emit_sentences(force=True, pause_ms=1000)  # pause => flush sentence
                    overlay.post_partial("")
                    continue

            gen_before = reset_gen[0] if reset_gen is not None else seen_gen
            committed, draft = proc.process(transcribe_words)
            if reset_gen is not None and reset_gen[0] != gen_before:
                continue
            for (_, _, t) in committed:
                committed_text = f"{committed_text} {t}".strip()

            if force_endpoint_process:
                tail = proc.finalize()
                if tail:
                    committed_text = f"{committed_text} {tail}".strip()
                proc.drop_active()
                endpoint_sent = True
                if committed_text.strip():
                    emit_sentences(force=True, pause_ms=trailing_silence_ms)
                overlay.post_partial("")
                overlay.post_status(audio_q.qsize())
                continue

            emit_sentences(force=False)

            draft_clean = strip_hallucinations(draft)
            live = f"{committed_text} {draft_clean}".strip()
            overlay.post_partial(sentence_case_text(live))
            overlay.post_status(audio_q.qsize())
        except Exception as exc:
            print(f"[stream] ошибка: {exc!r} — продолжаю", file=sys.stderr)
            time.sleep(0.2)

    Path(temp_path).unlink(missing_ok=True)


def build_translator(args):
    return OllamaTranslator(
        model=args.ollama_model,
        target=args.target,
        url=args.ollama_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning=args.reasoning,
        source=args.source,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time speech transcription + local LLM translation + macOS glass overlay."
    )
    p.add_argument("--list", action="store_true", help="показать аудио-устройства и выйти")
    p.add_argument("--device", type=int, default=None, help="номер входного аудио-устройства")
    p.add_argument("--source", default="auto", help="язык речи: auto/en/ru/es/... (по умолчанию auto)")
    p.add_argument("--target", default="ru", help="язык перевода: ru/en/es/... (по умолчанию ru)")
    p.add_argument(
        "--whisper",
        choices=WHISPER_MODELS.keys(),
        default="medium",
        help="MLX Whisper размер: small/medium/large",
    )
    p.add_argument(
        "--translator",
        choices=("ollama",),
        default="ollama",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ollama-model",
        choices=GEMMA_MODELS.keys(),
        default="gemma4:12b-mlx",
        help="модель Ollama Gemma 4",
    )
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="адрес Ollama")
    p.add_argument("--chunk-seconds", type=float, default=6.0, help="МАКС размер аудио-чанка (окна)")
    p.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=1.2,
        help="минимум аудио перед тем, как резать чанк на паузе",
    )
    p.add_argument(
        "--pause-emit-ms",
        type=float,
        default=350.0,
        help="пауза (мс), на которой эмитить чанк, не дожидаясь МАКС окна",
    )
    p.add_argument("--overlap-seconds", type=float, default=0.75, help="overlap между чанками")
    p.add_argument("--block-seconds", type=float, default=0.20, help="размер блока записи")
    p.add_argument(
        "--endpointing-ms",
        type=float,
        default=1000.0,
        help="сколько мс тишины считать концом utterance",
    )
    p.add_argument(
        "--endpoint-min-words",
        type=int,
        default=8,
        help="минимум слов для перевода на endpoint даже без пунктуации",
    )
    p.add_argument(
        "--max-partial-seconds",
        type=float,
        default=0.0,
        help="максимально держать live partial перед stable commit; 0 = ждать смысловой границы",
    )
    p.add_argument(
        "--mutable-tail-seconds",
        type=float,
        default=8.0,
        help="сколько секунд recent text держать изменяемым для overlap-дедупликации",
    )
    p.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="не использовать word timestamps Whisper для endpointing",
    )
    p.add_argument(
        "--audio-queue",
        type=int,
        default=DEFAULT_AUDIO_QUEUE_BLOCKS,
        help=(
            "буфер сырых аудио-блоков; при переполнении старое аудио пропускается, "
            "чтобы перевод восстановился; 0 = без лимита"
        ),
    )
    p.add_argument("--chunk-queue", type=int, default=8, help="буфер чанков на транскрибацию")
    p.add_argument(
        "--translation-queue",
        type=int,
        default=4,
        help="буфер текстовых блоков на перевод; при переполнении старые задачи пропускаются",
    )
    p.add_argument(
        "--no-sentence-mode",
        action="store_true",
        help="переводить каждый распознанный чанк сразу, без ожидания конца предложения",
    )
    p.add_argument(
        "--flush-after",
        type=float,
        default=0.0,
        help="legacy fallback; endpointing-ms является основным механизмом финализации",
    )
    p.add_argument(
        "--min-sentence-chars",
        type=int,
        default=20,
        help="совместимость: если --min-block-chars не задан, используется как минимум блока",
    )
    p.add_argument(
        "--min-block-chars",
        type=int,
        default=90,
        help="минимальный размер смыслового блока перед переводом",
    )
    p.add_argument(
        "--max-block-chars",
        type=int,
        default=1200,
        help="максимальный размер смыслового блока перед переводом",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="максимум предложений за один вызов переводчика",
    )
    p.add_argument("--silence-rms", type=float, default=0.006, help="порог тишины (строже = выше)")
    p.add_argument(
        "--no-vad",
        action="store_true",
        help="не выравнивать нарезку по тишине через Silero VAD (фиксированные чанки)",
    )
    p.add_argument(
        "--vad-min-speech-ms",
        type=float,
        default=250.0,
        help="минимум суммарной речи в чанке (мс), иначе чанк считается без речи и пропускается",
    )
    p.add_argument(
        "--legacy-chunking",
        action="store_true",
        help="старая нарезка на чанки вместо LocalAgreement-стриминга",
    )
    p.add_argument(
        "--update-seconds",
        type=float,
        default=1.5,
        help="как часто (с накопленного аудио) пере-распознавать буфер в LocalAgreement-режиме",
    )
    p.add_argument("--max-tokens", type=int, default=180, help="лимит токенов перевода")
    p.add_argument("--temperature", type=float, default=0.1, help="температура LLM")
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="включить reasoning/thinking у Ollama-модели; по умолчанию выключено",
    )
    p.add_argument("--width", type=int, default=1100, help="ширина окна")
    p.add_argument("--height", type=int, default=520, help="высота окна")
    p.add_argument("--opacity", type=float, default=0.92, help="прозрачность окна")
    p.add_argument(
        "--show-partial",
        action="store_true",
        help="показывать live draft до финализации; по умолчанию UI показывает только завершённые блоки",
    )
    p.add_argument("--no-window", action="store_true", help="выводить в терминал вместо окна")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_devices()
        return

    device = args.device
    if device is None:
        device, name = find_blackhole()
        if device is None:
            sys.exit(
                "Не нашёл BlackHole. Установи `brew install blackhole-2ch`, "
                "настрой Multi-Output Device или передай --device N после --list."
            )
        print(f"Найдено устройство: [{device}] {name}")

    info = sd.query_devices(device)
    samplerate = float(info["default_samplerate"])
    channels = min(2, int(info["max_input_channels"])) or 1

    stop_event = threading.Event()
    reset_gen = [0]  # bumped by the Clear button to wipe both workers' state
    settings = LanguageSettings(
        args.source,
        args.target,
        whisper_size=args.whisper,
        ollama_model=args.ollama_model,
    )
    min_block_chars = args.min_block_chars or args.min_sentence_chars
    audio_q = queue.Queue(maxsize=args.audio_queue)
    chunk_q = queue.Queue(maxsize=args.chunk_queue)
    translation_q = queue.Queue(maxsize=args.translation_queue)

    print("Загружаю переводчик...")
    translator = build_translator(args)
    print(
        f"Старт: {info['name']} -> Whisper {args.whisper} -> "
        f"Ollama {args.ollama_model} -> {args.target}"
    )

    overlay = (
        ConsoleOverlay(stop_event, show_partial=args.show_partial)
        if args.no_window
        else GlassOverlay(
            stop_event,
            settings=settings,
            title="Live Translation",
            width=args.width,
            height=args.height,
            opacity=args.opacity,
            show_partial=args.show_partial,
        )
    )
    overlay.reset_gen = reset_gen
    overlay.translator = translator  # lets the model menu unload the old model immediately

    workers = [
        threading.Thread(
            target=audio_capture_worker,
            args=(audio_q, stop_event, device, samplerate, channels, args.block_seconds),
            daemon=True,
        ),
        threading.Thread(
            target=translation_worker,
            args=(translation_q, overlay, translator, stop_event, settings, reset_gen),
            daemon=True,
        ),
    ]
    if args.legacy_chunking:
        workers += [
            threading.Thread(
                target=chunk_worker,
                args=(
                    audio_q,
                    chunk_q,
                    overlay,
                    stop_event,
                    samplerate,
                    args.chunk_seconds,
                    args.overlap_seconds,
                    args.endpointing_ms,
                    args.silence_rms,
                    not args.no_vad,
                    args.min_chunk_seconds,
                    args.pause_emit_ms,
                    reset_gen,
                    args.vad_min_speech_ms,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=transcribe_translate_worker,
                args=(
                    chunk_q,
                    translation_q,
                    overlay,
                    stop_event,
                    samplerate,
                    settings,
                    not args.no_sentence_mode,
                    args.flush_after,
                    min_block_chars,
                    args.max_block_chars,
                    args.max_sentences,
                    args.mutable_tail_seconds,
                    args.endpointing_ms,
                    args.endpoint_min_words,
                    args.max_partial_seconds,
                    not args.no_word_timestamps,
                    reset_gen,
                ),
                daemon=True,
            ),
        ]
    else:
        workers.append(
            threading.Thread(
                target=streaming_worker,
                args=(
                    audio_q,
                    translation_q,
                    overlay,
                    translator,
                    stop_event,
                    samplerate,
                    settings,
                    min_block_chars,
                    args.max_block_chars,
                    args.max_sentences,
                    args.endpoint_min_words,
                    args.endpointing_ms,
                    args.silence_rms,
                    reset_gen,
                    not args.no_vad,
                    args.vad_min_speech_ms,
                    args.update_seconds,
                    args.block_seconds,
                ),
                daemon=True,
            )
        )

    def handle_signal(signum, frame):
        overlay.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for worker in workers:
        worker.start()

    try:
        overlay.run()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
