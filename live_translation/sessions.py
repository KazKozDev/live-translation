"""Persistent session storage and transcript/subtitle exports."""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from pathlib import Path

APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "LiveTranslate"
SESSIONS_DIR = APP_SUPPORT_DIR / "sessions"


def utc_now_iso():
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_datetime_label(iso_value):
    try:
        value = str(iso_value or "").replace("Z", "+00:00")
        dt = _dt.datetime.fromisoformat(value).astimezone()
    except Exception:
        dt = _dt.datetime.now().astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def safe_filename(value, fallback="LiveTranslate"):
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"[^\w .()-]+", "", text, flags=re.UNICODE).strip(" .")
    return text[:120] or fallback


def model_label(code, labels):
    return labels.get(code, code or "")


def make_session_title(source_label, target_label, whisper_label, gemma_label, created_at=None):
    created = local_datetime_label(created_at or utc_now_iso())
    return f"{created}, {source_label} -> {target_label}, {whisper_label} / {gemma_label}"


def new_session(
    *,
    title=None,
    source_language="auto",
    target_language="ru",
    whisper_model="turbo",
    gemma_model="gemma4:26b-mlx",
    source_label="Auto",
    target_label="Russian",
    whisper_label="Whisper Turbo",
    gemma_label="Gemma 4 26B",
):
    created_at = utc_now_iso()
    return {
        "id": uuid.uuid4().hex,
        "title": title
        or make_session_title(source_label, target_label, whisper_label, gemma_label, created_at),
        "created_at": created_at,
        "updated_at": created_at,
        "source_language": source_language,
        "target_language": target_language,
        "whisper_model": whisper_model,
        "gemma_model": gemma_model,
        "duration_seconds": 0.0,
        "phrases": [],
    }


def phrase_from_pair(
    *,
    source="",
    translated="",
    start=None,
    end=None,
    language=None,
    confidence=None,
):
    source = str(source or "").strip()
    translated = str(translated or "").strip()
    start = 0.0 if start is None else max(0.0, float(start))
    if end is None:
        words = max(1, len(re.findall(r"\S+", source)))
        end = start + max(1.0, min(12.0, words * 0.42))
    end = max(start + 0.001, float(end))
    phrase = {
        "start": round(start, 3),
        "end": round(end, 3),
        "source": source,
        "translated": translated,
    }
    if language:
        phrase["language"] = str(language)
    if confidence is not None:
        phrase["confidence"] = float(confidence)
    return phrase


class SessionStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else SESSIONS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id):
        session_id = safe_filename(session_id, "session")
        return self.root / f"{session_id}.json"

    def save(self, session):
        session = dict(session or {})
        if not session.get("id"):
            session["id"] = uuid.uuid4().hex
        phrases = list(session.get("phrases") or [])
        session["phrases"] = phrases
        session["updated_at"] = utc_now_iso()
        session["duration_seconds"] = round(
            max((float(p.get("end") or 0.0) for p in phrases), default=0.0),
            3,
        )
        path = self.path_for(session["id"])
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
        return session

    def load(self, session_id):
        return json.loads(self.path_for(session_id).read_text(encoding="utf-8"))

    def delete(self, session_id):
        self.path_for(session_id).unlink(missing_ok=True)

    def list(self):
        sessions = []
        for path in sorted(self.root.glob("*.json")):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sessions.append(session)
        sessions.sort(key=lambda s: s.get("updated_at") or s.get("created_at") or "", reverse=True)
        return sessions

    def rename(self, session_id, title):
        session = self.load(session_id)
        session["title"] = str(title or "").strip() or session.get("title") or "Untitled"
        return self.save(session)


def session_summary(session, source_labels=None, target_labels=None, whisper_labels=None, gemma_labels=None):
    source_labels = source_labels or {}
    target_labels = target_labels or {}
    whisper_labels = whisper_labels or {}
    gemma_labels = gemma_labels or {}
    source = source_labels.get(session.get("source_language"), session.get("source_language") or "auto")
    target = target_labels.get(session.get("target_language"), session.get("target_language") or "")
    whisper = whisper_labels.get(session.get("whisper_model"), session.get("whisper_model") or "")
    gemma = gemma_labels.get(session.get("gemma_model"), session.get("gemma_model") or "")
    duration = int(round(float(session.get("duration_seconds") or 0.0)))
    minutes = max(1, round(duration / 60)) if duration else 0
    time_label = local_datetime_label(session.get("created_at"))
    return f"{time_label}, {source} -> {target}, {minutes} min, {whisper} / {gemma}"


def export_txt(session):
    original = []
    translated = []
    for phrase in session.get("phrases") or []:
        source = str(phrase.get("source") or "").strip()
        translation = str(phrase.get("translated") or "").strip()
        if source:
            original.append(source)
        if translation:
            translated.append(translation)
    return (
        "TRANSCRIPTION\n\n"
        + "\n\n".join(original)
        + "\n\n\nTRANSLATION\n\n"
        + "\n\n".join(translated)
        + "\n"
    )


def _srt_timestamp(seconds):
    millis = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _vtt_timestamp(seconds):
    return _srt_timestamp(seconds).replace(",", ".")


def export_subtitles(session, field="translated", fmt="srt"):
    fmt = str(fmt or "srt").lower()
    field = "source" if field in {"source", "original", "transcript"} else "translated"
    lines = ["WEBVTT", ""] if fmt == "vtt" else []
    idx = 1
    for phrase in session.get("phrases") or []:
        text = str(phrase.get(field) or "").strip()
        if not text:
            continue
        start = float(phrase.get("start") or 0.0)
        end = max(start + 0.001, float(phrase.get("end") or start + 1.0))
        if fmt == "vtt":
            lines.append(f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}")
        else:
            lines.append(str(idx))
            lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.extend(text.splitlines() or [text])
        lines.append("")
        idx += 1
    return "\n".join(lines).rstrip() + "\n"


def write_session_exports(session, base_path, formats=("txt", "srt", "vtt")):
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    stem = base.with_suffix("")
    if "txt" in formats:
        path = stem.with_suffix(".txt")
        path.write_text(export_txt(session), encoding="utf-8")
        written.append(path)
    for fmt in formats:
        if fmt not in {"srt", "vtt"}:
            continue
        for field, suffix in (("source", "original"), ("translated", "translation")):
            path = stem.parent / f"{stem.name}.{suffix}.{fmt}"
            path.write_text(export_subtitles(session, field=field, fmt=fmt), encoding="utf-8")
            written.append(path)
    return written
