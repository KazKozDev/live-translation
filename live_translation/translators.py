"""Local translation backends used by the live overlay."""

import contextlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request

from live_translation.text_pipeline import (
    live_translation_messages,
    strip_llm_noise,
)


class LanguageSettings:
    def __init__(self, source, target, whisper_size="turbo", ollama_model="gemma4:26b-mlx"):
        self._lock = threading.Lock()
        self._source = source
        self._target = target
        self._whisper_size = whisper_size
        self._ollama_model = ollama_model

    def get(self):
        with self._lock:
            return self._source, self._target

    def get_whisper_size(self) -> str:
        with self._lock:
            return self._whisper_size

    def get_ollama_model(self) -> str:
        with self._lock:
            return self._ollama_model

    def set_source(self, source):
        with self._lock:
            self._source = source

    def set_target(self, target):
        with self._lock:
            self._target = target

    def set_whisper_size(self, whisper_size):
        with self._lock:
            self._whisper_size = whisper_size

    def set_ollama_model(self, ollama_model):
        with self._lock:
            self._ollama_model = ollama_model


class OllamaTranslator:
    def __init__(
        self,
        model,
        target,
        url,
        max_tokens,
        temperature,
        reasoning,
        source="auto",
        num_ctx=4096,
    ):
        self.model = model
        self.target = target
        self.source = source
        self.chat_url = url.rstrip("/") + "/api/chat"
        self.generate_url = url.rstrip("/") + "/api/generate"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.num_ctx = num_ctx

    def set_model(self, model):
        if model and model != self.model:
            # Model switch: free the previous one from VRAM right away instead of letting
            # Ollama keep it resident for keep_alive minutes (which would hold both the old
            # and the new model in memory at once). The new model loads on the next translate.
            previous, self.model = self.model, model
            if previous and previous != model:
                self._unload(previous)

    def _unload(self, model):
        """Best-effort: ask Ollama to drop a model from memory (keep_alive=0)."""
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.generate_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
            if raw:
                body = json.loads(raw.decode("utf-8"))
                done_reason = body.get("done_reason")
                if done_reason not in (None, "unload"):
                    print(
                        f"[translate] Ollama did not confirm unload of {model}: {done_reason}",
                        file=sys.stderr,
                    )
        except urllib.error.URLError as exc:
            print(f"[translate] failed to unload {model}: {exc}", file=sys.stderr)

    def unload_models_except(self, models, keep_model):
        for model in models:
            if model and model != keep_model:
                self._unload(model)

    def set_target(self, target):
        self.target = target

    def set_source(self, source):
        self.source = source

    def translate(self, text, max_tokens=None, on_delta=None, history=None):
        num_predict = int(max_tokens or self.max_tokens)
        options = {
            "temperature": self.temperature,
            "num_predict": num_predict,
            "top_p": 0.8,
            "top_k": 20,
        }
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        # Stream tokens as they're generated so the UI can show the translation arriving
        # instead of waiting for the whole block. Disabled when reasoning is on (thinking
        # tokens would interleave) or when the caller doesn't want partials.
        stream = on_delta is not None and not self.reasoning
        payload = {
            "model": self.model,
            "messages": live_translation_messages(self.source, self.target, text, history),
            "stream": stream,
            "think": bool(self.reasoning),
            "keep_alive": "30m",
            "options": options,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.chat_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            if stream:
                return self._translate_stream(req, on_delta)
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama is not responding. Run `ollama serve` and pull the model: "
                f"`ollama pull {self.model}`."
            ) from exc
        response = body.get("message", {}).get("content", "")
        return strip_llm_noise(response)

    def _translate_stream(self, req, on_delta, throttle_seconds=0.1):
        """Read Ollama's newline-delimited streaming response, forwarding the growing
        translation to on_delta (throttled), and return the final cleaned text."""
        parts = []
        last_emit = 0.0
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        parts.append(delta)
                        now = time.monotonic()
                        if now - last_emit >= throttle_seconds:
                            last_emit = now
                            with contextlib.suppress(Exception):
                                on_delta(strip_llm_noise("".join(parts)))
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama is not responding. Run `ollama serve` and pull the model: "
                f"`ollama pull {self.model}`."
            ) from exc
        final = strip_llm_noise("".join(parts))
        # Throttling may have skipped the last tokens — push the complete text once so the
        # live draft is whole even if the commit that follows is briefly delayed.
        with contextlib.suppress(Exception):
            on_delta(final)
        return final
