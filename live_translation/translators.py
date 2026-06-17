"""Local translation backends used by the live overlay."""

import json
import sys
import threading
import urllib.error
import urllib.request

from live_translation.text_pipeline import (
    live_translation_messages,
    strip_llm_noise,
    translategemma_prompt,
)


class LanguageSettings:
    def __init__(self, source, target, whisper_size="medium", ollama_model="gemma4:26b-mlx"):
        self._lock = threading.Lock()
        self._source = source
        self._target = target
        self._whisper_size = whisper_size
        self._ollama_model = ollama_model

    def get(self):
        with self._lock:
            return self._source, self._target

    def get_whisper_size(self):
        with self._lock:
            return self._whisper_size

    def get_ollama_model(self):
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
    def __init__(self, model, target, url, max_tokens, temperature, reasoning, source="auto"):
        self.model = model
        self.target = target
        self.source = source
        self.chat_url = url.rstrip("/") + "/api/chat"
        self.generate_url = url.rstrip("/") + "/api/generate"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.translategemma = "translategemma" in model.lower() or "translate-gemma" in model.lower()

    def set_model(self, model):
        if model and model != self.model:
            # Model switch: free the previous one from VRAM right away instead of letting
            # Ollama keep it resident for keep_alive minutes (which would hold both the old
            # and the new model in memory at once). The new model loads on the next translate.
            previous, self.model = self.model, model
            self.translategemma = "translategemma" in model.lower() or "translate-gemma" in model.lower()
            if previous and previous != model:
                self._unload(previous)

    def _unload(self, model):
        """Best-effort: ask Ollama to drop a model from memory (keep_alive=0)."""
        payload = {"model": model, "keep_alive": 0}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.generate_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            print(f"[translate] не удалось выгрузить {model}: {exc}", file=sys.stderr)

    def set_target(self, target):
        self.target = target

    def set_source(self, source):
        self.source = source

    def translate(self, text, max_tokens=None):
        num_predict = int(max_tokens or self.max_tokens)
        if self.translategemma and self.source and self.source != "auto":
            prompt = translategemma_prompt(self.source, self.target, text)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": num_predict,
                    "top_p": 0.8,
                    "top_k": 20,
                },
            }
            url = self.generate_url
        else:
            payload = {
                "model": self.model,
                "messages": live_translation_messages(self.source, self.target, text),
                "stream": False,
                "think": bool(self.reasoning),
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": num_predict,
                    "top_p": 0.8,
                    "top_k": 20,
                },
            }
            url = self.chat_url
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama не отвечает. Запусти `ollama serve` и скачай модель: "
                f"`ollama pull {self.model}`."
            ) from exc
        if self.translategemma and self.source and self.source != "auto":
            response = body.get("response", "")
        else:
            response = body.get("message", {}).get("content", "")
        return strip_llm_noise(response)


class MLXTranslator:
    def __init__(self, model_repo, target, max_tokens, temperature, reasoning=False, source="auto"):
        # mlx_lm uses thread-local GPU streams, so import/load on the worker thread.
        self.model_repo = model_repo
        self.target = target
        self.source = source
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.model = None
        self.tokenizer = None
        self.generate = None
        self.make_sampler = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            try:
                from mlx_lm import generate, load
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise RuntimeError("Для --translator mlx установи `pip install mlx-lm`.") from exc
            self.generate = generate
            self.make_sampler = make_sampler
            loaded = load(self.model_repo)
            self.model, self.tokenizer = loaded[0], loaded[1]

    def set_target(self, target):
        self.target = target

    def set_source(self, source):
        self.source = source

    def translate(self, text, max_tokens=None):
        self._ensure_loaded()
        max_tokens = int(max_tokens or self.max_tokens)
        model, tokenizer = self.model, self.tokenizer
        generate, make_sampler = self.generate, self.make_sampler
        assert (
            model is not None
            and tokenizer is not None
            and generate is not None
            and make_sampler is not None
        ), "MLX translator not loaded"
        messages = live_translation_messages(self.source, self.target, text)
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.reasoning,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            system, user = messages
            prompt = f"{system['content']}\n\nUser:\n{user['content']}\n\nAssistant:\n"

        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=self.temperature, top_p=0.8),
            verbose=False,
        )
        return strip_llm_noise(out)
