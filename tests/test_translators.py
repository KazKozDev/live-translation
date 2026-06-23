import json

from live_translation.translators import OllamaTranslator


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


def test_ollama_qwen_uses_chat_api_and_disables_thinking(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "<think>skip</think>Привет"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="qwen3.5:9b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=220,
        temperature=0.1,
        reasoning=False,
        source="en",
    )

    assert translator.translate("hello") == "Привет"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {
        "temperature": 0.1,
        "num_predict": 220,
        "top_p": 0.8,
        "top_k": 20,
        "num_ctx": 4096,
    }
    messages = captured["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith("Source language: English (en)")
    assert "/no_think" not in json.dumps(messages)


class FakeStreamResponse:
    def __init__(self, chunks):
        # Each chunk is serialized as one NDJSON line, the way Ollama streams.
        self.lines = [json.dumps(c).encode("utf-8") + b"\n" for c in chunks]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_ollama_translate_streams_tokens_to_callback(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeStreamResponse(
            [
                {"message": {"content": "При"}},
                {"message": {"content": "вет"}},
                {"message": {"content": " мир"}, "done": True},
            ]
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="gemma4:12b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="en",
    )

    deltas = []
    # throttle_seconds=0 so every token fires the callback in the test
    result = translator.translate(
        "hello world", on_delta=lambda t: deltas.append(t)
    )

    assert result == "Привет мир"
    assert captured["payload"]["stream"] is True
    assert deltas[-1] == "Привет мир"
    assert deltas == sorted(deltas, key=len)  # text only grows


def test_ollama_translate_passes_history_as_prior_turns(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "продолжение"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="gemma4:12b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="en",
    )

    out = translator.translate(
        "the next part", history=[("the first part", "первая часть")]
    )

    assert out == "продолжение"
    messages = captured["payload"]["messages"]
    # system, prior user, prior assistant, current user
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "the first part" in messages[1]["content"]
    assert messages[2]["content"] == "первая часть"
    assert messages[3]["content"].endswith("the next part")


def test_ollama_translate_accepts_per_call_token_override(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "готово"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="gemma4:26b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="es",
    )

    assert translator.translate("hola", max_tokens=640) == "готово"
    assert captured["payload"]["options"]["num_predict"] == 640
    assert captured["payload"]["options"]["num_ctx"] == 4096


def test_ollama_translate_can_leave_context_to_server_default(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "готово"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="gemma4:26b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="es",
        num_ctx=0,
    )

    assert translator.translate("hola") == "готово"
    assert "num_ctx" not in captured["payload"]["options"]


def test_ollama_unload_uses_empty_generate_request(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"done": True, "done_reason": "unload"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="gemma4:26b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="es",
    )

    translator._unload("gemma4:12b-mlx")

    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 10
    assert captured["payload"] == {
        "model": "gemma4:12b-mlx",
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }


def test_ollama_unloads_known_models_except_selected(monkeypatch):
    unloaded = []

    def fake_urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        unloaded.append(payload["model"])
        return FakeResponse({"done": True, "done_reason": "unload"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="gemma4:12b-mlx",
        target="ru",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="es",
    )

    translator.unload_models_except(
        ["gemma4:26b-mlx", "gemma4:e4b-mlx", "gemma4:12b-mlx"],
        "gemma4:12b-mlx",
    )

    assert unloaded == ["gemma4:26b-mlx", "gemma4:e4b-mlx"]
