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
    }
    messages = captured["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith("Source language: English (en)")
    assert "/no_think" not in json.dumps(messages)


def test_ollama_translategemma_keeps_generate_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"response": "hola"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = OllamaTranslator(
        model="translategemma:12b",
        target="es",
        url="http://127.0.0.1:11434",
        max_tokens=180,
        temperature=0.1,
        reasoning=False,
        source="en",
    )

    assert translator.translate("hello") == "hola"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert "prompt" in captured["payload"]
    assert "messages" not in captured["payload"]


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
