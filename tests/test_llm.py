"""Provider adapters, with each SDK client replaced by a fake."""
from contextlib import contextmanager
from types import SimpleNamespace

import groq
import httpx
import pytest

from app import llm


def chunk(content=None, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)])


class FakeGroq:
    def __init__(self, chunks=None, error=None):
        self.chunks, self.error, self.calls = chunks or [], error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return iter(self.chunks)


@pytest.fixture
def groq_fake(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(llm, "_clients", {})

    def install(**kw):
        f = FakeGroq(**kw)
        monkeypatch.setitem(llm._clients, "groq", f)
        return f
    return install


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.provider() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.provider() == "anthropic"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm.provider() == "groq" and llm.model_name() == "openai/gpt-oss-120b"
    monkeypatch.setenv("NOVACELL_PROVIDER", "anthropic")
    assert llm.provider() == "anthropic"


def test_groq_streams_text_and_hides_reasoning(groq_fake):
    f = groq_fake(chunks=[chunk("Hello "), chunk("there [FAQ-1]."), chunk(None, "stop")])
    out = list(llm.stream_answer("SYS", "USER"))
    assert out == [("text", "Hello "), ("text", "there [FAQ-1]."), ("stop", "end")]
    kw = f.calls[0]
    assert kw["model"] == "openai/gpt-oss-120b" and kw["stream"] is True
    assert kw["reasoning_format"] == "hidden" and kw["reasoning_effort"] == "low"
    assert kw["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]


@pytest.mark.parametrize("finish,expected", [("content_filter", "refusal"), ("length", "length"), ("stop", "end")])
def test_groq_finish_reasons(groq_fake, finish, expected):
    groq_fake(chunks=[chunk("x", finish)])
    assert list(llm.stream_answer("s", "u"))[-1] == ("stop", expected)


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("err", response=httpx.Response(code, request=req), body=None)


def test_groq_errors_are_mapped(groq_fake):
    groq_fake(error=_status_error(groq.AuthenticationError, 401))
    with pytest.raises(llm.AuthError):
        list(llm.stream_answer("s", "u"))
    groq_fake(error=_status_error(groq.RateLimitError, 429))
    with pytest.raises(llm.ServiceError):
        list(llm.stream_answer("s", "u"))


def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("NOVACELL_PROVIDER", "anthropic")
    calls = []

    @contextmanager
    def stream(**kw):
        calls.append(kw)
        yield SimpleNamespace(text_stream=iter(["Hi [FAQ-1]"]),
                              get_final_message=lambda: SimpleNamespace(stop_reason="refusal"))

    monkeypatch.setattr(llm, "_clients", {"anthropic": SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))})
    out = list(llm.stream_answer("SYS", "USER"))
    assert out == [("text", "Hi [FAQ-1]"), ("stop", "refusal")]
    assert calls[0]["model"] == "claude-opus-5-5" and calls[0]["fallbacks"] == "default"
