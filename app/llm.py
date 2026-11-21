"""Model providers behind one streaming interface.

stream_answer(system, user) yields ("text", chunk) items, then one ("stop", reason) item.
reason is "end", "refusal" (safety filter declined) or "length".
Provider errors are mapped to AuthError / ServiceError so the chat flow handles both providers the same way.
"""
from __future__ import annotations

import os
from typing import Iterator

from . import config


class AuthError(Exception):
    pass


class ServiceError(Exception):
    pass


class Unsupported(Exception):
    """The configured provider has no adapter for this call; callers fall back to rules."""


def provider() -> str | None:
    forced = os.environ.get("NOVACELL_PROVIDER")
    if forced:
        return forced
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE"):
        return "anthropic"
    return None


def model_name() -> str | None:
    return {"groq": config.GROQ_MODEL, "anthropic": config.MODEL}.get(provider() or "")


_clients: dict[str, object] = {}


def _groq_client():
    if "groq" not in _clients:
        from groq import Groq
        _clients["groq"] = Groq(timeout=30.0)
    return _clients["groq"]


def _anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(timeout=30.0)
    return _clients["anthropic"]


def _stream_groq(system: str, user: str) -> Iterator[tuple[str, str]]:
    import groq
    try:
        stream = _groq_client().chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_completion_tokens=4096,  # includes the model's hidden reasoning tokens
            reasoning_effort=config.GROQ_REASONING_EFFORT,
            reasoning_format="hidden",  # never stream the model's reasoning to the customer
            stream=True,
        )
        finish = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                yield "text", choice.delta.content
            if choice.finish_reason:
                finish = choice.finish_reason
    except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
        raise AuthError(str(e)) from e
    except (groq.RateLimitError, groq.APIStatusError, groq.APIConnectionError) as e:
        raise ServiceError(type(e).__name__) from e
    yield "stop", {"content_filter": "refusal", "length": "length"}.get(finish, "end")


def _stream_anthropic(system: str, user: str) -> Iterator[tuple[str, str]]:
    import anthropic
    try:
        with _anthropic_client().beta.messages.stream(
            model=config.MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": config.EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            for chunk in stream.text_stream:
                yield "text", chunk
            final = stream.get_final_message()
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise AuthError(str(e)) from e
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        raise ServiceError(type(e).__name__) from e
    yield "stop", {"refusal": "refusal", "max_tokens": "length"}.get(final.stop_reason, "end")


def _groq_call(**kwargs):
    import groq
    try:
        return _groq_client().chat.completions.create(
            model=config.GROQ_MODEL, temperature=0, reasoning_effort=config.GROQ_REASONING_EFFORT,
            reasoning_format="hidden", **kwargs)
    except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
        raise AuthError(str(e)) from e
    except (groq.RateLimitError, groq.APIStatusError, groq.APIConnectionError) as e:
        raise ServiceError(type(e).__name__) from e


def complete_json(system: str, user: str, max_tokens: int = 600) -> dict:
    """One structured (JSON object) completion. Used for intent detection."""
    import json
    if provider() != "groq":
        raise Unsupported(provider() or "none")
    resp = _groq_call(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      response_format={"type": "json_object"}, max_completion_tokens=max_tokens)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError as e:
        raise ServiceError("invalid_json") from e


def chat_with_tools(messages: list[dict], tools: list[dict], max_tokens: int = 1500):
    """One tool-calling turn. Returns the assistant message (with .content and .tool_calls)."""
    if provider() != "groq":
        raise Unsupported(provider() or "none")
    resp = _groq_call(messages=messages, tools=tools, tool_choice="auto", parallel_tool_calls=False,
                      max_completion_tokens=max_tokens)
    return resp.choices[0].message


def stream_answer(system: str, user: str) -> Iterator[tuple[str, str]]:
    p = provider()
    if p == "groq":
        return _stream_groq(system, user)
    if p == "anthropic":
        return _stream_anthropic(system, user)
    raise AuthError("no model provider configured")
