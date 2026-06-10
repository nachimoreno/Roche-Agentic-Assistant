"""
llm.py
------
LLM provider seam.

`LLMClient` is the interface every consumer (ConversationLayer, RAGAgent)
talks to. `GroqClient` is the default implementation. Swapping to
Anthropic/OpenAI/self-hosted later is one new class — consumers don't change.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Protocol, Sequence

from groq import Groq


logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Return a JSON payload matching `schema`."""
        ...

    def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Yield plain-text deltas as the model produces them.

        Unlike `complete_structured`, the caller is responsible for any
        output structure via the system prompt (the response is not forced
        into JSON mode, so tokens can be streamed as they arrive).
        """
        ...


class GroqClient:
    """Default `LLMClient` backed by Groq + Llama 3."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        system_with_schema = (
            f"{system}\n\n"
            "Return ONLY a single JSON object matching this schema:\n"
            f"{json.dumps(schema)}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_with_schema}]
        messages.extend(history)
        messages.append({"role": "user", "content": user})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        raw = (response.choices[0].message.content or "").strip()
        logger.debug("llm.call.done", extra={"model": self._model, "bytes": len(raw)})

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GroqClient: model returned non-JSON output.\nRaw: {raw!r}"
            ) from exc

    def stream_text(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[dict[str, str]] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user})

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        n = 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                n += len(delta)
                yield delta
        logger.debug("llm.stream.done", extra={"model": self._model, "bytes": n})
