"""Testing helpers for Protocore.

This module provides lightweight fakes that implement core runtime protocols.
"""
from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .types import ApiMode, Message, StreamEvent, ToolDefinition


class FakeLLMClient:
    """Deterministic in-memory fake implementing the LLMClient protocol.

    Example:
        >>> from protocore.testing import FakeLLMClient
        >>> from protocore import Message
        >>> llm = FakeLLMClient(complete_responses=[Message(role="assistant", content="ok")])
    """

    def __init__(
        self,
        *,
        complete_responses: Sequence[Message] | None = None,
        structured_responses: Sequence[Any] | None = None,
        stream_sequences: Sequence[Sequence[StreamEvent]] | None = None,
        default_complete_response: Message | None = None,
        default_structured_response: Any | None = None,
        default_stream_sequence: Sequence[StreamEvent] | None = None,
    ) -> None:
        self._complete_responses: deque[Message] = deque(complete_responses or [])
        self._structured_responses: deque[Any] = deque(structured_responses or [])
        self._stream_sequences: deque[list[StreamEvent]] = deque(
            [list(events) for events in (stream_sequences or [])]
        )
        self._default_complete_response = default_complete_response
        self._default_structured_response = default_structured_response
        self._default_stream_sequence = list(default_stream_sequence or [{"type": "done", "usage": {}}])

        self.complete_calls: list[dict[str, Any]] = []
        self.complete_structured_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def enqueue_complete(self, response: Message) -> None:
        """Append a completion response to the queue."""
        self._complete_responses.append(response)

    def enqueue_structured(self, response: Any) -> None:
        """Append a structured response to the queue."""
        self._structured_responses.append(response)

    def enqueue_stream(self, events: Sequence[StreamEvent]) -> None:
        """Append a stream event sequence to the queue."""
        self._stream_sequences.append(list(events))

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = True,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> Message:
        """Return the next queued message response."""
        self.complete_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "system": system,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "api_mode": api_mode,
                "kwargs": kwargs,
            }
        )
        if self._complete_responses:
            return self._complete_responses.popleft()
        if self._default_complete_response is not None:
            return self._default_complete_response
        raise RuntimeError("FakeLLMClient has no configured response for complete()")

    async def complete_structured(
        self,
        *,
        messages: list[Message],
        schema: type,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the next queued structured response."""
        self.complete_structured_calls.append(
            {
                "messages": messages,
                "schema": schema,
                "system": system,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_mode": api_mode,
                "kwargs": kwargs,
            }
        )
        if self._structured_responses:
            return self._structured_responses.popleft()
        if self._default_structured_response is not None:
            return self._default_structured_response
        raise RuntimeError("FakeLLMClient has no configured response for complete_structured()")

    def stream_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Yield a queued stream sequence of events."""
        self.stream_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "system": system,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_mode": api_mode,
                "kwargs": kwargs,
            }
        )

        events = (
            self._stream_sequences.popleft()
            if self._stream_sequences
            else list(self._default_stream_sequence)
        )

        async def _iterate() -> AsyncIterator[StreamEvent]:
            for event in events:
                yield event

        return _iterate()

    def stream(
        self,
        *,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Alias for simple text streaming without tools."""
        return self.stream_with_tools(
            messages=messages,
            tools=None,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_mode=api_mode,
            **kwargs,
        )
