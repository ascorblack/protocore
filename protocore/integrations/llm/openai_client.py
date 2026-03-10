"""Concrete OpenAI LLM adapter implementing LLMClient protocol.

Supports two API modes:
  - ``responses`` (default): OpenAI Responses API (``client.responses.create``)
  - ``chat_completions``: standard Chat Completions API

Both modes accept ``base_url`` for OpenAI-compatible endpoints (vLLM, etc.).
Structured output is supported via ``complete_structured`` using pydantic schemas.
"""
from __future__ import annotations

import ast
import inspect
import logging
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any, Type, cast
from unittest.mock import Mock

from openai import AsyncOpenAI
from pydantic import ValidationError

from ...constants import DEFAULT_MODEL, MAX_STRUCTURED_JSON_CHARS
from ...json_utils import structured_json_candidates
from ...logging_utils import merge_log_context
from ...orchestrator_utils import serialize_messages_for_api
from ...protocols import LLMClient
from ...types import (
    ApiMode,
    attach_structured_usage,
    ContentPart,
    LLMUsage,
    Message,
    StreamEvent as CoreStreamEvent,
    SubagentResult,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)
StreamEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
StreamEvent = CoreStreamEvent
_MAX_LITERAL_EVAL_ARGUMENT_CHARS = 10_000
_NON_PROVIDER_KWARGS = frozenset(
    {
        "logging_context",
        "tool_definitions",
        "tool_registry",
    }
)


class ToolCallBuffer:
    """Accumulate partial tool call deltas and emit fully assembled calls."""

    def __init__(self) -> None:
        self._buffer: dict[int, dict[str, Any]] = {}
        self._implicit_index_by_key: dict[str, int] = {}
        self._last_implicit_index: int | None = None

    def _next_available_index(self) -> int:
        next_index = 0
        while next_index in self._buffer:
            next_index += 1
        return next_index

    def _resolve_index(self, delta: Any) -> int:
        raw_index = getattr(delta, "index", None)
        if isinstance(raw_index, int) and raw_index >= 0:
            self._last_implicit_index = raw_index
            return raw_index

        delta_id = getattr(delta, "id", None)
        if isinstance(delta_id, str) and delta_id:
            existing = self._implicit_index_by_key.get(delta_id)
            if existing is not None:
                self._last_implicit_index = existing
                return existing

        function_data = getattr(delta, "function", None)
        function_name = getattr(function_data, "name", None)
        if isinstance(function_name, str) and function_name:
            existing = self._implicit_index_by_key.get(f"name:{function_name}")
            if existing is not None:
                self._last_implicit_index = existing
                return existing

        if self._last_implicit_index is not None:
            return self._last_implicit_index

        allocated = self._next_available_index()
        self._last_implicit_index = allocated
        if isinstance(delta_id, str) and delta_id:
            self._implicit_index_by_key[delta_id] = allocated
        if isinstance(function_name, str) and function_name:
            self._implicit_index_by_key[f"name:{function_name}"] = allocated
        logger.warning(
            "chat stream tool_call delta missing index; using fallback slot=%d",
            allocated,
        )
        return allocated

    def add_delta(self, delta: Any) -> None:
        idx = self._resolve_index(delta)
        if idx not in self._buffer:
            self._buffer[idx] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        item = self._buffer[idx]
        if getattr(delta, "id", None):
            if not item["id"]:
                item["id"] = str(delta.id)
            self._implicit_index_by_key[str(delta.id)] = idx
        fn = getattr(delta, "function", None)
        if fn and getattr(fn, "name", None):
            item["function"]["name"] += str(fn.name)
            self._implicit_index_by_key[f"name:{str(fn.name)}"] = idx
        if fn and getattr(fn, "arguments", None):
            item["function"]["arguments"] += str(fn.arguments)

    def as_list(self) -> list[dict[str, Any]]:
        ordered_indexes = sorted(self._buffer)
        if ordered_indexes:
            expected = list(range(ordered_indexes[0], ordered_indexes[-1] + 1))
            if ordered_indexes != expected:
                logger.warning(
                    "chat stream tool_call indexes arrived with gaps: indexes=%s",
                    ordered_indexes,
                )
        return [self._buffer[i] for i in ordered_indexes]


def _pop_logging_context(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract structured log context without forwarding it to the SDK."""
    raw = request_kwargs.pop("logging_context", None)
    if isinstance(raw, dict):
        return merge_log_context(raw)
    return {}


def _log_extra_body_passthrough(*, operation: str, request_kwargs: dict[str, Any]) -> None:
    extra_body = request_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        logger.debug(
            "extra_body passthrough: operation=%s keys=%s payload=%s",
            operation,
            sorted(extra_body.keys()),
            extra_body,
        )


def _ensure_stream_usage_options(request_kwargs: dict[str, Any]) -> None:
    """Enable usage emission for chat streaming providers (for example vLLM)."""
    stream_options = request_kwargs.get("stream_options")
    if stream_options is None:
        # Request usage in final streaming chunk.
        request_kwargs["stream_options"] = {"include_usage": True}
        return
    if isinstance(stream_options, dict):
        merged_stream_options = dict(stream_options)
        merged_stream_options["include_usage"] = True
        request_kwargs["stream_options"] = merged_stream_options


def _filter_provider_kwargs(
    *,
    operation: str,
    create_method: Any,
    request_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Drop runtime-only kwargs before calling the OpenAI SDK."""
    filtered = {
        key: value
        for key, value in request_kwargs.items()
        if key not in _NON_PROVIDER_KWARGS
    }
    try:
        signature = inspect.signature(create_method)
    except (TypeError, ValueError):
        return filtered
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return filtered
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    dropped = sorted(key for key in filtered if key not in allowed)
    if dropped:
        logger.warning(
            "Dropping unsupported provider kwargs: operation=%s keys=%s",
            operation,
            dropped,
        )
    return {key: value for key, value in filtered.items() if key in allowed}


class StructuredOutputValidationError(ValueError):
    """Raised when structured output cannot be validated against schema."""

    def __init__(self, schema_name: str, raw_text: str, reason: str) -> None:
        preview = raw_text[:500]
        super().__init__(
            f"structured_output_schema_validation_failed:"
            f"schema={schema_name}:reason={reason}:raw_preview={preview}"
        )
        self.schema_name = schema_name
        self.raw_preview = preview


class OpenAILLMClient(LLMClient):
    """Production LLMClient backed by openai-python.

f    Args:
        api_key: OpenAI API key.
        base_url: Override endpoint for OpenAI-compatible servers.
        timeout: Optional request timeout (seconds) passed to AsyncOpenAI.
        api_mode: ``"responses"`` (default) or ``"chat_completions"``.
        default_model: Fallback model name when not provided per-call.
        allow_response_fallback_to_chat: Explicitly allow Responses API requests
            to downgrade to Chat Completions when the backend does not support
            Responses API.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str | None = None,
        timeout: float | None = None,
        api_mode: ApiMode = ApiMode.RESPONSES,
        default_model: str = DEFAULT_MODEL,
        allow_response_fallback_to_chat: bool = False,
        emit_reasoning_in_stream: bool = False,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._api_mode = api_mode
        self._default_model = default_model
        self._allow_response_fallback_to_chat = allow_response_fallback_to_chat
        self._emit_reasoning_in_stream = emit_reasoning_in_stream

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
        stream_event_callback: StreamEventCallback | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> Message:
        """Send messages to the LLM and return assistant response.

        Callers may pass ``logging_context={...}`` via ``kwargs`` to attach
        request/session identifiers to adapter log records without sending those
        fields to the provider API.
        """
        request_kwargs = dict(kwargs)
        logging_context = _pop_logging_context(request_kwargs)
        emit_reasoning_in_stream = bool(
            request_kwargs.get("emit_reasoning_in_stream", self._emit_reasoning_in_stream)
        )
        model = model or self._default_model
        temperature = temperature if temperature is not None else 0.7
        max_tokens = max_tokens if max_tokens is not None else 4096
        resolved_mode = api_mode or self._api_mode

        return await self._do_complete(
            messages=messages,
            tools=tools,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_event_callback=stream_event_callback,
            emit_reasoning_in_stream=emit_reasoning_in_stream,
            api_mode=resolved_mode,
            request_kwargs=request_kwargs,
            logging_context=logging_context,
        )

    async def _do_complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
        stream_event_callback: StreamEventCallback | None,
        emit_reasoning_in_stream: bool,
        api_mode: ApiMode,
        request_kwargs: dict[str, Any],
        logging_context: dict[str, Any],
    ) -> Message:
        if api_mode == ApiMode.RESPONSES:
            try:
                return await self._complete_responses(
                    messages=messages,
                    tools=tools,
                    system=system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    stream_event_callback=stream_event_callback,
                    emit_reasoning_in_stream=emit_reasoning_in_stream,
                    request_kwargs=request_kwargs,
                )
            except Exception as exc:
                if not (
                    self._allow_response_fallback_to_chat
                    and _should_fallback_to_chat(exc)
                ):
                    raise
                status_code = _error_status_code(exc)
                logger.warning(
                    "Responses API unsupported, falling back to chat completions: "
                    "status=%s error=%s",
                    status_code,
                    type(exc).__name__,
                    extra=merge_log_context(
                        logging_context,
                        model=model,
                        api_mode=api_mode.value,
                    ),
                )

        return await self._complete_chat(
            messages=messages,
            tools=tools,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_event_callback=stream_event_callback,
            emit_reasoning_in_stream=emit_reasoning_in_stream,
            request_kwargs=request_kwargs,
            logging_context=logging_context,
        )

    async def complete_structured(
        self,
        *,
        messages: list[Message],
        schema: Type[Any],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> Any:
        """Request structured JSON output validated against a pydantic schema.

        Uses the Responses API ``text.format`` or Chat Completions
        ``response_format`` depending on api_mode.
        """
        request_kwargs = dict(kwargs)
        logging_context = _pop_logging_context(request_kwargs)
        model = model or self._default_model
        temperature = temperature if temperature is not None else 0.3
        max_tokens = max_tokens if max_tokens is not None else 4096
        resolved_mode = api_mode or self._api_mode

        raw_text, usage = await self._do_complete_structured(
            messages=messages,
            schema=schema,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_mode=resolved_mode,
            request_kwargs=request_kwargs,
            logging_context=logging_context,
        )

        return attach_structured_usage(_validate_structured_output(raw_text, schema), usage)

    async def _do_complete_structured(
        self,
        *,
        messages: list[Message],
        schema: Type[Any],
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        api_mode: ApiMode,
        request_kwargs: dict[str, Any],
        logging_context: dict[str, Any],
    ) -> tuple[str, LLMUsage | None]:
        if api_mode == ApiMode.RESPONSES:
            try:
                return await self._complete_structured_responses(
                    messages=messages,
                    schema=schema,
                    system=system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_kwargs=request_kwargs,
                )
            except Exception as exc:
                if not (
                    self._allow_response_fallback_to_chat
                    and _should_fallback_to_chat(exc)
                ):
                    raise
                status_code = _error_status_code(exc)
                logger.warning(
                    "Responses API structured output unsupported, falling back to chat "
                    "completions: status=%s error=%s",
                    status_code,
                    type(exc).__name__,
                    extra=merge_log_context(
                        logging_context,
                        model=model,
                        api_mode=api_mode.value,
                        schema=schema.__name__,
                    ),
                )

        return await self._complete_structured_chat(
            messages=messages,
            schema=schema,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_kwargs=request_kwargs,
        )

    async def _complete_structured_chat(
        self,
        *,
        messages: list[Message],
        schema: Type[Any],
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        request_kwargs: dict[str, Any],
    ) -> tuple[str, LLMUsage | None]:
        raw_messages = _build_chat_messages(messages, system)
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": raw_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        }
        chat_kwargs.update(request_kwargs)
        chat_kwargs = _filter_provider_kwargs(
            operation="complete_structured.chat_completions",
            create_method=self._client.chat.completions.create,
            request_kwargs=chat_kwargs,
        )
        _log_extra_body_passthrough(
            operation="complete_structured.chat_completions",
            request_kwargs=chat_kwargs,
        )
        response = await self._client.chat.completions.create(**chat_kwargs)
        choice = _require_chat_choice(response, operation="complete_structured")
        return choice.message.content or "", _extract_usage(response)

    async def _complete_structured_responses(
        self,
        *,
        messages: list[Message],
        schema: Type[Any],
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        request_kwargs: dict[str, Any],
    ) -> tuple[str, LLMUsage | None]:
        input_items = _messages_to_responses_input(messages, system)
        base_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Respect explicit caller overrides. Compatibility retries are only applied
        # for the built-in schema format generated by the adapter.
        if "text" in request_kwargs:
            resp_kwargs = dict(base_kwargs)
            resp_kwargs.update(request_kwargs)
            resp_kwargs = _filter_provider_kwargs(
                operation="complete_structured.responses",
                create_method=self._client.responses.create,
                request_kwargs=resp_kwargs,
            )
            _log_extra_body_passthrough(
                operation="complete_structured.responses",
                request_kwargs=resp_kwargs,
            )
            response = await self._client.responses.create(**resp_kwargs)
            return _extract_responses_text(response), _extract_usage(response)

        base_kwargs.update(request_kwargs)
        last_exc: Exception | None = None
        text_formats = _structured_responses_text_formats(schema)
        for index, text_format in enumerate(text_formats):
            try:
                resp_kwargs = dict(base_kwargs)
                resp_kwargs["text"] = {"format": text_format}
                resp_kwargs = _filter_provider_kwargs(
                    operation="complete_structured.responses",
                    create_method=self._client.responses.create,
                    request_kwargs=resp_kwargs,
                )
                _log_extra_body_passthrough(
                    operation="complete_structured.responses",
                    request_kwargs=resp_kwargs,
                )
                response = await self._client.responses.create(**resp_kwargs)
                return _extract_responses_text(response), _extract_usage(response)
            except Exception as exc:
                last_exc = exc
                is_last_attempt = index == len(text_formats) - 1
                if is_last_attempt or not _should_retry_structured_responses_format(exc):
                    raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("structured_responses_request_failed")

    async def stream_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_registry: Any | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream normalized events from Responses or Chat Completions APIs.

        Event contract:
        - ``{"type": "delta", "kind": "text"|"reasoning", "text": ...}``
        - ``{"type": "tool_calls", "tool_calls": [...]}`` (optional)
        - ``{"type": "done", "usage": {...}}`` (always emitted once at the end)

        Important:
        - Text chunks are emitted with ``type="delta"`` and ``kind="text"``.
          Do not check for ``type == "text"``.

        Example:
            total_text = ""
            async for event in llm.stream_with_tools(messages=messages):
                if event.get("type") == "delta" and event.get("kind") == "text":
                    total_text += event.get("text", "")
                elif event.get("type") == "done":
                    usage = event.get("usage", {})
                    print("final text:", total_text)
                    print("usage:", usage)
        """
        _ = tool_registry
        request_kwargs = dict(kwargs)
        request_kwargs.pop("tool_registry", None)
        logging_context = _pop_logging_context(request_kwargs)
        emit_reasoning_in_stream = bool(
            request_kwargs.pop("emit_reasoning_in_stream", self._emit_reasoning_in_stream)
        )
        model = model or self._default_model
        temperature = temperature if temperature is not None else 0.7
        max_tokens = max_tokens if max_tokens is not None else 4096
        resolved_mode = api_mode or self._api_mode

        if resolved_mode == ApiMode.RESPONSES:
            try:
                input_items = _messages_to_responses_input(messages, system)
                resp_kwargs: dict[str, Any] = {
                    "model": model,
                    "input": input_items,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "stream": True,
                }
                if tools:
                    resp_kwargs["tools"] = _tools_to_responses_format(tools)
                resp_kwargs.update(request_kwargs)
                resp_kwargs = _filter_provider_kwargs(
                    operation="stream_with_tools.responses",
                    create_method=self._client.responses.create,
                    request_kwargs=resp_kwargs,
                )
                _log_extra_body_passthrough(
                    operation="stream_with_tools.responses",
                    request_kwargs=resp_kwargs,
                )
                final_response: Any | None = None
                usage: dict[str, Any] = {}
                async for event in await self._client.responses.create(**resp_kwargs):
                    delta_event = _extract_responses_stream_delta(
                        event,
                        emit_reasoning=emit_reasoning_in_stream,
                    )
                    if delta_event:
                        yield cast(StreamEvent, delta_event)
                    event_usage = _extract_usage(event)
                    if event_usage is not None:
                        usage = event_usage.model_dump(mode="json")
                    if getattr(event, "type", None) == "response.completed":
                        final_response = getattr(event, "response", None)
                if final_response is not None:
                    parsed = _parse_responses_output(final_response)
                    if parsed.tool_calls:
                        tool_calls_event: StreamEvent = {
                            "type": "tool_calls",
                            "tool_calls": [
                                _tool_call_to_dict(tc) for tc in parsed.tool_calls
                            ],
                        }
                        yield tool_calls_event
                    if parsed.usage is not None:
                        usage = parsed.usage.model_dump(mode="json")
                done_event: StreamEvent = {"type": "done", "usage": usage}
                yield done_event
                return
            except Exception as exc:
                if not (
                    self._allow_response_fallback_to_chat
                    and _should_fallback_to_chat(exc)
                ):
                    raise
                status_code = _error_status_code(exc)
                logger.warning(
                    "Responses API stream unsupported, falling back to chat completions: "
                    "status=%s error=%s",
                    status_code,
                    type(exc).__name__,
                    extra=merge_log_context(
                        logging_context,
                        model=model,
                        api_mode=resolved_mode.value,
                    ),
                )

        raw_messages = _build_chat_messages(messages, system)
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": raw_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            chat_kwargs["tools"] = _tools_to_chat_format(tools)
            chat_kwargs["tool_choice"] = "auto"
        chat_kwargs.update(request_kwargs)
        chat_kwargs = _filter_provider_kwargs(
            operation="stream_with_tools.chat_completions",
            create_method=self._client.chat.completions.create,
            request_kwargs=chat_kwargs,
        )
        _log_extra_body_passthrough(
            operation="stream_with_tools.chat_completions",
            request_kwargs=chat_kwargs,
        )
        if tools and "parallel_tool_calls" not in chat_kwargs:
            chat_kwargs["parallel_tool_calls"] = True
        _ensure_stream_usage_options(chat_kwargs)
        tool_calls_buf = ToolCallBuffer()
        chat_usage: dict[str, Any] = {}
        async for chunk in await self._client.chat.completions.create(**chat_kwargs):
            if getattr(chunk, "usage", None):
                u = chunk.usage
                chat_usage = u.model_dump() if hasattr(u, "model_dump") else dict(u)
            if not getattr(chunk, "choices", []):
                continue
            delta = chunk.choices[0].delta
            delta_event = _extract_chat_stream_delta(
                delta,
                emit_reasoning=emit_reasoning_in_stream,
            )
            if delta_event and delta_event.get("text"):
                yield cast(StreamEvent, delta_event)
            for tc in getattr(delta, "tool_calls", []) or []:
                tool_calls_buf.add_delta(tc)
        buffered = tool_calls_buf.as_list()
        if buffered:
            chat_tool_calls_event: StreamEvent = {"type": "tool_calls", "tool_calls": buffered}
            yield chat_tool_calls_event
        chat_done_event: StreamEvent = {"type": "done", "usage": chat_usage}
        yield chat_done_event

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
        """Alias for simple text streaming without runtime tools."""
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

    # ------------------------------------------------------------------
    # Responses API path
    # ------------------------------------------------------------------

    async def _complete_responses(
        self,
        *,
        messages: list[Message],
        tools: list[Any] | None,
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
        stream_event_callback: StreamEventCallback | None,
        emit_reasoning_in_stream: bool,
        request_kwargs: dict[str, Any],
    ) -> Message:
        input_items = _messages_to_responses_input(messages, system)

        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _tools_to_responses_format(tools)
        kwargs.update(request_kwargs)
        kwargs = _filter_provider_kwargs(
            operation="complete.responses",
            create_method=self._client.responses.create,
            request_kwargs=kwargs,
        )
        _log_extra_body_passthrough(
            operation="complete.responses",
            request_kwargs=kwargs,
        )

        if not stream:
            response = await self._client.responses.create(**kwargs)
            return _parse_responses_output(response)

        kwargs["stream"] = True

        response_or_stream = await self._client.responses.create(**kwargs)
        if not _is_async_iterable(response_or_stream):
            return _parse_responses_output(response_or_stream)

        final_response: Any | None = None
        streamed_text: list[str] = []
        last_seen_usage: LLMUsage | None = None
        async for event in response_or_stream:
            delta_event = _extract_responses_stream_delta(
                event,
                emit_reasoning=emit_reasoning_in_stream,
            )
            if delta_event is not None:
                streamed_text.append(delta_event["text"])
                await _emit_stream_event(stream_event_callback, delta_event)
            event_usage = _extract_usage(event)
            if event_usage is not None:
                last_seen_usage = event_usage
            if getattr(event, "type", None) == "response.completed":
                final_response = getattr(event, "response", None)

        if final_response is not None:
            return _parse_responses_output(final_response)
        return Message(
            role="assistant",
            content="".join(streamed_text) or None,
            usage=last_seen_usage,
        )

    async def _complete_chat(
        self,
        *,
        messages: list[Message],
        tools: list[Any] | None,
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
        stream_event_callback: StreamEventCallback | None,
        emit_reasoning_in_stream: bool,
        request_kwargs: dict[str, Any],
        logging_context: dict[str, Any],
    ) -> Message:
        _ = logging_context
        raw_messages = _build_chat_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": raw_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _tools_to_chat_format(tools)
            kwargs["tool_choice"] = "auto"
        kwargs.update(request_kwargs)
        kwargs = _filter_provider_kwargs(
            operation="complete.chat_completions",
            create_method=self._client.chat.completions.create,
            request_kwargs=kwargs,
        )
        _log_extra_body_passthrough(
            operation="complete.chat_completions",
            request_kwargs=kwargs,
        )
        if tools and "parallel_tool_calls" not in kwargs:
            kwargs["parallel_tool_calls"] = True

        if not stream:
            response = await self._client.chat.completions.create(**kwargs)
            return _parse_chat_output(response)

        kwargs["stream"] = True
        _ensure_stream_usage_options(kwargs)

        response_or_stream = await self._client.chat.completions.create(**kwargs)
        if not _is_async_iterable(response_or_stream):
            return _parse_chat_output(response_or_stream)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_buf = ToolCallBuffer()
        usage: LLMUsage | None = None

        async for chunk in response_or_stream:
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            delta = _safe_attr(choice, "delta")
            if delta is not None:
                delta_event = _extract_chat_stream_delta(
                    delta,
                    emit_reasoning=emit_reasoning_in_stream,
                )
                if delta_event is not None:
                    await _emit_stream_event(stream_event_callback, delta_event)
                content_text = _coerce_chat_content_text(_safe_attr(delta, "content"))
                if content_text:
                    content_parts.append(content_text)
                reasoning_text = _coerce_reasoning_text(_safe_attr(delta, "reasoning"))
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                for tool_call_delta in _safe_attr(delta, "tool_calls") or []:
                    tool_calls_buf.add_delta(tool_call_delta)
            chunk_usage = _extract_usage(chunk)
            if chunk_usage is not None:
                usage = chunk_usage

        content = "".join(content_parts)
        if not content:
            content = "".join(reasoning_parts)
        tool_calls = tool_calls_buf.as_list()
        for tc in tool_calls:
            tc["function"]["arguments"] = _sanitize_arguments_json(tc["function"]["arguments"])
        return Message(
            role="assistant",
            content=content or None,
            tool_calls=[_tool_call_model(tc) for tc in tool_calls] or None,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _build_chat_messages(
    messages: list[Message], system: str | None
) -> list[dict[str, Any]]:
    """Convert Message objects to dicts for Chat Completions API."""
    result = serialize_messages_for_api(messages, system=system, target_api="chat")
    for item in result:
        tool_calls = item.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        sanitized_tool_calls: list[dict[str, Any]] = []
        for tool_call_payload in tool_calls:
            if not isinstance(tool_call_payload, dict):
                continue
            function_data = tool_call_payload.get("function", {})
            arguments = (
                function_data.get("arguments", "{}")
                if isinstance(function_data, dict)
                else "{}"
            )
            sanitized_tool_calls.append(
                {
                    **tool_call_payload,
                    "function": {
                        **(function_data if isinstance(function_data, dict) else {}),
                        "arguments": _sanitize_arguments_json(arguments),
                    },
                }
            )
        item["tool_calls"] = sanitized_tool_calls
    return result


def _messages_to_responses_input(
    messages: list[Message], system: str | None
) -> list[dict[str, Any]]:
    """Convert Message objects to Responses API input format."""
    items = serialize_messages_for_api(messages, system=system, target_api="responses")
    for item in items:
        if item.get("type") != "function_call":
            continue
        item["arguments"] = _sanitize_arguments_json(item.get("arguments", "{}"))
    return items


def _content_to_function_output(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for part in content:
        if part.type == "text" and part.text:
            fragments.append(part.text)
        elif part.type == "input_json":
            fragments.append(json.dumps(part.json_data or {}, ensure_ascii=True))
        elif part.type == "image_url" and part.image_url:
            fragments.append(part.image_url.get("url", ""))
    return "\n".join(fragment for fragment in fragments if fragment)


def _tools_to_chat_format(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to Chat Completions tool format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "to_openai_function"):
            openai_function = tool.to_openai_function()
            if isinstance(openai_function, dict):
                result.append(openai_function)
                continue
        if hasattr(tool, "name"):
            params: dict[str, Any] = {}
            if hasattr(tool, "parameters") and tool.parameters:
                params = (
                    tool.parameters.to_openai_schema()
                    if hasattr(tool.parameters, "to_openai_schema")
                    else (
                        tool.parameters.model_dump(by_alias=True)
                        if hasattr(tool.parameters, "model_dump")
                        else dict(tool.parameters)
                    )
                )
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": params,
                        "strict": getattr(tool, "strict", False),
                    },
                }
            )
            continue
        if isinstance(tool, dict):
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function_data = tool["function"]
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function_data.get("name", ""),
                            "description": function_data.get("description", ""),
                            "parameters": function_data.get("parameters", {}),
                            "strict": function_data.get("strict", False),
                        },
                    }
                )
            else:
                result.append(tool)
    return result


def _tools_to_responses_format(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to Responses API tool format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "name"):
            params: dict[str, Any] = {}
            if hasattr(tool, "parameters") and tool.parameters:
                params = (
                    tool.parameters.to_openai_schema()
                    if hasattr(tool.parameters, "to_openai_schema")
                    else (
                        tool.parameters.model_dump(exclude_none=True, by_alias=True)
                        if hasattr(tool.parameters, "model_dump")
                        else dict(tool.parameters)
                    )
                )
            result.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": params,
                    "strict": getattr(tool, "strict", False),
                }
            )
            continue
        if isinstance(tool, dict):
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function_data = tool["function"]
                result.append(
                    {
                        "type": "function",
                        "name": function_data.get("name", ""),
                        "description": function_data.get("description", ""),
                        "parameters": function_data.get("parameters", {}),
                        "strict": function_data.get("strict", False),
                    }
                )
            else:
                result.append(tool)
    return result


def _content_to_chat(content: str | list[ContentPart] | None) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    converted: list[dict[str, Any]] = []
    for part in content:
        if part.type == "text":
            converted.append({"type": "text", "text": part.text or ""})
        elif part.type == "image_url":
            converted.append({"type": "image_url", "image_url": part.image_url or {}})
        elif part.type == "input_json":
            converted.append(
                {"type": "text", "text": json.dumps(part.json_data or {}, ensure_ascii=True)}
            )
    return converted


def _content_to_responses(
    content: str | list[ContentPart] | None,
) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    converted: list[dict[str, Any]] = []
    for part in content:
        if part.type == "text":
            converted.append({"type": "input_text", "text": part.text or ""})
        elif part.type == "image_url":
            payload = {"type": "input_image"}
            if part.image_url:
                payload["image_url"] = part.image_url.get("url", "")
                detail = part.image_url.get("detail")
                if detail:
                    payload["detail"] = detail
            converted.append(payload)
        elif part.type == "input_json":
            converted.append(
                {
                    "type": "input_text",
                    "text": json.dumps(part.json_data or {}, ensure_ascii=True),
                }
            )
    return converted


def _validate_structured_output(raw_text: str, schema: Type[Any]) -> Any:
    if len(raw_text) > MAX_STRUCTURED_JSON_CHARS:
        if issubclass(schema, SubagentResult):
            return schema.parse_with_fallback(raw_text, agent_id="")
        raise ValueError("structured_output_too_large")
    if issubclass(schema, SubagentResult):
        return schema.parse_with_fallback(raw_text, agent_id="")
    candidates = structured_json_candidates(raw_text)
    failures: list[str] = []
    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            reason = f"{type(exc).__name__}:{exc.errors()[0]['type'] if exc.errors() else 'unknown'}"
            failures.append(reason)
            logger.warning(
                "Structured output candidate validation failed: schema=%s reason=%s",
                schema.__name__,
                reason,
            )
        except ValueError as exc:
            reason = f"{type(exc).__name__}:{exc}"
            failures.append(reason)
            logger.warning(
                "Structured output candidate validation failed: schema=%s reason=%s",
                schema.__name__,
                reason,
            )
    raise StructuredOutputValidationError(
        schema_name=schema.__name__,
        raw_text=raw_text,
        reason="no_json_candidate_validated:" + ";".join(failures[:5]),
    )


def _should_fallback_to_chat(exc: Exception) -> bool:
    return _fallback_reason_for_chat(exc) is not None


def _fallback_reason_for_chat(exc: Exception) -> str | None:
    status_code = _error_status_code(exc)
    if status_code == 404:
        return "responses_endpoint_missing"

    if status_code not in {400, 422}:
        return None

    text = _error_text(exc)
    compatibility_markers = {
        "responses_disabled": (
            "responses unsupported",
            "responses api unsupported",
            "does not support responses",
            "not support responses",
        ),
        "response_format_unsupported": (
            "response_format not supported",
            "unsupported field text.format",
            "unsupported field response_format",
            "unknown field text.format",
            "unknown parameter text.format",
            "unsupported parameter text.format",
            "unknown parameter response_format",
            "unsupported parameter response_format",
        ),
        "structured_output_unsupported": (
            "json_schema not supported",
            "unsupported field json_schema",
            "unknown parameter json_schema",
        ),
        "max_output_tokens_unsupported": (
            "unsupported field max_output_tokens",
            "unknown parameter max_output_tokens",
            "unsupported parameter max_output_tokens",
        ),
    }
    for reason, markers in compatibility_markers.items():
        if any(marker in text for marker in markers):
            return reason
    return None


def _extract_responses_text(response: Any) -> str:
    """Extract text content from a Responses API response."""
    fragments: list[str] = []
    if hasattr(response, "output"):
        for item in response.output:
            fragments.extend(_extract_responses_item_fragments(item, include_reasoning=True))
    if fragments:
        return "".join(fragments)
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    return ""


def _parse_responses_output(response: Any) -> Message:
    """Parse Responses API response into a Message."""
    tool_calls: list[ToolCall] = []
    content_text = ""

    if hasattr(response, "output"):
        for item in response.output:
            item_type = _safe_attr(item, "type") or ""
            if item_type == "function_call":
                tool_calls.append(
                    _tool_call_model(
                        {
                            "id": _safe_attr(item, "call_id") or _safe_attr(item, "id") or "",
                            "type": "function",
                            "function": {
                                "name": _safe_attr(item, "name") or "",
                                "arguments": _sanitize_arguments_json(
                                    _safe_attr(item, "arguments") or "{}"
                                ),
                            },
                        }
                    )
                )
                continue
            content_text += "".join(
                _extract_responses_item_fragments(item, include_reasoning=True)
            )

    if not content_text and hasattr(response, "output_text"):
        content_text = response.output_text or ""

    usage = _extract_usage(response)

    return Message(
        role="assistant",
        content=content_text if content_text else None,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
    )


def _extract_usage(response: Any) -> LLMUsage | None:
    """Extract token usage from either API response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    # Support vLLM/OpenAI-compatible usage provided as dict/object.
    def _usage_int(value: Any, field: str) -> int:
        if isinstance(value, dict):
            raw = value.get(field, 0)
        else:
            raw = getattr(value, field, 0)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    # --- cached_tokens: top-level or nested in prompt_tokens_details ---
    cached = _usage_int(usage, "cached_tokens")
    if not cached:
        prompt_details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        cached = _usage_int(prompt_details, "cached_tokens")

    # --- reasoning_tokens: Chat API completion_tokens_details or Responses API output_tokens_details ---
    completion_details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "completion_tokens_details", None)
    )
    reasoning_chat = _usage_int(completion_details, "reasoning_tokens")

    output_details = (
        usage.get("output_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "output_tokens_details", None)
    )
    reasoning_resp = _usage_int(output_details, "reasoning_tokens")

    reasoning = reasoning_chat or reasoning_resp

    return LLMUsage(
        input_tokens=_usage_int(usage, "input_tokens")
        or _usage_int(usage, "prompt_tokens"),
        output_tokens=_usage_int(usage, "output_tokens")
        or _usage_int(usage, "completion_tokens"),
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


async def _emit_stream_event(
    callback: StreamEventCallback | None,
    event: dict[str, Any] | None,
) -> None:
    if callback is None or event is None:
        return
    try:
        maybe_awaitable = callback(event)
        if maybe_awaitable is not None:
            await maybe_awaitable
    except Exception:
        logger.exception("Stream event callback failed")


def _coerce_response_text(part: Any) -> str:
    raw_text = _safe_attr(part, "text")
    if raw_text is None:
        return ""
    if isinstance(raw_text, str):
        return raw_text
    value = _safe_attr(raw_text, "value")
    if isinstance(value, str):
        return value
    return str(raw_text)


def _parse_chat_output(response: Any) -> Message:
    """Parse Chat Completions response into a Message."""
    choice = _require_chat_choice(response, operation="parse_chat_output")
    msg = choice.message
    tool_calls: list[ToolCall] | None = None

    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            function_data = _safe_attr(tc, "function")
            tool_calls.append(
                _tool_call_model(
                    {
                        "id": _safe_attr(tc, "id") or "",
                        "type": "function",
                        "function": {
                            "name": _safe_attr(function_data, "name") or "",
                            "arguments": _sanitize_arguments_json(
                                _safe_attr(function_data, "arguments") or "{}"
                            ),
                        },
                    }
                )
            )

    usage = _extract_usage(response)
    content = _coerce_chat_content_text(_safe_attr(msg, "content"))
    if not content:
        content = _coerce_reasoning_text(_safe_attr(msg, "reasoning"))

    return Message(
        role="assistant",
        content=content or None,
        tool_calls=tool_calls,
        usage=usage,
    )


def _require_chat_choice(response: Any, *, operation: str) -> Any:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return choices[0]
    preview = _response_preview(response)
    raise ValueError(
        f"chat_response_missing_choices:{operation}:response_preview={preview}"
    )


def _response_preview(response: Any, *, limit: int = 300) -> str:
    try:
        if hasattr(response, "model_dump"):
            raw = json.dumps(response.model_dump(mode="json"), ensure_ascii=True)
        elif isinstance(response, dict):
            raw = json.dumps(response, ensure_ascii=True)
        else:
            raw = str(response)
    except Exception:
        raw = str(response)
    return raw[:limit]


def _sanitize_arguments_json(arguments: Any) -> str:
    """Ensure tool call arguments is a valid JSON string.

    Local models (e.g. Qwen via vLLM) sometimes produce arguments with
    single quotes instead of double quotes, which causes vLLM to reject
    the request with a 400 when replaying the conversation history.
    """
    if not arguments:
        return "{}"
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, ensure_ascii=False)
    arguments_str = arguments if isinstance(arguments, str) else str(arguments)
    try:
        json.loads(arguments_str)
        logger.debug("tool call arguments already valid JSON")
        return arguments_str
    except (json.JSONDecodeError, TypeError, RecursionError, ValueError):
        logger.debug("tool call arguments JSON decode failed; trying literal_eval fallback")
    if len(arguments_str) > _MAX_LITERAL_EVAL_ARGUMENT_CHARS:
        logger.warning(
            "tool call arguments too large for literal_eval fallback, wrapping as raw: %.200s",
            arguments_str,
        )
        return json.dumps({"raw": arguments_str[:1000]})
    try:
        obj = ast.literal_eval(arguments_str)
        logger.debug("tool call arguments normalized via literal_eval fallback")
        return json.dumps(obj, ensure_ascii=False)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        logger.debug("tool call arguments literal_eval fallback failed; wrapping as raw")
    logger.warning(
        "tool call arguments not valid JSON, wrapping as string: %.200s",
        arguments_str,
    )
    return json.dumps({"raw": arguments_str})


def _safe_attr(value: Any, name: str) -> Any:
    """Safely get an attribute from an object or dict.

    ``MagicMock`` fabricates child mocks for any missing attribute, which can
    make recursive response walkers explode in memory during parsing. To avoid
    that, only return mock attributes that already exist on the instance or its
    class, while still supporting SDK objects that populate attributes at
    runtime via ``__getattr__``.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    instance_dict = getattr(value, "__dict__", None)
    if isinstance(instance_dict, dict) and name in instance_dict:
        return instance_dict[name]
    try:
        inspect.getattr_static(value, name)
    except AttributeError:
        if isinstance(value, Mock):
            return None
        try:
            attr = getattr(value, name)
        except AttributeError:
            return None
        if isinstance(attr, Mock) and getattr(attr, "_mock_parent", None) is value:
            return None
        return attr
    try:
        return getattr(value, name)
    except AttributeError:
        return None


def _tool_call_to_dict(tool_call: ToolCall | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool_call, ToolCall):
        return tool_call.to_openai_dict()
    return dict(tool_call)


def _tool_call_model(tool_call: ToolCall | dict[str, Any]) -> ToolCall:
    if isinstance(tool_call, ToolCall):
        return tool_call
    return ToolCall.model_validate(tool_call)


def _is_async_iterable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mock):
        return "__aiter__" in getattr(value, "__dict__", {})
    try:
        from openai.types.chat import ChatCompletion
        from openai.types.responses import Response as ResponsesResponse

        if isinstance(value, (ChatCompletion, ResponsesResponse)):
            return False
    except ImportError:
        pass
    if isinstance(value, AsyncIterable):
        return True
    return callable(getattr(type(value), "__aiter__", None))


def _coerce_chat_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    fragments.append(part)
                continue
            text = _safe_attr(part, "text")
            if isinstance(text, str) and text:
                fragments.append(text)
        return "".join(fragments)
    return ""


def _coerce_reasoning_text(reasoning: Any) -> str:
    fragments = _collect_reasoning_fragments(reasoning)
    return "".join(fragment for fragment in fragments if fragment)


def _collect_reasoning_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        out: list[str] = []
        for item in value:
            out.extend(_collect_reasoning_fragments(item))
        return out

    fragments: list[str] = []
    text = _safe_attr(value, "text")
    if isinstance(text, str) and text:
        fragments.append(text)
    else:
        text_value = _safe_attr(text, "value")
        if isinstance(text_value, str) and text_value:
            fragments.append(text_value)

    for attr_name in ("content", "summary", "reasoning"):
        nested = _safe_attr(value, attr_name)
        if nested is not None and nested is not value:
            fragments.extend(_collect_reasoning_fragments(nested))
    return fragments


def _extract_responses_item_fragments(item: Any, *, include_reasoning: bool) -> list[str]:
    item_type = _safe_attr(item, "type") or ""
    if item_type == "function_call":
        return []
    if item_type == "reasoning" and not include_reasoning:
        return []

    fragments: list[str] = []
    content = _safe_attr(item, "content")
    if content is not None:
        fragments.extend(_extract_responses_content_fragments(content, include_reasoning=include_reasoning))

    if include_reasoning:
        reasoning = _safe_attr(item, "reasoning")
        if reasoning is not None:
            fragments.extend(_collect_reasoning_fragments(reasoning))
        summary = _safe_attr(item, "summary")
        if summary is not None:
            fragments.extend(_collect_reasoning_fragments(summary))

    if not fragments:
        text = _coerce_response_text(item)
        if text:
            fragments.append(text)
    return fragments


def _extract_responses_content_fragments(content: Any, *, include_reasoning: bool) -> list[str]:
    if content is None:
        return []
    if isinstance(content, list | tuple):
        out: list[str] = []
        for part in content:
            out.extend(_extract_responses_content_fragments(part, include_reasoning=include_reasoning))
        return out
    part_type = _safe_attr(content, "type") or ""
    if part_type == "reasoning" and not include_reasoning:
        return []
    if part_type == "reasoning_text" and include_reasoning:
        text = _coerce_response_text(content)
        return [text] if text else []
    if part_type == "summary_text" and include_reasoning:
        text = _coerce_response_text(content)
        return [text] if text else []

    fragments: list[str] = []
    if include_reasoning:
        reasoning = _safe_attr(content, "reasoning")
        if reasoning is not None:
            fragments.extend(_collect_reasoning_fragments(reasoning))
        summary = _safe_attr(content, "summary")
        if summary is not None:
            fragments.extend(_collect_reasoning_fragments(summary))

    text = _coerce_response_text(content)
    if text:
        fragments.append(text)

    nested_content = _safe_attr(content, "content")
    if nested_content is not None and nested_content is not content:
        fragments.extend(
            _extract_responses_content_fragments(
                nested_content,
                include_reasoning=include_reasoning,
            )
        )
    return fragments


def _structured_responses_text_formats(schema: Type[Any]) -> list[dict[str, Any]]:
    schema_json = schema.model_json_schema()
    schema_name = getattr(schema, "__name__", "StructuredOutput")
    return [
        {
            "type": "json_schema",
            "name": schema_name,
            "schema": schema_json,
        },
        {
            "type": "json_schema",
            "schema": schema_json,
        },
        {
            "type": "json_object",
        },
    ]


def _should_retry_structured_responses_format(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(
        marker in text
        for marker in (
            "response_format",
            "text.format",
            "json_schema",
            "validation error",
            "unsupported field",
            "unknown field",
            "unsupported parameter",
            "unknown parameter",
            "schema",
        )
    )


def _error_status_code(exc: Exception) -> int | None:
    for candidate in (
        _safe_attr(exc, "status_code"),
        _safe_attr(_safe_attr(exc, "response"), "status_code"),
        _safe_attr(_safe_attr(exc, "response"), "status"),
        _safe_attr(_safe_attr(exc, "body"), "status_code"),
    ):
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def _error_text(exc: Exception) -> str:
    parts = [str(exc).lower()]
    for candidate in (
        _safe_attr(exc, "message"),
        _safe_attr(exc, "body"),
        _safe_attr(_safe_attr(exc, "response"), "text"),
    ):
        if isinstance(candidate, str):
            parts.append(candidate.lower())
        elif isinstance(candidate, dict):
            parts.append(json.dumps(candidate, ensure_ascii=True).lower())
    return " ".join(part for part in parts if part)


def _extract_responses_stream_delta_text(event: Any, *, emit_reasoning: bool) -> str:
    event_type = str(_safe_attr(event, "type") or "").lower()
    delta = _safe_attr(event, "delta")
    if delta is None:
        return ""
    if "reasoning" in event_type and not emit_reasoning:
        return ""
    if isinstance(delta, str):
        return delta
    if "reasoning" in event_type:
        return _coerce_reasoning_text(delta) if emit_reasoning else ""
    text = _coerce_response_text(delta)
    if text:
        return text
    return _coerce_reasoning_text(delta) if emit_reasoning else ""


def _extract_responses_stream_delta(
    event: Any, *, emit_reasoning: bool
) -> dict[str, Any] | None:
    event_type = str(_safe_attr(event, "type") or "").lower()
    if not event_type:
        text = _extract_responses_stream_delta_text(event, emit_reasoning=False)
        if text:
            return {
                "type": "delta",
                "kind": "text",
                "text": text,
                "provider_event_type": "response.output_text.delta",
            }
        if emit_reasoning:
            text = _extract_responses_stream_delta_text(event, emit_reasoning=True)
            if text:
                return {
                    "type": "delta",
                    "kind": "reasoning",
                    "text": text,
                    "provider_event_type": "response.reasoning.delta",
                }
        return None
    if "delta" not in event_type:
        return None
    if "reasoning" in event_type and not emit_reasoning:
        return None
    if "reasoning" in event_type:
        text = _extract_responses_stream_delta_text(event, emit_reasoning=True)
        if not text:
            return None
        return {
            "type": "delta",
            "kind": "reasoning",
            "text": text,
            "provider_event_type": event_type,
        }
    text = _extract_responses_stream_delta_text(event, emit_reasoning=False)
    if not text and emit_reasoning:
        text = _extract_responses_stream_delta_text(event, emit_reasoning=True)
        if text:
            return {
                "type": "delta",
                "kind": "reasoning",
                "text": text,
                "provider_event_type": event_type or "responses.delta",
            }
    if not text:
        return None
    return {
        "type": "delta",
        "kind": "text",
        "text": text,
        "provider_event_type": event_type or "responses.delta",
    }


def _extract_chat_stream_delta(
    delta: Any, *, emit_reasoning: bool
) -> dict[str, Any] | None:
    content_text = _coerce_chat_content_text(_safe_attr(delta, "content"))
    if content_text:
        return {
            "type": "delta",
            "kind": "text",
            "text": content_text,
            "provider_event_type": "chat.completions.delta",
        }
    if emit_reasoning:
        reasoning_text = _coerce_reasoning_text(_safe_attr(delta, "reasoning"))
        if reasoning_text:
            return {
                "type": "delta",
                "kind": "reasoning",
                "text": reasoning_text,
                "provider_event_type": "chat.completions.reasoning",
            }
    return None
