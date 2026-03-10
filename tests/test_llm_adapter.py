"""Tests for OpenAILLMClient adapter (no real network calls).

All OpenAI SDK calls are mocked via AsyncMock patching.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from protocore import OpenAILLMClient, Message
from protocore.constants import MAX_STRUCTURED_JSON_CHARS
from protocore.context import _serialize_messages_for_estimation
from protocore.types import (
    ApiMode,
    ContentPart,
    StreamDeltaEvent,
    SubagentResult,
    SubagentStatus,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
)
from protocore.integrations.llm.openai_client import (
    ToolCallBuffer,
    _build_chat_messages,
    _coerce_response_text,
    _content_to_chat,
    _emit_stream_event,
    _ensure_stream_usage_options,
    _content_to_function_output,
    _content_to_responses,
    _extract_responses_content_fragments,
    _extract_chat_stream_delta,
    _extract_responses_item_fragments,
    _extract_responses_stream_delta,
    _extract_usage,
    _error_status_code,
    _error_text,
    _is_async_iterable,
    _response_preview,
    _pop_logging_context,
    _validate_structured_output,
    _messages_to_responses_input,
    _parse_chat_output,
    _parse_responses_output,
    _safe_attr,
    _sanitize_arguments_json,
    _should_retry_structured_responses_format,
    _should_fallback_to_chat,
    _tools_to_chat_format,
    _tools_to_responses_format,
    _extract_responses_text,
)


# ---------------------------------------------------------------------------
# Message conversion tests (pure functions, no mocks needed)
# ---------------------------------------------------------------------------


class TestBuildChatMessages:
    def test_with_system(self) -> None:
        msgs = [Message(role="user", content="hi")]
        result = _build_chat_messages(msgs, system="You are helpful")
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_without_system(self) -> None:
        msgs = [Message(role="user", content="hi")]
        result = _build_chat_messages(msgs, system=None)
        assert len(result) == 1

    def test_tool_role_message(self) -> None:
        msgs = [Message(role="tool", content="result", tool_call_id="tc1", name="echo")]
        result = _build_chat_messages(msgs, system=None)
        assert result[0]["tool_call_id"] == "tc1"
        assert result[0]["name"] == "echo"

    def test_assistant_with_tool_calls(self) -> None:
        tc = [
            ToolCall.model_validate(
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            )
        ]
        msgs = [Message(role="assistant", content=None, tool_calls=tc)]
        result = _build_chat_messages(msgs, system=None)
        assert result[0]["tool_calls"] == [tool_call.to_openai_dict() for tool_call in tc]

    def test_system_parameter_skips_existing_system_messages(self) -> None:
        msgs = [
            Message(role="system", content="explicit system"),
            Message(role="system", content="compaction summary"),
            Message(role="user", content="hi"),
        ]
        result = _build_chat_messages(msgs, system="explicit system")
        # Single system message at the beginning (merged)
        assert result == [
            {"role": "system", "content": "explicit system\n\ncompaction summary"},
            {"role": "user", "content": "hi"},
        ]


class TestToolCallBuffer:
    def test_repeated_id_delta_does_not_duplicate_id(self) -> None:
        delta = MagicMock()
        delta.index = 0
        delta.id = "call_abc"
        delta.function = None

        buffer = ToolCallBuffer()
        buffer.add_delta(delta)
        buffer.add_delta(delta)

        assert buffer.as_list()[0]["id"] == "call_abc"

    def test_missing_index_reuses_stable_fallback_slot(self) -> None:
        class FunctionDelta:
            def __init__(self, *, name: str | None, arguments: str) -> None:
                self.name = name
                self.arguments = arguments

        class ToolDelta:
            def __init__(self, *, call_id: str, name: str | None, arguments: str) -> None:
                self.id = call_id
                self.function = FunctionDelta(name=name, arguments=arguments)

        first = ToolDelta(call_id="call_xyz", name="echo", arguments="{")
        second = ToolDelta(call_id="call_xyz", name=None, arguments='"value":"ok"}')

        buffer = ToolCallBuffer()
        buffer.add_delta(first)
        buffer.add_delta(second)

        assert buffer.as_list() == [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"value":"ok"}'},
            }
        ]

    def test_missing_all_identifiers_reuses_last_implicit_slot(self) -> None:
        class ToolDelta:
            index = None
            id = None
            function = None

        buffer = ToolCallBuffer()
        buffer.add_delta(ToolDelta())
        buffer.add_delta(ToolDelta())

        assert len(buffer.as_list()) == 1

    def test_function_name_without_index_reuses_name_slot(self) -> None:
        class FunctionDelta:
            def __init__(self, *, name: str | None, arguments: str) -> None:
                self.name = name
                self.arguments = arguments

        class ToolDelta:
            def __init__(self, *, name: str | None, arguments: str) -> None:
                self.index = None
                self.id = None
                self.function = FunctionDelta(name=name, arguments=arguments)

        first = ToolDelta(name="echo", arguments="{")
        second = ToolDelta(name="echo", arguments='"a":1}')

        buffer = ToolCallBuffer()
        buffer.add_delta(first)
        buffer.add_delta(second)

        assert buffer.as_list()[0]["function"]["arguments"] == '{"a":1}'

    def test_as_list_handles_sparse_indexes(self) -> None:
        first = SimpleNamespace(index=1, id="call-1", function=None)
        second = SimpleNamespace(index=3, id="call-3", function=None)

        buffer = ToolCallBuffer()
        buffer.add_delta(first)
        buffer.add_delta(second)

        assert [item["id"] for item in buffer.as_list()] == ["call-1", "call-3"]


class TestMessagesToResponsesInput:
    def test_basic_conversion(self) -> None:
        msgs = [Message(role="user", content="hello")]
        items = _messages_to_responses_input(msgs, system="sys")
        assert items[0] == {
            "role": "system",
            "content": [{"type": "input_text", "text": "sys"}],
        }
        assert items[1]["content"] == [{"type": "input_text", "text": "hello"}]

    def test_multimodal_conversion(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="look"),
                    ContentPart(type="image_url", image_url={"url": "https://example.com/img.png"}),
                ],
            )
        ]
        items = _messages_to_responses_input(msgs, system=None)
        assert items[0]["content"][0] == {"type": "input_text", "text": "look"}
        assert items[0]["content"][1]["type"] == "input_image"

    def test_system_messages_are_deduplicated_when_system_prompt_is_provided(self) -> None:
        msgs = [
            Message(role="system", content="top-level system"),
            Message(role="system", content="summary kept"),
            Message(role="user", content="hello"),
        ]
        items = _messages_to_responses_input(msgs, system="top-level system")
        # Single system message at the beginning (merged)
        assert items == [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "top-level system\n\nsummary kept"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]

    def test_content_part_input_json_and_image_detail_are_converted(self) -> None:
        parts = [
            ContentPart(type="image_url", image_url={"url": "https://example.com/image.png", "detail": "high"}),
            ContentPart(type="input_json", json_data={"k": "v"}),
        ]
        converted = _content_to_responses(parts)
        assert converted[0] == {
            "type": "input_image",
            "image_url": "https://example.com/image.png",
            "detail": "high",
        }
        assert converted[1] == {"type": "input_text", "text": '{"k": "v"}'}

    def test_preserves_assistant_tool_calls_and_tool_outputs(self) -> None:
        msgs = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call-1",
                            "function": {"name": "search", "arguments": '{"query":"test"}'},
                        }
                    )
                ],
            ),
            Message(role="tool", content="result body", tool_call_id="call-1", name="search"),
        ]

        items = _messages_to_responses_input(msgs, system=None)

        assert items[0] == {
            "type": "function_call",
            "call_id": "call-1",
            "name": "search",
            "arguments": '{"query":"test"}',
        }
        assert items[1] == {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "result body",
        }


class TestMessageSerializationConsistency:
    def test_chat_serialization_matches_estimation_payload(self) -> None:
        messages = [
            Message(
                role="assistant",
                content="thinking",
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call-1",
                            "function": {"name": "search", "arguments": '{"q":"docs"}'},
                        }
                    )
                ],
            ),
            Message(role="tool", content="result body", tool_call_id="call-1", name="search"),
        ]

        assert _serialize_messages_for_estimation(messages, api_mode=ApiMode.CHAT_COMPLETIONS) == (
            _build_chat_messages(messages, system=None)
        )

    def test_responses_serialization_matches_estimation_payload(self) -> None:
        messages = [
            Message(
                role="assistant",
                content=[
                    ContentPart(type="text", text="look"),
                    ContentPart(type="input_json", json_data={"k": "v"}),
                ],
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "call-1",
                            "function": {"name": "search", "arguments": '{"q":"docs"}'},
                        }
                    )
                ],
            ),
            Message(role="tool", content="result body", tool_call_id="call-1", name="search"),
        ]

        assert _serialize_messages_for_estimation(messages, api_mode=ApiMode.RESPONSES) == (
            _messages_to_responses_input(messages, system=None)
        )


class TestToolsToFormat:
    def test_tool_definition_objects(self) -> None:
        tools = [
            ToolDefinition(
                name="echo",
                description="Echoes input",
                parameters=ToolParameterSchema(
                    type="object",
                    properties={"text": {"type": "string"}},
                ),
            )
        ]
        result = _tools_to_chat_format(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "echo"

    def test_tool_schema_uses_additional_properties_alias(self) -> None:
        tool = ToolDefinition(
            name="echo",
            description="Echoes input",
            parameters=ToolParameterSchema(
                type="object",
                properties={"text": {"type": "string"}},
                additionalProperties=False,
            ),
        )

        chat_result = _tools_to_chat_format([tool])
        responses_result = _tools_to_responses_format([tool])
        openai_function = tool.to_openai_function()

        assert chat_result[0]["function"]["parameters"]["additionalProperties"] is False
        assert "additional_properties" not in chat_result[0]["function"]["parameters"]
        assert responses_result[0]["parameters"]["additionalProperties"] is False
        assert "additional_properties" not in responses_result[0]["parameters"]
        assert openai_function["function"]["parameters"]["additionalProperties"] is False
        assert "additional_properties" not in openai_function["function"]["parameters"]

    def test_dict_passthrough(self) -> None:
        tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "x"}}]
        result = _tools_to_chat_format(tools)
        assert result[0]["function"]["name"] == "x"

    def test_custom_tool_object_uses_generic_parameters_fallback(self) -> None:
        class DictParams:
            def __init__(self) -> None:
                self._payload = {"type": "object", "properties": {"q": {"type": "string"}}}

            def __iter__(self) -> Any:
                return iter(self._payload.items())

        tool = SimpleNamespace(
            name="search",
            description="Search docs",
            strict=True,
            parameters=DictParams(),
        )

        result = _tools_to_chat_format([tool])

        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_non_function_tool_dict_is_preserved(self) -> None:
        tools: list[dict[str, Any]] = [{"type": "web_search", "provider": "custom"}]

        result = _tools_to_chat_format(tools)

        assert result == tools

    def test_non_function_tool_dict_is_preserved_for_responses_format(self) -> None:
        tools: list[dict[str, Any]] = [{"type": "web_search", "provider": "custom"}]

        result = _tools_to_responses_format(tools)

        assert result == tools


class TestStreamHelpers:
    @pytest.mark.asyncio
    async def test_emit_stream_event_swallow_callback_failures(self) -> None:
        seen: list[dict[str, Any]] = []

        async def callback(event: dict[str, Any]) -> None:
            seen.append(event)
            raise RuntimeError("callback failed")

        await _emit_stream_event(callback, {"text": "delta"})

        assert seen == [{"text": "delta"}]

    @pytest.mark.asyncio
    async def test_emit_stream_event_noop_when_callback_or_event_missing(self) -> None:
        await _emit_stream_event(None, {"text": "delta"})
        await _emit_stream_event(lambda event: None, None)

    def test_sanitize_arguments_json_accepts_non_string_input(self) -> None:
        assert _sanitize_arguments_json({"x": 1}) == '{"x": 1}'

    def test_sanitize_arguments_json_handles_empty_and_unparseable_strings(self) -> None:
        assert _sanitize_arguments_json(None) == "{}"
        wrapped = _sanitize_arguments_json("{not-valid-json")
        assert json.loads(wrapped)["raw"] == "{not-valid-json"

    def test_sanitize_arguments_json_limits_literal_eval_input_size(self) -> None:
        payload = "[" * 10_001
        sanitized = _sanitize_arguments_json(payload)
        assert '"raw"' in sanitized
        assert len(json.loads(sanitized)["raw"]) == 1000

    def test_responses_format_uses_top_level_function_shape(self) -> None:
        tools = [
            ToolDefinition(
                name="echo",
                description="Echoes input",
                parameters=ToolParameterSchema(
                    type="object",
                    properties={"text": {"type": "string"}},
                ),
                strict=True,
            )
        ]

        result = _tools_to_responses_format(tools)

        assert result == [
            {
                "type": "function",
                "name": "echo",
                "description": "Echoes input",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": [],
                },
                "strict": True,
            }
        ]

    def test_responses_format_flattens_nested_function_dict(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search docs",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    "strict": True,
                },
            }
        ]

        result = _tools_to_responses_format(tools)

        assert result == [
            {
                "type": "function",
                "name": "search",
                "description": "Search docs",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                "strict": True,
            }
        ]

    def test_extract_responses_stream_delta_ignores_missing_mock_delta_attributes(self) -> None:
        event = MagicMock()
        event.type = "response.completed"

        assert _extract_responses_stream_delta(event, emit_reasoning=True) is None

    def test_extract_responses_stream_delta_without_type_uses_text_fallback(self) -> None:
        event = SimpleNamespace(delta=SimpleNamespace(text="plain-text"))

        delta = _extract_responses_stream_delta(event, emit_reasoning=False)

        assert delta is not None
        assert delta["kind"] == "text"
        assert delta["provider_event_type"] == "response.output_text.delta"

    def test_extract_responses_stream_delta_reasoning_is_skipped_when_disabled(self) -> None:
        event = SimpleNamespace(
            type="response.reasoning.delta",
            delta=SimpleNamespace(text="hidden"),
        )

        assert _extract_responses_stream_delta(event, emit_reasoning=False) is None

    def test_extract_responses_stream_delta_text_event_can_fallback_to_reasoning_when_enabled(
        self,
    ) -> None:
        event = SimpleNamespace(
            type="response.output_text.delta",
            delta=SimpleNamespace(reasoning=SimpleNamespace(text="trace")),
        )

        delta = _extract_responses_stream_delta(event, emit_reasoning=True)

        assert delta is not None
        assert delta["kind"] == "reasoning"
        assert delta["text"] == "trace"

    def test_extract_chat_stream_delta_emits_reasoning_when_content_missing(self) -> None:
        delta = SimpleNamespace(content=None, reasoning=SimpleNamespace(text="chain"))

        event = _extract_chat_stream_delta(delta, emit_reasoning=True)

        assert event is not None
        assert event["kind"] == "reasoning"
        assert event["text"] == "chain"

    def test_is_async_iterable_accepts_stream_object_even_with_output_text_attr(self) -> None:
        class FakeStream:
            output_text = ""

            def __aiter__(self) -> Any:
                async def _iterate() -> Any:
                    if False:
                        yield None

                return _iterate()

        assert _is_async_iterable(FakeStream()) is True


class TestContentHelpers:
    def test_content_to_function_output_handles_none_and_plain_text(self) -> None:
        assert _content_to_function_output(None) == ""
        assert _content_to_function_output("raw") == "raw"

    def test_content_to_chat_handles_none_plain_text_and_text_parts(self) -> None:
        assert _content_to_chat(None) == ""
        assert _content_to_chat("plain") == "plain"
        assert _content_to_chat([ContentPart(type="text", text="hello")]) == [
            {"type": "text", "text": "hello"}
        ]

    def test_content_to_responses_handles_none_string_and_text_parts(self) -> None:
        assert _content_to_responses(None) == ""
        assert _content_to_responses("plain") == [{"type": "input_text", "text": "plain"}]
        assert _content_to_responses([ContentPart(type="text", text="hello")]) == [
            {"type": "input_text", "text": "hello"}
        ]

    def test_content_to_function_output_supports_text_json_and_image_parts(self) -> None:
        content = [
            ContentPart(type="text", text="alpha"),
            ContentPart(type="input_json", json_data={"k": "v"}),
            ContentPart(type="image_url", image_url={"url": "https://example.com/a.png"}),
        ]

        output = _content_to_function_output(content)

        assert output == 'alpha\n{"k": "v"}\nhttps://example.com/a.png'

    def test_content_to_chat_supports_image_and_input_json_parts(self) -> None:
        content = [
            ContentPart(type="image_url", image_url={"url": "https://example.com/a.png"}),
            ContentPart(type="input_json", json_data={"a": 1}),
        ]

        output = _content_to_chat(content)

        assert output == [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "text", "text": '{"a": 1}'},
        ]

    def test_content_to_responses_supports_image_part(self) -> None:
        content = [ContentPart(type="image_url", image_url={"url": "https://example.com/a.png"})]

        output = _content_to_responses(content)

        assert output == [{"type": "input_image", "image_url": "https://example.com/a.png"}]

    def test_extract_responses_content_fragments_does_not_recurse_into_missing_mock_attrs(self) -> None:
        content = MagicMock()

        assert _extract_responses_content_fragments(content, include_reasoning=True) == []

    def test_extract_responses_item_fragments_skips_function_call_items(self) -> None:
        item = SimpleNamespace(type="function_call", content="ignored")

        assert _extract_responses_item_fragments(item, include_reasoning=True) == []

    def test_extract_responses_item_fragments_skips_reasoning_when_not_requested(self) -> None:
        item = SimpleNamespace(type="reasoning", content="ignored")

        assert _extract_responses_item_fragments(item, include_reasoning=False) == []

    def test_extract_responses_content_fragments_extracts_nested_reasoning_and_summary(self) -> None:
        content = SimpleNamespace(
            type="message",
            text=None,
            reasoning=SimpleNamespace(text=SimpleNamespace(value="R")),
            summary=[SimpleNamespace(text="S")],
            content=[SimpleNamespace(text="T")],
        )

        fragments = _extract_responses_content_fragments(content, include_reasoning=True)

        assert "R" in fragments
        assert "S" in fragments
        assert "T" in fragments


# ---------------------------------------------------------------------------
# Parse response tests
# ---------------------------------------------------------------------------


class TestParseResponsesOutput:
    def test_text_response(self) -> None:
        response = MagicMock()
        response.output = []
        response.output_text = "Hello world"
        msg = _parse_responses_output(response)
        assert msg.role == "assistant"
        assert msg.content == "Hello world"
        assert msg.tool_calls is None

    def test_function_call_response(self) -> None:
        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.call_id = "call_123"
        fc_item.name = "echo"
        fc_item.arguments = '{"text": "hi"}'

        response = MagicMock()
        response.output = [fc_item]
        response.output_text = ""
        msg = _parse_responses_output(response)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["function"]["name"] == "echo"

    def test_message_with_content(self) -> None:
        part = MagicMock()
        part.text = "the answer"

        msg_item = MagicMock()
        msg_item.type = "message"
        msg_item.content = [part]

        response = MagicMock()
        response.output = [msg_item]
        response.output_text = ""
        msg = _parse_responses_output(response)
        assert msg.content == "the answer"

    def test_reasoning_item_with_reasoning_text_is_extracted(self) -> None:
        reasoning_part = MagicMock()
        reasoning_part.type = "reasoning_text"
        reasoning_part.text = "intermediate reasoning"

        reasoning_item = MagicMock()
        reasoning_item.type = "reasoning"
        reasoning_item.content = [reasoning_part]
        reasoning_item.reasoning = None
        reasoning_item.summary = None

        response = MagicMock()
        response.output = [reasoning_item]
        response.output_text = ""

        msg = _parse_responses_output(response)
        assert msg.content == "intermediate reasoning"


class TestParseChatOutput:
    def test_text_response(self) -> None:
        msg_obj = MagicMock()
        msg_obj.content = "answer"
        msg_obj.tool_calls = None

        choice = MagicMock()
        choice.message = msg_obj

        response = MagicMock()
        response.choices = [choice]

        msg = _parse_chat_output(response)
        assert msg.content == "answer"
        assert msg.tool_calls is None

    def test_tool_call_response(self) -> None:
        func = MagicMock()
        func.name = "echo"
        func.arguments = '{"x": 1}'

        tc = MagicMock()
        tc.id = "tc_1"
        tc.function = func

        msg_obj = MagicMock()
        msg_obj.content = None
        msg_obj.tool_calls = [tc]

        choice = MagicMock()
        choice.message = msg_obj

        response = MagicMock()
        response.choices = [choice]

        msg = _parse_chat_output(response)
        assert msg.tool_calls is not None
        assert msg.tool_calls[0]["function"]["name"] == "echo"

    def test_reasoning_only_response_falls_back_to_reasoning_text(self) -> None:
        reasoning_part = MagicMock()
        reasoning_part.text = "thoughts"

        msg_obj = MagicMock()
        msg_obj.content = None
        msg_obj.reasoning = [reasoning_part]
        msg_obj.tool_calls = None

        choice = MagicMock()
        choice.message = msg_obj

        response = MagicMock()
        response.choices = [choice]

        msg = _parse_chat_output(response)
        assert msg.content == "thoughts"


class TestExtractResponsesText:
    def test_with_output_text(self) -> None:
        response = MagicMock()
        response.output_text = "result"
        assert _extract_responses_text(response) == "result"

    def test_with_nested_output(self) -> None:
        part = MagicMock()
        part.text = "nested"

        item = MagicMock()
        item.content = [part]

        response = MagicMock()
        response.output_text = None
        response.output = [item]
        assert _extract_responses_text(response) == "nested"

    def test_concatenates_multiple_nested_text_parts_in_order(self) -> None:
        part1 = MagicMock()
        part1.text = '{"status":"success",'
        part2 = MagicMock()
        part2.text = '"summary":"ok"}'

        item = MagicMock()
        item.content = [part1, part2]

        response = MagicMock()
        response.output_text = None
        response.output = [item]

        assert _extract_responses_text(response) == '{"status":"success","summary":"ok"}'

    def test_extracts_reasoning_fragments_from_responses_output(self) -> None:
        reasoning_part = MagicMock()
        reasoning_part.type = "reasoning_text"
        reasoning_part.text = "reasoning"

        item = MagicMock()
        item.type = "reasoning"
        item.content = [reasoning_part]
        item.reasoning = None
        item.summary = None

        response = MagicMock()
        response.output_text = None
        response.output = [item]

        assert _extract_responses_text(response) == "reasoning"


# ---------------------------------------------------------------------------
# Integration-level tests (mock AsyncOpenAI client)
# ---------------------------------------------------------------------------


class TestOpenAILLMClientComplete:
    @pytest.mark.asyncio
    async def test_responses_mode_complete_streams_by_default_and_emits_deltas(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        reasoning_event = MagicMock()
        reasoning_event.type = "response.reasoning.delta"
        reasoning_event.delta = "thinking"

        text_event = MagicMock()
        text_event.type = "response.output_text.delta"
        text_event.delta = "Hello"

        completed_event = MagicMock()
        completed_event.type = "response.completed"
        completed_event.response = MagicMock()
        completed_event.response.output = []
        completed_event.response.output_text = "Hello"

        async def fake_stream() -> Any:
            yield reasoning_event
            yield text_event
            yield completed_event

        seen_events: list[dict[str, Any]] = []

        async def on_stream_event(event: dict[str, Any]) -> None:
            seen_events.append(event)

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                emit_reasoning_in_stream=True,
                stream_event_callback=on_stream_event,
            )

        assert result.content == "Hello"
        assert seen_events == [
            {
                "type": "delta",
                "kind": "reasoning",
                "text": "thinking",
                "provider_event_type": "response.reasoning.delta",
            },
            {
                "type": "delta",
                "kind": "text",
                "text": "Hello",
                "provider_event_type": "response.output_text.delta",
            },
        ]
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_complete_respects_stream_false_and_uses_non_stream_path(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = "Hello"

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                stream=False,
            )

        assert result.content == "Hello"
        assert mock_create.await_args is not None
        assert "stream" not in mock_create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_responses_mode_complete(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = "Hello"

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            )
            assert result.content == "Hello"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_mode_complete(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        msg_obj = MagicMock()
        msg_obj.content = "World"
        msg_obj.tool_calls = None

        choice = MagicMock()
        choice.message = msg_obj

        mock_response = MagicMock()
        mock_response.choices = [choice]

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            )
            assert result.content == "World"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_mode_complete_respects_stream_false(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        msg_obj = MagicMock()
        msg_obj.content = "non-stream"
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        response = MagicMock()
        response.choices = [choice]

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                stream=False,
            )

        assert result.content == "non-stream"
        assert mock_create.await_args is not None
        assert "stream" not in mock_create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_chat_mode_stream_complete_falls_back_to_reasoning_when_content_missing(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        delta = SimpleNamespace(
            content=None,
            reasoning=SimpleNamespace(text="reasoning-only"),
            tool_calls=[],
        )
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=delta)],
            usage=None,
        )

        async def fake_stream() -> Any:
            yield chunk

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            )

        assert result.content == "reasoning-only"
        assert mock_create.await_count == 1

    @pytest.mark.asyncio
    async def test_responses_mode_with_tools(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        fc_item = MagicMock()
        fc_item.type = "function_call"
        fc_item.call_id = "call_1"
        fc_item.name = "echo"
        fc_item.arguments = '{"text": "hi"}'

        mock_response = MagicMock()
        mock_response.output = [fc_item]
        mock_response.output_text = ""

        tools = [ToolDefinition(name="echo", description="echoes")]

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete(
                messages=[Message(role="user", content="call echo")],
                tools=tools,
                model="gpt-4o",
            )
            assert result.tool_calls is not None
            assert result.tool_calls[0]["function"]["name"] == "echo"
            assert mock_create.await_args is not None
            sent_tools = mock_create.await_args.kwargs["tools"]
            assert sent_tools[0]["name"] == "echo"
            assert "function" not in sent_tools[0]

    @pytest.mark.asyncio
    async def test_default_model_used(self) -> None:
        client = OpenAILLMClient(
            api_key="test",
            api_mode=ApiMode.RESPONSES,
            default_model="qwen-72b",
        )

        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = "ok"

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await client.complete(
                messages=[Message(role="user", content="hi")],
            )
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == "qwen-72b"

    @pytest.mark.asyncio
    async def test_base_url_set(self) -> None:
        client = OpenAILLMClient(
            api_key="test",
            base_url="http://localhost:8000/v1",
        )
        assert client._client.base_url is not None

    @pytest.mark.asyncio
    async def test_responses_fallbacks_to_chat_on_unsupported_error(self) -> None:
        client = OpenAILLMClient(
            api_key="test",
            api_mode=ApiMode.RESPONSES,
            allow_response_fallback_to_chat=True,
        )

        class FakeNotFoundError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("not found")
                self.status_code = 404

        msg_obj = MagicMock()
        msg_obj.content = "fallback answer"
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        chat_response = MagicMock()
        chat_response.choices = [choice]

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_responses_create, patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_chat_create:
            mock_responses_create.side_effect = FakeNotFoundError()
            mock_chat_create.return_value = chat_response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            )

        assert result.content == "fallback answer"
        mock_responses_create.assert_called_once()
        mock_chat_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_responses_mode_does_not_fallback_without_explicit_opt_in(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_responses_create, patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_chat_create:
            mock_responses_create.side_effect = RuntimeError("responses unsupported by backend")
            with pytest.raises(RuntimeError, match="responses unsupported by backend"):
                await client.complete(
                    messages=[Message(role="user", content="hi")],
                    model="gpt-4o",
                )

        mock_responses_create.assert_called_once()
        mock_chat_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_runtime_api_mode_override_uses_chat_path(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        msg_obj = MagicMock()
        msg_obj.content = "chat path"
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        chat_response = MagicMock()
        chat_response.choices = [choice]

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_responses_create, patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_chat_create:
            mock_chat_create.return_value = chat_response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                api_mode=ApiMode.CHAT_COMPLETIONS,
            )

        assert result.content == "chat path"
        mock_responses_create.assert_not_called()
        mock_chat_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_none_temperature_and_max_tokens_use_defaults(self) -> None:
        """Verify None values for temperature/max_tokens resolve to defaults."""
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        msg_obj = MagicMock()
        msg_obj.content = "ok"
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        chat_response = MagicMock()
        chat_response.choices = [choice]
        chat_response.usage = None

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = chat_response
            await client.complete(
                messages=[Message(role="user", content="hi")],
                model="test-model",
                temperature=None,
                max_tokens=None,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_kwargs_accepted_without_error(self) -> None:
        """Verify extra **kwargs are accepted (Protocol compliance)."""
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        msg_obj = MagicMock()
        msg_obj.content = "ok"
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        chat_response = MagicMock()
        chat_response.choices = [choice]
        chat_response.usage = None

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = chat_response
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="test-model",
                some_future_param="value",
            )
        assert result.content == "ok"
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["some_future_param"] == "value"

    @pytest.mark.asyncio
    async def test_chat_mode_stream_complete_collects_content_reasoning_usage_and_tool_calls(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        first_delta = SimpleNamespace(
            content=None,
            reasoning=SimpleNamespace(text="thinking"),
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="call-1",
                    function=SimpleNamespace(name="echo", arguments="{'x': 1}"),
                )
            ],
        )
        second_delta = SimpleNamespace(content="final", reasoning=None, tool_calls=[])

        chunk1 = SimpleNamespace(
            choices=[SimpleNamespace(delta=first_delta)],
            usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        )
        chunk2 = SimpleNamespace(
            choices=[SimpleNamespace(delta=second_delta)],
            usage=None,
        )

        async def fake_stream() -> Any:
            yield chunk1
            yield chunk2

        seen_events: list[dict[str, Any]] = []

        async def on_stream_event(event: dict[str, Any]) -> None:
            seen_events.append(event)

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            result = await client.complete(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                tools=[ToolDefinition(name="echo", description="Echo")],
                emit_reasoning_in_stream=True,
                stream_event_callback=on_stream_event,
            )

        assert result.content == "final"
        assert result.usage is not None
        assert result.usage.input_tokens == 3
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"x": 1}
        assert any(event["kind"] == "reasoning" for event in seen_events)
        assert any(event["kind"] == "text" for event in seen_events)
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["parallel_tool_calls"] is True
        assert mock_create.await_args.kwargs["stream_options"]["include_usage"] is True


class TestOpenAILLMClientStructured:
    class MySchema(BaseModel):
        answer: str
        confidence: float

    @pytest.mark.asyncio
    async def test_structured_responses_mode(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        mock_response = MagicMock()
        mock_response.output_text = '{"answer": "42", "confidence": 0.95}'

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
            )
            assert isinstance(result, self.MySchema)
            assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_structured_responses_respects_explicit_text_format_override(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        mock_response = MagicMock()
        mock_response.output_text = '{"answer": "ok", "confidence": 1.0}'

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
                text={"format": {"type": "json_object"}},
            )

        assert result.answer == "ok"
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["text"] == {"format": {"type": "json_object"}}

    @pytest.mark.asyncio
    async def test_structured_responses_fallbacks_to_chat_on_unsupported_error(self) -> None:
        client = OpenAILLMClient(
            api_key="test",
            api_mode=ApiMode.RESPONSES,
            allow_response_fallback_to_chat=True,
        )

        class FakeBadRequestError(RuntimeError):
            def __init__(self, message: str) -> None:
                super().__init__(message)
                self.status_code = 400

        msg_obj = MagicMock()
        msg_obj.content = '{"answer": "chat", "confidence": 0.7}'
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        chat_response = MagicMock()
        chat_response.choices = [choice]

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_responses_create, patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_chat_create:
            mock_responses_create.side_effect = FakeBadRequestError("responses api unsupported")
            mock_chat_create.return_value = chat_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
            )

        assert isinstance(result, self.MySchema)
        assert result.answer == "chat"
        mock_responses_create.assert_called_once()
        mock_chat_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_responses_bad_request_triggers_fallback_when_enabled(self) -> None:
        client = OpenAILLMClient(
            api_key="test",
            api_mode=ApiMode.RESPONSES,
            allow_response_fallback_to_chat=True,
        )

        class FakeBadRequestError(RuntimeError):
            def __init__(self, message: str) -> None:
                super().__init__(message)
                self.status_code = 400

        msg_obj = MagicMock()
        msg_obj.content = '{"answer": "chat", "confidence": 0.7}'
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        chat_response = MagicMock()
        chat_response.choices = [choice]

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_responses_create, patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_chat_create:
            mock_responses_create.side_effect = FakeBadRequestError(
                "validation error: unsupported field text.format"
            )
            mock_chat_create.return_value = chat_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
            )

        assert isinstance(result, self.MySchema)
        assert result.answer == "chat"
        assert mock_responses_create.await_count >= 1
        mock_chat_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_responses_retries_compatibility_formats_before_failing_over(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        class FakeSchemaError(RuntimeError):
            def __init__(self, message: str) -> None:
                super().__init__(message)
                self.status_code = 400

        mock_response = MagicMock()
        mock_response.output_text = '{"answer": "42", "confidence": 0.95}'

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [
                FakeSchemaError("validation error: unsupported field text.format.name"),
                FakeSchemaError("validation error: json_schema not supported"),
                mock_response,
            ]
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
            )

        assert isinstance(result, self.MySchema)
        assert result.answer == "42"
        assert mock_create.await_count == 3
        third_call = mock_create.await_args_list[2].kwargs
        assert third_call["text"]["format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_structured_chat_mode(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        msg_obj = MagicMock()
        msg_obj.content = '{"answer": "yes", "confidence": 0.8}'
        msg_obj.tool_calls = None

        choice = MagicMock()
        choice.message = msg_obj

        mock_response = MagicMock()
        mock_response.choices = [choice]

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
            )
            assert isinstance(result, self.MySchema)
            assert result.answer == "yes"

    @pytest.mark.asyncio
    async def test_structured_mode_forwards_extra_kwargs(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        msg_obj = MagicMock()
        msg_obj.content = '{"answer": "yes", "confidence": 0.8}'
        msg_obj.tool_calls = None
        choice = MagicMock()
        choice.message = msg_obj
        mock_response = MagicMock()
        mock_response.choices = [choice]

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=self.MySchema,
                extra_body={"top_k": 20},
            )

        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["extra_body"] == {"top_k": 20}

    @pytest.mark.asyncio
    async def test_subagent_result_uses_fallback_on_invalid_json(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        mock_response = MagicMock()
        mock_response.output_text = "not valid json"

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=SubagentResult,
            )
            assert isinstance(result, SubagentResult)
            assert result.status.value == "partial"

    @pytest.mark.asyncio
    async def test_subagent_result_parses_when_json_is_split_across_responses_parts(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        part1 = MagicMock()
        part1.text = '{"status":"success",'
        part2 = MagicMock()
        part2.text = '"summary":"ok","artifacts":[],"files_changed":[],"tool_calls_made":0,"errors":[]}'

        message_item = MagicMock()
        message_item.content = [part1, part2]

        mock_response = MagicMock()
        mock_response.output_text = None
        mock_response.output = [message_item]

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await client.complete_structured(
                messages=[Message(role="user", content="question")],
                schema=SubagentResult,
            )

        assert isinstance(result, SubagentResult)
        assert result.status == SubagentResult.model_validate(
            {
                "status": "success",
                "summary": "ok",
                "artifacts": [],
                "files_changed": [],
                "tool_calls_made": 0,
                "errors": [],
            }
        ).status
        assert result.summary == "ok"


# ---------------------------------------------------------------------------
# Usage extraction tests
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_responses_api_usage(self) -> None:
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cached_tokens = 10
        response = MagicMock()
        response.usage = usage
        result = _extract_usage(response)
        assert result is not None
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cached_tokens == 10

    def test_chat_completions_usage(self) -> None:
        usage = MagicMock()
        usage.input_tokens = 0
        usage.output_tokens = 0
        usage.prompt_tokens = 200
        usage.completion_tokens = 80
        usage.cached_tokens = 0
        response = MagicMock()
        response.usage = usage
        result = _extract_usage(response)
        assert result is not None
        assert result.input_tokens == 200
        assert result.output_tokens == 80

    def test_chat_completions_usage_dict_mapping_for_vllm(self) -> None:
        response = MagicMock()
        response.usage = {
            "prompt_tokens": 127,
            "completion_tokens": 39,
            "prompt_tokens_details": {"cached_tokens": 11},
        }

        result = _extract_usage(response)

        assert result is not None
        assert result.input_tokens == 127
        assert result.output_tokens == 39
        assert result.cached_tokens == 11

    def test_no_usage_returns_none(self) -> None:
        response = MagicMock(spec=[])
        assert _extract_usage(response) is None

    def test_invalid_usage_numbers_are_coerced_to_zero(self) -> None:
        response = MagicMock()
        response.usage = {
            "prompt_tokens": "invalid",
            "completion_tokens": None,
            "cached_tokens": "nan",
        }

        result = _extract_usage(response)

        assert result is not None
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cached_tokens == 0

    def test_parse_responses_includes_usage(self) -> None:
        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 20
        usage.cached_tokens = 0
        response = MagicMock()
        response.output = []
        response.output_text = "hello"
        response.usage = usage
        msg = _parse_responses_output(response)
        assert msg.usage is not None
        assert msg.usage.input_tokens == 10

    def test_parse_chat_includes_usage(self) -> None:
        usage = MagicMock()
        usage.input_tokens = 0
        usage.output_tokens = 0
        usage.prompt_tokens = 30
        usage.completion_tokens = 40
        usage.cached_tokens = 0

        msg_obj = MagicMock()
        msg_obj.content = "ok"
        msg_obj.tool_calls = None

        choice = MagicMock()
        choice.message = msg_obj

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        msg = _parse_chat_output(response)
        assert msg.usage is not None
        assert msg.usage.output_tokens == 40


class TestFallbackDetection:
    def test_generic_responses_error_does_not_silently_fallback(self) -> None:
        assert _should_fallback_to_chat(RuntimeError("responses request failed")) is False

    def test_bad_request_compatibility_error_can_trigger_fallback(self) -> None:
        class FakeBadRequest(RuntimeError):
            def __init__(self) -> None:
                super().__init__("response_format not supported")
                self.status_code = 400

        assert _should_fallback_to_chat(FakeBadRequest()) is True

    def test_image_input_error_does_not_trigger_chat_fallback(self) -> None:
        assert _should_fallback_to_chat(RuntimeError("unsupported image_url payload")) is False

    def test_http_404_triggers_fallback(self) -> None:
        class FakeNotFound(RuntimeError):
            def __init__(self) -> None:
                super().__init__("not found")
                self.status_code = 404

        assert _should_fallback_to_chat(FakeNotFound()) is True

    def test_unrelated_unknown_parameter_does_not_trigger_fallback(self) -> None:
        class FakeBadRequest(RuntimeError):
            def __init__(self) -> None:
                super().__init__("unknown parameter temperature_scale")
                self.status_code = 400

        assert _should_fallback_to_chat(FakeBadRequest()) is False

    def test_structured_format_errors_trigger_compatibility_retry(self) -> None:
        assert _should_retry_structured_responses_format(
            RuntimeError("validation error: unsupported field text.format")
        ) is True


class TestSafeAttr:
    def test_returns_dynamic_instance_attributes(self) -> None:
        class DynamicObject:
            call_id: str

        value = DynamicObject()
        value.call_id = "dynamic-123"

        assert _safe_attr(value, "call_id") == "dynamic-123"

    def test_missing_magicmock_attribute_returns_none(self) -> None:
        value = MagicMock()

        assert _safe_attr(value, "missing_field") is None

    def test_safe_attr_handles_dicts(self) -> None:
        value = {"call_id": "from-dict"}

        assert _safe_attr(value, "call_id") == "from-dict"

    def test_safe_attr_returns_none_for_none_value(self) -> None:
        assert _safe_attr(None, "anything") is None


class TestClientErrorHelpers:
    def test_response_preview_uses_model_dump_when_available(self) -> None:
        response = MagicMock()
        response.model_dump.return_value = {"hello": "world"}

        preview = _response_preview(response, limit=30)

        assert preview.startswith('{"hello": "world"}')

    def test_response_preview_falls_back_to_str_on_dump_error(self) -> None:
        class BadDump:
            def model_dump(self, mode: str = "json") -> Any:
                _ = mode
                raise RuntimeError("boom")

            def __str__(self) -> str:
                return "fallback-preview"

        assert _response_preview(BadDump()) == "fallback-preview"

    def test_error_status_code_extracts_numeric_strings(self) -> None:
        exc = RuntimeError("bad")
        setattr(exc, "response", SimpleNamespace(status_code="429"))

        assert _error_status_code(exc) == 429

    def test_error_status_code_uses_body_status_code(self) -> None:
        exc = RuntimeError("bad")
        setattr(exc, "body", {"status_code": 418})

        assert _error_status_code(exc) == 418

    def test_error_text_includes_message_body_and_response_text(self) -> None:
        exc = RuntimeError("base")
        setattr(exc, "message", "MSG")
        setattr(exc, "body", {"error": "oops"})
        setattr(exc, "response", SimpleNamespace(text="DETAIL"))

        text = _error_text(exc)

        assert "base" in text
        assert "msg" in text
        assert '"error": "oops"' in text
        assert "detail" in text


class TestRequestKwargHelpers:
    def test_pop_logging_context_extracts_dict_and_removes_key(self) -> None:
        kwargs: dict[str, Any] = {"logging_context": {"trace_id": "t-1"}, "other": 1}

        log_ctx = _pop_logging_context(kwargs)

        assert "logging_context" not in kwargs
        assert log_ctx["trace_id"] == "t-1"

    def test_pop_logging_context_returns_empty_for_non_dict(self) -> None:
        kwargs: dict[str, Any] = {"logging_context": "not-a-dict"}

        assert _pop_logging_context(kwargs) == {}

    def test_ensure_stream_usage_options_sets_default_and_merges_dict(self) -> None:
        kwargs: dict[str, Any] = {}
        _ensure_stream_usage_options(kwargs)
        assert kwargs["stream_options"]["include_usage"] is True

        with_existing: dict[str, Any] = {"stream_options": {"foo": "bar"}}
        _ensure_stream_usage_options(with_existing)
        assert with_existing["stream_options"]["foo"] == "bar"
        assert with_existing["stream_options"]["include_usage"] is True


# ---------------------------------------------------------------------------
# Structured output / coercion helper tests
# ---------------------------------------------------------------------------


class TestStructuredOutputHelpers:
    class SmallSchema(BaseModel):
        answer: str

    def test_validate_structured_output_raises_for_non_subagent_large_payload(self) -> None:
        raw = "x" * (MAX_STRUCTURED_JSON_CHARS + 1)
        with pytest.raises(ValueError, match="structured_output_too_large"):
            _validate_structured_output(raw, self.SmallSchema)

    def test_validate_structured_output_accepts_markdown_fenced_json(self) -> None:
        raw = '```json\n{"answer":"ok"}\n```'
        parsed = _validate_structured_output(raw, self.SmallSchema)
        assert parsed.answer == "ok"

    def test_validate_structured_output_raises_diagnostic_error_on_invalid_payload(self) -> None:
        raw = "not-json"
        with pytest.raises(ValueError, match="structured_output_schema_validation_failed"):
            _validate_structured_output(raw, self.SmallSchema)

    def test_validate_structured_output_aggregates_candidate_failures(self) -> None:
        raw = '```json\n{"wrong":"shape"}\n```\n{"still":"wrong"}'
        with pytest.raises(ValueError, match="no_json_candidate_validated:ValidationError"):
            _validate_structured_output(raw, self.SmallSchema)

    def test_validate_structured_output_collects_value_error_candidates(self) -> None:
        class ValueErrorSchema:
            __name__ = "ValueErrorSchema"

            @classmethod
            def model_validate_json(cls, value: str) -> Any:
                _ = value
                raise ValueError("bad payload")

        with pytest.raises(ValueError, match="no_json_candidate_validated:ValueError:bad payload"):
            _validate_structured_output('{"x":1}', ValueErrorSchema)

    def test_validate_structured_output_large_subagent_payload_uses_subagent_fallback(self) -> None:
        raw = "x" * (MAX_STRUCTURED_JSON_CHARS + 1)
        parsed = _validate_structured_output(raw, SubagentResult)
        assert isinstance(parsed, SubagentResult)
        assert parsed.status == SubagentStatus.PARTIAL


class TestResponseTextCoercion:
    class ValueWrapper:
        def __init__(self, value: str | None) -> None:
            self.value = value

    class TextPart:
        def __init__(self, text: Any) -> None:
            self.text = text

    def test_coerce_response_text_handles_string(self) -> None:
        assert _coerce_response_text(self.TextPart("hello")) == "hello"

    def test_coerce_response_text_handles_wrapped_value(self) -> None:
        assert _coerce_response_text(self.TextPart(self.ValueWrapper("wrapped"))) == "wrapped"

    def test_coerce_response_text_handles_missing_text(self) -> None:
        assert _coerce_response_text(self.TextPart(None)) == ""

    def test_coerce_response_text_falls_back_to_str(self) -> None:
        class Unknown:
            def __str__(self) -> str:
                return "fallback"

        assert _coerce_response_text(self.TextPart(Unknown())) == "fallback"


# ---------------------------------------------------------------------------
# Stream tests
# ---------------------------------------------------------------------------


class TestOpenAILLMClientStream:
    @pytest.mark.asyncio
    async def test_stream_responses_mode(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        event1 = MagicMock()
        event1.delta = "Hello"
        event2 = MagicMock()
        event2.delta = " world"

        async def fake_stream() -> Any:
            for e in [event1, event2]:
                yield e

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])
            assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_responses_mode_emits_tool_calls_from_completed_response(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        fc_item = SimpleNamespace(
            type="function_call",
            call_id="call_1",
            name="echo",
            arguments='{"text":"hi"}',
        )
        final_response = SimpleNamespace(
            output=[fc_item],
            output_text="",
            usage={"input_tokens": 9, "output_tokens": 4},
        )
        completed_event = SimpleNamespace(type="response.completed", response=final_response, usage=None)

        async def fake_stream() -> Any:
            yield completed_event

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            events = [
                event
                async for event in client.stream_with_tools(
                    messages=[Message(role="user", content="hi")],
                )
            ]

        assert events[0]["type"] == "tool_calls"
        assert events[0]["tool_calls"][0]["function"]["name"] == "echo"
        assert events[-1]["type"] == "done"
        assert events[-1]["usage"]["input_tokens"] == 9
        assert mock_create.await_count == 1

    @pytest.mark.asyncio
    async def test_stream_responses_mode_sends_tools_in_responses_shape(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)
        tools = [ToolDefinition(name="echo", description="Echoes")]

        event = MagicMock()
        event.delta = "ok"

        async def fake_stream() -> Any:
            yield event

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                tools=tools,
                model="gpt-4o",
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])

        assert chunks == ["ok"]
        assert mock_create.await_args is not None
        sent_tools = mock_create.await_args.kwargs["tools"]
        assert sent_tools[0]["name"] == "echo"
        assert "function" not in sent_tools[0]

    @pytest.mark.asyncio
    async def test_stream_chat_mode(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        delta1 = MagicMock()
        delta1.content = "Hi"
        choice1 = MagicMock()
        choice1.delta = delta1
        chunk1 = MagicMock()
        chunk1.choices = [choice1]

        delta2 = MagicMock()
        delta2.content = " there"
        choice2 = MagicMock()
        choice2.delta = delta2
        chunk2 = MagicMock()
        chunk2.choices = [choice2]

        async def fake_stream() -> Any:
            for c in [chunk1, chunk2]:
                yield c

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                extra_body={"top_k": 20},
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])
            assert chunks == ["Hi", " there"]
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["extra_body"] == {"top_k": 20}

    @pytest.mark.asyncio
    async def test_stream_chat_mode_enables_include_usage_for_stream_options(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        delta = MagicMock()
        delta.content = "Hi"
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]

        async def fake_stream() -> Any:
            yield chunk

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            async for _ in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            ):
                pass

        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_stream_chat_mode_respects_explicit_parallel_tool_calls_override(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)
        tool = ToolDefinition(name="echo", description="Echo")

        delta = MagicMock()
        delta.content = None
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]

        async def fake_stream() -> Any:
            yield chunk

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            async for _ in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                tools=[tool],
                model="gpt-4o",
                parallel_tool_calls=False,
            ):
                pass

        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_stream_chat_reasoning_is_suppressed_by_default(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        reasoning_delta = MagicMock()
        reasoning_part = MagicMock()
        reasoning_part.text = "hidden reasoning"
        reasoning_delta.content = None
        reasoning_delta.reasoning = [reasoning_part]
        choice = MagicMock()
        choice.delta = reasoning_delta
        chunk = MagicMock()
        chunk.choices = [choice]

        async def fake_stream() -> Any:
            yield chunk

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])

        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_chat_can_emit_reasoning_when_opted_in(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        reasoning_delta = MagicMock()
        reasoning_part = MagicMock()
        reasoning_part.text = "visible reasoning"
        reasoning_delta.content = None
        reasoning_delta.reasoning = [reasoning_part]
        choice = MagicMock()
        choice.delta = reasoning_delta
        chunk = MagicMock()
        chunk.choices = [choice]

        async def fake_stream() -> Any:
            yield chunk

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                emit_reasoning_in_stream=True,
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])

        assert chunks == ["visible reasoning"]

    @pytest.mark.asyncio
    async def test_stream_events_chat_separates_reasoning_kind(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        reasoning_delta = MagicMock()
        reasoning_part = MagicMock()
        reasoning_part.text = "visible reasoning"
        reasoning_delta.content = None
        reasoning_delta.reasoning = [reasoning_part]
        choice = MagicMock()
        choice.delta = reasoning_delta
        chunk = MagicMock()
        chunk.choices = [choice]

        async def fake_stream() -> Any:
            yield chunk

        with patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            events = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
                emit_reasoning_in_stream=True,
            ):
                events.append(stream_event)

        assert len(events) == 2
        assert events[0] == {
            "type": "delta",
            "text": "visible reasoning",
            "kind": "reasoning",
            "provider_event_type": "chat.completions.reasoning",
        }
        assert events[1]["type"] == "done"
        assert "usage" in events[1]

    @pytest.mark.asyncio
    async def test_stream_responses_reasoning_events_are_suppressed_by_default(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        event = MagicMock()
        event.type = "response.reasoning.delta"
        event.delta = "hidden reasoning"

        async def fake_stream() -> Any:
            yield event

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])

        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_events_responses_labels_text_delta_kind(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        event = MagicMock()
        event.type = "response.output_text.delta"
        event.delta = "Hello"

        async def fake_stream() -> Any:
            yield event

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_stream()
            events = []
            async for delta in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            ):
                events.append(delta)

        assert events == [
            {
                "type": "delta",
                "kind": "text",
                "text": "Hello",
                "provider_event_type": "response.output_text.delta",
            },
            {
                "type": "done",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cached_tokens": 1,
                    "reasoning_tokens": 1,
                },
            },
        ]

    @pytest.mark.asyncio
    async def test_stream_responses_fallbacks_to_chat_on_unsupported_error(self) -> None:
        client = OpenAILLMClient(
            api_key="test",
            api_mode=ApiMode.RESPONSES,
            allow_response_fallback_to_chat=True,
        )

        class FakeNotFoundError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("not found")
                self.status_code = 404

        delta = MagicMock()
        delta.content = "chat token"
        choice = MagicMock()
        choice.delta = delta
        chat_chunk = MagicMock()
        chat_chunk.choices = [choice]

        async def fake_chat_stream() -> Any:
            yield chat_chunk

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_responses_create, patch.object(
            client._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_chat_create:
            mock_responses_create.side_effect = FakeNotFoundError()
            mock_chat_create.return_value = fake_chat_stream()

            chunks = []
            async for stream_event in client.stream_with_tools(
                messages=[Message(role="user", content="hi")],
                model="gpt-4o",
            ):
                if stream_event.get("type") == "delta":
                    delta_event = cast(StreamDeltaEvent, stream_event)
                    chunks.append(delta_event["text"])

        assert chunks == ["chat token"]
        mock_responses_create.assert_called_once()
        mock_chat_create.assert_called_once()
