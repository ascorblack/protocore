from __future__ import annotations

import os
from typing import cast

import pytest
from pydantic import BaseModel

from protocore import ApiMode, Message, OpenAILLMClient
from protocore.types import StreamDeltaEvent, ToolDefinition, ToolParameterSchema


LIVE_FLAG = os.getenv("PROTOCORE_RUN_LIVE_LLM_TESTS") == "1"
LIVE_BASE_URL = os.getenv("PROTOCORE_LIVE_BASE_URL")
LIVE_MODEL = os.getenv("PROTOCORE_LIVE_MODEL")
LIVE_API_KEY = os.getenv("PROTOCORE_LIVE_API_KEY", "EMPTY")
LIVE_SUPPORTS_RESPONSES = os.getenv("PROTOCORE_LIVE_SUPPORTS_RESPONSES") == "1"

pytestmark = pytest.mark.skipif(
    not (LIVE_FLAG and LIVE_BASE_URL and LIVE_MODEL),
    reason=(
        "set PROTOCORE_RUN_LIVE_LLM_TESTS=1, "
        "PROTOCORE_LIVE_BASE_URL and PROTOCORE_LIVE_MODEL to run live tests"
    ),
)


def _live_client(
    *,
    api_mode: ApiMode,
    allow_response_fallback_to_chat: bool = False,
) -> OpenAILLMClient:
    return OpenAILLMClient(
        api_key=LIVE_API_KEY,
        base_url=LIVE_BASE_URL,
        api_mode=api_mode,
        default_model=LIVE_MODEL or "",
        allow_response_fallback_to_chat=allow_response_fallback_to_chat,
    )


class _AnswerSchema(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_complete_chat_reasoning_only_returns_non_empty_content_or_reasoning() -> None:
    client = _live_client(api_mode=ApiMode.CHAT_COMPLETIONS)

    result = await client.complete(
        messages=[
            Message(
                role="user",
                content=(
                    "Реши 347 * 219 пошагово, но из-за малого лимита ответа ты можешь не успеть "
                    "дать финальный ответ. Все равно верни хоть какой-то осмысленный текст."
                ),
            )
        ],
        max_tokens=48,
    )

    assert isinstance(result.content, str)
    assert result.content.strip()


@pytest.mark.asyncio
async def test_stream_chat_emits_reasoning_deltas_when_content_absent() -> None:
    client = _live_client(api_mode=ApiMode.CHAT_COMPLETIONS)

    chunks: list[str] = []
    async for event in client.stream_with_tools(
        messages=[
            Message(
                role="user",
                content="Коротко подумай над задачей 913 * 47 и начни вывод сразу.",
            )
        ],
        max_tokens=64,
        emit_reasoning_in_stream=True,
    ):
        if event.get("type") == "delta":
            delta_event = cast(StreamDeltaEvent, event)
            chunks.append(delta_event["text"])
        if len("".join(chunks)) >= 24:
            break

    assert "".join(chunks).strip()


@pytest.mark.asyncio
async def test_structured_responses_bad_request_triggers_fallback_when_enabled() -> None:
    client = _live_client(
        api_mode=ApiMode.RESPONSES,
        allow_response_fallback_to_chat=True,
    )

    result = await client.complete_structured(
        messages=[
            Message(
                role="user",
                content='Верни JSON вида {"answer":"ok"} и ничего кроме JSON.',
            )
        ],
        schema=_AnswerSchema,
        max_tokens=256,
    )

    assert isinstance(result, _AnswerSchema)
    assert result.answer.strip()


@pytest.mark.asyncio
async def test_api_mode_override_forces_chat_without_touching_responses() -> None:
    client = _live_client(api_mode=ApiMode.RESPONSES)

    result = await client.complete(
        messages=[Message(role="user", content="Напиши только слово ok.")],
        api_mode=ApiMode.CHAT_COMPLETIONS,
        max_tokens=32,
    )

    assert isinstance(result.content, str)
    assert result.content.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not LIVE_SUPPORTS_RESPONSES,
    reason="set PROTOCORE_LIVE_SUPPORTS_RESPONSES=1 for live Responses API coverage",
)
async def test_complete_responses_parses_reasoning_output() -> None:
    client = _live_client(api_mode=ApiMode.RESPONSES)

    result = await client.complete(
        messages=[
            Message(
                role="user",
                content=(
                    "Реши задачу 1287 / 9 пошагово. Если финальный ответ не поместится, "
                    "верни хотя бы reasoning."
                ),
            )
        ],
        max_tokens=64,
    )

    assert isinstance(result.content, str)
    assert result.content.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not LIVE_SUPPORTS_RESPONSES,
    reason="set PROTOCORE_LIVE_SUPPORTS_RESPONSES=1 for live Responses API coverage",
)
async def test_complete_responses_normalizes_function_call_ids() -> None:
    client = _live_client(api_mode=ApiMode.RESPONSES)

    result = await client.complete(
        messages=[
            Message(
                role="user",
                content="Use the tool to get weather for Paris. Return only the tool call.",
            )
        ],
        tools=[
            ToolDefinition(
                name="lookup_weather",
                description="Lookup weather by city",
                parameters=ToolParameterSchema(
                    type="object",
                    properties={"city": {"type": "string"}},
                    required=["city"],
                    additionalProperties=False,
                ),
                strict=True,
            )
        ],
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert result.tool_calls is not None
    assert result.tool_calls[0].id
    assert result.tool_calls[0].call_id == result.tool_calls[0].id
    assert result.tool_calls[0].function.name == "lookup_weather"
