from __future__ import annotations

from typing import cast

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from protocore import ApiMode, Message, OpenAILLMClient
from protocore.types import StreamDeltaEvent

class _StructuredAnswer(BaseModel):
    answer: str


QWEN_DISABLE_THINKING_KWARGS = {
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False},
    }
}


def _chat_client(cfg: object) -> OpenAILLMClient:
    api_key = getattr(cfg, "api_key")
    base_url = getattr(cfg, "base_url")
    model = getattr(cfg, "model")
    return OpenAILLMClient(
        api_key=api_key,
        base_url=base_url,
        api_mode=ApiMode.CHAT_COMPLETIONS,
        default_model=model,
    )


def _responses_client_with_fallback(cfg: object) -> OpenAILLMClient:
    api_key = getattr(cfg, "api_key")
    base_url = getattr(cfg, "base_url")
    model = getattr(cfg, "model")
    return OpenAILLMClient(
        api_key=api_key,
        base_url=base_url,
        api_mode=ApiMode.RESPONSES,
        default_model=model,
        allow_response_fallback_to_chat=True,
    )


@pytest.mark.asyncio
async def test_vllm_models_endpoint_contains_qwen35(live_vllm_config: object) -> None:
    client = AsyncOpenAI(
        api_key=getattr(live_vllm_config, "api_key"),
        base_url=getattr(live_vllm_config, "base_url"),
    )
    response = await client.models.list()
    model_ids = [item.id for item in response.data]

    assert getattr(live_vllm_config, "model") in model_ids


@pytest.mark.asyncio
async def test_complete_chat_returns_non_empty_answer(live_vllm_config: object) -> None:
    client = _chat_client(live_vllm_config)

    result = await client.complete(
        messages=[Message(role="user", content="Ответь ровно одним словом: pong")],
        temperature=0.0,
        max_tokens=32,
        **QWEN_DISABLE_THINKING_KWARGS,  # type: ignore[arg-type]
    )

    assert isinstance(result.content, str)
    assert result.content.strip()
    assert "pong" in result.content.lower()


@pytest.mark.asyncio
async def test_stream_chat_emits_text_chunks(live_vllm_config: object) -> None:
    client = _chat_client(live_vllm_config)

    chunks: list[str] = []
    async for event in client.stream_with_tools(
        messages=[Message(role="user", content="Напиши 3 коротких слова через пробел.")],
        max_tokens=48,
        **QWEN_DISABLE_THINKING_KWARGS,  # type: ignore[arg-type]
    ):
        if event.get("type") == "delta":
            delta_event = cast(StreamDeltaEvent, event)
            chunks.append(delta_event["text"])
        if len("".join(chunks)) >= 12:
            break

    assert "".join(chunks).strip()


@pytest.mark.asyncio
async def test_complete_structured_chat_returns_valid_schema(
    live_vllm_config: object,
) -> None:
    client = _chat_client(live_vllm_config)

    result = await client.complete_structured(
        messages=[
            Message(
                role="user",
                content='Верни JSON строго вида {"answer":"ok"} без дополнительного текста.',
            )
        ],
        schema=_StructuredAnswer,
        temperature=0.0,
        max_tokens=96,
        **QWEN_DISABLE_THINKING_KWARGS,  # type: ignore[arg-type]
    )

    assert isinstance(result, _StructuredAnswer)
    assert result.answer.strip()


@pytest.mark.asyncio
async def test_complete_structured_with_system_param_and_system_message_merged(
    live_vllm_config: object,
) -> None:
    """complete_structured with both system= and Message(role='system') — single merged system."""
    client = _chat_client(live_vllm_config)

    result = await client.complete_structured(
        messages=[
            Message(role="user", content='Верни JSON вида {"answer":"merged"} без пояснений.'),
            Message(role="system", content="Return only the requested JSON object."),
        ],
        schema=_StructuredAnswer,
        system="You are a strict information extractor.",
        temperature=0.0,
        max_tokens=96,
        **QWEN_DISABLE_THINKING_KWARGS,  # type: ignore[arg-type]
    )

    assert isinstance(result, _StructuredAnswer)
    assert result.answer.strip()


@pytest.mark.asyncio
async def test_responses_mode_with_fallback_produces_answer(
    live_vllm_config: object,
) -> None:
    client = _responses_client_with_fallback(live_vllm_config)

    result = await client.complete(
        messages=[Message(role="user", content="Коротко: почему вода мокрая?")],
        max_tokens=80,
        **QWEN_DISABLE_THINKING_KWARGS,  # type: ignore[arg-type]
    )

    assert isinstance(result.content, str)
    assert result.content.strip()


@pytest.mark.asyncio
async def test_manual_compact_on_live_llm_produces_summary_with_stats(
    live_vllm_config: object,
) -> None:
    """manual_compact against real model returns CompactionSummary with statistics."""
    from protocore.compression import manual_compact

    client = _chat_client(live_vllm_config)
    messages = [
        Message(role="user", content="First: what is 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="Second: capital of France?"),
        Message(role="assistant", content="Paris."),
        Message(role="user", content="Third: summarize the above in one line."),
    ]
    new_messages, summary = await manual_compact(
        messages,
        llm_client=client,
        model=getattr(live_vllm_config, "model"),
        config=None,
    )
    assert len(new_messages) <= len(messages)
    assert summary.original_count >= 0
    assert summary.compacted_count >= 0
