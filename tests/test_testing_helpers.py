from __future__ import annotations

import pytest

from protocore import Message
from protocore.testing import FakeLLMClient


@pytest.mark.asyncio
async def test_fake_llm_client_enqueue_and_call_records() -> None:
    client = FakeLLMClient(
        default_complete_response=Message(role="assistant", content="default"),
        default_structured_response={"ok": True},
        default_stream_sequence=[{"type": "delta", "text": "d", "kind": "text"}],
    )
    client.enqueue_complete(Message(role="assistant", content="queued"))
    client.enqueue_structured({"answer": 42})
    client.enqueue_stream([{"type": "done", "usage": {}}])

    complete = await client.complete(messages=[Message(role="user", content="hi")], stream=False)
    structured = await client.complete_structured(
        messages=[Message(role="user", content="hi")],
        schema=dict,
    )
    streamed = [event async for event in client.stream_with_tools(messages=[])]

    assert complete.content == "queued"
    assert structured == {"answer": 42}
    assert streamed == [{"type": "done", "usage": {}}]
    assert client.complete_calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_fake_llm_client_uses_defaults_and_raises_without_configuration() -> None:
    with_defaults = FakeLLMClient(
        default_complete_response=Message(role="assistant", content="fallback"),
        default_structured_response={"fallback": True},
    )

    assert (await with_defaults.complete(messages=[])).content == "fallback"
    assert await with_defaults.complete_structured(messages=[], schema=dict) == {"fallback": True}

    without_defaults = FakeLLMClient(default_stream_sequence=[{"type": "done", "usage": {}}])
    with pytest.raises(RuntimeError, match="no configured response for complete\\(\\)"):
        await without_defaults.complete(messages=[])
    with pytest.raises(RuntimeError, match="no configured response for complete_structured\\(\\)"):
        await without_defaults.complete_structured(messages=[], schema=dict)
