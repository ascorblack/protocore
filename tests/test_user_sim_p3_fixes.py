from __future__ import annotations

from typing import Any

import pytest

from protocore import (
    AgentConfig,
    CapabilityBasedSelectionPolicy,
    FirstAvailablePolicy,
    Message,
    NoOpPlanningStrategy,
    make_agent_context,
)
from protocore.testing import FakeLLMClient
from protocore.types import StreamEvent


class TestFix011NoOpPlanningStrategy:
    @pytest.mark.asyncio
    async def test_no_op_planning_strategy_is_exported_and_returns_raw_task(self) -> None:
        planning = NoOpPlanningStrategy()
        context = make_agent_context(config=AgentConfig(model="test-model"))

        plan = await planning.build_plan(
            "calculate fibonacci",
            context=context,
            llm_client=FakeLLMClient(default_complete_response=Message(role="assistant", content="unused")),
        )

        assert plan.raw_plan == "calculate fibonacci"


class TestFix012BuiltInSelectionPolicies:
    @pytest.mark.asyncio
    async def test_first_available_policy_selects_first_agent(self) -> None:
        policy = FirstAvailablePolicy()
        context = make_agent_context(config=AgentConfig(model="test-model"))

        selected = await policy.select(
            task="Solve math question",
            available_agents=["math-subagent", "code-subagent"],
            context=context,
        )

        assert selected == "math-subagent"

    @pytest.mark.asyncio
    async def test_capability_based_policy_selects_agent_from_llm_json_response(self) -> None:
        fake_llm = FakeLLMClient(
            complete_responses=[
                Message(role="assistant", content='{"agent_id":"code-subagent"}'),
            ]
        )
        policy = CapabilityBasedSelectionPolicy(
            llm_client=fake_llm,
            agent_descriptions={
                "math-subagent": "Math computations and formulas",
                "code-subagent": "Code generation and debugging",
            },
        )
        context = make_agent_context(config=AgentConfig(model="test-model"))

        selected = await policy.select(
            task="Write Python code for quicksort",
            available_agents=["math-subagent", "code-subagent"],
            context=context,
        )

        assert selected == "code-subagent"
        assert fake_llm.complete_calls

    @pytest.mark.asyncio
    async def test_capability_based_policy_falls_back_when_llm_output_is_invalid(self) -> None:
        fake_llm = FakeLLMClient(
            complete_responses=[Message(role="assistant", content="unknown-agent")]
        )
        policy = CapabilityBasedSelectionPolicy(llm_client=fake_llm)
        context = make_agent_context(config=AgentConfig(model="test-model"))

        selected = await policy.select(
            task="Any task",
            available_agents=["a", "b"],
            context=context,
        )

        assert selected == "a"


class TestFix014FakeLLMClient:
    @pytest.mark.asyncio
    async def test_fake_llm_client_supports_complete_structured_and_stream(self) -> None:
        stream_sequence: list[StreamEvent] = [
            {"type": "delta", "kind": "text", "text": "hel"},
            {"type": "delta", "kind": "text", "text": "lo"},
            {"type": "done", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
        fake = FakeLLMClient(
            complete_responses=[Message(role="assistant", content="final")],
            structured_responses=[{"answer": 42}],
            stream_sequences=[stream_sequence],
        )

        message = await fake.complete(messages=[Message(role="user", content="hi")], stream=False)
        structured: Any = await fake.complete_structured(
            messages=[Message(role="user", content="schema")],
            schema=dict,
        )
        events = [event async for event in fake.stream_with_tools(messages=[])]

        assert message.content == "final"
        assert structured == {"answer": 42}
        assert events[-1]["type"] == "done"
        assert fake.complete_calls
        assert fake.complete_structured_calls
        assert fake.stream_calls
