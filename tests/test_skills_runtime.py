from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from protocore import (
    AgentConfig,
    AgentRole,
    AgentOrchestrator,
    CoreEvent,
    EventBus,
    ExecutionMode,
    ExecutionStatus,
    Message,
    RunKind,
    SkillIndexEntry,
    SkillLoadResult,
    SkillManifest,
    SkillTrustLevel,
    StopReason,
    make_agent_context,
    make_execution_report,
)
from protocore.events import (
    EV_SKILL_BUDGET_EXCEEDED,
    EV_SKILL_INDEX_INJECTED,
    EV_SKILL_LOAD_END,
    EV_SKILL_LOAD_START,
)
from protocore.types import ToolCall


class FakeSkillManager:
    def __init__(self) -> None:
        self._catalog: dict[str, tuple[SkillManifest, str]] = {
            "testing-code": (
                SkillManifest(
                    name="testing-code",
                    description="Testing workflow",
                    tags=["testing"],
                    trust_level=SkillTrustLevel.MANAGED,
                ),
                "Run focused tests first.",
            ),
            "git-workflow": (
                SkillManifest(
                    name="git-workflow",
                    description="Git workflow hints",
                    tags=["git"],
                    trust_level=SkillTrustLevel.MANAGED,
                ),
                "Check git status before commit.",
            ),
        }
        self._session_loaded: dict[str, set[str]] = {}

    async def get_index(
        self,
        agent_id: str,
        skill_names: list[str],
        max_chars: int = 1500,
    ) -> list[SkillIndexEntry]:
        _ = (agent_id, max_chars)
        entries: list[SkillIndexEntry] = []
        for name in skill_names:
            if name not in self._catalog:
                continue
            manifest, _ = self._catalog[name]
            entries.append(
                SkillIndexEntry(
                    name=manifest.name,
                    description=manifest.description,
                    tags=manifest.tags,
                    trust_level=manifest.trust_level,
                )
            )
        return entries

    async def load_skill(
        self,
        name: str,
        agent_id: str,
        max_chars: int = 8000,
    ) -> SkillLoadResult:
        _ = agent_id
        manifest, body = self._catalog[name]
        truncated = len(body) > max_chars
        body_out = body[:max_chars] if truncated else body
        return SkillLoadResult(
            name=name,
            body=body_out,
            manifest=manifest,
            estimated_tokens=max(1, len(body_out) // 4),
            truncated=truncated,
            from_cache=False,
        )

    async def mark_loaded(self, name: str, session_id: str) -> None:
        self._session_loaded.setdefault(session_id, set()).add(name)

    async def is_loaded(self, name: str, session_id: str) -> bool:
        return name in self._session_loaded.get(session_id, set())

    async def get_load_count(self, session_id: str) -> int:
        return len(self._session_loaded.get(session_id, set()))


@pytest.mark.asyncio
async def test_skills_index_injection_and_load_tool_flow() -> None:
    llm = MagicMock()
    calls = 0

    async def _complete(*args: Any, **kwargs: Any) -> Message:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert "## Available skills" in str(kwargs.get("system", ""))
            tool_names = [tool.name for tool in kwargs.get("tools", [])]
            assert "load_skill" in tool_names
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall.model_validate({
                        "id": "tc-skill-1",
                        "function": {
                            "name": "load_skill",
                            "arguments": '{"name":"testing-code"}',
                        },
                    })
                ],
            )
        return Message(role="assistant", content="done with skill")

    llm.complete = AsyncMock(side_effect=_complete)

    events: list[str] = []
    bus = EventBus()

    async def _collect(event: CoreEvent) -> None:
        events.append(event.name)

    bus.subscribe("*", _collect)

    cfg = AgentConfig(
        name="leader",
        role=AgentRole.LEADER,
        execution_mode=ExecutionMode.BYPASS,
        skill_set=["testing-code"],
        max_skill_loads_per_run=2,
    )
    context = make_agent_context(config=cfg)
    context.messages.append(Message(role="user", content="Run tests"))

    orch = AgentOrchestrator(
        llm_client=llm,
        event_bus=bus,
        skill_manager=FakeSkillManager(),
    )
    result, report = await orch.run(context)

    assert result.status == ExecutionStatus.COMPLETED
    assert report.stop_reason == StopReason.END_TURN
    assert "skill_loaded:testing-code" in report.artifacts
    assert EV_SKILL_INDEX_INJECTED in events
    assert EV_SKILL_LOAD_START in events
    assert EV_SKILL_LOAD_END in events


@pytest.mark.asyncio
async def test_skills_budget_exceeded_event_is_emitted() -> None:
    llm = MagicMock()
    calls = 0

    async def _complete(*args: Any, **kwargs: Any) -> Message:
        nonlocal calls
        _ = args
        calls += 1
        if calls == 1:
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall.model_validate({
                        "id": "tc-skill-a",
                        "function": {
                            "name": "load_skill",
                            "arguments": '{"name":"testing-code"}',
                        },
                    })
                ],
            )
        if calls == 2:
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall.model_validate({
                        "id": "tc-skill-b",
                        "function": {
                            "name": "load_skill",
                            "arguments": '{"name":"git-workflow"}',
                        },
                    })
                ],
            )
        return Message(role="assistant", content="finished")

    llm.complete = AsyncMock(side_effect=_complete)

    events: list[str] = []
    bus = EventBus()

    async def _collect(event: CoreEvent) -> None:
        events.append(event.name)

    bus.subscribe("*", _collect)

    cfg = AgentConfig(
        role=AgentRole.LEADER,
        execution_mode=ExecutionMode.BYPASS,
        skill_set=["testing-code", "git-workflow"],
        max_skill_loads_per_run=1,
        max_iterations=4,
    )
    context = make_agent_context(config=cfg)
    context.messages.append(Message(role="user", content="Need two skill loads"))

    orch = AgentOrchestrator(
        llm_client=llm,
        event_bus=bus,
        skill_manager=FakeSkillManager(),
    )
    result, report = await orch.run(context)

    assert result.status == ExecutionStatus.COMPLETED
    assert "skill_budget_exceeded" in report.warnings
    assert EV_SKILL_BUDGET_EXCEEDED in events


@pytest.mark.asyncio
async def test_spawned_children_isolate_load_skill_handlers() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=Message(role="assistant", content="done"))
    skill_manager = FakeSkillManager()

    parent = AgentOrchestrator(
        llm_client=llm,
        skill_manager=skill_manager,
    )
    child_one = parent._spawn_child_orchestrator()
    child_two = parent._spawn_child_orchestrator()

    cfg_one = AgentConfig(
        agent_id="child-one",
        role=AgentRole.SUBAGENT,
        execution_mode=ExecutionMode.BYPASS,
        skill_set=["testing-code", "git-workflow"],
        max_skill_loads_per_run=1,
    )
    cfg_two = AgentConfig(
        agent_id="child-two",
        role=AgentRole.SUBAGENT,
        execution_mode=ExecutionMode.BYPASS,
        skill_set=["testing-code", "git-workflow"],
        max_skill_loads_per_run=2,
    )
    ctx_one = make_agent_context(config=cfg_one)
    ctx_two = make_agent_context(config=cfg_two)

    await child_one._prepare_config_with_skills(
        context=ctx_one,
        report=make_execution_report(context=ctx_one, run_kind=RunKind.SUBAGENT),
    )
    await child_two._prepare_config_with_skills(
        context=ctx_two,
        report=make_execution_report(context=ctx_two, run_kind=RunKind.SUBAGENT),
    )

    assert "load_skill" not in parent._tool_registry
    handler_one = child_one._tool_registry.get_handler("load_skill")
    handler_two = child_two._tool_registry.get_handler("load_skill")
    assert handler_one is not None
    assert handler_two is not None
    assert handler_one is not handler_two

    result_one_a = await child_one._tool_registry.dispatch(
        "load_skill",
        {"name": "testing-code"},
        ctx_one.tool_context,
    )
    result_one_b = await child_one._tool_registry.dispatch(
        "load_skill",
        {"name": "git-workflow"},
        ctx_one.tool_context,
    )
    result_two_a = await child_two._tool_registry.dispatch(
        "load_skill",
        {"name": "testing-code"},
        ctx_two.tool_context,
    )
    result_two_b = await child_two._tool_registry.dispatch(
        "load_skill",
        {"name": "git-workflow"},
        ctx_two.tool_context,
    )

    assert result_one_a is not None and result_one_a.is_error is False
    assert result_one_b is not None and result_one_b.is_error is True
    assert result_two_a is not None and result_two_a.is_error is False
    assert result_two_b is not None and result_two_b.is_error is False

