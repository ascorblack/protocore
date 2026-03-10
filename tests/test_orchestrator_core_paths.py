from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    AgentContextMeta,
    CancellationContext,
    EventBus,
    ExecutionMode,
    ExecutionStatus,
    Message,
    RunKind,
    SubagentResult,
    SubagentRunSummary,
    SubagentStatus,
    make_agent_context,
    make_execution_report,
)
from protocore.constants import LOAD_SKILL_TOOL_NAME
from protocore.events import EV_SKILL_INDEX_INJECTED
from protocore.types import SkillIndexEntry, SkillTrustLevel, ToolContextMeta


def _final_llm(content: str = "ok") -> Any:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=Message(role="assistant", content=content))
    return llm


def _valid_parallel_policy() -> Any:
    return SimpleNamespace(
        max_concurrency=2,
        timeout_seconds=5.0,
        cancellation_mode="graceful",
    )


def _summary(agent_id: str, status: ExecutionStatus = ExecutionStatus.COMPLETED) -> SubagentRunSummary:
    now = datetime.now(timezone.utc).isoformat()
    return SubagentRunSummary(
        agent_id=agent_id,
        run_kind=RunKind.SUBAGENT,
        status=status,
        started_at=now,
        finished_at=now,
        duration_ms=1.0,
    )


class TestOrchestratorSkillPreparation:
    @pytest.mark.asyncio
    async def test_prepare_config_with_skills_injects_tool_metadata_and_index(self) -> None:
        skill_manager = MagicMock()
        skill_manager.get_index = AsyncMock(
            return_value=[
                SkillIndexEntry(
                    name="openai-docs",
                    description="Official OpenAI docs lookup",
                    tags=["docs", "openai"],
                    trust_level=SkillTrustLevel.MANAGED,
                )
            ]
        )

        bus = EventBus()
        seen: list[Any] = []

        async def on_index_injected(event: Any) -> None:
            seen.append(event)

        bus.subscribe(EV_SKILL_INDEX_INJECTED, on_index_injected)

        cfg = AgentConfig(
            agent_id="leader",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            skill_set=["openai-docs"],
            system_prompt="Base system",
        )
        context = make_agent_context(config=cfg)
        report = make_execution_report(context=context, run_kind=RunKind.LEADER)

        orch = AgentOrchestrator(llm_client=_final_llm(), skill_manager=skill_manager, event_bus=bus)
        updated = await orch._prepare_config_with_skills(context=context, report=report)

        assert any(tool.name == LOAD_SKILL_TOOL_NAME for tool in updated.tool_definitions)
        assert "## Available skills" in updated.system_prompt
        assert "openai-docs" in updated.system_prompt
        assert context.tool_context.metadata[ToolContextMeta.SKILL_SET] == ["openai-docs"]
        assert context.tool_context.metadata[ToolContextMeta.SKILL_LOAD_MAX_CHARS] == cfg.skill_load_max_chars
        assert (
            context.tool_context.metadata[ToolContextMeta.MAX_SKILL_LOADS_PER_RUN]
            == cfg.max_skill_loads_per_run
        )
        assert len(seen) == 1
        assert seen[0].payload["skill_count"] == 1

    @pytest.mark.asyncio
    async def test_prepare_config_with_skills_handles_index_errors_without_breaking_run(self) -> None:
        skill_manager = MagicMock()
        skill_manager.get_index = AsyncMock(side_effect=RuntimeError("index down"))

        cfg = AgentConfig(
            agent_id="leader",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            skill_set=["broken-skill"],
            system_prompt="Base",
        )
        context = make_agent_context(config=cfg)
        report = make_execution_report(context=context, run_kind=RunKind.LEADER)

        orch = AgentOrchestrator(llm_client=_final_llm(), skill_manager=skill_manager)
        updated = await orch._prepare_config_with_skills(context=context, report=report)

        assert updated.system_prompt == "Base"
        assert any(tool.name == LOAD_SKILL_TOOL_NAME for tool in updated.tool_definitions)
        assert any(w.startswith("skill_index_load_failed:RuntimeError") for w in report.warnings)


class TestOrchestratorParallelMode:
    @pytest.mark.asyncio
    async def test_parallel_mode_maps_pending_shell_approvals_into_context_and_result(self) -> None:
        cfg = AgentConfig(
            agent_id="leader",
            model="gpt-4o",
            execution_mode=ExecutionMode.PARALLEL,
        )
        context = make_agent_context(
            config=cfg,
            metadata={AgentContextMeta.PARALLEL_AGENT_IDS: ["worker-b", "worker-a"]},
        )
        context.messages.append(Message(role="user", content="dispatch this task"))
        report = make_execution_report(context=context, run_kind=RunKind.LEADER)

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            parallel_execution_policy=_valid_parallel_policy(),
        )
        orch._build_subagent_context = MagicMock(return_value=context)  # type: ignore[method-assign]

        merged = SubagentResult(
            status=SubagentStatus.SUCCESS,
            summary="merged summary",
            artifacts=["artifact:a"],
            files_changed=["changed.txt"],
            tool_calls_made=0,
            errors=[],
        )
        summaries = [_summary("worker-a"), _summary("worker-b")]

        async def fake_run_parallel(
            tasks: list[tuple[str, Any]],
            cancel_ctx: CancellationContext | None = None,
            report: Any | None = None,
        ) -> tuple[SubagentResult, list[SubagentRunSummary]]:
            assert [agent_id for agent_id, _ in tasks] == ["worker-a", "worker-b"]
            assert cancel_ctx is not None
            assert report is not None
            report.metadata["pending_shell_approvals"] = [
                {"agent_id": "worker-a", "plan": {"plan_id": "plan-a"}},
                {"agent_id": "worker-b", "plan": {"plan_id": "plan-b"}},
                "invalid",
            ]
            return merged, summaries

        with patch(
            "protocore.orchestrator.ParallelSubagentRunner.run_parallel",
            new=AsyncMock(side_effect=fake_run_parallel),
        ):
            result = await orch._run_parallel_mode(
                context=context,
                report=report,
                cancel_ctx=CancellationContext(),
            )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.content == "merged summary"
        assert report.status == ExecutionStatus.COMPLETED
        assert context.metadata["subagent_pending_shell_approvals"] == {
            "worker-a": {"plan_id": "plan-a"},
            "worker-b": {"plan_id": "plan-b"},
        }
        assert context.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] == {"plan_id": "plan-a"}
        assert result.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] == {"plan_id": "plan-a"}
        assert "pending_shell_approvals" in result.metadata

    @pytest.mark.asyncio
    async def test_parallel_mode_clears_stale_pending_approval_metadata_when_none_reported(self) -> None:
        cfg = AgentConfig(
            agent_id="leader",
            model="gpt-4o",
            execution_mode=ExecutionMode.PARALLEL,
        )
        context = make_agent_context(
            config=cfg,
            metadata={
                AgentContextMeta.PARALLEL_AGENT_IDS: ["worker-a"],
                "subagent_pending_shell_approvals": {"stale": {"plan_id": "old"}},
                AgentContextMeta.PENDING_SHELL_APPROVAL: {"plan_id": "old"},
            },
        )
        context.messages.append(Message(role="user", content="dispatch this task"))
        report = make_execution_report(context=context, run_kind=RunKind.LEADER)

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            parallel_execution_policy=_valid_parallel_policy(),
        )
        orch._build_subagent_context = MagicMock(return_value=context)  # type: ignore[method-assign]

        merged = SubagentResult(
            status=SubagentStatus.SUCCESS,
            summary="ok",
            artifacts=[],
            files_changed=[],
            tool_calls_made=0,
            errors=[],
        )
        summaries = [_summary("worker-a")]

        with patch(
            "protocore.orchestrator.ParallelSubagentRunner.run_parallel",
            new=AsyncMock(return_value=(merged, summaries)),
        ):
            result = await orch._run_parallel_mode(
                context=context,
                report=report,
                cancel_ctx=CancellationContext(),
            )

        assert result.metadata == {}
        assert "subagent_pending_shell_approvals" not in context.metadata
        assert AgentContextMeta.PENDING_SHELL_APPROVAL not in context.metadata


class TestOrchestratorDescribeSubagent:
    def test_describe_subagent_prefers_description_then_legacy_then_name(self) -> None:
        explicit = AgentConfig(agent_id="a1", model="gpt-4o", description="explicit description")
        assert AgentOrchestrator._describe_subagent(explicit) == "explicit description"

        legacy = AgentConfig(
            agent_id="a2",
            model="gpt-4o",
            description="",
            name="a2",
            extra={"description": "legacy description"},
        )
        assert AgentOrchestrator._describe_subagent(legacy) == "legacy description"

        by_name = AgentConfig(agent_id="a3", model="gpt-4o", description="", name="Planner")
        assert AgentOrchestrator._describe_subagent(by_name) == "Agent name: Planner"

        fallback = AgentConfig(agent_id="a4", model="gpt-4o", description="", name="a4")
        assert AgentOrchestrator._describe_subagent(fallback) == "No description provided."
