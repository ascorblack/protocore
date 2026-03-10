"""Pluggy hook specifications for Protocore.

Service plugins implement these to extend the immutable loop WITHOUT
modifying orchestrator.py. All extension points are here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pluggy

if TYPE_CHECKING:
    from ..types import (
        AgentContext,
        ExecutionReport,
        Message,
        PlanArtifact,
        PolicyDecision,
        Result,
        SubagentResult,
        ToolContext,
        ToolDefinition,
        ToolResult,
        WorkflowDefinition,
    )

from ..constants import HOOKS_PROJECT_NAME as PROJECT_NAME

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class AgentHookSpecs:
    """Complete set of extension points for the agent loop.

    Implementations in lan-agent-service register via pluggy plugin manager.
    Hook methods that return values use firstresult=True so the first
    non-None result wins.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookspec
    def on_session_start(self, context: "AgentContext", report: "ExecutionReport") -> None:
        """Called once when a new agent session begins."""

    @hookspec
    def on_session_end(self, context: "AgentContext", report: "ExecutionReport") -> None:
        """Called once when a session ends (success, error, or cancel)."""

    # ------------------------------------------------------------------
    # Pre-LLM hooks (called before every LLM invocation in the loop)
    # ------------------------------------------------------------------

    @hookspec
    def pre_llm_call(
        self,
        messages: list["Message"],
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Inspect/mutate messages before the LLM call.

        Use for micro_compact, inbox drain, identity re-injection, etc.
        Mutations to *messages* list are in-place.
        """

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    @hookspec
    def on_tool_registered(self, tool: "ToolDefinition") -> None:
        """Called when a tool is added to the registry."""

    @hookspec(firstresult=True)
    def on_tool_pre_execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: "ToolContext",
        report: "ExecutionReport",
    ) -> "PolicyDecision | None":
        """Pre-execution policy check.

        Return DENY to block execution, CONFIRM to require user approval,
        ALLOW or None to proceed. First non-None result wins.
        """

    @hookspec
    def on_tool_post_execute(
        self,
        result: "ToolResult",
        context: "ToolContext",
        report: "ExecutionReport",
    ) -> None:
        """Post-execution hook: log, validate, update state, etc."""

    # ------------------------------------------------------------------
    # Compression hooks
    # ------------------------------------------------------------------

    @hookspec
    def on_micro_compact(
        self,
        messages: list["Message"],
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called after micro_compact is applied."""

    @hookspec
    def on_auto_compact(
        self,
        messages: list["Message"],
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called after auto_compact is applied."""

    @hookspec
    def on_manual_compact(
        self,
        messages: list["Message"],
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called after manual_compact is applied."""

    # ------------------------------------------------------------------
    # Planning hooks
    # ------------------------------------------------------------------

    @hookspec
    def on_plan_created(
        self,
        plan: "PlanArtifact",
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called after the leader builds a plan (pre-dispatch gate)."""

    @hookspec
    def on_workflow_start(
        self,
        workflow: "WorkflowDefinition",
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called before a workflow engine is invoked."""

    @hookspec
    def on_workflow_end(
        self,
        workflow: "WorkflowDefinition",
        result: "Result",
        report: "ExecutionReport",
    ) -> None:
        """Called after a workflow engine completes."""

    # ------------------------------------------------------------------
    # Subagent hooks
    # ------------------------------------------------------------------

    @hookspec
    def on_subagent_start(
        self,
        agent_id: str,
        envelope_payload: dict[str, Any],
        report: "ExecutionReport",
    ) -> None:
        """Called before a subagent is dispatched."""

    @hookspec
    def on_subagent_end(
        self,
        agent_id: str,
        result: "SubagentResult",
        report: "ExecutionReport",
    ) -> None:
        """Called after a subagent returns its structured output."""

    # ------------------------------------------------------------------
    # Destructive action confirmation
    # ------------------------------------------------------------------

    @hookspec(firstresult=True)
    def on_destructive_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: "ToolContext",
    ) -> bool | None:
        """Confirmation hook for destructive tool calls.

        Return True to allow, False to deny. None means no opinion.
        """

    # ------------------------------------------------------------------
    # Error / cancellation
    # ------------------------------------------------------------------

    @hookspec
    def on_error(
        self,
        error: Exception,
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called on any unhandled exception inside the loop."""

    @hookspec
    def on_cancelled(
        self,
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> None:
        """Called when a cancellation signal is received."""

    # ------------------------------------------------------------------
    # Response / output
    # ------------------------------------------------------------------

    @hookspec(firstresult=True)
    def on_response_generated(
        self,
        content: str,
        context: "AgentContext",
        report: "ExecutionReport",
    ) -> str | None:
        """Post-process final assistant response. Return modified string or None."""
