"""Plugin manager for Protocore hook system (pluggy-based)."""
from __future__ import annotations

import copy
import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

import pluggy

from ..constants import HOOKS_PROJECT_NAME
from ..orchestrator_errors import ContractViolationError
from ..orchestrator_utils import normalize_policy_decision
from ..types import PolicyDecision
from .specs import AgentHookSpecs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..types import (
        AgentContext,
        ExecutionReport,
        Message,
        PlanArtifact,
        Result,
        SubagentResult,
        ToolContext,
        ToolDefinition,
        ToolResult,
        WorkflowDefinition,
    )


class _TypedHookRelay(Protocol):
    def pre_llm_call(
        self, *, messages: list[Message], context: AgentContext, report: ExecutionReport
    ) -> None: ...
    def on_error(
        self, *, error: Exception, context: AgentContext, report: ExecutionReport
    ) -> None: ...
    def on_cancelled(self, *, context: AgentContext, report: ExecutionReport) -> None: ...
    def on_session_start(self, *, context: AgentContext, report: ExecutionReport) -> None: ...
    def on_session_end(self, *, context: AgentContext, report: ExecutionReport) -> None: ...
    def on_tool_post_execute(
        self, *, result: ToolResult, context: ToolContext, report: ExecutionReport
    ) -> None: ...
    def on_tool_registered(self, *, tool: ToolDefinition) -> None: ...
    def on_tool_pre_execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        report: ExecutionReport,
    ) -> PolicyDecision | None: ...
    def on_plan_created(
        self, *, plan: PlanArtifact, context: AgentContext, report: ExecutionReport
    ) -> None: ...
    def on_workflow_start(
        self,
        *,
        workflow: WorkflowDefinition,
        context: AgentContext,
        report: ExecutionReport,
    ) -> None: ...
    def on_workflow_end(
        self,
        *,
        workflow: WorkflowDefinition,
        result: Result,
        report: ExecutionReport,
    ) -> None: ...
    def on_subagent_start(
        self, *, agent_id: str, envelope_payload: dict[str, Any], report: ExecutionReport
    ) -> None: ...
    def on_subagent_end(
        self, *, agent_id: str, result: SubagentResult, report: ExecutionReport
    ) -> None: ...
    def on_micro_compact(
        self, *, messages: list[Message], context: AgentContext, report: ExecutionReport
    ) -> None: ...
    def on_auto_compact(
        self, *, messages: list[Message], context: AgentContext, report: ExecutionReport
    ) -> None: ...
    def on_manual_compact(
        self, *, messages: list[Message], context: AgentContext, report: ExecutionReport
    ) -> None: ...
    def on_destructive_action(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> bool | None: ...
    def on_response_generated(
        self, *, content: str, context: AgentContext, report: ExecutionReport
    ) -> str | None: ...


def create_plugin_manager() -> pluggy.PluginManager:
    """Create and return a configured PluginManager.

    Call ``pm.register(plugin_instance)`` to add service-level plugins.
    """
    pm = pluggy.PluginManager(HOOKS_PROJECT_NAME)
    pm.add_hookspecs(AgentHookSpecs)
    return pm


def _normalize_hook_decision(value: Any) -> PolicyDecision | None:
    """Normalize a plugin return value to PolicyDecision or None.

    Accepts PolicyDecision instances, string values (``"deny"``, ``"confirm"``,
    ``"allow"``), and returns ``None`` for unrecognised inputs.
    """
    return normalize_policy_decision(value)


class HookManager:
    """Thin wrapper over pluggy.PluginManager for ergonomic use in orchestrator."""

    def __init__(self, pm: pluggy.PluginManager | None = None) -> None:
        self._pm = pm or create_plugin_manager()

    @property
    def hook(self) -> pluggy.HookRelay:
        return self._pm.hook

    def _typed_hook(self) -> _TypedHookRelay:
        return cast(_TypedHookRelay, self._pm.hook)

    def register(self, plugin: object) -> None:
        """Register a plugin object implementing hookimpl methods."""
        self._pm.register(plugin)

    def unregister(self, plugin: object) -> None:
        """Remove a previously registered plugin."""
        self._pm.unregister(plugin)

    def get_plugins(self) -> list[object]:
        return list(self._pm.get_plugins())

    @staticmethod
    def _record_hook_failure(
        *,
        hook_name: str,
        report: "ExecutionReport | None",
        exc: Exception,
    ) -> None:
        logger.exception("HookManager.%s failed", hook_name)
        if report is not None:
            report.add_warning(f"hook_failed:{hook_name}:{type(exc).__name__}")

    def _safe_hook_call(
        self,
        *,
        hook_name: str,
        report: "ExecutionReport | None",
        callback: Any,
        default: Any = None,
    ) -> Any:
        try:
            return callback()
        except Exception as exc:
            # Hook failures are isolated from the main loop.
            self._record_hook_failure(hook_name=hook_name, report=report, exc=exc)
            return default

    def clone(self) -> "HookManager":
        """Return a best-effort isolated clone of the registered plugins."""
        return self.clone_with_mode(strict=False)

    def clone_with_mode(self, *, strict: bool) -> "HookManager":
        """Clone plugins for a child run."""
        cloned = HookManager()
        for plugin in self.get_plugins():
            plugin_copy = plugin
            clone_method = getattr(plugin, "clone", None)
            if callable(clone_method):
                plugin_copy = clone_method()
            else:
                try:
                    plugin_copy = copy.deepcopy(plugin)
                except Exception as exc:
                    if strict:
                        raise ContractViolationError(
                            "HOOK_PLUGIN_CLONE_REQUIRED",
                            "hook plugins must implement clone() or support deepcopy()",
                        ) from exc
                    plugin_copy = plugin
                else:
                    if plugin_copy is plugin and not inspect.isfunction(plugin):
                        if strict:
                            raise ContractViolationError(
                                "HOOK_PLUGIN_CLONE_REQUIRED",
                                "hook plugins must not be shared by reference across runs",
                            )
            self._link_shared_observer_state(plugin, plugin_copy)
            cloned.register(plugin_copy)
        return cloned

    @staticmethod
    def _link_shared_observer_state(original: object, cloned: object) -> None:
        """Share common append-only observer sinks across parent/child clones.

        Some telemetry plugins keep a public ``events`` list for later assertions.
        When such a plugin is deep-copied for a subagent, hook callbacks still
        fire but append into the clone-local list, making the parent observer
        look incomplete. Keep the plugin object isolated while reusing the
        caller-visible sink when both instances expose a list-like ``events``
        attribute.
        """
        if cloned is original:
            return
        original_events = getattr(original, "events", None)
        cloned_events = getattr(cloned, "events", None)
        if isinstance(original_events, list) and isinstance(cloned_events, list):
            setattr(cloned, "events", original_events)

    # ------------------------------------------------------------------
    # Convenience callers (avoid attribute-chain noise in orchestrator)
    # ------------------------------------------------------------------

    def call_pre_llm(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="pre_llm_call",
            report=report,
            callback=lambda: hook.pre_llm_call(
                messages=messages,
                context=context,
                report=report,
            ),
        )

    def call_on_error(
        self, *, error: Exception, context: AgentContext, report: ExecutionReport
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_error",
            report=report,
            callback=lambda: hook.on_error(error=error, context=context, report=report),
        )

    def call_on_cancelled(self, *, context: AgentContext, report: ExecutionReport) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_cancelled",
            report=report,
            callback=lambda: hook.on_cancelled(context=context, report=report),
        )

    def call_session_start(self, *, context: AgentContext, report: ExecutionReport) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_session_start",
            report=report,
            callback=lambda: hook.on_session_start(context=context, report=report),
        )

    def call_session_end(self, *, context: AgentContext, report: ExecutionReport) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_session_end",
            report=report,
            callback=lambda: hook.on_session_end(context=context, report=report),
        )

    def call_tool_post_execute(
        self, *, result: ToolResult, context: ToolContext, report: ExecutionReport
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_tool_post_execute",
            report=report,
            callback=lambda: hook.on_tool_post_execute(
                result=result,
                context=context,
                report=report,
            ),
        )

    def call_tool_registered(self, *, tool: ToolDefinition) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_tool_registered",
            report=None,
            callback=lambda: hook.on_tool_registered(tool=tool),
        )

    def call_tool_pre_execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        report: ExecutionReport,
    ) -> PolicyDecision | None:
        hook = self._typed_hook()
        result: Any = self._safe_hook_call(
            hook_name="on_tool_pre_execute",
            report=report,
            callback=lambda: hook.on_tool_pre_execute(
                tool_name=tool_name,
                arguments=arguments,
                context=context,
                report=report,
            ),
            default=None,
        )
        return _normalize_hook_decision(result)

    def call_plan_created(
        self, *, plan: PlanArtifact, context: AgentContext, report: ExecutionReport
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_plan_created",
            report=report,
            callback=lambda: hook.on_plan_created(
                plan=plan,
                context=context,
                report=report,
            ),
        )

    def call_workflow_start(
        self,
        *,
        workflow: WorkflowDefinition,
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_workflow_start",
            report=report,
            callback=lambda: hook.on_workflow_start(
                workflow=workflow,
                context=context,
                report=report,
            ),
        )

    def call_workflow_end(
        self,
        *,
        workflow: WorkflowDefinition,
        result: Result,
        report: ExecutionReport,
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_workflow_end",
            report=report,
            callback=lambda: hook.on_workflow_end(
                workflow=workflow,
                result=result,
                report=report,
            ),
        )

    def call_subagent_start(
        self, *, agent_id: str, envelope_payload: dict[str, Any], report: ExecutionReport
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_subagent_start",
            report=report,
            callback=lambda: hook.on_subagent_start(
                agent_id=agent_id,
                envelope_payload=envelope_payload,
                report=report,
            ),
        )

    def call_subagent_end(
        self, *, agent_id: str, result: SubagentResult, report: ExecutionReport
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_subagent_end",
            report=report,
            callback=lambda: hook.on_subagent_end(
                agent_id=agent_id,
                result=result,
                report=report,
            ),
        )

    def call_micro_compact(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_micro_compact",
            report=report,
            callback=lambda: hook.on_micro_compact(
                messages=messages,
                context=context,
                report=report,
            ),
        )

    def call_auto_compact(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_auto_compact",
            report=report,
            callback=lambda: hook.on_auto_compact(
                messages=messages,
                context=context,
                report=report,
            ),
        )

    def call_manual_compact(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        hook = self._typed_hook()
        self._safe_hook_call(
            hook_name="on_manual_compact",
            report=report,
            callback=lambda: hook.on_manual_compact(
                messages=messages,
                context=context,
                report=report,
            ),
        )

    def call_destructive_action(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> bool | None:
        hook = self._typed_hook()
        results: Any = self._safe_hook_call(
            hook_name="on_destructive_action",
            report=None,
            callback=lambda: hook.on_destructive_action(
                tool_name=tool_name,
                arguments=arguments,
                context=context,
            ),
            default=None,
        )
        if results is None:
            return None
        if isinstance(results, list):
            for item in results:
                if item is not None:
                    return cast(bool, item)
            return None
        return cast(bool, results)

    def call_response_generated(
        self, *, content: str, context: AgentContext, report: ExecutionReport
    ) -> str | None:
        hook = self._typed_hook()
        result: Any = self._safe_hook_call(
            hook_name="on_response_generated",
            report=report,
            callback=lambda: hook.on_response_generated(
                content=content,
                context=context,
                report=report,
            ),
            default=None,
        )
        if result is None:
            return None
        if isinstance(result, list):
            for r in result:
                if r is not None:
                    return cast(str, r)
            return None
        return cast(str, result)
