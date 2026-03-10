"""Immutable Agent Loop Orchestrator.

The loop structure NEVER changes. All extensions happen through:
- HookManager (pluggy hooks: pre_llm_call, on_tool_post_execute, …)
- EventBus (CoreEvent emissions for telemetry)
- Injected strategies (LLMClient, ToolExecutor, PlanningStrategy, …)

Anti-pattern: branching on tool_name or condition INSIDE the loop.
""" 
from __future__ import annotations

import asyncio
from functools import partial
import inspect
import json
import logging
import re
import time
import warnings
from typing import Any, Awaitable, Callable, cast

from .compression import micro_compact
from .context import (
    CancellationContext,
    estimate_llm_prompt_tokens,
    estimate_tokens,
)
from .events import (
    EV_CANCELLATION,
    EV_COMPACTION_AUTO_START,
    EV_COMPACTION_CHECK,
    EV_COMPACTION_END,
    EV_COMPACTION_MANUAL_START,
    EV_COMPRESSION_AUTO,
    EV_COMPRESSION_MANUAL,
    EV_COMPRESSION_MICRO,
    EV_ERROR,
    EV_INJECTION_SIGNAL,
    EV_LLM_CALL_END,
    EV_LLM_CALL_FAILED,
    EV_LLM_CALL_START,
    EV_LLM_OUTPUT_EMPTY,
    EV_LLM_OUTPUT_PARSED,
    EV_LLM_REQUEST_PREPARED,
    EV_LLM_STREAM_COMPLETED,
    EV_LLM_STREAM_DELTA,
    EV_LOOP_ITERATION_END,
    EV_LOOP_ITERATION_START,
    EV_PARALLEL_RUN_END,
    EV_PARALLEL_RUN_START,
    EV_PLANNING_FAILED,
    EV_PLANNING_INPUT_PREPARED,
    EV_PLANNING_END,
    EV_PLANNING_START,
    EV_SESSION_END,
    EV_SESSION_START,
    EV_SKILL_INDEX_INJECTED,
    EV_SUBAGENT_DISPATCH_START,
    EV_SUBAGENT_END,
    EV_SUBAGENT_RESULT_FALLBACK,
    EV_SUBAGENT_RESULT_PARSE_END,
    EV_SUBAGENT_RESULT_PARSE_START,
    EV_SUBAGENT_RESULT_RAW,
    EV_SUBAGENT_SELECTION_END,
    EV_SUBAGENT_SELECTION_START,
    EV_SUBAGENT_START,
    EV_TOOL_CALL_DETECTED,
    EV_TOOL_BUDGET_EXCEEDED,
    EV_TOOL_RESULT_APPENDED,
    EV_WORKFLOW_END,
    EV_WORKFLOW_START,
    EventBus,
)
from .constants import (
    DEFAULT_STATE_MANAGER_TIMEOUT_SECONDS,
    FORCED_FINALIZATION_MSG,
    LOAD_SKILL_TOOL_NAME,
    MANUAL_COMPACT_TOOL_NAME,
    MAX_PARALLEL_AGENT_IDS,
)
from .factories import (
    make_control_envelope,
    make_execution_report,
    make_task_envelope,
    register_load_skill_tool,
)
from .hooks.manager import HookManager
from .logging_utils import context_log_extra
from .orchestrator_errors import ContractViolationError, PendingShellApprovalError
from .orchestrator_state import (
    accumulate_llm_usage,
    build_session_snapshot,
    build_subagent_context,
    ensure_active_child_agent_ids,
    get_active_child_agent_ids,
    set_active_child_agent_ids,
    set_queue_wait_metric,
)
from .orchestrator_utils import (
    PolicyRunner,
    append_tool_results_as_messages,
    build_llm_request_kwargs,
    build_subagent_summary,
    default_stop_reason_for_status,
    ensure_terminal_report,
    event_context_payload,
    execution_status_from_subagent_status,
    extract_task,
    load_existing_plan,
    merge_nested_dict,
    merge_execution_report,
    recover_tool_calls_from_assistant_text,
    redact_sensitive_keys,
    resolve_effective_llm_config,
    resolve_max_tokens,
    resolve_shell_capability,
    session_refs,
    string_preview,
    subagent_result_used_fallback,
    tool_requests_manual_compact,
    validate_parallel_policy,
)
from .parallel import ParallelSubagentRunner, run_bypass
from .protocols import (
    CapabilityBasedSelectionPolicy,
    ExecutionPolicy,
    CompressionStrategy,
    LLMClient,
    ParallelExecutionPolicy,
    PlanningPolicy,
    PlanningStrategy,
    RetryPolicy,
    ShellExecutor,
    ShellSafetyPolicy,
    SkillManager,
    StateManager,
    SubagentSelectionPolicy,
    TelemetryCollector,
    TimeoutPolicy,
    ToolExecutor,
    Transport,
    WorkflowEngine,
 )
from .registry import AgentRegistry, ToolRegistry
from .shell_handler import ShellHandler
from .shell_safety import DefaultShellSafetyPolicy
from .tool_dispatch import ToolDispatcher
from .types import (
    AgentConfig,
    AgentContext,
    AgentContextMeta,
    CompressionEvent,
    ControlCommand,
    ExecutionMode,
    ExecutionReport,
    ExecutionStatus,
    extract_structured_usage,
    Message,
    PolicyDecision,
    Result,
    RunKind,
    ShellCommandPlan,
    ShellExecutionRequest,
    ShellToolConfig,
    SkillIndexEntry,
    SessionSnapshot,
    StopReason,
    SubagentResult,
    ToolCallRecord,
    ToolCall,
    ToolContextMeta,
    ToolDefinition,
    ToolResult,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)
_STATE_MANAGER_TIMEOUT = object()
_STATE_MANAGER_INFO_LOGGED = False
_PUBLIC_ERROR_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|credential|bearer)\b\s*[:=]\s*([^\s,;]+)"
)


# ---------------------------------------------------------------------------
# AgentOrchestrator — single-agent immutable loop
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Runs the immutable loop for a single agent (leader or subagent).
    
    Inject dependencies via constructor. No singletons, no global state.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_executor: ToolExecutor | None = None,
        tool_registry: ToolRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
        hook_manager: HookManager | None = None,
        event_bus: EventBus | None = None,
        planning_strategy: PlanningStrategy | None = None,
        planning_policy: PlanningPolicy | None = None,
        parallel_execution_policy: ParallelExecutionPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
        shell_executor: ShellExecutor | None = None,
        shell_safety_policy: ShellSafetyPolicy | None = None,
        skill_manager: SkillManager | None = None,
        compressor: CompressionStrategy | None = None,
        state_manager: StateManager | None = None,
        subagent_selection_policy: SubagentSelectionPolicy | None = None,
        selection_policy: SubagentSelectionPolicy | None = None,
        suppress_state_manager_warning: bool = True,
        transport: Transport | None = None,
        workflow_engine: WorkflowEngine | None = None,
        telemetry_collector: TelemetryCollector | None = None,
        timeout_policy: TimeoutPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._llm = llm_client
        self._tool_executor = tool_executor
        if hook_manager is not None and not isinstance(hook_manager, HookManager):
            raise ContractViolationError(
                "HOOK_MANAGER_INVALID",
                (
                    "hook_manager must be HookManager. "
                    "create_plugin_manager() returns low-level pluggy.PluginManager; "
                    "use HookManager() or HookManager(pm=...)."
                ),
            )
        self._hooks = hook_manager if hook_manager is not None else HookManager()
        self._tool_registry = (
            tool_registry
            if tool_registry is not None
            else ToolRegistry(hook_manager=self._hooks)
        )
        self._agent_registry = (
            agent_registry if agent_registry is not None else AgentRegistry()
        )
        self._bus = event_bus if event_bus is not None else EventBus()
        self._planning = planning_strategy
        self._planning_policy = planning_policy
        self._parallel_policy = parallel_execution_policy
        self._policy = execution_policy
        self._shell_executor = shell_executor
        self._shell_safety_policy = (
            shell_safety_policy
            if shell_safety_policy is not None
            else (DefaultShellSafetyPolicy() if shell_executor is not None else None)
        )
        self._skill_manager = skill_manager
        self._compressor = compressor
        self._state_manager = state_manager
        if (
            subagent_selection_policy is not None
            and selection_policy is not None
            and subagent_selection_policy is not selection_policy
        ):
            raise TypeError(
                "Provide only one of subagent_selection_policy or selection_policy"
            )
        if selection_policy is not None:
            warnings.warn(
                "'selection_policy' is deprecated; use 'subagent_selection_policy'",
                DeprecationWarning,
                stacklevel=2,
            )
        self._selection_policy = (
            subagent_selection_policy
            if subagent_selection_policy is not None
            else selection_policy
        )
        global _STATE_MANAGER_INFO_LOGGED
        if (
            self._state_manager is None
            and not suppress_state_manager_warning
            and not _STATE_MANAGER_INFO_LOGGED
        ):
            logger.info("No StateManager provided; running in stateless mode")
            _STATE_MANAGER_INFO_LOGGED = True
        self._transport = transport
        self._workflow_engine = workflow_engine
        self._telemetry = telemetry_collector
        self._timeout_policy = timeout_policy
        self._retry_policy = retry_policy
        self._policy_runner = PolicyRunner(
            timeout_policy=timeout_policy,
            retry_policy=retry_policy,
        )
        self._shell_handler = ShellHandler(
            shell_executor=self._shell_executor,
            policy_runner=self._policy_runner,
            shell_safety_policy=self._shell_safety_policy,
            append_tool_results_as_messages=append_tool_results_as_messages,
        )
        self._tool_dispatcher = ToolDispatcher(
            hooks=self._hooks,
            event_bus=self._bus,
            policy=self._policy,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
            shell_executor=self._shell_executor,
            shell_safety_policy=self._shell_safety_policy,
            shell_handler=self._shell_handler,
            policy_runner=self._policy_runner,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def event_bus(self) -> EventBus:
        """Event bus for this orchestrator. Use e.g. ``event_bus.handler_count("*")`` for load balancing."""
        return self._bus

    def set_subagent_selection_policy(
        self,
        policy: SubagentSelectionPolicy | None,
    ) -> None:
        """Set or replace subagent selection policy at runtime."""
        self._selection_policy = policy

    async def _call_state_manager_with_timeout(
        self,
        *,
        operation: str,
        warning_code: str,
        report: ExecutionReport,
        awaitable: Awaitable[Any],
    ) -> Any:
        """Await a StateManager operation with a fixed timeout.

        Timeouts are downgraded to warnings so persistence failures do not block
        the orchestrator shutdown path.
        """
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=DEFAULT_STATE_MANAGER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # Degrade gracefully when persistence storage stalls.
            report.state_manager_timeout_count += 1
            report.add_warning(warning_code)
            logger.warning(
                "StateManager operation timed out: operation=%s timeout_seconds=%.1f",
                operation,
                DEFAULT_STATE_MANAGER_TIMEOUT_SECONDS,
            )
            return _STATE_MANAGER_TIMEOUT

    async def run(
        self,
        context: AgentContext,
        *,
        run_kind: RunKind = RunKind.LEADER,
        cancel_ctx: CancellationContext | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[Result, ExecutionReport]:
        """Execute the immutable loop. Always returns (Result, ExecutionReport).
        
        Planning gate is enforced here for LEADER mode.
        AUTO_SELECT mode delegates to SubagentSelectionPolicy.
        StateManager is called for session persistence if injected.
        Hook lifecycle by mode:
        - ``BYPASS``/``LEADER``/``PARALLEL``: this run executes LLM turns and
          fires ``pre_llm_call`` in ``_immutable_loop``.
        - ``AUTO_SELECT``: leader run only selects and delegates, so leader-level
          ``pre_llm_call`` does not fire; delegated subagent run fires it normally.
        """
        report = make_execution_report(context=context, run_kind=run_kind)
        prepared_config = await self._prepare_config_with_skills(
            context=context,
            report=report,
        )
        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise TypeError("extra_body must be a dict[str, Any] when provided")
            # Allow per-run extra_body overrides.
            prepared_config = prepared_config.model_copy(
                update={
                    "llm_extra_body": merge_nested_dict(
                        dict(prepared_config.llm_extra_body),
                        extra_body,
                    )
                },
                deep=True,
            )
        if prepared_config is not context.config:
            context = context.model_copy(update={"config": prepared_config}, deep=True)
        self._set_queue_wait_metric(report=report, context=context)
        cancel_ctx = cancel_ctx or CancellationContext()
        ensure_active_child_agent_ids(context.metadata)
        context.tool_context.metadata[ToolContextMeta.RUN_KIND] = run_kind.value
        context.tool_context.metadata[ToolContextMeta.REQUEST_ID] = context.request_id
        background_tasks: set[asyncio.Task[None]] = set()

        def _track_background_task(task: asyncio.Task[None], *, warning_code: str) -> None:
            background_tasks.add(task)

            def _on_done(done_task: asyncio.Task[None]) -> None:
                background_tasks.discard(done_task)
                try:
                    done_task.result()
                except Exception as exc:
                    logger.exception(
                        "%s",
                        warning_code,
                        extra=context_log_extra(context, report=report),
                    )
                    report.warnings.append(f"{warning_code}:{type(exc).__name__}")

            task.add_done_callback(_on_done)

        def _error_sink(event: Any, exc: Exception) -> None:
            _ = exc
            report.add_warning(f"event_handler_failed:{event.name}:{type(exc).__name__}")

        request_id = context.request_id
        def event_filter(event: Any) -> bool:
            return bool(event.payload.get("request_id") == request_id)
        error_sink_token = self._bus.push_error_sink(
            _error_sink,
            event_filter=event_filter,
        )
        telemetry_subscription: Any | None = None
        if self._telemetry is not None:
            collector = self._telemetry

            async def _emit_telemetry(event: Any) -> None:
                await collector.record_event(event.name, event.payload, report)

            async def _telemetry_handler(event: Any) -> None:
                # Telemetry should not block the main orchestration loop.
                _track_background_task(
                    asyncio.create_task(_emit_telemetry(event)),
                    warning_code="telemetry_handler_failed",
                )

            telemetry_subscription = self._bus.subscribe(
                "*",
                _telemetry_handler,
                event_filter=event_filter,
            )

        try:
            if not context.messages:
                raise ContractViolationError(
                    "EMPTY_MESSAGES",
                    "run requires at least one message in context.messages",
                )
            await self._bus.emit_simple(
                EV_SESSION_START,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="subagent" if run_kind == RunKind.SUBAGENT else "main_turn",
                    scope_id=f"session:{context.request_id}",
                    source_component="orchestrator.run",
                ),
                model=context.config.model,
                api_mode=context.config.api_mode.value,
                stream=context.config.stream,
                max_iterations=context.config.max_iterations,
                max_tool_calls=context.config.max_tool_calls,
                stream_reasoning_enabled=context.config.emit_reasoning_in_stream,
            )
            try:
                self._hooks.call_session_start(context=context, report=report)
            except Exception as exc:
                logger.exception(
                    "session_start hook failed: agent=%s",
                    context.config.agent_id,
                    extra=context_log_extra(context, report=report),
                )
                report.warnings.append(
                    f"session_start_hook_failed:{type(exc).__name__}"
                )

            # --- Save initial session snapshot ---
            if self._state_manager:
                try:
                    message_history_ref, execution_metadata_ref = session_refs(context)
                    snapshot = SessionSnapshot(
                        session_id=context.session_id,
                        trace_id=context.trace_id,
                        agent_id=context.config.agent_id,
                        message_history_ref=message_history_ref,
                        execution_metadata_ref=execution_metadata_ref,
                        messages=list(context.messages),
                    )
                    await self._call_state_manager_with_timeout(
                        operation="save_session_snapshot",
                        warning_code="session_snapshot_save_timed_out",
                        report=report,
                        awaitable=self._state_manager.save_session_snapshot(snapshot),
                    )
                except Exception:
                    logger.exception(
                        "Failed to save initial session snapshot",
                        extra=context_log_extra(context, report=report),
                    )
                    report.warnings.append("session_snapshot_save_failed")

            mode = context.config.execution_mode
            await self._enforce_mode_policy(context)
            should_run_planning_gate = await self._should_run_planning_gate(context)
            workflow_definition = context.metadata.get(AgentContextMeta.WORKFLOW_DEFINITION)

            if workflow_definition is not None:
                if should_run_planning_gate:
                    await self._planning_gate(context, report, cancel_ctx)
                workflow = (
                    workflow_definition
                    if isinstance(workflow_definition, WorkflowDefinition)
                    else WorkflowDefinition.model_validate(workflow_definition)
                )
                result, workflow_report = await self.run_workflow(
                    workflow,
                    context,
                    run_kind=run_kind,
                    report=report,
                )
                report = workflow_report
            elif mode == ExecutionMode.AUTO_SELECT:
                result = await self._run_auto_select_mode(context, report, cancel_ctx)
            elif mode == ExecutionMode.PARALLEL:
                result = await self._run_parallel_mode(context, report, cancel_ctx)
            else:
                if should_run_planning_gate:
                    await self._planning_gate(context, report, cancel_ctx)
                result = await self._immutable_loop(context, report, cancel_ctx, run_kind)

        except asyncio.CancelledError:
            result = await self._handle_cancellation(context, report, cancel_ctx)

        except Exception as exc:
            result = await self._handle_error(exc, context, report)

        finally:
            self._sync_report_metadata(report=report, context=context)
            await self._bus.emit_simple(
                EV_SESSION_END,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="finalization",
                    scope_id=f"session:{context.request_id}",
                    source_component="orchestrator.run",
                ),
                status=report.status.value,
                stop_reason=report.stop_reason.value if report.stop_reason else None,
                duration_ms=report.duration_ms,
                error_code=report.error_code,
                error_message=report.error_message,
                warnings=list(report.warnings),
            )
            try:
                self._hooks.call_session_end(context=context, report=report)
            except Exception as exc:
                logger.exception(
                    "session_end hook failed: agent=%s",
                    context.config.agent_id,
                    extra=context_log_extra(context, report=report),
                )
                report.warnings.append(f"session_end_hook_failed:{type(exc).__name__}")

            # --- Save execution report ---
            if self._state_manager:
                try:
                    await self._update_session_snapshot(context=context, report=report)
                except Exception:
                    logger.exception(
                        "Failed to update session snapshot",
                        extra=context_log_extra(context, report=report),
                    )
                    report.warnings.append("session_snapshot_update_failed")
                try:
                    await self._call_state_manager_with_timeout(
                        operation="save_execution_report",
                        warning_code="execution_report_save_timed_out",
                        report=report,
                        awaitable=self._state_manager.save_execution_report(report),
                    )
                except Exception:
                    logger.exception(
                        "Failed to save execution report",
                        extra=context_log_extra(context, report=report),
                    )
                    report.warnings.append("execution_report_save_failed")
            if telemetry_subscription is not None:
                self._bus.unsubscribe("*", telemetry_subscription)
            if background_tasks:
                done, pending = await asyncio.wait(background_tasks, timeout=5.0)
                for task in pending:
                    task.cancel()
                if pending:
                    report.warnings.append(
                        f"background_tasks_pending:{len(pending)}"
                    )
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            self._bus.pop_error_sink(error_sink_token)

        return result, report

    # ------------------------------------------------------------------
    # AUTO_SELECT handler
    # ------------------------------------------------------------------

    @staticmethod
    def _require_dispatch_task(
        messages: list[Message],
        *,
        purpose: str,
    ) -> str:
        task_text = extract_task(messages, strategy="all")
        if task_text is None or not task_text.strip():
            raise ContractViolationError(
                "EMPTY_TASK",
                f"{purpose} requires a non-empty user task",
            )
        return task_text

    async def _run_auto_select_mode(
        self,
        context: AgentContext,
        report: ExecutionReport,
        cancel_ctx: CancellationContext,
    ) -> Result:
        """Fast-path auto-select: choose a subagent and dispatch it immediately.

        The leader does not call the LLM in this mode, so leader-level
        ``pre_llm_call`` is intentionally not emitted. Hook callbacks for
        subagent lifecycle still fire, and the delegated subagent run emits
        ``pre_llm_call`` when it enters ``_immutable_loop``.
        """
        task_text = self._require_dispatch_task(
            context.messages,
            purpose="AUTO_SELECT dispatch",
        )
        selection_policy = (
            self._selection_policy
            if self._selection_policy is not None
            else self._make_default_selection_policy()
        )
        available = [cfg.agent_id for cfg in self._agent_registry.list_subagents()]
        await self._bus.emit_simple(
            EV_SUBAGENT_SELECTION_START,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="subagent",
                scope_id=f"subagent-selection:{context.request_id}",
                source_component="orchestrator.auto_select",
            ),
            candidate_agent_ids=available,
            policy_name=type(selection_policy).__name__,
        )
        selected = await selection_policy.select(task_text, available, context)
        selection_debug = getattr(selection_policy, "last_selection_debug", None)
        await self._bus.emit_simple(
            EV_SUBAGENT_SELECTION_END,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="subagent",
                scope_id=f"subagent-selection:{context.request_id}",
                source_component="orchestrator.auto_select",
            ),
            selected_agent_ids=[selected],
            selection_reason=(
                selection_debug.get("reason", "policy.select")
                if isinstance(selection_debug, dict)
                else "policy.select"
            ),
        )
        report.add_artifact(f"auto_selected_agent:{selected}")
        context.metadata[AgentContextMeta.AUTO_SELECTED_AGENT] = selected
        report.metadata[AgentContextMeta.AUTO_SELECTED_AGENT] = selected
        if isinstance(selection_debug, dict):
            report.metadata["auto_select"] = dict(selection_debug)
        return await self._run_child_subagent(
            parent_context=context,
            report=report,
            cancel_ctx=cancel_ctx,
            agent_id=selected,
            task_text=task_text,
        )

    async def _run_parallel_mode(
        self,
        context: AgentContext,
        report: ExecutionReport,
        cancel_ctx: CancellationContext,
    ) -> Result:
        """Dispatch multiple subagents concurrently using the configured policy."""
        if self._parallel_policy is None:
            raise ContractViolationError(
                "PARALLEL_POLICY_REQUIRED",
                "PARALLEL mode requires ParallelExecutionPolicy",
            )
        validate_parallel_policy(self._parallel_policy)

        raw_agent_ids = context.metadata.get(AgentContextMeta.PARALLEL_AGENT_IDS)
        if not isinstance(raw_agent_ids, list) or not raw_agent_ids:
            raise ContractViolationError(
                "PARALLEL_AGENT_IDS_REQUIRED",
                "PARALLEL mode requires context.metadata['parallel_agent_ids']",
            )
        if not all(isinstance(agent_id, str) for agent_id in raw_agent_ids):
            raise ContractViolationError(
                "PARALLEL_AGENT_IDS_INVALID",
                "parallel_agent_ids must contain only strings",
            )
        if len(raw_agent_ids) != len(set(raw_agent_ids)):
            raise ContractViolationError(
                "PARALLEL_AGENT_IDS_DUPLICATE",
                "parallel_agent_ids must not contain duplicates",
            )
        if len(raw_agent_ids) > MAX_PARALLEL_AGENT_IDS:
            raise ContractViolationError(
                "PARALLEL_AGENT_IDS_TOO_MANY",
                (
                    f"parallel_agent_ids has {len(raw_agent_ids)} entries, "
                    f"max is {MAX_PARALLEL_AGENT_IDS}"
                ),
            )

        tasks: list[tuple[str, AgentContext]] = []
        task_text = self._require_dispatch_task(
            context.messages,
            purpose="PARALLEL dispatch",
        )
        for agent_id in sorted(raw_agent_ids):
            tasks.append((agent_id, self._build_subagent_context(context, agent_id, task_text)))

        runner = ParallelSubagentRunner(
            policy=self._parallel_policy,
            orchestrator_factory=self._spawn_child_orchestrator,
            transport=self._transport,
            parent_context=context,
            event_bus=self._bus,
        )
        await self._bus.emit_simple(
            EV_PARALLEL_RUN_START,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="subagent",
                scope_id=f"parallel:{context.request_id}",
                source_component="orchestrator.parallel",
            ),
            agent_ids=[agent_id for agent_id, _ in tasks],
            max_concurrency=self._parallel_policy.max_concurrency,
            timeout_seconds=self._parallel_policy.timeout_seconds,
            cancellation_mode=self._parallel_policy.cancellation_mode,
        )
        set_active_child_agent_ids(
            context.metadata,
            [agent_id for agent_id, _ in tasks],
        )
        try:
            merged, summaries = await runner.run_parallel(
                tasks,
                cancel_ctx=cancel_ctx,
                report=report,
            )
        finally:
            set_active_child_agent_ids(context.metadata, [])
        report.subagent_runs = summaries
        subagent_input_tokens = sum(max(s.input_tokens, 0) for s in summaries)
        subagent_output_tokens = sum(max(s.output_tokens, 0) for s in summaries)
        report.input_tokens += subagent_input_tokens
        report.output_tokens += subagent_output_tokens
        report.metadata["subagent_input_tokens"] = subagent_input_tokens
        report.metadata["subagent_output_tokens"] = subagent_output_tokens
        report.metadata["child_tokens_sum_input"] = subagent_input_tokens
        report.metadata["child_tokens_sum_output"] = subagent_output_tokens
        report.metadata["parent_tokens_include_subagents"] = True
        pending_subagent_approvals_raw = report.metadata.get("pending_shell_approvals")
        pending_subagent_approvals = (
            pending_subagent_approvals_raw
            if isinstance(pending_subagent_approvals_raw, list)
            else []
        )
        pending_map: dict[str, dict[str, Any]] = {}
        first_pending_plan: dict[str, Any] | None = None
        for item in pending_subagent_approvals:
            if not isinstance(item, dict):
                continue
            agent_id = item.get("agent_id")
            plan = item.get("plan")
            if isinstance(agent_id, str) and isinstance(plan, dict):
                pending_map[agent_id] = plan
                if first_pending_plan is None:
                    first_pending_plan = plan
        if pending_map:
            context.metadata["subagent_pending_shell_approvals"] = pending_map
            context.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = first_pending_plan
        else:
            context.metadata.pop("subagent_pending_shell_approvals", None)
            context.metadata.pop(AgentContextMeta.PENDING_SHELL_APPROVAL, None)
        for artifact in merged.artifacts:
            if artifact not in report.artifacts:
                report.add_artifact(artifact)
        for file_path in merged.files_changed:
            if file_path not in report.files_changed:
                report.add_file_changed(file_path)
        if cancel_ctx.is_cancelled:
            report.finalize(
                ExecutionStatus.CANCELLED,
                stop_reason=StopReason.CANCELLED,
                error_code="CANCELLED",
                error_message=cancel_ctx.reason,
            )
            return Result(
                content=merged.summary or "[cancelled]",
                status=ExecutionStatus.CANCELLED,
                artifacts=merged.artifacts,
            )
        status = execution_status_from_subagent_status(merged.status)
        report.finalize(
            status,
            stop_reason=default_stop_reason_for_status(status),
            error_message=merged.errors[0] if status in {ExecutionStatus.PARTIAL, ExecutionStatus.FAILED} and merged.errors else None,
        )
        result_metadata: dict[str, Any] = {}
        if first_pending_plan is not None:
            result_metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = first_pending_plan
            result_metadata["pending_shell_approvals"] = pending_subagent_approvals
        await self._bus.emit_simple(
            EV_PARALLEL_RUN_END,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="synthesis",
                scope_id=f"parallel:{context.request_id}",
                source_component="orchestrator.parallel",
            ),
            status=report.status.value,
            duration_ms=report.duration_ms,
            subagent_count=len(tasks),
        )
        return Result(
            content=merged.summary,
            status=status,
            artifacts=merged.artifacts,
            metadata=result_metadata,
        )

    # ------------------------------------------------------------------
    # Planning gate (leader mode)
    # ------------------------------------------------------------------

    async def _planning_gate(
        self,
        context: AgentContext,
        report: ExecutionReport,
        cancel_ctx: CancellationContext,
    ) -> None:
        """Build/update plan before dispatch — mandatory for LEADER mode."""
        if self._planning is None:
            raise ContractViolationError(
                "PLANNING_REQUIRED",
                (
                    "Leader mode requires PlanningStrategy before execution; "
                    f"agent={context.config.agent_id}. "
                    "For simple single-agent chat use execution_mode=ExecutionMode.BYPASS "
                    "or pass planning_strategy=NoOpPlanningStrategy()."
                ),
            )

        cancel_ctx.check()
        task_text = self._require_dispatch_task(
            context.messages,
            purpose="Planning",
        )
        existing_plan = load_existing_plan(context)
        await self._bus.emit_simple(
            EV_PLANNING_START,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="planning",
                scope_id=f"plan:{context.request_id}",
                source_component="orchestrator.planning",
            ),
            has_existing_plan=existing_plan is not None,
        )
        await self._bus.emit_simple(
            EV_PLANNING_INPUT_PREPARED,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="planning",
                scope_id=f"plan:{context.request_id}",
                source_component="orchestrator.planning",
            ),
            task_preview=string_preview(task_text, limit=600),
            message_count=len(context.messages),
        )
        try:
            if existing_plan is None:
                plan = await self._planning.build_plan(task_text, context, self._llm)
            else:
                plan = await self._planning.update_plan(existing_plan, context, self._llm)
            report.plan_created = True
            report.plan_id = plan.plan_id
            report.plan_artifact = plan
            context.metadata[AgentContextMeta.PLAN_ARTIFACT] = plan.model_dump()
            report.add_artifact(f"plan:{plan.plan_id}")
            self._hooks.call_plan_created(plan=plan, context=context, report=report)
        except Exception as exc:
            await self._bus.emit_simple(
                EV_PLANNING_FAILED,
                **event_context_payload(
                    context,
                    run_kind=RunKind.LEADER,
                    phase="planning",
                    scope_id=f"plan:{context.request_id}",
                    source_component="orchestrator.planning",
                ),
                error_code="PLANNING_FAILED",
                error_message=str(exc),
            )
            raise ContractViolationError(
                "PLANNING_FAILED",
                f"Planning strategy failed for agent={context.config.agent_id}: {exc}",
            ) from exc

        raw_plan = report.plan_artifact.raw_plan if report.plan_artifact is not None else None
        plan_step_count = len(
            [line for line in (raw_plan or "").splitlines() if line.strip()]
        )
        steps_payload: list[dict[str, Any]] = []
        if report.plan_artifact is not None and report.plan_artifact.steps:
            steps_payload = [
                s.model_dump(mode="json") for s in report.plan_artifact.steps
            ]
        await self._bus.emit_simple(
            EV_PLANNING_END,
            **event_context_payload(
                context,
                run_kind=RunKind.LEADER,
                phase="planning",
                scope_id=f"plan:{context.request_id}",
                source_component="orchestrator.planning",
            ),
            plan_id=report.plan_id,
            plan_created=report.plan_created,
            plan_step_count=plan_step_count,
            raw_plan_preview=string_preview(raw_plan or "", limit=1200),
            raw_plan=raw_plan,
            steps=steps_payload,
        )

    async def _enforce_mode_policy(self, context: AgentContext) -> None:
        """Validate policy-gated execution invariants before dispatch."""
        if self._planning_policy is None:
            return

        mode = context.config.execution_mode
        is_top_level_bypass = mode == ExecutionMode.BYPASS and context.parent_agent_id is None
        if is_top_level_bypass:
            if not await self._planning_policy.allow_bypass(context):
                raise ContractViolationError(
                    "BYPASS_NOT_ALLOWED",
                    (
                        "Explicit bypass mode was requested but denied by PlanningPolicy; "
                        f"agent={context.config.agent_id}"
                    ),
                )
            return

    async def _should_run_planning_gate(self, context: AgentContext) -> bool:
        """Return True when LEADER execution must pass through planning gate."""
        if context.config.execution_mode != ExecutionMode.LEADER:
            return False
        if self._planning_policy is None:
            return True
        return await self._planning_policy.should_plan(context)

    def _build_runtime_tool_definitions(
        self,
        config: AgentConfig,
        run_kind: RunKind,
    ) -> tuple[list[ToolDefinition], int]:
        runtime_tools = list(config.tool_definitions)
        if self._tool_registry is not None and not runtime_tools:
            configured_names = {tool.name for tool in runtime_tools}
            for definition in self._tool_registry.list_definitions():
                if definition.name not in configured_names:
                    runtime_tools.append(definition.model_copy(deep=True))
                    configured_names.add(definition.name)
        capability = resolve_shell_capability(config, run_kind)
        if capability is not None and self._shell_executor is not None:
            if not any(tool.name == capability.tool_name for tool in runtime_tools):
                runtime_tools.append(capability.to_tool_definition())

        total_candidate_tools = len(runtime_tools)
        if (
            config.max_visible_tools is not None
            and total_candidate_tools > config.max_visible_tools
        ):
            core_tool_names = {
                MANUAL_COMPACT_TOOL_NAME,
                LOAD_SKILL_TOOL_NAME,
            }
            if capability is not None:
                core_tool_names.add(capability.tool_name)
            core_tools = [tool for tool in runtime_tools if tool.name in core_tool_names]
            user_tools = [tool for tool in runtime_tools if tool.name not in core_tool_names]
            max_user_tools = max(config.max_visible_tools - len(core_tools), 0)
            runtime_tools = user_tools[:max_user_tools] + core_tools
            hidden_tools = [tool.name for tool in user_tools[max_user_tools:]]
            if hidden_tools:
                logger.warning(
                    "tools_capped: visible=%d total=%d hidden=%s",
                    config.max_visible_tools,
                    total_candidate_tools,
                    hidden_tools,
                    extra={
                        "agent_id": config.agent_id,
                        "visible_tool_count": config.max_visible_tools,
                    },
                )

        return runtime_tools, total_candidate_tools

    @staticmethod
    def _describe_subagent(config: AgentConfig) -> str:
        """Return concise routing text for a subagent."""
        description = config.description.strip()
        if description:
            return description
        legacy_description = str(config.custom_data.get("description", "")).strip()
        if legacy_description:
            return legacy_description
        name = config.name.strip()
        if name and name != config.agent_id:
            return f"Agent name: {name}"
        return "No description provided."

    def _make_default_selection_policy(self) -> SubagentSelectionPolicy:
        """Build the default capability-based subagent selector."""
        descriptions = {
            cfg.agent_id: self._describe_subagent(cfg)
            for cfg in self._agent_registry.list_subagents()
        }
        return CapabilityBasedSelectionPolicy(
            llm_client=self._llm,
            agent_descriptions=descriptions,
        )

    async def _prepare_config_with_skills(
        self,
        *,
        context: AgentContext,
        report: ExecutionReport,
    ) -> AgentConfig:
        cfg = context.config
        if self._skill_manager is None or not cfg.skill_set:
            return cfg

        load_skill_def = register_load_skill_tool(
            self._tool_registry,
            skill_manager=self._skill_manager,
            max_skill_loads_per_run=cfg.max_skill_loads_per_run,
        )
        tool_definitions = list(cfg.tool_definitions)
        if not any(defn.name == load_skill_def.name for defn in tool_definitions):
            tool_definitions.append(load_skill_def)

        context.tool_context.metadata[ToolContextMeta.SKILL_SET] = list(cfg.skill_set)
        context.tool_context.metadata[ToolContextMeta.SKILL_LOAD_MAX_CHARS] = cfg.skill_load_max_chars
        context.tool_context.metadata[ToolContextMeta.MAX_SKILL_LOADS_PER_RUN] = cfg.max_skill_loads_per_run

        try:
            index_entries = await self._skill_manager.get_index(
                context.config.agent_id,
                list(cfg.skill_set),
                max_chars=cfg.skill_index_max_chars,
            )
        except Exception as exc:
            logger.exception(
                "Failed to fetch skill index",
                extra=context_log_extra(context, report=report),
            )
            report.warnings.append(f"skill_index_load_failed:{type(exc).__name__}")
            return cfg.model_copy(update={"tool_definitions": tool_definitions}, deep=True)

        if not index_entries:
            return cfg.model_copy(update={"tool_definitions": tool_definitions}, deep=True)

        skill_block = self._build_skill_index_block(index_entries)
        merged_system = (
            f"{cfg.system_prompt}\n\n{skill_block}" if cfg.system_prompt.strip() else skill_block
        )
        await self._bus.emit_simple(
            EV_SKILL_INDEX_INJECTED,
            **event_context_payload(
                context,
                phase="planning",
                scope_id=f"skill-index:{context.request_id}",
                source_component="orchestrator.skills",
            ),
            skill_count=len(index_entries),
            max_chars=cfg.skill_index_max_chars,
        )
        return cfg.model_copy(
            deep=True,
            update={
                "tool_definitions": tool_definitions,
                "system_prompt": merged_system,
            }
        )

    @staticmethod
    def _build_skill_index_block(entries: list[SkillIndexEntry]) -> str:
        lines = [
            "## Available skills",
            (
                "Use the `load_skill` tool only when you need full instructions "
                "for a specific skill."
            ),
        ]
        for entry in entries:
            tags_text = ",".join(entry.tags) if entry.tags else "-"
            lines.append(
                "- "
                f"{entry.name}: {entry.description} "
                f"[trust={entry.trust_level.value}; tags={tags_text}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _normalize_shell_request(
        arguments: dict[str, Any],
        capability: ShellToolConfig,
        tool_context: Any,
    ) -> ShellExecutionRequest:
        return ShellHandler.normalize_shell_request(arguments, capability, tool_context)

    @staticmethod
    def _shell_result_to_tool_result(
        *,
        tool_call_id: str,
        tool_name: str,
        request: ShellExecutionRequest,
        result: Any,
        capability: ShellToolConfig,
    ) -> ToolResult:
        return ShellHandler.shell_result_to_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            request=request,
            result=result,
            capability=capability,
        )

    @staticmethod
    def _build_shell_command_plan(
        *,
        tool_call_id: str,
        tool_name: str,
        request: ShellExecutionRequest,
        capability: ShellToolConfig,
    ) -> ShellCommandPlan:
        return ShellHandler.build_shell_command_plan(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            request=request,
            capability=capability,
        )

    async def _resume_pending_shell_approval(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
        pending_plan: ShellCommandPlan,
    ) -> None:
        await self._shell_handler.resume_pending_shell_approval(
            messages=messages,
            context=context,
            report=report,
            pending_plan=pending_plan,
        )

    async def _invoke_llm_turn(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
        config: AgentConfig,
        runtime_tools: list[ToolDefinition],
        stream_event_callback: Any,
        llm_request_kwargs: dict[str, Any],
    ) -> Message:
        if config.response_format is not None and not runtime_tools:
            structured = await self._llm.complete_structured(
                messages=messages,
                schema=config.response_format,
                system=config.system_prompt,
                model=config.model,
                temperature=config.temperature,
                max_tokens=resolve_max_tokens(config),
                api_mode=config.api_mode,
                logging_context=context_log_extra(context, report=report, model=config.model),
                **llm_request_kwargs,
            )
            if hasattr(structured, "model_dump_json"):
                content = str(structured.model_dump_json())
                structured_payload = structured.model_dump(mode="json")
            else:
                content = str(structured)
                structured_payload = structured
            if isinstance(structured_payload, dict):
                structured_payload = dict(structured_payload)
                structured_payload.pop("__protocore_usage__", None)
            structured_usage = extract_structured_usage(structured)
            return Message.model_validate(
                {
                    "role": "assistant",
                    "content": content,
                    "structured_output": structured_payload,
                    "usage": structured_usage,
                }
            )
        return await self._llm.complete(
            messages=messages,
            tools=runtime_tools,
            system=config.system_prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=resolve_max_tokens(config),
            api_mode=config.api_mode,
            stream=config.stream,
            stream_event_callback=stream_event_callback if config.stream else None,
            logging_context=context_log_extra(context, report=report, model=config.model),
            **llm_request_kwargs,
        )

    @staticmethod
    def _record_output_token_guardrail_warnings(
        report: ExecutionReport,
        config: AgentConfig,
        assistant_message: Message,
    ) -> None:
        usage = assistant_message.usage
        if usage is None:
            return
        output_tokens = usage.output_tokens
        if (
            config.output_token_soft_limit is not None
            and output_tokens >= config.output_token_soft_limit
        ):
            report.add_warning(
                "output_token_soft_limit_exceeded:"
                f"{output_tokens}/{config.output_token_soft_limit}"
            )
        if (
            config.output_token_hard_limit is not None
            and output_tokens >= config.output_token_hard_limit
        ):
            report.add_warning(
                "output_token_hard_limit_exceeded:"
                f"{output_tokens}/{config.output_token_hard_limit}"
            )

    # ------------------------------------------------------------------
    # Immutable loop
    # ------------------------------------------------------------------

    async def _immutable_loop(
        self,
        context: AgentContext,
        report: ExecutionReport,
        cancel_ctx: CancellationContext,
        run_kind: RunKind,
    ) -> Result:
        """The loop that NEVER changes structure.

        pre-hooks → LLM call → stop-check → tool dispatch → post-hooks → repeat

        Hook note:
        ``pre_llm_call`` is emitted only for runs that actually execute this
        loop. AUTO_SELECT leader runs delegate directly and do not enter here.
        """
        messages = list(context.messages)
        cfg = resolve_effective_llm_config(
            context.config,
            run_kind=run_kind,
            call_purpose="main_loop",
        )
        runtime_tools, total_candidate_tools = self._build_runtime_tool_definitions(
            cfg, run_kind
        )
        context.tool_context.metadata[ToolContextMeta.VISIBLE_TOOL_NAMES] = [
            tool.name for tool in runtime_tools
        ]
        clipped_tool_count = total_candidate_tools - len(runtime_tools)
        if clipped_tool_count > 0:
            context.tool_context.metadata[ToolContextMeta.VISIBLE_TOOL_NAMES_CLIPPED] = clipped_tool_count
            report.add_warning(
                f"runtime_tools_capped:{len(runtime_tools)}/{total_candidate_tools}"
            )
        else:
            context.tool_context.metadata.pop(ToolContextMeta.VISIBLE_TOOL_NAMES_CLIPPED, None)
        tool_calls_count = 0
        iteration = 0
        try:
            decisions_raw = context.metadata.get(AgentContextMeta.SHELL_APPROVAL_DECISIONS)
            pending_raw = context.metadata.get(AgentContextMeta.PENDING_SHELL_APPROVAL)
            if pending_raw is None and isinstance(decisions_raw, dict) and decisions_raw:
                raise ContractViolationError(
                    "PENDING_SHELL_APPROVAL_REQUIRED",
                    (
                        "shell_approval_decisions were provided without "
                        "pending_shell_approval. Resume requires the original "
                        "pending plan payload. Use resume_from_pending(result, decision) "
                        "or persist Result.metadata['pending_shell_approval'] from the "
                        "partial run."
                    ),
                )
            if isinstance(pending_raw, dict):
                pending_plan = ShellCommandPlan.model_validate(pending_raw)
                context.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = pending_plan.model_dump()
                await self._resume_pending_shell_approval(
                    messages=messages,
                    context=context,
                    report=report,
                    pending_plan=pending_plan,
                )

            while iteration < cfg.max_iterations:
                cancel_ctx.check()
                iteration += 1
                await self._bus.emit_simple(
                    EV_LOOP_ITERATION_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="main_turn" if run_kind == RunKind.LEADER else "subagent",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.loop",
                    ),
                    message_count=len(messages),
                    tool_calls_used=tool_calls_count,
                    token_estimate_before=estimate_llm_prompt_tokens(
                        messages,
                        system=cfg.system_prompt,
                        api_mode=cfg.api_mode,
                        estimate_tokens_func=cfg.estimate_tokens_func,
                        model=cfg.model,
                        profile=cfg.token_estimator_profile,
                        chars_per_token=cfg.chars_per_token_estimate,
                    ),
                )

                # ----- Pre-LLM hooks (compression, drain, etc.) -----
                messages, report = await self._pre_llm_hooks(
                    messages,
                    context,
                    report,
                    run_kind=run_kind,
                )

                # ----- LLM call -----
                t0 = time.monotonic()
                llm_request_kwargs = build_llm_request_kwargs(
                    cfg,
                    has_tools=bool(runtime_tools),
                )
                llm_messages = list(messages)
                visible_tool_names = list(
                    context.tool_context.metadata.get(ToolContextMeta.VISIBLE_TOOL_NAMES, [])
                )
                clipped_visible_tools = context.tool_context.metadata.get(
                    ToolContextMeta.VISIBLE_TOOL_NAMES_CLIPPED, 0
                )
                await self._bus.emit_simple(
                    EV_LLM_REQUEST_PREPARED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.llm",
                    ),
                    message_count=len(llm_messages),
                    system_chars=len(cfg.system_prompt),
                    input_preview=string_preview(llm_messages[-1].content if llm_messages else "", limit=1200),
                    tool_count=len(runtime_tools),
                    tool_names=visible_tool_names,
                )
                await self._bus.emit_simple(
                    EV_LLM_CALL_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.llm",
                    ),
                    model=cfg.model,
                    api_mode=cfg.api_mode.value,
                    stream=cfg.stream,
                    temperature=cfg.temperature,
                    max_tokens=resolve_max_tokens(cfg),
                    enable_thinking=cfg.enable_thinking,
                    emit_reasoning_in_stream=cfg.emit_reasoning_in_stream,
                    thinking_profile=(
                        cfg.thinking_profile.value if cfg.thinking_profile else None
                    ),
                    thinking_run_policy=cfg.thinking_run_policy.value,
                    visible_tool_names=visible_tool_names,
                    visible_tool_count=len(visible_tool_names),
                    visible_tool_names_clipped=clipped_visible_tools,
                    llm_request_kwargs=llm_request_kwargs,
                )
                # This hook is only emitted on call sites that
                # execute an LLM turn (i.e. runs that enter _immutable_loop).
                self._hooks.call_pre_llm(messages=messages, context=context, report=report)
                stream_event_callback, stream_stats = self._make_stream_event_callback(
                    context=context,
                    iteration=iteration,
                )
                assistant_message = await self._policy_runner.call(
                    operation="llm.complete",
                    report=report,
                    latency_ms_sink=report.llm_latency_ms,
                    fn=partial(
                        self._invoke_llm_turn,
                        messages=llm_messages,
                        context=context,
                        report=report,
                        config=cfg,
                        runtime_tools=runtime_tools,
                        stream_event_callback=stream_event_callback,
                        llm_request_kwargs=llm_request_kwargs,
                    ),
                )

                llm_latency = (time.monotonic() - t0) * 1000

                # Accumulate token usage
                if assistant_message.usage:
                    self._accumulate_llm_usage(
                        report=report,
                        context=context,
                        input_tokens=assistant_message.usage.input_tokens,
                        output_tokens=assistant_message.usage.output_tokens,
                        cached_tokens=assistant_message.usage.cached_tokens,
                        reasoning_tokens=assistant_message.usage.reasoning_tokens,
                    )
                self._record_output_token_guardrail_warnings(
                    report=report,
                    config=cfg,
                    assistant_message=assistant_message,
                )

                messages.append(assistant_message)

                tool_calls = assistant_message.tool_calls or []
                recovered_tool_calls_count = 0
                if not tool_calls and cfg.allow_fallback_tool_call_recovery:
                    shell_tool_name = (
                        context.config.shell_tool_config.tool_name
                        if context.config.shell_tool_enabled_for_run(run_kind)
                        else None
                    )
                    recovered_tool_calls = recover_tool_calls_from_assistant_text(
                        assistant_message.content,
                        runtime_tools,
                        blocked_tool_names={shell_tool_name} if shell_tool_name else None,
                    )
                    if recovered_tool_calls:
                        assistant_message.tool_calls = [
                            ToolCall.model_validate(tool_call)
                            for tool_call in recovered_tool_calls
                        ]
                        tool_calls = assistant_message.tool_calls or []
                        recovered_tool_calls_count = len(recovered_tool_calls)
                        report.add_warning(
                            f"fallback_tool_call_parsed:{len(recovered_tool_calls)}"
                        )
                        await self._bus.emit_simple(
                            EV_INJECTION_SIGNAL,
                            **event_context_payload(
                                context,
                                run_kind=run_kind,
                                phase="tool_use",
                                scope_id=f"turn:{context.request_id}:{iteration}",
                                iteration=iteration,
                                source_component="orchestrator.llm",
                            ),
                            tool_name="fallback_recovery",
                            tool_call_id=None,
                            result_summary=string_preview(
                                assistant_message.content
                                if isinstance(assistant_message.content, str)
                                else str(assistant_message.content or ""),
                                limit=300,
                            ),
                        )
                assistant_content = (
                    assistant_message.content
                    if isinstance(assistant_message.content, str)
                    else str(assistant_message.content or "")
                )
                assistant_usage = (
                    assistant_message.usage.model_dump(mode="json")
                    if assistant_message.usage is not None
                    else None
                )
                await self._bus.emit_simple(
                    EV_LLM_OUTPUT_PARSED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.llm",
                    ),
                    content_chars=len(assistant_content),
                    tool_call_count=len(tool_calls),
                )
                if not assistant_content.strip() and not tool_calls:
                    await self._bus.emit_simple(
                        EV_LLM_OUTPUT_EMPTY,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="tool_use",
                            scope_id=f"turn:{context.request_id}:{iteration}",
                            iteration=iteration,
                            source_component="orchestrator.llm",
                        ),
                        finish_reason=StopReason.END_TURN.value,
                    )
                await self._bus.emit_simple(
                    EV_LLM_CALL_END,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.llm",
                    ),
                    latency_ms=llm_latency,
                    usage=assistant_usage,
                    assistant_content_chars=len(assistant_content),
                    assistant_content_preview=string_preview(
                        assistant_content, limit=1200
                    ),
                    assistant_tool_call_count=len(tool_calls),
                    assistant_tool_call_names=[
                        tc.function.name
                        for tc in tool_calls
                    ],
                    stream=cfg.stream,
                    stream_delta_count=stream_stats["delta_count"],
                    stream_text_chars=stream_stats["text_chars"],
                    stream_reasoning_chars=stream_stats["reasoning_chars"],
                    recovered_tool_calls_count=recovered_tool_calls_count,
                    finish_reason=(
                        StopReason.TOOL_USE.value if tool_calls else StopReason.END_TURN.value
                    ),
                )
                await self._bus.emit_simple(
                    EV_LLM_STREAM_COMPLETED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.llm",
                    ),
                    delta_count=stream_stats["delta_count"],
                    text_chars=stream_stats["text_chars"],
                    reasoning_chars=stream_stats["reasoning_chars"],
                )

                # ----- Stop condition check -----
                if not tool_calls:
                    # Model produced final answer — exit loop
                    content = (
                        assistant_message.content
                        if isinstance(assistant_message.content, str)
                        else str(assistant_message.content or "")
                    )
                    # Post-process via hook (firstresult)
                    modified = self._hooks.call_response_generated(
                        content=content, context=context, report=report
                    )
                    final_content = modified if modified is not None else content
                    report.finalize(
                        ExecutionStatus.COMPLETED,
                        stop_reason=StopReason.END_TURN,
                    )
                    result_metadata: dict[str, Any] = {}
                    structured_output = getattr(assistant_message, "structured_output", None)
                    if structured_output is not None:
                        result_metadata["structured"] = structured_output
                    return Result(
                        content=final_content,
                        status=ExecutionStatus.COMPLETED,
                        artifacts=report.artifacts,
                        metadata=result_metadata,
                    )
                for index, tc in enumerate(tool_calls):
                    function = tc.function
                    raw_arguments = (
                        function.arguments
                        if isinstance(function.arguments, (str, dict))
                        else str(function.arguments)
                    )
                    detected_arguments: dict[str, Any] | None = None
                    if isinstance(raw_arguments, dict):
                        detected_arguments = raw_arguments
                    elif isinstance(raw_arguments, str):
                        try:
                            parsed_arguments = json.loads(raw_arguments)
                        except Exception:
                            detected_arguments = None
                        else:
                            if isinstance(parsed_arguments, dict):
                                detected_arguments = parsed_arguments
                    detected_payload: dict[str, Any] = {
                        "arguments_json": (
                            raw_arguments
                            if isinstance(raw_arguments, str)
                            else json.dumps(raw_arguments, ensure_ascii=True, sort_keys=True, default=str)
                        )
                    }
                    if detected_arguments is not None:
                        detected_payload["arguments"] = redact_sensitive_keys(detected_arguments)
                        detected_payload["argument_keys"] = sorted(detected_arguments.keys())
                        shell_tool_name = context.config.shell_tool_config.tool_name
                        if function.name == shell_tool_name:
                            shell_command = detected_arguments.get("command")
                            if isinstance(shell_command, str):
                                detected_payload["shell_command"] = shell_command
                            shell_cwd = detected_arguments.get("cwd")
                            if isinstance(shell_cwd, str):
                                detected_payload["shell_cwd"] = shell_cwd
                    await self._bus.emit_simple(
                        EV_TOOL_CALL_DETECTED,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="tool_use",
                            scope_id=f"tool:{tc.id}",
                            iteration=iteration,
                            source_component="orchestrator.loop",
                        ),
                        tool_call_id=tc.id,
                        tool_name=function.name,
                        arguments_preview=string_preview(function.arguments, limit=600),
                        tool_call_index=index,
                        **detected_payload,
                    )

                # ----- Tool dispatch (dict-lookup, no branching on names) -----
                tool_results: list[ToolResult] = []
                if getattr(cfg, "parallel_tool_calls", False) and len(tool_calls) > 1:
                    cancel_ctx.check()
                    slot = max(0, cfg.max_tool_calls - tool_calls_count)
                    batch = tool_calls[:slot] if slot < len(tool_calls) else tool_calls
                    truncated_by_budget = len(batch) < len(tool_calls)
                    tool_tasks = [
                        asyncio.create_task(
                            self._dispatch_tool(tc, context, report, run_kind)
                        )
                        for tc in batch
                    ]
                    cancel_waiter = asyncio.create_task(cancel_ctx.wait())
                    try:
                        pending_tasks: set[asyncio.Task[Any]] = set(tool_tasks)
                        while pending_tasks:
                            done, pending = await asyncio.wait(
                                pending_tasks | {cancel_waiter},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if cancel_waiter in done:
                                for task in pending:
                                    if task is not cancel_waiter:
                                        task.cancel()
                                await asyncio.gather(
                                    *(task for task in tool_tasks),
                                    return_exceptions=True,
                                )
                                raise asyncio.CancelledError(cancel_ctx.reason)
                            pending_tasks = cast(
                                set[asyncio.Task[Any]],
                                {task for task in pending if task is not cancel_waiter},
                            )
                        results = await asyncio.gather(*tool_tasks, return_exceptions=True)
                    finally:
                        cancel_waiter.cancel()
                        await asyncio.gather(cancel_waiter, return_exceptions=True)
                    for i, r in enumerate(results):
                        if isinstance(r, asyncio.CancelledError):
                            # Cancellation must propagate, not become ToolResult.
                            raise r
                        if isinstance(r, PendingShellApprovalError):
                            # Shell approval required — save pending state
                            # and return PARTIAL, mirroring sequential behavior.
                            if tool_results:
                                append_tool_results_as_messages(messages, tool_results)
                            context.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = r.plan.model_dump()
                            report.add_artifact(
                                f"shell_approval_pending:{r.plan.plan_id}"
                            )
                            report.finalize(
                                ExecutionStatus.PARTIAL,
                                stop_reason=StopReason.APPROVAL_REQUIRED,
                                error_code="APPROVAL_REQUIRED",
                                error_message=(
                                    f"Shell approval required for plan_id={r.plan.plan_id}"
                                ),
                            )
                            return Result(
                                content="[approval required before shell execution]",
                                status=ExecutionStatus.PARTIAL,
                                artifacts=report.artifacts,
                                metadata={
                                    AgentContextMeta.PENDING_SHELL_APPROVAL: r.plan.model_dump()
                                },
                            )
                        elif isinstance(r, BaseException):
                            exc_str = (
                                str(r)
                                if isinstance(r, Exception)
                                else repr(r)
                            )
                            logger.error(
                                "Parallel tool call failed: tool_call=%s error=%s",
                                batch[i],
                                exc_str,
                                extra=context_log_extra(context, report=report),
                            )
                            fn = batch[i].get("function") or {}
                            tool_results.append(
                                ToolResult(
                                    tool_call_id=batch[i].get("id", ""),
                                    tool_name=(
                                        fn.get("name", "unknown")
                                        if isinstance(fn, dict)
                                        else getattr(fn, "name", "unknown")
                                    ),
                                    content=str(r),
                                    is_error=True,
                                )
                            )
                        else:
                            tool_results.append(r)
                        tool_calls_count += 1
                        report.tool_calls_total += 1
                        name = tool_results[-1].tool_name
                        report.increment_tool_call(name)
                        self._record_tool_call_detail(
                            report=report,
                            tool_call=batch[i],
                            result=tool_results[-1],
                        )
                        if tool_results[-1].is_error:
                            report.tool_failures += 1
                        if tool_results[-1].prompt_injection_signal:
                            report.prompt_injection_signals += 1
                    if truncated_by_budget:
                        cancel_ctx.check()
                        append_tool_results_as_messages(messages, tool_results)
                        messages.append(
                            Message(role="system", content=FORCED_FINALIZATION_MSG)
                        )
                        report.forced_finalization_triggered = True
                        await self._bus.emit_simple(
                            EV_TOOL_BUDGET_EXCEEDED,
                            **event_context_payload(
                                context,
                                run_kind=run_kind,
                                phase="tool_use",
                                scope_id=f"turn:{context.request_id}:{iteration}",
                                iteration=iteration,
                                source_component="orchestrator.loop",
                            ),
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=cfg.max_tool_calls,
                            pending_tool_calls_dropped=len(tool_calls) - len(batch),
                        )
                        logger.info(
                            "Tool budget exceeded: agent=%s limit=%d",
                            cfg.agent_id,
                            cfg.max_tool_calls,
                        )
                        return await self._finalize_after_tool_budget(
                            messages=messages,
                            context=context,
                            report=report,
                            run_kind=run_kind,
                        )
                else:
                    for tc in tool_calls:
                        cancel_ctx.check()
                        if tool_calls_count >= cfg.max_tool_calls:
                            return await self._handle_tool_budget_exhausted(
                                messages=messages,
                                context=context,
                                report=report,
                                tool_results=tool_results,
                                tool_calls_count=tool_calls_count,
                                run_kind=run_kind,
                            )
                        try:
                            tool_result = await self._dispatch_tool(
                                tc,
                                context,
                                report,
                                run_kind,
                            )
                        except PendingShellApprovalError as pending:
                            if tool_results:
                                append_tool_results_as_messages(messages, tool_results)
                            context.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = pending.plan.model_dump()
                            report.add_artifact(
                                f"shell_approval_pending:{pending.plan.plan_id}"
                            )
                            report.finalize(
                                ExecutionStatus.PARTIAL,
                                stop_reason=StopReason.APPROVAL_REQUIRED,
                                error_code="APPROVAL_REQUIRED",
                                error_message=(
                                    f"Shell approval required for plan_id={pending.plan.plan_id}"
                                ),
                            )
                            return Result(
                                content="[approval required before shell execution]",
                                status=ExecutionStatus.PARTIAL,
                                artifacts=report.artifacts,
                                metadata={
                                    AgentContextMeta.PENDING_SHELL_APPROVAL: pending.plan.model_dump()
                                },
                            )
                        tool_results.append(tool_result)
                        tool_calls_count += 1
                        report.tool_calls_total += 1
                        name = tool_result.tool_name
                        report.increment_tool_call(name)
                        self._record_tool_call_detail(
                            report=report,
                            tool_call=tc,
                            result=tool_result,
                        )
                        if tool_result.is_error:
                            report.tool_failures += 1
                        if tool_result.prompt_injection_signal:
                            report.prompt_injection_signals += 1

                # Append tool results as tool-role messages
                cancel_ctx.check()
                append_tool_results_as_messages(messages, tool_results)
                for message_index, result in enumerate(tool_results, start=len(messages) - len(tool_results)):
                    await self._bus.emit_simple(
                        EV_TOOL_RESULT_APPENDED,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="tool_use",
                            scope_id=f"tool:{result.tool_call_id}",
                            iteration=iteration,
                            source_component="orchestrator.loop",
                        ),
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        message_index=message_index,
                    )

                if any(tool_requests_manual_compact(tr) for tr in tool_results):
                    cancel_ctx.check()
                    # NOTE: Intentional in-place mutation so hooks/state snapshots
                    # keep observing the live conversation object.
                    context.messages[:] = messages
                    messages = await self.trigger_manual_compact(
                        context,
                        report,
                        run_kind=run_kind,
                    )

                report.loop_count = iteration
                await self._bus.emit_simple(
                    EV_LOOP_ITERATION_END,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="main_turn" if run_kind == RunKind.LEADER else "subagent",
                        scope_id=f"turn:{context.request_id}:{iteration}",
                        iteration=iteration,
                        source_component="orchestrator.loop",
                    ),
                    tool_calls_in_iteration=len(tool_results),
                    tool_call_names=[result.tool_name for result in tool_results],
                    tool_failures=sum(1 for result in tool_results if result.is_error),
                    token_estimate_after=estimate_llm_prompt_tokens(
                        messages,
                        system=cfg.system_prompt,
                        api_mode=cfg.api_mode,
                        estimate_tokens_func=cfg.estimate_tokens_func,
                        model=cfg.model,
                        profile=cfg.token_estimator_profile,
                        chars_per_token=cfg.chars_per_token_estimate,
                    ),
                )

            # Max iterations reached
            report.finalize(
                ExecutionStatus.FAILED,
                stop_reason=StopReason.MAX_ITERATIONS,
                error_code="MAX_ITERATIONS_EXCEEDED",
                error_message=f"Exceeded max_iterations={cfg.max_iterations}",
            )
            report.add_warning("max_iterations_exceeded")
            return Result(
                content="[max iterations reached]",
                status=ExecutionStatus.FAILED,
            )
        except PendingShellApprovalError as pending:
            context.metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = pending.plan.model_dump()
            if report.status == ExecutionStatus.RUNNING:
                report.finalize(
                    ExecutionStatus.PARTIAL,
                    stop_reason=StopReason.APPROVAL_REQUIRED,
                    error_code="APPROVAL_REQUIRED",
                    error_message=(
                        f"Shell approval required for plan_id={pending.plan.plan_id}"
                    ),
                )
            return Result(
                content="[approval required before shell execution]",
                status=ExecutionStatus.PARTIAL,
                artifacts=report.artifacts,
                metadata={AgentContextMeta.PENDING_SHELL_APPROVAL: pending.plan.model_dump()},
            )
        finally:
            # NOTE: Intentional in-place mutation so external holders of
            # context.messages keep the latest conversation reference.
            context.messages[:] = messages

    async def _finalize_after_tool_budget(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
        run_kind: RunKind,
    ) -> Result:
        """Force one final no-tools LLM turn after tool budget exhaustion."""
        finalization_messages = list(messages)
        effective_config = resolve_effective_llm_config(
            context.config,
            run_kind=run_kind,
            call_purpose="finalize",
        )
        stream_event_callback, _stream_stats = self._make_stream_event_callback(
            context=context,
            iteration=len(report.llm_latency_ms) + 1,
        )
        async def invoke_finalizer() -> Message:
            return await self._llm.complete(
                messages=finalization_messages,
                tools=[],
                system=(
                    f"{context.config.system_prompt}\n\n{FORCED_FINALIZATION_MSG}"
                    if context.config.system_prompt
                    else FORCED_FINALIZATION_MSG
                ),
                model=effective_config.model,
                temperature=effective_config.temperature,
                max_tokens=resolve_max_tokens(effective_config),
                api_mode=effective_config.api_mode,
                stream=effective_config.stream,
                stream_event_callback=(
                    stream_event_callback if effective_config.stream else None
                ),
                logging_context=context_log_extra(
                    context,
                    report=report,
                    phase="forced_finalization",
                ),
                **build_llm_request_kwargs(effective_config),
            )
        try:
            final_message = await self._policy_runner.call(
                operation="llm.finalize",
                report=report,
                latency_ms_sink=report.llm_latency_ms,
                fn=invoke_finalizer,
            )
        except Exception:
            report.finalize(
                ExecutionStatus.FAILED,
                stop_reason=StopReason.ERROR,
                error_code="FINALIZATION_LLM_FAILED",
                error_message="LLM call failed during forced finalization",
            )
            raise
        if final_message.usage:
            self._accumulate_llm_usage(
                report=report,
                context=context,
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
                cached_tokens=final_message.usage.cached_tokens,
                reasoning_tokens=final_message.usage.reasoning_tokens,
            )
        self._record_output_token_guardrail_warnings(
            report=report,
            config=effective_config,
            assistant_message=final_message,
        )

        messages.append(final_message)
        content = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content or "")
        )
        modified = self._hooks.call_response_generated(
            content=content,
            context=context,
            report=report,
        )
        final_content = modified if modified is not None else content
        report.finalize(
            ExecutionStatus.COMPLETED,
            stop_reason=StopReason.TOOL_BUDGET_EXCEEDED,
        )
        # NOTE: Intentional in-place mutation so external holders of
        # context.messages keep the latest conversation reference.
        context.messages[:] = messages
        return Result(
            content=final_content,
            status=ExecutionStatus.COMPLETED,
            artifacts=report.artifacts,
        )

    async def _handle_tool_budget_exhausted(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
        tool_results: list[ToolResult],
        tool_calls_count: int,
        run_kind: RunKind,
    ) -> Result:
        """Append completed results and force finalization after budget exhaustion."""
        if tool_results:
            append_tool_results_as_messages(messages, tool_results)
        messages.append(Message(role="system", content=FORCED_FINALIZATION_MSG))
        report.forced_finalization_triggered = True
        await self._bus.emit_simple(
            EV_TOOL_BUDGET_EXCEEDED,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="tool_use",
                scope_id=f"turn:{context.request_id}:{report.loop_count + 1}",
                iteration=report.loop_count + 1,
                source_component="orchestrator.loop",
            ),
            tool_calls_count=tool_calls_count,
            max_tool_calls=context.config.max_tool_calls,
        )
        logger.info(
            "Tool budget exceeded: agent=%s limit=%d",
            context.config.agent_id,
            context.config.max_tool_calls,
        )
        return await self._finalize_after_tool_budget(
            messages=messages,
            context=context,
            report=report,
            run_kind=run_kind,
        )

    # ------------------------------------------------------------------
    # Pre-LLM hooks (compression applied here)
    # ------------------------------------------------------------------

    async def _pre_llm_hooks(
        self,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
        *,
        run_kind: RunKind,
    ) -> tuple[list[Message], ExecutionReport]:
        cfg = context.config

        # Layer 1: micro_compact every turn
        if self._compressor is not None and callable(
            getattr(type(self._compressor), "apply_micro", None)
        ):
            compressor_with_micro = cast(Any, self._compressor)
            new_msgs, micro_count = compressor_with_micro.apply_micro(messages, cfg)
        else:
            new_msgs, micro_count = micro_compact(
                messages,
                keep_recent=cfg.micro_compact_keep_recent,
                max_tool_result_size=cfg.max_tool_result_size,
            )
        if micro_count:
            report.micro_compact_applied += micro_count
            self._hooks.call_micro_compact(messages=new_msgs, context=context, report=report)
            await self._bus.emit_simple(
                EV_COMPRESSION_MICRO,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="main_turn" if run_kind == RunKind.LEADER else "subagent",
                    scope_id=f"turn:{context.request_id}:{report.loop_count + 1}",
                    iteration=report.loop_count + 1,
                    source_component="compression.micro",
                ),
                count=micro_count,
                message_count_before=len(messages),
                message_count_after=len(new_msgs),
            )

        # Layer 2: auto_compact when threshold exceeded
        if self._compressor:
            token_est = estimate_llm_prompt_tokens(
                new_msgs,
                system=cfg.system_prompt,
                api_mode=cfg.api_mode,
                estimate_tokens_func=cfg.estimate_tokens_func,
                model=cfg.model,
                profile=cfg.token_estimator_profile,
                chars_per_token=cfg.chars_per_token_estimate,
            )
            await self._bus.emit_simple(
                EV_COMPACTION_CHECK,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="auto_compact",
                    scope_id=f"compact:{context.request_id}:{report.auto_compact_applied + 1}",
                    iteration=report.loop_count + 1,
                    source_component="compression.auto",
                ),
                token_estimate=token_est,
                threshold=cfg.auto_compact_threshold,
                message_count=len(new_msgs),
            )
            if token_est >= cfg.auto_compact_threshold:
                before = token_est
                msgs_count_before = len(new_msgs)
                await self._bus.emit_simple(
                    EV_COMPACTION_AUTO_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="auto_compact",
                        scope_id=f"compact:{context.request_id}:{report.auto_compact_applied + 1}",
                        iteration=report.loop_count + 1,
                        source_component="compression.auto",
                    ),
                    tokens_before=before,
                    threshold=cfg.auto_compact_threshold,
                    message_count_before=msgs_count_before,
                )
                apply_auto_sig = inspect.signature(self._compressor.apply_auto)
                apply_auto_kwargs: dict[str, Any] = {
                    "precomputed_tokens": token_est,
                    "run_kind": run_kind,
                }
                if "event_bus" in apply_auto_sig.parameters:
                    apply_auto_kwargs["event_bus"] = self._bus
                if "context" in apply_auto_sig.parameters:
                    apply_auto_kwargs["context"] = context
                new_msgs, summary, parse_ok = await self._compressor.apply_auto(
                    new_msgs,
                    cfg,
                    **apply_auto_kwargs,
                )
                if summary is not None:
                    after = estimate_llm_prompt_tokens(
                        new_msgs,
                        system=cfg.system_prompt,
                        api_mode=cfg.api_mode,
                        estimate_tokens_func=cfg.estimate_tokens_func,
                        model=cfg.model,
                        profile=cfg.token_estimator_profile,
                        chars_per_token=cfg.chars_per_token_estimate,
                    )
                    report.auto_compact_applied += 1
                    if not parse_ok:
                        report.auto_compact_failed += 1
                        report.warnings.append("auto_compact_summary_parse_failed")
                    report.tokens_before_compression_total = (
                        (report.tokens_before_compression_total or 0) + before
                    )
                    report.tokens_after_compression_total = (
                        (report.tokens_after_compression_total or 0) + after
                    )
                    report.compression_events.append(
                        CompressionEvent(
                            kind="auto",
                            tokens_before=before,
                            tokens_after=after,
                            messages_affected=msgs_count_before - len(new_msgs),
                            summary_parse_success=parse_ok,
                        )
                    )
                    self._hooks.call_auto_compact(
                        messages=new_msgs, context=context, report=report
                    )
                    await self._bus.emit_simple(
                        EV_COMPRESSION_AUTO,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="auto_compact",
                            scope_id=f"compact:{context.request_id}:{report.auto_compact_applied}",
                            iteration=report.loop_count + 1,
                            source_component="compression.auto",
                        ),
                        tokens_before=before,
                        tokens_after=after,
                    )
                    await self._bus.emit_simple(
                        EV_COMPACTION_END,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="auto_compact",
                            scope_id=f"compact:{context.request_id}:{report.auto_compact_applied}",
                            iteration=report.loop_count + 1,
                            source_component="compression.auto",
                        ),
                        kind="auto",
                        duration_ms=None,
                    )

        return new_msgs, report

    # ------------------------------------------------------------------
    # Manual compact (Layer 3)
    # ------------------------------------------------------------------

    async def trigger_manual_compact(
        self,
        context: AgentContext,
        report: ExecutionReport,
        *,
        run_kind: RunKind = RunKind.LEADER,
    ) -> list[Message]:
        """Trigger manual compression of the current message history.

        Can be invoked from a tool handler or hook. Updates report metrics
        and emits the compression event.
        """
        if self._compressor is None:
            report.warnings.append("manual_compact_no_compressor")
            return list(context.messages)

        before = estimate_tokens(
            context.messages,
            estimate_tokens_func=context.config.estimate_tokens_func,
            model=context.config.model,
            profile=context.config.token_estimator_profile,
            chars_per_token=context.config.chars_per_token_estimate,
        )
        msgs_count_before = len(context.messages)
        await self._bus.emit_simple(
            EV_COMPACTION_MANUAL_START,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="manual_compact",
                scope_id=f"compact:{context.request_id}:manual:{report.manual_compact_applied + 1}",
                source_component="compression.manual",
            ),
            tokens_before=before,
            message_count_before=msgs_count_before,
            requested_by_tool=True,
            reason="manual_compact_requested",
        )
        apply_manual_sig = inspect.signature(self._compressor.apply_manual)
        apply_manual_kwargs: dict[str, Any] = {
            "model": context.config.model,
            "config": context.config,
            "run_kind": run_kind,
        }
        if "event_bus" in apply_manual_sig.parameters:
            apply_manual_kwargs["event_bus"] = self._bus
        if "context" in apply_manual_sig.parameters:
            apply_manual_kwargs["context"] = context
        new_msgs, summary = await self._compressor.apply_manual(
            list(context.messages),
            **apply_manual_kwargs,
        )
        after = estimate_tokens(
            new_msgs,
            estimate_tokens_func=context.config.estimate_tokens_func,
            model=context.config.model,
            profile=context.config.token_estimator_profile,
            chars_per_token=context.config.chars_per_token_estimate,
        )
        report.manual_compact_applied += 1
        report.tokens_before_compression_total = (
            (report.tokens_before_compression_total or 0) + before
        )
        report.tokens_after_compression_total = (
            (report.tokens_after_compression_total or 0) + after
        )
        report.compression_events.append(
            CompressionEvent(
                kind="manual",
                tokens_before=before,
                tokens_after=after,
                messages_affected=msgs_count_before - len(new_msgs),
            )
        )
        if summary is not None:
            report.add_artifact(f"manual_compact_goal:{summary.current_goal[:80]}")
        # NOTE: Intentional in-place mutation so hooks/state snapshots keep
        # observing the canonical list object stored on the context.
        context.messages[:] = new_msgs
        self._hooks.call_manual_compact(
            messages=new_msgs, context=context, report=report
        )
        await self._bus.emit_simple(
            EV_COMPRESSION_MANUAL,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="manual_compact",
                scope_id=f"compact:{context.request_id}:manual:{report.manual_compact_applied}",
                source_component="compression.manual",
            ),
            tokens_before=before,
            tokens_after=after,
        )
        await self._bus.emit_simple(
            EV_COMPACTION_END,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="manual_compact",
                scope_id=f"compact:{context.request_id}:manual:{report.manual_compact_applied}",
                source_component="compression.manual",
            ),
            kind="manual",
            duration_ms=None,
        )
        return new_msgs

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        workflow: WorkflowDefinition,
        context: AgentContext,
        *,
        run_kind: RunKind = RunKind.LEADER,
        report: ExecutionReport | None = None,
    ) -> tuple[Result, ExecutionReport]:
        """Execute a DAG workflow via the injected WorkflowEngine.

        Raises ContractViolationError if no WorkflowEngine is configured.
        Updates ExecutionReport with workflow metrics.
        """
        if self._workflow_engine is None:
            raise ContractViolationError(
                "WORKFLOW_ENGINE_REQUIRED",
                "No WorkflowEngine configured. Inject workflow_engine in constructor.",
            )

        report = report or make_execution_report(context=context, run_kind=run_kind)
        report.workflow_id = workflow.workflow_id
        report.node_count = len(workflow.nodes)
        report.edge_count = len(workflow.edges)
        await self._bus.emit_simple(
            EV_WORKFLOW_START,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="workflow",
                scope_id=f"workflow:{workflow.workflow_id}",
                workflow_id=workflow.workflow_id,
                source_component="orchestrator.workflow",
            ),
            node_count=len(workflow.nodes),
            edge_count=len(workflow.edges),
        )
        self._hooks.call_workflow_start(workflow=workflow, context=context, report=report)

        result, engine_report = await self._workflow_engine.run(workflow, context)
        merge_execution_report(report, engine_report)
        ensure_terminal_report(report, result)
        await self._bus.emit_simple(
            EV_WORKFLOW_END,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="workflow",
                scope_id=f"workflow:{workflow.workflow_id}",
                workflow_id=workflow.workflow_id,
                source_component="orchestrator.workflow",
            ),
            status=report.status.value,
            duration_ms=report.duration_ms,
            node_count=report.node_count,
            edge_count=report.edge_count,
        )
        self._hooks.call_workflow_end(workflow=workflow, result=result, report=report)

        return result, report

    @staticmethod
    def _set_queue_wait_metric(*, report: ExecutionReport, context: AgentContext) -> None:
        set_queue_wait_metric(report=report, context=context)

    @staticmethod
    def _accumulate_llm_usage(
        *,
        report: ExecutionReport,
        context: AgentContext,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int = 0,
    ) -> None:
        accumulate_llm_usage(
            report=report,
            context=context,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @staticmethod
    def _sync_report_metadata(*, report: ExecutionReport, context: AgentContext) -> None:
        merged = dict(report.metadata)
        merged.update(context.metadata)
        report.metadata = merged

    @staticmethod
    def _tool_call_arguments_payload(tool_call: ToolCall | dict[str, Any]) -> Any:
        function_data = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        raw_arguments = (
            function_data.get("arguments")
            if isinstance(function_data, dict)
            else getattr(function_data, "arguments", None)
        )
        if raw_arguments in (None, "", {}, []):
            return {}
        parsed_arguments: Any = raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return {"raw": string_preview(raw_arguments, limit=600)}
        if isinstance(parsed_arguments, (dict, list)):
            return redact_sensitive_keys(parsed_arguments)
        return parsed_arguments

    @classmethod
    def _record_tool_call_detail(
        cls,
        *,
        report: ExecutionReport,
        tool_call: ToolCall | dict[str, Any],
        result: ToolResult,
    ) -> None:
        report.add_tool_call_detail(
            ToolCallRecord(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                arguments=cls._tool_call_arguments_payload(tool_call),
                latency_ms=result.latency_ms,
                success=not result.is_error,
                error_message=(
                    string_preview(result.content, limit=300) if result.is_error else None
                ),
            )
        )

    def _spawn_child_orchestrator(self) -> "AgentOrchestrator":
        """Spawn a child orchestrator with the same injected dependencies."""
        return AgentOrchestrator(
            llm_client=self._llm,
            tool_executor=self._tool_executor,
            tool_registry=self._tool_registry.clone(strict=True),
            agent_registry=self._agent_registry.clone(),
            hook_manager=self._hooks.clone_with_mode(strict=True),
            event_bus=EventBus(parent_bus=self._bus),
            planning_strategy=self._planning,
            planning_policy=self._planning_policy,
            parallel_execution_policy=self._parallel_policy,
            execution_policy=self._policy,
            shell_executor=self._shell_executor,
            shell_safety_policy=self._shell_safety_policy,
            skill_manager=self._skill_manager,
            compressor=self._compressor,
            state_manager=self._state_manager,
            subagent_selection_policy=self._selection_policy,
            transport=self._transport,
            workflow_engine=self._workflow_engine,
            telemetry_collector=self._telemetry,
            timeout_policy=self._timeout_policy,
            retry_policy=self._retry_policy,
        )

    def _make_stream_event_callback(
        self,
        *,
        context: AgentContext,
        iteration: int,
    ) -> tuple[Callable[[dict[str, Any]], Awaitable[None]], dict[str, int]]:
        stats = {
            "delta_count": 0,
            "text_chars": 0,
            "reasoning_chars": 0,
        }

        async def _callback(delta: dict[str, Any]) -> None:
            text = delta.get("text")
            if not isinstance(text, str) or not text:
                return
            stats["delta_count"] += 1
            kind = str(delta.get("kind", "text"))
            if kind == "reasoning":
                stats["reasoning_chars"] += len(text)
            else:
                stats["text_chars"] += len(text)
            await self._bus.emit_simple(
                EV_LLM_STREAM_DELTA,
                **event_context_payload(
                    context,
                    phase="tool_use",
                    scope_id=f"turn:{context.request_id}:{iteration}",
                    iteration=iteration,
                    source_component="orchestrator.stream",
                ),
                delta_index=stats["delta_count"],
                kind=kind,
                chars=len(text),
                text=text,
                provider_event_type=delta.get("provider_event_type"),
            )

        return _callback, stats

    def _build_subagent_context(
        self,
        parent_context: AgentContext,
        agent_id: str,
        task_text: str,
        *,
        inject_structured_contract: bool = False,
    ) -> AgentContext:
        """Create an isolated child context from minimal task payload."""
        child_cfg = self._agent_registry.get(agent_id)
        if child_cfg is None:
            raise ContractViolationError(
                "SUBAGENT_NOT_REGISTERED",
                f"Subagent agent_id={agent_id} is not registered",
            )
        child_cfg = child_cfg.model_copy(
            update={"execution_mode": ExecutionMode.BYPASS},
            deep=True,
        )
        if inject_structured_contract:
            base_system_prompt = child_cfg.system_prompt.strip()
            structured_contract = SubagentResult.prompt_instructions()
            child_system_prompt = (
                f"{base_system_prompt}\n\n{structured_contract}"
                if base_system_prompt
                else structured_contract
            )
            child_cfg = child_cfg.model_copy(
                update={"system_prompt": child_system_prompt},
                deep=True,
            )
        return build_subagent_context(
            parent_context=parent_context,
            child_config=child_cfg,
            task_text=task_text,
        )

    async def _run_child_subagent(
        self,
        *,
        parent_context: AgentContext,
        report: ExecutionReport,
        cancel_ctx: CancellationContext,
        agent_id: str,
        task_text: str,
    ) -> Result:
        child_context = self._build_subagent_context(
            parent_context,
            agent_id,
            task_text,
            inject_structured_contract=True,
        )
        envelope = make_task_envelope(
            sender_id=parent_context.config.agent_id,
            recipient_id=agent_id,
            payload={"task": task_text},
            trace_id=parent_context.trace_id,
            session_id=parent_context.session_id,
            report=report,
        )
        set_active_child_agent_ids(parent_context.metadata, [agent_id])
        self._hooks.call_subagent_start(
            agent_id=agent_id,
            envelope_payload=envelope.payload,
            report=report,
        )
        await self._bus.emit_simple(
            EV_SUBAGENT_START,
            **event_context_payload(
                parent_context,
                run_kind=RunKind.LEADER,
                phase="subagent",
                scope_id=f"subagent:{child_context.request_id}",
                correlation_id=child_context.request_id,
                source_component="orchestrator.subagent",
            ),
            child_agent_id=agent_id,
            child_request_id=child_context.request_id,
            task_preview=string_preview(task_text, limit=300),
        )
        await self._bus.emit_simple(
            EV_SUBAGENT_DISPATCH_START,
            **event_context_payload(
                parent_context,
                run_kind=RunKind.LEADER,
                phase="subagent",
                scope_id=f"subagent:{child_context.request_id}",
                correlation_id=child_context.request_id,
                source_component="orchestrator.subagent",
            ),
            child_agent_id=agent_id,
            task_preview=string_preview(task_text, limit=300),
            envelope_payload=envelope.payload,
        )
        try:
            child_orchestrator = self._spawn_child_orchestrator()
            child_result, child_report = await child_orchestrator.run(
                child_context,
                run_kind=RunKind.SUBAGENT,
                cancel_ctx=cancel_ctx,
            )
            merge_execution_report(report, child_report, include_terminal=False)
            await self._bus.emit_simple(
                EV_SUBAGENT_RESULT_RAW,
                **event_context_payload(
                    parent_context,
                    run_kind=RunKind.LEADER,
                    phase="subagent",
                    scope_id=f"subagent:{child_context.request_id}",
                    correlation_id=child_context.request_id,
                    source_component="orchestrator.subagent",
                ),
                child_agent_id=agent_id,
                raw_chars=len(child_result.content),
                raw_preview=string_preview(child_result.content, limit=600),
            )
            await self._bus.emit_simple(
                EV_SUBAGENT_RESULT_PARSE_START,
                **event_context_payload(
                    parent_context,
                    run_kind=RunKind.LEADER,
                    phase="subagent",
                    scope_id=f"subagent:{child_context.request_id}",
                    correlation_id=child_context.request_id,
                    source_component="orchestrator.subagent",
                ),
                child_agent_id=agent_id,
                schema_name="SubagentResult",
            )
            structured_result = SubagentResult.parse_with_fallback(
                child_result.content,
                agent_id=agent_id,
            )
            if subagent_result_used_fallback(structured_result):
                report.warnings.append(f"subagent_result_fallback:{agent_id}")
                await self._bus.emit_simple(
                    EV_SUBAGENT_RESULT_FALLBACK,
                    **event_context_payload(
                        parent_context,
                        run_kind=RunKind.LEADER,
                        phase="subagent",
                        scope_id=f"subagent:{child_context.request_id}",
                        correlation_id=child_context.request_id,
                        source_component="orchestrator.subagent",
                    ),
                    child_agent_id=agent_id,
                    reason="schema_fallback",
                    fallback_summary_preview=string_preview(
                        structured_result.summary,
                        limit=600,
                    ),
                )
            await self._bus.emit_simple(
                EV_SUBAGENT_RESULT_PARSE_END,
                **event_context_payload(
                    parent_context,
                    run_kind=RunKind.LEADER,
                    phase="subagent",
                    scope_id=f"subagent:{child_context.request_id}",
                    correlation_id=child_context.request_id,
                    source_component="orchestrator.subagent",
                ),
                child_agent_id=agent_id,
                parse_success=not subagent_result_used_fallback(structured_result),
                status=structured_result.status.value,
                tool_calls_made=child_report.tool_calls_total,
            )
            normalized_status = (
                execution_status_from_subagent_status(structured_result.status)
                if child_report.status == ExecutionStatus.COMPLETED
                else child_report.status
            )
            normalized_stop_reason = child_report.stop_reason
            if normalized_stop_reason is None:
                normalized_stop_reason = default_stop_reason_for_status(normalized_status)
            normalized_error_message = child_report.error_message
            if (
                normalized_status in {ExecutionStatus.PARTIAL, ExecutionStatus.FAILED}
                and normalized_error_message is None
                and structured_result.errors
            ):
                normalized_error_message = structured_result.errors[0]
            child_summary = build_subagent_summary(child_report).model_copy(
                deep=True,
                update={
                    "status": normalized_status,
                    "errors": [] if normalized_status == ExecutionStatus.COMPLETED else structured_result.errors,
                }
            )
            report.subagent_runs.append(child_summary)
            report.add_artifact(f"subagent:{agent_id}")
            for artifact in structured_result.artifacts:
                if artifact not in report.artifacts:
                    report.add_artifact(artifact)
            for file_path in structured_result.files_changed:
                if file_path not in report.files_changed:
                    report.add_file_changed(file_path)
            await self._bus.emit_simple(
                EV_SUBAGENT_END,
                **event_context_payload(
                    parent_context,
                    run_kind=RunKind.LEADER,
                    phase="subagent",
                    scope_id=f"subagent:{child_context.request_id}",
                    correlation_id=child_context.request_id,
                    source_component="orchestrator.subagent",
                ),
                child_agent_id=agent_id,
                status=child_report.status.value,
                duration_ms=child_report.duration_ms,
                tool_calls_total=child_report.tool_calls_total,
                tool_failures=child_report.tool_failures,
                warnings=list(child_report.warnings),
            )
            self._hooks.call_subagent_end(
                agent_id=agent_id,
                result=structured_result,
                report=report,
            )
            report.finalize(
                normalized_status,
                stop_reason=normalized_stop_reason,
                error_code=child_report.error_code,
                error_message=normalized_error_message,
            )
            return Result(
                content=structured_result.summary,
                status=normalized_status,
                artifacts=structured_result.artifacts,
                metadata={AgentContextMeta.AUTO_SELECTED_AGENT: agent_id},
            )
        finally:
            set_active_child_agent_ids(parent_context.metadata, [])

    async def _update_session_snapshot(
        self,
        *,
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        """Persist the final session snapshot while preserving stored ``created_at``.

        The StateManager protocol is fully async; this method always awaits
        ``load_session_snapshot`` and degrades gracefully on storage timeouts.
        """
        if self._state_manager is None:
            return
        message_history_ref, execution_metadata_ref = session_refs(context)
        existing_snapshot = await self._call_state_manager_with_timeout(
            operation="load_session_snapshot",
            warning_code="session_snapshot_load_timed_out",
            report=report,
            awaitable=self._state_manager.load_session_snapshot(context.session_id),
        )
        if existing_snapshot is _STATE_MANAGER_TIMEOUT:
            return
        if not isinstance(existing_snapshot, SessionSnapshot):
            existing_snapshot = None
        snapshot = build_session_snapshot(
            context=context,
            report=report,
            message_history_ref=message_history_ref,
            execution_metadata_ref=execution_metadata_ref,
            existing_snapshot=existing_snapshot,
        )
        await_result = await self._call_state_manager_with_timeout(
            operation="update_session_snapshot",
            warning_code="session_snapshot_update_timed_out",
            report=report,
            awaitable=self._state_manager.update_session_snapshot(snapshot),
        )
        if await_result is _STATE_MANAGER_TIMEOUT:
            return

    async def _propagate_cancel(
        self,
        *,
        context: AgentContext,
        report: ExecutionReport,
    ) -> None:
        child_ids = get_active_child_agent_ids(context.metadata)
        if not child_ids:
            return
        if self._transport is None:
            report.warnings.append("cancel_transport_not_configured")
            return
        for child_id in child_ids:
            envelope = make_control_envelope(
                sender_id=context.config.agent_id,
                recipient_id=child_id,
                command=ControlCommand.CANCEL,
                trace_id=context.trace_id,
                session_id=context.session_id,
                report=report,
            )
            try:
                await asyncio.wait_for(
                    self._transport.send(envelope, destination=child_id),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "_propagate_cancel: transport send timed out after 5s for child=%s",
                    child_id,
                    extra=context_log_extra(context, report=report, child_agent_id=child_id),
                )
                report.add_warning(f"cancel_propagation_timed_out:{child_id}")
            except Exception as exc:
                logger.exception(
                    "Failed to propagate cancel to child=%s",
                    child_id,
                    extra=context_log_extra(context, report=report, child_agent_id=child_id),
                )
                report.warnings.append(
                    f"cancel_propagation_failed:{child_id}:{type(exc).__name__}"
                )

    # ------------------------------------------------------------------
    # Tool dispatch (dict-lookup pattern — no if/elif on names)
    # ------------------------------------------------------------------

    async def _dispatch_tool(
        self,
        tool_call: Any,
        context: AgentContext,
        report: ExecutionReport,
        run_kind: RunKind,
    ) -> ToolResult:
        return await self._tool_dispatcher.dispatch_tool(
            tool_call=tool_call,
            context=context,
            report=report,
            run_kind=run_kind,
        )

    async def _evaluate_tool_preflight(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: AgentContext,
        report: ExecutionReport,
        shell_capability: ShellToolConfig | None,
        shell_tool_name: str,
    ) -> tuple[PolicyDecision | None, str | None]:
        return await self._tool_dispatcher.evaluate_tool_preflight(
            tool_name=tool_name,
            arguments=arguments,
            context=context,
            report=report,
            shell_capability=shell_capability,
            shell_tool_name=shell_tool_name,
        )

    def _confirm_destructive_action(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: AgentContext,
    ) -> bool:
        return self._tool_dispatcher.confirm_destructive_action(
            tool_name=tool_name,
            arguments=arguments,
            context=context,
        )

    # ------------------------------------------------------------------
    # Error & cancellation handlers
    # ------------------------------------------------------------------

    async def _handle_cancellation(
        self,
        context: AgentContext,
        report: ExecutionReport,
        cancel_ctx: CancellationContext,
    ) -> Result:
        await self._propagate_cancel(context=context, report=report)
        report.finalize(
            ExecutionStatus.CANCELLED,
            stop_reason=StopReason.CANCELLED,
            error_code="CANCELLED",
            error_message=cancel_ctx.reason,
        )
        try:
            self._hooks.call_on_cancelled(context=context, report=report)
        except Exception as exc:
            logger.exception(
                "cancelled hook failed: agent=%s",
                context.config.agent_id,
                extra=context_log_extra(context, report=report),
            )
            report.warnings.append(f"cancelled_hook_failed:{type(exc).__name__}")
        await self._bus.emit_simple(
            EV_CANCELLATION,
            **event_context_payload(
                context,
                phase="finalization",
                scope_id=f"session:{context.request_id}",
                source_component="orchestrator.run",
            ),
            reason=cancel_ctx.reason,
        )
        logger.info(
            "Agent run cancelled: agent=%s reason=%s",
            context.config.agent_id,
            cancel_ctx.reason,
            extra=context_log_extra(context, report=report),
        )
        return Result(content="[cancelled]", status=ExecutionStatus.CANCELLED)

    async def _handle_error(
        self,
        exc: Exception,
        context: AgentContext,
        report: ExecutionReport,
    ) -> Result:
        is_timeout = isinstance(exc, TimeoutError)
        error_code = "TIMEOUT" if is_timeout else getattr(exc, "error_code", type(exc).__name__)
        error_details = self._public_error_details(exc=exc, error_code=str(error_code), is_timeout=is_timeout)
        public_reason = self._public_error_reason(
            exc=exc,
            is_timeout=is_timeout,
            error_details=error_details,
        )
        report.finalize(
            ExecutionStatus.TIMEOUT if is_timeout else ExecutionStatus.FAILED,
            stop_reason=StopReason.ERROR,
            error_code=str(error_code),
            error_message=str(exc),
        )
        try:
            self._hooks.call_on_error(error=exc, context=context, report=report)
        except Exception as hook_exc:
            logger.exception(
                "error hook failed: agent=%s",
                context.config.agent_id,
                extra=context_log_extra(context, report=report),
            )
            report.warnings.append(f"error_hook_failed:{type(hook_exc).__name__}")
        await self._bus.emit_simple(
            EV_ERROR,
            **event_context_payload(
                context,
                phase="finalization",
                scope_id=f"session:{context.request_id}",
                source_component="orchestrator.run",
            ),
            error_code=str(error_code),
            error_message=str(exc),
        )
        await self._bus.emit_simple(
            EV_LLM_CALL_FAILED,
            **event_context_payload(
                context,
                phase="tool_use",
                scope_id=f"session:{context.request_id}",
                source_component="orchestrator.run",
            ),
            error_code=str(error_code),
            error_message=str(exc),
            attempt=1,
        )
        if isinstance(exc, ContractViolationError):
            logger.warning(
                "Agent run contract violation: agent=%s error=%s",
                context.config.agent_id,
                exc,
                extra=context_log_extra(context, report=report),
            )
        elif is_timeout:
            logger.warning(
                "Agent run timeout: agent=%s error=%s",
                context.config.agent_id,
                exc,
                extra=context_log_extra(context, report=report),
            )
        else:
            logger.exception(
                "Agent run error: agent=%s error=%s",
                context.config.agent_id,
                exc,
                extra=context_log_extra(context, report=report),
            )
        return Result(
            content=f"[error: {public_reason}]",
            status=ExecutionStatus.TIMEOUT if is_timeout else ExecutionStatus.FAILED,
            error_details=error_details,
        )

    @staticmethod
    def _sanitize_public_error_text(value: str, *, limit: int = 240) -> str:
        sanitized = _PUBLIC_ERROR_VALUE_RE.sub(r"\1=[REDACTED]", value).strip()
        if len(sanitized) > limit:
            return sanitized[:limit] + "...[truncated]"
        return sanitized

    @classmethod
    def _public_error_details(
        cls,
        *,
        exc: Exception,
        error_code: str,
        is_timeout: bool,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "type": type(exc).__name__,
            "error_code": error_code,
            "is_timeout": is_timeout,
        }
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            details["status_code"] = status_code
        raw_message = str(exc).strip()
        if raw_message:
            details["message"] = cls._sanitize_public_error_text(raw_message)
        return details

    @classmethod
    def _public_error_reason(
        cls,
        *,
        exc: Exception,
        is_timeout: bool,
        error_details: dict[str, Any] | None = None,
    ) -> str:
        """Return a short safe reason suitable for end-user Result.content."""
        if is_timeout:
            return "timeout"
        if isinstance(exc, ContractViolationError):
            return f"contract:{exc.error_code.lower()}"
        detail_message = None
        if isinstance(error_details, dict):
            maybe_message = error_details.get("message")
            if isinstance(maybe_message, str) and maybe_message:
                detail_message = maybe_message
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if detail_message:
                return f"upstream {status_code}: {detail_message}"
            return f"upstream {status_code}"
        return type(exc).__name__.lower()

__all__ = [
    "AgentOrchestrator",
    "ContractViolationError",
    "ParallelSubagentRunner",
    "run_bypass",
]
