"""Factory helpers for building core objects from config.

Factories centralise object construction, making DI and testing easy.
No concrete vendor dependencies — adapters are injected by the service.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Any

from .context import build_tool_context
from .hooks.manager import HookManager, create_plugin_manager
from .protocols import (
    ExecutionPolicy,
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
from .constants import LOAD_SKILL_TOOL_NAME, MANUAL_COMPACT_TOOL_NAME, PROTOCOL_VERSION
from .registry import AgentRegistry, StrategyRegistry, ToolRegistry
from .types import (
    AgentConfig,
    AgentContext,
    AgentIdentity,
    AgentRole,
    AgentEnvelope,
    ControlCommand,
    EnvelopeMeta,
    ExecutionReport,
    MessageType,
    RunKind,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultMeta,
)

if TYPE_CHECKING:
    from .compression import ContextCompressor
    from .events import EventBus
    from .orchestrator import AgentOrchestrator



_MAX_EXTERNAL_ID_LENGTH = 128
_SAFE_EXTERNAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _normalize_factory_uuid(*, field_name: str, value: str | None) -> str:
    """Validate externally supplied IDs while preserving legacy safe identifiers.

    UUIDs remain the preferred format for externally provided IDs. For backward
    compatibility, short opaque identifiers are still accepted when they only
    use a constrained safe character set.
    """
    if value is None:
        return str(uuid.uuid4())
    normalized = str(value).strip()
    if len(normalized) > _MAX_EXTERNAL_ID_LENGTH:
        raise ValueError(
            f"{field_name} must be <= {_MAX_EXTERNAL_ID_LENGTH} characters when provided"
        )
    if not normalized:
        raise ValueError(f"{field_name} must not be blank when provided explicitly")
    try:
        return str(uuid.UUID(normalized))
    except (AttributeError, TypeError, ValueError) as exc:
        if _SAFE_EXTERNAL_ID_RE.fullmatch(normalized):
            return normalized
        raise ValueError(
            f"{field_name} must be a valid UUID or a safe opaque identifier"
        ) from exc


# ---------------------------------------------------------------------------
# AgentContext factory
# ---------------------------------------------------------------------------


def make_agent_context(
    *,
    config: AgentConfig,
    session_id: str | None = None,
    trace_id: str | None = None,
    request_id: str | None = None,
    parent_agent_id: str | None = None,
    allowed_paths: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    message_history_ref: str | None = None,
    execution_metadata_ref: str | None = None,
    **kwargs: Any,
) -> AgentContext:
    """Build ``AgentContext`` with validated identities.

    Explicit ``session_id`` / ``trace_id`` / ``request_id`` values are treated as
    externally provided input. UUIDs are preferred; legacy opaque identifiers
    remain allowed when they use a constrained safe character set. Omitted
    values are generated automatically.

    ``AgentContext`` intentionally does not accept runtime infrastructure objects
    such as ``ToolRegistry``. Registry wiring happens in ``AgentOrchestrator``
    so leader and subagents can keep focused, per-agent tool sets and distinct
    system prompts (for example a minimal delegating leader plus specialized
    subagents).
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        if "tool_registry" in kwargs:
            raise TypeError(
                "make_agent_context() got unexpected keyword argument 'tool_registry'. "
                "ToolRegistry is injected into AgentOrchestrator(...), not AgentContext. "
                f"Unexpected kwargs: {unknown}."
            )
        raise TypeError(
            "make_agent_context() got unexpected keyword argument(s): "
            f"{unknown}. AgentContext accepts runtime identity/message fields only."
        )

    sid = _normalize_factory_uuid(field_name="session_id", value=session_id)
    tid = _normalize_factory_uuid(field_name="trace_id", value=trace_id)
    rid = _normalize_factory_uuid(field_name="request_id", value=request_id)

    tool_ctx = build_tool_context(
        session_id=sid,
        trace_id=tid,
        agent_id=config.agent_id,
        allowed_paths=allowed_paths,
        metadata=dict(metadata) if metadata else None,
    )

    return AgentContext(
        session_id=sid,
        trace_id=tid,
        request_id=rid,
        parent_agent_id=parent_agent_id,
        message_history_ref=message_history_ref,
        execution_metadata_ref=execution_metadata_ref,
        config=config,
        tool_context=tool_ctx,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# ExecutionReport factory
# ---------------------------------------------------------------------------


def make_execution_report(
    *,
    context: AgentContext,
    run_kind: RunKind = RunKind.LEADER,
) -> ExecutionReport:
    """Create a fresh ``ExecutionReport`` from an ``AgentContext``.

    This factory assumes context identity fields were already normalized by
    ``make_agent_context`` or equivalent service-side validation.
    """
    return ExecutionReport(
        request_id=context.request_id,
        trace_id=context.trace_id,
        session_id=context.session_id,
        agent_id=context.config.agent_id,
        parent_agent_id=context.parent_agent_id,
        run_kind=run_kind,
        model=context.config.model,
        api_mode=context.config.api_mode,
        execution_mode=context.config.execution_mode,
        metadata=dict(context.metadata) if context.metadata else {},
    )


# ---------------------------------------------------------------------------
# Inter-agent envelope factories
# ---------------------------------------------------------------------------


def make_task_envelope(
    *,
    sender_id: str,
    recipient_id: str,
    payload: dict[str, Any],
    trace_id: str,
    session_id: str,
    report: ExecutionReport | None = None,
    protocol_version: str | None = None,
) -> AgentEnvelope:
    """Build a task envelope from leader to subagent."""
    return _build_envelope(
        message_type=MessageType.TASK,
        trace_id=trace_id,
        session_id=session_id,
        sender=AgentIdentity(agent_id=sender_id, role=AgentRole.LEADER),
        recipient=AgentIdentity(agent_id=recipient_id, role=AgentRole.SUBAGENT),
        payload=payload,
        report=report,
        protocol_version=protocol_version,
    )


def make_result_envelope(
    *,
    sender_id: str,
    recipient_id: str,
    result_payload: dict[str, Any],
    trace_id: str,
    session_id: str,
    report: ExecutionReport | None = None,
    protocol_version: str | None = None,
) -> AgentEnvelope:
    """Build a result envelope from subagent to leader."""
    return _build_envelope(
        message_type=MessageType.RESULT,
        trace_id=trace_id,
        session_id=session_id,
        sender=AgentIdentity(agent_id=sender_id, role=AgentRole.SUBAGENT),
        recipient=AgentIdentity(agent_id=recipient_id, role=AgentRole.LEADER),
        payload=result_payload,
        report=report,
        protocol_version=protocol_version,
    )


def make_error_envelope(
    *,
    sender_id: str,
    recipient_id: str,
    error_message: str,
    error_code: str,
    trace_id: str,
    session_id: str,
    sender_role: AgentRole = AgentRole.SUBAGENT,
    recipient_role: AgentRole = AgentRole.LEADER,
    report: ExecutionReport | None = None,
    protocol_version: str | None = None,
) -> AgentEnvelope:
    """Build an error envelope."""
    return _build_envelope(
        message_type=MessageType.ERROR,
        trace_id=trace_id,
        session_id=session_id,
        sender=AgentIdentity(agent_id=sender_id, role=sender_role),
        recipient=AgentIdentity(agent_id=recipient_id, role=recipient_role),
        payload={"error": error_message, "error_code": error_code},
        report=report,
        protocol_version=protocol_version,
    )


def make_control_envelope(
    *,
    sender_id: str,
    recipient_id: str,
    command: ControlCommand,
    trace_id: str,
    session_id: str,
    sender_role: AgentRole = AgentRole.LEADER,
    recipient_role: AgentRole = AgentRole.SUBAGENT,
    report: ExecutionReport | None = None,
    protocol_version: str | None = None,
) -> AgentEnvelope:
    """Build a control envelope (e.g. cancel signal)."""
    return _build_envelope(
        message_type=MessageType.CONTROL,
        trace_id=trace_id,
        session_id=session_id,
        sender=AgentIdentity(agent_id=sender_id, role=sender_role),
        recipient=AgentIdentity(agent_id=recipient_id, role=recipient_role),
        payload={"command": command},
        report=report,
        protocol_version=protocol_version,
    )


def _build_envelope(
    *,
    message_type: MessageType,
    trace_id: str,
    session_id: str,
    sender: AgentIdentity,
    recipient: AgentIdentity,
    payload: dict[str, Any],
    report: ExecutionReport | None = None,
    protocol_version: str | None = None,
) -> AgentEnvelope:
    version = protocol_version or PROTOCOL_VERSION
    envelope = AgentEnvelope(
        protocol_version=version,
        message_type=message_type,
        trace_id=trace_id,
        session_id=session_id,
        sender=sender,
        recipient=recipient,
        payload=payload,
        meta=EnvelopeMeta(protocol_version=version),
    )
    if report is not None:
        envelope.apply_version_compatibility(report)
    return envelope


def make_manual_compact_tool_definition(
    *,
    tool_name: str = MANUAL_COMPACT_TOOL_NAME,
) -> ToolDefinition:
    """Create the standard core tool definition for manual context compaction."""
    return ToolDefinition(
        name=tool_name,
        description=(
            "Compact the current conversation context into a structured summary "
            "before continuing with the task."
        ),
        parameters=ToolParameterSchema(
            properties={
                "reason": {
                    "type": "string",
                    "description": "Why compaction is needed right now.",
                }
            }
        ),
        strict=False,
    )


def register_manual_compact_tool(
    registry: ToolRegistry,
    *,
    tool_name: str = MANUAL_COMPACT_TOOL_NAME,
) -> ToolDefinition:
    """Register the standard core tool that requests manual compaction.

    The handler does not perform compaction itself. Instead it emits a
    ``ToolResult`` metadata flag that the orchestrator consumes after the
    tool step, preserving the immutable loop structure and avoiding
    branching on tool names inside the loop.
    """

    definition = make_manual_compact_tool_definition(tool_name=tool_name)

    async def _handler(arguments: dict[str, Any], context: Any) -> ToolResult:
        _ = context
        reason = arguments.get("reason")
        metadata: dict[str, Any] = {ToolResultMeta.MANUAL_COMPACT_REQUESTED: True}
        if isinstance(reason, str) and reason.strip():
            metadata[ToolResultMeta.MANUAL_COMPACT_REASON] = reason.strip()
        return ToolResult(
            tool_call_id="",
            tool_name=tool_name,
            content="[manual compact requested]",
            metadata=metadata,
        )

    registry.register(definition, _handler, tags=["core", "compression"])
    return definition


def make_load_skill_tool_definition(
    *,
    tool_name: str = LOAD_SKILL_TOOL_NAME,
) -> ToolDefinition:
    """Create the standard core tool definition for lazy skill loading."""
    return ToolDefinition(
        name=tool_name,
        description=(
            "Load full instructions for a named skill from the injected skill index. "
            "Use only when additional task-specific guidance is needed."
        ),
        parameters=ToolParameterSchema(
            properties={
                "name": {
                    "type": "string",
                    "description": "Skill name from the available skills index.",
                }
            },
            required=["name"],
            additionalProperties=False,
        ),
        strict=True,
    )


def register_load_skill_tool(
    registry: ToolRegistry,
    *,
    skill_manager: SkillManager,
    max_skill_loads_per_run: int,
    tool_name: str = LOAD_SKILL_TOOL_NAME,
) -> ToolDefinition:
    """Register the built-in core tool that loads skill bodies lazily."""
    definition = make_load_skill_tool_definition(tool_name=tool_name)

    async def _handler(arguments: dict[str, Any], context: Any) -> ToolResult:
        name_raw = arguments.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            return ToolResult(
                tool_call_id="",
                tool_name=tool_name,
                content="[ERROR: 'name' must be a non-empty string]",
                is_error=True,
            )

        skill_name = name_raw.strip()
        session_id = getattr(context, "session_id", "")
        agent_id = getattr(context, "agent_id", "")
        if not session_id or not agent_id:
            return ToolResult(
                tool_call_id="",
                tool_name=tool_name,
                content="[ERROR: missing session/agent context for skill loading]",
                is_error=True,
            )

        already_loaded = await skill_manager.is_loaded(skill_name, session_id)
        loaded_count = await skill_manager.get_load_count(session_id)
        if not already_loaded and loaded_count >= max_skill_loads_per_run:
            return ToolResult(
                tool_call_id="",
                tool_name=tool_name,
                content=(
                    f"[ERROR: skill load budget exceeded "
                    f"({loaded_count}/{max_skill_loads_per_run})]"
                ),
                is_error=True,
                metadata={
                    ToolResultMeta.SKILL_BUDGET_EXCEEDED: True,
                    ToolResultMeta.SKILL_NAME: skill_name,
                    ToolResultMeta.SKILL_LOAD_COUNT: loaded_count,
                    ToolResultMeta.MAX_SKILL_LOADS_PER_RUN: max_skill_loads_per_run,
                },
            )

        max_chars = max(
            500,
            int(getattr(context, "metadata", {}).get("skill_load_max_chars", 8_000)),
        )
        loaded = await skill_manager.load_skill(
            skill_name,
            agent_id=agent_id,
            max_chars=max_chars,
        )
        await skill_manager.mark_loaded(skill_name, session_id)
        payload = {
            "name": loaded.name,
            "body": loaded.body,
            "manifest": loaded.manifest.model_dump(exclude_none=True),
            "estimated_tokens": loaded.estimated_tokens,
            "truncated": loaded.truncated,
            "from_cache": loaded.from_cache,
        }
        return ToolResult(
            tool_call_id="",
            tool_name=tool_name,
            content=json.dumps(payload, ensure_ascii=True),
            metadata={
                ToolResultMeta.SKILL_NAME: loaded.name,
                ToolResultMeta.SKILL_FROM_CACHE: loaded.from_cache,
                ToolResultMeta.SKILL_TRUNCATED: loaded.truncated,
            },
        )

    registry.register(definition, _handler, tags=["core", "skills"])
    return definition


# ---------------------------------------------------------------------------
# Infrastructure factories
# ---------------------------------------------------------------------------


class CoreFactory:
    """Aggregated factory for core infrastructure objects.

    Service typically creates one CoreFactory and uses it for DI setup.
    Concrete persistence backends remain a service concern; the factory wires
    runtime boundaries such as registries and hooks, not Storage adapters.
    """

    def __init__(self) -> None:
        self._hook_manager = HookManager(create_plugin_manager())
        self._tool_registry = ToolRegistry(hook_manager=self._hook_manager)
        self._agent_registry = AgentRegistry()
        self._strategy_registry = StrategyRegistry()

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    @property
    def strategy_registry(self) -> StrategyRegistry:
        return self._strategy_registry

    @property
    def hook_manager(self) -> HookManager:
        return self._hook_manager

    def register_plugin(self, plugin: object) -> None:
        """Register a pluggy plugin with the hook manager."""
        self._hook_manager.register(plugin)

    def register_manual_compact_tool(
        self,
        *,
        tool_name: str = MANUAL_COMPACT_TOOL_NAME,
    ) -> ToolDefinition:
        """Register the built-in manual compaction tool in the factory registry."""
        return register_manual_compact_tool(self._tool_registry, tool_name=tool_name)

    def build_agent_context(
        self,
        *,
        config: AgentConfig,
        session_id: str | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
        parent_agent_id: str | None = None,
        allowed_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        message_history_ref: str | None = None,
        execution_metadata_ref: str | None = None,
    ) -> AgentContext:
        """Convenience: build AgentContext through this factory."""
        return make_agent_context(
            config=config,
            session_id=session_id,
            trace_id=trace_id,
            request_id=request_id,
            parent_agent_id=parent_agent_id,
            allowed_paths=allowed_paths,
            metadata=metadata,
            message_history_ref=message_history_ref,
            execution_metadata_ref=execution_metadata_ref,
        )

    def build_orchestrator(
        self,
        *,
        llm_client: LLMClient,
        tool_executor: ToolExecutor | None = None,
        planning_strategy: PlanningStrategy | None = None,
        planning_policy: PlanningPolicy | None = None,
        parallel_execution_policy: ParallelExecutionPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
        shell_executor: ShellExecutor | None = None,
        shell_safety_policy: ShellSafetyPolicy | None = None,
        skill_manager: SkillManager | None = None,
        compressor: "ContextCompressor | None" = None,
        state_manager: StateManager | None = None,
        subagent_selection_policy: SubagentSelectionPolicy | None = None,
        selection_policy: SubagentSelectionPolicy | None = None,
        transport: Transport | None = None,
        workflow_engine: WorkflowEngine | None = None,
        telemetry_collector: TelemetryCollector | None = None,
        timeout_policy: TimeoutPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        event_bus: "EventBus | None" = None,
    ) -> "AgentOrchestrator":
        """Build an AgentOrchestrator with factory-managed registries and hooks.

        This provides a single DI entrypoint for wiring runtime adapters/policies
        without touching core internals.
        """
        from .orchestrator import AgentOrchestrator

        return AgentOrchestrator(
            llm_client=llm_client,
            tool_executor=tool_executor,
            tool_registry=self._tool_registry,
            agent_registry=self._agent_registry,
            hook_manager=self._hook_manager,
            event_bus=event_bus,
            planning_strategy=planning_strategy,
            planning_policy=planning_policy,
            parallel_execution_policy=parallel_execution_policy,
            execution_policy=execution_policy,
            shell_executor=shell_executor,
            shell_safety_policy=shell_safety_policy,
            skill_manager=skill_manager,
            compressor=compressor,
            state_manager=state_manager,
            subagent_selection_policy=subagent_selection_policy,
            selection_policy=selection_policy,
            transport=transport,
            workflow_engine=workflow_engine,
            telemetry_collector=telemetry_collector,
            timeout_policy=timeout_policy,
            retry_policy=retry_policy,
        )
