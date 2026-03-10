"""Protocol definitions for Protocore.

All external dependencies are expressed through typing.Protocol — no hard
vendor imports allowed here. Concrete adapters live in lan-agent-service or
integrations.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .constants import SUBAGENT_CAPABILITY_SELECTION_PROMPT_TEMPLATE
from .types import (
    AgentContext,
    AgentEnvelope,
    ApiMode,
    ExecutionReport,
    LLMUsage,
    Message,
    PlanArtifact,
    PolicyDecision,
    Result,
    SkillIndexEntry,
    SkillLoadResult,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellToolConfig,
    SessionSnapshot,
    StreamEvent,
    SubagentResult,
    ToolContext,
    ToolDefinition,
    ToolResult,
    WorkflowDefinition,
)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Adapter for any OpenAI-compatible LLM endpoint.

    Default implementation uses Responses API. api_mode on AgentConfig controls
    which API surface is used (responses | chat_completions).

    Internal canonical naming uses ``tool_call_id`` / ``ToolCall.id`` even when
    a provider surface emits Responses-style ``call_id`` fields. Implementations
    should normalize provider-specific IDs before returning ``Message`` objects.
    """

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = True,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> Message:
        """Send messages and return the assistant reply.

        Implementations may accept optional transport-neutral kwargs such as
        ``logging_context`` as long as those fields are not forwarded to the
        provider API verbatim.
        """
        ...

    async def complete_structured(
        self,
        *,
        messages: list[Message],
        schema: type,  # Pydantic BaseModel subclass
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> Any:
        """Structured output call — returns validated pydantic instance."""
        ...

    def stream_with_tools(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_registry: Any | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_mode: ApiMode | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Return an async iterator of structured stream events.

        Callers consume the returned iterator with ``async for`` and should not
        ``await`` the method result separately. Implementations commonly expose
        this as an async generator.
        """
        ...


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolExecutor(Protocol):
    """Executes tool calls and returns results."""

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Run a single tool call."""
        ...

    def list_tools(self) -> list[ToolDefinition]:
        """Return available tools for this executor."""
        ...


# ---------------------------------------------------------------------------
# Shell Executor + Safety Policy
# ---------------------------------------------------------------------------


@runtime_checkable
class ShellExecutor(Protocol):
    """Executes normalized shell requests inside a service-provided sandbox."""

    async def execute(
        self,
        request: ShellExecutionRequest,
        context: ToolContext,
        capability: ShellToolConfig,
    ) -> ShellExecutionResult:
        """Run a shell command using the service runtime."""
        ...


@runtime_checkable
class ShellSafetyPolicy(Protocol):
    """Allow / deny / confirm shell execution requests before dispatch.

    Implementations may additionally expose debugging helpers such as
    ``explain_decision()`` for operator-facing audit trails.
    """

    async def evaluate(
        self,
        request: ShellExecutionRequest,
        context: ToolContext,
        capability: ShellToolConfig,
    ) -> PolicyDecision:
        """Return ALLOW, DENY, or CONFIRM for a shell request."""
        ...


# ---------------------------------------------------------------------------
# State Manager (session boundary)
# ---------------------------------------------------------------------------


@runtime_checkable
class StateManager(Protocol):
    """Persistent state adapter.

    Core never holds concrete storage; service provides this orchestration-facing
    boundary for session snapshots and execution reports. A service may
    implement StateManager on top of a generic Storage adapter or repository.

    Call points used by the core runtime:
    - before execution: ``save_session_snapshot``
    - after execution: ``update_session_snapshot``
    - after execution: ``save_execution_report``
    - loading/resume paths: ``load_session_snapshot``
    """

    async def get(self, key: str) -> Any:
        """Retrieve value by key. Return None if absent."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store value. Optional TTL in seconds."""
        ...

    async def delete(self, key: str) -> None:
        """Remove key."""
        ...

    async def save_session_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Persist full session snapshot."""
        ...

    async def load_session_snapshot(self, session_id: str) -> SessionSnapshot | None:
        """Load session snapshot or return None if not found."""
        ...

    async def update_session_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Persist a full post-run snapshot replacement.

        Implementations must treat ``snapshot`` as authoritative and preserve
        caller-supplied ``created_at`` rather than silently resetting it.
        """
        ...

    async def save_execution_report(self, report: ExecutionReport) -> None:
        """Persist execution report after each run."""
        ...


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Message delivery adapter (RabbitMQ, Kafka, HTTP, in-process, …)."""

    async def send(self, envelope: AgentEnvelope, destination: str) -> None:
        """Deliver envelope to named destination."""
        ...

    async def receive(self, source: str) -> AsyncIterator[AgentEnvelope]:
        """Consume envelopes from named source."""
        ...


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@runtime_checkable
class Storage(Protocol):
    """Generic backing store adapter (Postgres, MongoDB, Redis, …).

    This protocol exists as a replaceability boundary for service/repository
    layers. ``AgentOrchestrator`` intentionally depends on narrower runtime
    contracts such as ``StateManager`` instead of a generic Storage surface.
    """

    async def query(self, collection: str, query: dict[str, Any]) -> list[dict[str, Any]]:
        ...

    async def insert(self, collection: str, data: dict[str, Any]) -> str:
        ...

    async def update(self, collection: str, id: str, data: dict[str, Any]) -> None:
        ...

    async def delete(self, collection: str, id: str) -> None:
        ...


# ---------------------------------------------------------------------------
# Compression Strategy
# ---------------------------------------------------------------------------


@runtime_checkable
class CompressionStrategy(Protocol):
    """Runtime compressor contract used by the orchestrator."""

    async def apply_auto(
        self,
        messages: list[Message],
        config: Any,  # AgentConfig — avoid circular at protocol level
        *,
        precomputed_tokens: int | None = None,
        run_kind: Any = None,
        event_bus: Any = None,
        context: Any = None,
    ) -> tuple[list[Message], Any, bool]:
        """Apply auto-compaction and return (messages, summary, parse_ok)."""
        ...

    async def apply_manual(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        config: Any = None,
        run_kind: Any = None,
        event_bus: Any = None,
        context: Any = None,
    ) -> tuple[list[Message], Any]:
        """Apply manual compaction and return (messages, summary)."""
        ...


# ---------------------------------------------------------------------------
# Skill Manager (IDEA-007/017)
# ---------------------------------------------------------------------------


@runtime_checkable
class SkillManager(Protocol):
    """Manages skill catalog access and lazy loading."""

    async def get_index(
        self,
        agent_id: str,
        skill_names: list[str],
        max_chars: int = 1500,
    ) -> list[SkillIndexEntry]:
        """Return compact index entries visible to this agent."""
        ...

    async def load_skill(
        self,
        name: str,
        agent_id: str,
        max_chars: int = 8000,
    ) -> SkillLoadResult:
        """Load full skill body for runtime usage."""
        ...

    async def mark_loaded(self, name: str, session_id: str) -> None:
        """Record that skill was loaded in this session."""
        ...

    async def is_loaded(self, name: str, session_id: str) -> bool:
        """Return whether skill is already loaded in session cache."""
        ...

    async def get_load_count(self, session_id: str) -> int:
        """Return number of loaded skills in this session."""
        ...


# ---------------------------------------------------------------------------
# Orchestration Strategy (execution modes)
# ---------------------------------------------------------------------------


@runtime_checkable
class OrchestrationStrategy(Protocol):
    """Leader / bypass / parallel orchestration strategy.

    Each mode (leader, bypass, auto-select, parallel) is a concrete
    implementation of this protocol registered by the service.
    """

    async def execute(
        self,
        task: str,
        context: AgentContext,
    ) -> tuple[Result, ExecutionReport]:
        """Execute one strategy turn and always return ``(Result, ExecutionReport)``.

        Implementations must not mutate ``context.config`` in place and must
        finalize the returned report on every terminal path, including partial
        or failed execution.
        """
        ...


# ---------------------------------------------------------------------------
# Planning Policy + Strategy (planning gate)
# ---------------------------------------------------------------------------


@runtime_checkable
class PlanningPolicy(Protocol):
    """Decide when planning is mandatory and whether bypass is allowed."""

    async def should_plan(self, context: AgentContext) -> bool:
        """Return True when leader execution must pass through planning.

        The decision should be deterministic for the provided ``context`` and
        should not have side effects.
        """
        ...

    async def allow_bypass(self, context: AgentContext) -> bool:
        """Return True when explicit bypass mode is allowed for this request.

        Returning False means the orchestrator must route the run through its
        normal planning / delegation path instead of executing directly.
        """
        ...


@runtime_checkable
class PlanningStrategy(Protocol):
    """Build or update a plan before leader dispatch."""

    async def build_plan(
        self,
        task: str,
        context: AgentContext,
        llm_client: LLMClient,
    ) -> PlanArtifact:
        """Return a serialisable plan artifact for the current task.

        The returned artifact must be stable enough to persist inside
        ``ExecutionReport.plan_artifact`` and safe to round-trip through JSON.
        """
        ...

    async def update_plan(
        self,
        plan: PlanArtifact,
        context: AgentContext,
        llm_client: LLMClient,
    ) -> PlanArtifact:
        """Update an existing plan given new context.

        Implementations must preserve the semantic identity of ``plan`` instead
        of returning unrelated free-form text.
        """
        ...


class NoOpPlanningStrategy:
    """Minimal planning strategy that stores the incoming task verbatim.

    Useful when LEADER mode requires a planning strategy but you do not need
    LLM-generated steps yet.

    Example:
        >>> from protocore import NoOpPlanningStrategy
        >>> planning = NoOpPlanningStrategy()
    """

    async def build_plan(
        self,
        task: str,
        context: AgentContext,
        llm_client: LLMClient,
    ) -> PlanArtifact:
        """Build a no-op plan where ``raw_plan`` equals ``task``."""
        _ = (context, llm_client)
        # Provide built-in planning strategy for LEADER mode.
        return PlanArtifact(raw_plan=task)

    async def update_plan(
        self,
        plan: PlanArtifact,
        context: AgentContext,
        llm_client: LLMClient,
    ) -> PlanArtifact:
        """Return the existing plan unchanged."""
        _ = (context, llm_client)
        return plan


# ---------------------------------------------------------------------------
# Subagent Selection Policy (auto-select mode)
# ---------------------------------------------------------------------------


@runtime_checkable
class SubagentSelectionPolicy(Protocol):
    """Choose which subagent to route a task to (auto-select mode)."""

    async def select(
        self,
        task: str,
        available_agents: list[str],
        context: AgentContext,
    ) -> str:
        """Return agent_id of the selected subagent."""
        ...


class FirstAvailablePolicy:
    """Simple fallback policy that selects the first available subagent.

    Example:
        >>> from protocore import FirstAvailablePolicy
        >>> policy = FirstAvailablePolicy()
    """

    async def select(
        self,
        task: str,
        available_agents: list[str],
        context: AgentContext,
    ) -> str:
        """Select the first available subagent id."""
        _ = (task, context)
        # Built-in no-LLM selection fallback.
        if not available_agents:
            raise ValueError("No subagents available")
        return available_agents[0]


class CapabilityBasedSelectionPolicy:
    """LLM-based policy that picks a subagent from capabilities/descriptions.

    Example:
        >>> from protocore import CapabilityBasedSelectionPolicy
        >>> policy = CapabilityBasedSelectionPolicy(
        ...     llm_client=my_llm,
        ...     agent_descriptions={"math": "Arithmetic and equations"},
        ... )
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        agent_descriptions: dict[str, str] | None = None,
        prompt_template: str = SUBAGENT_CAPABILITY_SELECTION_PROMPT_TEMPLATE,
        fallback_policy: SubagentSelectionPolicy | None = None,
        confidence_threshold: float = 0.55,
    ) -> None:
        self._llm_client = llm_client
        self._agent_descriptions = dict(agent_descriptions or {})
        self._prompt_template = prompt_template
        self._fallback_policy = fallback_policy or FirstAvailablePolicy()
        self._confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.last_selection_debug: dict[str, Any] = {}

    async def select(
        self,
        task: str,
        available_agents: list[str],
        context: AgentContext,
    ) -> str:
        """Select a subagent via LLM with deterministic fallback."""
        if not available_agents:
            raise ValueError("No subagents available")

        agent_lines = []
        for idx, agent_id in enumerate(available_agents, start=1):
            description = self._agent_descriptions.get(agent_id, "No description provided.")
            agent_lines.append(f"{idx}. {agent_id}: {description}")
        prompt = self._prompt_template.format(task=task, available_agents="\n".join(agent_lines))

        response = await self._llm_client.complete(
            messages=[Message(role="user", content=prompt)],
            model=context.config.model,
            temperature=0.0,
            max_tokens=128,
            stream=False,
            api_mode=context.config.api_mode,
        )
        selected, confidence, reason = self._extract_agent_id(response.content, available_agents)
        if (
            selected is not None
            and confidence >= self._confidence_threshold
            and self._looks_compatible_with_task(task, selected)
        ):
            self.last_selection_debug = {
                "selected_agent": selected,
                "confidence": confidence,
                "reason": reason,
                "used_fallback": False,
            }
            return selected
        fallback_selected = await self._fallback_policy.select(task, available_agents, context)
        self.last_selection_debug = {
            "selected_agent": fallback_selected,
            "confidence": confidence,
            "reason": reason or "fallback_policy",
            "used_fallback": True,
            "llm_candidate": selected,
        }
        return fallback_selected

    @staticmethod
    def _extract_agent_id(
        content: str | list[Any] | None,
        available_agents: list[str],
    ) -> tuple[str | None, float, str]:
        if not isinstance(content, str):
            return None, 0.0, "non_text_response"
        raw = content.strip()
        if not raw:
            return None, 0.0, "empty_response"
        if raw in available_agents:
            return raw, 1.0, "exact_text_match"

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            candidate = parsed.get("agent_id")
            if isinstance(candidate, str) and candidate in available_agents:
                raw_confidence = parsed.get("confidence", 0.75)
                confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 0.75
                reason = str(parsed.get("reason", "json.agent_id"))
                return candidate, max(0.0, min(1.0, confidence)), reason
        elif isinstance(parsed, str) and parsed in available_agents:
            return parsed, 0.7, "json_string_match"

        for line in raw.splitlines():
            candidate = line.strip().strip("`\"'")
            if candidate in available_agents:
                return candidate, 0.6, "line_match"
        return None, 0.0, "no_match"

    def _looks_compatible_with_task(self, task: str, selected_agent: str) -> bool:
        description = self._agent_descriptions.get(selected_agent, "").strip().lower()
        if not description:
            return True
        task_terms = {
            token
            for token in re.split(r"[^a-z0-9_]+", task.lower())
            if len(token) >= 4
        }
        if not task_terms:
            return True
        overlap = sum(1 for token in task_terms if token in description)
        return overlap >= 1


def accumulate_usage_from_llm_calls(
    llm_calls: Iterable[LLMUsage | Message | dict[str, Any] | None],
) -> tuple[int, int, int, int]:
    """Aggregate usage from workflow-engine-owned LLM calls.

    Accepts ``LLMUsage`` objects, ``Message`` instances with ``usage``, or raw
    dict payloads shaped like ``{"input_tokens": ..., "output_tokens": ...}``.
    Returns ``(input_tokens, output_tokens, cached_tokens, reasoning_tokens)``.
    """
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    for item in llm_calls:
        usage: LLMUsage | None = None
        if item is None:
            continue
        if isinstance(item, LLMUsage):
            usage = item
        elif isinstance(item, Message):
            usage = item.usage
        elif isinstance(item, dict):
            try:
                usage = LLMUsage.model_validate(item)
            except Exception:
                usage = None
        if usage is None:
            continue
        input_tokens += max(usage.input_tokens, 0)
        output_tokens += max(usage.output_tokens, 0)
        cached_tokens += max(usage.cached_tokens, 0)
        reasoning_tokens += max(usage.reasoning_tokens, 0)
    return input_tokens, output_tokens, cached_tokens, reasoning_tokens


# ---------------------------------------------------------------------------
# Parallel Execution Policy
# ---------------------------------------------------------------------------


@runtime_checkable
class ParallelExecutionPolicy(Protocol):
    """Control parallel subagent concurrency and merge semantics."""

    @property
    def max_concurrency(self) -> int:
        """Maximum parallel subagent runs."""
        ...

    @property
    def timeout_seconds(self) -> float:
        """Per-subagent timeout."""
        ...

    @property
    def cancellation_mode(self) -> str:
        """Cancellation semantics (for example: propagate, graceful)."""
        ...

    async def merge_results(
        self,
        results: list[SubagentResult | None],
        agent_ids: list[str],
    ) -> SubagentResult:
        """Deterministically merge parallel subagent results into one.

        ``results`` and ``agent_ids`` are positional peers; implementations
        must preserve ordering semantics and tolerate ``None`` entries for
        missing/failed children.
        """
        ...


# ---------------------------------------------------------------------------
# Execution Policy (safety)
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionPolicy(Protocol):
    """Allow / deny / require confirmation for tool calls."""

    async def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> PolicyDecision:
        """Return ALLOW, DENY, or CONFIRM.

        The policy result is authoritative for the current dispatch attempt and
        should not rely on mutating ``arguments`` or ``context`` as an output
        channel.
        """
        ...


# ---------------------------------------------------------------------------
# Workflow Engine
# ---------------------------------------------------------------------------


@runtime_checkable
class WorkflowEngine(Protocol):
    """Execute a workflow DAG. Integration point for LangGraph adapter.

    Implementations are responsible for filling ``ExecutionReport`` usage
    fields. Use ``accumulate_usage_from_llm_calls(...)`` when a workflow engine
    makes multiple LLM calls and wants to aggregate them into one report.
    """

    async def run(
        self,
        workflow: WorkflowDefinition,
        context: AgentContext,
    ) -> tuple[Result, ExecutionReport]:
        """Execute all nodes respecting DAG dependency order."""
        ...


# ---------------------------------------------------------------------------
# Telemetry Collector
# ---------------------------------------------------------------------------


@runtime_checkable
class TelemetryCollector(Protocol):
    """Receive lifecycle events from orchestrator for external aggregation."""

    async def record_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        report: ExecutionReport,
    ) -> None:
        """Record a single telemetry event."""
        ...


# ---------------------------------------------------------------------------
# Timeout / Retry boundary contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class TimeoutPolicy(Protocol):
    """Per-call timeout contract."""

    def get_timeout(self, operation: str) -> float:
        """Return timeout in seconds for the named operation."""
        ...


@runtime_checkable
class RetryPolicy(Protocol):
    """Retry strategy contract."""

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Return True if a retry should be attempted."""
        ...

    def delay_seconds(self, attempt: int) -> float:
        """Return wait time before next retry."""
        ...
