"""Core types and pydantic models for Protocore.

All data structures shared between core and service. No business logic here.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
import warnings
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, NotRequired, SupportsIndex, TypeAlias, TypedDict, TypeVar

from .constants import (
    AUTO_COMPACT_MAX_TOKENS,
    AUTO_COMPACT_TEMPERATURE,
    COMPACTION_SUMMARY_MARKER,
    DEFAULT_AUTO_COMPACT_KEEP_TRAILING,
    DEFAULT_AUTO_COMPACT_TIMEOUT_SECONDS,
    DEFAULT_AUTO_COMPACT_THRESHOLD,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SKILL_LOADS_PER_RUN,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOOL_RESULT_SIZE,
    DEFAULT_MICRO_COMPACT_KEEP_RECENT,
    DEFAULT_MODEL,
    DEFAULT_SHELL_DESCRIPTION,
    DEFAULT_SHELL_MAX_COMMAND_LENGTH,
    DEFAULT_SHELL_MAX_TIMEOUT_MS,
    DEFAULT_SHELL_TIMEOUT_MS,
    DEFAULT_SHELL_TOOL_NAME,
    DEFAULT_SKILL_INDEX_MAX_CHARS,
    DEFAULT_SKILL_LOAD_MAX_CHARS,
    FORBIDDEN_PAYLOAD_KEYS,
    MAX_ARTIFACTS,
    MAX_ENVELOPE_PAYLOAD_CHARS,
    MAX_ENVELOPE_PAYLOAD_DEPTH,
    MAX_FILES_CHANGED,
    MAX_STRUCTURED_JSON_CHARS,
    MAX_SUBAGENT_ERRORS,
    MAX_SUMMARY_CHARS,
    PROTOCOL_VERSION,
    QWEN_NO_THINKING_EXTRA_BODY,
    ThinkingProfileRegistry,
)
from .json_utils import structured_json_candidates

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

logger = logging.getLogger(__name__)

STRUCTURED_USAGE_ATTR = "__protocore_usage__"
_StructuredModelT = TypeVar("_StructuredModelT", bound=BaseModel)


def _coerce_message(value: Any, *, field_name: str = "messages") -> "Message":
    if isinstance(value, Message):
        return value
    raise TypeError(
        f"{field_name} items must be Message instances; "
        f"got {type(value).__name__}. Convert Result objects explicitly, "
        "for example Message(role='assistant', content=result.content)."
    )


class MessageList(list["Message"]):
    """List wrapper that validates Message instances on mutation."""

    def __init__(self, iterable: list["Message"] | None = None) -> None:
        super().__init__()
        if iterable:
            self.extend(iterable)

    def append(self, value: Any) -> None:
        super().append(_coerce_message(value))

    def extend(self, values: Any) -> None:
        super().extend(_coerce_message(value) for value in values)

    def insert(self, index: SupportsIndex, value: Any) -> None:
        super().insert(index, _coerce_message(value))

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            super().__setitem__(index, [_coerce_message(item) for item in value])
            return
        super().__setitem__(index, _coerce_message(value))

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        list_schema = handler.generate_schema(list["Message"])
        return core_schema.no_info_after_validator_function(cls, list_schema)


def get_text_content(message: "Message") -> str:
    """Return a plain-text view of ``Message.content`` for callers/UI helpers."""
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.type == "text" and part.text is not None:
            parts.append(part.text)
        elif part.type == "input_json" and part.json_data is not None:
            parts.append(json.dumps(part.json_data, ensure_ascii=False, sort_keys=True))
        elif part.type == "image_url":
            image_url = part.image_url or {}
            url = image_url.get("url")
            parts.append(url if isinstance(url, str) else "[image_url]")
        else:
            parts.append(f"[{part.type}]")
    return "\n".join(part for part in parts if part).strip()


def _validate_iso_datetime_string(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 datetime") from exc
    return value


def _json_serialize_strict(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise ValueError("Envelope payload must be JSON-serializable") from exc


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ApiMode(str, Enum):
    """LLM API mode selector."""

    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


class RunKind(str, Enum):
    """Whether this execution is a leader or subagent run."""

    LEADER = "leader"
    SUBAGENT = "subagent"


class AgentRole(str, Enum):
    """Identity role for message routing in envelopes (sender/recipient).

    Distinct from RunKind which tracks execution context (leader vs subagent run).
    Values intentionally match RunKind for serialization compatibility.
    """

    LEADER = "leader"
    SUBAGENT = "subagent"


class ExecutionMode(str, Enum):
    """Execution mode for orchestration."""

    LEADER = "leader"
    BYPASS = "bypass"
    AUTO_SELECT = "auto-select"
    PARALLEL = "parallel"
    TOOL_ORCHESTRATED = "tool_orchestrated"


class ThinkingProfilePreset(str, Enum):
    """Named generation presets for reasoning/instruct workloads."""

    THINKING_PLANNER = "thinking_planner"
    THINKING_PRECISE_CODING = "thinking_precise_coding"
    THINKING_ANALYTICAL = "thinking_analytical"
    INSTRUCT_TOOL_WORKER = "instruct_tool_worker"


class ThinkingRunPolicy(str, Enum):
    """Per-run override for enabling/disabling model reasoning."""

    AUTO = "auto"
    FORCE_ON = "force_on"
    FORCE_OFF = "force_off"


class TokenEstimatorProfile(str, Enum):
    """Which built-in token counting strategy to use."""

    AUTO = "auto"
    OPENAI = "openai"
    QWEN3 = "qwen3"
    HEURISTIC = "heuristic"


class ShellAccessMode(str, Enum):
    """Which agents may see and use the built-in shell tool."""

    DISABLED = "disabled"
    LEADER_ONLY = "leader_only"
    ALL_AGENTS = "all_agents"


class ShellToolProfile(str, Enum):
    """High-level shell capability preset exposed to policies/runtimes."""

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    FULL_ACCESS = "full_access"


class ShellSafetyMode(str, Enum):
    """Safety path selection for shell preflight and approval flow."""

    ENFORCED = "enforced"
    YOLO = "yolo"


class SkillTrustLevel(str, Enum):
    """Trust classification for skills."""

    MANAGED = "managed"
    PROJECT = "project"
    USER = "user"


class MessageType(str, Enum):
    """Types of inter-agent envelope messages."""

    TASK = "task"
    RESULT = "result"
    CONTROL = "control"
    ERROR = "error"


class ControlCommand(str, Enum):
    """Control-plane commands for inter-agent runtime coordination."""

    CANCEL = "cancel"


class SubagentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class StopReason(str, Enum):
    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_ITERATIONS = "max_iterations"
    TOOL_BUDGET_EXCEEDED = "tool_budget_exceeded"
    APPROVAL_REQUIRED = "approval_required"
    CANCELLED = "cancelled"
    ERROR = "error"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


class ToolContextMetadata(TypedDict, total=False):
    """Known keys stored in ``ToolContext.metadata``."""

    message_history_ref: str
    execution_metadata_ref: str
    request_id: str
    parent_agent_id: str
    skill_set: list[str]
    skill_load_max_chars: int
    max_skill_loads_per_run: int
    visible_tool_names: list[str]
    visible_tool_names_clipped: bool
    run_kind: str
    workspace_root: str


class AgentContextMetadata(TypedDict, total=False):
    """Known keys stored in ``AgentContext.metadata``."""

    active_child_agent_ids: list[str]
    _active_child_agent_ids: list[str]
    workflow_definition: dict[str, Any]
    auto_selected_agent: str
    parallel_agent_ids: list[str]
    plan_artifact: dict[str, Any]
    queue_wait_ms: float
    pending_shell_approval: dict[str, Any]
    shell_approval_decisions: dict[str, Any]
    shell_approval_rules: list[dict[str, Any]]


class ExecutionMetadata(TypedDict, total=False):
    """Known keys stored in persisted execution/session metadata."""

    status: str
    stop_reason: str | None
    warnings: list[str]
    pending_shell_approval: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Typed metadata keys
# ---------------------------------------------------------------------------


class ToolResultMeta(str, Enum):
    """Known keys for ``ToolResult.metadata``.

    Using these constants instead of bare strings prevents silent typos and
    gives IDE autocompletion.  The dict type stays ``dict[str, Any]`` for
    extensibility; these enums are just "well-known" keys.
    """

    def __str__(self) -> str:  # noqa: D105
        return self.value

    MANUAL_COMPACT_REQUESTED = "manual_compact_requested"
    MANUAL_COMPACT_REASON = "manual_compact_reason"
    SKILL_BUDGET_EXCEEDED = "skill_budget_exceeded"
    SKILL_NAME = "skill_name"
    SKILL_FROM_CACHE = "skill_from_cache"
    SKILL_TRUNCATED = "skill_truncated"
    SKILL_LOAD_COUNT = "skill_load_count"
    MAX_SKILL_LOADS_PER_RUN = "max_skill_loads_per_run"
    SHELL_RISK_FLAGS = "shell_risk_flags"


class ToolContextMeta(str, Enum):
    """Known keys for ``ToolContext.metadata``."""

    def __str__(self) -> str:  # noqa: D105
        return self.value

    MESSAGE_HISTORY_REF = "message_history_ref"
    EXECUTION_METADATA_REF = "execution_metadata_ref"
    REQUEST_ID = "request_id"
    PARENT_AGENT_ID = "parent_agent_id"
    SKILL_SET = "skill_set"
    SKILL_LOAD_MAX_CHARS = "skill_load_max_chars"
    MAX_SKILL_LOADS_PER_RUN = "max_skill_loads_per_run"
    VISIBLE_TOOL_NAMES = "visible_tool_names"
    VISIBLE_TOOL_NAMES_CLIPPED = "visible_tool_names_clipped"
    RUN_KIND = "run_kind"
    WORKSPACE_ROOT = "workspace_root"


class AgentContextMeta(str, Enum):
    """Known keys for ``AgentContext.metadata``."""

    def __str__(self) -> str:  # noqa: D105
        return self.value

    ACTIVE_CHILD_AGENT_IDS = "active_child_agent_ids"
    LEGACY_ACTIVE_CHILD_AGENT_IDS = "_active_child_agent_ids"
    WORKFLOW_DEFINITION = "workflow_definition"
    AUTO_SELECTED_AGENT = "auto_selected_agent"
    PARALLEL_AGENT_IDS = "parallel_agent_ids"
    PLAN_ARTIFACT = "plan_artifact"
    QUEUE_WAIT_MS = "queue_wait_ms"
    PENDING_SHELL_APPROVAL = "pending_shell_approval"
    SHELL_APPROVAL_DECISIONS = "shell_approval_decisions"
    SHELL_APPROVAL_RULES = "shell_approval_rules"


# ---------------------------------------------------------------------------
# Base message types
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """Single content part in a message (text or image)."""

    type: Literal["text", "image_url", "input_json"]
    text: str | None = None
    image_url: dict[str, str] | None = None  # {"url": "data:...", "detail": "auto"}
    json_data: dict[str, Any] | None = None

    model_config = {"extra": "forbid"}


class LLMUsage(BaseModel):
    """Token usage returned by LLM APIs."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    model_config = {"extra": "allow"}


class ToolCallFunction(BaseModel):
    name: str = Field(min_length=1)
    arguments: str = "{}"

    model_config = {"extra": "allow"}

    @field_validator("arguments", mode="before")
    @classmethod
    def normalize_arguments(cls, value: Any) -> str:
        if value in (None, ""):
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class ToolCall(BaseModel):
    """Canonical internal tool-call model.

    External providers may use ``call_id`` (Responses API). The model accepts
    both spellings and normalizes them to ``id`` internally.
    """

    id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("id", "call_id"),
    )
    type: Literal["function"] = "function"
    function: ToolCallFunction

    model_config = {"extra": "forbid"}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_openai_dict(self) -> dict[str, Any]:
        """Return a plain dict for OpenAI-compatible payload serialization."""
        return self.model_dump(exclude_none=True)

    @property
    def call_id(self) -> str:
        """Responses-API compatibility alias for ``id``."""
        return self.id


class Message(BaseModel):
    """Conversation message compatible with OpenAI message format.

    ``tool_call_id`` is the canonical internal name for tool-result linkage.
    Provider payloads using ``call_id`` are accepted on input and normalized.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tool_call_id", "call_id"),
    )
    name: str | None = None
    usage: LLMUsage | None = None

    model_config = {"extra": "allow", "validate_assignment": True}

    @field_validator("tool_calls", mode="before")
    @classmethod
    def normalize_tool_calls(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        if not isinstance(value, list):
            raise ValueError("tool_calls must be a list")
        normalized: list[ToolCall] = []
        for item in value:
            normalized.append(ToolCall.model_validate(item))
        return normalized

    @model_validator(mode="after")
    def validate_message_contract(self) -> "Message":
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages must include tool_call_id")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls are only allowed on assistant messages")
        return self

    @property
    def call_id(self) -> str | None:
        """Responses-API compatibility alias for ``tool_call_id``."""
        return self.tool_call_id


def attach_structured_usage(payload: Any, usage: LLMUsage | dict[str, Any] | None) -> Any:
    """Attach provider usage to a structured-output payload when possible."""
    if usage is None:
        return payload
    usage_model = usage if isinstance(usage, LLMUsage) else LLMUsage.model_validate(usage)
    if isinstance(payload, dict):
        enriched = dict(payload)
        enriched[STRUCTURED_USAGE_ATTR] = usage_model.model_dump(mode="json")
        return enriched
    try:
        object.__setattr__(payload, STRUCTURED_USAGE_ATTR, usage_model)
    except (AttributeError, TypeError):
        logger.debug(
            "attach_structured_usage failed for payload type=%s",
            type(payload).__name__,
        )
    return payload


def extract_structured_usage(payload: Any) -> LLMUsage | None:
    """Return usage previously attached to a structured-output payload."""
    if isinstance(payload, dict):
        usage = payload.get(STRUCTURED_USAGE_ATTR)
    else:
        usage = getattr(payload, STRUCTURED_USAGE_ATTR, None)
    if usage is None:
        return None
    if isinstance(usage, LLMUsage):
        return usage
    try:
        return LLMUsage.model_validate(usage)
    except ValidationError:
        logger.debug(
            "extract_structured_usage failed for payload type=%s",
            type(payload).__name__,
        )
        return None


class StreamDeltaEvent(TypedDict):
    """Incremental text/reasoning chunk emitted during streaming.

    Notes:
    - ``type`` is always ``"delta"`` (not ``"text"``).
    - ``kind`` tells whether ``text`` belongs to final answer text or reasoning.
    - ``provider_event_type`` is optional provider-specific telemetry.
    """

    type: Literal["delta"]
    text: str
    kind: Literal["text", "reasoning"]
    provider_event_type: NotRequired[str]


class StreamToolCallsEvent(TypedDict):
    """Finalized tool call list emitted when streaming includes tool calls."""

    type: Literal["tool_calls"]
    tool_calls: list[dict[str, Any]]


class StreamDoneEvent(TypedDict):
    """Terminal streaming event emitted exactly once at stream end.

    ``usage`` contains provider usage metrics when available.
    """

    type: Literal["done"]
    usage: dict[str, Any]


StreamEvent: TypeAlias = StreamDeltaEvent | StreamToolCallsEvent | StreamDoneEvent


def is_text_delta_event(event: StreamEvent) -> bool:
    """Return True when a stream event is a text delta chunk.

    Use this helper when handling streamed events so callers do not accidentally
    treat ``event["type"]`` as ``"text"``.
    """

    return event.get("type") == "delta" and event.get("kind") == "text"


def is_done_event(event: StreamEvent) -> bool:
    """Return True when a stream event is the terminal ``done`` event."""

    return event.get("type") == "done"


TokenEstimateFunc: TypeAlias = Callable[[str], int]


# ---------------------------------------------------------------------------
# Tool definitions and results
# ---------------------------------------------------------------------------


class ToolParameterSchema(BaseModel):
    """JSON Schema for tool parameters."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additional_properties: bool | None = Field(
        default=None,
        alias="additionalProperties",
    )

    model_config = {"extra": "allow", "populate_by_name": True}

    def to_openai_schema(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)


class ToolDefinition(BaseModel):
    """Tool schema sent to LLM (OpenAI function format)."""

    name: str
    description: str
    parameters: ToolParameterSchema = Field(default_factory=ToolParameterSchema)
    strict: bool = False
    filesystem_access: bool = False
    path_fields: list[str] = Field(default_factory=list)
    idempotent: bool = False

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_filesystem_contract(self) -> "ToolDefinition":
        if not self.filesystem_access:
            return self
        normalized = [field.strip() for field in self.path_fields if field.strip()]
        if normalized:
            self.path_fields = list(dict.fromkeys(normalized))
            return self
        canonical_path_fields = {
            "path",
            "paths",
            "file_path",
            "source_path",
            "destination_path",
            "target_path",
            "cwd",
            "root",
        }
        inferred = [
            key
            for key in self.parameters.properties
            if isinstance(key, str) and key in canonical_path_fields
        ]
        if not inferred:
            raise ValueError(
                "filesystem_access tools must declare path_fields or use canonical "
                "path parameter names"
            )
        self.path_fields = inferred
        return self

    def to_openai_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_openai_schema(),
                "strict": self.strict,
            },
        }

    @classmethod
    def simple(
        cls,
        *,
        name: str,
        description: str,
        params: Mapping[str, tuple[str, bool, str | None] | tuple[str, bool] | str],
        strict: bool = True,
        filesystem_access: bool = False,
        path_fields: list[str] | None = None,
        idempotent: bool = False,
        additional_properties: bool = False,
    ) -> "ToolDefinition":
        """Convenience builder for simple object-shaped tool schemas.

        ``params`` accepts either a raw JSON-schema type string or tuples in the form
        ``(json_type, required[, description])``.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, spec in params.items():
            if isinstance(spec, str):
                json_type = spec
                is_required = False
                param_description = None
            else:
                json_type = spec[0]
                is_required = bool(spec[1])
                param_description = spec[2] if len(spec) > 2 else None
            schema: dict[str, Any] = {"type": json_type}
            if param_description:
                schema["description"] = param_description
            properties[param_name] = schema
            if is_required:
                required.append(param_name)
        return cls(
            name=name,
            description=description,
            parameters=ToolParameterSchema(
                type="object",
                properties=properties,
                required=required,
                additionalProperties=additional_properties,
            ),
            strict=strict,
            filesystem_access=filesystem_access,
            path_fields=path_fields or [],
            idempotent=idempotent,
        )


class ShellExecutionRequest(BaseModel):
    """Normalized shell request passed to a ShellExecutor."""

    command: str = Field(min_length=1)
    cwd: str | None = None
    timeout_ms: int | None = Field(default=None, ge=1)
    env: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None

    model_config = {"extra": "forbid"}


class ShellExecutionResult(BaseModel):
    """Structured result returned by a ShellExecutor."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int
    duration_ms: float | None = None
    truncated: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ShellToolConfig(BaseModel):
    """Capability and guardrail settings for the built-in shell tool."""

    access_mode: ShellAccessMode = ShellAccessMode.DISABLED
    tool_name: str = DEFAULT_SHELL_TOOL_NAME
    profile: ShellToolProfile = ShellToolProfile.READ_ONLY
    safety_mode: ShellSafetyMode = ShellSafetyMode.ENFORCED
    description: str = DEFAULT_SHELL_DESCRIPTION
    default_timeout_ms: int = Field(default=DEFAULT_SHELL_TIMEOUT_MS, ge=1)
    max_timeout_ms: int = Field(default=DEFAULT_SHELL_MAX_TIMEOUT_MS, ge=1)
    max_command_length: int = Field(default=DEFAULT_SHELL_MAX_COMMAND_LENGTH, ge=1)
    env_allowlist: list[str] = Field(default_factory=list)
    allow_network: bool = False
    require_cwd: bool = False

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_timeout_bounds(self) -> "ShellToolConfig":
        if self.max_timeout_ms < self.default_timeout_ms:
            raise ValueError(
                "shell_tool_config.max_timeout_ms must be greater than or equal to "
                "shell_tool_config.default_timeout_ms"
            )
        return self

    def to_tool_definition(self) -> ToolDefinition:
        """Return the function schema exposed to the model."""
        return ToolDefinition(
            name=self.tool_name,
            description=(
                f"{self.description} Profile={self.profile.value}, "
                f"network_allowed={self.allow_network}, "
                f"safety_mode={self.safety_mode.value}."
            ),
            parameters=ToolParameterSchema(
                properties={
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory inside the sandbox.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional execution timeout in milliseconds.",
                    },
                    "env": {
                        "type": "object",
                        "description": (
                            "Optional environment variables. Only allowlisted keys "
                            "may be forwarded by the runtime."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of why this command is needed.",
                    },
                },
                required=["command"],
                additionalProperties=False,
            ),
            strict=True,
        )


class ShellCommandPlan(BaseModel):
    """Serializable pending shell command plan requiring external approval."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_call_id: str = ""
    tool_name: str = "shell_exec"
    command: str = Field(min_length=1)
    cwd: str | None = None
    timeout_ms: int | None = Field(default=None, ge=1)
    env: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    requested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    approval_status: Literal["pending", "approved", "rejected", "executed"] = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def transition_to(
        self,
        status: Literal["pending", "approved", "rejected", "executed"],
    ) -> "ShellCommandPlan":
        allowed_transitions = {
            "pending": {"approved", "rejected"},
            "approved": {"executed", "rejected"},
            "rejected": set(),
            "executed": set(),
        }
        if status == self.approval_status:
            return self
        if status not in allowed_transitions[self.approval_status]:
            raise ValueError(
                "invalid shell approval_status transition: "
                f"{self.approval_status} -> {status}"
            )
        self.approval_status = status
        return self


class ShellApprovalRule(BaseModel):
    """Session-scoped matcher for shell approvals."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name_pattern: str | None = None
    command_pattern: str = Field(min_length=1)
    cwd_pattern: str | None = None
    reason_pattern: str | None = None
    description: str | None = None
    added_via: Literal["preconfigured", "session_approval"] = "preconfigured"
    added_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("tool_name_pattern", "command_pattern", "cwd_pattern", "reason_pattern")
    @classmethod
    def validate_regex(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern: {exc}") from exc
        return value


class SkillManifest(BaseModel):
    """Parsed frontmatter metadata for a single skill."""

    name: str = Field(min_length=1)
    description: str = ""
    version: int = 1
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    trust_level: SkillTrustLevel = SkillTrustLevel.MANAGED
    body_size_bytes: int | None = None
    supporting_files: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class SkillIndexEntry(BaseModel):
    """Compact catalog entry injected into system prompt."""

    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    trust_level: SkillTrustLevel = SkillTrustLevel.MANAGED

    model_config = {"extra": "forbid"}


class SkillLoadResult(BaseModel):
    """Result returned by lazy load_skill call."""

    name: str
    body: str
    manifest: SkillManifest
    estimated_tokens: int = 0
    truncated: bool = False
    from_cache: bool = False

    model_config = {"extra": "forbid"}


class ToolContext(BaseModel):
    """Execution context passed to tool handlers (path isolation)."""

    allowed_paths: list[str] = Field(
        default_factory=list,
        description="FS paths the tool is allowed to access. Empty = no FS access.",
    )
    allow_symlinks: bool = Field(
        default=False,
        description=(
            "Whether path validation may follow symlink targets. Disabled by "
            "default for sandbox safety."
        ),
    )
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    # Tool handlers can read dispatcher-provided tool call id.
    tool_call_id: str = ""
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extensible tool-scoped metadata bag. Known keys are documented by "
            "ToolContextMeta / ToolContextMetadata."
        ),
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def for_manual_tests(
        cls,
        *,
        agent_id: str = "",
        session_id: str | None = None,
        trace_id: str | None = None,
        tool_call_id: str = "",
        allowed_paths: list[str] | None = None,
        allow_symlinks: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolContext":
        """Convenience constructor for shell/tool demos and manual tests."""
        return cls(
            session_id=session_id or str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            allowed_paths=list(allowed_paths or []),
            allow_symlinks=allow_symlinks,
            metadata=dict(metadata or {}),
        )


class ToolResult(BaseModel):
    """Result returned by a tool executor.

    ``tool_call_id`` is the canonical internal field. ``call_id`` is accepted
    as an input alias for compatibility with Responses-style payloads.
    When a tool is invoked through the orchestrator dispatcher, the framework
    auto-fills ``tool_call_id`` from the incoming tool call if the handler
    returns an empty value.
    """

    tool_call_id: str = Field(
        default="",
        validation_alias=AliasChoices("tool_call_id", "call_id"),
    )
    tool_name: str
    content: str
    is_error: bool = False
    prompt_injection_signal: bool = False  # flag suspicious content
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    artifact_summary: str | None = None
    next_recommended_tools: list[str] | None = None

    model_config = {"extra": "forbid"}

    @property
    def call_id(self) -> str:
        """Responses-API compatibility alias for ``tool_call_id``."""
        return self.tool_call_id


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Per-agent configuration knobs (optimised for local <80B models)."""

    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "agent"
    description: str = Field(
        default="",
        description=(
            "Short capability summary used for routing/delegation decisions. "
            "Keep it concise and focused on what this agent is good at."
        ),
    )
    role: AgentRole = AgentRole.SUBAGENT
    model: str = DEFAULT_MODEL
    compaction_model: str | None = Field(
        default=None,
        description=(
            "Optional cheaper/faster model used for auto/manual compaction. "
            "When unset, compaction reuses the main agent model."
        ),
    )
    api_mode: ApiMode = ApiMode.RESPONSES
    base_url: str | None = None  # OpenAI-compatible endpoint for local/vLLM
    max_tool_calls: int = Field(
        default=DEFAULT_MAX_TOOL_CALLS,
        ge=1,
        description="Tool budget per run. Forced finalization at limit.",
    )
    max_visible_tools: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on the number of model-visible runtime tools for this "
            "agent. Counts the built-in shell tool when it is exposed."
        ),
    )
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    response_format: type[BaseModel] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Optional pydantic model for structured final output. "
            "When set and no runtime tools are exposed, the orchestrator uses "
            "LLMClient.complete_structured and returns parsed payload in "
            "Result.metadata['structured']."
        ),
    )
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    thinking_profile: ThinkingProfilePreset | None = Field(
        default=None,
        description=(
            "Optional named preset for selective thinking/instruct behavior. "
            "Preset defaults apply only when the corresponding sampling fields "
            "are not explicitly set on this config."
        ),
    )
    thinking_run_policy: ThinkingRunPolicy = Field(
        default=ThinkingRunPolicy.AUTO,
        description=(
            "Per-run override for model reasoning. FORCE_ON/FORCE_OFF only "
            "affect enable_thinking when it is not explicitly configured."
        ),
    )
    thinking_tokens_reserve: int = Field(
        default=0,
        ge=0,
        description=(
            "Optional extra output-token reserve added to max_tokens for "
            "reasoning-heavy models."
        ),
    )
    enable_thinking: bool | None = Field(
        default=None,
        description=(
            "Optional thinking-mode hint for backends that support it via "
            "OpenAI-compatible extra_body (for example Qwen on vLLM/SGLang)."
        ),
    )
    stream: bool = Field(
        default=True,
        description=(
            "Whether LLM turns should use streaming transport by default. "
            "Set to False to force classic non-stream completions."
        ),
    )
    emit_reasoning_in_stream: bool = Field(
        default=False,
        description=(
            "Whether reasoning deltas may be emitted to callers in stream mode. "
            "Defaults to False for safer UX."
        ),
    )
    output_token_soft_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional warning threshold for output tokens produced by a single "
            "LLM turn."
        ),
    )
    output_token_hard_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional hard threshold for output tokens produced by a single LLM "
            "turn. The core records a warning when exceeded."
        ),
    )
    llm_request_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional provider-specific LLM request kwargs passed through "
            "to the LLM adapter (for example frequency_penalty, seed, etc.)."
        ),
    )
    llm_extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific OpenAI-compatible extra_body merged into each "
            "LLM request. Useful for backend knobs such as top_k/min_p."
        ),
    )
    cost_per_token: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional per-token price used to estimate LLM cost in ExecutionReport. "
            "Applied to input/output tokens."
        ),
    )
    cost_per_cached_token: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional per-cached-token price. Cached tokens are typically 2-10x "
            "cheaper than regular input tokens. When None, cached tokens are "
            "excluded from cost estimation."
        ),
    )
    estimate_tokens_func: TokenEstimateFunc | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Optional custom token-estimation function. When provided, it takes "
            "precedence over built-in tokenization profiles."
        ),
    )
    token_estimator_profile: TokenEstimatorProfile = Field(
        default=TokenEstimatorProfile.AUTO,
        description=(
            "Built-in token counting profile. AUTO prefers a Qwen-oriented setup "
            "for Qwen models and OpenAI model-aware encoding for GPT/o-series."
        ),
    )
    # Compression thresholds rely on a configurable fallback char-based estimate.
    chars_per_token_estimate: float = Field(
        default=4.0,
        gt=0.0,
        description=(
            "Fallback number of characters per token used when heuristic counting "
            "is selected or when tokenizer-backed counting is unavailable."
        ),
    )
    auto_compact_threshold: int = Field(
        default=DEFAULT_AUTO_COMPACT_THRESHOLD,
        ge=0,
        description="Token-estimate threshold to trigger auto_compact.",
    )
    auto_compact_keep_trailing: int = Field(
        default=DEFAULT_AUTO_COMPACT_KEEP_TRAILING,
        ge=0,
        description="Number of trailing messages kept after auto_compact summary injection.",
    )
    auto_compact_max_tokens: int = Field(
        default=AUTO_COMPACT_MAX_TOKENS,
        ge=1,
        description="Max output tokens for the LLM summarization call during auto_compact.",
    )
    auto_compact_temperature: float = Field(
        default=AUTO_COMPACT_TEMPERATURE,
        ge=0.0,
        description="Temperature for the LLM summarization call during auto_compact.",
    )
    auto_compact_timeout_seconds: float = Field(
        default=DEFAULT_AUTO_COMPACT_TIMEOUT_SECONDS,
        gt=0.0,
        description=(
            "Timeout in seconds for each LLM summarization attempt during "
            "auto/manual compact."
        ),
    )
    micro_compact_keep_recent: int = Field(
        default=DEFAULT_MICRO_COMPACT_KEEP_RECENT,
        ge=0,
        description="Number of recent tool results to keep in micro_compact.",
    )
    max_tool_result_size: int = Field(
        default=DEFAULT_MAX_TOOL_RESULT_SIZE,
        ge=1,
        description="Max chars kept per tool result before placeholder.",
    )
    shell_tool_config: ShellToolConfig = Field(
        default_factory=ShellToolConfig,
        description=(
            "Configuration for the built-in shell capability. The core only "
            "defines the contract; the service provides the actual sandboxed "
            "runtime via ShellExecutor."
        ),
    )
    execution_mode: ExecutionMode = ExecutionMode.LEADER
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)
    parallel_tool_calls: bool = Field(
        default=False,
        description="When True, execute multiple tool calls from one LLM turn concurrently (asyncio.gather).",
    )
    allow_fallback_tool_call_recovery: bool = Field(
        default=False,
        description=(
            "Whether the orchestrator may recover tool calls from plain assistant "
            "text when the model failed to return structured tool_calls. Disabled "
            "by default because plain-text recovery must only be used on trusted "
            "assistant messages, never on tool outputs."
        ),
    )
    skill_set: list[str] = Field(
        default_factory=list,
        description=(
            "Names of skills available to this agent. The service SkillManager "
            "resolves these names into index entries and load_skill responses."
        ),
    )
    skill_index_max_chars: int = Field(
        default=DEFAULT_SKILL_INDEX_MAX_CHARS,
        ge=100,
        description=(
            "Hard cap for the injected compact skills index block in system prompt."
        ),
    )
    skill_load_max_chars: int = Field(
        default=DEFAULT_SKILL_LOAD_MAX_CHARS,
        ge=500,
        description="Hard cap on skill body returned by load_skill.",
    )
    max_skill_loads_per_run: int = Field(
        default=DEFAULT_MAX_SKILL_LOADS_PER_RUN,
        ge=1,
        description="Maximum allowed load_skill calls per run.",
    )
    system_prompt: str = ""
    custom_data: dict[str, Any] = Field(default_factory=dict, alias="extra")
    thinking_profile_defaults_snapshot: dict[str, dict[str, Any]] = Field(
        default_factory=ThinkingProfileRegistry.all_profiles,
        exclude=True,
        repr=False,
        description=(
            "Legacy snapshot kept for backward compatibility. Runtime resolution "
            "uses ThinkingProfileRegistry as the single source of truth."
        ),
    )

    model_config = {"extra": "forbid", "populate_by_name": True}

    @field_validator("model", mode="before")
    @classmethod
    def validate_model_name(cls, value: Any) -> str:
        """Reject empty or whitespace-only model identifiers early."""
        if not isinstance(value, str):
            raise TypeError("model must be a string")
        model_name = value.strip()
        if not model_name:
            raise ValueError("model must be a non-empty string")
        return model_name

    @model_validator(mode="after")
    def validate_output_token_limits(self) -> "AgentConfig":
        if (
            self.output_token_soft_limit is not None
            and self.output_token_hard_limit is not None
            and self.output_token_hard_limit < self.output_token_soft_limit
        ):
            raise ValueError(
                "output_token_hard_limit must be greater than or equal to "
                "output_token_soft_limit"
            )
        if (
            self.max_tokens is not None
            and self.max_tokens >= 1
            and self.thinking_tokens_reserve > self.max_tokens
        ):
            raise ValueError(
                "thinking_tokens_reserve cannot exceed max_tokens"
            )
        return self

    @property
    def extra(self) -> dict[str, Any]:
        """Backward-compatible alias for ``custom_data``."""
        return self.custom_data

    @extra.setter
    def extra(self, value: dict[str, Any]) -> None:
        """Set custom_data without re-validating the full config model."""
        if not isinstance(value, dict):
            raise TypeError(f"extra must be a dict, got {type(value).__name__}")
        self.custom_data = value

    def resolved_with_selective_thinking(self) -> "AgentConfig":
        """Return a copy with preset defaults applied for the current step.

        Profiles are looked up via ``ThinkingProfileRegistry`` which supports
        runtime-registered custom profiles in addition to built-in presets.
        """
        updates: dict[str, Any] = {}
        preset_defaults: dict[str, Any] = {}
        if self.thinking_profile is not None:
            preset_defaults = ThinkingProfileRegistry.get(self.thinking_profile.value) or {}
        for field_name, value in preset_defaults.items():
            if getattr(self, field_name) is None:
                updates[field_name] = value

        if self.enable_thinking is None:
            if self.thinking_run_policy == ThinkingRunPolicy.FORCE_ON:
                updates["enable_thinking"] = True
            elif self.thinking_run_policy == ThinkingRunPolicy.FORCE_OFF:
                updates["enable_thinking"] = False

        if not updates:
            return self
        return self.model_copy(update=updates)

    def shell_tool_enabled_for_run(self, run_kind: RunKind) -> bool:
        """Return True when the built-in shell tool should be exposed."""
        access_mode = self.shell_tool_config.access_mode
        if access_mode == ShellAccessMode.DISABLED:
            return False
        if access_mode == ShellAccessMode.ALL_AGENTS:
            return True
        return run_kind == RunKind.LEADER

    def with_thinking_disabled(self) -> "AgentConfig":
        """Return a copy that disables Qwen thinking in one call.

        This helper overlays ``QWEN_NO_THINKING_EXTRA_BODY`` on top of existing
        ``llm_extra_body`` so callers do not have to handcraft
        ``chat_template_kwargs`` payloads.
        """
        merged_extra_body = dict(self.llm_extra_body)
        chat_template_kwargs = dict(merged_extra_body.get("chat_template_kwargs", {}))
        chat_template_kwargs.update(
            QWEN_NO_THINKING_EXTRA_BODY["chat_template_kwargs"]
        )
        merged_extra_body["chat_template_kwargs"] = chat_template_kwargs
        return self.model_copy(
            update={
                "enable_thinking": False,
                "llm_extra_body": merged_extra_body,
            },
            deep=True,
        )


class AgentContext(BaseModel):
    """Runtime context for an agent execution."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_agent_id: str | None = None
    message_history_ref: str | None = None
    execution_metadata_ref: str | None = None
    messages: MessageList = Field(default_factory=MessageList)
    config: AgentConfig = Field(default_factory=AgentConfig)
    tool_context: ToolContext = Field(default_factory=ToolContext)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extensible agent-scoped metadata bag. Known keys are documented by "
            "AgentContextMeta / AgentContextMetadata."
        ),
    )

    model_config = {"extra": "forbid"}

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, value: Any) -> MessageList:
        if value is None:
            return MessageList()
        if isinstance(value, MessageList):
            return value
        if not isinstance(value, list):
            raise TypeError("messages must be a list[Message]")
        return MessageList(value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "messages" and isinstance(value, list) and not isinstance(value, MessageList):
            value = MessageList(value)
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def ensure_session_refs(self) -> "AgentContext":
        legacy_key = AgentContextMeta.LEGACY_ACTIVE_CHILD_AGENT_IDS.value
        canonical_key = AgentContextMeta.ACTIVE_CHILD_AGENT_IDS.value
        if canonical_key not in self.metadata and legacy_key in self.metadata:
            legacy_child_ids = self.metadata.get(legacy_key)
            if isinstance(legacy_child_ids, list):
                # Expose canonical key while keeping legacy alias.
                self.metadata[canonical_key] = list(legacy_child_ids)
        canonical_child_ids = self.metadata.get(canonical_key)
        if isinstance(canonical_child_ids, list):
            self.metadata[legacy_key] = list(canonical_child_ids)
        if not self.message_history_ref:
            self.message_history_ref = f"session:{self.session_id}:messages"
        if not self.execution_metadata_ref:
            self.execution_metadata_ref = f"request:{self.request_id}:metadata"
        metadata_updates: dict[str, Any] = {
            ToolContextMeta.MESSAGE_HISTORY_REF.value: self.message_history_ref,
            ToolContextMeta.EXECUTION_METADATA_REF.value: self.execution_metadata_ref,
            ToolContextMeta.REQUEST_ID.value: self.request_id,
        }
        if self.parent_agent_id is not None:
            metadata_updates[ToolContextMeta.PARENT_AGENT_ID.value] = self.parent_agent_id

        new_metadata = dict(self.tool_context.metadata)
        new_metadata.update(metadata_updates)
        if self.parent_agent_id is None:
            new_metadata.pop(ToolContextMeta.PARENT_AGENT_ID.value, None)

        self.tool_context = self.tool_context.model_copy(
            update={
                "session_id": self.session_id,
                "trace_id": self.trace_id,
                "agent_id": self.config.agent_id,
                "metadata": new_metadata,
            }
        )
        return self


# ---------------------------------------------------------------------------
# Inter-agent envelope
# ---------------------------------------------------------------------------



class TaskPayload(BaseModel):
    """Minimal task payload sent from leader to subagent."""

    task: str = Field(min_length=1)
    context_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ControlPayload(BaseModel):
    """Control-plane payload for runtime commands such as cancel."""

    command: ControlCommand

    model_config = {"extra": "forbid"}


class ErrorPayload(BaseModel):
    """Structured error payload for inter-agent failures."""

    error: str = Field(min_length=1)
    error_code: str = Field(min_length=1)

    model_config = {"extra": "forbid"}


class AgentIdentity(BaseModel):
    agent_id: str
    role: AgentRole

    model_config = {"extra": "forbid"}


class EnvelopeMeta(BaseModel):
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    protocol_version: str = PROTOCOL_VERSION
    compatibility_warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _validate_iso_datetime_string(value, field_name="created_at") or value


class AgentEnvelope(BaseModel):
    """Inter-agent JSON envelope.

    Payload contains only minimal task context (no full message history).
    """

    protocol_version: str = PROTOCOL_VERSION
    message_type: MessageType
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentIdentity
    recipient: AgentIdentity
    payload: dict[str, Any] = Field(default_factory=dict)
    meta: EnvelopeMeta = Field(default_factory=EnvelopeMeta)

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def align_meta_protocol_version(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        protocol_version = data.get("protocol_version", PROTOCOL_VERSION)
        raw_meta = data.get("meta")
        if raw_meta is None:
            data["meta"] = {"protocol_version": protocol_version}
            return data
        if isinstance(raw_meta, EnvelopeMeta):
            if not raw_meta.protocol_version:
                raw_meta.protocol_version = protocol_version
            return data
        if isinstance(raw_meta, dict) and "protocol_version" not in raw_meta:
            data["meta"] = {
                **raw_meta,
                "protocol_version": protocol_version,
            }
        return data

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, v: str) -> str:
        major = v.split(".")[0]
        expected_major = PROTOCOL_VERSION.split(".")[0]
        if major != expected_major:
            raise ValueError(
                f"PROTOCOL_VERSION_MISMATCH: got major={major}, "
                f"expected={expected_major}"
            )
        return v

    @field_validator("payload")
    @classmethod
    def validate_minimal_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        forbidden_paths = _find_forbidden_payload_keys(payload)
        if forbidden_paths:
            raise ValueError(
                "Envelope payload must contain only minimal task context; "
                f"forbidden keys present: {', '.join(forbidden_paths)}"
            )
        return payload

    @model_validator(mode="after")
    def validate_payload_contract(self) -> "AgentEnvelope":
        if self.meta.protocol_version != self.protocol_version:
            raise ValueError(
                "Envelope protocol_version must match meta.protocol_version; "
                f"got={self.protocol_version}, meta={self.meta.protocol_version}"
            )
        payload_size = len(_json_serialize_strict(self.payload))
        if payload_size > MAX_ENVELOPE_PAYLOAD_CHARS:
            raise ValueError(
                "Envelope payload exceeds the maximum allowed size; "
                f"got={payload_size}, limit={MAX_ENVELOPE_PAYLOAD_CHARS}"
            )

        payload_depth = _compute_payload_depth(self.payload)
        if payload_depth > MAX_ENVELOPE_PAYLOAD_DEPTH:
            raise ValueError(
                "Envelope payload exceeds the maximum allowed nesting depth; "
                f"got={payload_depth}, limit={MAX_ENVELOPE_PAYLOAD_DEPTH}"
            )

        payload_model = _PAYLOAD_MODELS.get(self.message_type)
        if payload_model is None:
            raise ValueError(
                "No payload model registered for message_type="
                f"{self.message_type.value}"
            )
        structured = payload_model.model_validate(self.payload)
        self.payload = structured.model_dump(exclude_none=True)
        warning = self.check_minor_version()
        if warning and warning not in self.meta.compatibility_warnings:
            self.meta.compatibility_warnings.append(warning)
        return self

    def check_minor_version(self) -> str | None:
        """Return warning string if minor version differs from expected, else None."""
        parts = self.protocol_version.split(".")
        expected_parts = PROTOCOL_VERSION.split(".")
        if len(parts) >= 2 and len(expected_parts) >= 2:
            if parts[1] != expected_parts[1]:
                return (
                    f"protocol_minor_version_mismatch: "
                    f"got={self.protocol_version}, expected={PROTOCOL_VERSION}"
                )
        return None

    def apply_version_compatibility(self, report: "ExecutionReport") -> None:
        """Record compatibility warnings into an execution report."""
        warnings = list(self.meta.compatibility_warnings)
        if not warnings:
            warning = self.check_minor_version()
            if warning:
                warnings.append(warning)
        for warning in warnings:
            if warning not in report.warnings:
                report.add_warning(warning)

    @classmethod
    def parse_with_report(
        cls,
        data: str | bytes | dict[str, Any] | "AgentEnvelope",
        report: "ExecutionReport | None" = None,
    ) -> "AgentEnvelope":
        """Parse envelope and always apply minor-version compatibility warnings."""
        if isinstance(data, cls):
            envelope = cls.model_validate(data.model_dump(mode="python"))
        elif isinstance(data, (str, bytes)):
            envelope = cls.model_validate_json(data)
        else:
            envelope = cls.model_validate(data)
        if report is not None:
            envelope.apply_version_compatibility(report)
        return envelope


# ---------------------------------------------------------------------------
# Subagent structured output
# ---------------------------------------------------------------------------



def _find_forbidden_payload_keys(value: Any, path: str = "payload") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key)
            key_lower = key_str.lower()
            next_path = f"{path}.{key_str}"
            if key_lower in FORBIDDEN_PAYLOAD_KEYS:
                found.append(next_path)
            found.extend(_find_forbidden_payload_keys(item, next_path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            found.extend(_find_forbidden_payload_keys(item, f"{path}[{idx}]"))
    return found


def _compute_payload_depth(
    value: Any,
    current_depth: int = 1,
    max_depth: int = 100,
) -> int:
    """Compute nested payload depth with a hard recursion ceiling."""
    if current_depth >= max_depth:
        return current_depth
    if isinstance(value, dict):
        if not value:
            return current_depth
        return max(
            _compute_payload_depth(item, current_depth + 1, max_depth)
            for item in value.values()
        )
    if isinstance(value, list):
        if not value:
            return current_depth
        return max(
            _compute_payload_depth(item, current_depth + 1, max_depth)
            for item in value
        )
    return current_depth


class SubagentResult(BaseModel):
    """Structured output from a subagent.

    Subagents are expected to return a JSON object matching this schema.
    The strict parser uses this model during ``AUTO_SELECT``/delegation flows.

    Example JSON payload:
    {
      "status": "success",
      "summary": "Computed factorial(5)=120 and explained recursion.",
      "artifacts": [],
      "files_changed": [],
      "tool_calls_made": 0,
      "errors": [],
      "next_steps": null
    }

    To build a system prompt template for subagents, use:
    ``SubagentResult.prompt_instructions()``.
    """

    status: SubagentStatus
    summary: str = Field(max_length=MAX_SUMMARY_CHARS)
    artifacts: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    tool_calls_made: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list, max_length=MAX_SUBAGENT_ERRORS)
    next_steps: str | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def prompt_instructions(cls) -> str:
        """Return system-prompt instructions for strict SubagentResult JSON output."""

        return (
            "Return ONLY valid JSON (no markdown, no prose) matching this schema:\n"
            "{\n"
            '  "status": "success" | "partial" | "failed",\n'
            '  "summary": "<short summary>",\n'
            '  "artifacts": ["<optional artifact paths>"],\n'
            '  "files_changed": ["<optional modified files>"],\n'
            '  "tool_calls_made": 0,\n'
            '  "errors": ["<optional error strings>"],\n'
            '  "next_steps": "<optional next step>" | null\n'
            "}\n"
            "Do not include keys outside this schema."
        )

    @classmethod
    def parse_with_fallback(cls, raw: str, agent_id: str = "") -> "SubagentResult":
        """Strict parse with graceful fallback.

        The parser accepts raw JSON, markdown-fenced JSON, and extracted object/array
        candidates. Invalid payloads degrade to a partial result instead of raising.
        """
        if len(raw) > MAX_STRUCTURED_JSON_CHARS:
            return cls(
                status=SubagentStatus.PARTIAL,
                summary=(
                    "Structured output exceeded the maximum allowed JSON size "
                    "and was replaced with a fallback summary."
                ),
                errors=[f"JSON_TOO_LARGE: agent={agent_id}"],
            )
        normalized_raw = raw.strip()
        approval_markers = (
            "[approval required before shell execution]",
            "approval_required",
            "shell approval pending",
        )
        if normalized_raw and any(marker in normalized_raw.lower() for marker in approval_markers):
            return cls(
                status=SubagentStatus.PARTIAL,
                summary=normalized_raw[:MAX_SUMMARY_CHARS],
                errors=["APPROVAL_REQUIRED"],
            )

        last_exc: Exception | None = None
        for candidate in structured_json_candidates(raw):
            try:
                data = json.loads(candidate)
                return cls.model_validate(data)
            except (
                json.JSONDecodeError,
                ValidationError,
                TypeError,
                ValueError,
            ) as exc:
                last_exc = exc
        truncated = raw[:MAX_SUMMARY_CHARS]
        return cls(
            status=SubagentStatus.PARTIAL,
            summary=truncated,
            errors=[
                "SUBAGENT_RESULT_SCHEMA_VIOLATION",
                f"Failed to parse structured output from agent={agent_id}: {last_exc}",
            ],
        )


class ResultPayload(SubagentResult):
    """Structured result payload sent from subagent to leader."""


_PAYLOAD_MODELS: dict[MessageType, type[BaseModel]] = {
    MessageType.TASK: TaskPayload,
    MessageType.RESULT: ResultPayload,
    MessageType.CONTROL: ControlPayload,
    MessageType.ERROR: ErrorPayload,
}


# ---------------------------------------------------------------------------
# Compression summary
# ---------------------------------------------------------------------------


# COMPACTION_SUMMARY_MARKER is imported from .constants and re-exported


class CompactionSummary(BaseModel):
    """Structured summary stored after auto/manual compact."""

    marker: str = Field(
        default=COMPACTION_SUMMARY_MARKER,
        validation_alias=AliasChoices("marker", "__marker__"),
    )
    completed_tasks: list[str] = Field(default_factory=list)
    current_goal: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    next_steps: str = ""
    original_count: int = Field(default=0, ge=0)
    compacted_count: int = Field(default=0, ge=0)
    tokens_saved: int = Field(default=0, ge=0)
    duration_ms: float | None = Field(default=None, ge=0.0)

    model_config = {"extra": "forbid", "populate_by_name": True}

    @field_validator("marker", mode="before")
    @classmethod
    def normalize_marker(cls, value: Any) -> str:
        marker = str(value or COMPACTION_SUMMARY_MARKER)
        if marker != COMPACTION_SUMMARY_MARKER:
            raise ValueError(
                f"marker must be {COMPACTION_SUMMARY_MARKER!r}"
            )
        return COMPACTION_SUMMARY_MARKER

    @property
    def messages_removed(self) -> int:
        """DX alias: how many messages were removed during compaction."""
        return max(self.original_count - self.compacted_count, 0)


# ---------------------------------------------------------------------------
# Workflow / DAG types
# ---------------------------------------------------------------------------


class WorkflowNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowNode(BaseModel):
    node_id: str
    label: str
    agent_id: str | None = None
    tool_name: str | None = None
    status: WorkflowNodeStatus = WorkflowNodeStatus.PENDING
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class WorkflowEdge(BaseModel):
    from_node: str
    to_node: str
    condition: str | None = None  # optional guard condition

    model_config = {"extra": "forbid"}


class WorkflowDefinition(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_graph(self) -> "WorkflowDefinition":
        node_ids = [node.node_id for node in self.nodes]
        unique_node_ids = set(node_ids)
        if len(unique_node_ids) != len(node_ids):
            raise ValueError("WorkflowDefinition contains duplicate node_id values")

        for edge in self.edges:
            if edge.from_node not in unique_node_ids:
                raise ValueError(f"WorkflowEdge.from_node references unknown node: {edge.from_node}")
            if edge.to_node not in unique_node_ids:
                raise ValueError(f"WorkflowEdge.to_node references unknown node: {edge.to_node}")

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in unique_node_ids}
        in_degree: dict[str, int] = {node_id: 0 for node_id in unique_node_ids}
        for edge in self.edges:
            adjacency[edge.from_node].append(edge.to_node)
            in_degree[edge.to_node] += 1

        # Kahn's algorithm avoids deep recursion for large DAGs.
        zero_in_degree = [node_id for node_id, degree in in_degree.items() if degree == 0]
        processed = 0

        while zero_in_degree:
            node_id = zero_in_degree.pop()
            processed += 1
            for next_node in adjacency.get(node_id, []):
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    zero_in_degree.append(next_node)

        if processed != len(unique_node_ids):
            raise ValueError("WorkflowDefinition must be acyclic")

        if len(unique_node_ids) > 1:
            undirected: dict[str, set[str]] = {
                node_id: set() for node_id in unique_node_ids
            }
            for edge in self.edges:
                undirected[edge.from_node].add(edge.to_node)
                undirected[edge.to_node].add(edge.from_node)

            start = next(iter(unique_node_ids))
            visited: set[str] = set()
            stack = [start]
            while stack:
                node_id = stack.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                stack.extend(undirected[node_id] - visited)

            if visited != unique_node_ids:
                unreachable = ", ".join(sorted(unique_node_ids - visited))
                raise ValueError(
                    "WorkflowDefinition must be weakly connected; "
                    f"disconnected nodes: {unreachable}"
                )

        return self


# ---------------------------------------------------------------------------
# Plan artifact
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    agent_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "skipped"] = "pending"

    model_config = {"extra": "forbid"}


class PlanArtifact(BaseModel):
    """Serialisable plan created by leader before dispatch."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    steps: list[PlanStep] = Field(default_factory=list)
    raw_plan: str = ""  # optional free-text plan from LLM

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Result — final output of an agent run
# ---------------------------------------------------------------------------


class Result(BaseModel):
    """Final output returned by OrchestrationStrategy.execute()."""

    content: str
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_details: dict[str, Any] | None = None

    model_config = {"extra": "forbid"}

    def to_message(self) -> Message:
        """Convert the final result back into an assistant message."""
        return Message(role="assistant", content=self.content)

    def get_structured(self, model: type[_StructuredModelT]) -> _StructuredModelT | None:
        """Return validated structured payload from ``metadata['structured']``."""
        data = self.metadata.get("structured")
        if data is None:
            return None
        if isinstance(data, model):
            return data
        return model.model_validate(data)


# ---------------------------------------------------------------------------
# Subagent run summary (nested in ExecutionReport)
# ---------------------------------------------------------------------------


class SubagentRunSummary(BaseModel):
    agent_id: str
    run_kind: RunKind = RunKind.SUBAGENT
    status: ExecutionStatus
    started_at: str
    finished_at: str
    duration_ms: float
    tool_calls_total: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_iso_datetime_string(value, field_name=str(info.field_name)) or value


# ---------------------------------------------------------------------------
# CompressionEvent — per-event compression detail
# ---------------------------------------------------------------------------


class CompressionEvent(BaseModel):
    """Single compression event with before/after token counts."""

    kind: Literal["micro", "auto", "manual"]
    tokens_before: int
    tokens_after: int
    messages_affected: int = 0
    summary_parse_success: bool = True
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    model_config = {"extra": "forbid"}

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validate_iso_datetime_string(value, field_name="timestamp") or value


# ---------------------------------------------------------------------------
# ToolCallRecord — detailed per-call telemetry
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """Single tool call with arguments, timing, and success state."""

    tool_call_id: str = ""
    tool_name: str
    arguments: Any = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float | None = Field(default=None, ge=0.0)
    success: bool = True
    error_message: str | None = None

    model_config = {"extra": "forbid"}

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validate_iso_datetime_string(value, field_name="timestamp") or value

    @property
    def status(self) -> str:
        """Read-only compatibility alias expected by older user scripts."""
        return "success" if self.success else "failed"


# ---------------------------------------------------------------------------
# ExecutionReport — telemetry
# ---------------------------------------------------------------------------



class ExecutionReport(BaseModel):
    """Full telemetry report for a single agent run.

    Returned on every execution path including error, timeout, cancelled.
    Format is stable and versioned.
    """

    report_version: str = "1.1"

    # Identification
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    parent_agent_id: str | None = None
    run_kind: RunKind = RunKind.LEADER

    # Status
    status: ExecutionStatus = ExecutionStatus.RUNNING
    stop_reason: StopReason | None = None
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    # Timing
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    duration_ms: float | None = None
    llm_latency_ms: list[float] = Field(default_factory=list)
    tool_latency_ms: list[float] = Field(default_factory=list)
    queue_wait_ms: float | None = None
    loop_count: int = 0

    # LLM metrics
    model: str = ""
    api_mode: ApiMode = ApiMode.RESPONSES
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost: float | None = None

    # Tool metrics
    tool_calls_total: int = 0
    tool_calls_by_name: dict[str, int] = Field(default_factory=dict)
    tool_failures: int = 0
    tool_call_details: list[ToolCallRecord] = Field(default_factory=list)
    forced_finalization_triggered: bool = False
    state_manager_timeout_count: int = 0

    # Compression metrics
    micro_compact_applied: int = 0
    auto_compact_applied: int = 0
    auto_compact_failed: int = 0
    manual_compact_applied: int = 0
    tokens_before_compression_total: int | None = None
    tokens_after_compression_total: int | None = None
    compression_events: list[CompressionEvent] = Field(default_factory=list)

    # Workflow metrics
    workflow_id: str | None = None
    node_count: int | None = None
    edge_count: int | None = None
    node_durations_ms: dict[str, float] = Field(default_factory=dict)

    # Execution mode + planning
    execution_mode: ExecutionMode = ExecutionMode.LEADER
    plan_created: bool = False
    plan_id: str | None = None
    plan_artifact: PlanArtifact | None = None
    subagents_parallel_max: int = 0

    # Safety signals
    destructive_action_requested: int = 0
    destructive_action_confirmed: int = 0
    prompt_injection_signals: int = 0
    shell_calls_total: int = 0
    shell_calls_denied: int = 0
    shell_calls_confirm_required: int = 0
    shell_approvals_granted: int = 0
    shell_approvals_rejected: int = 0
    shell_risk_flags: list[str] = Field(default_factory=list)

    # Artifacts (capped to prevent unbounded growth)
    artifacts: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    artifacts_dropped: int = 0
    files_changed_dropped: int = 0
    artifacts_overflow: bool = False
    files_changed_overflow: bool = False
    subagent_runs: list[SubagentRunSummary] = Field(default_factory=list)

    # Additional metadata (e.g., chat_id for linking)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: str) -> str:
        return _validate_iso_datetime_string(value, field_name="started_at") or value

    @field_validator("finished_at")
    @classmethod
    def validate_finished_at(cls, value: str | None) -> str | None:
        return _validate_iso_datetime_string(value, field_name="finished_at")

    @property
    def iterations(self) -> int:
        """Deprecated alias for ``loop_count``."""
        warnings.warn(
            "ExecutionReport.iterations is deprecated; use loop_count instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.loop_count

    @property
    def tool_calls_count(self) -> int:
        """Deprecated alias for ``tool_calls_total``."""
        warnings.warn(
            "ExecutionReport.tool_calls_count is deprecated; use tool_calls_total instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tool_calls_total

    def total_tokens_including_subagents(self) -> tuple[int, int]:
        """Return inclusive token totals with child runs."""
        if self.metadata.get("parent_tokens_include_subagents") is True:
            return self.input_tokens, self.output_tokens
        return (
            self.input_tokens + sum(max(s.input_tokens, 0) for s in self.subagent_runs),
            self.output_tokens + sum(max(s.output_tokens, 0) for s in self.subagent_runs),
        )

    def parent_tokens(self) -> tuple[int, int]:
        """Return leader-only token totals without child runs."""
        child_input = self.metadata.get("child_tokens_sum_input")
        child_output = self.metadata.get("child_tokens_sum_output")
        if (
            self.metadata.get("parent_tokens_include_subagents") is True
            and isinstance(child_input, int)
            and isinstance(child_output, int)
        ):
            return (
                max(self.input_tokens - child_input, 0),
                max(self.output_tokens - child_output, 0),
            )
        return self.input_tokens, self.output_tokens

    def child_tokens_sum(self) -> tuple[int, int]:
        """Return the sum of child subagent token usage."""
        child_input = self.metadata.get("child_tokens_sum_input")
        child_output = self.metadata.get("child_tokens_sum_output")
        if isinstance(child_input, int) and isinstance(child_output, int):
            return max(child_input, 0), max(child_output, 0)
        return (
            sum(max(s.input_tokens, 0) for s in self.subagent_runs),
            sum(max(s.output_tokens, 0) for s in self.subagent_runs),
        )

    def add_artifact(self, value: str) -> None:
        """Append an artifact and surface overflow instead of dropping silently."""
        if len(self.artifacts) < MAX_ARTIFACTS:
            self.artifacts.append(value)
            return
        # Surface artifact overflow instead of silently dropping data.
        self.artifacts_dropped += 1
        self.artifacts_overflow = True
        if self.artifacts_dropped == 1:
            self.add_warning(
                f"Artifact limit reached ({MAX_ARTIFACTS}), dropping new entries"
            )

    def add_file_changed(self, path: str) -> None:
        """Append a file path and surface overflow instead of silent data loss."""
        if path in self.files_changed:
            return
        if len(self.files_changed) < MAX_FILES_CHANGED:
            self.files_changed.append(path)
            return
        # Surface files_changed overflow instead of silently dropping data.
        self.files_changed_dropped += 1
        self.files_changed_overflow = True
        if self.files_changed_dropped == 1:
            self.add_warning(
                f"File change limit reached ({MAX_FILES_CHANGED}), dropping new entries"
            )

    def add_warning(self, warning: str) -> None:
        """Record a warning in one place for consistent report mutation."""
        self.warnings.append(warning)

    def increment_tool_call(self, tool_name: str, count: int = 1) -> None:
        """Update the per-tool counter without duplicating dict plumbing."""
        self.tool_calls_by_name[tool_name] = (
            self.tool_calls_by_name.get(tool_name, 0) + count
        )

    def add_tool_call_detail(self, detail: ToolCallRecord) -> None:
        """Append a detailed tool-call record."""
        self.tool_call_details.append(detail)

    def finalize(
        self,
        status: ExecutionStatus,
        stop_reason: StopReason | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark report as finished, compute duration."""
        if self.status != ExecutionStatus.RUNNING or self.finished_at is not None:
            logger.debug(
                "ExecutionReport.finalize ignored repeated call: "
                "agent_id=%s current_status=%s attempted_status=%s",
                self.agent_id,
                self.status.value,
                status.value,
            )
            return
        self.status = status
        self.stop_reason = stop_reason
        self.error_code = error_code
        self.error_message = error_message
        now = datetime.now(timezone.utc).isoformat()
        self.finished_at = now
        if self.started_at:
            started = datetime.fromisoformat(self.started_at)
            finished = datetime.fromisoformat(now)
            self.duration_ms = (finished - started).total_seconds() * 1000

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "ExecutionReport":
        if self.status in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.PARTIAL,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        }:
            if self.finished_at is None:
                self.finished_at = datetime.now(timezone.utc).isoformat()
            if self.duration_ms is None:
                if self.started_at:
                    started = datetime.fromisoformat(self.started_at)
                    finished = datetime.fromisoformat(self.finished_at)
                    self.duration_ms = (finished - started).total_seconds() * 1000
                else:
                    self.duration_ms = 0.0
        return self


# ---------------------------------------------------------------------------
# Session snapshot
# ---------------------------------------------------------------------------


class SessionSnapshot(BaseModel):
    """Minimal session state persisted via StateManager.

    ``model_copy(update=...)`` automatically refreshes ``updated_at`` unless the
    caller explicitly supplies a custom timestamp.
    """

    session_id: str
    trace_id: str
    agent_id: str
    message_history_ref: str = Field(min_length=1)
    execution_metadata_ref: str = Field(min_length=1)
    messages: list[Message] = Field(default_factory=list)
    execution_report_id: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_snapshot_timestamps(cls, value: str, info: Any) -> str:
        return _validate_iso_datetime_string(value, field_name=str(info.field_name)) or value

    def touch(self) -> "SessionSnapshot":
        """Return a copy with ``updated_at`` set to the current UTC timestamp."""

        return self.model_copy(update={"updated_at": datetime.now(timezone.utc).isoformat()})

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "SessionSnapshot":
        """Copy the snapshot and keep ``updated_at`` fresh on partial updates."""

        updates = None if update is None else dict(update)
        if updates is not None and "updated_at" not in updates:
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        return super().model_copy(update=updates, deep=deep)
