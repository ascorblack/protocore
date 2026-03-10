"""Centralized constants for Protocore.

All magic numbers, default limits, and hardcoded strings live here.
Grouped by domain. Import specific constants where needed.
"""
from __future__ import annotations
from threading import RLock
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping

# ---------------------------------------------------------------------------
# Protocol & Envelope
# ---------------------------------------------------------------------------
PROTOCOL_VERSION: Final[str] = "1.0"
MAX_ENVELOPE_PAYLOAD_CHARS: Final[int] = 10_000
MAX_ENVELOPE_PAYLOAD_DEPTH: Final[int] = 5
MAX_STRUCTURED_JSON_CHARS: Final[int] = 10_000

# ---------------------------------------------------------------------------
# Subagent Result Limits
# ---------------------------------------------------------------------------
MAX_SUMMARY_CHARS: Final[int] = 8_000
MAX_SUBAGENT_ERRORS: Final[int] = 10
FORBIDDEN_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset({
    "messages", "history", "conversation",
    "conversation_history", "message_history",
})

# ---------------------------------------------------------------------------
# Execution Report Limits
# ---------------------------------------------------------------------------
MAX_ARTIFACTS: Final[int] = 500
MAX_FILES_CHANGED: Final[int] = 500
DEFAULT_STATE_MANAGER_TIMEOUT_SECONDS: Final[float] = 5.0

# ---------------------------------------------------------------------------
# Compression Defaults
# ---------------------------------------------------------------------------
DEFAULT_AUTO_COMPACT_THRESHOLD: Final[int] = 30_000
DEFAULT_AUTO_COMPACT_KEEP_TRAILING: Final[int] = 4
DEFAULT_AUTO_COMPACT_TIMEOUT_SECONDS: Final[float] = 20.0
DEFAULT_MICRO_COMPACT_KEEP_RECENT: Final[int] = 2
DEFAULT_MAX_TOOL_RESULT_SIZE: Final[int] = 3_000

TRANSCRIPT_CONTENT_LIMIT: Final[int] = 2_000
TRANSCRIPT_JSON_SUMMARY_LIMIT: Final[int] = 1_500
TRANSCRIPT_PROTECT_LIMIT: Final[int] = 40_000

AUTO_COMPACT_MAX_TOKENS: Final[int] = 1_024
AUTO_COMPACT_TEMPERATURE: Final[float] = 0.1

COMPACTION_SUMMARY_MARKER: Final[str] = "__compaction_summary__"

# ---------------------------------------------------------------------------
# Auto-compact LLM Prompts
# ---------------------------------------------------------------------------
AUTO_COMPACT_SYSTEM_PROMPT: Final[str] = (
    "You are a context summarization assistant. "
    "Given the conversation history, produce a structured JSON summary "
    "of what happened. Return ONLY valid JSON with no extra text."
)

AUTO_COMPACT_USER_TEMPLATE: Final[str] = """\
Summarize the following agent conversation into a structured JSON object:

{transcript}

Return exactly this JSON structure (no other text):
{{
  "completed_tasks": ["<list of completed tasks>"],
  "current_goal": "<what the agent is currently working on>",
  "key_decisions": ["<important decisions made>"],
  "files_modified": ["<files that were created or changed>"],
  "next_steps": "<what needs to happen next>"
}}"""

# ---------------------------------------------------------------------------
# Token Estimation
# ---------------------------------------------------------------------------
DEFAULT_CHARS_PER_TOKEN: Final[float] = 4.0
DEFAULT_OPENAI_ENCODING: Final[str] = "o200k_base"
DEFAULT_QWEN_ENCODING: Final[str] = "o200k_base"
QWEN_SPECIAL_TOKENS_PER_MESSAGE: Final[int] = 5
QWEN_REPLY_PRIMING: Final[int] = 3
QWEN_NO_THINKING_EXTRA_BODY: Final[dict[str, dict[str, bool]]] = {
    "chat_template_kwargs": {"enable_thinking": False}
}

# ---------------------------------------------------------------------------
# Shell Tool Defaults
# ---------------------------------------------------------------------------
DEFAULT_SHELL_TIMEOUT_MS: Final[int] = 30_000
DEFAULT_SHELL_MAX_TIMEOUT_MS: Final[int] = 300_000
DEFAULT_SHELL_MAX_COMMAND_LENGTH: Final[int] = 4_000
DEFAULT_SHELL_TOOL_NAME: Final[str] = "shell_exec"
DEFAULT_SHELL_DESCRIPTION: Final[str] = (
    "Execute a shell command in an isolated sandbox runtime. "
    "Prefer inspection and deterministic commands; avoid destructive changes "
    "unless explicitly required."
)

# ---------------------------------------------------------------------------
# Agent Config Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL: Final[str] = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_MAX_TOOL_CALLS: Final[int] = 10
DEFAULT_MAX_ITERATIONS: Final[int] = 30

# ---------------------------------------------------------------------------
# Skill Defaults
# ---------------------------------------------------------------------------
DEFAULT_SKILL_INDEX_MAX_CHARS: Final[int] = 1_500
DEFAULT_SKILL_LOAD_MAX_CHARS: Final[int] = 8_000
DEFAULT_MAX_SKILL_LOADS_PER_RUN: Final[int] = 4

# ---------------------------------------------------------------------------
# Orchestrator Internal
# ---------------------------------------------------------------------------
MAX_EXTRA_BODY_MERGE_DEPTH: Final[int] = 5
MAX_RETRY_ATTEMPTS: Final[int] = 10
MAX_PARALLEL_AGENT_IDS: Final[int] = 50
MAX_PARALLEL_POLICY_CONCURRENCY: Final[int] = 100
MAX_RECOVERY_TEXT_SIZE: Final[int] = 10_000
MAX_SHELL_ENV_VALUE_LENGTH: Final[int] = 4_096
MIN_SHELL_EXECUTION_INTERVAL_MS: Final[int] = 100
BLOCKED_SHELL_ENV_KEYS: Final[frozenset[str]] = frozenset({
    "PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
})
MAX_RESULT_ERROR_MESSAGE_LENGTH: Final[int] = 500
SUBAGENT_CAPABILITY_SELECTION_PROMPT_TEMPLATE: Final[str] = """\
You are selecting the best subagent for a task.
Choose exactly one agent_id from the available list.
Return ONLY JSON: {{"agent_id":"<selected_agent_id>"}}.

Task:
{task}

Available subagents:
{available_agents}
"""

FORCED_FINALIZATION_MSG: Final[str] = (
    "You have reached the maximum number of tool calls for this session. "
    "Provide your FINAL answer now based on available data, even if incomplete. "
    "Do not attempt any more tool calls."
)

SUPPORTED_PARALLEL_CANCELLATION_MODES: Final[frozenset[str]] = frozenset({
    "propagate", "graceful",
})

# ---------------------------------------------------------------------------
# Built-in Tool Names
# ---------------------------------------------------------------------------
MANUAL_COMPACT_TOOL_NAME: Final[str] = "manual_compact"
LOAD_SKILL_TOOL_NAME: Final[str] = "load_skill"

# ---------------------------------------------------------------------------
# Thinking Profile Registry
# ---------------------------------------------------------------------------


class ThinkingProfileRegistry:
    """Extensible registry for named generation presets.

    Built-in presets are registered at import time.  Services and plugins can
    register additional profiles or override existing ones at runtime via
    ``ThinkingProfileRegistry.register()`` without modifying core code.

    Keys are string names matching ``ThinkingProfilePreset`` enum values (or
    any custom string for user-defined profiles).
    """

    _profiles: dict[str, dict[str, Any]] = {}
    _lock = RLock()

    @classmethod
    def register(cls, name: str, defaults: dict[str, Any]) -> None:
        """Register (or replace) a named thinking profile."""
        with cls._lock:
            cls._profiles[name] = dict(defaults)

    @classmethod
    def get(cls, name: str) -> dict[str, Any] | None:
        """Return profile defaults or ``None`` if not registered."""
        with cls._lock:
            profile = cls._profiles.get(name)
        return dict(profile) if profile is not None else None

    @classmethod
    def all_profiles(cls) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the entire registry."""
        with cls._lock:
            return {k: dict(v) for k, v in cls._profiles.items()}

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a profile. Returns ``True`` if it existed."""
        with cls._lock:
            return cls._profiles.pop(name, None) is not None

    @classmethod
    def reset_to_defaults(cls) -> None:
        """Restore only the built-in profiles.

        This is primarily intended for tests that temporarily register or
        override profiles and need a deterministic baseline afterwards.
        """
        with cls._lock:
            cls._profiles = {
                name: dict(defaults) for name, defaults in _BUILTIN_THINKING_PROFILES
            }


class _ThinkingProfileDefaultsView(Mapping[str, Mapping[str, Any]]):
    """Read-only live view backed by ThinkingProfileRegistry."""

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        profile = ThinkingProfileRegistry.get(key)
        if profile is None:
            raise KeyError(key)
        return MappingProxyType(profile)

    def __iter__(self) -> Iterator[str]:
        return iter(ThinkingProfileRegistry.all_profiles())

    def __len__(self) -> int:
        return len(ThinkingProfileRegistry.all_profiles())


_BUILTIN_THINKING_PROFILES: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    ("thinking_planner", {
        "enable_thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }),
    ("thinking_precise_coding", {
        "enable_thinking": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }),
    ("thinking_analytical", {
        "enable_thinking": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.05,
    }),
    ("instruct_tool_worker", {
        "enable_thinking": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }),
)


def _register_builtin_thinking_profiles() -> None:
    """Populate the registry with the built-in profile set."""
    ThinkingProfileRegistry.reset_to_defaults()


_register_builtin_thinking_profiles()

# Backward-compatible alias: read-only view used by types.py resolution.
THINKING_PROFILE_DEFAULTS: Final[Mapping[str, Mapping[str, Any]]] = (
    _ThinkingProfileDefaultsView()
)

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
HOOKS_PROJECT_NAME: Final[str] = "core"
