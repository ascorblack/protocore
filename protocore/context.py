"""Context helpers: CancellationContext, path validation, ToolContext helpers.

Path isolation is enforced here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any, cast
import weakref

from .constants import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_OPENAI_ENCODING,
    QWEN_REPLY_PRIMING,
    QWEN_SPECIAL_TOKENS_PER_MESSAGE,
)
from .orchestrator_utils import serialize_messages_for_api
from .types import (
    ApiMode,
    Message,
    TokenEstimateFunc,
    TokenEstimatorProfile,
    ToolContext,
    ToolContextMetadata,
    ToolContextMeta,
)

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency should normally be present
    tiktoken = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cancellation contract
# ---------------------------------------------------------------------------


@dataclass
class CancellationContext:
    """Coroutine-safe cancellation signal propagated to child runs.

    Usage::

        ctx = CancellationContext()
        ...
        ctx.cancel()           # from another coroutine
        ...
        ctx.check()            # raises CancelledError inside loop
    """

    _events_by_loop: weakref.WeakKeyDictionary[
        asyncio.AbstractEventLoop, asyncio.Event
    ] = field(
        default_factory=weakref.WeakKeyDictionary,
        init=False,
        repr=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    reason: str = ""
    _cancelled: bool = field(default=False, init=False, repr=False)

    def _event_for_current_loop(self) -> asyncio.Event:
        """Return the current loop's event without losing a prior cancel signal."""
        loop = asyncio.get_running_loop()
        with self._lock:
            event = self._events_by_loop.get(loop)
            if event is None:
                event = asyncio.Event()
                self._events_by_loop[loop] = event
                if self._cancelled:
                    event.set()
        return event

    def cancel(self, reason: str = "cancelled") -> None:
        """Signal cancellation. Idempotent — only the first call sets the reason."""
        loop_events: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self.reason = reason
            loop_events = list(self._events_by_loop.items())

        if not loop_events:
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        closed_loops: list[asyncio.AbstractEventLoop] = []
        for event_loop, event_to_set in loop_events:
            if current_loop is not event_loop:
                try:
                    event_loop.call_soon_threadsafe(event_to_set.set)
                except RuntimeError:
                    # Closed loops must not block propagation
                    # to remaining waiters during cascading cancellation.
                    closed_loops.append(event_loop)
                continue
            event_to_set.set()
        if closed_loops:
            with self._lock:
                for event_loop in closed_loops:
                    self._events_by_loop.pop(event_loop, None)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        """Raise asyncio.CancelledError if cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError(self.reason)

    async def wait(self) -> None:
        """Await until cancelled."""
        if self._cancelled:
            return
        await self._event_for_current_loop().wait()


# ---------------------------------------------------------------------------
# Path isolation helpers
# ---------------------------------------------------------------------------


class PathIsolationError(PermissionError):
    """Raised when a tool attempts to access a path outside allowed_paths."""


DEFAULT_PATH_ARGUMENT_KEYS = frozenset(
    {
        "path",
        "paths",
        "file_path",
        "source_path",
        "destination_path",
        "target_path",
        "cwd",
        "root",
    }
)


def _first_symlink_component(candidate: Path) -> Path | None:
    """Return the first symlink component in *candidate*, if any.

    For absolute paths, the first component below root is ignored to avoid
    false positives on platform aliases like ``/var`` or ``/tmp`` on macOS.
    Effective isolation is still enforced by resolved ``allowed_paths`` checks.
    """
    current = Path(candidate.anchor) if candidate.is_absolute() else Path()
    absolute_parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for idx, part in enumerate(absolute_parts):
        current = current / part
        if candidate.is_absolute() and idx == 0:
            continue
        try:
            if current.is_symlink():
                return current
        except OSError:
            return current
    return None


def validate_path_access(path: str | os.PathLike[str], tool_context: ToolContext) -> Path:
    """Return a resolved path if allowed; raise ``PathIsolationError`` otherwise.

    Relative paths are resolved against, in order:
    1. ``ToolContextMeta.WORKSPACE_ROOT`` from ``tool_context.metadata``.
    2. The first entry of ``tool_context.allowed_paths`` as a documented fallback.
    3. The current working directory when no workspace root is available.

    If ``allowed_paths`` is empty, all filesystem access is denied. Symlinks are
    blocked by default; callers must opt in via ``ToolContext.allow_symlinks``.
    """
    workspace_root_raw = tool_context.metadata.get(ToolContextMeta.WORKSPACE_ROOT)
    if isinstance(workspace_root_raw, (str, os.PathLike)):
        workspace_root = Path(workspace_root_raw).resolve()
    elif tool_context.allowed_paths:
        workspace_root = Path(tool_context.allowed_paths[0]).resolve()
        logger.debug(
            "validate_path_access: WORKSPACE_ROOT not set, using first allowed_path as root: %s",
            workspace_root,
        )
    else:
        workspace_root = None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (workspace_root or Path.cwd()) / candidate
    candidate = candidate.absolute()

    if not tool_context.allowed_paths:
        raise PathIsolationError(
            f"No allowed_paths configured — FS access denied for: {candidate}"
        )

    symlink_component = _first_symlink_component(candidate)
    if symlink_component is not None:
        logger.warning(
            "Path validation encountered symlink: candidate=%s symlink=%s allow_symlinks=%s",
            candidate,
            symlink_component,
            tool_context.allow_symlinks,
        )
        if not tool_context.allow_symlinks:
            # Deny symlink traversal unless explicitly allowed.
            raise PathIsolationError(f"Symlink access denied: {symlink_component}")
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PathIsolationError(f"Failed to resolve path: {candidate}") from exc

    for allowed in tool_context.allowed_paths:
        allowed_resolved = Path(allowed).resolve(strict=False)
        try:
            resolved.relative_to(allowed_resolved)
            return resolved  # Path is inside an allowed directory
        except ValueError:
            continue

    raise PathIsolationError(
        f"Path {resolved!r} is outside allowed_paths: {tool_context.allowed_paths}"
    )


def validate_path_arguments(
    arguments: dict[str, Any],
    tool_context: ToolContext,
    *,
    path_keys: set[str] | None = None,
) -> list[Path]:
    """Validate filesystem-like arguments against ``allowed_paths``.

    This helper is intended for mandatory pre-dispatch validation of tools
    registered with filesystem access (for example, ``tags=["fs"]``).
    """
    if not tool_context.allowed_paths:
        raise PathIsolationError("No allowed_paths configured — FS access denied")

    candidate_keys = set(path_keys or DEFAULT_PATH_ARGUMENT_KEYS)
    validated: list[Path] = []

    def collect(value: Any) -> list[str | os.PathLike[str]]:
        if isinstance(value, (str, os.PathLike)):
            return [value]
        if isinstance(value, list):
            collected: list[str | os.PathLike[str]] = []
            for item in value:
                collected.extend(collect(item))
            return collected
        if isinstance(value, dict):
            collected = []
            for nested_value in value.values():
                collected.extend(collect(nested_value))
            return collected
        return []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                key_str = str(key)
                if key_str in candidate_keys:
                    for candidate in collect(nested_value):
                        validated.append(validate_path_access(candidate, tool_context))
                walk(nested_value)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)

    walk(arguments)
    return validated


def contains_path_argument(
    arguments: dict[str, Any],
    *,
    path_keys: set[str] | None = None,
) -> bool:
    """Return True when a path-like argument key is present anywhere in payload."""

    candidate_keys = set(path_keys or DEFAULT_PATH_ARGUMENT_KEYS)

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if str(key) in candidate_keys:
                    return True
                if walk(nested_value):
                    return True
            return False
        if isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(arguments)


def build_tool_context(
    *,
    session_id: str,
    trace_id: str,
    agent_id: str,
    allowed_paths: list[str] | None = None,
    allow_symlinks: bool = False,
    metadata: ToolContextMetadata | dict[str, Any] | None = None,
) -> ToolContext:
    """Convenience constructor for ``ToolContext`` with normalized paths."""
    metadata_payload = dict(metadata) if metadata else {}
    resolved_allowed_paths: list[str] = []
    for raw_path in allowed_paths or []:
        resolved = Path(raw_path).resolve(strict=False)
        if not resolved.exists():
            logger.warning(
                "build_tool_context: allowed_path does not exist: %s",
                resolved,
            )
        resolved_allowed_paths.append(str(resolved))
    if resolved_allowed_paths and ToolContextMeta.WORKSPACE_ROOT not in metadata_payload:
        metadata_payload[ToolContextMeta.WORKSPACE_ROOT] = resolved_allowed_paths[0]
    return ToolContext(
        session_id=session_id,
        trace_id=trace_id,
        agent_id=agent_id,
        allowed_paths=resolved_allowed_paths,
        allow_symlinks=allow_symlinks,
        metadata=metadata_payload,
    )


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------



def estimate_text_tokens(
    text: str,
    *,
    estimate_tokens_func: TokenEstimateFunc | None = None,
    model: str | None = None,
    profile: TokenEstimatorProfile = TokenEstimatorProfile.AUTO,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate tokens for a single text fragment."""
    estimator = resolve_token_estimator(
        model=model,
        estimate_tokens_func=estimate_tokens_func,
        profile=profile,
        chars_per_token=chars_per_token,
    )
    return _normalize_token_count(estimator(text))


def resolve_token_estimator(
    *,
    model: str | None = None,
    estimate_tokens_func: TokenEstimateFunc | None = None,
    profile: TokenEstimatorProfile = TokenEstimatorProfile.AUTO,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> TokenEstimateFunc:
    """Resolve a token estimator callable for the given model/profile."""
    if estimate_tokens_func is not None:
        return estimate_tokens_func

    resolved_profile = _resolve_profile(model=model, profile=profile)
    return _resolve_estimator_for_profile(
        model=model,
        resolved_profile=resolved_profile,
        chars_per_token=chars_per_token,
    )


def _resolve_estimator_for_profile(
    *,
    model: str | None,
    resolved_profile: TokenEstimatorProfile,
    chars_per_token: float,
) -> TokenEstimateFunc:
    if resolved_profile in (TokenEstimatorProfile.OPENAI, TokenEstimatorProfile.QWEN3):
        estimator = _resolve_tiktoken_estimator(model=model, profile=resolved_profile)
        if estimator is not None:
            return estimator
        logger.warning(
            "tiktoken unavailable for profile=%s model=%s; falling back to heuristic estimation",
            resolved_profile.value,
            model or "",
        )

    return _make_heuristic_estimator(chars_per_token=chars_per_token)


def estimate_tokens(
    messages: list[Any],
    *,
    estimate_tokens_func: TokenEstimateFunc | None = None,
    model: str | None = None,
    profile: TokenEstimatorProfile = TokenEstimatorProfile.AUTO,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate tokens across message content using custom or built-in tokenizers."""
    if estimate_tokens_func is not None:
        estimator = estimate_tokens_func
        resolved_profile = _resolve_profile(model=model, profile=profile)
    else:
        resolved_profile = _resolve_profile(model=model, profile=profile)
        estimator = _resolve_estimator_for_profile(
            model=model,
            resolved_profile=resolved_profile,
            chars_per_token=chars_per_token,
        )
    is_qwen = resolved_profile == TokenEstimatorProfile.QWEN3

    total_tokens = 0
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            total_tokens += _normalize_token_count(estimator(content))
        elif isinstance(content, list):
            for part in content:
                text = getattr(part, "text", "") or ""
                total_tokens += _normalize_token_count(estimator(text))
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            payload = tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc
            if isinstance(payload, dict):
                total_tokens += _normalize_token_count(
                    estimator(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                )
        # Add Qwen chat template overhead
        if is_qwen:
            total_tokens += QWEN_SPECIAL_TOKENS_PER_MESSAGE

    # Buffer for assistant reply priming (last unclosed tag)
    if is_qwen and messages:
        total_tokens += QWEN_REPLY_PRIMING

    return total_tokens


def estimate_llm_prompt_tokens(
    messages: list[Any],
    *,
    system: str | None = None,
    api_mode: ApiMode = ApiMode.RESPONSES,
    tools: list[Any] | None = None,
    estimate_tokens_func: TokenEstimateFunc | None = None,
    model: str | None = None,
    profile: TokenEstimatorProfile = TokenEstimatorProfile.AUTO,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate a serialized LLM request payload, including system/tools.

    This intentionally counts the request envelope that will be sent to the
    adapter rather than only the raw message text. That keeps auto-compaction
    thresholds closer to actual transport size for system prompts, tool schemas,
    and structured content parts such as ``input_json``.
    """

    estimator = resolve_token_estimator(
        model=model,
        estimate_tokens_func=estimate_tokens_func,
        profile=profile,
        chars_per_token=chars_per_token,
    )
    payload = {
        "api_mode": api_mode.value,
        "system": system or "",
        "messages": _serialize_messages_for_estimation(messages, api_mode=api_mode),
    }
    if tools:
        payload["tools"] = _serialize_tools_for_estimation(tools, api_mode=api_mode)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _normalize_token_count(estimator(serialized))


def _serialize_messages_for_estimation(
    messages: list[Any],
    *,
    api_mode: ApiMode,
) -> list[dict[str, Any]]:
    return serialize_messages_for_api(
        cast(list[Message], messages),
        system=None,
        target_api="responses" if api_mode == ApiMode.RESPONSES else "chat",
    )


def _serialize_content_for_estimation(
    content: Any,
    *,
    api_mode: ApiMode = ApiMode.RESPONSES,
) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        if api_mode == ApiMode.RESPONSES:
            return [{"type": "input_text", "text": content}]
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, Any]] = []
    for part in content:
        part_type = getattr(part, "type", "text")
        if part_type == "text":
            text = getattr(part, "text", "") or ""
            if api_mode == ApiMode.RESPONSES:
                parts.append({"type": "input_text", "text": text})
            else:
                parts.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            image_url = getattr(part, "image_url", None) or {}
            if api_mode == ApiMode.RESPONSES:
                payload = {
                    "type": "input_image",
                    "image_url": image_url.get("url", ""),
                }
                detail = image_url.get("detail")
                if detail:
                    payload["detail"] = detail
                parts.append(payload)
            else:
                parts.append({"type": "image_url", "image_url": image_url})
            continue
        if part_type == "input_json":
            json_text = json.dumps(
                getattr(part, "json_data", None) or {},
                ensure_ascii=True,
                sort_keys=True,
            )
            if api_mode == ApiMode.RESPONSES:
                parts.append({"type": "input_text", "text": json_text})
            else:
                parts.append({"type": "text", "text": json_text})
            continue
        parts.append({"type": str(part_type)})
    return parts


def _serialize_tools_for_estimation(
    tools: list[Any],
    *,
    api_mode: ApiMode,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "to_openai_function"):
            openai_format = tool.to_openai_function()
            if api_mode == ApiMode.RESPONSES and isinstance(openai_format, dict):
                function_data = openai_format.get("function", {})
                if isinstance(function_data, dict):
                    serialized.append(
                        {
                            "type": "function",
                            "name": function_data.get("name", ""),
                            "description": function_data.get("description", ""),
                            "parameters": function_data.get("parameters", {}),
                            "strict": function_data.get("strict", False),
                        }
                    )
                    continue
            if isinstance(openai_format, dict):
                serialized.append(openai_format)
                continue
        if isinstance(tool, dict):
            serialized.append(tool)
    return serialized


def _normalize_token_count(value: int) -> int:
    count = int(value)
    if count < 0:
        raise ValueError("token estimate must be non-negative")
    return count


_MODEL_PREFIX_PROFILE: dict[str, TokenEstimatorProfile] = {
    "gpt": TokenEstimatorProfile.OPENAI,
    "o1": TokenEstimatorProfile.OPENAI,
    "o3": TokenEstimatorProfile.OPENAI,
    "o4": TokenEstimatorProfile.OPENAI,
    "claude": TokenEstimatorProfile.HEURISTIC,
    "qwen": TokenEstimatorProfile.QWEN3,
}


def _resolve_profile(
    *,
    model: str | None,
    profile: TokenEstimatorProfile,
) -> TokenEstimatorProfile:
    if profile != TokenEstimatorProfile.AUTO:
        return profile

    model_name = (model or "").lower()
    # Strip provider namespace prefix (e.g. "Qwen/Qwen2.5-32B" → "qwen2.5-32b")
    base_name = model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name
    if not base_name:
        return TokenEstimatorProfile.OPENAI
    for prefix, resolved in _MODEL_PREFIX_PROFILE.items():
        if base_name.startswith(prefix):
            return resolved
    return TokenEstimatorProfile.HEURISTIC


def _make_heuristic_estimator(*, chars_per_token: float) -> TokenEstimateFunc:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than 0")

    def _estimate(text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / chars_per_token)

    return _estimate


_TIKTOKEN_FULL_ENCODE_LIMIT = 40_000
_TIKTOKEN_SAMPLE_SIZE = 10_000


def _resolve_tiktoken_estimator(
    *,
    model: str | None,
    profile: TokenEstimatorProfile,
) -> TokenEstimateFunc | None:
    if tiktoken is None:
        return None
    encoding_name = _resolve_tiktoken_encoding_name(model=model, profile=profile)
    encoding = _get_tiktoken_encoding(encoding_name)

    def _estimate(text: str) -> int:
        if not text:
            return 0
        if len(text) <= _TIKTOKEN_FULL_ENCODE_LIMIT:
            return len(encoding.encode(text))
        # For large texts, sample + extrapolate to avoid pathological BPE
        # performance on repetitive patterns (e.g. repeated chars can cause
        # O(n^2) merge behavior in tiktoken, turning 200K chars into a
        # 17-second / 7GB operation).
        sample_tokens = len(encoding.encode(text[:_TIKTOKEN_SAMPLE_SIZE]))
        return int(sample_tokens * len(text) / _TIKTOKEN_SAMPLE_SIZE)

    return _estimate


def _resolve_tiktoken_encoding_name(
    *,
    model: str | None,
    profile: TokenEstimatorProfile,
) -> str:
    # For Qwen2/3/3.5 — o200k_base is correct (officially documented)
    if profile == TokenEstimatorProfile.QWEN3:
        return "o200k_base"
    if tiktoken is None:
        return DEFAULT_OPENAI_ENCODING
    if model:
        try:
            encoding_name_for_model = getattr(tiktoken, "encoding_name_for_model", None)
            if callable(encoding_name_for_model):
                return str(encoding_name_for_model(model))
            return tiktoken.encoding_for_model(model).name
        except KeyError:
            logger.debug(
                "No tiktoken model mapping for %s; using %s",
                model, DEFAULT_OPENAI_ENCODING,
            )
    return DEFAULT_OPENAI_ENCODING


@lru_cache(maxsize=8)
def _get_tiktoken_encoding(name: str) -> Any:
    if tiktoken is None:
        raise RuntimeError("tiktoken is not available")
    return tiktoken.get_encoding(name)
