"""3-layer context compression.

Layer 1: micro_compact  — replace old tool results with placeholders each turn.
Layer 2: auto_compact   — LLM summarization when token estimate exceeds threshold.
Layer 3: manual_compact — explicit trigger via tool or hook call.

All thresholds are configurable on AgentConfig; no hard-coded assumptions
about "infinite" context. Defaults are optimised for local models with 40K
context windows.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .context import estimate_llm_prompt_tokens
from .events import (
    EV_COMPACTION_LLM_DELTA,
    EV_COMPACTION_LLM_END,
    EV_COMPACTION_LLM_START,
    EV_COMPACTION_SUMMARY_PARSE_END,
    EV_COMPACTION_SUMMARY_PARSE_FAILED,
    EV_COMPACTION_SUMMARY_PARSE_START,
    EventBus,
)
from .constants import (
    AUTO_COMPACT_MAX_TOKENS,
    AUTO_COMPACT_SYSTEM_PROMPT,
    AUTO_COMPACT_TEMPERATURE,
    AUTO_COMPACT_USER_TEMPLATE,
    COMPACTION_SUMMARY_MARKER,
    DEFAULT_AUTO_COMPACT_KEEP_TRAILING,
    DEFAULT_AUTO_COMPACT_TIMEOUT_SECONDS,
    DEFAULT_AUTO_COMPACT_THRESHOLD,
    TRANSCRIPT_CONTENT_LIMIT,
    TRANSCRIPT_JSON_SUMMARY_LIMIT,
    TRANSCRIPT_PROTECT_LIMIT,
)
from .json_utils import structured_json_candidates
from .orchestrator_utils import (
    build_llm_request_kwargs,
    event_context_payload,
    resolve_effective_llm_config,
    string_preview,
)
from .types import AgentContext, ApiMode, CompactionSummary, Message, RunKind, get_text_content

if TYPE_CHECKING:
    from .protocols import LLMClient
    from .types import AgentConfig

logger = logging.getLogger(__name__)

AUTO_COMPACT_RETRY_COUNT = 2
AUTO_COMPACT_RETRY_DELAY_SECONDS = 0.5


def _force_no_thinking_kwargs(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ensure compaction requests disable reasoning-preface outputs."""
    merged = dict(request_kwargs)
    raw_extra_body = merged.get("extra_body")
    extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
    raw_chat_template_kwargs = extra_body.get("chat_template_kwargs")
    chat_template_kwargs = (
        dict(raw_chat_template_kwargs) if isinstance(raw_chat_template_kwargs, dict) else {}
    )
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    merged["extra_body"] = extra_body
    return merged


def _content_to_text(content: str | list[Any] | None) -> str:
    """Normalize message content into plain text for size checks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text is not None:
            parts.append(text)
            continue
        part_type = getattr(part, "type", "unknown")
        parts.append(f"[{part_type}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Layer 1: micro_compact
# ---------------------------------------------------------------------------


def micro_compact(
    messages: list[Message],
    *,
    keep_recent: int = 2,
    max_tool_result_size: int = 3_000,
) -> tuple[list[Message], int]:
    """Replace old tool-result messages with compact placeholders.

    Args:
        messages: Current message list. The original list is not mutated.
        keep_recent: Number of most-recent tool results to keep verbatim.
        max_tool_result_size: Max chars to keep for any single result.

    Returns:
        (new_messages, count_compacted) where count_compacted is the number
        of messages that were replaced.
    """
    tool_message_count = sum(1 for message in messages if message.role == "tool")
    compact_tool_count = max(0, tool_message_count - keep_recent)

    if compact_tool_count == 0:
        return messages, 0

    compacted_count = 0
    new_messages: list[Message] = []
    seen_tool_messages = 0

    for msg in messages:
        if msg.role == "tool" and seen_tool_messages < compact_tool_count:
            content = _content_to_text(msg.content)
            length = len(content)
            placeholder = (
                f"[micro_compact: {msg.name or msg.tool_call_id} "
                f"returned {length} chars]"
            )
            new_messages.append(
                Message(
                    role="tool",
                    content=placeholder,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
            compacted_count += 1
            seen_tool_messages += 1
        elif msg.role == "tool":
            seen_tool_messages += 1
            # Keep-recent: still truncate if oversized
            content = _content_to_text(msg.content)
            if len(content) > max_tool_result_size:
                truncated = content[:max_tool_result_size]
                placeholder = (
                    f"{truncated}\n[truncated: original {len(content)} chars]"
                )
                new_messages.append(msg.model_copy(update={"content": placeholder}))
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)

    logger.debug("micro_compact: compacted=%d messages", compacted_count)
    return new_messages, compacted_count


# ---------------------------------------------------------------------------
# Layer 2: auto_compact
# ---------------------------------------------------------------------------



async def auto_compact(
    messages: list[Message],
    *,
    llm_client: "LLMClient",
    model: str,
    threshold_tokens: int | None = None,
    keep_trailing: int | None = None,
    config: "AgentConfig | None" = None,
    precomputed_tokens: int | None = None,
    run_kind: RunKind = RunKind.LEADER,
    event_bus: EventBus | None = None,
    context: AgentContext | None = None,
) -> tuple[list[Message], CompactionSummary | None, bool]:
    """Summarize conversation when token estimate exceeds threshold.

    Returns (new_messages, summary, parse_success) where new_messages contains
    the summary injected as a system message plus any trailing messages.
    If threshold not reached, returns (messages, None, True) unchanged.
    parse_success is False when recoverable LLM/JSON failures force a degraded
    summary. Cancellation is always re-raised. Transient summarization errors
    get one bounded retry with a short delay before degrading.
    """
    compaction_started = asyncio.get_running_loop().time()
    effective_threshold_tokens = threshold_tokens
    effective_keep_trailing = keep_trailing
    compact_max_tokens = AUTO_COMPACT_MAX_TOKENS
    compact_temperature = AUTO_COMPACT_TEMPERATURE
    resolved_config = None
    if config is not None:
        resolved_config = resolve_effective_llm_config(
            config,
            run_kind=run_kind,
            call_purpose="auto_compact",
        )
        if effective_threshold_tokens is None:
            effective_threshold_tokens = config.auto_compact_threshold
        if effective_keep_trailing is None:
            effective_keep_trailing = config.auto_compact_keep_trailing
        compact_max_tokens = config.auto_compact_max_tokens
        compact_temperature = config.auto_compact_temperature
    if effective_threshold_tokens is None:
        effective_threshold_tokens = DEFAULT_AUTO_COMPACT_THRESHOLD
    if effective_keep_trailing is None:
        effective_keep_trailing = DEFAULT_AUTO_COMPACT_KEEP_TRAILING

    estimate_kwargs: dict[str, Any] = {}
    if config is not None:
        estimate_kwargs = {
            "estimate_tokens_func": config.estimate_tokens_func,
            "model": config.model,
            "profile": config.token_estimator_profile,
            "chars_per_token": config.chars_per_token_estimate,
        }
    current_tokens = (
        precomputed_tokens
        if precomputed_tokens is not None
        else estimate_llm_prompt_tokens(
            messages,
            system=(config.system_prompt if config is not None else None),
            api_mode=(config.api_mode if config is not None else ApiMode.RESPONSES),
            **estimate_kwargs,
        )
    )
    if current_tokens < effective_threshold_tokens:
        return messages, None, True

    logger.info(
        "auto_compact triggered: estimated_tokens=%d threshold=%d",
        current_tokens,
        effective_threshold_tokens,
    )

    compaction_model = (
        config.compaction_model
        if config is not None and config.compaction_model
        else model
    )

    # Build transcript for summarization
    transcript = _build_transcript(messages, limit=TRANSCRIPT_PROTECT_LIMIT)
    summary_messages = [
        Message(role="user", content=AUTO_COMPACT_USER_TEMPLATE.format(
            transcript=transcript
        )),
    ]

    parse_success = True
    try:
        timeout_seconds = (
            config.auto_compact_timeout_seconds
            if config is not None
            else DEFAULT_AUTO_COMPACT_TIMEOUT_SECONDS
        )
        summary: CompactionSummary | None = None
        last_error: Exception | None = None
        for attempt in range(1, AUTO_COMPACT_RETRY_COUNT + 1):
            try:
                stream_stats = {"text_chars": 0, "reasoning_chars": 0, "delta_count": 0}
                stream_event_callback = None
                if event_bus is not None and context is not None:
                    await event_bus.emit_simple(
                        EV_COMPACTION_LLM_START,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="auto_compact",
                            scope_id=f"compact:{context.request_id}:{attempt}",
                            attempt=attempt,
                            source_component="compression.auto",
                        ),
                        model=compaction_model,
                        max_tokens=compact_max_tokens,
                        temperature=compact_temperature,
                    )

                    async def _stream_callback(delta: dict[str, Any]) -> None:
                        text = delta.get("text")
                        if not isinstance(text, str) or not text:
                            return
                        stream_stats["delta_count"] += 1
                        kind = str(delta.get("kind", "text"))
                        if kind == "reasoning":
                            stream_stats["reasoning_chars"] += len(text)
                        else:
                            stream_stats["text_chars"] += len(text)
                        await event_bus.emit_simple(
                            EV_COMPACTION_LLM_DELTA,
                            **event_context_payload(
                                context,
                                run_kind=run_kind,
                                phase="auto_compact",
                                scope_id=f"compact:{context.request_id}:{attempt}",
                                attempt=attempt,
                                source_component="compression.auto",
                            ),
                            kind=kind,
                            text=text,
                            chars=len(text),
                        )

                    stream_event_callback = _stream_callback
                llm_started = asyncio.get_running_loop().time()
                complete_structured = getattr(llm_client, "complete_structured", None)
                compaction_request_kwargs = _force_no_thinking_kwargs(
                    build_llm_request_kwargs(resolved_config)
                    if resolved_config is not None
                    else {}
                )
                if callable(complete_structured) and inspect.iscoroutinefunction(complete_structured):
                    summary = await asyncio.wait_for(
                        complete_structured(
                            messages=summary_messages,
                            schema=CompactionSummary,
                            system=AUTO_COMPACT_SYSTEM_PROMPT,
                            model=compaction_model,
                            temperature=compact_temperature,
                            max_tokens=compact_max_tokens,
                            api_mode=(
                                resolved_config.api_mode
                                if resolved_config is not None
                                else ApiMode.RESPONSES
                            ),
                            **compaction_request_kwargs,
                        ),
                        timeout=timeout_seconds,
                    )
                    summary = CompactionSummary.model_validate(summary)
                else:
                    reply = await asyncio.wait_for(
                        llm_client.complete(
                            messages=summary_messages,
                            tools=[],
                            system=AUTO_COMPACT_SYSTEM_PROMPT,
                            model=compaction_model,
                            temperature=compact_temperature,
                            max_tokens=compact_max_tokens,
                            stream=stream_event_callback is not None,
                            stream_event_callback=stream_event_callback,
                            **compaction_request_kwargs,
                        ),
                        timeout=timeout_seconds,
                    )
                    raw_content = (
                        reply.content if isinstance(reply.content, str)
                        else str(reply.content or "")
                    )
                    summary, parse_success = _parse_summary(raw_content)
                if event_bus is not None and context is not None:
                    await event_bus.emit_simple(
                        EV_COMPACTION_LLM_END,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="auto_compact",
                            scope_id=f"compact:{context.request_id}:{attempt}",
                            attempt=attempt,
                            source_component="compression.auto",
                        ),
                        latency_ms=(asyncio.get_running_loop().time() - llm_started) * 1000,
                        usage=None,
                        raw_output_preview=string_preview(summary.model_dump_json(), limit=600),
                        delta_count=stream_stats["delta_count"],
                        text_chars=stream_stats["text_chars"],
                        reasoning_chars=stream_stats["reasoning_chars"],
                )
                break
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "auto_compact attempt failed: attempt=%d/%d error=%s",
                    attempt,
                    AUTO_COMPACT_RETRY_COUNT,
                    type(exc).__name__,
                )
                if attempt < AUTO_COMPACT_RETRY_COUNT:
                    await asyncio.sleep(AUTO_COMPACT_RETRY_DELAY_SECONDS)
        if summary is None:
            assert last_error is not None
            raise last_error
        if event_bus is not None and context is not None:
            await event_bus.emit_simple(
                EV_COMPACTION_SUMMARY_PARSE_START,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="auto_compact",
                    scope_id=f"compact:{context.request_id}:parse",
                    source_component="compression.auto",
                ),
                schema_name="CompactionSummary",
            )
        if event_bus is not None and context is not None:
            await event_bus.emit_simple(
                EV_COMPACTION_SUMMARY_PARSE_END,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="auto_compact",
                    scope_id=f"compact:{context.request_id}:parse",
                    source_component="compression.auto",
                ),
                parse_success=parse_success,
                current_goal_preview=string_preview(summary.current_goal, limit=300),
            )
        parse_success = bool(parse_success)
    except asyncio.CancelledError:
        raise
    except (
        asyncio.TimeoutError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ):
        logger.exception("auto_compact: LLM summarization failed; using minimal summary")
        summary = CompactionSummary(current_goal="[summarization failed; continuing]")
        parse_success = False
        if event_bus is not None and context is not None:
            await event_bus.emit_simple(
                EV_COMPACTION_SUMMARY_PARSE_FAILED,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="auto_compact",
                    scope_id=f"compact:{context.request_id}:parse",
                    source_component="compression.auto",
                ),
                raw_preview="[summarization failed; continuing]",
                fallback_applied=True,
            )
    if not parse_success and event_bus is not None and context is not None:
        await event_bus.emit_simple(
            EV_COMPACTION_SUMMARY_PARSE_FAILED,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="auto_compact",
                scope_id=f"compact:{context.request_id}:parse",
                source_component="compression.auto",
            ),
            raw_preview=string_preview(summary.current_goal, limit=300),
            fallback_applied=True,
        )

    new_messages = [Message(role="system", content="")]
    # Keep trailing non-summary messages for continuity and avoid recursive
    # inclusion of previous compaction summaries.
    trailing_source = [m for m in messages if not _is_compaction_summary_message(m)]
    effective_trailing = max(0, effective_keep_trailing)
    if effective_trailing == 0:
        trailing = []
    elif len(trailing_source) >= effective_trailing:
        trailing = trailing_source[-effective_trailing:]
    else:
        trailing = trailing_source
    new_messages.extend(trailing)

    summary = summary.model_copy(
        update={
            "original_count": len(messages),
            "compacted_count": len(new_messages),
            "tokens_saved": max(
                current_tokens
                - estimate_llm_prompt_tokens(
                    new_messages,
                    system=(config.system_prompt if config is not None else None),
                    api_mode=(config.api_mode if config is not None else ApiMode.RESPONSES),
                    **estimate_kwargs,
                ),
                0,
            ),
            "duration_ms": (asyncio.get_running_loop().time() - compaction_started) * 1000,
        },
        deep=True,
    )
    new_messages[0] = Message(role="system", content=summary.model_dump_json())

    logger.info(
        "auto_compact done: before=%d after=%d messages parse_ok=%s",
        len(messages),
        len(new_messages),
        parse_success,
    )
    return new_messages, summary, parse_success




def _truncate_for_transcript(text: str, limit: int = TRANSCRIPT_CONTENT_LIMIT) -> str:
    """Truncate content for transcript at a meaningful boundary.

    For JSON content: parse and re-serialize to a compact summary so
    the LLM summarizer receives valid, coherent data rather than mid-stream
    truncated bytes.

    For plain text: truncate at the last newline within the limit to
    preserve logical line boundaries.

    Always appends a [truncated] marker when content is shortened.
    """
    if len(text) <= limit:
        return text

    # Fast-path: try to parse as JSON and produce a compact, valid excerpt
    stripped = text.lstrip()
    if stripped and stripped[0] in ("{", "["):
        try:
            parsed = json.loads(text)
            compact = json.dumps(
                parsed,
                ensure_ascii=False,
                indent=None,
                separators=(",", ":"),
            )
            if len(compact) <= limit:
                return compact  # fits without truncation after compaction
            # Still too large — summarize structure instead of raw bytes
            return _json_structure_summary(
                parsed, limit=TRANSCRIPT_JSON_SUMMARY_LIMIT
            )
        except (json.JSONDecodeError, ValueError):
            logger.debug("Content is not valid JSON, using plain-text truncation")

    # Plain-text path: break at last newline within limit
    window = text[:limit]
    last_newline = window.rfind("\n")
    cut = (
        window[: last_newline + 1]
        if last_newline > limit // 2
        else window
    )
    return cut + f" … [truncated, {len(text) - len(cut)} chars omitted]"


def _json_structure_summary(value: Any, *, limit: int) -> str:
    """Produce a human-readable structural summary of a JSON value.

    Instead of raw truncated bytes, the LLM summarizer sees something like:
      {keys: [result, status, rows], rows: [<150 items>], ...}
    This preserves semantic shape while staying within the token budget.
    """

    def _trim(fragment: str, budget: int) -> str:
        if budget <= 0:
            return ""
        if len(fragment) <= budget:
            return fragment
        if budget <= 1:
            return "…"
        return fragment[: budget - 1] + "…"

    def _summarize(v: Any, *, depth: int = 0, budget: int) -> str:
        if budget <= 0:
            return ""
        if depth > 3:
            return _trim("…", budget)
        if isinstance(v, dict):
            keys = list(v.keys())
            if not keys:
                return _trim("{}", budget)
            parts = ["{"]
            remaining = budget - 1
            shown_count = min(len(keys), 5)
            for index, key in enumerate(keys[:shown_count]):
                if remaining <= 0:
                    break
                if index > 0:
                    separator = ", "
                    parts.append(_trim(separator, remaining))
                    remaining -= len(parts[-1])
                    if remaining <= 0:
                        break
                key_repr = _trim(str(key), min(remaining, 40))
                key_fragment = f"{key_repr}: "
                key_fragment = _trim(key_fragment, remaining)
                parts.append(key_fragment)
                remaining -= len(key_fragment)
                if remaining <= 0:
                    break
                value_summary = _summarize(v[key], depth=depth + 1, budget=remaining)
                parts.append(value_summary)
                remaining -= len(value_summary)
            if len(keys) > shown_count and remaining > 0:
                extra = _trim(f", … +{len(keys) - shown_count} more keys", remaining)
                parts.append(extra)
                remaining -= len(extra)
            if remaining > 0:
                parts.append("}")
            return "".join(parts)
        if isinstance(v, list):
            if not v:
                return _trim("[]", budget)
            sample_budget = max(1, budget - 6)
            sample = _summarize(v[0], depth=depth + 1, budget=sample_budget)
            rendered = f"[{sample}]"
            if len(v) > 1:
                rendered = f"[{sample}, …×{len(v)}]"
            return _trim(rendered, budget)
        if isinstance(v, str):
            rendered = repr(v[:60] + "…") if len(v) > 60 else repr(v)
            return _trim(rendered, budget)
        return _trim(repr(v), budget)

    prefix = "<JSON structure> "
    if limit <= len(prefix):
        return _trim(prefix, limit)
    return prefix + _summarize(value, budget=limit - len(prefix))


def _build_transcript(
    messages: list[Message], *, limit: int = TRANSCRIPT_PROTECT_LIMIT
) -> str:
    return _build_transcript_limited(messages, limit=limit)


def _summarize_tool_call_arguments(raw_arguments: Any) -> str:
    if raw_arguments in (None, "", {}, []):
        return ""
    candidate = raw_arguments
    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return ""
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            return _truncate_for_transcript(stripped, limit=TRANSCRIPT_JSON_SUMMARY_LIMIT)
    if isinstance(candidate, (dict, list)):
        return _json_structure_summary(candidate, limit=TRANSCRIPT_JSON_SUMMARY_LIMIT)
    return _truncate_for_transcript(str(candidate), limit=TRANSCRIPT_JSON_SUMMARY_LIMIT)


def _summarize_tool_calls_for_transcript(message: Message) -> str:
    tool_calls = message.tool_calls or []
    rendered: list[str] = []
    for tool_call in tool_calls:
        function = tool_call.function
        name = str(function.name or "unknown")
        arguments = _summarize_tool_call_arguments(function.arguments)
        rendered.append(f"{name}({arguments})" if arguments else name)
    return ", ".join(rendered)


def _build_transcript_limited(messages: list[Message], *, limit: int) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.role
        content = get_text_content(msg)
        line = f"[{role.upper()}]: {_truncate_for_transcript(content)}"
        tool_calls_summary = _summarize_tool_calls_for_transcript(msg)
        if tool_calls_summary:
            line += f" [tool_calls: {tool_calls_summary}]"
        lines.append(line)

    transcript = "\n".join(lines)
    if len(transcript) <= limit:
        return transcript

    omission_marker = "\n...[middle omitted for compaction]...\n"
    marker_len = len(omission_marker)
    if limit <= marker_len + 16:
        return _truncate_for_transcript(transcript, limit=limit)

    head_budget = max(1, (limit - marker_len) // 2)
    tail_budget = max(1, limit - marker_len - head_budget)

    head_parts: list[str] = []
    head_len = 0
    head_index = -1
    for index, line in enumerate(lines):
        separator_len = 1 if head_parts else 0
        if head_len + separator_len + len(line) > head_budget:
            break
        head_parts.append(line)
        head_len += separator_len + len(line)
        head_index = index

    tail_parts: list[str] = []
    tail_len = 0
    tail_index = len(lines)
    for index in range(len(lines) - 1, head_index, -1):
        line = lines[index]
        separator_len = 1 if tail_parts else 0
        if tail_len + separator_len + len(line) > tail_budget:
            break
        tail_parts.append(line)
        tail_len += separator_len + len(line)
        tail_index = index

    if not head_parts or tail_index <= head_index + 1:
        fallback = _truncate_for_transcript(transcript, limit=limit)
        return fallback[:limit] if len(fallback) > limit else fallback

    tail_parts.reverse()
    combined = "\n".join(head_parts) + omission_marker + "\n".join(tail_parts)
    if len(combined) <= limit:
        return combined
    fallback = _truncate_for_transcript(combined, limit=limit)
    return fallback[:limit] if len(fallback) > limit else fallback


def _parse_summary(raw: str) -> tuple[CompactionSummary, bool]:
    """Parse LLM output into a CompactionSummary.

    Returns (summary, success) where success=False indicates the LLM
    returned invalid JSON and the summary is a degraded fallback.
    """
    for candidate in structured_json_candidates(raw):
        try:
            data: dict[str, Any] = json.loads(candidate)
            return CompactionSummary.model_validate(data), True
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            continue
    logger.warning(
        "auto_compact: LLM returned invalid JSON for summary; "
        "using degraded fallback (raw length=%d)",
        len(raw),
    )
    return CompactionSummary(
        current_goal="[summary_parse_failed] " + raw[:480]
    ), False


def _is_compaction_summary_message(message: Message) -> bool:
    """Check whether a message is a compaction summary."""
    if message.role != "system" or not isinstance(message.content, str):
        return False
    if COMPACTION_SUMMARY_MARKER not in message.content:
        return False
    try:
        data = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and data.get("marker") == COMPACTION_SUMMARY_MARKER


# ---------------------------------------------------------------------------
# Layer 3: manual_compact
# ---------------------------------------------------------------------------


async def manual_compact(
    messages: list[Message],
    *,
    llm_client: "LLMClient",
    model: str,
    config: "AgentConfig | None" = None,
    run_kind: RunKind = RunKind.LEADER,
    event_bus: EventBus | None = None,
    context: AgentContext | None = None,
) -> tuple[list[Message], CompactionSummary]:
    """Force compression regardless of current token count.

    Used by explicit tool call or hook. Always returns (new_messages, summary).
    """
    new_messages, summary, _parse_ok = await auto_compact(
        messages,
        llm_client=llm_client,
        model=model,
        threshold_tokens=0,  # force
        config=config,
        run_kind=run_kind,
        event_bus=event_bus,
        context=context,
    )
    return new_messages, summary or CompactionSummary()


# ---------------------------------------------------------------------------
# Composite compressor: applies all three layers in order
# ---------------------------------------------------------------------------


class ContextCompressor:
    """Applies micro → auto → manual compression based on config."""

    def __init__(self, llm_client: "LLMClient", model: str) -> None:
        self._llm = llm_client
        self._model = model

    def apply_micro(
        self,
        messages: list[Message],
        config: "AgentConfig",
    ) -> tuple[list[Message], int]:
        """Apply micro_compact layer.

        This helper is intentionally synchronous because ``micro_compact`` is a
        CPU-light in-process transformation. Keeping one call site prevents
        drift between the direct helper and the compressor wrapper.
        """
        return micro_compact(
            messages,
            keep_recent=config.micro_compact_keep_recent,
            max_tool_result_size=config.max_tool_result_size,
        )

    async def apply_auto(
        self,
        messages: list[Message],
        config: "AgentConfig",
        *,
        precomputed_tokens: int | None = None,
        run_kind: RunKind = RunKind.LEADER,
        event_bus: EventBus | None = None,
        context: AgentContext | None = None,
    ) -> tuple[list[Message], CompactionSummary | None, bool]:
        """Apply auto_compact layer if threshold exceeded."""
        return await auto_compact(
            messages,
            llm_client=self._llm,
            model=config.model,
            config=config,
            precomputed_tokens=precomputed_tokens,
            run_kind=run_kind,
            event_bus=event_bus,
            context=context,
        )

    async def apply_manual(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        config: "AgentConfig | None" = None,
        run_kind: RunKind = RunKind.LEADER,
        event_bus: EventBus | None = None,
        context: AgentContext | None = None,
    ) -> tuple[list[Message], CompactionSummary]:
        """Force manual compression."""
        effective_model = model or self._model
        if config is not None:
            effective_model = config.compaction_model or config.model
        return await manual_compact(
            messages,
            llm_client=self._llm,
            model=effective_model,
            config=config,
            run_kind=run_kind,
            event_bus=event_bus,
            context=context,
        )
