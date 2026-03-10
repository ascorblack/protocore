from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from protocore import PathIsolationError
from protocore.constants import DEFAULT_OPENAI_ENCODING
from protocore import context as core_context
from protocore.types import ApiMode, TokenEstimatorProfile, ToolContext


class _ToolWithOpenAIFunction:
    def to_openai_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search docs",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                "strict": True,
            },
        }


class TestContextSerializationForEstimation:
    def test_serialize_content_for_estimation_handles_modalities_and_unknown_parts(self) -> None:
        parts = [
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="image_url", image_url={"url": "https://img", "detail": "high"}),
            SimpleNamespace(type="input_json", json_data={"b": 2, "a": 1}),
            SimpleNamespace(type="other", payload="ignored"),
        ]

        resp = core_context._serialize_content_for_estimation(parts, api_mode=ApiMode.RESPONSES)
        chat = core_context._serialize_content_for_estimation(parts, api_mode=ApiMode.CHAT_COMPLETIONS)

        assert resp[0] == {"type": "input_text", "text": "hello"}
        assert resp[1] == {"type": "input_image", "image_url": "https://img", "detail": "high"}
        assert resp[2] == {"type": "input_text", "text": '{"a": 1, "b": 2}'}
        assert resp[3] == {"type": "other"}
        assert chat[0] == {"type": "text", "text": "hello"}
        assert chat[1] == {"type": "image_url", "image_url": {"url": "https://img", "detail": "high"}}
        assert chat[2] == {"type": "text", "text": '{"a": 1, "b": 2}'}

    def test_serialize_content_for_estimation_handles_none_string_and_non_list(self) -> None:
        assert core_context._serialize_content_for_estimation(None) == ""
        assert core_context._serialize_content_for_estimation("hi", api_mode=ApiMode.RESPONSES) == [
            {"type": "input_text", "text": "hi"}
        ]
        assert core_context._serialize_content_for_estimation("hi", api_mode=ApiMode.CHAT_COMPLETIONS) == "hi"
        assert core_context._serialize_content_for_estimation(123, api_mode=ApiMode.CHAT_COMPLETIONS) == "123"

    def test_serialize_tools_for_estimation_flattens_for_responses_and_preserves_chat_shape(self) -> None:
        tool = _ToolWithOpenAIFunction()
        raw = {"type": "web_search", "provider": "custom"}

        resp_tools = core_context._serialize_tools_for_estimation([tool, raw], api_mode=ApiMode.RESPONSES)
        chat_tools = core_context._serialize_tools_for_estimation([tool, raw], api_mode=ApiMode.CHAT_COMPLETIONS)

        assert resp_tools[0]["name"] == "search"
        assert "function" not in resp_tools[0]
        assert resp_tools[1] == raw
        assert chat_tools[0]["function"]["name"] == "search"
        assert chat_tools[1] == raw


class TestCancellationContextCore:
    @pytest.mark.asyncio
    async def test_cancelled_context_sets_current_loop_event_and_is_idempotent(self) -> None:
        ctx = core_context.CancellationContext()
        ctx.cancel("first")
        ctx.cancel("second")

        event = ctx._event_for_current_loop()

        assert event.is_set()
        assert ctx.reason == "first"
        assert ctx.is_cancelled is True


class TestPathValidationCore:
    def test_validate_path_access_wraps_resolve_failures(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tool_context = ToolContext(allowed_paths=[str(tmp_path)])
        original_resolve = Path.resolve

        def patched_resolve(self: Path, strict: bool = False) -> Path:
            if self.name == "boom.txt":
                raise OSError("resolve failed")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", patched_resolve)

        with pytest.raises(PathIsolationError, match="Failed to resolve path"):
            core_context.validate_path_access("boom.txt", tool_context)


class TestTiktokenResolutionCore:
    def test_resolve_tiktoken_encoding_name_falls_back_to_default_on_keyerror(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeTiktoken:
            @staticmethod
            def encoding_name_for_model(model: str) -> str:
                _ = model
                raise KeyError("unknown model")

        monkeypatch.setattr(core_context, "tiktoken", FakeTiktoken)

        assert (
            core_context._resolve_tiktoken_encoding_name(
                model="unknown-model",
                profile=TokenEstimatorProfile.OPENAI,
            )
            == DEFAULT_OPENAI_ENCODING
        )
