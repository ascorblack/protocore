from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protocore import (
    AgentConfig,
    AgentEnvelope,
    ApiMode,
    ExecutionMode,
    ExecutionStatus,
    ExecutionReport,
    OpenAILLMClient,
    Result,
    StopReason,
    SubagentResult,
    make_agent_context,
    make_execution_report,
)


def test_report_version_stays_stable() -> None:
    ctx = make_agent_context(
        config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
    )
    report = make_execution_report(context=ctx)
    assert report.report_version == "1.1"


def test_envelope_parse_helper_records_minor_version_warning() -> None:
    report = make_execution_report(
        context=make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
    )
    envelope = AgentEnvelope.parse_with_report(
        {
            "protocol_version": "1.1",
            "message_type": "task",
            "trace_id": "trace",
            "session_id": "session",
            "sender": {"agent_id": "leader", "role": "leader"},
            "recipient": {"agent_id": "sub", "role": "subagent"},
            "payload": {"task": "hello"},
            "meta": {"created_at": "2026-03-07T00:00:00+00:00", "protocol_version": "1.1"},
        },
        report,
    )
    assert envelope.protocol_version == "1.1"
    assert any(w.startswith("protocol_minor_version_mismatch") for w in report.warnings)


def test_ingress_parse_envelope_is_canonical_api() -> None:
    """parse_envelope is canonical ingress; direct validate does not auto-append report warnings."""
    from protocore import parse_envelope

    report = make_execution_report(
        context=make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
    )
    data = {
        "protocol_version": "1.1",  # minor version differs from PROTOCOL_VERSION
        "message_type": "task",
        "trace_id": "trace",
        "session_id": "session",
        "sender": {"agent_id": "leader", "role": "leader"},
        "recipient": {"agent_id": "sub", "role": "subagent"},
        "payload": {"task": "hello"},
        "meta": {"created_at": "2026-03-07T00:00:00+00:00", "protocol_version": "1.1"},
    }

    # Ingress API records warning
    envelope = parse_envelope(data, report)
    assert envelope.protocol_version == "1.1"
    assert any(w.startswith("protocol_minor_version_mismatch") for w in report.warnings)

    # Direct model_validate keeps warning inside envelope meta; report warning still needs apply/ingress.
    report2 = make_execution_report(
        context=make_agent_context(
            config=AgentConfig(agent_id="agent2", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
    )
    envelope_direct = AgentEnvelope.model_validate(data)
    assert any(
        warning.startswith("protocol_minor_version_mismatch")
        for warning in envelope_direct.meta.compatibility_warnings
    )
    envelope_direct.apply_version_compatibility(report2)
    assert any(w.startswith("protocol_minor_version_mismatch") for w in report2.warnings)


def test_direct_model_validate_stores_minor_warning_in_envelope_meta() -> None:
    data = {
        "protocol_version": "1.1",
        "message_type": "task",
        "trace_id": "trace",
        "session_id": "session",
        "sender": {"agent_id": "leader", "role": "leader"},
        "recipient": {"agent_id": "sub", "role": "subagent"},
        "payload": {"task": "hello"},
    }
    envelope = AgentEnvelope.model_validate(data)
    assert any(
        warning.startswith("protocol_minor_version_mismatch")
        for warning in envelope.meta.compatibility_warnings
    )


def test_ingress_parse_envelope_accepts_json_string_and_bytes() -> None:
    from protocore import parse_envelope

    data = {
        "protocol_version": "1.1",
        "message_type": "task",
        "trace_id": "trace",
        "session_id": "session",
        "sender": {"agent_id": "leader", "role": "leader"},
        "recipient": {"agent_id": "sub", "role": "subagent"},
        "payload": {"task": "hello"},
    }

    report_from_str = make_execution_report(
        context=make_agent_context(
            config=AgentConfig(agent_id="agent-str", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
    )
    envelope_from_str = parse_envelope(AgentEnvelope.model_validate(data).model_dump_json(), report_from_str)
    assert envelope_from_str.protocol_version == "1.1"
    assert any(w.startswith("protocol_minor_version_mismatch") for w in report_from_str.warnings)

    report_from_bytes = make_execution_report(
        context=make_agent_context(
            config=AgentConfig(agent_id="agent-bytes", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
    )
    envelope_from_bytes = parse_envelope(AgentEnvelope.model_validate(data).model_dump_json().encode(), report_from_bytes)
    assert envelope_from_bytes.protocol_version == "1.1"
    assert any(w.startswith("protocol_minor_version_mismatch") for w in report_from_bytes.warnings)


def test_ingress_parse_envelope_revalidates_existing_envelope_and_applies_warnings() -> None:
    from protocore import parse_envelope

    envelope = AgentEnvelope.model_validate(
        {
            "protocol_version": "1.1",
            "message_type": "task",
            "trace_id": "trace",
            "session_id": "session",
            "sender": {"agent_id": "leader", "role": "leader"},
            "recipient": {"agent_id": "sub", "role": "subagent"},
            "payload": {"task": "hello"},
        }
    )
    report = make_execution_report(
        context=make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
    )

    parsed = parse_envelope(envelope, report)

    assert parsed is not envelope
    assert parsed.model_dump() == envelope.model_dump()
    assert any(w.startswith("protocol_minor_version_mismatch") for w in report.warnings)


@pytest.mark.asyncio
async def test_structured_output_falls_back_in_both_api_modes() -> None:
    for mode in (ApiMode.RESPONSES, ApiMode.CHAT_COMPLETIONS):
        client = OpenAILLMClient(api_key="test", api_mode=mode)
        if mode == ApiMode.RESPONSES:
            mock_response = MagicMock()
            mock_response.output_text = "not valid json"
            with patch.object(client._client.responses, "create", new_callable=AsyncMock) as create:
                create.return_value = mock_response
                result = await client.complete_structured(
                    messages=[],
                    schema=SubagentResult,
                )
        else:
            msg_obj = MagicMock()
            msg_obj.content = "not valid json"
            msg_obj.tool_calls = None
            choice = MagicMock()
            choice.message = msg_obj
            mock_response = MagicMock()
            mock_response.choices = [choice]
            with patch.object(
                client._client.chat.completions, "create", new_callable=AsyncMock
            ) as create:
                create.return_value = mock_response
                result = await client.complete_structured(
                    messages=[],
                    schema=SubagentResult,
                )
        assert isinstance(result, SubagentResult)
        assert result.status.value == "partial"


def test_core_layout_modules_are_importable() -> None:
    from protocore import compression, context, events, factories, ingress, orchestrator, protocols, registry, types
    from protocore.hooks import manager, specs

    assert compression is not None
    assert context is not None
    assert events is not None
    assert factories is not None
    assert ingress is not None
    assert orchestrator is not None
    assert protocols is not None
    assert registry is not None
    assert types is not None
    assert manager is not None
    assert specs is not None


@pytest.mark.asyncio
async def test_optional_langgraph_adapter_integration() -> None:
    """Optional integration: a real langgraph-backed adapter can satisfy WorkflowEngine."""
    graph_mod = pytest.importorskip("langgraph.graph")
    from protocore import WorkflowEngine
    from protocore.types import WorkflowDefinition

    StateGraph = graph_mod.StateGraph
    START = graph_mod.START
    END = graph_mod.END

    class LangGraphAdapter:
        def __init__(self) -> None:
            graph = StateGraph(dict)
            graph.add_node("work", lambda state: {"result": "ok"})
            graph.add_edge(START, "work")
            graph.add_edge("work", END)
            self._compiled = graph.compile()

        async def run(
            self, workflow: WorkflowDefinition, context: Any
        ) -> tuple[Result, ExecutionReport]:
            _ = await self._compiled.ainvoke({"input": "x"})
            report = make_execution_report(context=context)
            report.workflow_id = workflow.workflow_id
            report.finalize(ExecutionStatus.COMPLETED, StopReason.END_TURN)
            return Result(content="ok", status=ExecutionStatus.COMPLETED), report

    adapter = LangGraphAdapter()
    assert isinstance(adapter, WorkflowEngine)


def test_canonical_package_import_path_is_available() -> None:
    import protocore as canonical

    assert canonical.AgentOrchestrator is not None
    assert canonical.AgentConfig is not None
    assert canonical.ExecutionReport is not None
    assert canonical.PlanningPolicy is not None


def test_core_exports_validate_path_arguments() -> None:
    from protocore import validate_path_arguments

    assert callable(validate_path_arguments)


def test_py_typed_marker_exists() -> None:
    marker = Path(__file__).resolve().parents[1] / "protocore" / "py.typed"
    assert marker.exists()
