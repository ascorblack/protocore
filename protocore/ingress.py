"""Unified ingress API for envelope parsing.

This module provides the single canonical entry point for parsing AgentEnvelope
inside the core. All envelope ingress must go through parse_envelope() to ensure
minor-version compatibility warnings are recorded in the execution report.

Anti-pattern: parsing envelope directly via AgentEnvelope.model_validate() or
AgentEnvelope.model_validate_json() — this bypasses version warning recording.
"""
from __future__ import annotations

from typing import Any

from .types import AgentEnvelope, ExecutionReport


def parse_envelope(
    data: str | bytes | dict[str, Any] | AgentEnvelope,
    report: ExecutionReport | None = None,
) -> AgentEnvelope:
    """Parse envelope and record minor-version compatibility warnings.

    This is the single ingress point for envelope parsing inside the core.
    Always use this function instead of direct Pydantic validation to ensure
    version warnings are captured in the execution report.

    Args:
        data: Raw envelope data (JSON string, bytes, dict, or already parsed).
        report: Execution report to record warnings into. If None, warnings are
            still checked but not persisted anywhere.

    Returns:
        Parsed and validated AgentEnvelope.

    Raises:
        ValueError: If major version mismatch or payload validation fails.
    """
    return AgentEnvelope.parse_with_report(data, report)
