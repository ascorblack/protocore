"""Internal error types used by orchestration helpers."""

from __future__ import annotations

from .types import ShellCommandPlan


class ContractViolationError(RuntimeError):
    """Raised when a mandatory core contract cannot be satisfied."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class PendingShellApprovalError(RuntimeError):
    """Raised to pause execution until external shell approval is provided."""

    def __init__(self, plan: ShellCommandPlan) -> None:
        super().__init__(f"Shell approval required for plan_id={plan.plan_id}")
        self.plan = plan


__all__ = ["ContractViolationError", "PendingShellApprovalError"]
