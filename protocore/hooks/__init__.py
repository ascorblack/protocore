"""Hook package public API."""

from .manager import HookManager, create_plugin_manager
from .specs import AgentHookSpecs, hookimpl, hookspec

__all__ = [
    "AgentHookSpecs",
    "HookManager",
    "create_plugin_manager",
    "hookimpl",
    "hookspec",
]
