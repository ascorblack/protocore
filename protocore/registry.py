"""Tool registry and agent registry for Protocore.

Provides centralised lookup for tool handlers and registered agents.
No business logic — purely mechanical registration / lookup.
"""
from __future__ import annotations

import copy
import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol

from .context import contains_path_argument, validate_path_arguments
from .orchestrator_errors import ContractViolationError
from .types import AgentConfig, ToolDefinition, ToolResult, ToolContext

if TYPE_CHECKING:
    from .hooks.manager import HookManager

logger = logging.getLogger(__name__)


def _clone_registered_object(
    obj: Any,
    *,
    strict: bool,
    error_code: str,
    object_label: str,
) -> Any:
    clone_method = getattr(obj, "clone", None)
    if callable(clone_method):
        return clone_method()
    try:
        cloned = copy.deepcopy(obj)
    except Exception as exc:
        if strict:
            raise ContractViolationError(
                error_code,
                f"{object_label} must implement clone() or support deepcopy()",
            ) from exc
        logger.warning(
            "%s shared by reference; clone()/deepcopy() unavailable (%s)",
            object_label,
            type(exc).__name__,
        )
        return obj
    if cloned is obj and not inspect.isfunction(obj) and not inspect.ismethod(obj):
        if strict:
            raise ContractViolationError(
                error_code,
                f"{object_label} must not be shared by reference across runs",
            )
        logger.warning("%s deepcopy returned original object; sharing by reference", object_label)
    return cloned

class ToolHandler(Protocol):
    async def __call__(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of available tools and their handlers.

    Tools registered here can be passed to LLM and dispatched by orchestrator.
    """

    def __init__(self, hook_manager: "HookManager | None" = None) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._definitions: dict[str, ToolDefinition] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._hook_manager = hook_manager

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
        *,
        tags: list[str] | None = None,
    ) -> None:
        """Register a tool definition and its async handler."""
        name = definition.name
        if definition.filesystem_access and not definition.path_fields:
            raise ValueError(
                f"filesystem tool '{name}' must declare path_fields or canonical path parameters"
            )
        if name in self._handlers:
            logger.warning("ToolRegistry: overwriting existing tool=%s", name)
        new_definitions = dict(self._definitions)
        new_handlers = dict(self._handlers)
        new_meta = copy.deepcopy(self._meta)
        new_definitions[name] = definition
        new_handlers[name] = handler
        new_meta[name] = {"tags": list(tags or [])}
        try:
            if self._hook_manager is not None:
                self._hook_manager.call_tool_registered(tool=definition)
        except Exception as exc:
            logger.warning("Hook call failed during tool registration: %s (tool=%s)", exc, name)
            raise
        self._definitions = new_definitions
        self._handlers = new_handlers
        self._meta = new_meta
        if not self.registry_is_consistent():
            raise RuntimeError("ToolRegistry invariant violated after register()")
        logger.debug("ToolRegistry: registered tool=%s", name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._definitions.pop(name, None)
        self._handlers.pop(name, None)
        self._meta.pop(name, None)

    def get_handler(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    def get_definition(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def get_tags(self, name: str) -> list[str]:
        return list(self._meta.get(name, {}).get("tags", []))

    def list_definitions(self, tags: list[str] | None = None) -> list[ToolDefinition]:
        """Return all tool definitions, optionally filtered by tags."""
        if not tags:
            return list(self._definitions.values())
        return [
            defn
            for name, defn in self._definitions.items()
            if any(t in self._meta[name].get("tags", []) for t in tags)
        ]

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        validate_paths: bool = True,
        tool_call_id: str = "",
    ) -> ToolResult | None:
        """Dispatch a tool call. Returns None if tool not found."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            logger.error("ToolRegistry: unknown tool=%s", tool_name)
            return None
        # Expose dispatcher tool_call_id inside handler context.
        if tool_call_id:
            context = context.model_copy(update={"tool_call_id": tool_call_id}, deep=True)
        if validate_paths:
            tags = self.get_tags(tool_name)
            definition = self.get_definition(tool_name)
            requires_path_validation = (
                "fs" in tags
                or (definition is not None and definition.filesystem_access)
                or contains_path_argument(arguments)
            )
            if requires_path_validation:
                path_keys = set(definition.path_fields) if definition is not None else None
                validate_path_arguments(arguments, context, path_keys=path_keys)
        return await handler(arguments=arguments, context=context)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers

    def __len__(self) -> int:
        return len(self._handlers)

    def registry_is_consistent(self) -> bool:
        """Return True when handlers/definitions/meta contain the same tool names."""
        keys = set(self._handlers)
        return keys == set(self._definitions) == set(self._meta)

    def clone(self, *, strict: bool = False) -> "ToolRegistry":
        """Return a best-effort isolated registry clone.

        Handlers that are callable objects with a ``clone()`` method or that
        support ``copy.deepcopy`` are copied; ``_meta`` is deep-copied. Plain
        function handlers (e.g. ``async def my_tool(...)``) have no clone
        support in Python and are left shared by reference.

        **Known limitation (contract):** If a handler is a plain function that
        closes over mutable state (e.g. a list or dict), that state is shared
        between the original registry and the clone. So parent and child
        orchestrators can observe each other's mutations via such handlers.
        This core does not fix that for the following reasons:

        1. Python cannot truly clone arbitrary functions or their closures
           without serialization or other heavy mechanisms.
        2. The only use of ``clone()`` here is when spawning a child
           orchestrator; the risk is limited to that parent/child pair.
        3. Handlers with mutable closure state are a design smell; the
           recommended contract is that tool handlers be stateless or get
           state from ``context`` only.

        If you need full isolation for a stateful handler, implement a
        callable class with a ``clone()`` method (or ensure it is
        deepcopy-able) and register that instead of a plain function.
        """
        cloned = ToolRegistry(hook_manager=self._hook_manager)
        cloned._handlers = {}
        for name, handler in self._handlers.items():
            handler_copy = _clone_registered_object(
                handler,
                strict=strict,
                error_code="TOOL_HANDLER_CLONE_REQUIRED",
                object_label=f"tool handler '{name}'",
            )
            cloned._handlers[name] = handler_copy
        cloned._definitions = {
            name: definition.model_copy(deep=True)
            for name, definition in self._definitions.items()
        }
        cloned._meta = copy.deepcopy(self._meta)
        return cloned


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Registry of available agent configurations (subagent profiles).

    Service registers its subagent configs here; orchestrator resolves them
    by agent_id to know which config to use when dispatching.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentConfig] = {}

    def register(self, config: AgentConfig) -> None:
        """Register an agent configuration."""
        self._agents[config.agent_id] = config
        logger.debug("AgentRegistry: registered agent_id=%s name=%s", config.agent_id, config.name)

    def get(self, agent_id: str) -> AgentConfig | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[AgentConfig]:
        return list(self._agents.values())

    def list_subagents(self) -> list[AgentConfig]:
        return [
            config for config in self._agents.values() if config.role.value == "subagent"
        ]

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def clone(self) -> "AgentRegistry":
        """Return an isolated clone of the agent registry."""
        cloned = AgentRegistry()
        cloned._agents = {
            agent_id: config.model_copy(deep=True)
            for agent_id, config in self._agents.items()
        }
        return cloned


# ---------------------------------------------------------------------------
# Strategy Registry (OrchestrationStrategy, etc.)
# ---------------------------------------------------------------------------


class StrategyRegistry:
    """Registry of named strategies (OrchestrationStrategy, PlanningStrategy, etc.).

    Keys are arbitrary names used by the service to select strategies at
    runtime (e.g. "fast", "expert", "parallel").
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Any] = {}

    def register(self, name: str, strategy: Any) -> None:
        """Register a strategy instance under a name."""
        self._strategies[name] = strategy
        logger.debug("StrategyRegistry: registered strategy=%s", name)

    def get(self, name: str) -> Any | None:
        return self._strategies.get(name)

    def list_names(self) -> list[str]:
        return list(self._strategies.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._strategies
