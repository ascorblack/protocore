from __future__ import annotations

import asyncio
from typing import cast

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    ExecutionMode,
    LLMClient,
    Message,
    PlanArtifact,
    make_agent_context,
)
from protocore.integrations.openai import OpenAILLMClient
from protocore.types import ThinkingProfilePreset


class DemoPlanningStrategy:
    async def build_plan(
        self,
        task: str,
        context: object,
        llm_client: object,
    ) -> PlanArtifact:
        _ = (context, llm_client)
        return PlanArtifact(
            trace_id="demo-trace",
            raw_plan=f"1) Analyze task: {task}\n2) Produce direct answer",
        )

    async def update_plan(
        self,
        plan: PlanArtifact,
        context: object,
        llm_client: object,
    ) -> PlanArtifact:
        _ = (context, llm_client)
        return plan


async def run_bypass_example(llm: OpenAILLMClient, registry: AgentRegistry) -> None:
    bypass_config = AgentConfig(
        name="qwen-bypass-agent",
        model="Qwen/Qwen2.5-32B-Instruct",
        execution_mode=ExecutionMode.BYPASS,
        thinking_profile=ThinkingProfilePreset.INSTRUCT_TOOL_WORKER,
        system_prompt="You are a helpful agent.",
    )
    context = make_agent_context(config=bypass_config)
    context.messages.append(Message(role="user", content="Hello from bypass mode"))
    orchestrator = AgentOrchestrator(
        llm_client=cast(LLMClient, llm),
        agent_registry=registry,
    )
    result, report = await orchestrator.run(context)
    print("[BYPASS]", result.content)
    print(f"[BYPASS] tokens: {report.input_tokens}+{report.output_tokens}")


async def run_leader_example(llm: OpenAILLMClient, registry: AgentRegistry) -> None:
    leader_config = AgentConfig(
        name="qwen-leader-agent",
        model="Qwen/Qwen2.5-32B-Instruct",
        execution_mode=ExecutionMode.LEADER,
        thinking_profile=ThinkingProfilePreset.THINKING_PLANNER,
        thinking_tokens_reserve=1024,
        system_prompt="You are a helpful leader agent.",
    )
    context = make_agent_context(config=leader_config)
    context.messages.append(Message(role="user", content="Build a short plan and answer"))
    orchestrator = AgentOrchestrator(
        llm_client=cast(LLMClient, llm),
        agent_registry=registry,
        planning_strategy=DemoPlanningStrategy(),
    )
    result, report = await orchestrator.run(context)
    print("[LEADER]", result.content)
    print(f"[LEADER] plan_created={report.plan_created} plan_id={report.plan_id}")


async def main() -> None:
    llm = OpenAILLMClient(
        api_key="local-token",
        base_url="http://127.0.0.1:8000/v1",
        default_model="Qwen/Qwen2.5-32B-Instruct",
    )

    registry = AgentRegistry()
    registry.register(
        AgentConfig(
            agent_id="qwen-subagent",
            name="qwen-subagent",
            model="Qwen/Qwen2.5-32B-Instruct",
            execution_mode=ExecutionMode.BYPASS,
            thinking_profile=ThinkingProfilePreset.INSTRUCT_TOOL_WORKER,
            system_prompt="You are a helpful subagent.",
        )
    )

    await run_bypass_example(llm, registry)
    await run_leader_example(llm, registry)


if __name__ == "__main__":
    asyncio.run(main())
