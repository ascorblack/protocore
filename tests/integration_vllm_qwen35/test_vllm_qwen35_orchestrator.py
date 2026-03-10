from __future__ import annotations

import pytest

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    ApiMode,
    ExecutionMode,
    ExecutionStatus,
    Message,
    OpenAILLMClient,
    StopReason,
    make_agent_context,
)

def _orchestrator_client(cfg: object) -> OpenAILLMClient:
    return OpenAILLMClient(
        api_key=getattr(cfg, "api_key"),
        base_url=getattr(cfg, "base_url"),
        api_mode=ApiMode.CHAT_COMPLETIONS,
        default_model=getattr(cfg, "model"),
    )


@pytest.mark.asyncio
async def test_orchestrator_single_turn_vllm_qwen35(live_vllm_config: object) -> None:
    config = AgentConfig(
        agent_id="integration-leader-vllm-qwen35",
        model=getattr(live_vllm_config, "model"),
        api_mode=ApiMode.CHAT_COMPLETIONS,
        execution_mode=ExecutionMode.BYPASS,
        enable_thinking=False,
        max_iterations=2,
        max_tokens=96,
        system_prompt="Ты краткий технический ассистент.",
    )
    context = make_agent_context(config=config)
    context.messages.append(
        Message(
            role="user",
            content="В одном предложении объясни, зачем нужны интеграционные тесты.",
        )
    )

    orchestrator = AgentOrchestrator(llm_client=_orchestrator_client(live_vllm_config))
    result, report = await orchestrator.run(context)

    assert result.content.strip()
    assert result.status == ExecutionStatus.COMPLETED
    assert report.status == ExecutionStatus.COMPLETED
    assert report.stop_reason == StopReason.END_TURN
    assert report.model == getattr(live_vllm_config, "model")
    assert report.api_mode == ApiMode.CHAT_COMPLETIONS


@pytest.mark.asyncio
async def test_orchestrator_leader_without_planning_strategy_fails_contract(
    live_vllm_config: object,
) -> None:
    config = AgentConfig(
        agent_id="integration-leader-contract-vllm-qwen35",
        model=getattr(live_vllm_config, "model"),
        api_mode=ApiMode.CHAT_COMPLETIONS,
        execution_mode=ExecutionMode.LEADER,
        enable_thinking=False,
        max_iterations=2,
        max_tokens=96,
    )
    context = make_agent_context(config=config)
    context.messages.append(
        Message(
            role="user",
            content="Коротко объясни, как устроены smoke-тесты.",
        )
    )

    orchestrator = AgentOrchestrator(llm_client=_orchestrator_client(live_vllm_config))
    result, report = await orchestrator.run(context)

    assert result.status == ExecutionStatus.FAILED
    assert report.status == ExecutionStatus.FAILED
    assert report.stop_reason == StopReason.ERROR
    assert report.error_code == "PLANNING_REQUIRED"
