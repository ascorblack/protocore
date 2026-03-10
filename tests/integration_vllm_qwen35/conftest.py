from __future__ import annotations

import os
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class LiveVllmConfig:
    run_live_tests: bool
    base_url: str
    model: str
    api_key: str


def _read_live_vllm_config() -> LiveVllmConfig:
    return LiveVllmConfig(
        run_live_tests=os.getenv("PROTOCORE_RUN_VLLM_INTEGRATION_TESTS") == "1",
        base_url=os.getenv("PROTOCORE_VLLM_BASE_URL", ""),
        model=os.getenv("PROTOCORE_VLLM_MODEL", "Qwen3.5"),
        api_key=os.getenv("PROTOCORE_VLLM_API_KEY", "EMPTY"),
    )


LIVE_VLLM_CONFIG = _read_live_vllm_config()


@pytest.fixture(scope="session")
def live_vllm_config() -> LiveVllmConfig:
    return LIVE_VLLM_CONFIG


@pytest.fixture(autouse=True)
def _skip_unless_live_flag_enabled(live_vllm_config: LiveVllmConfig) -> None:
    if not live_vllm_config.run_live_tests:
        pytest.skip(
            "set PROTOCORE_RUN_VLLM_INTEGRATION_TESTS=1 "
            "to run vLLM integration tests"
        )
