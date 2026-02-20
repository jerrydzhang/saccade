"""Pytest configuration for LiteLLM integration tests.

Configuration is loaded from .env file at workspace root.

Required variables for cassette recording:
    PROVIDER_MODEL - LiteLLM model string (e.g., "my-provider/glm-4.7")
    PROVIDER_API_KEY - API key for your provider

Optional variables:
    PROVIDER_API_BASE - Custom API endpoint (for OpenAI-compatible providers)
"""

import os
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import load_dotenv

if TYPE_CHECKING:
    from saccade.integrations.litellm import TracedLiteLLM

load_dotenv(Path(__file__).parent.parent.parent.parent.parent.parent / ".env")


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["Authorization"],
        "decode_compressed_response": True,
    }


@pytest.fixture
def llm() -> "TracedLiteLLM":
    from saccade.integrations.litellm import OpenAICompatibleProvider, TracedLiteLLM

    model = os.getenv("PROVIDER_MODEL", "openai/gpt-4o-mini")
    api_base = os.getenv("PROVIDER_API_BASE")
    api_key = os.getenv("PROVIDER_API_KEY")

    if api_base and api_key:
        provider_name = model.split("/")[0] if "/" in model else "custom"
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_base=api_base,
            api_key=api_key,
        )
        return TracedLiteLLM(model=model, provider=provider)

    return TracedLiteLLM(model=model)


@pytest.fixture
def llm_with_pricing() -> "TracedLiteLLM":
    from saccade.integrations.litellm import (
        ModelPricing,
        OpenAICompatibleProvider,
        TracedLiteLLM,
    )

    model = os.getenv("PROVIDER_MODEL", "openai/gpt-4o-mini")
    api_base = os.getenv("PROVIDER_API_BASE")
    api_key = os.getenv("PROVIDER_API_KEY")

    if api_base and api_key:
        provider_name = model.split("/")[0] if "/" in model else "custom"
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_base=api_base,
            api_key=api_key,
            default_pricing=ModelPricing(
                input_cost_per_token=Decimal("0.000001"),
                output_cost_per_token=Decimal("0.000002"),
            ),
        )
        return TracedLiteLLM(model=model, provider=provider)

    return TracedLiteLLM(model=model)
