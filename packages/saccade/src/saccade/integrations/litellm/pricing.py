"""Pricing models and provider base class for custom LiteLLM providers.

Usage:
    from saccade.integrations.litellm import ModelPricing, RegisterableProvider
    from decimal import Decimal

    pricing = ModelPricing(
        input_cost_per_token=Decimal("0.0000005"),
        output_cost_per_token=Decimal("0.0000015"),
    )
    cost = pricing.calculate(input_tokens=1000, output_tokens=500)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    """Pricing configuration for a model.

    Costs are in USD per token (e.g., $0.50/1M tokens = Decimal("0.0000005")).
    """

    input_cost_per_token: Decimal = Decimal(0)
    output_cost_per_token: Decimal = Decimal(0)
    reasoning_cost_per_token: Decimal | None = None

    def calculate(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> Decimal:
        cost = self.input_cost_per_token * input_tokens + self.output_cost_per_token * output_tokens
        if self.reasoning_cost_per_token is not None and reasoning_tokens > 0:
            cost += self.reasoning_cost_per_token * reasoning_tokens
        return cost


class RegisterableProvider(ABC):
    """Base class for custom LiteLLM providers with optional pricing.

    Inherit from this class alongside CustomLLM to create a provider
    that can be registered with LiteLLM and calculate costs.

    Example:
        class MyProvider(CustomLLM, RegisterableProvider):
            def register(self) -> None:
                import litellm
                litellm.custom_provider_map = [
                    {"provider": "my-provider", "custom_handler": self}
                ]

            def calculate_cost(self, model, input_tokens, output_tokens, reasoning=0):
                return self.pricing[model].calculate(input_tokens, output_tokens, reasoning)
    """

    @abstractmethod
    def register(self) -> None: ...

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> Decimal | None:
        return None


__all__ = ["ModelPricing", "RegisterableProvider"]
