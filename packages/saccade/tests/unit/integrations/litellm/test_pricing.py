"""Unit tests for ModelPricing and pricing functionality."""

from decimal import Decimal

import pytest

from saccade.integrations.litellm import ModelPricing, OpenAICompatibleProvider


class TestModelPricing:
    def test_calculate_with_zero_costs(self) -> None:
        pricing = ModelPricing()
        cost = pricing.calculate(input_tokens=1000, output_tokens=500)
        assert cost == Decimal("0")

    def test_calculate_input_only(self) -> None:
        pricing = ModelPricing(
            input_cost_per_token=Decimal("0.000001"),
        )
        cost = pricing.calculate(input_tokens=1000, output_tokens=500)
        assert cost == Decimal("0.001")

    def test_calculate_output_only(self) -> None:
        pricing = ModelPricing(
            output_cost_per_token=Decimal("0.000002"),
        )
        cost = pricing.calculate(input_tokens=1000, output_tokens=500)
        assert cost == Decimal("0.001")

    def test_calculate_input_and_output(self) -> None:
        pricing = ModelPricing(
            input_cost_per_token=Decimal("0.000001"),
            output_cost_per_token=Decimal("0.000002"),
        )
        cost = pricing.calculate(input_tokens=1000, output_tokens=500)
        assert cost == Decimal("0.002")

    def test_calculate_with_reasoning_tokens(self) -> None:
        pricing = ModelPricing(
            input_cost_per_token=Decimal("0.000001"),
            output_cost_per_token=Decimal("0.000002"),
            reasoning_cost_per_token=Decimal("0.000003"),
        )
        cost = pricing.calculate(input_tokens=1000, output_tokens=500, reasoning_tokens=200)
        assert cost == Decimal("0.0026")

    def test_calculate_reasoning_tokens_ignored_when_not_configured(self) -> None:
        pricing = ModelPricing(
            input_cost_per_token=Decimal("0.000001"),
            output_cost_per_token=Decimal("0.000002"),
        )
        cost = pricing.calculate(input_tokens=1000, output_tokens=500, reasoning_tokens=200)
        assert cost == Decimal("0.002")

    def test_calculate_with_real_world_pricing(self) -> None:
        pricing = ModelPricing(
            input_cost_per_token=Decimal("0.0000005"),
            output_cost_per_token=Decimal("0.0000015"),
        )
        cost = pricing.calculate(input_tokens=1_000_000, output_tokens=500_000)
        assert cost == Decimal("1.25")

    def test_frozen_dataclass(self) -> None:
        pricing = ModelPricing(input_cost_per_token=Decimal("0.000001"))
        with pytest.raises(AttributeError):
            pricing.input_cost_per_token = Decimal("0.000002")


class TestOpenAICompatibleProviderPricing:
    def test_calculate_cost_with_default_pricing(self) -> None:
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            default_pricing=ModelPricing(
                input_cost_per_token=Decimal("0.000001"),
                output_cost_per_token=Decimal("0.000002"),
            ),
        )
        cost = provider.calculate_cost("test/model", 1000, 500)
        assert cost == Decimal("0.002")

    def test_calculate_cost_with_model_specific_pricing(self) -> None:
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            pricing={
                "premium-model": ModelPricing(
                    input_cost_per_token=Decimal("0.00001"),
                    output_cost_per_token=Decimal("0.00003"),
                ),
            },
            default_pricing=ModelPricing(
                input_cost_per_token=Decimal("0.000001"),
                output_cost_per_token=Decimal("0.000002"),
            ),
        )

        cost_premium = provider.calculate_cost("test/premium-model", 1000, 500)
        assert cost_premium == Decimal("0.025")

        cost_default = provider.calculate_cost("test/standard-model", 1000, 500)
        assert cost_default == Decimal("0.002")

    def test_calculate_cost_no_pricing_returns_none(self) -> None:
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.test.com/v1",
            api_key="test-key",
        )
        cost = provider.calculate_cost("test/model", 1000, 500)
        assert cost is None

    def test_calculate_cost_strips_provider_prefix(self) -> None:
        provider = OpenAICompatibleProvider(
            provider_name="my-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            pricing={
                "specific-model": ModelPricing(
                    input_cost_per_token=Decimal("0.000001"),
                ),
            },
        )
        cost = provider.calculate_cost("my-provider/specific-model", 1000, 0)
        assert cost == Decimal("0.001")

    def test_calculate_cost_with_reasoning_tokens(self) -> None:
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            default_pricing=ModelPricing(
                input_cost_per_token=Decimal("0.000001"),
                output_cost_per_token=Decimal("0.000002"),
                reasoning_cost_per_token=Decimal("0.000003"),
            ),
        )
        cost = provider.calculate_cost("test/model", 1000, 500, reasoning_tokens=200)
        assert cost == Decimal("0.0026")
