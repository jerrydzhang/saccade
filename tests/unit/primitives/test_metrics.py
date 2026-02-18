"""Tests for Talos metric types: TokenMetrics, CostMetrics, LatencyMetrics, OperationMeta."""

import pytest
from decimal import Decimal

pytestmark = pytest.mark.unit


class TestTokenMetrics:
    """Tests for TokenMetrics - token usage counts with aggregation."""

    def test_default_values(self):
        """TokenMetrics should have all zero defaults."""
        from cadence.primitives.events import TokenMetrics

        m = TokenMetrics()
        assert m.input == 0
        assert m.output == 0
        assert m.reasoning == 0
        assert m.cached == 0
        assert m.cache_write == 0

    def test_addition(self):
        """TokenMetrics should support + operator for aggregation."""
        from cadence.primitives.events import TokenMetrics

        m1 = TokenMetrics(input=100, output=50)
        m2 = TokenMetrics(input=200, output=75, reasoning=100)
        result = m1 + m2

        assert result.input == 300
        assert result.output == 125
        assert result.reasoning == 100
        assert result.cached == 0
        assert result.cache_write == 0

    def test_addition_with_cached_tokens(self):
        """TokenMetrics should aggregate cached and cache_write fields."""
        from cadence.primitives.events import TokenMetrics

        m1 = TokenMetrics(cached=50, cache_write=100)
        m2 = TokenMetrics(cached=25, cache_write=0)
        result = m1 + m2

        assert result.cached == 75
        assert result.cache_write == 100

    def test_frozen(self):
        """TokenMetrics should be immutable (frozen)."""
        from cadence.primitives.events import TokenMetrics

        m = TokenMetrics(input=100)
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            m.input = 200


class TestCostMetrics:
    """Tests for CostMetrics - estimated cost in USD using Decimal."""

    def test_default_value(self):
        """CostMetrics should default to Decimal('0') USD."""
        from cadence.primitives.events import CostMetrics

        c = CostMetrics()
        assert c.usd == Decimal(0)

    def test_decimal_type(self):
        """CostMetrics.usd should be a Decimal for precision."""
        from cadence.primitives.events import CostMetrics

        c = CostMetrics(usd=Decimal("0.01"))
        assert isinstance(c.usd, Decimal)

    def test_addition(self):
        """CostMetrics should support + operator for aggregation."""
        from cadence.primitives.events import CostMetrics

        c1 = CostMetrics(usd=Decimal("0.01"))
        c2 = CostMetrics(usd=Decimal("0.02"))
        result = c1 + c2

        assert result.usd == Decimal("0.03")

    def test_precision_no_drift(self):
        """Decimal should avoid floating point drift."""
        from cadence.primitives.events import CostMetrics

        # This would fail with float: 0.1 + 0.2 != 0.3
        c1 = CostMetrics(usd=Decimal("0.1"))
        c2 = CostMetrics(usd=Decimal("0.2"))
        result = c1 + c2

        assert result.usd == Decimal("0.3")

    def test_fractional_pricing(self):
        """Decimal should handle fractional LLM pricing accurately."""
        from cadence.primitives.events import CostMetrics

        # Real example: 4 tokens when 1M cached tokens = $0.015
        # Cost per token = 0.000000015
        cost_per_token = Decimal("0.000000015")
        token_count = 4
        c = CostMetrics(usd=cost_per_token * token_count)

        assert c.usd == Decimal("0.000000060")

    def test_frozen(self):
        """CostMetrics should be immutable (frozen)."""
        from cadence.primitives.events import CostMetrics

        c = CostMetrics(usd=Decimal("0.01"))
        with pytest.raises(Exception):
            c.usd = Decimal("0.05")


class TestLatencyMetrics:
    """Tests for LatencyMetrics - timing data."""

    def test_default_values(self):
        """LatencyMetrics should have zero total_ms and None TTFT."""
        from cadence.primitives.events import LatencyMetrics

        latency = LatencyMetrics()
        assert latency.total_ms == 0.0
        assert latency.time_to_first_token_ms is None
        assert latency.has_clock_skew is False

    def test_clock_skew_flag(self):
        """LatencyMetrics should flag when end < start (clock skew)."""
        from cadence.primitives.events import LatencyMetrics

        latency = LatencyMetrics(total_ms=0.0, has_clock_skew=True)
        assert latency.has_clock_skew is True

    def test_addition_ignores_ttft(self):
        """LatencyMetrics + should sum total_ms but not TTFT (TTFT is per-span)."""
        from cadence.primitives.events import LatencyMetrics

        l1 = LatencyMetrics(total_ms=100.0, time_to_first_token_ms=50.0)
        l2 = LatencyMetrics(total_ms=200.0, time_to_first_token_ms=75.0)
        result = l1 + l2

        assert result.total_ms == 300.0

    def test_addition_propagates_clock_skew(self):
        """If any LatencyMetrics has has_clock_skew, result should too."""
        from cadence.primitives.events import LatencyMetrics

        l1 = LatencyMetrics(total_ms=100.0, has_clock_skew=True)
        l2 = LatencyMetrics(total_ms=200.0, has_clock_skew=False)
        result = l1 + l2

        assert result.has_clock_skew is True

    def test_frozen(self):
        """LatencyMetrics should be immutable (frozen)."""
        from cadence.primitives.events import LatencyMetrics

        latency = LatencyMetrics(total_ms=100.0)
        with pytest.raises(Exception):
            latency.total_ms = 200.0


class TestOperationMeta:
    """Tests for OperationMeta - start-time metadata."""

    def test_default_values(self):
        """OperationMeta should have sensible defaults."""
        from cadence.primitives.events import OperationMeta

        op = OperationMeta()
        assert op.model is None
        assert op.provider is None
        assert op.host is None
        assert op.kind == "generic"
        assert op.correlation_id is None

    def test_llm_metadata(self):
        """OperationMeta should capture LLM call details."""
        from cadence.primitives.events import OperationMeta

        op = OperationMeta(model="gpt-4o", provider="openai", host="api.openai.com", kind="llm")
        assert op.model == "gpt-4o"
        assert op.provider == "openai"
        assert op.kind == "llm"

    def test_external_correlation_id(self):
        """OperationMeta should store external system correlation IDs."""
        from cadence.primitives.events import OperationMeta

        op = OperationMeta(model="gpt-4o", provider="openai", correlation_id="chatcmpl-abc123")
        assert op.correlation_id == "chatcmpl-abc123"

    def test_frozen(self):
        """OperationMeta should be immutable (frozen)."""
        from cadence.primitives.events import OperationMeta

        op = OperationMeta(model="gpt-4o")
        with pytest.raises(Exception):
            op.model = "gpt-3.5-turbo"
