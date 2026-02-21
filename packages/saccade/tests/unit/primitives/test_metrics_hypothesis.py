from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

import pytest

from saccade.primitives.events import CostMetrics, LatencyMetrics, TokenMetrics

pytestmark = pytest.mark.unit


class TestTokenMetricsProperties:
    @given(
        input1=st.integers(min_value=0, max_value=10**9),
        output1=st.integers(min_value=0, max_value=10**9),
        reasoning1=st.integers(min_value=0, max_value=10**9),
        input2=st.integers(min_value=0, max_value=10**9),
        output2=st.integers(min_value=0, max_value=10**9),
        reasoning2=st.integers(min_value=0, max_value=10**9),
    )
    def test_addition_is_commutative(
        self, input1, output1, reasoning1, input2, output2, reasoning2
    ):
        m1 = TokenMetrics(input=input1, output=output1, reasoning=reasoning1)
        m2 = TokenMetrics(input=input2, output=output2, reasoning=reasoning2)

        assert m1 + m2 == m2 + m1

    @given(
        input1=st.integers(min_value=0, max_value=10**9),
        output1=st.integers(min_value=0, max_value=10**9),
        input2=st.integers(min_value=0, max_value=10**9),
        output2=st.integers(min_value=0, max_value=10**9),
        input3=st.integers(min_value=0, max_value=10**9),
        output3=st.integers(min_value=0, max_value=10**9),
    )
    def test_addition_is_associative(self, input1, output1, input2, output2, input3, output3):
        m1 = TokenMetrics(input=input1, output=output1)
        m2 = TokenMetrics(input=input2, output=output2)
        m3 = TokenMetrics(input=input3, output=output3)

        assert (m1 + m2) + m3 == m1 + (m2 + m3)

    @given(
        input_val=st.integers(min_value=0, max_value=10**9),
        output_val=st.integers(min_value=0, max_value=10**9),
        reasoning_val=st.integers(min_value=0, max_value=10**9),
        cached_val=st.integers(min_value=0, max_value=10**9),
        cache_write_val=st.integers(min_value=0, max_value=10**9),
    )
    def test_addition_with_zero_is_identity(
        self, input_val, output_val, reasoning_val, cached_val, cache_write_val
    ):
        m = TokenMetrics(
            input=input_val,
            output=output_val,
            reasoning=reasoning_val,
            cached=cached_val,
            cache_write=cache_write_val,
        )
        zero = TokenMetrics()

        assert m + zero == m
        assert zero + m == m

    @given(
        input1=st.integers(min_value=0, max_value=10**9),
        input2=st.integers(min_value=0, max_value=10**9),
    )
    def test_addition_sums_inputs(self, input1, input2):
        m1 = TokenMetrics(input=input1)
        m2 = TokenMetrics(input=input2)

        result = m1 + m2
        assert result.input == input1 + input2

    @given(
        input_val=st.integers(min_value=0, max_value=10**9),
        output_val=st.integers(min_value=0, max_value=10**9),
    )
    def test_total_property(self, input_val, output_val):
        m = TokenMetrics(input=input_val, output=output_val)
        assert m.total == input_val + output_val


class TestCostMetricsProperties:
    @given(
        usd1=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=6),
        usd2=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=6),
    )
    def test_addition_is_commutative(self, usd1, usd2):
        c1 = CostMetrics(usd=usd1)
        c2 = CostMetrics(usd=usd2)

        result1 = c1 + c2
        result2 = c2 + c1
        assert result1.usd == result2.usd

    @given(
        usd1=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=6),
        usd2=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=6),
        usd3=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=6),
    )
    def test_addition_is_associative(self, usd1, usd2, usd3):
        c1 = CostMetrics(usd=usd1)
        c2 = CostMetrics(usd=usd2)
        c3 = CostMetrics(usd=usd3)

        result1 = (c1 + c2) + c3
        result2 = c1 + (c2 + c3)
        assert result1.usd == result2.usd

    @given(usd=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=6))
    def test_addition_with_zero_is_identity(self, usd):
        c = CostMetrics(usd=usd)
        zero = CostMetrics()

        result = c + zero
        assert result.usd == usd

    @given(usd=st.decimals(min_value=Decimal("0.000001"), max_value=Decimal("100"), places=6))
    def test_addition_preserves_precision(self, usd):
        c1 = CostMetrics(usd=usd)
        c2 = CostMetrics(usd=usd)

        result = c1 + c2
        expected = usd + usd
        assert result.usd == expected


class TestLatencyMetricsProperties:
    @given(
        ms1=st.floats(min_value=0, max_value=10**6, allow_nan=False, allow_infinity=False),
        ms2=st.floats(min_value=0, max_value=10**6, allow_nan=False, allow_infinity=False),
    )
    def test_addition_sums_total_ms(self, ms1, ms2):
        l1 = LatencyMetrics(total_ms=ms1)
        l2 = LatencyMetrics(total_ms=ms2)

        result = l1 + l2
        assert result.total_ms == ms1 + ms2

    @given(
        skew1=st.booleans(),
        skew2=st.booleans(),
    )
    def test_addition_propagates_clock_skew(self, skew1, skew2):
        l1 = LatencyMetrics(has_clock_skew=skew1)
        l2 = LatencyMetrics(has_clock_skew=skew2)

        result = l1 + l2
        expected_skew = skew1 or skew2
        assert result.has_clock_skew == expected_skew

    @given(
        ms=st.floats(min_value=0, max_value=10**6, allow_nan=False, allow_infinity=False),
        skew=st.booleans(),
    )
    def test_addition_with_zero_is_identity(self, ms, skew):
        l = LatencyMetrics(total_ms=ms, has_clock_skew=skew)
        zero = LatencyMetrics()

        result = l + zero
        assert result.total_ms == ms
        assert result.has_clock_skew == skew


class TestAggregationList:
    @given(
        values=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=1000),
                st.integers(min_value=0, max_value=1000),
            ),
            min_size=0,
            max_size=20,
        )
    )
    def test_aggregating_list_of_token_metrics(self, values):
        metrics = [TokenMetrics(input=i, output=o) for i, o in values]

        result = TokenMetrics()
        for m in metrics:
            result = result + m

        expected_input = sum(i for i, _ in values)
        expected_output = sum(o for _, o in values)

        assert result.input == expected_input
        assert result.output == expected_output

    @given(
        values=st.lists(
            st.decimals(min_value=Decimal("0"), max_value=Decimal("100"), places=4),
            min_size=0,
            max_size=20,
        )
    )
    def test_aggregating_list_of_cost_metrics(self, values):
        metrics = [CostMetrics(usd=v) for v in values]

        result = CostMetrics()
        for c in metrics:
            result = result + c

        expected = sum(values, Decimal("0"))
        assert result.usd == expected
