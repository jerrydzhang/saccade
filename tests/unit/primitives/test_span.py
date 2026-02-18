"""Tests for Span - the context manager for tracing a unit of work."""

import pytest
import asyncio
import warnings
from decimal import Decimal

pytestmark = pytest.mark.unit


class TestSpanInit:
    """Tests for Span initialization."""

    def test_auto_generated_id(self):
        """Span should auto-generate a ULID id."""
        from cadence.primitives.span import Span

        span = Span("test")
        assert span.id is not None
        assert len(span.id) == 26  # ULID string length

    def test_unique_ids(self):
        """Each Span should have a unique ID."""
        from cadence.primitives.span import Span

        spans = [Span("test") for _ in range(100)]
        ids = [s.id for s in spans]
        assert len(set(ids)) == 100

    def test_name_and_kind(self):
        """Span should store name and kind."""
        from cadence.primitives.span import Span

        span = Span("agent", kind="llm")
        assert span.name == "agent"
        assert span.kind == "llm"

    def test_default_kind(self):
        """Span should default to 'generic' kind."""
        from cadence.primitives.span import Span

        span = Span("test")
        assert span.kind == "generic"

    def test_inputs(self):
        """Span should store inputs."""
        from cadence.primitives.span import Span

        span = Span("test", inputs={"query": "hello", "count": 5})
        assert span.inputs == {"query": "hello", "count": 5}

    def test_initial_status(self):
        """Span should start in PENDING status."""
        from cadence.primitives.span import Span

        span = Span("test")
        assert span.status == "PENDING"


class TestSpanRelations:
    """Tests for Span relations handling."""

    def test_no_relations_by_default(self):
        """Span should have no relations by default."""
        from cadence.primitives.span import Span

        span = Span("test")
        assert span.relations == {}

    def test_user_provided_relations(self):
        """Span should accept user-provided relations."""
        from cadence.primitives.span import Span

        span = Span("test", relations={"depends_on": ["span-123"]})
        assert span.relations == {"depends_on": ["span-123"]}

    def test_relations_validation_list_required(self):
        """Relations values must be lists."""
        from cadence.primitives.span import Span

        with pytest.raises(TypeError, match="must be list"):
            Span("test", relations={"depends_on": "not-a-list"})

    def test_relations_validation_str_elements(self):
        """Relations list elements must be strings."""
        from cadence.primitives.span import Span

        with pytest.raises(TypeError, match="must contain str"):
            Span("test", relations={"depends_on": [123]})

    def test_relate_adds_relation(self):
        """relate() should add a new relation."""
        from cadence.primitives.span import Span

        span = Span("test")
        span.relate("depends_on", "span-123")

        assert span.relations == {"depends_on": ["span-123"]}

    def test_relate_idempotent(self):
        """relate() should be idempotent - duplicate calls should not duplicate entries."""
        from cadence.primitives.span import Span

        span = Span("test")
        span.relate("depends_on", "span-123")
        span.relate("depends_on", "span-123")
        span.relate("depends_on", "span-123")

        assert span.relations == {"depends_on": ["span-123"]}
        assert len(span.relations["depends_on"]) == 1

    def test_relate_multiple_targets(self):
        """relate() should support multiple targets for same relation type."""
        from cadence.primitives.span import Span

        span = Span("test")
        span.relate("depends_on", "span-1")
        span.relate("depends_on", "span-2")

        assert span.relations["depends_on"] == ["span-1", "span-2"]

    def test_relate_validation_relation_type_str(self):
        """relate() should validate relation_type is a string."""
        from cadence.primitives.span import Span

        span = Span("test")
        with pytest.raises(TypeError, match="relation_type must be str"):
            span.relate(123, "span-123")

    def test_relate_validation_span_id_str(self):
        """relate() should validate span_id is a string."""
        from cadence.primitives.span import Span

        span = Span("test")
        with pytest.raises(TypeError, match="span_id must be str"):
            span.relate("depends_on", 123)


class TestSpanContext:
    """Tests for Span as a context manager."""

    def test_enter_changes_status(self):
        """__enter__ should change status to RUNNING."""
        from cadence.primitives.span import Span

        with Span("test") as span:
            assert span.status == "RUNNING"

    def test_exit_changes_status_completed(self):
        """__exit__ without error should change status to COMPLETED."""
        from cadence.primitives.span import Span

        with Span("test") as span:
            pass

        assert span.status == "COMPLETED"

    def test_exit_on_error(self):
        """__exit__ with error should change status to FAILED."""
        from cadence.primitives.span import Span

        with pytest.raises(ValueError):
            with Span("test") as span:
                raise ValueError("test error")

        assert span.status == "FAILED"
        assert span.error == "test error"

    def test_exit_on_cancelled_error(self):
        """__exit__ with CancelledError should change status to CANCELLED."""
        from cadence.primitives.span import Span

        with pytest.raises(asyncio.CancelledError):
            with Span("test") as span:
                raise asyncio.CancelledError

        assert span.status == "CANCELLED"

    def test_exception_propagates(self):
        """Span should record error AND re-raise the exception."""
        from cadence.primitives.span import Span

        with pytest.raises(ValueError, match="boom"):
            with Span("test"):
                raise ValueError("boom")


class TestSpanConcurrency:
    """Tests for Span thread/task safety."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_single_span(self):
        """Span should handle concurrent stream calls safely."""
        from cadence.primitives.span import Span

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with Span("parallel-writer") as span:

                async def writer(val: str):
                    for _ in range(10):
                        span.stream(val)
                        await asyncio.sleep(0)

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(writer("A"))
                    tg.create_task(writer("B"))

            assert len(span._chunks) == 20


class TestSpanOutputs:
    """Tests for Span output handling."""

    def test_set_output(self):
        """set_output() should store the output value."""
        from cadence.primitives.span import Span

        with Span("test") as span:
            span.set_output("result")

        assert span.output == "result"

    def test_set_output_multiple_times_last_wins(self):
        """Multiple set_output() calls - last value wins."""
        from cadence.primitives.span import Span

        with Span("test") as span:
            span.set_output("v1")
            span.set_output("v2")
            span.set_output("final")

        assert span.output == "final"


class TestSpanStreaming:
    """Tests for Span streaming functionality."""

    def test_stream_tracks_chunks(self):
        """stream() should store chunks locally."""
        from cadence.primitives.span import Span

        with Span("test") as span:
            span.stream("Hello")
            span.stream(" ")
            span.stream("world")

        assert span._chunks == ["Hello", " ", "world"]

    def test_stream_tracks_ttft(self):
        """stream() should track time-to-first-token on first call."""
        import time
        from cadence.primitives.span import Span

        with Span("test") as span:
            time.sleep(0.01)
            span.stream("first")
            first_token_time = span._first_token_time
            assert first_token_time is not None
            time.sleep(0.01)
            span.stream("second")

        # First token time should not change
        assert span._first_token_time == first_token_time


class TestSpanMetrics:
    """Tests for Span metrics handling."""

    def test_set_metrics_tokens(self):
        """set_metrics() should store token metrics."""
        from cadence.primitives.span import Span
        from cadence.primitives.events import TokenMetrics

        with Span("test") as span:
            span.set_metrics(tokens=TokenMetrics(input=100, output=50))

        assert span._pending_tokens.input == 100
        assert span._pending_tokens.output == 50

    def test_set_metrics_cost(self):
        """set_metrics() should store cost metrics."""
        from cadence.primitives.span import Span
        from cadence.primitives.events import CostMetrics

        with Span("test") as span:
            span.set_metrics(cost=CostMetrics(usd=Decimal("0.01")))

        assert span._pending_cost.usd == Decimal("0.01")

    def test_set_metrics_accumulates(self):
        """set_metrics() should accumulate metrics across calls."""
        from cadence.primitives.span import Span
        from cadence.primitives.events import TokenMetrics

        with Span("test") as span:
            span.set_metrics(tokens=TokenMetrics(input=100, output=50))
            span.set_metrics(tokens=TokenMetrics(input=50, output=25))

        assert span._pending_tokens.input == 150
        assert span._pending_tokens.output == 75


class TestSpanMeta:
    """Tests for Span operation metadata."""

    def test_set_meta(self):
        """set_meta() should store operation metadata."""
        from cadence.primitives.span import Span
        from cadence.primitives.events import OperationMeta

        with Span("test") as span:
            span.set_meta(OperationMeta(model="gpt-4o", provider="openai"))

        assert span._pending_operation.model == "gpt-4o"
        assert span._pending_operation.provider == "openai"

    def test_set_meta_updates(self):
        """set_meta() can be called multiple times - last wins."""
        from cadence.primitives.span import Span
        from cadence.primitives.events import OperationMeta

        with Span("test") as span:
            span.set_meta(OperationMeta(model="gpt-4o"))
            span.set_meta(OperationMeta(model="gpt-4o", correlation_id="chatcmpl-abc"))

        assert span._pending_operation.model == "gpt-4o"
        assert span._pending_operation.correlation_id == "chatcmpl-abc"


class TestSpanWithoutBus:
    """Tests for Span behavior without a TraceBus (orphan mode)."""

    def test_warns_without_bus(self):
        """Span should warn when emitting events without a bus."""
        from cadence.primitives.span import Span

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with Span("test") as span:
                span.set_output("result")

            # Should have warning about no bus
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) > 0
            assert "no tracebus" in str(runtime_warnings[0].message).lower()

    def test_works_locally_without_bus(self):
        """Span should still work locally even without a bus."""
        from cadence.primitives.span import Span

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Span("test") as span:
                span.set_output("result")
                assert span.output == "result"

            assert span.status == "COMPLETED"


class TestSpanContextPropagation:
    """Tests for Span context variable propagation."""

    def test_auto_context_relation(self):
        """Nested spans should auto-capture context relation."""
        from cadence.primitives.span import Span

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with Span("parent") as parent:
                parent_id = parent.id

                with Span("child") as child:
                    assert "context" in child.relations
                    assert child.relations["context"] == [parent_id]

    def test_context_stack_multiple_levels(self):
        """Deep nesting should capture correct context chain."""
        from cadence.primitives.span import Span

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with Span("level1") as l1:
                l1_id = l1.id
                with Span("level2") as l2:
                    assert l2.relations["context"] == [l1_id]
                    l2_id = l2.id
                    with Span("level3") as l3:
                        assert l3.relations["context"] == [l2_id]

    def test_context_resets_on_exit(self):
        """Context variable should reset when span exits."""
        from cadence.primitives.span import Span, _current_span_id

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            assert _current_span_id.get() is None

            with Span("outer") as outer:
                assert _current_span_id.get() == outer.id

                with Span("inner") as inner:
                    assert _current_span_id.get() == inner.id

                # After inner exits, should be back to outer
                assert _current_span_id.get() == outer.id

            # After outer exits, should be None
            assert _current_span_id.get() is None
