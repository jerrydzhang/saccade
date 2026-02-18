"""Tests for TraceEvent - the immutable event record."""

import time
from decimal import Decimal

import pytest

pytestmark = pytest.mark.unit


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """All event types should be defined."""
        from saccade.primitives.events import EventType

        assert EventType.START.value == "START"
        assert EventType.CHUNK.value == "CHUNK"
        assert EventType.OUTPUT.value == "OUTPUT"
        assert EventType.SUCCESS.value == "SUCCESS"
        assert EventType.ERROR.value == "ERROR"
        assert EventType.CANCEL.value == "CANCEL"


class TestSpanKind:
    """Tests for SpanKind constants."""

    def test_builtin_kinds_exist(self):
        """Built-in span kinds should be defined."""
        from saccade.primitives.events import SpanKind

        assert SpanKind.AGENT == "agent"
        assert SpanKind.TOOL == "tool"
        assert SpanKind.LLM == "llm"
        assert SpanKind.RETRIEVAL == "retrieval"
        assert SpanKind.EMBEDDING == "embedding"

    def test_custom_kind_allowed(self):
        """Users should be able to use custom kind strings."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.START,
            span_id="span-123",
            kind="custom_workflow",
        )

        assert event.kind == "custom_workflow"


class TestTraceEvent:
    """Tests for TraceEvent - immutable record of state change."""

    def test_auto_generated_id(self):
        """TraceEvent should auto-generate a ULID id."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(type=EventType.START, span_id="span-123")
        assert event.id is not None
        assert len(event.id) == 26

    def test_auto_generated_timestamp(self):
        """TraceEvent should auto-generate a timestamp."""
        from saccade.primitives.events import EventType, TraceEvent

        before = time.time()
        event = TraceEvent(type=EventType.START, span_id="span-123")
        after = time.time()

        assert before <= event.timestamp <= after

    def test_trace_id_required(self):
        """TraceEvent should require a trace_id."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.START,
            trace_id="trace-abc123",
            span_id="span-123",
        )

        assert event.trace_id == "trace-abc123"

    def test_trace_id_groups_events(self):
        """Events with same trace_id belong to same trace."""
        from saccade.primitives.events import EventType, TraceEvent

        trace_id = "trace-xyz"
        events = [
            TraceEvent(type=EventType.START, trace_id=trace_id, span_id="span-1"),
            TraceEvent(type=EventType.CHUNK, trace_id=trace_id, span_id="span-1", chunk="hi"),
            TraceEvent(type=EventType.SUCCESS, trace_id=trace_id, span_id="span-1"),
        ]

        assert all(e.trace_id == trace_id for e in events)

    def test_attributes_default_empty_dict(self):
        """TraceEvent should default attributes to empty dict."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123")

        assert event.attributes == {}

    def test_attributes_stores_arbitrary_metadata(self):
        """TraceEvent should store arbitrary attributes."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.START,
            trace_id="t1",
            span_id="span-123",
            attributes={
                "user_id": "user-456",
                "session_id": "session-789",
                "environment": "production",
                "custom_flag": True,
            },
        )

        assert event.attributes["user_id"] == "user-456"
        assert event.attributes["session_id"] == "session-789"
        assert event.attributes["environment"] == "production"
        assert event.attributes["custom_flag"] is True

    def test_attributes_immutable(self):
        """TraceEvent attributes should be immutable (frozen copy)."""
        from saccade.primitives.events import EventType, TraceEvent

        original_attrs = {"key": "value"}
        event = TraceEvent(
            type=EventType.START,
            trace_id="t1",
            span_id="span-123",
            attributes=original_attrs,
        )

        original_attrs["key"] = "modified"

        assert event.attributes["key"] == "value"

    def test_start_event_with_context(self):
        """START event should capture name, inputs, kind, relations."""
        from saccade.primitives.events import EventType, Relation, TraceEvent

        event = TraceEvent(
            type=EventType.START,
            trace_id="trace-123",
            span_id="span-123",
            name="agent",
            inputs={"query": "hello"},
            kind="llm",
            relations={Relation.CONTEXT: ["parent-456"]},
        )

        assert event.name == "agent"
        assert event.inputs == {"query": "hello"}
        assert event.kind == "llm"
        assert event.relations == {Relation.CONTEXT: ["parent-456"]}

    def test_start_event_with_dataflow(self):
        """START event should capture dataflow relations."""
        from saccade.primitives.events import EventType, Relation, TraceEvent

        event = TraceEvent(
            type=EventType.START,
            trace_id="trace-123",
            span_id="analyst",
            name="analyst",
            relations={
                Relation.CONTEXT: ["orchestrator"],
                Relation.DATAFLOW: ["researcher"],
            },
        )

        assert Relation.CONTEXT in event.relations
        assert Relation.DATAFLOW in event.relations
        assert "researcher" in event.relations[Relation.DATAFLOW]

    def test_chunk_event(self):
        """CHUNK event should capture streaming data."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.CHUNK,
            trace_id="trace-123",
            span_id="span-123",
            chunk="Hello",
        )

        assert event.chunk == "Hello"

    def test_chunk_event_with_dict(self):
        """CHUNK event should support dict chunks."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.CHUNK,
            trace_id="trace-123",
            span_id="span-123",
            chunk={"delta": {"content": "hello"}},
        )

        assert event.chunk == {"delta": {"content": "hello"}}

    def test_output_event(self):
        """OUTPUT event should capture the output value."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.OUTPUT,
            trace_id="trace-123",
            span_id="span-123",
            output="final result",
        )

        assert event.output == "final result"

    def test_success_event_with_metrics(self):
        """SUCCESS event should capture final metrics."""
        from saccade.primitives.events import (
            CostMetrics,
            EventType,
            LatencyMetrics,
            OperationMeta,
            TokenMetrics,
            TraceEvent,
        )

        event = TraceEvent(
            type=EventType.SUCCESS,
            trace_id="trace-123",
            span_id="span-123",
            tokens=TokenMetrics(input=100, output=50),
            cost=CostMetrics(usd=Decimal("0.01")),
            latency=LatencyMetrics(total_ms=150.0, time_to_first_token_ms=50.0),
            operation=OperationMeta(model="gpt-4o"),
            response_id="chatcmpl-abc",
        )

        assert event.tokens.input == 100
        assert event.tokens.output == 50
        assert event.cost.usd == Decimal("0.01")
        assert event.latency.total_ms == 150.0
        assert event.operation.model == "gpt-4o"
        assert event.response_id == "chatcmpl-abc"

    def test_error_event_with_metrics(self):
        """ERROR event should capture error message and metrics."""
        from saccade.primitives.events import EventType, LatencyMetrics, TraceEvent

        event = TraceEvent(
            type=EventType.ERROR,
            trace_id="trace-123",
            span_id="span-123",
            error="Connection failed",
            latency=LatencyMetrics(total_ms=500.0),
        )

        assert event.error == "Connection failed"
        assert event.latency.total_ms == 500.0

    def test_cancel_event_with_metrics(self):
        """CANCEL event should capture metrics."""
        from saccade.primitives.events import EventType, LatencyMetrics, TraceEvent

        event = TraceEvent(
            type=EventType.CANCEL,
            trace_id="trace-123",
            span_id="span-123",
            latency=LatencyMetrics(total_ms=100.0),
        )

        assert event.type == EventType.CANCEL
        assert event.latency.total_ms == 100.0

    def test_frozen(self):
        """TraceEvent should be immutable (frozen)."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123")
        with pytest.raises(Exception):
            event.span_id = "different-span"

    def test_unique_ids(self):
        """Each TraceEvent should have a unique ID."""
        from saccade.primitives.events import EventType, TraceEvent

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123") for _ in range(100)
        ]
        ids = [e.id for e in events]
        assert len(set(ids)) == 100

    def test_chronological_ids(self):
        """ULID IDs should have increasing timestamp component."""
        from saccade.primitives.events import EventType, TraceEvent

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id=f"span-{i}") for i in range(10)
        ]
        ids = [e.id for e in events]

        timestamps = [eid[:10] for eid in ids]
        assert timestamps == sorted(timestamps)
