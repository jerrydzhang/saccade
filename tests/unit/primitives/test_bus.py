"""Tests for TraceBus - the central event aggregator."""

import pytest

pytestmark = pytest.mark.unit


class TestTraceBus:
    """Tests for TraceBus - synchronous event collection and subscriber notification."""

    def test_emit_and_retrieve_events(self):
        """Events emitted to bus should be immediately available."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123", name="test")
        bus.emit(event)

        events = bus.events
        assert len(events) == 1
        assert events[0].span_id == "span-123"
        assert events[0].name == "test"

    def test_subscriber_notification(self):
        """Subscribers should receive events immediately on emit."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()
        received = []

        def callback(event: TraceEvent):
            received.append(event)

        bus.subscribe(callback)

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123")
        bus.emit(event)

        assert len(received) == 1
        assert received[0].span_id == "span-123"

    def test_subscriber_error_isolation(self):
        """Subscriber errors should not crash the bus or affect other subscribers."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()
        received = []

        def bad_callback(event: TraceEvent):
            raise RuntimeError("Subscriber error")

        def good_callback(event: TraceEvent):
            received.append(event)

        bus.subscribe(bad_callback)
        bus.subscribe(good_callback)

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123")
        bus.emit(event)

        assert len(received) == 1
        assert len(bus.events) == 1

    def test_multiple_subscribers(self):
        """Multiple subscribers should all receive events."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()
        received1 = []
        received2 = []
        received3 = []

        bus.subscribe(lambda e: received1.append(e))
        bus.subscribe(lambda e: received2.append(e))
        bus.subscribe(lambda e: received3.append(e))

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123")
        bus.emit(event)

        assert len(received1) == 1
        assert len(received2) == 1
        assert len(received3) == 1

    def test_events_returns_copy(self):
        """events property should return a copy, not the internal list."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()

        event = TraceEvent(type=EventType.START, trace_id="t1", span_id="span-123")
        bus.emit(event)

        events1 = bus.events
        events2 = bus.events

        assert events1 is not events2
        assert events1 == events2

    def test_order_preserved(self):
        """Events should be stored in emission order."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()

        for i in range(10):
            bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id=f"span-{i}"))

        events = bus.events
        span_ids = [e.span_id for e in events]
        expected = [f"span-{i}" for i in range(10)]
        assert span_ids == expected

    def test_trace_id_auto_generated(self):
        """TraceBus should auto-generate a trace_id if not provided."""
        from cadence.primitives.bus import TraceBus

        bus = TraceBus()
        assert bus.trace_id is not None
        assert len(bus.trace_id) == 26

    def test_trace_id_provided(self):
        """TraceBus should use provided trace_id."""
        from cadence.primitives.bus import TraceBus

        bus = TraceBus(trace_id="custom-trace-id")
        assert bus.trace_id == "custom-trace-id"

    def test_events_property_reflects_real_time(self):
        """Events should be available immediately after emit (no drain needed)."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()

        assert len(bus.events) == 0

        bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id="s1"))
        assert len(bus.events) == 1

        bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id="s2"))
        assert len(bus.events) == 2

    def test_subscriber_can_be_added_after_emit(self):
        """Subscribers added after emit should receive future events."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()
        received = []

        bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id="s1"))
        assert len(received) == 0

        bus.subscribe(lambda e: received.append(e))
        bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id="s2"))

        assert len(received) == 1
        assert received[0].span_id == "s2"


class TestTraceBusSubscribe:
    """Tests for TraceBus.subscribe() method."""

    def test_subscribe_accepts_callable(self):
        """subscribe should accept any callable."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()
        received = []

        class Callable:
            def __call__(self, event: TraceEvent):
                received.append(event)

        bus.subscribe(Callable())
        bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id="s1"))

        assert len(received) == 1

    def test_subscriber_receives_all_event_types(self):
        """Subscribers should receive all event types."""
        from cadence.primitives.bus import TraceBus
        from cadence.primitives.events import TraceEvent, EventType

        bus = TraceBus()
        types_received = []

        bus.subscribe(lambda e: types_received.append(e.type))

        bus.emit(TraceEvent(type=EventType.START, trace_id="t1", span_id="s1"))
        bus.emit(TraceEvent(type=EventType.OUTPUT, trace_id="t1", span_id="s1"))
        bus.emit(TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"))

        assert EventType.START in types_received
        assert EventType.OUTPUT in types_received
        assert EventType.SUCCESS in types_received
