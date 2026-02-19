from __future__ import annotations

from typing import TYPE_CHECKING

from ulid import ULID

if TYPE_CHECKING:
    from collections.abc import Callable

    from saccade.primitives.events import TraceEvent


class TraceBus:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id: str = trace_id or str(ULID())
        self._events: list[TraceEvent] = []
        self._subscribers: list[Callable[[TraceEvent], None]] = []
        self._running_spans: list[str] = []

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def subscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, event: TraceEvent) -> None:
        self._events.append(event)
        for callback in list(self._subscribers):
            try:  # noqa: SIM105
                callback(event)
            except Exception:  # noqa: S110, BLE001
                pass

    def _register_span(self, span_name: str) -> None:
        self._running_spans.append(span_name)

    def _unregister_span(self, span_name: str) -> None:
        if span_name in self._running_spans:
            self._running_spans.remove(span_name)

    @property
    def running_spans(self) -> list[str]:
        return list(self._running_spans)
