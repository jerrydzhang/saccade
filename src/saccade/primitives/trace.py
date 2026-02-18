from __future__ import annotations

import warnings
from contextvars import ContextVar
from typing import TYPE_CHECKING, Self

from saccade.primitives.bus import TraceBus

if TYPE_CHECKING:
    from collections.abc import Callable

    from saccade.primitives.events import TraceEvent


_current_bus: ContextVar[TraceBus | None] = ContextVar("_current_bus", default=None)


class Trace:
    def __init__(self, trace_id: str | None = None) -> None:
        self.bus = TraceBus(trace_id=trace_id)
        self._token = None

    def __enter__(self) -> Self:
        self._token = _current_bus.set(self.bus)
        return self

    def __exit__(self, *args: object) -> None:
        if self.bus.running_spans:
            for span_name in self.bus.running_spans:
                warnings.warn(
                    f"Span '{span_name}' is still running",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self._token:
            _current_bus.reset(self._token)

    @property
    def events(self) -> list[TraceEvent]:
        return self.bus.events

    @property
    def trace_id(self) -> str:
        return self.bus.trace_id

    def subscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        self.bus.subscribe(callback)
