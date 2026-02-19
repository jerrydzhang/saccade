from __future__ import annotations

import asyncio
import time
import warnings
from contextvars import ContextVar, Token
from decimal import Decimal
from typing import Any, Self

from ulid import ULID

from saccade.primitives.events import (
    CostMetrics,
    EventType,
    LatencyMetrics,
    OperationMeta,
    TokenMetrics,
    TraceEvent,
)
from saccade.primitives.trace import _current_bus

_current_span: ContextVar[Span | None] = ContextVar("_current_span", default=None)


class Span:
    def __init__(
        self,
        name: str,
        *,
        kind: str = "generic",
        inputs: dict[str, Any] | None = None,
        relations: dict[str, list[str]] | None = None,
    ) -> None:
        self.id: str = str(ULID())
        self.name = name
        self.kind = kind
        self.inputs = inputs or {}
        self.status: str = "PENDING"
        self.relations: dict[str, list[str]] = {}
        self.output: Any = None
        self.error: str | None = None

        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._chunks: list[str] = []
        self._pending_tokens: TokenMetrics = TokenMetrics()
        self._pending_cost: CostMetrics = CostMetrics()
        self._pending_operation: OperationMeta | None = None
        self._token: Token[Span | None] | None = None

        if relations:
            self._validate_relations(relations)
            self.relations = dict(relations)

    def _validate_relations(self, relations: dict[str, Any]) -> None:
        for rel_type, targets in relations.items():
            if not isinstance(targets, list):
                msg = f"Relations value for '{rel_type}' must be list"
                raise TypeError(msg)
            for target in targets:
                if not isinstance(target, str):
                    msg = f"Relations list for '{rel_type}' must contain str"
                    raise TypeError(msg)

    @classmethod
    def current(cls) -> Span | None:
        """Get the currently active span from context, if any."""
        return _current_span.get()

    def relate(self, relation_type: str, span_id: str) -> None:
        if not isinstance(relation_type, str):
            msg = "relation_type must be str"
            raise TypeError(msg)
        if not isinstance(span_id, str):
            msg = "span_id must be str"
            raise TypeError(msg)

        if relation_type not in self.relations:
            self.relations[relation_type] = []
        if span_id not in self.relations[relation_type]:
            self.relations[relation_type].append(span_id)

    def __enter__(self) -> Self:
        self._start_time = time.time()
        self.status = "RUNNING"

        parent_span = _current_span.get()
        if parent_span:
            self.relate("context", parent_span.id)

        self._token = _current_span.set(self)

        bus = _current_bus.get()
        if bus:
            if hasattr(bus, "_register_span"):
                bus._register_span(self.name)  # noqa: SLF001

            event = TraceEvent(
                type=EventType.START,
                span_id=self.id,
                trace_id=bus.trace_id,
                name=self.name,
                inputs=self.inputs or None,
                kind=self.kind,
                relations=dict(self.relations) or None,
            )
            bus.emit(event)
        else:
            warnings.warn(
                "Span has no TraceBus - events will not be recorded",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        end_time = time.time()
        total_ms = (end_time - (self._start_time or end_time)) * 1000

        if self._token:
            _current_span.reset(self._token)

        bus = _current_bus.get()

        if bus and hasattr(bus, "_unregister_span"):
            bus._unregister_span(self.name)  # noqa: SLF001

        if exc_type is asyncio.CancelledError:
            self.status = "CANCELLED"
            if bus:
                event = TraceEvent(
                    type=EventType.CANCEL,
                    span_id=self.id,
                    trace_id=bus.trace_id,
                    name=self.name,
                    latency=LatencyMetrics(total_ms=total_ms),
                )
                bus.emit(event)
        elif exc_type is not None:
            self.status = "FAILED"
            self.error = str(exc_val) if exc_val else "Unknown error"
            if bus:
                # Emit OUTPUT if set, so partial results are preserved on error
                if self.output is not None:
                    output_event = TraceEvent(
                        type=EventType.OUTPUT,
                        span_id=self.id,
                        trace_id=bus.trace_id,
                        name=self.name,
                        output=self.output,
                    )
                    bus.emit(output_event)

                event = TraceEvent(
                    type=EventType.ERROR,
                    span_id=self.id,
                    trace_id=bus.trace_id,
                    name=self.name,
                    error=self.error,
                    latency=LatencyMetrics(total_ms=total_ms),
                )
                bus.emit(event)
        else:
            self.status = "COMPLETED"
            if bus:
                if self.output is not None:
                    output_event = TraceEvent(
                        type=EventType.OUTPUT,
                        span_id=self.id,
                        trace_id=bus.trace_id,
                        name=self.name,
                        output=self.output,
                    )
                    bus.emit(output_event)

                latency = LatencyMetrics(
                    total_ms=total_ms,
                    time_to_first_token_ms=(
                        (self._first_token_time - self._start_time) * 1000
                        if self._first_token_time and self._start_time
                        else None
                    ),
                )

                success_event = TraceEvent(
                    type=EventType.SUCCESS,
                    span_id=self.id,
                    trace_id=bus.trace_id,
                    name=self.name,
                    tokens=self._pending_tokens if self._pending_tokens.input > 0 else None,
                    cost=self._pending_cost if self._pending_cost.usd > Decimal(0) else None,
                    latency=latency,
                    operation=self._pending_operation,
                )
                bus.emit(success_event)

        return False

    def stream(self, chunk: str) -> None:
        if self._first_token_time is None and self._start_time:
            self._first_token_time = time.time()
        self._chunks.append(chunk)

        bus = _current_bus.get()
        if bus:
            event = TraceEvent(
                type=EventType.CHUNK,
                span_id=self.id,
                trace_id=bus.trace_id,
                name=self.name,
                chunk=chunk,
            )
            bus.emit(event)

    def set_output(self, value: Any) -> None:
        self.output = value

    def set_metrics(
        self,
        tokens: TokenMetrics | None = None,
        cost: CostMetrics | None = None,
    ) -> None:
        if tokens:
            self._pending_tokens = self._pending_tokens + tokens
        if cost:
            self._pending_cost = self._pending_cost + cost

    def set_meta(self, operation: OperationMeta) -> None:
        self._pending_operation = operation
