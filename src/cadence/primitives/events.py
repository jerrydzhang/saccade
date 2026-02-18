from __future__ import annotations

import time
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID


class TokenMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cached: int = 0
    cache_write: int = 0

    def __add__(self, other: TokenMetrics) -> TokenMetrics:
        return TokenMetrics(
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
            cached=self.cached + other.cached,
            cache_write=self.cache_write + other.cache_write,
        )

    @property
    def total(self) -> int:
        return self.input + self.output


class CostMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    usd: Decimal = Decimal("0.0")

    def __add__(self, other: CostMetrics) -> CostMetrics:
        return CostMetrics(usd=self.usd + other.usd)


class LatencyMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    has_clock_skew: bool = False

    def __add__(self, other: LatencyMetrics) -> LatencyMetrics:
        return LatencyMetrics(
            total_ms=self.total_ms + other.total_ms,
            has_clock_skew=self.has_clock_skew or other.has_clock_skew,
        )


class OperationMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    provider: str | None = None
    host: str | None = None
    kind: str = "generic"
    correlation_id: str | None = None


class EventType(StrEnum):
    START = "START"
    CHUNK = "CHUNK"
    OUTPUT = "OUTPUT"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCEL = "CANCEL"


class SpanKind:
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"


class Relation(StrEnum):
    CONTEXT = "context"
    DATAFLOW = "dataflow"


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(ULID()))
    type: EventType
    span_id: str
    trace_id: str | None = None
    timestamp: float = Field(default_factory=time.time)

    name: str | None = None
    inputs: dict[str, Any] | None = None
    kind: str | None = None
    relations: dict[str, list[str]] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    chunk: str | dict[str, Any] | None = None
    output: Any = None
    error: str | None = None
    response_id: str | None = None

    operation: OperationMeta | None = None
    tokens: TokenMetrics | None = None
    cost: CostMetrics | None = None
    latency: LatencyMetrics | None = None
