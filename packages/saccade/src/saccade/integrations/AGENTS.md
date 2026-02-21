# SACCADE INTEGRATIONS - AGENTS.md

External library integrations with saccade tracing.

## OVERVIEW

LiteLLM wrapper with automatic saccade span integration. Streaming-first design with cost tracking.

## STRUCTURE

```
integrations/
├── __init__.py          # Re-exports: TracedLiteLLM, OpenAICompatibleProvider, etc.
└── litellm/
    ├── __init__.py      # Module exports
    ├── traced_llm.py    # TracedLiteLLM class (310 lines)
    ├── openai_provider.py  # Custom provider for OpenAI-compatible APIs (344 lines)
    └── pricing.py       # ModelPricing, RegisterableProvider (76 lines)
```

## WHERE TO LOOK

| Task | File | Key Symbols |
|------|------|-------------|
| Add LLM method | `traced_llm.py` | `TracedLiteLLM.stream()`, `TracedLiteLLM.complete()` |
| Add custom provider | `openai_provider.py` | `OpenAICompatibleProvider` |
| Add pricing model | `pricing.py` | `ModelPricing`, `RegisterableProvider` |
| Change exports | `__init__.py` | `__all__` |

## CODE MAP

| Symbol | Type | Role |
|--------|------|------|
| `TracedLiteLLM` | class | LiteLLM wrapper with saccade span integration |
| `OpenAICompatibleProvider` | class | Custom LiteLLM provider for OpenAI-compatible APIs |
| `ModelPricing` | dataclass | Per-token cost configuration |
| `RegisterableProvider` | ABC | Base class for custom providers with pricing |
| `Usage` | dataclass | Token usage (input, output, reasoning) |
| `StreamChunk` | dataclass | Streaming chunk with delta, tool_calls, usage |
| `Completion` | dataclass | Non-streaming completion result |

## KEY PATTERNS

### TracedLiteLLM Usage
```python
from saccade.integrations.litellm import TracedLiteLLM

llm = TracedLiteLLM(model="openai/gpt-4o-mini")

# Streaming (default)
async for chunk in llm.stream(messages=[{"role": "user", "content": "Hello"}]):
    print(chunk.delta, end="")

# Non-streaming
result = await llm.complete(messages=[...])
print(result.content)
```

### Custom Provider Registration
```python
from saccade.integrations.litellm import OpenAICompatibleProvider, ModelPricing
from decimal import Decimal

provider = OpenAICompatibleProvider(
    provider_name="my-provider",
    api_base="https://api.example.com/v1",
    api_key="sk-xxx",
    pricing={
        "llama-3.1-70b": ModelPricing(
            input_cost_per_token=Decimal("0.0000005"),
            output_cost_per_token=Decimal("0.0000015"),
        )
    },
)

llm = TracedLiteLLM(model="my-provider/llama-3.1-70b", provider=provider)
```

### Span Integration
- `TracedLiteLLM` calls `Span.current()` to get active span
- Streaming emits `CHUNK` events via `span.stream(delta)`
- Completion sets output via `span.set_output(content)`
- Metrics captured via `span.set_metrics(tokens=..., cost=...)`
- Metadata via `span.set_meta(OperationMeta(model=..., kind="llm"))`

### Cost Extraction Flow
```
1. Try LiteLLM hidden_params.response_cost
2. Fall back to provider.calculate_cost()
3. Return None if no cost available
```

## ANTI-PATTERNS

| Avoid | Use Instead |
|-------|-------------|
| Direct `litellm.acompletion()` | `TracedLiteLLM.stream()` or `.complete()` |
| Manual span creation for LLM calls | Let `TracedLiteLLM` handle it |
| Hardcoded pricing | `ModelPricing` dataclass |

## NOTES

- Streaming requires `stream_options: {include_usage: true}` for token counts
- `OpenAICompatibleProvider` inherits from `CustomLLM` + `RegisterableProvider`
- Provider registration happens in `__init__` if provider is passed
- `_raw` field on `StreamChunk`/`Completion` preserves original response
- LiteLLM private access (`_hidden_params`, `_usage`) allowed via ruff overrides
