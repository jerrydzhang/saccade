"""Shared helpers for validation scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from vcr import VCR

    from research_agent.memory import WorkingMemory
    from research_agent.tools import ToolRegistry
    from saccade import Trace
    from saccade.integrations.litellm import TracedLiteLLM

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

VCR_CASSETTE_DIR = Path(__file__).parent / "cassettes"


def create_vcr() -> VCR:
    import vcr

    return vcr.VCR(
        filter_headers=["Authorization"],
        decode_compressed_response=True,
        cassette_library_dir=str(VCR_CASSETTE_DIR),
        record_mode="new_episodes",
    )


def create_llm_from_env() -> TracedLiteLLM:
    from saccade.integrations.litellm import OpenAICompatibleProvider, TracedLiteLLM

    model = os.getenv("PROVIDER_MODEL")
    api_key = os.getenv("PROVIDER_API_KEY")
    api_base = os.getenv("PROVIDER_API_BASE")

    if not model:
        raise RuntimeError("PROVIDER_MODEL environment variable is required")

    if api_base and api_key:
        provider_name = model.split("/")[0] if "/" in model else "custom"
        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_base=api_base,
            api_key=api_key,
        )
        print(f"[setup] Using custom provider: {provider_name}")
        print(f"[setup] Model: {model}")
        print(f"[setup] API base: {api_base}")
        return TracedLiteLLM(model=model, provider=provider)

    print(f"[setup] Using standard provider (no custom config)")
    print(f"[setup] Model: {model}")
    return TracedLiteLLM(model=model)


def create_registry_with_tools() -> ToolRegistry:
    from research_agent.tools import ToolRegistry, tool

    @tool
    def add(a: int, b: int) -> str:
        return str(a + b)

    @tool
    def multiply(a: int, b: int) -> str:
        return str(a * b)

    @tool
    def echo(text: str) -> str:
        return f"Echo: {text}"

    registry = ToolRegistry()
    registry.register(add)
    registry.register(multiply)
    registry.register(echo)

    print(f"[setup] Registered tools: {list(registry._tools.keys())}")
    return registry


def print_trace_projections(trace: Trace) -> None:
    from saccade import project_cost, project_state, project_timeline, project_tree

    tree = project_tree(trace.events)
    cost = project_cost(trace.events)
    state = project_state(trace.events)
    timeline = project_timeline(trace.events)

    print("\n" + "=" * 60)
    print("TRACE TREE")
    print("=" * 60)

    if not tree.roots:
        print("  (no spans)")
        return

    def print_node(node, indent: int = 0) -> None:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        print(f"{prefix}{node.name} (kind={node.kind or 'unknown'})")

        if node.output:
            output_str = str(node.output)
            if len(output_str) > 50:
                output_str = output_str[:47] + "..."
            print(f"{'  ' * (indent + 1)}output: {output_str!r}")

        if node.tokens and (node.tokens.input > 0 or node.tokens.output > 0):
            print(f"{'  ' * (indent + 1)}tokens: in={node.tokens.input}, out={node.tokens.output}")

        for child in node.children:
            print_node(child, indent + 1)

    for root in tree.roots:
        print_node(root)

    print("\n" + "-" * 60)
    print(f"Total events: {len(trace.events)}")
    print(f"Total tokens: in={tree.total_tokens.input}, out={tree.total_tokens.output}")
    if cost.cost.usd:
        print(f"Total cost: ${cost.cost.usd}")

    print("\n" + "=" * 60)
    print("COST VIEW")
    print("=" * 60)
    print(f"Span count: {cost.span_count}")
    print(f"Tokens: in={cost.tokens.input}, out={cost.tokens.output}, cached={cost.tokens.cached}")
    if cost.cost.usd:
        print(f"Cost: ${cost.cost.usd}")
    if cost.latency.total_ms:
        print(f"Latency: {cost.latency.total_ms:.1f}ms")

    if cost.by_kind:
        print("\nBy kind:")
        for kind, group in cost.by_kind.items():
            print(
                f"  {kind}: tokens(in={group.tokens.input}, out={group.tokens.output}), count={group.count}"
            )

    if cost.by_name:
        print("\nBy name:")
        for name, group in cost.by_name.items():
            print(f"  {name}: tokens(in={group.tokens.input}, out={group.tokens.output})")

    print("\n" + "=" * 60)
    print("TIMELINE VIEW")
    print("=" * 60)
    print(f"Total duration: {timeline.total_duration_ms:.1f}ms")
    print(f"Span count: {len(timeline.spans)}")
    print("\nSpans:")
    for span in timeline.spans:
        duration = f"{span.duration_ms:.1f}ms" if span.end_ms else "running"
        print(
            f"  {span.name}: [{span.start_ms:.1f}ms - {span.end_ms or '?'}ms] duration={duration}, depth={span.depth}"
        )

    print("\n" + "=" * 60)
    print("STATE VIEW")
    print("=" * 60)
    print(f"Span records: {len(state.snapshots)}")
    for record in state.snapshots:
        chunk_count = len(record.chunks)
        output_count = len(record.outputs)
        print(f"  {record.span_id[:16]}...: chunks={chunk_count}, outputs={output_count}")


def print_memory_state(memory: WorkingMemory) -> None:
    print("\n" + "=" * 60)
    print("MEMORY STATE")
    print("=" * 60)

    messages = memory.to_list()
    print(f"Message count: {len(messages)}")
    print(f"Estimated tokens: {memory.estimate_total_tokens()}")
    print(f"Fits in context: {memory.fits_in_context()}")

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if content and len(content) > 60:
            content = content[:57] + "..."

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_names = [tc["function"]["name"] for tc in tool_calls]
            print(f"  [{i}] {role}: tool_calls={tc_names}")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "?")
            print(f"  [{i}] {role}: id={tool_call_id[:16]}... content={content!r}")
        else:
            print(f"  [{i}] {role}: {content!r}")


def format_tool_call(tc: dict) -> str:
    fn = tc.get("function", {})
    name = fn.get("name", "?")
    args = fn.get("arguments", "{}")
    try:
        args_dict = json.loads(args) if isinstance(args, str) else args
        args_str = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
    except json.JSONDecodeError:
        args_str = args
    return f"{name}({args_str})"
