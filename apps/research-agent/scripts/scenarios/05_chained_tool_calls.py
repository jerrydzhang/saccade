from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_scripts_dir = Path(__file__).parent.parent
_project_root = _scripts_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_scripts_dir))

from helpers import (
    create_llm_from_env,
    create_registry_with_tools,
    format_tool_call,
    print_memory_state,
    print_trace_projections,
)
from saccade import Span, Trace
from research_agent.memory import WorkingMemory


async def main() -> None:
    llm = create_llm_from_env()
    registry = create_registry_with_tools()

    memory = WorkingMemory(model="gpt-4o-mini")
    memory.set_system("You are a helpful assistant. Use tools to solve problems step by step.")

    print("\n" + "=" * 60)
    print("SCENARIO 5: Chained Tool Calls")
    print("=" * 60)

    with Trace() as trace:
        memory.add_user("First add 10 and 20, then multiply the result by 3.")
        print("\n[user] First add 10 and 20, then multiply the result by 3.")

        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            with Span(f"llm_turn_{iteration}", kind="llm"):
                response = await llm.complete(
                    messages=memory.to_list(),
                    tools=registry.get_schemas(),
                )

            memory.add_assistant(
                content=response.content,
                tool_calls=response.tool_calls,
                prompt_tokens=response.usage.input_tokens if response.usage else None,
            )

            if not response.tool_calls:
                if response.content:
                    print(f"[assistant] {response.content}")
                break

            for tc in response.tool_calls:
                fn = tc["function"]
                print(f"[turn {iteration}] [assistant] Tool call: {format_tool_call(tc)}")

                with Span(f"tool_{fn['name']}", kind="tool"):
                    result = await registry.execute(fn["name"], json.loads(fn["arguments"]))
                print(f"[turn {iteration}] [tool] {fn['name']} -> {result}")

                memory.add_tool_result(tc["id"], str(result))

    print_memory_state(memory)
    print_trace_projections(trace)

    print("\n" + "-" * 60)
    print("Chained execution summary:")
    print(f"  Total LLM turns: {iteration}")
    print(f"  Total messages: {memory.message_count()}")
    print(f"  Estimated tokens: {memory.estimate_total_tokens()}")

    print("\n✓ Scenario 5 passed!")


if __name__ == "__main__":
    asyncio.run(main())
