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
    memory.set_system("You are a helpful assistant. Use tools when appropriate.")

    print("\n" + "=" * 60)
    print("SCENARIO 2: Single Tool Call")
    print("=" * 60)

    with Trace() as trace:
        memory.add_user("What is 25 + 17?")
        print("\n[user] What is 25 + 17?")

        with Span("llm_turn_1", kind="llm"):
            response1 = await llm.complete(
                messages=memory.to_list(),
                tools=registry.get_schemas(),
            )

        memory.add_assistant(
            content=response1.content,
            tool_calls=response1.tool_calls,
            prompt_tokens=response1.usage.input_tokens if response1.usage else None,
        )

        if response1.tool_calls:
            for tc in response1.tool_calls:
                fn = tc["function"]
                print(f"[assistant] Tool call: {format_tool_call(tc)}")

                result = await registry.execute(fn["name"], json.loads(fn["arguments"]))
                print(f"[tool] {fn['name']} -> {result}")

                memory.add_tool_result(tc["id"], str(result))

            with Span("llm_turn_2", kind="llm"):
                response2 = await llm.complete(
                    messages=memory.to_list(),
                    tools=registry.get_schemas(),
                )

            memory.add_assistant(
                content=response2.content,
                prompt_tokens=response2.usage.input_tokens if response2.usage else None,
            )
            print(f"[assistant] {response2.content}")
        else:
            print(f"[assistant] {response1.content}")

    print_memory_state(memory)
    print_trace_projections(trace)

    assert len(memory.to_list()) >= 3, "Expected at least 3 messages"

    print("\n✓ Scenario 2 passed!")


if __name__ == "__main__":
    asyncio.run(main())
