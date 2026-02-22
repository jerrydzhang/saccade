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
    memory.set_system("You are a helpful assistant. Keep track of our conversation.")

    print("\n" + "=" * 60)
    print("SCENARIO 4: Multi-Turn Conversation")
    print("=" * 60)

    with Trace() as trace:
        with Span("turn_1", kind="agent"):
            memory.add_user("My name is Alice.")
            print("\n[turn 1] [user] My name is Alice.")

            with Span("llm", kind="llm"):
                response1 = await llm.complete(messages=memory.to_list())

            memory.add_assistant(
                content=response1.content,
                prompt_tokens=response1.usage.input_tokens if response1.usage else None,
            )
            print(f"[turn 1] [assistant] {response1.content}")

        with Span("turn_2", kind="agent"):
            memory.add_user("What is 10 + 5?")
            print("\n[turn 2] [user] What is 10 + 5?")

            with Span("llm", kind="llm"):
                response2 = await llm.complete(
                    messages=memory.to_list(),
                    tools=registry.get_schemas(),
                )

            memory.add_assistant(
                content=response2.content,
                tool_calls=response2.tool_calls,
                prompt_tokens=response2.usage.input_tokens if response2.usage else None,
            )

            if response2.tool_calls:
                for tc in response2.tool_calls:
                    fn = tc["function"]
                    print(f"[turn 2] [assistant] Tool call: {format_tool_call(tc)}")

                    result = await registry.execute(fn["name"], json.loads(fn["arguments"]))
                    print(f"[turn 2] [tool] {fn['name']} -> {result}")

                    memory.add_tool_result(tc["id"], str(result))

                with Span("llm", kind="llm"):
                    response2b = await llm.complete(
                        messages=memory.to_list(),
                        tools=registry.get_schemas(),
                    )

                memory.add_assistant(
                    content=response2b.content,
                    prompt_tokens=response2b.usage.input_tokens if response2b.usage else None,
                )
                print(f"[turn 2] [assistant] {response2b.content}")
            else:
                print(f"[turn 2] [assistant] {response2.content}")

        with Span("turn_3", kind="agent"):
            memory.add_user("Do you remember my name?")
            print("\n[turn 3] [user] Do you remember my name?")

            with Span("llm", kind="llm"):
                response3 = await llm.complete(messages=memory.to_list())

            memory.add_assistant(
                content=response3.content,
                prompt_tokens=response3.usage.input_tokens if response3.usage else None,
            )
            print(f"[turn 3] [assistant] {response3.content}")

    print_memory_state(memory)
    print_trace_projections(trace)

    print("\n" + "-" * 60)
    print("Context tracking:")
    print(f"  Total turns: 3")
    print(f"  Total messages: {memory.message_count()}")
    print(f"  Estimated tokens: {memory.estimate_total_tokens()}")
    print(f"  Fits in context: {memory.fits_in_context()}")

    print("\n✓ Scenario 4 passed!")


if __name__ == "__main__":
    asyncio.run(main())
