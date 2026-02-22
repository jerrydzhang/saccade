from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_scripts_dir = Path(__file__).parent.parent
_project_root = _scripts_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_scripts_dir))

from helpers import create_llm_from_env, print_memory_state, print_trace_projections
from saccade import Span, Trace
from research_agent.memory import WorkingMemory


async def main() -> None:
    llm = create_llm_from_env()

    memory = WorkingMemory(model="gpt-4o-mini")
    memory.set_system("You are a helpful assistant. Be concise.")

    print("\n" + "=" * 60)
    print("SCENARIO 1: Simple Conversation")
    print("=" * 60)

    with Trace() as trace:
        memory.add_user("What is 2 + 2? Answer with just the number.")
        print("\n[user] What is 2 + 2?")

        with Span("llm_turn", kind="llm"):
            response = await llm.complete(messages=memory.to_list())

        memory.add_assistant(
            content=response.content,
            prompt_tokens=response.usage.input_tokens if response.usage else None,
        )

        print(f"[assistant] {response.content}")

    print_memory_state(memory)
    print_trace_projections(trace)

    assert len(memory.to_list()) == 3, "Expected 3 messages (system, user, assistant)"
    assert response.content, "Expected non-empty response"

    print("\n✓ Scenario 1 passed!")


if __name__ == "__main__":
    asyncio.run(main())
