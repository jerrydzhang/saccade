#!/usr/bin/env python3
"""
Test script to verify ContextVar propagation through asyncio primitives.

This validates that our _current_span_id contextvar correctly propagates to:
1. asyncio.create_task()
2. asyncio.TaskGroup
3. asyncio.gather()
4. Nested context managers
"""

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any

# Simulate our contextvar setup
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "talos_span_id", default=None
)


@dataclass
class SpanInfo:
    """Captured span context at a point in time."""

    name: str
    span_id: str
    parent_id: str | None
    task_name: str


def capture_context(name: str) -> SpanInfo:
    """Helper to capture current context."""
    try:
        current_task = asyncio.current_task()
        task_name = current_task.get_name() if current_task else "no-task"
    except RuntimeError:
        task_name = "no-task"

    return SpanInfo(
        name=name,
        span_id=_current_span_id.get() or "NONE",
        parent_id=None,  # Will be filled by caller
        task_name=task_name,
    )


class FakeSpan:
    """Minimal span that sets contextvar."""

    def __init__(self, name: str):
        self.name = name
        self.id = f"span_{name}_{id(self)}"
        self._token = None

    def __enter__(self):
        self._token = _current_span_id.set(self.id)
        return self

    def __exit__(self, *args):
        if self._token:
            _current_span_id.reset(self._token)


async def test_nested_context_managers():
    """Test 1: Basic nesting (baseline - should always work)."""
    print("\n=== Test 1: Nested Context Managers ===")
    results = []

    with FakeSpan("root") as root:
        results.append(("root", _current_span_id.get(), None))

        with FakeSpan("child") as child:
            results.append(("child", _current_span_id.get(), root.id))

        results.append(("root_after", _current_span_id.get(), None))

    for name, span_id, parent in results:
        print(
            f"  {name}: span_id={span_id[:20]}... parent={parent[:20] if parent else 'None'}..."
        )

    assert results[0][1] == root.id
    assert results[1][1] == child.id
    assert results[2][1] == root.id
    print("  ✓ Nested context managers work correctly")


async def test_create_task():
    """Test 2: Does contextvar propagate through create_task?"""
    print("\n=== Test 2: asyncio.create_task() ===")
    results = []

    async def worker(name: str, expected_parent: str | None):
        current = _current_span_id.get()
        results.append((name, current, expected_parent))
        print(
            f"  {name}: span_id={current[:20] if current else 'None'}... expected_parent={expected_parent[:20] if expected_parent else 'None'}..."
        )

    with FakeSpan("root") as root:
        # Create tasks while inside the span context
        t1 = asyncio.create_task(worker("task1", root.id))
        t2 = asyncio.create_task(worker("task2", root.id))
        await asyncio.gather(t1, t2)

    # Verify
    assert results[0][1] == root.id, f"Expected {root.id}, got {results[0][1]}"
    assert results[1][1] == root.id, f"Expected {root.id}, got {results[1][1]}"
    print("  ✓ ContextVar propagates through create_task()")


async def test_taskgroup():
    """Test 3: Does contextvar propagate through TaskGroup?"""
    print("\n=== Test 3: asyncio.TaskGroup ===")
    results = []

    async def worker(name: str, expected_parent: str | None):
        current = _current_span_id.get()
        results.append((name, current, expected_parent))
        print(
            f"  {name}: span_id={current[:20] if current else 'None'}... expected_parent={expected_parent[:20] if expected_parent else 'None'}..."
        )

    with FakeSpan("root") as root:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(worker("tg1", root.id))
            tg.create_task(worker("tg2", root.id))
            tg.create_task(worker("tg3", root.id))

    assert all(r[1] == root.id for r in results), "Not all tasks got parent context"
    print("  ✓ ContextVar propagates through TaskGroup")


async def test_gather():
    """Test 4: Does contextvar propagate through gather?"""
    print("\n=== Test 4: asyncio.gather() ===")
    results = []

    async def worker(name: str, expected_parent: str | None):
        current = _current_span_id.get()
        results.append((name, current, expected_parent))
        print(
            f"  {name}: span_id={current[:20] if current else 'None'}... expected_parent={expected_parent[:20] if expected_parent else 'None'}..."
        )

    with FakeSpan("root") as root:
        await asyncio.gather(
            worker("g1", root.id),
            worker("g2", root.id),
        )

    assert all(r[1] == root.id for r in results), (
        "Not all coroutines got parent context"
    )
    print("  ✓ ContextVar propagates through gather()")


async def test_nested_spans_in_tasks():
    """Test 5: Each task can have its own nested span."""
    print("\n=== Test 5: Nested spans within tasks ===")
    results = []

    async def worker(name: str, parent_id: str):
        # At this point, we should have the parent context
        outer = _current_span_id.get()
        results.append((f"{name}.enter", outer, parent_id))
        print(
            f"  {name}.enter: span_id={outer[:20] if outer else 'None'}... expected={parent_id[:20]}..."
        )

        # Create our own nested span
        with FakeSpan(name) as my_span:
            inner = _current_span_id.get()
            results.append((f"{name}.nested", inner, my_span.id))
            print(
                f"  {name}.nested: span_id={inner[:20] if inner else 'None'}... expected={my_span.id[:20]}..."
            )

        # Should be back to parent
        outer_again = _current_span_id.get()
        results.append((f"{name}.exit", outer_again, parent_id))
        print(
            f"  {name}.exit: span_id={outer_again[:20] if outer_again else 'None'}... expected={parent_id[:20]}..."
        )

    with FakeSpan("root") as root:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(worker("w1", root.id))
            tg.create_task(worker("w2", root.id))

    # Verify each worker got the parent context initially
    for name, actual, expected in results:
        if ".enter" in name or ".exit" in name:
            assert actual == expected, f"{name}: expected {expected}, got {actual}"
        elif ".nested" in name:
            assert actual == expected, (
                f"{name}: expected own span {expected}, got {actual}"
            )

    print("  ✓ Nested spans within tasks work correctly")


async def test_task_created_outside_span():
    """Test 6: Task created outside span context shouldn't have it."""
    print("\n=== Test 6: Task created outside span context ===")
    results = []

    async def worker(name: str):
        current = _current_span_id.get()
        results.append((name, current))
        print(f"  {name}: span_id={current}")

    # Create task OUTSIDE any span
    t1 = asyncio.create_task(worker("outside"))

    # Now enter a span
    with FakeSpan("root") as root:
        t2 = asyncio.create_task(worker("inside"))
        await asyncio.gather(t1, t2)

    assert results[0][1] is None, (
        f"Task created outside should have None, got {results[0][1]}"
    )
    assert results[1][1] == root.id, (
        f"Task created inside should have root.id, got {results[1][1]}"
    )
    print("  ✓ Tasks created outside span context correctly have no parent")


async def test_deeply_nested_parallel():
    """Test 7: Complex nesting with parallel tasks."""
    print("\n=== Test 7: Complex nested parallel execution ===")
    results = []

    async def leaf(name: str, expected_ancestors: list[str]):
        current = _current_span_id.get()
        results.append((name, current))
        print(f"  {name}: span_id={current[:20] if current else 'None'}...")

    async def middle(name: str, parent_id: str):
        with FakeSpan(name) as m:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(leaf(f"{name}_leaf1", [m.id, parent_id]))
                tg.create_task(leaf(f"{name}_leaf2", [m.id, parent_id]))

    with FakeSpan("root") as root:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(middle("m1", root.id))
            tg.create_task(middle("m2", root.id))

    # All results should have some span_id (not None)
    assert all(r[1] is not None for r in results), "Some tasks had no context"
    print("  ✓ Complex nested parallel execution maintains context")


async def main():
    print("=" * 60)
    print("ContextVar Propagation Tests for Talos")
    print("=" * 60)

    try:
        await test_nested_context_managers()
        await test_create_task()
        await test_taskgroup()
        await test_gather()
        await test_nested_spans_in_tasks()
        await test_task_created_outside_span()
        await test_deeply_nested_parallel()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nConclusion: ContextVar correctly propagates through all")
        print("asyncio primitives (create_task, TaskGroup, gather).")
        print("Our auto-context capture design will work correctly.")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
