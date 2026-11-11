// The composition skeleton's construction tests (slice 1): the ratified
// glue behaves per call, the projection preserves every state shape,
// latest-intent refresh wins, and destruction forwards through the
// registry facade. Pure node tests over the same modules the app
// builds with; no DOM, no svelte runtime (the component type enters
// the glue only as opaque data).

import test from "node:test";
import assert from "node:assert/strict";
import type { Component } from "svelte";
import {
  createBufferFactory,
  createDiagnostics,
  createViewport,
  mapBufferState,
} from "../src/components/Buffer.ts";
import type { BufferAccess, ContentFrame, InstrumentAttachment } from "../src/components/Buffer.ts";
import {
  constructInstrument,
  registerInstrument,
  type RunningInstrument,
  type WindowEnvironment,
} from "../src/lib/instruments/registry.ts";
import type { InstrumentDefinition, InstrumentProps } from "../src/lib/instruments/contract.ts";
import { knownIssue } from "../src/lib/contracts/buffer.ts";
import { commandContext } from "../src/lib/contracts/input.ts";
import type { BoardAnchor, ContentCapture, ContentPlaces } from "../src/lib/contracts/places.ts";

interface Frame {
  readonly label: string;
}
interface Controller {
  readonly tag: string;
}

// An inert component stand-in: the glue treats it as opaque data.
const stubComponent = (() => null) as unknown as Component<InstrumentProps<Frame, Controller>>;

function stubPlaces(): ContentPlaces<BoardAnchor> {
  return {
    captureDisplayed(): ContentCapture<BoardAnchor> {
      return { address: { machine: "", path: "" }, position: { kind: "default" } };
    },
    async request() {
      return "applied" as const;
    },
    acceptsAnchor(data): data is BoardAnchor {
      return (
        typeof data === "object" &&
        data !== null &&
        (data as { kind?: unknown }).kind === "board-part"
      );
    },
    revealAnchor() {
      return { kind: "missing" as const };
    },
    defaultPosition() {
      return { left: 0, top: 0 };
    },
    subscribeChanges() {
      return () => {};
    },
  };
}

function stubDefinition(
  tag: string,
  constructed: Controller[],
): InstrumentDefinition<Frame, Controller, BoardAnchor> {
  return {
    kind: "stub",
    Component: stubComponent,
    connect: (buffer: BufferAccess<Frame>): InstrumentAttachment<Controller, BoardAnchor> => {
      const controller: Controller = { tag };
      constructed.push(controller);
      void buffer;
      return {
        presentation: controller,
        input: commandContext(() => []),
        places: stubPlaces(),
        release() {},
      };
    },
  };
}

let minted = 0;

function environment(): WindowEnvironment {
  // the window host mints a fresh buffer id per buffer
  return {
    context: { bufferId: `win-buffer-${++minted}`, windowId: "win" },
    viewport: createViewport(),
    diagnostics: createDiagnostics(),
    buffers: createBufferFactory("win"),
  };
}

test("registration constructs separate controller/input/state per call", () => {
  const constructed: Controller[] = [];
  const registration = registerInstrument(stubDefinition("one", constructed));
  // the window mints a fresh environment per buffer: constructInstrument
  // is called only when the window needs a NEW buffer
  const first = constructInstrument(registration, environment());
  const second = constructInstrument(registration, environment());
  assert.equal(constructed.length, 2, "connect ran exactly once per construct");
  assert.notEqual(first.id, second.id, "each buffer carries its own id");
  const firstSnapshot = first.snapshot();
  const secondSnapshot = second.snapshot();
  assert.deepEqual(firstSnapshot, secondSnapshot, "both start at the same empty state");
  assert.notEqual(firstSnapshot, secondSnapshot, "state objects are distinct");
  first.destroy();
  second.destroy();
});

test("mapBufferState preserves every state shape and projects values", () => {
  const project = (frame: Readonly<ContentFrame<Frame>>): Readonly<{ label: string }> => ({
    label: frame.value.label,
  });
  const frame = { value: { label: "shown" }, targets: [] };

  const destroyed = mapBufferState({ kind: "destroyed" }, project);
  assert.equal(destroyed.kind, "destroyed");

  const empty = mapBufferState(
    { kind: "alive", active: true, content: { kind: "empty" } },
    project,
  );
  assert.ok(empty.kind === "alive" && empty.content.kind === "empty");

  const pending = mapBufferState(
    { kind: "alive", active: false, content: { kind: "pending", previous: frame } },
    project,
  );
  assert.ok(pending.kind === "alive" && pending.content.kind === "pending");
  if (pending.kind === "alive" && pending.content.kind === "pending") {
    assert.ok(pending.content.previous !== null && pending.content.previous.label === "shown");
  }

  const pendingBare = mapBufferState(
    { kind: "alive", active: false, content: { kind: "pending", previous: null } },
    project,
  );
  if (pendingBare.kind === "alive" && pendingBare.content.kind === "pending") {
    assert.equal(pendingBare.content.previous, null);
  }

  const available = mapBufferState(
    {
      kind: "alive",
      active: true,
      content: { kind: "available", value: frame, lastFailure: null },
    },
    project,
  );
  if (available.kind === "alive" && available.content.kind === "available") {
    assert.equal(available.content.value.label, "shown");
    assert.equal(available.content.lastFailure, null);
  } else {
    assert.fail("available must stay available through the projection");
  }

  const unavailable = mapBufferState(
    { kind: "alive", active: false, content: { kind: "unavailable", failure: "d-1" } },
    project,
  );
  if (unavailable.kind === "alive" && unavailable.content.kind === "unavailable") {
    assert.equal(unavailable.content.failure, "d-1");
  } else {
    assert.fail("unavailable must stay unavailable through the projection");
  }
});

test("latest intent wins: a superseded refresh never publishes", async () => {
  // Two refreshes race: the first resolves late, the second early.
  // The late answer is superseded - the shown frame stays the second's.
  const gate: { release: () => void } = { release: () => {} };
  const firstGate = new Promise<void>((resolve) => {
    gate.release = resolve;
  });
  const registration = registerInstrument({
    kind: "stub",
    Component: stubComponent,
    connect: (buffer: BufferAccess<Frame>): InstrumentAttachment<Controller, BoardAnchor> => {
      const first = buffer.updates.refresh(async () => {
        await firstGate;
        return {
          kind: "ready" as const,
          value: { value: { label: "first" }, targets: [] },
        };
      });
      const second = buffer.updates.refresh(() => ({
        kind: "ready" as const,
        value: { value: { label: "second" }, targets: [] },
      }));
      void first.then((outcome) => {
        assert.equal(outcome, "superseded", "the late refresh is superseded");
      });
      void second.then((outcome) => {
        assert.equal(outcome, "applied", "the early refresh applies");
      });
      return {
        presentation: { tag: "x" },
        input: commandContext(() => []),
        places: stubPlaces(),
        release() {},
      };
    },
  });
  const instrument = constructInstrument(registration, environment());
  await new Promise((resolve) => setTimeout(resolve, 0));
  const shown = instrument.snapshot();
  if (shown.kind === "alive" && shown.content.kind === "available") {
    const frame = shown.content.value.props["content"] as Frame;
    assert.equal(frame.label, "second");
  } else {
    assert.fail("the winning refresh published an available frame");
  }
  gate.release();
  await new Promise((resolve) => setTimeout(resolve, 0));
  if (shown.kind === "alive" && shown.content.kind === "available") {
    const frame = shown.content.value.props["content"] as Frame;
    assert.equal(frame.label, "second", "the superseded answer never overwrites");
  }
  instrument.destroy();
});

test("a failed refresh retains the last usable frame and reports", async () => {
  const registration = registerInstrument({
    kind: "stub",
    Component: stubComponent,
    connect: (buffer: BufferAccess<Frame>): InstrumentAttachment<Controller, BoardAnchor> => {
      void buffer.updates.refresh(() => ({
        kind: "ready" as const,
        value: { value: { label: "first" }, targets: [] },
      }));
      return {
        presentation: { tag: "x" },
        input: commandContext(() => []),
        places: {
          ...stubPlaces(),
          async request() {
            await buffer.updates.refresh(() => ({
              kind: "failed" as const,
              issue: knownIssue({ code: "asset-missing", message: "no such board" }),
            }));
            return "applied" as const;
          },
        },
        release() {},
      };
    },
  });
  const instrument: RunningInstrument = constructInstrument(registration, environment());
  await new Promise((resolve) => setTimeout(resolve, 0));
  const before = instrument.snapshot();
  if (before.kind === "alive" && before.content.kind === "available") {
    const frame = before.content.value.props["content"] as Frame;
    assert.equal(frame.label, "first", "the usable frame stays");
  } else {
    assert.fail("the first refresh should have published");
  }
  await instrument.places.restore({
    bufferId: instrument.id,
    address: { machine: "", path: "" },
    position: { kind: "default" },
  });
  const after = instrument.snapshot();
  if (after.kind === "alive" && after.content.kind === "available") {
    const frame = after.content.value.props["content"] as Frame;
    assert.equal(frame.label, "first", "failure retains the last usable frame");
    assert.ok(after.content.lastFailure !== null, "the failure is reported on the frame");
  } else {
    assert.fail("a failed refresh must not drop a usable frame");
  }
  instrument.destroy();
});

test("destruction forwards through the registry facade and idempotence holds", () => {
  const registration = registerInstrument(stubDefinition("one", []));
  const instrument = constructInstrument(registration, environment());
  instrument.destroy();
  assert.equal(instrument.snapshot().kind, "destroyed");
  instrument.destroy();
  assert.equal(instrument.snapshot().kind, "destroyed", "destroy is idempotent");
});
