import assert from "node:assert/strict";
import { test, onTestFinished, vi } from "vitest";
import { createViewport, SliceBufferHost } from "../src/components/Buffer.ts";
import type { InstrumentAttachment } from "../src/components/Buffer.ts";
import { connectBoard } from "../src/instruments/boardController.ts";
import type { BoardData, BoardUI } from "../src/instruments/boardController.ts";
import type { BoardAnchor } from "../src/lib/contracts/places.ts";
import { sourceFor } from "../src/lib/sources.ts";
import { TrackerSource } from "../src/lib/tracker.ts";

test("retained-frame hydration and folding cannot supersede a pending board open", async (t) => {
  const machine = "tracker:publication-test";
  const source = sourceFor(machine) as TrackerSource;
  let finish!: () => void;
  const gate = new Promise<void>((resolve) => { finish = resolve; });
  vi.spyOn(source, "loadText").mockImplementation(async (location: { path: string }) => {
    if (location.path.startsWith("t-2")) await gate;
    if (location.path.startsWith("t-3")) throw new Error("C could not load");
    return {
      text: `grammar: 1\ntask: ${location.path.split("@")[0]}\ncontent:\n  - group:\n      title: the group\n      content:\n        - text: child\n        - figure:\n            input: figure.png\n            caption: retained media\n`,
      location: { machine, path: location.path },
    };
  });
  let finishAsset!: () => void;
  const bytes = new Uint8Array([1, 2, 3]);
  vi.spyOn(source, "loadAsset").mockImplementation(async () => {
    await new Promise<void>((resolve) => { finishAsset = resolve; });
    return { bytes } as never;
  });
  let attachment!: InstrumentAttachment<BoardUI, BoardAnchor>;
  const controller = new SliceBufferHost<BoardData, BoardUI, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" }, viewport: createViewport(),
    diagnostics: { report: () => "diagnostic", unexpected: () => "diagnostic" },
    connect(buffer) { attachment = connectBoard(buffer); return attachment; },
  });
  const built = controller.wire();
  onTestFinished(() => built.host.destroy());
  await attachment.places.request({ machine, path: "t-1@r1" });
  const a = built.host.snapshot();
  assert.ok(a.kind === "alive" && a.content.kind === "available");
  const original = a.content.value.value;
  const group = original.board?.content[0];
  assert.ok(group?.kind === "group");
  const figure = group.content.find((block) => block.kind === "figure");
  assert.ok(figure?.kind === "figure");
  assert.ok(original.resolved);
  const opening = attachment.places.request({ machine, path: "t-2@r1" });
  const pending = built.host.snapshot();
  assert.ok(pending.kind === "alive" && pending.content.kind === "pending");
  assert.equal(pending.content.previous?.value.generation, original.generation,
    "retained A keeps its token, never borrows the pending B token");
  assert.equal(attachment.presentation.displayedGeneration(), original.generation,
    "the hydration host names the displayed A, not requested B");
  const retainedAsset = attachment.presentation.loadFigureAsset(original.resolved, figure);
  finishAsset();
  assert.deepEqual(await retainedAsset, bytes, "A's bytes remain usable while A is displayed");
  const lateAsset = attachment.presentation.loadFigureAsset(original.resolved, figure);
  const finishLateAsset = finishAsset;
  // A guessed newer-request token is not proof of displayed identity.
  // Correct late A hydration is valid while A remains displayed.
  const stale = [{ kind: "node" as const, status: "error" as const, message: "A's late hydration" }];
  attachment.presentation.reportRenderOutcomes(original.generation + 1, stale);
  attachment.presentation.reportRenderOutcomes(original.generation, stale);
  attachment.presentation.toggleFold("/content/0");
  const folded = built.host.snapshot();
  assert.ok(folded.kind === "alive" && folded.content.kind === "pending");
  assert.deepEqual(folded.content.previous?.value.folded, ["/content/0"]);
  assert.deepEqual(folded.content.previous?.value.renderOutcomes, stale);
  finish();
  assert.equal(await opening, "applied");
  const b = built.host.snapshot();
  assert.ok(b.kind === "alive" && b.content.kind === "available");
  assert.match(b.content.value.value.location, /t-2@r1$/);
  finishLateAsset();
  assert.equal(await lateAsset, "stale", "A's bytes are stale only after B actually replaces A");
  assert.equal(b.content.value.value.generation, attachment.presentation.displayedGeneration());
  attachment.presentation.reportRenderOutcomes(b.content.value.value.generation, stale);
  const repeated = built.host.snapshot();
  assert.ok(repeated.kind === "alive" && repeated.content.kind === "available");
  assert.deepEqual(repeated.content.value.value.renderOutcomes, stale,
    "equal outcome payloads in different generations still publish independently");
  const current = [{ kind: "node" as const, status: "ok" as const, message: "B's hydration" }];
  attachment.presentation.reportRenderOutcomes(b.content.value.value.generation, current);
  attachment.presentation.reportRenderOutcomes(original.generation, stale);
  const after = built.host.snapshot();
  assert.ok(after.kind === "alive" && after.content.kind === "available");
  assert.deepEqual(after.content.value.value.renderOutcomes, current);
  assert.equal(await attachment.places.request({ machine, path: "t-3@r1" }), "failed");
  assert.equal(attachment.presentation.displayedGeneration(), b.content.value.value.generation,
    "failed C leaves B as the valid hydration identity");
  const retained = [{ kind: "node" as const, status: "ok" as const, message: "B rehydrated after failed C" }];
  attachment.presentation.reportRenderOutcomes(b.content.value.value.generation, retained);
  const failed = built.host.snapshot();
  assert.ok(failed.kind === "alive" && failed.content.kind === "available");
  assert.deepEqual(failed.content.value.value.renderOutcomes, retained);
  assert.equal(failed.content.lastFailure, "diagnostic");
  assert.ok(failed.content.value.value.resolved);
  const recoveredAsset = attachment.presentation.loadFigureAsset(failed.content.value.value.resolved, figure);
  finishAsset();
  assert.deepEqual(await recoveredAsset, bytes, "retained B can hydrate after failed C");
});
