import assert from "node:assert/strict";
import test from "node:test";
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
  t.mock.method(source, "loadText", async (location: { path: string }) => {
    if (location.path.startsWith("t-2")) await gate;
    return {
      text: `grammar: 1\ntask: ${location.path.split("@")[0]}\ncontent:\n  - group:\n      title: the group\n      content:\n        - text: child\n`,
      location: { machine, path: location.path },
    };
  });
  let attachment!: InstrumentAttachment<BoardUI, BoardAnchor>;
  const controller = new SliceBufferHost<BoardData, BoardUI, BoardAnchor>({
    context: { windowId: "w", bufferId: "b" }, viewport: createViewport(),
    diagnostics: { report: () => "diagnostic", unexpected: () => "diagnostic" },
    connect(buffer) { attachment = connectBoard(buffer); return attachment; },
  });
  const built = controller.wire();
  t.after(() => built.host.destroy());
  await attachment.places.request({ machine, path: "t-1@r1" });
  const a = built.host.snapshot();
  assert.ok(a.kind === "alive" && a.content.kind === "available");
  const original = a.content.value.value;
  const opening = attachment.places.request({ machine, path: "t-2@r1" });
  const pending = built.host.snapshot();
  assert.ok(pending.kind === "alive" && pending.content.kind === "pending");
  assert.equal(pending.content.previous?.value.generation, original.generation,
    "retained A keeps its token, never borrows the pending B token");
  assert.notEqual(attachment.presentation.currentGeneration(), original.generation);
  // Reproduce the old component's bad attribution as well as correct
  // late A hydration. Neither may cancel B or attach A's outcomes to B.
  const stale = [{ kind: "node" as const, status: "error" as const, message: "A's late hydration" }];
  attachment.presentation.reportRenderOutcomes(attachment.presentation.currentGeneration(), stale);
  attachment.presentation.reportRenderOutcomes(original.generation, stale);
  attachment.presentation.toggleFold("/content/0");
  const folded = built.host.snapshot();
  assert.ok(folded.kind === "alive" && folded.content.kind === "pending");
  assert.deepEqual(folded.content.previous?.value.folded, ["/content/0"]);
  assert.deepEqual(folded.content.previous?.value.renderOutcomes, []);
  finish();
  assert.equal(await opening, "applied");
  const b = built.host.snapshot();
  assert.ok(b.kind === "alive" && b.content.kind === "available");
  assert.match(b.content.value.value.location, /t-2@r1$/);
  assert.equal(b.content.value.value.generation, attachment.presentation.currentGeneration());
  const current = [{ kind: "node" as const, status: "ok" as const, message: "B's hydration" }];
  attachment.presentation.reportRenderOutcomes(b.content.value.value.generation, current);
  attachment.presentation.reportRenderOutcomes(original.generation, stale);
  const after = built.host.snapshot();
  assert.ok(after.kind === "alive" && after.content.kind === "available");
  assert.deepEqual(after.content.value.value.renderOutcomes, current);
});
