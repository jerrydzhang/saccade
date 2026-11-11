import assert from "node:assert/strict";
import test from "node:test";
import { createFindInteraction } from "../src/lib/find/interaction.ts";
import type { FindUnit } from "../src/lib/find/session.ts";
import { createHostMatcher } from "../src/lib/input/host.ts";
import { describeCommands } from "../src/lib/contracts/input.ts";
import type { InputContext, KeyInput } from "../src/lib/contracts/input.ts";
const stroke = (key: string, shift = false): KeyInput => ({ key, shift, ctrl: false, alt: false, meta: false });

test("Find captures a declared text-entry context; Enter/Shift-Enter/Escape use the shared matcher", () => {
  const events: string[] = [];
  let captured!: InputContext;
  const find = createFindInteraction({
    units: () => [
      { id: "/content/0", text: "needle", source: "needle" },
      { id: "/content/1", text: "needle", source: "needle" },
    ],
    capture(context) { captured = context; events.push("capture"); return { release: () => { events.push("release"); } }; },
    reveal: (occurrence) => { events.push(occurrence.unit.id); },
  });
  find.enter(); find.enter();
  assert.deepEqual(events, ["capture"], "re-entry does not acquire a second claim");
  assert.equal(captured.kind, "text-entry");
  assert.deepEqual(describeCommands(captured).map((command) => [command.actionId, command.keyLabels]), [
    ["find.next", ["Enter"]], ["find.previous", ["Shift-Enter"]], ["find.close", ["Escape"]],
  ]);
  const host = createHostMatcher(captured);
  try {
    assert.equal(host.apply(host.stroke(stroke("n"))), "allow-default", "ordinary typing stays native");
    find.type("needle");
    assert.equal(find.view().count, 2);
    assert.equal(host.apply(host.stroke(stroke("Enter"))), "consume");
    assert.equal(host.apply(host.stroke(stroke("Enter", true))), "consume");
    assert.deepEqual(events, ["capture", "/content/1", "/content/0"]);
    host.apply(host.stroke(stroke("Escape")));
    assert.equal(find.view().open, false);
    assert.deepEqual(events, ["capture", "/content/1", "/content/0", "release"]);
    find.close();
    assert.equal(events.filter((event) => event === "release").length, 1);
  } finally { host.dispose(); }
});

test("live units remain current on Find exit; explicit navigation releases without restoring old content", () => {
  let units: readonly FindUnit[] = [{ id: "old", text: "needle", source: "needle" }];
  let releases = 0;
  const find = createFindInteraction({
    units: () => units,
    capture: () => ({ release: () => { releases += 1; } }),
    reveal() {},
  });
  const counts: number[] = [];
  const off = find.subscribe((view) => { counts.push(view.count); });
  find.enter(); find.type("needle");
  units = [{ id: "new", text: "needle needle", source: "needle needle" }];
  find.contentChanged();
  assert.equal(counts.at(-1), 2);
  find.navigateAway();
  assert.equal(releases, 1);
  assert.equal(units[0]?.id, "new");
  assert.equal(find.view().open, false);
  find.enter();
  assert.equal(find.view().query, "needle", "re-entry retains query, not an old revision");
  assert.equal(find.view().count, 2);
  find.close(); off();
});
