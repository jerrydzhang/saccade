import test from "node:test";
import assert from "node:assert/strict";
import { helpKeys } from "../src/lib/commands.ts";

test("each window kind has its own key table (round 16's showHelp)", () => {
  const viewer = helpKeys({ kind: "viewer", history: false, deck: false });
  assert.ok(viewer.some(([key, label]) => key === "Ctrl-f Ctrl-b" && label === "page"), "Ctrl-f pages");
  assert.ok(!viewer.some(([, label]) => label.includes("search")), "find is a space command, never Ctrl-f");
  assert.equal(helpKeys({ kind: "viewer", history: true, deck: false }).find(([key]) => key === "h l")?.[1], "step time");
  const agents = helpKeys({ kind: "agents", history: false, deck: false });
  assert.ok(agents.some(([key]) => key === "a w b"), "agents name their copies");
  const tasks = helpKeys({ kind: "tasks", history: false, deck: false });
  assert.ok(!tasks.some(([key]) => key === "a w b"));
  for (const table of [viewer, agents, tasks]) {
    for (const [key, label] of table) assert.ok(/^[\x20-\x7e]+$/.test(key + label), `ASCII: ${key}`);
  }
});
