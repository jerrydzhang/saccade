// Incorporation expansion (t-58): an incorporation is collapsed until
// its act expands it; expanded, it holds exactly the components its
// select paths name in the cited revision; the act collapses it again;
// the state follows the part AND its source. Pure: the loader is a
// fake answering decoded boards.

import assert from "node:assert/strict";
import { test } from "vitest";
import { decodeBoard } from "../src/lib/board/decode.ts";
import type { Board } from "../src/lib/board/model.ts";
import { citedPart, createExpansions, selectBlock, type CitedSource } from "../src/lib/board/expansion.ts";
import { actsFor } from "../src/lib/menu/acts.ts";

function board(yaml: string): Board {
  const result = decodeBoard(yaml);
  assert.equal(result.status, "ok");
  if (result.status !== "ok") throw new Error("does not decode");
  return result.board;
}

const CITED = board(`grammar: 1
task: t-7
content:
  - group:
      title: Recorded setup
      content:
        - text: the first child
        - text: the second child
  - text: after the group
`);

test("a select path names exactly its component, through groups", () => {
  const first = selectBlock(CITED, "/content/0/group/content/0");
  assert.equal(first?.kind, "text");
  assert.equal(first?.kind === "text" ? first.markdown : null, "the first child");
  assert.equal(selectBlock(CITED, "/content/1")?.kind, "text");
  assert.equal(selectBlock(CITED, "/content/0")?.kind, "group");
  // a path naming nothing, or entering through the wrong kind
  assert.equal(selectBlock(CITED, "/content/2"), null);
  assert.equal(selectBlock(CITED, "/content/0/group/content/5"), null);
  assert.equal(selectBlock(CITED, "/content/0/incorporation/content/0"), null);
  assert.equal(selectBlock(CITED, "/content/1/group/content/0"), null);
  assert.equal(selectBlock(CITED, "content/0"), null);
});

function harness(initial: CitedSource | null) {
  let source = initial;
  const loads: string[] = [];
  let changes = 0;
  const pending: (() => void)[] = [];
  const expansions = createExpansions({
    load(machine, path) {
      loads.push(`${machine} ${path}`);
      return new Promise((resolve, reject) => {
        pending.push(() => {
          if (path === "t-404@r1") reject(new Error("no such revision"));
          else resolve({ board: CITED, location: { machine, path } });
        });
      });
    },
    machine: () => "tracker:seed",
    sourceAt: (part) => (part === "/content/1" ? source : null),
    onChange: () => {
      changes += 1;
    },
  });
  const settle = async (): Promise<void> => {
    for (const answer of pending.splice(0)) answer();
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  return {
    expansions,
    loads,
    changes: () => changes,
    settle,
    setSource: (next: CitedSource | null) => {
      source = next;
    },
  };
}

const SOURCE: CitedSource = { task: "t-7", revision: 1, select: ["/content/0/group/content/0"] };

test("collapsed by default; expanding shows only the cited components; the act collapses", async () => {
  const run = harness(SOURCE);
  assert.equal(run.expansions.isExpanded("/content/1"), false);
  assert.deepEqual(run.expansions.view(), []);
  run.expansions.toggle("/content/1");
  assert.equal(run.expansions.isExpanded("/content/1"), true);
  assert.equal(run.expansions.view()[0]?.status, "loading");
  assert.deepEqual(run.loads, ["tracker:seed t-7@r1"]);
  await run.settle();
  const [shown] = run.expansions.view();
  assert.equal(shown?.status, "loaded");
  assert.deepEqual(shown?.location, { machine: "tracker:seed", path: "t-7@r1" });
  // exactly the one cited component, never its group or its sibling
  assert.equal(shown?.components.length, 1);
  assert.equal(shown?.components[0]?.part, citedPart("/content/1", 0));
  assert.equal(shown?.components[0]?.part, "/content/1/cited/0");
  const block = shown?.components[0]?.block;
  assert.equal(block?.kind === "text" ? block.markdown : null, "the first child");
  run.expansions.toggle("/content/1");
  assert.equal(run.expansions.isExpanded("/content/1"), false);
  assert.deepEqual(run.expansions.view(), []);
  assert.ok(run.changes() >= 3);
});

test("a part that is no incorporation never expands", () => {
  const run = harness(SOURCE);
  run.expansions.toggle("/content/0");
  assert.deepEqual(run.expansions.view(), []);
  assert.deepEqual(run.loads, []);
});

test("an answer arriving after the collapse is dropped", async () => {
  const run = harness(SOURCE);
  run.expansions.toggle("/content/1");
  run.expansions.toggle("/content/1");
  await run.settle();
  assert.deepEqual(run.expansions.view(), []);
});

test("an unreadable revision or a select naming nothing says so", async () => {
  const run = harness({ task: "t-404", revision: 1, select: ["/content/0"] });
  run.expansions.toggle("/content/1");
  await run.settle();
  assert.equal(run.expansions.view()[0]?.status, "failed");
  assert.match(run.expansions.view()[0]?.message ?? "", /no such revision/);
  const missing = harness({ task: "t-7", revision: 1, select: ["/content/9", "/content/1"] });
  missing.expansions.toggle("/content/1");
  await missing.settle();
  const components = missing.expansions.view()[0]?.components ?? [];
  assert.deepEqual(components.map((component) => component.block?.kind ?? null), [null, "text"]);
});

test("the expansion follows the part and its source: another citation there shows collapsed", async () => {
  const run = harness(SOURCE);
  run.expansions.toggle("/content/1");
  await run.settle();
  run.setSource({ ...SOURCE, revision: 2 });
  assert.equal(run.expansions.isExpanded("/content/1"), false);
  assert.deepEqual(run.expansions.view(), []);
  // the same citation back (a return, a refresh): expanded again,
  // without a second read
  run.setSource(SOURCE);
  assert.equal(run.expansions.view()[0]?.status, "loaded");
  assert.equal(run.loads.length, 1);
});

test("an incorporation's acts: z expands or collapses first (a plain click), y copies its citation", () => {
  const expanded = new Set<string>();
  const toggled: string[] = [];
  const copied: string[] = [];
  const deps = {
    isFolded: () => false,
    toggleFold: () => {},
    isExpanded: (part: string) => expanded.has(part),
    toggleExpansion: (part: string) => {
      toggled.push(part);
      expanded.add(part);
    },
    referenceOf: () => null,
    referencePorts: () => ({ copy: (payload: string) => void copied.push(payload), open: () => {} }),
    referenceMachine: () => null,
    partCitation: (part: string) => `t-9@r4:${part}`,
  };
  const target = { kind: "incorporation", label: "incorporates t-7@r1", part: "/content/1" };
  const acts = actsFor(target, deps);
  assert.deepEqual(acts.map((act) => [act.key, act.action.label]), [["z", "expand"], ["y", "copy citation"]]);
  acts[0]?.action.run();
  assert.deepEqual(toggled, ["/content/1"]);
  assert.equal(actsFor(target, deps)[0]?.action.label, "collapse");
  actsFor(target, deps)[1]?.action.run();
  assert.deepEqual(copied, ["t-9@r4:/content/1"]);
  // without the expansion seam the incorporation still copies
  const bare = actsFor(target, { ...deps, isExpanded: undefined, toggleExpansion: undefined });
  assert.deepEqual(bare.map((act) => act.key), ["y"]);
});
