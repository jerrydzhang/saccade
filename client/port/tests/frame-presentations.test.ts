// The frame's presentation of published targets (t-73, the frame's
// half), tested on controlled ports the way the retired window-store
// tests worked: memory fakes, no DOM, no transport. In-view filtering
// through the geometry, nested labels per `inside`, the menu built
// from the acts list in order, the click paths' mapping (plain =
// first act, modified = the act its key names, with the fallback), the
// reading anchor Back rides, and the stale lease that acts nothing.

import assert from "node:assert/strict";
import { test } from "vitest";

import { createTargetPresentations, type TargetPresentationPorts } from "../src/lib/frame/presentations.ts";
import {
  commandsOf,
  formatSequence,
  key,
  sequence,
  type Action,
  type CommandBinding,
  type InputContext,
  type KeySequence,
} from "../src/lib/contracts/input.ts";
import { knownIssue, type Bounds, type KnownIssue, type TargetDefinition, type Viewport } from "../src/lib/contracts/buffer.ts";

const VIEW = { width: 300, height: 100 };

/** A viewport whose reading position the test moves; the scroll calls
 * are recorded so the reading context's keys can be seen driving it. */
function fakeViewport(top = 0): Viewport & { scrolled: string[] } {
  const scrolled: string[] = [];
  return {
    scrolled,
    metrics: () => ({ position: { left: 0, top }, width: VIEW.width, height: VIEW.height }),
    scrollLines(amount: number) {
      scrolled.push(`lines:${amount}`);
    },
    scrollPage(direction: -1 | 1, extent: "half" | "full") {
      scrolled.push(`page:${direction}:${extent}`);
    },
    scrollFraction(direction: -1 | 1, fraction: number) {
      scrolled.push(`fraction:${direction}:${fraction}`);
    },
    start() {
      scrolled.push("start");
    },
    end() {
      scrolled.push("end");
    },
    reveal() {},
    moveTo() {},
  };
}

const box = (top: number, bottom: number): Bounds => ({ left: 0, top, right: 100, bottom });

/** The acts record their runs; the definition helpers keep the shape
 * the instruments will publish. */
function harness(defs: readonly TargetDefinition[], geometry: ReadonlyMap<string, Bounds>, viewport = fakeViewport()) {
  const runs: string[] = [];
  const reported: KnownIssue[] = [];
  const captured: InputContext[] = [];
  let released = 0;
  const ports: TargetPresentationPorts = {
    targets: () => defs,
    bounds: (id) => geometry.get(id) ?? null,
    capture: (context) => {
      captured.push(context);
      return {
        release: () => {
          released += 1;
        },
      };
    },
    report: (issue) => reported.push(issue),
    viewport,
  };
  return { presentations: createTargetPresentations(ports), reported, captured, releasedOf: () => released, viewport };
}

/** An act whose run records itself, so a test sees exactly which acts
 * the frame's paths ran. */
function recorder(runs: string[], id: string): Action {
  return { id, label: id, run: () => { runs.push(id); } };
}

const plain = (over: Partial<TargetDefinition> & { id: string }): TargetDefinition => ({
  kind: "task",
  label: over.id,
  inside: null,
  anchor: null,
  acts: () => [],
  ...over,
});

test("select labels go to the targets in view; out-of-view and unmeasurable targets get none", () => {
  const defs = [
    plain({ id: "a", acts: () => [] }),
    plain({ id: "b", acts: () => [] }),
    plain({ id: "c", acts: () => [] }),
    plain({ id: "d", acts: () => [] }),
  ];
  const geometry = new Map([
    ["a", box(10, 40)],
    ["b", box(50, 80)],
    ["c", box(120, 160)],
  ]);
  const { presentations } = harness(defs, geometry);
  const select = commandsOf(presentations.readingContext()).find((entry) => entry.action.id === "frame.select");
  assert.ok(select, "the reading context binds the select entry");
  select!.action.run();
  const view = presentations.selectView();
  assert.ok(view, "s enters select over visible targets");
  // in view: a and b; c is below the viewport, d has no geometry
  assert.deepEqual(
    view!.labels.map((chip) => [chip.part, chip.nested]),
    [
      ["a", false],
      ["b", false],
    ],
  );
});

test("a label is nested when its target sits inside another labelled one", () => {
  const defs = [
    plain({ id: "a", acts: () => [] }),
    plain({ id: "b", inside: "a", acts: () => [] }),
    plain({ id: "e", inside: "c", acts: () => [] }),
    plain({ id: "c", acts: () => [] }),
  ];
  const geometry = new Map([
    ["a", box(10, 40)],
    ["b", box(20, 30)],
    ["c", box(150, 180)],
    ["e", box(50, 60)],
  ]);
  const { presentations } = harness(defs, geometry);
  const select = commandsOf(presentations.readingContext()).find((entry) => entry.action.id === "frame.select")!;
  select.action.run();
  const view = presentations.selectView()!;
  // b sits in labelled a: outlined (nested). e's inside (c) is out of
  // view and unlabelled: a solid chip, the same look as the top level
  assert.deepEqual(
    view.labels.map((chip) => [chip.part, chip.nested]),
    [
      ["a", false],
      ["b", true],
      ["e", false],
    ],
  );
});

test("the act menu is the acts list in order, headed by the published label", () => {
  const defs = [
    plain({
      id: "row",
      label: "t-9 the row",
      acts: () => [],
    }),
    plain({
      id: "cite",
      label: "repo/t-3@r1",
      acts: () => [
        { key: "f", action: { id: "cite.follow", label: "follow here", run: () => {} } },
        { key: "o", action: { id: "cite.open", label: "open at the link", run: () => {} } },
        { key: "c", action: { id: "cite.copy", label: "copy", run: () => {} } },
      ],
    }),
  ];
  const geometry = new Map([["cite", box(0, 20)]]);
  const { presentations } = harness(defs, geometry);
  presentations.presentTarget("cite", () => box(0, 20)).showActions({ left: 4, top: 4 });
  const view = presentations.menuView();
  assert.ok(view, "the menu opens for a target with acts");
  assert.equal(view!.label, "repo/t-3@r1");
  assert.deepEqual(
    view!.acts.map((act) => [act.key, act.label]),
    [
      ["f", "follow here"],
      ["o", "open at the link"],
      ["c", "copy"],
    ],
  );
});

test("a plain click runs the first act; a modified click runs the act its key names", () => {
  const runs: string[] = [];
  const defs = [
    plain({
      id: "cite",
      acts: () => [
        { key: "f", action: recorder(runs, "cite.follow") },
        { key: "o", action: recorder(runs, "cite.open") },
      ],
    }),
  ];
  const geometry = new Map([["cite", box(0, 20)]]);
  const { presentations } = harness(defs, geometry);
  const lease = presentations.presentTarget("cite", () => box(0, 20));
  assert.equal(lease.activatePointer("plain"), true);
  assert.deepEqual(runs, ["cite.follow"], "plain click means the first act");
  assert.equal(lease.activatePointer("modified"), true);
  assert.deepEqual(runs, ["cite.follow", "cite.open"], "shift/middle is the o act (t-47)");
});

test("a modified click on a kind with no o act acts nothing", () => {
  const runs: string[] = [];
  const defs = [
    plain({
      id: "row",
      acts: () => [{ key: "o", action: recorder(runs, "row.open") }],
    }),
    plain({
      id: "agent",
      acts: () => [{ key: "f", action: recorder(runs, "agent.follow") }],
    }),
  ];
  const geometry = new Map([
    ["row", box(0, 20)],
    ["agent", box(30, 50)],
  ]);
  const { presentations } = harness(defs, geometry);
  assert.equal(presentations.presentTarget("row", () => box(0, 20)).activatePointer("plain"), true);
  assert.equal(presentations.presentTarget("agent", () => box(30, 50)).activatePointer("modified"), false);
  assert.deepEqual(runs, ["row.open"], "a key the kind does not declare acts nothing");
});

test("a lease whose element has no acts clicks nothing and opens no menu", () => {
  const runs: string[] = [];
  const defs = [plain({ id: "quiet", acts: () => [] })];
  const geometry = new Map([["quiet", box(0, 20)]]);
  const { presentations } = harness(defs, geometry);
  const lease = presentations.presentTarget("quiet", () => box(0, 20));
  assert.equal(lease.activatePointer("plain"), false);
  assert.equal(lease.activatePrimary(), undefined);
  lease.showActions({ left: 0, top: 0 });
  assert.equal(presentations.menuView(), null, "no acts, no menu");
  assert.deepEqual(runs, []);
});

test("a lease for an id the frame does not publish reports and acts nothing", () => {
  const runs: string[] = [];
  const { presentations, reported } = harness([], new Map());
  const lease = presentations.presentTarget("ghost", () => box(0, 20));
  assert.equal(lease.activatePointer("plain"), false);
  assert.deepEqual(runs, []);
  assert.equal(reported.length, 1);
  assert.equal(reported[0]!.code, "target-unavailable");
});

test("a lease whose element has left the surface acts nothing", () => {
  const runs: string[] = [];
  const defs = [
    plain({
      id: "gone",
      acts: () => [{ key: "o", action: recorder(runs, "gone.open") }],
    }),
  ];
  const { presentations } = harness(defs, new Map([["gone", box(0, 20)]]));
  const lease = presentations.presentTarget("gone", () => null);
  assert.equal(lease.activatePointer("plain"), false);
  lease.activatePrimary();
  lease.showActions({ left: 0, top: 0 });
  assert.equal(presentations.menuView(), null);
  assert.deepEqual(runs, []);
});

test("a released lease acts nothing", () => {
  const runs: string[] = [];
  const defs = [
    plain({
      id: "row",
      acts: () => [{ key: "o", action: recorder(runs, "row.open") }],
    }),
  ];
  const geometry = new Map([["row", box(0, 20)]]);
  const { presentations } = harness(defs, geometry);
  const lease = presentations.presentTarget("row", () => box(0, 20));
  lease.release();
  assert.equal(lease.activatePointer("plain"), false);
  assert.deepEqual(runs, []);
});

test("Enter on the open menu runs the first act - the primary", () => {
  const runs: string[] = [];
  const defs = [
    plain({
      id: "cite",
      acts: () => [
        { key: "f", action: recorder(runs, "cite.follow") },
        { key: "o", action: recorder(runs, "cite.open") },
      ],
    }),
  ];
  const geometry = new Map([["cite", box(0, 20)]]);
  const { presentations } = harness(defs, geometry);
  presentations.presentTarget("cite", () => box(0, 20)).showActions({ left: 4, top: 4 });
  const outcome = presentations.menuSession().key("Enter");
  assert.equal(outcome.kind, "act");
  assert.ok(outcome.kind === "act" && outcome.action.id === "cite.follow");
  assert.deepEqual(runs, ["cite.follow"]);
});

test("the reading anchor is the anchored target the reading position sits inside", () => {
  const defs = [
    plain({ id: "a", anchor: null, acts: () => [] }),
    plain({ id: "b", anchor: "t-9", acts: () => [] }),
    plain({ id: "c", anchor: "t-3", acts: () => [] }),
  ];
  const geometry = new Map([
    ["a", box(10, 40)],
    ["b", box(50, 80)],
    ["c", box(80, 90)],
  ]);
  const { presentations } = harness(defs, geometry, fakeViewport(55));
  // the reading top holds b: its durable anchor and bounds ride Back
  assert.deepEqual(presentations.readingAnchor(), { kind: "anchor", anchor: "t-9", bounds: box(50, 80) });
});

test("targets without an anchor cannot carry a return place; none in view is the default", () => {
  const defs = [
    plain({ id: "a", anchor: null, acts: () => [] }),
    plain({ id: "far", anchor: "t-2", acts: () => [] }),
  ];
  const geometry = new Map([
    ["a", box(10, 40)],
    ["far", box(500, 540)],
  ]);
  const { presentations } = harness(defs, geometry, fakeViewport(20));
  assert.deepEqual(presentations.readingAnchor(), { kind: "default" }, "an unanchored target under the reading top is skipped");
  const { presentations: empty } = harness(defs, geometry, fakeViewport(300));
  assert.deepEqual(empty.readingAnchor(), { kind: "default" });
});

test("the anchor's place locates through the published targets", () => {
  const defs = [plain({ id: "b", anchor: "t-9", acts: () => [] })];
  const geometry = new Map([["b", box(50, 80)]]);
  const { presentations } = harness(defs, geometry);
  assert.deepEqual(presentations.locateAnchor("t-9"), box(50, 80));
  assert.equal(presentations.locateAnchor("t-404"), null);
});

test("the reading context's keys drive the viewport; s captures input for the labels", () => {
  const defs = [plain({ id: "a", acts: () => [] })];
  const geometry = new Map([["a", box(10, 40)]]);
  const { presentations, viewport, captured, releasedOf } = harness(defs, geometry);
  const context = presentations.readingContext();
  const runOf = (id: string): (() => void) => {
    const binding = commandsOf(context).find((entry) => entry.action.id === id);
    assert.ok(binding, `the reading context binds ${id}`);
    return () => binding!.action.run();
  };
  runOf("frame.scroll-down")();
  runOf("frame.scroll-up")();
  runOf("frame.page-half-down")();
  runOf("frame.page-mostly-up")();
  runOf("frame.scroll-top")();
  runOf("frame.scroll-bottom")();
  assert.deepEqual(viewport.scrolled, ["lines:1", "lines:-1", "fraction:1:0.5", "fraction:-1:0.9", "start", "end"]);
  runOf("frame.select")();
  assert.ok(presentations.selectView(), "s enters select");
  assert.equal(captured.length, 1, "the select session holds the input while active");
  // exit hands the input back
  const exit = commandsOf(captured[0]!).find((entry) => entry.action.id === "select.exit");
  exit!.action.run();
  assert.equal(releasedOf(), 1);
  assert.equal(presentations.selectView(), null);
});

test("the reading context's table is the shared keys and nothing else", () => {
  const { presentations } = harness([], new Map());
  const table = new Map(
    commandsOf(presentations.readingContext()).map((entry) => [
      entry.action.id,
      entry.keys.map((keys: KeySequence) => formatSequence(keys)),
    ]),
  );
  assert.deepEqual(
    [...table.entries()].sort(([a], [b]) => (a < b ? -1 : 1)),
    [
      ["frame.page-ctrl-half-down", ["Ctrl-d"]],
      ["frame.page-ctrl-half-up", ["Ctrl-u"]],
      ["frame.page-half-down", ["d"]],
      ["frame.page-half-up", ["u"]],
      ["frame.page-mostly-down", ["Ctrl-f"]],
      ["frame.page-mostly-up", ["Ctrl-b"]],
      ["frame.scroll-bottom", ["G"]],
      ["frame.scroll-down", ["j"]],
      ["frame.scroll-top", ["g g"]],
      ["frame.scroll-up", ["k"]],
      ["frame.select", ["s"]],
    ],
  );
});

test("an instrument's own key extends the reading context; a frame key cannot be rebound", () => {
  const { presentations } = harness([], new Map());
  const addProject: CommandBinding = {
    action: { id: "tasks.add-project", label: "add a project", run: () => {} },
    keys: [sequence(key("a"))],
  };
  const extended = presentations.readingContext([addProject]);
  const ids = commandsOf(extended).map((entry) => entry.action.id);
  assert.ok(ids.includes("tasks.add-project"), "the instrument's key coexists");
  assert.ok(ids.includes("frame.select"), "the frame's keys stay");
  assert.throws(
    () => presentations.readingContext([{ action: { id: "tasks.other", label: "other", run: () => {} }, keys: [sequence(key("j"))] }]),
    "a binding that names a frame key is a collision, never a silent rebinding",
  );
});
