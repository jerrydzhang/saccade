// The Tasks instrument's controller, unit-tested through a real
// SliceBufferHost with fake measurement elements (the geometry.test.ts
// pattern): publication through the refresh pipeline, the reading
// context (j/k pure scrolling), label selection over visible rows,
// the row act menu (f/o/i/w - one action object per invocation path),
// and the return places (the departed row's identity plus viewport
// offset).

import assert from "node:assert/strict";
import { test, onTestFinished, vi } from "vitest";
import { SliceBufferHost } from "../src/components/Buffer.ts";
import type { InstrumentAttachment } from "../src/components/Buffer.ts";
import { connectTasks } from "../src/instruments/tasksController.ts";
import type { TasksData, TasksUI } from "../src/instruments/tasksController.ts";
import type { TaskAnchor } from "../src/lib/contracts/places.ts";
import type { ViewPage } from "../src/lib/pages.ts";
import type { ElsewhereAct } from "../src/lib/elsewhere.ts";
import { TrackerSource } from "../src/lib/tracker.ts";
import { sourceFor } from "../src/lib/sources.ts";
import { projectPart } from "../src/lib/tasks/rail.ts";

const MACHINE = "tracker:tasks-test";
// the control plane showing the project (t-54): the plane address's
// path names the project it shows
const PLANE = { machine: "plane", path: "tasks-test" };

function tasksPageFixture(rows: { task: string; title: string }[]): ViewPage {
  return {
    view: { major: 1 },
    kind: "document",
    location: "tasks",
    seq: null,
    content: [
      {
        kind: "group",
        title: "waiting",
        content: [
          {
            kind: "table",
            columns: ["task", "title", "agents", "latest commit", "latest revision"],
            rows: rows.map((row) => ({
              cells: [row.task, row.title, "", "cafe", "r1"],
              location: row.task,
              at: 1794467029,
            })),
          },
        ],
      },
    ],
  };
}

// The fake measurement root: the scroller the viewport attaches to,
// with a part element per row at its recorded bounds (content
// coordinates; scrollTop contributes to the metrics position).
// the classes are module-level: the HTMLElement shim binds the FIRST
// call's class, so every call must share it
class FakePart {
  private readonly bounds: { left: number; top: number; right: number; bottom: number };
  constructor(bounds: { left: number; top: number; right: number; bottom: number }) {
    this.bounds = bounds;
  }
  getBoundingClientRect() {
    return this.bounds;
  }
}
class FakeScroller {
  scrollTop: number;
  scrollLeft = 0;
  clientTop = 0;
  clientLeft = 0;
  clientWidth = 400;
  clientHeight = 200;
  private readonly parts: Map<string, { top: number; bottom: number }>;
  constructor(parts: Map<string, { top: number; bottom: number }>, scrollTop: number) {
    this.parts = parts;
    this.scrollTop = scrollTop;
  }
  getBoundingClientRect() {
    return { left: 0, top: 0 };
  }
  addEventListener() {}
  removeEventListener() {}
  querySelector(selector: string): FakePart | null {
    const match = /data-port-part="([^"]+)"/.exec(selector);
    const id = match?.[1];
    const bounds = id === undefined ? undefined : this.parts.get(id);
    if (bounds === undefined) return null;
    // the DOM box is viewport-relative: it moves up as the list
    // scrolls down, and the geometry service translates back into
    // content coordinates by adding the scroll position
    return new FakePart({
      left: 0,
      top: bounds.top - this.scrollTop,
      right: 400,
      bottom: bounds.bottom - this.scrollTop,
    });
  }
}

if (typeof globalThis.HTMLElement === "undefined") {
  // the geometry service's instanceof gate: node has no HTMLElement,
  // so the shim must be the part class itself
  Object.defineProperty(globalThis, "HTMLElement", { value: FakePart, configurable: true });
}

function fakeScroller(parts: Map<string, { top: number; bottom: number }>, scrollTop = 0) {
  const scroller = new FakeScroller(parts, scrollTop);
  return { scroller: scroller as unknown as HTMLElement };
}

interface Harness {
  built: ReturnType<SliceBufferHost<TasksData, TasksUI, TaskAnchor>["wire"]>;
  attachment: InstrumentAttachment<TasksUI, TaskAnchor>;
  opened: string[];
  elsewhere: [string, ElsewhereAct][];
  copied: string[];
  press(key: string): string;
  selectView(): unknown;
  menuView(): unknown;
}

function harness(t: import("vitest").TestContext, rows: { task: string; title: string }[], bounds: Map<string, { top: number; bottom: number }>, scrollTop = 0): Harness {
  const machine = MACHINE;
  const source = sourceFor(machine) as TrackerSource;
  const page = tasksPageFixture(rows);
  vi.spyOn(source, "instrumentPage").mockImplementation(async () => page);
  const { scroller } = fakeScroller(bounds, scrollTop);
  const viewport = {
    metrics: () => ({
      position: { left: scroller.scrollLeft, top: scroller.scrollTop },
      width: scroller.clientWidth,
      height: scroller.clientHeight,
    }),
    scrollLines: () => {},
    scrollPage: () => {},
    start: () => {
      scroller.scrollTop = 0;
    },
    end: () => {
      scroller.scrollTop = 1000;
    },
    reveal: () => {},
    moveTo: (next: { left: number; top: number }) => {
      scroller.scrollTop = next.top;
    },
    attachElement: (element: HTMLElement | null) => {},
    attachedElement: () => scroller,
  };
  const opened: string[] = [];
  const elsewhere: [string, ElsewhereAct][] = [];
  let attachment!: InstrumentAttachment<TasksUI, TaskAnchor>;
  const host = new SliceBufferHost<TasksData, TasksUI, TaskAnchor>({
    context: { windowId: "w", bufferId: "tasks-1" },
    viewport: viewport as unknown as import("../src/lib/contracts/buffer.ts").Viewport,
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) {
      attachment = connectTasks(buffer);
      return attachment;
    },
  });
  const built = host.wire();
  built.host.activate();
  onTestFinished(() => built.host.destroy());
  const copied: string[] = [];
  attachment.presentation.setTaskPorts({
    open: (spec, part) => opened.push(part == null ? spec : `${spec} ${part}`),
    elsewhere: (spec, act) => elsewhere.push([spec, act]),
    copy: (text) => copied.push(text),
  });
  const press = (key: string): string =>
    built.host.handleKey({ key, ctrl: false, shift: false, alt: false, meta: false });
  return {
    built,
    attachment,
    opened,
    elsewhere,
    copied,
    press,
    selectView: () => {
      let view: unknown = null;
      attachment.presentation.subscribeSelect((v) => {
        view = v;
      })();
      return view;
    },
    menuView: () => {
      let view: unknown = null;
      attachment.presentation.subscribeMenu((v) => {
        view = v;
      })();
      return view;
    },
  };
}

const ROW_BOUNDS = new Map<string, { top: number; bottom: number }>([
  ["/content/0/content/0/row/0", { top: 0, bottom: 40 }],
  ["/content/0/content/0/row/1", { top: 40, bottom: 80 }],
]);

test("the tasks page publishes through the refresh pipeline with its rows", async (t) => {
  const h = harness(t, [
    { task: "t-0", title: "the cited seed account" },
    { task: "t-1", title: "the citing seed account" },
  ], ROW_BOUNDS);
  const outcome = await h.attachment.places.request(PLANE);
  assert.equal(outcome, "applied");
  const state = h.built.host.snapshot();
  assert.equal(state.kind, "alive");
  if (state.kind === "alive" && state.content.kind === "available") {
    // the host's snapshot is the raw state: the frame is the content
    // frame's VALUE (the registry's projection wraps it for rendering)
    const frame = state.content.value.value;
    assert.equal(frame.status, "loaded");
    assert.equal(frame.location, "plane");
    assert.equal(frame.project, "tasks-test");
    assert.equal(frame.asked, "plane", "the plane's location names no tracker");
    assert.equal(frame.targets.length, 2);
    assert.equal(frame.targets[0]?.task, "t-0");
  } else {
    assert.fail("the tasks page published");
  }
});

test("an address that is not the plane refuses with the named gap, never a guess", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  assert.equal(await h.attachment.places.request({ machine: MACHINE, path: "t-0" }), "failed");
  assert.equal(
    await h.attachment.places.request({ machine: MACHINE, path: "tasks" }),
    "failed",
    "a tracker's pipeline is no location of its own (t-54)",
  );
});

test("j/k is pure scrolling (no row cursor); gg/G hit the ends", (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  assert.equal(h.press("j"), "consume");
  assert.equal(h.press("k"), "consume");
  assert.equal(h.press("g"), "consume", "gg is a prefix");
  h.press("g");
  assert.equal(h.press("G"), "consume");
  assert.equal(h.press("x"), "consume", "unmatched application keys are consumed by the reading context");
});

test("label selection lands on the row and opens its act menu", async (t) => {
  const h = harness(t, [
    { task: "t-0", title: "the cited seed account" },
    { task: "t-1", title: "the citing seed account" },
  ], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  const state0 = h.built.host.snapshot();
  console.log("DBG state:", state0.kind, state0.kind === "alive" ? state0.content.kind : "-", state0.kind === "alive" && state0.content.kind === "available" ? JSON.stringify(state0.content.value.value.targets.map((t2) => t2.part)) : "-");
  assert.equal(h.press("s"), "consume");
  console.log("DBG select view:", JSON.stringify(h.selectView()));
  // labels paint while the session is active (it exits on selection)
  const view = h.selectView() as { active: boolean; labels: { part: string; label: string }[] };
  assert.equal(view.active, true);
  // one namespace, the prototype's home-row order (m-545)
  assert.deepEqual(view.labels.map((entry) => entry.label), ["A", "S"]);
  assert.equal(h.press("a"), "consume");
  assert.equal(h.attachment.presentation.getSelection(), "t-0");
  assert.equal((h.selectView() as { active: boolean } | null), null, "exact-single selects and exits");
  const menu = h.menuView() as { open: boolean; label: string; acts: { key: string }[] };
  assert.equal(menu.open, true);
  assert.equal(menu.label, "t-0 the cited seed account");
  assert.deepEqual(menu.acts.map((act) => act.key), ["o", "i", "w", "y"]);
});

test("Enter (the primary) opens the task's board THROUGH THE LINK - F6's ratified primary", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "the cited seed account" }], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  h.press("s");
  h.press("a");
  assert.deepEqual(h.opened, [], "no act ran on selection alone");
  assert.deepEqual(h.elsewhere, []);
  h.press("Enter");
  // the primary is the link act: with one window the elsewhere
  // surface opens a new linked window without asking - never the
  // in-place open that reversed the ratified order
  assert.deepEqual(h.elsewhere, [["tasks-test/t-0", "link"]]);
  assert.deepEqual(h.opened, [], "the primary never opens in place");
});

test("y (the last act) copies the task's reference", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "the cited seed account" }], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  h.press("s");
  h.press("a");
  h.press("y");
  assert.deepEqual(h.elsewhere, [], "copy never opens anything");
  assert.deepEqual(h.copied, ["tasks-test/t-0"]);
});

test("o/i/w target the label-selected task through the elsewhere surface", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "the cited seed account" }], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  // each OPEN act runs with the menu open: the act closes the menu
  // (close-before-execute), so the next act re-selects first - the
  // scenario 18 script's shape ("on a reference run o; then i; then
  // w")
  const selectFirst = (): void => {
    h.press("s");
    h.press("a");
  };
  selectFirst();
  h.press("o");
  selectFirst();
  h.press("i");
  selectFirst();
  h.press("w");
  assert.deepEqual(h.elsewhere, [
    ["tasks-test/t-0", "link"],
    ["tasks-test/t-0", "once"],
    ["tasks-test/t-0", "new"],
  ]);
  assert.deepEqual(h.opened, [], "the OPEN acts never open here");
});

test("with no selection no task act runs (the menu is not open)", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  h.press("o");
  h.press("i");
  h.press("w");
  h.press("Enter");
  assert.deepEqual(h.opened, []);
  assert.deepEqual(h.elsewhere, []);
});

test("a plain click runs the clicked row's first act directly", async (t) => {
  const h = harness(t, [
    { task: "t-0", title: "the cited seed account" },
    { task: "t-1", title: "the citing seed account" },
  ], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  h.attachment.presentation.rowPrimaryClick("/content/0/content/0/row/1");
  // F6: the row's first act is o through the link - a plain click
  // delivers through the elsewhere surface, never in place
  assert.deepEqual(h.elsewhere, [["tasks-test/t-1", "link"]]);
  assert.deepEqual(h.opened, []);
});

test("departure capture records the row under the reading position; restore returns by identity plus offset", async (t) => {
  // the second row sits at 40..80; the window rests at scrollTop 50,
  // so the row under the reading position (view top) is row 1
  const h = harness(t, [
    { task: "t-0", title: "the cited seed account" },
    { task: "t-1", title: "the citing seed account" },
  ], ROW_BOUNDS, 50);
  await h.attachment.places.request(PLANE);
  // the window's trail captures through the buffer's return places
  // (RunningInstrument.places), not the content-places adapter
  const capture = h.built.host.places.capture();
  // the plane's address carries the project it shows (t-54)
  assert.equal(capture.address.machine, "plane");
  assert.equal(capture.address.path, "tasks-test");
  if (capture.position.kind !== "anchor") {
    assert.fail("the reading position sits inside row 1");
  }
  // the trail carries the unconstrained AnchorData; the surface's own
  // anchor contract is the task
  const anchor = capture.position.anchor as { kind: string; task: string };
  assert.equal(anchor.kind, "task");
  assert.equal(anchor.task, "t-1");

  // the restore: open away, then restore the captured place - the row
  // returns at its recorded viewport offset (bounds.top - offset.top)
  await h.attachment.places.request(PLANE);
  const job = h.built.host.places.restore({
    bufferId: "tasks-1",
    address: PLANE,
    position: {
      kind: "anchor",
      anchor: { kind: "task", task: "t-1" },
      viewportOffset: { left: 0, top: -10 },
    },
  });
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(job.snapshot(), "restored");
  assert.equal(h.built.host.viewport.metrics().position.top, 50, "row bounds 40 minus offset -10");
});

test("an anchor naming a row the page does not carry is refused, not guessed", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  await h.attachment.places.request(PLANE);
  const accepts = (anchor: unknown): boolean => h.attachment.places.acceptsAnchor(anchor as never);
  assert.equal(accepts({ kind: "task", task: "t-99" }), false);
  assert.equal(accepts({ kind: "task", task: "t-0" }), true);
  assert.equal(
    accepts({ kind: "board-part", path: "/content/0" }),
    false,
    "a board anchor is not a task anchor",
  );
});

test("a selection whose row leaves the page clears (scenario 12 identity)", async (t) => {
  const machine = MACHINE;
  const source = sourceFor(machine) as TrackerSource;
  let page = tasksPageFixture([
    { task: "t-0", title: "the cited seed account" },
    { task: "t-1", title: "the citing seed account" },
  ]);
  vi.spyOn(source, "instrumentPage").mockImplementation(async () => page);
  const { scroller } = fakeScroller(ROW_BOUNDS);
  const viewport = {
    metrics: () => ({
      position: { left: 0, top: scroller.scrollTop },
      width: 400,
      height: 200,
    }),
    scrollLines: () => {},
    scrollPage: () => {},
    start: () => {},
    end: () => {},
    reveal: () => {},
    moveTo: () => {},
    attachElement: () => {},
    attachedElement: () => scroller,
  };
  let attachment!: InstrumentAttachment<TasksUI, TaskAnchor>;
  const host = new SliceBufferHost<TasksData, TasksUI, TaskAnchor>({
    context: { windowId: "w", bufferId: "tasks-1" },
    viewport: viewport as unknown as import("../src/lib/contracts/buffer.ts").Viewport,
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) {
      attachment = connectTasks(buffer);
      return attachment;
    },
  });
  const built = host.wire();
  built.host.activate();
  onTestFinished(() => built.host.destroy());
  attachment.presentation.setTaskPorts({ open: () => {} });
  await attachment.places.request(PLANE);
  built.host.handleKey({ key: "s", ctrl: false, shift: false, alt: false, meta: false });
  built.host.handleKey({ key: "a", ctrl: false, shift: false, alt: false, meta: false });
  assert.equal(attachment.presentation.getSelection(), "t-0");
  page = tasksPageFixture([{ task: "t-1", title: "the citing seed account" }]);
  await attachment.places.request(PLANE);
  assert.equal(
    attachment.presentation.getSelection(),
    null,
    "t-0 left the page: the selection clears, never a silent neighbour",
  );
});

test("the control plane: rail entries are targets - s labels them after the rows; a project's Enter switches here (t-48)", async (t) => {
  const project = projectPart("tasks-test");
  const bounds = new Map([...ROW_BOUNDS, [project, { top: 0, bottom: 20 }]]);
  const h = harness(t, [
    { task: "t-0", title: "x" },
    { task: "t-1", title: "y" },
  ], bounds);
  await h.attachment.places.request(PLANE);
  h.press("s");
  const view = h.selectView() as { labels: { part: string; label: string }[] };
  assert.deepEqual(
    view.labels.map((entry) => [entry.part, entry.label]),
    [
      ["/content/0/content/0/row/0", "A"],
      ["/content/0/content/0/row/1", "S"],
      [project, "D"],
    ],
    "rows keep the labels they had; the rail labels after them",
  );
  h.press("d");
  const menu = h.menuView() as { label: string; acts: { key: string }[] };
  assert.deepEqual(menu.acts.map((act) => act.key), ["f", "a", "d"], "even the shown project can be forgotten");
  h.press("Enter");
  assert.deepEqual(h.opened, ["plane tasks-test"]);
  assert.equal(h.attachment.presentation.getSelection(), null, "a rail entry is not a task selection");
  // a plain click on the entry runs the same first act
  h.attachment.presentation.rowPrimaryClick(project);
  assert.deepEqual(h.opened, ["plane tasks-test", "plane tasks-test"]);
  // the rail marks the shown project; outside the native shell it
  // says Herdr's words are not here rather than showing nobody
  let rail: import("../src/lib/tasks/rail.ts").Rail | null = null;
  const off = h.attachment.presentation.subscribeRail((next) => {
    rail = next;
  });
  off();
  const shown = rail as import("../src/lib/tasks/rail.ts").Rail | null;
  assert.ok(shown !== null);
  assert.ok(shown.projects.some((entry) => entry.name === "tasks-test" && entry.current));

  // d on the shown project: it leaves the rail though the list stays
  h.attachment.presentation.openMenuFor({ part: project, kind: "project", label: "tasks-test" });
  h.press("d");
  const after = (() => {
    let next: import("../src/lib/tasks/rail.ts").Rail | null = null;
    h.attachment.presentation.subscribeRail((value) => {
      next = value;
    })();
    return next as unknown as import("../src/lib/tasks/rail.ts").Rail;
  })();
  assert.ok(!after.projects.some((entry) => entry.name === "tasks-test"), "the forgotten shown project leaves the rail");
});

// -- the plane's own address and the add field (t-54) ----------------

import { knownTrackers } from "../src/lib/tracker.ts";

function stubRegistry(seed: Record<string, string> = {}): void {
  const store = new Map<string, string>();
  if (Object.keys(seed).length > 0) {
    store.set("saccade.trackers", JSON.stringify(seed));
    store.set("saccade.trackers.home", Object.keys(seed)[0]!);
  }
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
}

function frameOf(h: Harness): TasksData {
  const state = h.built.host.snapshot();
  if (state.kind !== "alive" || state.content.kind !== "available") assert.fail("the plane published a frame");
  return state.content.value.value;
}

function railOfHarness(h: Harness): import("../src/lib/tasks/rail.ts").Rail {
  let rail: import("../src/lib/tasks/rail.ts").Rail | null = null;
  h.attachment.presentation.subscribeRail((next) => {
    rail = next;
  })();
  return rail as unknown as import("../src/lib/tasks/rail.ts").Rail;
}

test("an empty plane: no project known is no error; a opens the add field, Enter registers host:port/name and opens nothing", async (t) => {
  stubRegistry();
  const h = harness(t, [], ROW_BOUNDS);
  assert.equal(await h.attachment.places.request({ machine: "plane", path: "" }), "applied");
  const empty = frameOf(h);
  assert.equal(empty.status, "loaded");
  assert.equal(empty.project, null);
  assert.equal(empty.problem, null);
  assert.equal(empty.asked, "plane");

  assert.equal(h.press("a"), "consume");
  assert.equal(h.attachment.presentation.addView().open, true);
  assert.equal(h.press("x"), "allow-default", "typing stays the field's own");
  h.attachment.presentation.typeAdd("not a project");
  h.press("Enter");
  assert.equal(h.attachment.presentation.addView().open, true, "a refused address keeps the field open");
  assert.match(h.attachment.presentation.addView().problem ?? "", /host:port\/name/);
  assert.deepEqual(knownTrackers(), {});

  h.attachment.presentation.typeAdd("127.0.0.1:9/pyrimel");
  assert.equal(h.attachment.presentation.addView().problem, null, "typing clears the refusal");
  h.press("Enter");
  assert.equal(h.attachment.presentation.addView().open, false);
  assert.deepEqual(knownTrackers(), { pyrimel: "127.0.0.1:9" });
  assert.ok(railOfHarness(h).projects.some((project) => project.name === "pyrimel"), "the rail lists it");
  assert.deepEqual(h.opened, [], "registering opens nothing");
  assert.deepEqual(h.elsewhere, []);
  assert.equal(frameOf(h).project, null, "the shown project stays");

  // Escape leaves the field and returns the keys to the plane
  h.press("a");
  assert.equal(h.attachment.presentation.addView().open, true);
  h.press("Escape");
  assert.equal(h.attachment.presentation.addView().open, false);
  assert.equal(h.press("j"), "consume", "the reading context has the keys again");
});

test("a bare plane shows the home project; an unreachable one keeps the rail and says why", async (t) => {
  stubRegistry({ "tasks-test": "127.0.0.1:9" });
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  await h.attachment.places.request({ machine: "plane", path: "" });
  const shown = frameOf(h);
  assert.equal(shown.project, "tasks-test");
  assert.equal(shown.status, "loaded");
  assert.equal(shown.targets.length, 1);

  // a project this device never reached: the plane names the problem
  // and the rail stays, so another project is one act away
  assert.equal(await h.attachment.places.request({ machine: "plane", path: "ghost" }), "applied");
  const dead = frameOf(h);
  assert.equal(dead.status, "invalid");
  assert.equal(dead.project, "ghost");
  assert.match(dead.problem ?? "", /ghost/);
  assert.equal(dead.targets.length, 0);
  assert.ok(railOfHarness(h).projects.some((project) => project.name === "tasks-test"));

  // a bare plane again keeps a known shown project, else the home
  await h.attachment.places.request({ machine: "plane", path: "" });
  assert.equal(frameOf(h).project, "tasks-test", "ghost is not known: the home shows");
});

test("an agent no tracker registered is a rail target: s labels it, its menu is y alone, Enter copies hc attach <address> (t-55)", async (t) => {
  const { terminalPart } = await import("../src/lib/tasks/rail.ts");
  const source = sourceFor(MACHINE) as TrackerSource;
  // the tracker's root names its repo; its agents page registers no one
  vi.spyOn(source, "identity").mockImplementation(async () => ({ machineId: null, repoPath: "/srv/tasks-test" }));
  vi.spyOn(source, "agentsPage").mockImplementation(async () => ({
    view: { major: 1 },
    kind: "document",
    location: "agents",
    seq: null,
    content: [],
  }));
  // the shell's feed: a stray agent in the repo through a federating
  // core, one nowhere a project knows, and a plain shell
  const snapshot = {
    generation: 1,
    connected: true,
    words: {
      "other/t63": { state: "working", app: "claude-code", cwd: "/srv/tasks-test", terminal_title: "tasks-test / general", at: 1 },
      t7r: { state: "done", app: "claude-code", cwd: "/elsewhere", at: 2 },
      t3k: { cwd: "/srv/tasks-test", terminal_title: "tasks-test / service", at: 3 },
    },
  };
  vi.stubGlobal("window", {
    __TAURI_INTERNALS__: {
      invoke: (command: string, args: { since: number }) =>
        command === "words_wait" && args.since < 1 ? Promise.resolve(snapshot) : new Promise(() => {}),
    },
  });
  // the project is one this device reached: the rail's first read
  // covers it
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  const { registerTracker } = await import("../src/lib/tracker.ts");
  registerTracker("tasks-test", "localhost:8811");
  const stray = terminalPart("other/t63");
  const foot = terminalPart("t7r");
  const bounds = new Map([
    ...ROW_BOUNDS,
    [projectPart("tasks-test"), { top: 0, bottom: 20 }],
    [stray, { top: 20, bottom: 40 }],
    [foot, { top: 40, bottom: 60 }],
  ]);
  const h = harness(t, [{ task: "t-0", title: "x" }], bounds);
  await h.attachment.places.request(PLANE);
  let rail: import("../src/lib/tasks/rail.ts").Rail | null = null;
  const off = h.attachment.presentation.subscribeRail((next) => {
    rail = next;
  });
  for (let tries = 0; tries < 50; tries++) {
    const shown = rail as import("../src/lib/tasks/rail.ts").Rail | null;
    if (shown !== null && shown.unregistered.length > 0 && shown.projects.some((p) => p.groups.length > 0)) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  off();
  const shown = rail as unknown as import("../src/lib/tasks/rail.ts").Rail;
  const project = shown.projects.find((entry) => entry.name === "tasks-test");
  assert.deepEqual(
    project?.groups.map((group) => [group.state, group.agents.map((agent) => [agent.incarnation, agent.address, agent.label])]),
    [["working", [[null, "other/t63", "tasks-test / general"]]]],
    "under its project, unregistered, named by the terminal's title; the shell never shows",
  );
  assert.deepEqual(shown.unregistered.map((agent) => agent.address), ["t7r"]);
  h.press("s");
  const view = h.selectView() as { labels: { part: string; label: string }[] };
  const label = view.labels.find((entry) => entry.part === stray)?.label;
  assert.ok(label !== undefined, "the stray is labelled with the rail");
  assert.ok(view.labels.some((entry) => entry.part === foot), "the foot group's agent too");
  for (const key of label) h.press(key);
  const menu = h.menuView() as { label: string; acts: { key: string; label: string }[] };
  assert.equal(menu.label, "tasks-test / general other/t63");
  assert.deepEqual(menu.acts.map((act) => [act.key, act.label]), [["y", "copy hc attach other/t63"]]);
  h.press("Enter");
  assert.deepEqual(h.copied, ["hc attach other/t63"]);
  assert.deepEqual(h.opened, [], "copying opens nothing");
  // a plain click on the foot group's entry runs its one act
  h.attachment.presentation.rowPrimaryClick(foot);
  assert.deepEqual(h.copied, ["hc attach other/t63", "hc attach t7r"]);
});
