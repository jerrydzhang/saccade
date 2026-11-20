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
import { PageRefusal, type ViewPage } from "../src/lib/pages.ts";
import type { ElsewhereAct } from "../src/lib/elsewhere.ts";
import { TrackerSource } from "../src/lib/tracker.ts";
import { sourceFor } from "../src/lib/sources.ts";
import { projectPart } from "../src/lib/tasks/rail.ts";

const MACHINE = "tracker:tasks-test";
// a project's task list: the location names the project's tracker
// machine and the list
const TASKS = { machine: MACHINE, path: "tasks" };
// the words feed's observers, for the rows' agent marks (the frame
// rail's feed, injected)
const wordsObservers: ((rail: import("../src/lib/tasks/rail.ts").Rail) => void)[] = [];
function frameRailWords(rail: import("../src/lib/tasks/rail.ts").Rail): void {
  for (const observer of [...wordsObservers]) observer(rail);
}

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
  presentations(): import("../src/components/Buffer.ts").TargetPresentations | null;
  selectView(): unknown;
  menuView(): unknown;
  showActions(part: string): void;
  plainClick(part: string): boolean;
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
  const copied: string[] = [];
  // the window services ride the definition: the harness hands
  // them at connect, exactly as the window assembly does
  let attachment!: InstrumentAttachment<TasksUI, TaskAnchor>;
  const host = new SliceBufferHost<TasksData, TasksUI, TaskAnchor>({
    context: { windowId: "w", bufferId: "tasks-1" },
    viewport: viewport as unknown as import("../src/lib/contracts/buffer.ts").Viewport,
    diagnostics: { report: () => "", unexpected: () => "" },
    connect(buffer) {
      attachment = connectTasks(buffer, {
        open: (spec) => opened.push(spec),
        elsewhere: (spec, act) => elsewhere.push([spec, act]),
        copy: (text) => copied.push(text),
        railWords: (observer) => {
          wordsObservers.push(observer);
          return () => {
            wordsObservers.length = 0;
          };
        },
      });
      return attachment;
    },
  });
  const built = host.wire();
  built.host.activate();
  onTestFinished(() => built.host.destroy());
  const press = (key: string): string =>
    built.host.handleKey({ key, ctrl: false, shift: false, alt: false, meta: false });
  // the select and menu sessions are the frame's, behind the lease
  // (the frame's face): the views read off the buffer's presentations
  const presentations = (): import("../src/components/Buffer.ts").TargetPresentations | null => {
    const state = built.host.snapshot();
    if (state.kind !== "alive" || state.content.kind !== "available") return null;
    return built.host.presentationsFor(state.content.value);
  };
  return {
    built,
    attachment,
    opened,
    elsewhere,
    copied,
    press,
    presentations,
    selectView: () => presentations()?.selectView() ?? null,
    menuView: () => presentations()?.menuView() ?? null,
    showActions: (part: string): void => {
      const lease = presentations()?.presentTarget(part, () => ({ left: 0, top: 0, right: 0, bottom: 0 }));
      lease?.showActions({ left: 0, top: 0 });
    },
    /** The frame's own click path: the lease the surface would
     * create for the part, activated as a plain click. */
    plainClick: (part: string): boolean =>
      presentations()?.presentTarget(part, () => ({ left: 0, top: 0, right: 0, bottom: 0 })).activatePointer("plain") ?? false,
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
  const outcome = await h.attachment.places.request(TASKS);
  assert.equal(outcome, "applied");
  const state = h.built.host.snapshot();
  assert.equal(state.kind, "alive");
  if (state.kind === "alive" && state.content.kind === "available") {
    // the host's snapshot is the raw state: the frame is the content
    // frame (the registry's projection wraps it for rendering), its
    // targets in the publication's own slot - the rows, and nothing
    // else (the rail is the frame's)
    const published = state.content.value;
    const frame = published.value;
    assert.equal(frame.status, "loaded");
    assert.equal(frame.location, "tasks-test/tasks");
    assert.equal(frame.project, "tasks-test");
    assert.equal(frame.asked, "tasks-test/tasks");
    assert.equal(published.targets.length, 2);
    assert.equal(published.targets[0]?.anchor, "t-0");
    assert.equal(published.targets[0]?.id, "/content/0/content/0/row/0");
    assert.equal(published.targets[0]?.label, "t-0 the cited seed account");
    assert.deepEqual(
      published.targets.map((target) => target.kind),
      ["task", "task"],
    );
  } else {
    assert.fail("the tasks page published");
  }
});

test("an address that is not a project's task list refuses with the named gap, never a guess", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  assert.equal(await h.attachment.places.request({ machine: MACHINE, path: "t-0" }), "failed");
  assert.equal(
    await h.attachment.places.request({ machine: "plane", path: "" }),
    "failed",
    "the plane is gone: a window with no location is the empty window",
  );
});

test("a late answer for a list no longer shown changes nothing: the newer load's rows stay", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  const source = sourceFor(MACHINE) as TrackerSource;
  let answerLate: (page: ViewPage) => void = () => {};
  const late = new Promise<ViewPage>((resolve) => {
    answerLate = resolve;
  });
  vi.spyOn(source, "instrumentPage")
    .mockImplementationOnce(() => late)
    .mockImplementationOnce(async () => tasksPageFixture([{ task: "t-1", title: "the newer list" }]));
  const older = h.attachment.places.request(TASKS);
  const newer = h.attachment.places.request(TASKS);
  assert.equal(await newer, "applied");
  answerLate(tasksPageFixture([{ task: "t-0", title: "the older list, answered late" }]));
  assert.equal(await older, "superseded");
  const state = h.built.host.snapshot();
  if (state.kind !== "alive" || state.content.kind !== "available") assert.fail("the newer list is shown");
  assert.deepEqual(
    state.content.value.targets.map((target) => target.label),
    ["t-1 the newer list"],
    "the late answer never overwrote the newer list",
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
  await h.attachment.places.request(TASKS);
  assert.equal(h.press("s"), "consume");
  // labels paint while the session is active (it exits on selection)
  const view = h.selectView() as { active: boolean; labels: { part: string; label: string }[] };
  assert.equal(view.active, true);
  // one namespace, the prototype's home-row order
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
  await h.attachment.places.request(TASKS);
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
  await h.attachment.places.request(TASKS);
  h.press("s");
  h.press("a");
  h.press("y");
  assert.deepEqual(h.elsewhere, [], "copy never opens anything");
  assert.deepEqual(h.copied, ["tasks-test/t-0"]);
});

test("o/i/w target the label-selected task through the elsewhere surface", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "the cited seed account" }], ROW_BOUNDS);
  await h.attachment.places.request(TASKS);
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
  await h.attachment.places.request(TASKS);
  h.press("o");
  h.press("i");
  h.press("w");
  h.press("Enter");
  assert.deepEqual(h.opened, []);
  assert.deepEqual(h.elsewhere, []);
});

test("departure capture records the row under the reading position; restore returns by identity plus offset", async (t) => {
  // the second row sits at 40..80; the window rests at scrollTop 50,
  // so the row under the reading position (view top) is row 1
  const h = harness(t, [
    { task: "t-0", title: "the cited seed account" },
    { task: "t-1", title: "the citing seed account" },
  ], ROW_BOUNDS, 50);
  await h.attachment.places.request(TASKS);
  // the window's trail captures through the buffer's return places
  // (RunningInstrument.places), not the content-places adapter
  const capture = h.built.host.places.capture();
  // the address names the project's list
  assert.equal(capture.address.machine, MACHINE);
  assert.equal(capture.address.path, "tasks");
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
  await h.attachment.places.request(TASKS);
  const job = h.built.host.places.restore({
    bufferId: "tasks-1",
    address: TASKS,
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
  await h.attachment.places.request(TASKS);
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
      attachment = connectTasks(buffer, { open: () => {} });
      return attachment;
    },
  });
  const built = host.wire();
  built.host.activate();
  onTestFinished(() => built.host.destroy());
  await attachment.places.request(TASKS);
  built.host.handleKey({ key: "s", ctrl: false, shift: false, alt: false, meta: false });
  built.host.handleKey({ key: "a", ctrl: false, shift: false, alt: false, meta: false });
  assert.equal(attachment.presentation.getSelection(), "t-0");
  page = tasksPageFixture([{ task: "t-1", title: "the citing seed account" }]);
  await attachment.places.request(TASKS);
  assert.equal(
    attachment.presentation.getSelection(),
    null,
    "t-0 left the page: the selection clears, never a silent neighbour",
  );
});

test("a narrowed list asks the tracker for its narrowing, shows the location canonical, and holds the page to the echo", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  const source = sourceFor(MACHINE) as TrackerSource;
  const asked: (string | undefined)[] = [];
  vi.spyOn(source, "instrumentPage").mockImplementation(async (_path, query) => {
    asked.push(query);
    return { ...tasksPageFixture([{ task: "t-0", title: "x" }]), narrowing: { tags: ["UI", "auth"], untagged: false } };
  });
  assert.equal(await h.attachment.places.request({ machine: MACHINE, path: "tasks?tag=UI,auth,ui" }), "applied");
  assert.deepEqual(asked, ["tag=auth&tag=ui"]);
  const state = h.built.host.snapshot();
  assert.ok(state.kind === "alive" && state.content.kind === "available");
  const frame = state.content.value.value;
  assert.equal(frame.status, "loaded");
  assert.equal(frame.location, "tasks-test/tasks?tag=auth,ui");
  assert.equal(frame.resolved.path, "tasks?tag=auth,ui");
  assert.equal(frame.page?.groups[0]?.rows.length, 1);
});

test("a tracker that does not echo the narrowing is refused: the whole list never stands in for it", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  const source = sourceFor(MACHINE) as TrackerSource;
  // an older tracker ignores the query it does not know and answers everything
  vi.spyOn(source, "instrumentPage").mockImplementation(async () => tasksPageFixture([{ task: "t-0", title: "x" }]));
  await h.attachment.places.request({ machine: MACHINE, path: "tasks?tag=auth" });
  const state = h.built.host.snapshot();
  assert.ok(state.kind === "alive" && state.content.kind === "available");
  const frame = state.content.value.value;
  assert.equal(frame.status, "invalid");
  assert.equal(frame.page, null);
  assert.match(frame.problem ?? "", /does not narrow by tag/);
  assert.equal(state.content.value.targets.length, 0);
});

test("an unknown tag shows the tracker's refusal with the tags that exist, on the problem line", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  const source = sourceFor(MACHINE) as TrackerSource;
  vi.spyOn(source, "instrumentPage").mockRejectedValue(
    new PageRefusal("unknown_tag", "no tag 'nope'; the tags are:\nauth - sign-in\nui - screens", 400),
  );
  await h.attachment.places.request({ machine: MACHINE, path: "tasks?tag=nope" });
  const state = h.built.host.snapshot();
  assert.ok(state.kind === "alive" && state.content.kind === "available");
  const frame = state.content.value.value;
  assert.equal(frame.status, "invalid");
  assert.match(frame.problem ?? "", /no tag 'nope'/);
  assert.match(frame.problem ?? "", /auth - sign-in/);
  assert.match(frame.problem ?? "", /ui - screens/);
});

test("a refusal the transport reduced to a failed status is explained from the repository's tags", async (t) => {
  const h = harness(t, [{ task: "t-0", title: "x" }], ROW_BOUNDS);
  const source = sourceFor(MACHINE) as TrackerSource;
  vi.spyOn(source, "instrumentPage").mockRejectedValue(new Error("the tracker answered 400 for /view/v1/tasks?tag=nope"));
  vi.spyOn(source, "tags").mockResolvedValue([{ name: "auth", description: "sign-in" }]);
  await h.attachment.places.request({ machine: MACHINE, path: "tasks?tag=nope" });
  const state = h.built.host.snapshot();
  assert.ok(state.kind === "alive" && state.content.kind === "available");
  assert.equal(state.content.value.value.problem, "no tag 'nope'; the tags are:\nauth - sign-in");
});
