// The rail keeps views: a view is a kept location shown beneath its
// repository. The declaration offers some (the file's spelling is
// parseDeclaration's alone), the device keeps others (Space k, the
// store in lib/views.ts), and the frame holds only canonical
// locations the claiming instrument spelled. These are the pure
// halves: the declaration's views, canonicalization making two
// spellings one view, the device store and its forgetting, the rail's
// keep and forget acts, the label hook, and the key bound once.

import assert from "node:assert/strict";
import { onTestFinished, test, vi } from "vitest";

import { directAsks, theProjects } from "../src/lib/asks.ts";
import { createFrameRail } from "../src/lib/frame/rail.ts";
import { readView, resolveViews } from "../src/lib/frame/views.ts";
import { locationTitleFor } from "../src/lib/instruments/claims.ts";
import "../src/lib/instruments/index.ts";
import { WINDOW_CHORDS, windowContext } from "../src/lib/input/contexts.ts";
import { createHostMatcher } from "../src/lib/input/host.ts";
import { parseDeclaration, parseRegisteredPath } from "../src/lib/projects.ts";
import { locationTitleOf } from "../src/lib/tasks/narrowing.ts";
import { deviceViews, forgetDeviceView, keepDeviceView } from "../src/lib/views.ts";

function stubStore(initial: Record<string, string> = {}): Map<string, string> {
  const store = new Map<string, string>(Object.entries(initial));
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  return store;
}

// -- the declaration's views ----------------------------------------

test("a declaration's views: one line, several lines with a trailing comma, and none", () => {
  const one = parseDeclaration('name = "p"\nviews = ["tasks?tag=app", "tasks?untagged"]\n\n[tracker]\nport = 9\n');
  assert.deepEqual(one, { name: "p", port: 9, views: ["tasks?tag=app", "tasks?untagged"], viewsProblem: null });
  const many = parseDeclaration('name = "p"\nviews = [\n  "tasks?tag=app", # the app\n  "t-5",\n]\n[tracker]\nport = 9\n');
  assert.deepEqual(many?.views, ["tasks?tag=app", "t-5"]);
  assert.equal(many?.port, 9, "the port after a multi-line array still reads");
  assert.deepEqual(parseDeclaration('name = "p"\n')?.views, []);
  assert.deepEqual(parseDeclaration('name = "p"\nviews = []\n')?.views, []);
});

test("a malformed views line is reported and never breaks the name or the port", () => {
  for (const line of ['views = "tasks"', 'views = [tasks]', 'views = ["a" "b"]', 'views = ["a",', 'views = [1, 2]']) {
    const parsed = parseDeclaration(`name = "p"\n${line}\n[tracker]\nport = 9\n`);
    assert.equal(parsed?.name, "p", line);
    assert.equal(parsed?.port, 9, line);
    assert.deepEqual(parsed?.views, [], line);
    assert.ok(parsed?.viewsProblem !== null, line);
  }
});

test("a views line after a section is no top-level key and is ignored", () => {
  const parsed = parseDeclaration('name = "p"\n[tracker]\nport = 9\nviews = ["tasks?tag=app"]\n');
  assert.deepEqual(parsed, { name: "p", port: 9, views: [], viewsProblem: null });
});

// -- canonicalization and resolution --------------------------------

test("two spellings of one view are one view; the file's wins over the device's", () => {
  const resolved = resolveViews("proj", ["tasks?tag=UI,app", "tasks?tag=app,ui"], ["tasks?tag=app,ui", "t-5"]);
  assert.deepEqual(
    resolved.views.map((view) => [view.location, view.source]),
    [["proj/tasks?tag=app,ui", "file"], ["proj/t-5", "device"]],
  );
  assert.deepEqual(resolved.problems, []);
});

test("a file view nothing can read is a problem naming it, and the others still stand", () => {
  const resolved = resolveViews("proj", ["tasks?tag=", "tasks?untagged", "nonsense", "agents?x=1"], []);
  assert.deepEqual(resolved.views.map((view) => view.location), ["proj/tasks?untagged"]);
  assert.equal(resolved.problems.length, 3);
  assert.match(resolved.problems[0]!, /^workbench\.toml view 'tasks\?tag=' cannot be read: tag= names no tag/);
  assert.match(resolved.problems[1]!, /view 'nonsense' cannot be read: 'nonsense' is not a location inside proj/);
  assert.match(resolved.problems[2]!, /view 'agents\?x=1' cannot be read: 'agents\?x=1' is not a location inside proj/);
});

test("any location of the repository can be kept, and the label is the words or the path", () => {
  const verdict = readView("proj", "t-5@r2");
  assert.deepEqual(verdict, { ok: true, path: "t-5@r2", location: "proj/t-5@r2" });
  const views = resolveViews("proj", ["tasks", "tasks?tag=app,ui", "tasks?untagged", "t-5"], []).views;
  assert.deepEqual(views.map((view) => view.label), ["tasks", "app, ui", "untagged", "t-5"]);
});

test("the tasks instrument words a kept view; others offer none", () => {
  assert.equal(locationTitleOf("tasks"), "tasks");
  assert.equal(locationTitleOf("tasks?tag=app,ui"), "app, ui");
  assert.equal(locationTitleOf("tasks?tag=UI,app"), "app, ui");
  assert.equal(locationTitleOf("tasks?untagged"), "untagged");
  assert.equal(locationTitleOf("tasks?tag="), null);
  assert.equal(locationTitleOf("t-5"), null);
  assert.equal(locationTitleFor({ machine: "tracker:proj", path: "tasks?untagged" }), "untagged");
  assert.equal(locationTitleFor({ machine: "tracker:proj", path: "t-5" }), null, "no hook answers: the path stands");
});

// -- the device store -----------------------------------------------

test("the device's views round-trip beneath each registered path, keep once, and forget one", () => {
  const store = stubStore();
  keepDeviceView("box:/srv/a", "tasks?tag=app");
  keepDeviceView("box:/srv/a", "tasks?untagged");
  keepDeviceView("box:/srv/a", "tasks?tag=app");
  keepDeviceView("box:/srv/b", "t-5");
  assert.deepEqual(JSON.parse(store.get("saccade.views") ?? "{}"), {
    "box:/srv/a": ["tasks?tag=app", "tasks?untagged"],
    "box:/srv/b": ["t-5"],
  }, "paths inside the repository, no name: a rename keeps them");
  assert.deepEqual(deviceViews("box:/srv/a"), ["tasks?tag=app", "tasks?untagged"]);
  forgetDeviceView("box:/srv/a", "tasks?tag=app");
  assert.deepEqual(deviceViews("box:/srv/a"), ["tasks?untagged"]);
  assert.deepEqual(deviceViews("box:/srv/none"), []);
});

test("a refused or garbled store reads as no views and never throws", () => {
  stubStore({ "saccade.views": "{not json" });
  assert.deepEqual(deviceViews("box:/srv/a"), []);
  vi.stubGlobal("localStorage", {
    getItem: () => {
      throw new Error("refused");
    },
    setItem: () => {
      throw new Error("refused");
    },
  });
  assert.deepEqual(deviceViews("box:/srv/a"), []);
  keepDeviceView("box:/srv/a", "tasks");
});

function stubWindowWithDeclaration(text: string): void {
  vi.stubGlobal("window", {
    __saccadeAsks: directAsks({
      readFile: async () => text,
      readAsset: async () => {
        throw new Error("no assets here");
      },
      requestLocal: async () => {
        throw new Error("no requests here");
      },
    }),
  });
}

test("unregistering a repository forgets the views the device kept beneath it", async () => {
  const store = stubStore();
  stubWindowWithDeclaration('name = "proj"\n');
  const projects = theProjects();
  const registered = parseRegisteredPath("box:/srv/proj");
  assert.ok(registered !== null);
  await projects.register(registered);
  keepDeviceView("box:/srv/proj", "tasks?untagged");
  keepDeviceView("box:/srv/other", "tasks");
  projects.forget(registered);
  assert.deepEqual(JSON.parse(store.get("saccade.views") ?? "{}"), { "box:/srv/other": ["tasks"] });
});

// -- the rail's keep and forget -------------------------------------

async function railOver(declaration: string, shown: { at: string | null }) {
  const store = stubStore({ "saccade.projects": JSON.stringify(["box:/srv/proj"]) });
  stubWindowWithDeclaration(declaration);
  const opened: string[] = [];
  const said: string[] = [];
  const elsewhere: string[] = [];
  await theProjects().projects();
  const rail = createFrameRail({
    capture: () => ({ release: () => {} }),
    open: (spec) => opened.push(spec),
    shownProject: () => null,
    shownLocation: () => shown.at,
    elsewhere: (spec, act) => elsewhere.push(`${act} ${spec}`),
    say: (text) => said.push(text),
  });
  rail.start();
  await vi.waitFor(() => assert.deepEqual(rail.projectsNow().map((project) => project.name), ["proj"]));
  onTestFinished(() => rail.stop());
  return { rail, store, opened, said, elsewhere };
}

const viewTargets = (rail: Awaited<ReturnType<typeof railOver>>["rail"]) =>
  rail.targets().filter((target) => target.kind === "view");

test("Space k keeps the shown location once, says so, and says why when it cannot", async () => {
  const shown: { at: string | null } = { at: null };
  const { rail, store, said } = await railOver('name = "proj"\n', shown);
  rail.keepShown();
  assert.match(said.pop()!, /shows no location/);
  shown.at = "elsewhere/tasks";
  rail.keepShown();
  assert.match(said.pop()!, /belongs to no registered repository/);
  shown.at = "box:/some/file.md";
  rail.keepShown();
  assert.match(said.pop()!, /belongs to no registered repository/);

  shown.at = "proj/tasks?tag=app,ui";
  rail.keepShown();
  assert.deepEqual(said.pop(), "kept proj/tasks?tag=app,ui beneath proj");
  assert.deepEqual(JSON.parse(store.get("saccade.views") ?? "{}"), { "box:/srv/proj": ["tasks?tag=app,ui"] });
  rail.keepShown();
  assert.deepEqual(said.pop(), "already kept: proj/tasks?tag=app,ui");
  assert.equal(viewTargets(rail).length, 1, "keeping twice adds nothing");
  assert.deepEqual(rail.railViews("proj").entries.map((entry) => [entry.label, entry.shown]), [["app, ui", true]]);
  shown.at = "proj/tasks";
  assert.deepEqual(rail.railViews("proj").entries.map((entry) => entry.shown), [false], "the mark sits only on the exact location");
});

test("a view's acts: f opens here, o/i/w elsewhere, d forgets a device view; a file view has no d and says who offers it", async () => {
  const shown: { at: string | null } = { at: "proj/tasks?untagged" };
  const { rail, opened, elsewhere, store } = await railOver(
    'name = "proj"\nviews = ["tasks?tag=app", "tasks?tag="]\n',
    shown,
  );
  rail.keepShown();
  const targets = viewTargets(rail);
  assert.deepEqual(targets.map((target) => target.label), [
    "proj/tasks?tag=app (offered by workbench.toml)",
    "proj/tasks?untagged",
  ]);
  assert.ok(targets.every((target) => target.inside === "rail/project/proj"));
  const file = targets[0]!;
  const device = targets[1]!;
  assert.deepEqual(file.acts().map((act) => act.key), ["f", "o", "i", "w"]);
  assert.deepEqual(device.acts().map((act) => act.key), ["f", "o", "i", "w", "d"]);
  file.acts()[0]!.action.run();
  device.acts()[2]!.action.run();
  assert.deepEqual(opened, ["proj/tasks?tag=app"]);
  assert.deepEqual(elsewhere, ["once proj/tasks?untagged"]);
  device.acts()[4]!.action.run();
  assert.deepEqual(viewTargets(rail).map((target) => target.id), [file.id]);
  assert.deepEqual(JSON.parse(store.get("saccade.views") ?? "{}"), {});
  assert.deepEqual(rail.railViews("proj").notes.length, 1);
  assert.match(rail.railViews("proj").notes[0]!, /^workbench\.toml view 'tasks\?tag=' cannot be read: /);
});

test("a view the file offers and the device kept shows once, as the file's", async () => {
  const shown: { at: string | null } = { at: "proj/tasks?tag=app" };
  const { rail, said } = await railOver('name = "proj"\nviews = ["tasks?tag=app"]\n', shown);
  rail.keepShown();
  assert.deepEqual(said.pop(), "already kept: proj/tasks?tag=app");
  assert.equal(viewTargets(rail).length, 1);
});

// -- the key --------------------------------------------------------

test("the keep key is bound once, to the one action, in the window's table", () => {
  assert.deepEqual(WINDOW_CHORDS.keep.map((stroke) => stroke.key), [" ", "k"]);
  const ran: string[] = [];
  const host = createHostMatcher(
    windowContext({
      openPicker: () => {}, closeOverlays: () => {}, pickerIsOpen: () => false,
      trailBack: () => true, trailForward: () => true,
      linkByLabels: () => {}, unlink: () => {}, toggleRail: () => {}, registerProject: () => {},
      keepView: () => void ran.push("keep"), showHelp: () => {},
    }),
  );
  const press = (k: string) => host.stroke({ key: k, ctrl: false, shift: false, alt: false, meta: false });
  const first = press(" ");
  assert.equal(first.kind, "prefix");
  const second = press("k");
  assert.equal(second.kind, "invoke");
  assert.equal(second.action.id, "window.keep-view");
  host.apply(second);
  assert.deepEqual(ran, ["keep"]);
  const all = host.allView().entries.filter((entry) => entry.actionId === "window.keep-view");
  assert.deepEqual(all.map((entry) => entry.keys), ["spc k"]);
});
