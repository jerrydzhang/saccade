// The board menu wiring (chunk C + the reference acts, t-36@r90): one
// action object behind every invocation path, close-before-execute,
// fold as the first real act, part identity riding the target VALUE
// OBJECT (no kind:label side map - duplicate labels cannot collide),
// and the citation's reference acts: 'f' follow here first (so a
// plain click follows), 'y' copy reference with the prototype-parity
// payload. Pure tests - no DOM, no sources.

import assert from "node:assert/strict";
import { test } from "vitest";
import { createBoardMenu, actsFor, type ReferencePorts } from "../src/lib/menu/acts.ts";
import { createMenuSession } from "../src/lib/menu/session.ts";
import type { Action } from "../src/lib/contracts/input.ts";
import type { InputContext } from "../src/lib/contracts/input.ts";

test("groups carry fold/unfold; other kinds carry nothing yet", () => {
  const folded = new Set<string>(["/content/1"]);
  const acts = actsFor({ kind: "group", label: "the group", part: "/content/1" }, {
    isFolded: (part) => folded.has(part),
    toggleFold: (part) => { folded.delete(part); },
    referenceOf: () => null,
    referencePorts: () => null,
    referenceMachine: () => null,
    partCitation: () => null,
  });
  assert.equal(acts.length, 1);
  assert.equal(acts[0]?.key, "z");
  assert.equal(acts[0]?.action.label, "unfold");
  // F5: a text block carries y copy citation when its citation and
  // the ports exist; without them it stays act-less (no menu)
  const textActs = actsFor({ kind: "text", label: "prose", part: "/content/0" }, {
    isFolded: () => false,
    toggleFold: () => {},
    referenceOf: () => null,
    referencePorts: () => ({ copy: () => {}, open: () => {}, figureEnlarge: () => {} }),
    referenceMachine: () => null,
    partCitation: () => "t-9@r4:/content/0",
  });
  assert.equal(textActs.length, 1);
  assert.equal(textActs[0]?.key, "y");
  assert.equal(textActs[0]?.action.label, "copy citation");
  assert.deepEqual(actsFor({ kind: "text", label: "prose", part: "/content/0" }, {
    isFolded: () => false,
    toggleFold: () => {},
    referenceOf: () => null,
    referencePorts: () => null,
    referenceMachine: () => null,
    partCitation: () => null,
  }), []);
  assert.deepEqual(actsFor({ kind: "group", label: "g", part: null }, {
    isFolded: () => false,
    toggleFold: () => {},
    referenceOf: () => null,
    referencePorts: () => null,
    referenceMachine: () => null,
    partCitation: () => null,
  }), []);
});

test("the menu session: four invocation paths run the SAME action object; close before execute", () => {
  const journal: string[] = [];
  const action: Action = { id: "board.fold", label: "fold", run: () => { journal.push("run"); } };
  const session = createMenuSession(() => [{ key: "z", action }]);
  const target = { kind: "group", label: "g", part: "/content/1" };
  // path 1: selection opens; Enter fires the first act
  session.openFromSelection(target);
  assert.deepEqual(session.key("Enter"), { kind: "act", index: 0, action });
  // path 2: right-click opens; Enter fires
  session.openAtPointer(target, { x: 1, y: 1 });
  assert.deepEqual(session.key("Enter"), { kind: "act", index: 0, action });
  // path 3: right-click opens; the act item is clicked (choose)
  session.openAtPointer(target, { x: 1, y: 1 });
  assert.deepEqual(session.choose(0), { kind: "act", index: 0, action });
  // path 4: the declared act key fires while open
  session.openFromSelection(target);
  assert.deepEqual(session.key("z"), { kind: "act", index: 0, action });
  assert.equal(journal.length, 4);
  // close-before-execute ordering: the close event precedes the act event
  const closeIndex = session.journal().findIndex((e) => e.kind === "close" && e.reason === "act");
  const actIndex = session.journal().findIndex((e) => e.kind === "act");
  assert.ok(closeIndex >= 0 && actIndex >= 0 && closeIndex < actIndex);
  // escape dismisses without acting; an unrelated key is consumed and
  // leaves the menu open (no universal close rule)
  session.openFromSelection(target);
  assert.deepEqual(session.key("Escape"), { kind: "dismissed" });
  session.openFromSelection(target);
  assert.deepEqual(session.key("q"), { kind: "ignored" });
  assert.equal(session.isOpen(), true);
  session.outsidePointerDown();
  assert.equal(session.isOpen(), false);
  assert.equal(journal.length, 4, "dismissals never act");
});

test("part identity rides the target value object: two same-kind same-label targets resolve their OWN parts", () => {
  // the r90 sequencing constraint: the kind:label side map is gone;
  // a stale map entry can never fold or follow the wrong part
  const folded = new Set<string>();
  const parts: string[] = [];
  const menu = createBoardMenu({
    isFolded: (part) => folded.has(part),
    toggleFold: (part) => { parts.push(part); folded.add(part); },
    referenceOf: () => null,
    referencePorts: () => null,
    referenceMachine: () => null,
    partCitation: () => null,
    capture: () => ({ release: () => {} }),
  });
  // two groups with IDENTICAL labels open back to back: the second
  // open's acts fold the SECOND part
  menu.openFor({ part: "/content/0", kind: "group", label: "same" });
  menu.openFor({ part: "/content/2", kind: "group", label: "same" });
  const outcome = menu.session.key("z");
  assert.equal(outcome.kind, "act");
  assert.deepEqual(parts, ["/content/2"]);
});

function menuHarness(folded: Set<string>) {
  const events: string[] = [];
  let readingNow: InputContext | null = null;
  const menu = createBoardMenu({
    isFolded: (part) => folded.has(part),
    toggleFold: (part) => {
      if (folded.has(part)) folded.delete(part);
      else folded.add(part);
      events.push(`fold:${part}`);
    },
    referenceOf: () => null,
    referencePorts: () => null,
    referenceMachine: () => null,
    partCitation: () => null,
    capture: (context) => {
      events.push("capture");
      readingNow = context;
      return { release: () => {
        events.push("release", "restore");
        readingNow = null; // the host lease owns the return
      } };
    },
  });
  return { menu, events, context: () => readingNow };
}

test("the board menu host: selection opens for groups, capture/restore ride the lease, z folds", () => {
  const folded = new Set<string>();
  const { menu, events, context } = menuHarness(folded);
  // F5: a text part OPENS with its copy act (the ruled primary);
  // an act-less kind (no citation composed) still opens nothing
  menu.openFor({ part: "/content/0", kind: "text", label: "prose" });
  assert.deepEqual(events, [], "no ports injected: nothing opens");
  assert.equal(context(), null);
  // a group opens, captures, and the declared act key folds it
  menu.openFor({ part: "/content/1", kind: "group", label: "the group" });
  assert.ok(menu.session.isOpen());
  assert.notEqual(context(), null, "the menu context installed");
  const outcome = menu.session.key("z");
  assert.equal(outcome.kind, "act");
  assert.deepEqual(events, ["capture", "release", "restore", "fold:/content/1"]);
  assert.deepEqual(folded, new Set(["/content/1"]));
  assert.equal(menu.session.isOpen(), false, "acting closes the menu");
});

// -- the reference acts (t-36@r90) --------------------------------------

function referenceHarness(options?: {
  source?: { task: string; revision: number; select: readonly string[] | null } | null;
  machine?: string | null;
  ports?: ReferencePorts | null;
}) {
  const opened: { spec: string; part: string | null }[] = [];
  const copied: string[] = [];
  const ports: ReferencePorts | null =
    options?.ports === undefined
      ? {
          open: (spec, part) => { opened.push({ spec, part }); },
          copy: (payload) => { copied.push(payload); },
        }
      : options.ports;
  const machine = options?.machine === undefined ? "tracker:seedtracker" : options.machine;
  const source =
    options?.source === undefined
      ? { task: "t-0", revision: 1, select: ["/content/0/group/content/1"] as readonly string[] }
      : options.source;
  const deps = {
    isFolded: () => false,
    toggleFold: () => {},
    referenceOf: () => source,
    referencePorts: () => ports,
    referenceMachine: () => machine,
    partCitation: (part: string) => `t-0@r1:${part}`,
  };
  return { deps, opened, copied };
}

test("a citation's first act follows here through the parseable location form, anchored at the select", () => {
  const { deps, opened } = referenceHarness();
  const acts = actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, deps);
  assert.equal(acts.length, 2);
  assert.equal(acts[0]?.key, "f");
  assert.equal(acts[0]?.action.label, "follow here");
  assert.equal(acts[1]?.key, "y");
  assert.equal(acts[1]?.action.label, "copy reference");
  // the follow composes the tracker location form (name/spec - never
  // a colon join, the 5a44c71 finding) and anchors at the canonical
  // part (the grammar select mapped)
  acts[0]?.action.run();
  assert.deepEqual(opened, [{ spec: "seedtracker/t-0@r1", part: "/content/0/content/1" }]);
});

test("the copy payload is the prototype parity encoding: part path when the citation names one, bare otherwise", () => {
  const { deps, copied } = referenceHarness();
  actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, deps)[1]?.action.run();
  assert.deepEqual(copied, ["t-0@r1:/content/0/content/1"]);
  const bare = referenceHarness({ source: { task: "t-3", revision: 7, select: null } });
  actsFor({ kind: "reference", label: "t-3@r7", part: "/content/0" }, bare.deps)[1]?.action.run();
  assert.deepEqual(bare.copied, ["t-3@r7"]);
});

test("a group's own citation copies the SHOWN revision plus the group's full path", () => {
  const copied: string[] = [];
  const deps = {
    isFolded: () => false,
    toggleFold: () => {},
    referenceOf: () => null,
    referencePorts: () => ({ open: () => {}, copy: (payload: string) => { copied.push(payload); } }),
    referenceMachine: () => null,
    partCitation: (part: string) => `t-0@r1:${part}`,
  };
  const acts = actsFor({ kind: "group", label: "Recorded setup", part: "/content/0" }, deps);
  assert.equal(acts.length, 2);
  assert.equal(acts[0]?.key, "z");
  assert.equal(acts[1]?.key, "y");
  assert.equal(acts[1]?.action.label, "copy citation");
  acts[1]?.action.run();
  assert.deepEqual(copied, ["t-0@r1:/content/0"]);
  // without a citable shown revision or ports, no copy act is offered
  assert.equal(
    actsFor({ kind: "group", label: "g", part: "/content/0" }, { ...deps, partCitation: () => null }).length,
    1,
  );
  assert.equal(
    actsFor({ kind: "group", label: "g", part: "/content/0" }, { ...deps, referencePorts: () => null }).length,
    1,
  );
});

test("no reference acts without the model source, the window ports, or the machine", () => {
  assert.deepEqual(
    actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, referenceHarness({ source: null }).deps),
    [],
  );
  assert.deepEqual(
    actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, referenceHarness({ ports: null }).deps),
    [],
  );
  assert.deepEqual(
    actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, referenceHarness({ machine: null }).deps),
    [],
  );
});

test("a plain click on the citation runs the same ACT as the menu paths (id and behavior; each open builds its list)", () => {
  const { deps, opened } = referenceHarness();
  const menu = createBoardMenu({ ...deps, capture: () => ({ release: () => {} }) });
  const target = { kind: "reference", label: "t-0@r1", part: "/content/1" };
  // every open rebuilds the act list (the prototype's select()
  // re-calls KINDS, inst16 :58) - object identity is per-open and
  // pinned by the session test's constant act; AGREEMENT across the
  // four paths is id, label, and behavior
  menu.openFor(target);
  const viaEnter = menu.session.key("Enter");
  assert.equal(viaEnter.kind, "act");
  menu.openFor(target);
  const viaKey = menu.session.key("f");
  assert.equal(viaKey.kind, "act");
  menu.openFor(target);
  const viaChoose = menu.session.choose(0);
  assert.equal(viaChoose.kind, "act");
  // the plain click path - primaryClick builds its own list, exactly
  // as the prototype's primary() re-calls KINDS (inst16 :69)
  const viaPrimary = menu.session.primaryClick(target);
  assert.equal(viaPrimary.kind, "act");
  if (viaEnter.kind === "act") {
    for (const outcome of [viaKey, viaChoose, viaPrimary]) {
      if (outcome.kind !== "act") assert.fail("an invocation path did not act");
      else {
        assert.equal(outcome.action.id, viaEnter.action.id);
        assert.equal(outcome.action.label, viaEnter.action.label);
      }
    }
  }
  assert.equal(opened.length, 4, "all four invocation paths followed here");
  for (const entry of opened) {
    assert.deepEqual(entry, { spec: "seedtracker/t-0@r1", part: "/content/0/content/1" });
  }
});

// -- the OPEN acts (S18): o/i/w ride the window's elsewhere port ------

test("with the elsewhere port the citation carries the OPEN acts in the KINDS order: f, o, i, w, y", () => {
  const { deps } = referenceHarness({
    ports: {
      open: () => {},
      copy: () => {},
      elsewhere: () => {},
    },
  });
  const acts = actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, deps);
  assert.deepEqual(
    acts.map((act) => act.key),
    ["f", "o", "i", "w", "y"],
  );
  assert.equal(acts[1]?.action.label, "open at the link");
  assert.equal(acts[2]?.action.label, "open elsewhere once");
  assert.equal(acts[3]?.action.label, "open in a new viewer");
});

test("the OPEN acts deliver the composed spec with their act; without the port they never appear", () => {
  const elsewhere: { spec: string; act: string }[] = [];
  const withPort = referenceHarness({
    ports: {
      open: () => {},
      copy: () => {},
      elsewhere: (spec, act) => { elsewhere.push({ spec, act }); },
    },
  });
  const acts = actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, withPort.deps);
  acts[1]?.action.run();
  acts[2]?.action.run();
  acts[3]?.action.run();
  assert.deepEqual(elsewhere, [
    { spec: "seedtracker/t-0@r1", act: "link" },
    { spec: "seedtracker/t-0@r1", act: "once" },
    { spec: "seedtracker/t-0@r1", act: "new" },
  ]);
  // the unbound window (no state yet) offers follow and copy only -
  // the same honest absence as the ports-null case
  const without = referenceHarness();
  const bare = actsFor({ kind: "reference", label: "t-0@r1", part: "/content/1" }, without.deps);
  assert.deepEqual(
    bare.map((act) => act.key),
    ["f", "y"],
  );
});

// -- the figure acts (S04): x enlarges first, y copies, a/o wait ------

test("a figure's acts: x enlarge first (the primary a plain click runs), then y copy citation; a/o wait for their surfaces", () => {
  const copied: string[] = [];
  const enlarged: string[] = [];
  const { deps } = referenceHarness({
    ports: {
      open: () => {},
      copy: (payload) => { copied.push(payload); },
      figureEnlarge: (part) => { enlarged.push(part); },
    },
  });
  const acts = actsFor({ kind: "figure", label: "t-0@r1:/content/0", part: "/content/0" }, deps);
  assert.deepEqual(
    acts.map((act) => act.key),
    ["x", "y"],
  );
  assert.equal(acts[0]?.action.label, "enlarge");
  acts[0]?.action.run();
  acts[1]?.action.run();
  assert.deepEqual(enlarged, ["/content/0"]);
  assert.deepEqual(copied, ["t-0@r1:/content/0"]);
  // no ports, no acts - the honest absence
  const bare = referenceHarness({ ports: null });
  assert.deepEqual(
    actsFor({ kind: "figure", label: "x", part: "/content/0" }, bare.deps),
    [],
  );
});
