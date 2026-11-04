import { describe, it } from "node:test";
import assert from "node:assert/strict";

// A synthetic state root: the rule under test is pure string logic, and
// nothing here may depend on the machine the tests run on.
import {
  freshId,
  machineOf,
  localId,
  parseRemote,
  workspaceOf,
  plan,
} from "../mods/saccade/plan.js";

const ROOT = "/state/saccade";
const WS = (repo, label) => `${ROOT}/${repo}/worktrees/${label}`;

describe("workspaceOf", () => {
  it("names the segment after worktrees, across every project", () => {
    assert.equal(workspaceOf(`${WS("repo-alpha", "gate-demo")}/src/x.rs`, ROOT), "gate-demo");
    assert.equal(workspaceOf(WS("repo-beta", "some-ws"), ROOT), "some-ws");
    assert.equal(workspaceOf(WS("repo-alpha", "gate-demo-two"), ROOT), "gate-demo-two");
  });

  it("rejects state-root content that is not a provisioned worktree", () => {
    assert.equal(workspaceOf(`${ROOT}/herdr-core-dev`, ROOT), null);
    assert.equal(workspaceOf(`${ROOT}/repo-alpha/development`, ROOT), null);
    assert.equal(workspaceOf(ROOT, ROOT), null);
  });

  it("rejects everything outside the state root", () => {
    assert.equal(workspaceOf("/home/someone/Projects/repo/src/x.rs", ROOT), null);
    assert.equal(workspaceOf("/srv/worktrees/gate-demo", ROOT), null);
    assert.equal(workspaceOf(null, ROOT), null);
  });

  it("uses the first worktrees component, whatever sits deeper", () => {
    assert.equal(
      workspaceOf(`${WS("repo-alpha", "gate-demo")}/build/worktrees`, ROOT),
      "gate-demo",
    );
  });

  it("follows a moved state home: the rule is the segment, not the path", () => {
    assert.equal(
      workspaceOf("/data/state/saccade/repo-alpha/worktrees/moved-ws", "/data/state/saccade"),
      "moved-ws",
    );
  });
});

describe("machineOf and localId", () => {
  it("splits the app's machine-prefixed remote addresses", () => {
    assert.equal(machineOf("box/t3b"), "box");
    assert.equal(localId("box/t3b"), "t3b");
    assert.equal(machineOf("t3b"), null);
    assert.equal(localId("t3b"), "t3b");
  });
});

describe("parseRemote", () => {
  it("prefixes a machine's own ids with the machine", () => {
    const rows = parseRemote(
      JSON.stringify({ terminals: [{ address: "t3f", cwd: "/x", state: "live", labels: {} }] }),
      "box",
    );
    assert.equal(rows[0].address, "box/t3f");
    assert.equal(rows[0].cwd, "/x");
    assert.equal(machineOf(rows[0].address), "box");
  });

  it("refuses an unreadable table whole", () => {
    assert.throws(() => parseRemote("not json", "box"), /unreadable/);
    assert.throws(() => parseRemote(JSON.stringify({ nope: 1 }), "box"), /names no rows/);
  });
});

describe("freshId", () => {
  it("mints the app's own id shape: kind prefix plus 32 hex", () => {
    const id = freshId("spa");
    assert.match(id, /^spa[0-9a-f]{32}$/);
    assert.notEqual(freshId("spa"), freshId("spa"));
  });
});

describe("plan", () => {
  const live = (address, cwd, labels = {}) => ({
    address,
    cwd,
    labels,
    state: "live",
  });
  const space = (id, folder, name, machine = null, named = false) => ({
    id,
    folder,
    name,
    machine,
    named,
  });

  it("partitions by machine: a box terminal labels on the box with the prefix stripped", () => {
    const rows = [
      live("t3b", WS("repo-alpha", "gate-demo")),
      live("box/t3d", WS("repo-beta", "gate-demo-two")),
    ];
    const jobs = plan(rows, [], {}, ROOT);
    const local = jobs.find((j) => j.terminal === "t3b");
    const remote = jobs.find((j) => j.terminal === "box/t3d");
    assert.equal(local.kind, "label");
    assert.equal(local.machine, null);
    assert.equal(local.localId, "t3b");
    assert.equal(remote.machine, "box");
    assert.equal(remote.localId, "t3d");
    // different machines, same label: separate map entries, separate ids
    const keys = jobs.filter((j) => j.kind === "map").map((j) => j.key);
    assert.deepEqual(keys.sort(), ["box::gate-demo-two", "local::gate-demo"]);
  });

  it("labels nothing for terminals outside the roots, already labeled, or dead", () => {
    const rows = [
      live("t1", "/home/someone/Projects/repo/src/x.rs"),
      live("t2", WS("repo-alpha", "floop-a1b2"), { "herdr.space": "spa1" }),
      { address: "box/t3", cwd: `${ROOT}/gone/x`, labels: {}, state: "finished" },
    ];
    assert.deepEqual(plan(rows, [], {}, ROOT), []);
  });

  it("once the space stands on a machine, new terminals label with its own id and first tab", () => {
    const rows = [live("box/t5", WS("repo-alpha", "gate-demo"))];
    const map = {
      "box::gate-demo": { spaceLabel: "spaMinted", name: null },
    };
    const spaces_ = [
      {
        id: "spaReal",
        folder: WS("repo-alpha", "gate-demo"),
        name: "gate-demo",
        machine: "box",
        named: true,
        tabs: [{ id: "tbReal" }],
      },
    ];
    const jobs = plan(rows, spaces_, map, ROOT);
    assert.deepEqual(
      jobs.find((j) => j.kind === "label"),
      { kind: "label", terminal: "box/t5", localId: "t5", machine: "box", space: "spaReal", tab: "tbReal" },
    );
    assert.equal(map["box::gate-demo"].realId, "spaReal");
  });

  it("renames an unnamed space under a root to its label, per machine", () => {
    const rows = [];
    const map = {
      "box::gate-demo": { spaceLabel: "spaStable", name: null },
    };
    const unnamed = [
      space("spaApp", WS("repo-alpha", "gate-demo"), "gate-demo-something", "box"),
    ];
    assert.deepEqual(
      plan(rows, unnamed, map, ROOT).find((j) => j.kind === "rename"),
      { kind: "rename", id: "spaApp", name: "gate-demo", key: "box::gate-demo" },
    );
    const named = [space("spaApp", WS("repo-alpha", "gate-demo"), "gate-demo", "box", true)];
    assert.equal(
      plan(rows, named, map, ROOT).some((j) => j.kind === "rename"),
      false,
      "our own correct name stays",
    );
    const human = [space("spaApp", WS("repo-alpha", "gate-demo"), "Jerry's gate", "box", true)];
    assert.equal(
      plan(rows, human, map, ROOT).some((j) => j.kind === "rename"),
      false,
      "a human rename is never stomped",
    );
    // a space on another machine with a folder outside the roots never matches
    const elsewhere = [space("spaOther", "/srv/other/gate-demo", "", "box")];
    assert.equal(plan(rows, elsewhere, map, ROOT).some((j) => j.kind === "rename"), false);
  });
});
