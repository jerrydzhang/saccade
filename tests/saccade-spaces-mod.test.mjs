import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { freshId, workspaceOf, plan } from "../mods/saccade/plan.js";

const ROOT = "/home/jerry/Projects/saccade-workspaces";

describe("workspaceOf", () => {
  it("names the first path segment under the root", () => {
    assert.equal(
      workspaceOf(`${ROOT}/floop-a1b2/src/x.rs`, ROOT),
      "floop-a1b2",
    );
    assert.equal(workspaceOf(`${ROOT}/migrate-floop`, ROOT), "migrate-floop");
  });

  it("rejects everything outside the root", () => {
    assert.equal(workspaceOf("/home/jerry/Projects/saccade/src/x.rs", ROOT), null);
    assert.equal(workspaceOf(`${ROOT}-elsewhere/floop/file`, ROOT), null);
    assert.equal(workspaceOf(ROOT, ROOT), null);
    assert.equal(workspaceOf(null, ROOT), null);
  });
});

describe("freshId", () => {
  it("mints the app's own id shape: kind prefix plus 32 hex", () => {
    for (const kind of ["spa", "tb"]) {
      const id = freshId(kind);
      assert.match(id, new RegExp(`^${kind}[0-9a-f]{32}$`));
    }
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
  const space = (id, folder, name, named = false) => ({ id, folder, name, named });

  it("labels nothing for terminals outside the root, already labeled, or dead", () => {
    const rows = [
      live("t1", "/home/jerry/Projects/saccade/src/x.rs"),
      live("t2", `${ROOT}/floop-a1b2`, { "herdr.space": "spa1" }),
      { address: "t3", cwd: `${ROOT}/gone/x`, labels: {}, state: "finished" },
      { address: "t4", cwd: null, labels: {}, state: "live" },
    ];
    assert.deepEqual(plan(rows, [], {}, ROOT), []);
  });

  it("labels each live terminal with its workspace's one stable space label", () => {
    const rows = [
      live("t1", `${ROOT}/floop-a1b2`),
      live("t2", `${ROOT}/floop-a1b2/build`),
      live("t3", `${ROOT}/migrate-floop`),
    ];
    const jobs = plan(rows, [], {}, ROOT);
    const labels = jobs.filter((j) => j.kind === "label");
    const maps = jobs.filter((j) => j.kind === "map");
    assert.equal(maps.length, 2);
    assert.equal(labels.length, 3);
    const floop = maps.find((m) => m.label === "floop-a1b2").entry;
    for (const job of labels.filter((j) => j.terminal !== "t3")) {
      assert.equal(job.space, floop.spaceLabel);
      // no space stands yet: a fresh tab id per terminal
      assert.match(job.tab, /^tb[0-9a-f]{32}$/);
    }
    assert.notEqual(
      labels.find((j) => j.terminal === "t1").tab,
      labels.find((j) => j.terminal === "t2").tab,
    );
  });

  it("once the space stands, new terminals label with its own id and first tab", () => {
    const rows = [live("t5", `${ROOT}/floop-a1b2`)];
    const map = {
      "floop-a1b2": { spaceLabel: "spaMinted", name: null },
    };
    const spaces_ = [
      {
        id: "spaReal",
        folder: `${ROOT}/floop-a1b2`,
        name: "floop-a1b2",
        named: true,
        tabs: [{ id: "tbReal" }],
      },
    ];
    const jobs = plan(rows, spaces_, map, ROOT);
    assert.deepEqual(
      jobs.find((j) => j.kind === "label"),
      { kind: "label", terminal: "t5", space: "spaReal", tab: "tbReal" },
    );
    assert.equal(map["floop-a1b2"].realId, "spaReal");
    // the id the app knows is the durable key: it survives restarts,
    // because the doc carries it
    const later = plan([live("t6", `${ROOT}/floop-a1b2`)], spaces_, map, ROOT);
    assert.equal(later.find((j) => j.kind === "label").space, "spaReal");
  });

  it("renames an unnamed space under the root to its label; a named space keeps its name", () => {
    const rows = [];
    const map = {
      "floop-a1b2": { spaceLabel: "spaStable", name: null },
    };
    // the app's own derived name on an unnamed space: ours to set
    const unnamed = [space("spaApp", `${ROOT}/floop-a1b2`, "spaces-accept-1")];
    assert.deepEqual(
      plan(rows, unnamed, map, ROOT).find((j) => j.kind === "rename"),
      { kind: "rename", id: "spaApp", name: "floop-a1b2", label: "floop-a1b2" },
    );
    const correct = [space("spaApp", `${ROOT}/floop-a1b2`, "floop-a1b2", true)];
    assert.equal(
      plan(rows, correct, map, ROOT).some((j) => j.kind === "rename"),
      false,
      "our own correct name stays",
    );
    const human = [space("spaHuman", `${ROOT}/floop-a1b2`, "Jerry's floop", true)];
    assert.equal(
      plan(rows, human, map, ROOT).some((j) => j.kind === "rename"),
      false,
      "a human rename is never stomped",
    );
  });
});
