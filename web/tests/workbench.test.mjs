import test from "node:test";
import assert from "node:assert/strict";
import { Workbench, commandFor, confirmsLanded, createTaskCommand, confirmsTaskCreated, launchRequest, launchOutcome } from "../src/workbench.mjs";

function workbench() {
  const bench = new Workbench();
  bench.register({ id: "home", kind: "home", text: "Find work" });
  bench.register({ id: "comment:19", kind: "comment", text: "Keep the draft cursor while choosing a citation." });
  bench.register({ id: "draft:cursor", kind: "draft", home: 0, text: "We should preserve the cursor." });
  bench.open("home");
  return bench;
}

test("buffer switching restores its position and preserves draft content", () => {
  const bench = workbench();
  bench.open("draft:cursor");
  bench.remember({ anchor: 9, head: 9, scrollTop: 50 });
  bench.open("comment:19");
  bench.remember({ anchor: 20, head: 20, scrollTop: 100 });
  bench.open("draft:cursor");
  assert.deepEqual(bench.position(), { anchor: 9, head: 9, scrollTop: 50 });
  assert.equal(bench.buffer("draft:cursor").text, "We should preserve the cursor.");
  bench.open("comment:19");
  assert.equal(bench.position().head, 20);
});

test("two panes share a buffer but retain independent positions", () => {
  const bench = workbench();
  bench.open("comment:19");
  bench.remember({ anchor: 4, head: 4, scrollTop: 20 });
  const first = bench.focus;
  bench.split("row");
  const second = bench.focus;
  bench.remember({ anchor: 25, head: 25, scrollTop: 120 });
  assert.equal(bench.view(first).bufferId, bench.view(second).bufferId);
  assert.deepEqual(bench.position(first), { anchor: 4, head: 4, scrollTop: 20 });
  assert.deepEqual(bench.position(second), { anchor: 25, head: 25, scrollTop: 120 });
});

test("swapping panes moves their document and position without replacement", () => {
  const bench = workbench();
  bench.open("draft:cursor");
  bench.remember({ anchor: 9, head: 9, scrollTop: 50 });
  const first = bench.focus;
  bench.split("row", "comment:19");
  const second = bench.focus;
  bench.remember({ anchor: 20, head: 20, scrollTop: 100 });
  bench.swapPanes(first, second);
  assert.deepEqual(bench.layout.children, [{ view: second }, { view: first }]);
  assert.equal(bench.view(first).bufferId, "draft:cursor");
  assert.equal(bench.view(second).bufferId, "comment:19");
  assert.equal(bench.position(first).head, 9);
  assert.equal(bench.position(second).head, 20);
  assert.equal(bench.focus, first);
});

test("closing a pane keeps its buffers and a position for reopening", () => {
  const bench = workbench();
  bench.split("column", "draft:cursor");
  bench.remember({ anchor: 7, head: 7, scrollTop: 40 });
  assert.equal(bench.closePane(), true);
  bench.split("row", "draft:cursor");
  assert.deepEqual(bench.position(), { anchor: 7, head: 7, scrollTop: 40 });
  assert.equal(bench.buffers.size, 3);
  assert.equal(bench.buffer("draft:cursor").text, "We should preserve the cursor.");
});

test("nested row and column splits collapse without losing the surviving layout", () => {
  const bench = workbench();
  const first = bench.focus;
  bench.split("row", "comment:19");
  const second = bench.focus;
  bench.split("column", "draft:cursor");
  const third = bench.focus;
  bench.closePane(second);
  assert.deepEqual(bench.layout, {
    axis: "row", ratio: 0.5, children: [{ view: first }, { view: third }],
  });
  assert.equal(bench.closePane(first), true);
  assert.deepEqual(bench.layout, { view: third });
  assert.equal(bench.closePane(third), false);
});

test("showing only one pane does not discard other buffers or their positions", () => {
  const bench = workbench();
  bench.split("row", "comment:19");
  bench.remember({ anchor: 11, head: 11, scrollTop: 90 });
  bench.split("column", "draft:cursor");
  bench.onlyPane();
  assert.equal(bench.views.size, 1);
  assert.equal(bench.buffers.size, 3);
  bench.open("comment:19");
  assert.equal(bench.position().head, 11);
});

test("citation insertion uses the saved draft position and appends one mapping", () => {
  const bench = workbench();
  const ticket = bench.beginCitation("draft:cursor", 9);
  const result = bench.insertCitation(ticket, 19);
  assert.equal(result.buffer.text, "We should[^1] preserve the cursor.");
  assert.deepEqual(result.buffer.references, [19]);
  assert.equal(result.cursor, 13);
});

test("citation insertion reuses existing numbering without normalizing prior mappings", () => {
  const bench = workbench();
  const draft = bench.buffer("draft:cursor");
  draft.references = [7, 19, 19, 23];
  const result = bench.insertCitation(bench.beginCitation(draft.id, 0), 19);
  assert.equal(result.marker, "[^2]");
  assert.deepEqual(result.buffer.references, [7, 19, 19, 23]);
});

test("cancelling citation selection makes no draft or mapping change", () => {
  const bench = workbench();
  const before = structuredClone(bench.buffer("draft:cursor"));
  bench.beginCitation("draft:cursor", 4);
  assert.deepEqual(bench.buffer("draft:cursor"), before);
});

test("a changed draft refuses a stale citation insertion without adding a mapping", () => {
  const bench = workbench();
  const ticket = bench.beginCitation("draft:cursor", 9);
  bench.edit("draft:cursor", "Keep this correction.");
  assert.throws(() => bench.insertCitation(ticket, 19), /draft changed/);
  assert.equal(bench.buffer("draft:cursor").text, "Keep this correction.");
  assert.deepEqual(bench.buffer("draft:cursor").references, []);
});

test("published buffers refuse edits and citation insertion", () => {
  const bench = workbench();
  assert.throws(() => bench.edit("comment:19", "rewrite history"), /read-only/);
  assert.throws(() => bench.beginCitation("comment:19", 0), /needs a draft/);
  assert.equal(bench.buffer("comment:19").text, "Keep the draft cursor while choosing a citation.");
});

test("citation references and positions refuse invalid numbers before mutation", () => {
  const bench = workbench();
  const before = structuredClone(bench.buffer("draft:cursor"));
  for (const position of [-1, 0.5, 100]) {
    assert.throws(() => bench.beginCitation("draft:cursor", position), /outside/);
  }
  for (const reference of [-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => bench.insertCitation(bench.beginCitation("draft:cursor", 2), reference), /birth-record/);
  }
  assert.deepEqual(bench.buffer("draft:cursor"), before);
});

test("draft Unicode remains unchanged around inserted citation markers", () => {
  const bench = workbench();
  bench.edit("draft:cursor", "\u03bb and \ud83d\udc41 stay data.");
  bench.insertCitation(bench.beginCitation("draft:cursor", 1), 0);
  assert.equal(bench.buffer("draft:cursor").text, "\u03bb[^1] and \ud83d\udc41 stay data.");
  assert.deepEqual(bench.buffer("draft:cursor").references, [0]);
});


test("a landed revision confirms against the surface's own baseline, not a live pending", () => {
  const surface = revisionSurface();
  const landed = { kind: "comment_revised", actor: "jerry", tier: "human",
    payload: { id: 11, body: "the corrected body" } };
  assert.equal(confirmsLanded(surface, structuredClone(landed)), true);
  surface.buffer.revisionPending = null;
  assert.equal(confirmsLanded(surface, structuredClone(landed)), true,
    "a repair discarded mid-flight still confirms its own sent bytes");
});

function rejectSurface() {
  const buffer = { id: "draft:grounds", kind: "draft", text: "the edge case still fails",
    eventual: { kind: "reject", offer: "c-2", offerSeq: 7 }, submission: null };
  return { kind: "reject", buffer, title: null, body: buffer.text,
    references: [], home: 0, target: "c-2",
    get submission() { return buffer.submission; }, set submission(value) { buffer.submission = value; } };
}

test("a reject command reads the stored offer birth, never a source on the grounds draft", () => {
  const command = commandFor(rejectSurface(), "jerry");
  assert.deepEqual(command.comment.kind.reject, { offer: 7, grounds: "the edge case still fails" });
});

test("a landed reject confirms the same offer birth and the exact grounds bytes", () => {
  const surface = rejectSurface();
  const landed = { kind: "commented", actor: "jerry", tier: "human",
    payload: { home: 0, kind: { reject: { offer: 7, grounds: "the edge case still fails" } } } };
  assert.equal(confirmsLanded(surface, landed), true);
  const otherOffer = structuredClone(landed);
  otherOffer.payload.kind.reject.offer = 9;
  assert.equal(confirmsLanded(surface, otherOffer), false);
});

function revisionSurface() {
  const buffer = { id: "c-0", kind: "comment", text: "the corrected body",
    source: { seq: 11, kind: "note", title: "The original title", body: "the original body" },
    revisionPending: {
      title: "The original title", savedTitle: "the locally saved title",
      savedBody: "the locally saved body",
      originalTitle: "The original title", originalBody: "the original body",
      submission: null,
    } };
  return { kind: "revision", buffer, title: buffer.revisionPending.title, body: buffer.text,
    references: null, home: 0, target: "c-0",
    originalTitle: buffer.revisionPending.originalTitle, originalBody: buffer.revisionPending.originalBody,
    get submission() { return buffer.revisionPending?.submission; }, set submission(value) { if (buffer.revisionPending) buffer.revisionPending.submission = value; } };
}

test("the revision patch diffs against the original record bytes, not the local savepoint", () => {
  const command = commandFor(revisionSurface(), "jerry");
  assert.deepEqual(command.revise_comment, { id: 11, body: "the corrected body" },
    "an untouched title sends no title field even after :w moved the savepoint");
});

test("saving locally never changes the revision patch intent", () => {
  const surface = revisionSurface();
  surface.buffer.revisionPending.savedTitle = surface.buffer.revisionPending.title;
  surface.buffer.revisionPending.savedBody = surface.buffer.text;
  const command = commandFor(surface, "jerry");
  assert.deepEqual(command.revise_comment, { id: 11, body: "the corrected body" });
});

test("a landed revision confirms exactly the fields the patch sent", () => {
  const surface = revisionSurface();
  const landed = { kind: "comment_revised", actor: "jerry", tier: "human",
    payload: { id: 11, body: "the corrected body" } };
  assert.equal(confirmsLanded(surface, landed), true);
  const extraField = structuredClone(landed);
  extraField.payload.title = "a concurrent title lands unbothered";
  assert.equal(confirmsLanded(surface, extraField), false,
    "a title the patch never sent is not claimed by the confirmation");
});

test("task creation rides the human write door and checks its own landed record", () => {
  assert.deepEqual(createTaskCommand("The browser-authored task"),
    { create_task: { name: "The browser-authored task" } });
  const landed = { kind: "task_created", actor: "jerry", tier: "human",
    payload: { name: "The browser-authored task" } };
  assert.equal(confirmsTaskCreated(landed, "The browser-authored task"), true);
  assert.equal(confirmsTaskCreated(landed, "Another name"), false,
    "a different name is not claimed as this creation");
  assert.equal(confirmsTaskCreated({ kind: "commented", payload: landed.payload }, "The browser-authored task"), false);
});

test("the launch request names one explicit existing workspace and nothing else", () => {
  assert.deepEqual(launchRequest("/repo/wt/workspace-abc"), { workspace: { existing: "/repo/wt/workspace-abc" } });
});

test("a lost launch reply is unknown and a server answer is a confirmed fact", () => {
  assert.deepEqual(launchOutcome(new Error("Failed to fetch")), { kind: "unknown", detail: "Failed to fetch" });
  const answered = Object.assign(new Error("refused"), { data: { error: { code: "launch_host_refused", detail: "the host refused the create" } } });
  assert.deepEqual(launchOutcome(answered), { kind: "answered", code: "launch_host_refused", detail: "the host refused the create" });
  const shaped = Object.assign(new Error("refused"), { data: { error: { code: "launch_host_no_handle", detail: { text: "created" } } } });
  assert.equal(launchOutcome(shaped).code, "launch_host_no_handle");
  assert.equal(launchOutcome(shaped).detail, '{"text":"created"}', "a non-string detail still travels as the server's own fact");
});
