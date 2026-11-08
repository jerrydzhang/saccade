import test from "node:test";
import assert from "node:assert/strict";
import { resolveRelative } from "../src/lib/resolve.ts";

test("a relative path joins the declaration directory", () => {
  const result = resolveRelative("/boards/first", "assets/fig.svg");
  assert.deepEqual(result, { ok: true, joined: "/boards/first/assets/fig.svg" });
});

test("dot segments are dropped", () => {
  const result = resolveRelative("/boards/first", "./assets/./fig.svg");
  assert.ok(result.ok);
  assert.equal(result.ok ? result.joined : "", "/boards/first/assets/fig.svg");
});

test("parent traversal is denied", () => {
  for (const attempt of ["../secret", "assets/../../secret", "a/../.."]) {
    const result = resolveRelative("/boards/first", attempt);
    assert.ok(!result.ok, attempt);
  }
});

test("absolute paths are denied", () => {
  assert.ok(!resolveRelative("/boards/first", "/etc/passwd").ok);
  assert.ok(!resolveRelative("/boards/first", "C:/etc/passwd").ok);
});

test("backslash separators are denied outright", () => {
  assert.ok(!resolveRelative("/boards/first", "assets\\fig.svg").ok);
  assert.ok(!resolveRelative("/boards/first", "..\\secret").ok);
});

test("NUL bytes are denied", () => {
  assert.ok(!resolveRelative("/boards/first", "a\u0000b").ok);
});

test("unicode paths resolve untouched: data stays data", () => {
  const result = resolveRelative("/boards/first", "assets/\u56fe\u7247/\u03b7.svg");
  assert.ok(result.ok);
  assert.equal(result.ok ? result.joined : "", "/boards/first/assets/\u56fe\u7247/\u03b7.svg");
});

test("empty and segment-free paths are denied", () => {
  assert.ok(!resolveRelative("/boards/first", "").ok);
  assert.ok(!resolveRelative("/boards/first", "./").ok);
});

test("an empty declaration base resolves repo-relative: no leading slash", () => {
  const result = resolveRelative("", "figures/error-render.png");
  assert.ok(result.ok);
  assert.equal(result.ok ? result.joined : "", "figures/error-render.png");
  const nested = resolveRelative("", "a/b/c.png");
  assert.ok(nested.ok);
  assert.equal(nested.ok ? nested.joined : "", "a/b/c.png");
});
