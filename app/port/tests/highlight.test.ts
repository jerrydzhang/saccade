// Code highlighting (t-40): a fence's source becomes coloured runs, and
// the runs are presentation only - joined, they are the source, byte for
// byte, so copy and find keep reading the authored text.
import assert from "node:assert/strict";
import { test } from "vitest";
import { highlightCode, type CodeRun } from "../src/lib/highlight.ts";

const joined = (runs: readonly CodeRun[]): string => runs.map((run) => run.text).join("");
const kindsOf = (runs: readonly CodeRun[], text: string): string[] =>
  runs.filter((run) => run.text === text).flatMap((run) => run.kinds);

const RUST = 'fn main() {\n    // a comment\n    let n: u32 = 42;\n    println!("n < {} & \'x\'", n);\n}\n';

test("a known language's runs join back to the source exactly", () => {
  const runs = highlightCode(RUST, "rust");
  assert.ok(runs !== null);
  assert.equal(joined(runs), RUST);
});

test("keywords, strings, numbers, comments and titles come out as their kinds", () => {
  const runs = highlightCode(RUST, "rust")!;
  assert.ok(kindsOf(runs, "fn").includes("keyword"));
  assert.ok(kindsOf(runs, "let").includes("keyword"));
  assert.ok(kindsOf(runs, "42").includes("number"));
  assert.ok(kindsOf(runs, "// a comment").includes("comment"));
  assert.ok(kindsOf(runs, "main").includes("title"));
  assert.ok(runs.some((run) => run.kinds.includes("string") && run.text.includes("n < {} & 'x'")));
});

test("HTML-special characters stay text: nothing is markup", () => {
  const source = 'if a < b && c > d { s = "<b>&amp;</b>"; }';
  const runs = highlightCode(source, "rust")!;
  assert.equal(joined(runs), source);
});

test("the chosen set covers the board's usual languages, by name and alias", () => {
  for (const [lang, source] of [
    ["rust", "fn f() {}"],
    ["rs", "fn f() {}"],
    ["typescript", "const x: number = 1;"],
    ["ts", "const x: number = 1;"],
    ["javascript", "const x = 1;"],
    ["python", "def f():\n    return 1\n"],
    ["py", "def f():\n    return 1\n"],
    ["bash", "echo hi\n"],
    ["sh", "echo hi\n"],
    ["shell", "$ echo hi\n"],
    ["json", '{"a": 1}'],
    ["yaml", "a: 1\n"],
    ["yml", "a: 1\n"],
    ["toml", "a = 1\n"],
    ["nix", "{ pkgs }: pkgs.hello"],
    ["sql", "select 1;"],
    ["diff", "-a\n+b\n"],
    ["css", "a { color: red; }"],
    ["html", "<p>hi</p>"],
    ["markdown", "# hi\n"],
  ] as const) {
    const runs = highlightCode(source, lang);
    assert.ok(runs !== null, `${lang} is highlighted`);
    assert.equal(joined(runs), source, `${lang} keeps its source`);
  }
});

test("an absent or unknown language stays plain code (no guessing)", () => {
  assert.equal(highlightCode("fn main() {}", null), null);
  assert.equal(highlightCode("fn main() {}", undefined), null);
  assert.equal(highlightCode("fn main() {}", "no-such-language"), null);
  assert.equal(highlightCode("graph TD; a-->b", "mermaid"), null);
});
