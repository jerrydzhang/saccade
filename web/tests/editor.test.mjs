import test from "node:test";
import assert from "node:assert/strict";
import { EditorState } from "@codemirror/state";
import { markdown } from "@codemirror/lang-markdown";
import { sections } from "../src/editor.mjs";

test("Markdown section folds follow document headings and stop at peers", () => {
  const raw = "# Overview\nIntroduction.\n\n## Details\nImplementation.\n\n### Edge cases\nLimits.\n\n## Next\nContinuation.\n\n# Appendix\nSources.\n";
  const state = EditorState.create({ doc: raw, extensions: [markdown()] });
  const folds = sections(state);
  assert.deepEqual(folds.map((section) => section.level), [1, 2, 3, 2, 1]);
  assert.equal(folds[0].to, raw.indexOf("# Appendix") - 1);
  assert.equal(folds[1].to, raw.indexOf("## Next") - 1);
  assert.equal(folds[2].to, folds[1].to);
  const data = "Paragraph.\n\n- a list\n- another item\n\n```md\n# Not a section\n```\n\n> # Quoted heading\n> Quoted prose.\n\nMain section\n============\nBody.\n\nSubsection\n----------\nDetails.\n";
  const other = EditorState.create({ doc: data, extensions: [markdown()] });
  assert.deepEqual(sections(other).map((section) => section.level), [1, 2]);
  assert.equal(sections(other)[0].start, data.indexOf("Main section"));
  assert.equal(sections(EditorState.create({ doc: "# Empty", extensions: [markdown()] })).length, 0);
});
