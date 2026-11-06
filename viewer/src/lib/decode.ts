// YAML -> read model decoding over the provisional grammar. Structural
// failures are fatal errors; unknown keys inside known maps are surfaced as
// diagnostics and never silently dropped. Positions ride every block so the
// viewer can always point back at the authored bytes.

import { parseDocument, isMap, isSeq, isScalar, type Document, type Pair } from "yaml";
import type { ParsedNode } from "yaml";
import type { Diagnostic, SourceRange, StructuredIssue } from "./errors.ts";
import { issueAt } from "./errors.ts";
import type { Block, Board, FigureBlock, GroupBlock, IncorporationBlock } from "./model.ts";

const BLOCK_KEYS = new Set(["text", "group", "annotation", "incorporation", "figure"]);

export type DecodeResult =
  | { status: "ok"; board: Board }
  | { status: "invalid"; errors: StructuredIssue[]; diagnostics: Diagnostic[] };

export function offsetToPos(text: string, offset: number) {
  const clamped = Math.min(Math.max(offset, 0), text.length);
  let line = 1;
  let column = 1;
  for (let i = 0; i < clamped; i++) {
    if (text.charCodeAt(i) === 10) {
      line += 1;
      column = 1;
    } else {
      column += 1;
    }
  }
  return { line, column, offset: clamped };
}

function rangeOf(text: string, node: unknown): SourceRange | null {
  const range = (node as { range?: [number, number, ...number[]] } | null | undefined)?.range;
  if (!range || range.length < 2) return null;
  return {
    start: offsetToPos(text, range[0]),
    end: offsetToPos(text, range[1]),
  };
}

function rangeStartOf(error: unknown): number {
  const range = (error as { range?: [number, number, ...number[]] }).range;
  return range?.[0] ?? 0;
}

export function decodeBoard(text: string): DecodeResult {
  const doc: Document.Parsed = parseDocument(text, { uniqueKeys: true });
  const diagnostics: Diagnostic[] = [];
  const errors: StructuredIssue[] = [];

  for (const error of doc.errors) {
    const pos = offsetToPos(text, rangeStartOf(error));
    errors.push({
      code: error.code === "DUPLICATE_KEY" ? "duplicate-key" : "yaml-syntax",
      message: error.message,
      line: pos.line,
      column: pos.column,
    });
  }
  for (const warning of doc.warnings) {
    const pos = offsetToPos(text, rangeStartOf(warning));
    diagnostics.push({
      code: "yaml-syntax",
      message: warning.message,
      line: pos.line,
      column: pos.column,
    });
  }
  if (errors.length > 0) {
    return { status: "invalid", errors, diagnostics };
  }

  const root = doc.contents;
  if (!isMap(root)) {
    return {
      status: "invalid",
      errors: [
        issueAt(
          { code: "wrong-type", message: "the declaration must be a mapping" },
          rangeOf(text, root),
        ),
      ],
      diagnostics,
    };
  }

  let task: string | null = null;
  const content: Block[] = [];

  for (const pair of root.items as Pair[]) {
    const keyNode = pair.key;
    const key = isScalar(keyNode) && typeof keyNode.value === "string" ? keyNode.value : null;
    const keyRange = rangeOf(text, keyNode);
    if (key !== "task" && key !== "content") {
      diagnostics.push(
        issueAt(
          {
            code: "unknown-key",
            message: `unknown top-level key: ${JSON.stringify(String(key ?? keyNode))} (provisional grammar)`,
          },
          keyRange,
        ),
      );
      continue;
    }
    if (key === "task") {
      if (isScalar(pair.value) && (pair.value.value === null || typeof pair.value.value === "string")) {
        task = (pair.value.value as string | null) ?? null;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: "task must be a string (authored metadata only)", path: ["task"] },
            rangeOf(text, pair.value),
          ),
        );
      }
      continue;
    }
    // content
    if (!isSeq(pair.value)) {
      errors.push(
        issueAt(
          { code: "wrong-type", message: "content must be a list of blocks", path: ["content"] },
          rangeOf(text, pair.value),
        ),
      );
      continue;
    }
    const decodedContent: Block[] = [];
    pair.value.items.forEach((item, index) => {
      decodeBlock(text, item, ["content", index], decodedContent, errors, diagnostics);
    });
    content.push(...decodedContent);
  }

  if (content.length === 0) {
    errors.push(
      issueAt(
        { code: "empty-board", message: "the declaration carries no content blocks" },
        rangeOf(text, root),
      ),
    );
  }
  if (errors.length > 0) {
    return { status: "invalid", errors, diagnostics };
  }

  return { status: "ok", board: { task, content, diagnostics } };
}

function decodeBlock(
  text: string,
  node: unknown,
  path: (string | number)[],
  out: Block[],
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
): void {
  const range = rangeOf(text, node);
  if (!isMap(node)) {
    errors.push(
      issueAt({ code: "wrong-type", message: "each content item must be a mapping", path }, range),
    );
    return;
  }
  const pairs = node.items as Pair[];
  if (pairs.length !== 1) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `a content item must carry exactly one block key; found ${pairs.length}`,
          path,
        },
        range,
      ),
    );
    return;
  }
  const pair = pairs[0];
  if (!pair) {
    errors.push(
      issueAt({ code: "wrong-type", message: "an empty content item is not a block", path }, range),
    );
    return;
  }
  const keyNode = pair.key;
  const key = isScalar(keyNode) && typeof keyNode.value === "string" ? keyNode.value : null;
  const keyRange = rangeOf(text, keyNode);
  if (key === null || !BLOCK_KEYS.has(key)) {
    errors.push(
      issueAt(
        {
          code: "unknown-block",
          message: `unknown block kind: ${JSON.stringify(String(key ?? keyNode))}`,
          path,
        },
        keyRange ?? range,
      ),
    );
    return;
  }
  const valueRange = rangeOf(text, pair.value);
  const blockPath = [...path, key];

  if (key === "text" || key === "annotation") {
    if (!isScalar(pair.value) || typeof pair.value.value !== "string") {
      errors.push(
        issueAt({ code: "wrong-type", message: `${key} must be a string`, path: blockPath }, valueRange),
      );
      return;
    }
    const markdown = pair.value.value;
    out.push(
      key === "text"
        ? { kind: "text", markdown, pos: valueRange ?? range ?? zeroRange() }
        : { kind: "annotation", markdown, pos: valueRange ?? range ?? zeroRange() },
    );
    return;
  }

  if (key === "figure") {
    decodeFigure(text, pair, blockPath, out, errors, diagnostics);
    return;
  }

  if (key === "group") {
    const group: GroupBlock = {
      kind: "group",
      content: [],
      pos: valueRange ?? range ?? zeroRange(),
    };
    decodeGroupLike(text, pair.value, blockPath, group, errors, diagnostics, (title, relation) => {
      if (title !== undefined) group.title = title;
      if (relation !== undefined) group.relation = relation;
    });
    out.push(group);
    return;
  }

  // incorporation
  decodeIncorporation(text, pair, blockPath, out, errors, diagnostics);
}

function decodeFigure(
  text: string,
  pair: Pair,
  path: (string | number)[],
  out: Block[],
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
): void {
  const node = pair.value as ParsedNode | null;
  const range = rangeOf(text, node);
  if (!isMap(node)) {
    errors.push(
      issueAt({ code: "wrong-type", message: "figure must be a mapping with path and optional caption", path }, range),
    );
    return;
  }
  const figure: FigureBlock = { kind: "figure", path: "", pos: range ?? zeroRange() };
  for (const item of node.items as Pair[]) {
    const key = isScalar(item.key) && typeof item.key.value === "string" ? item.key.value : null;
    const keyRange = rangeOf(text, item.key);
    if (key === "path") {
      if (isScalar(item.value) && typeof item.value.value === "string") {
        figure.path = item.value.value;
      } else {
        errors.push(
          issueAt({ code: "wrong-type", message: "figure path must be a string", path: [...path, "path"] }, rangeOf(text, item.value)),
        );
      }
    } else if (key === "caption") {
      if (isScalar(item.value) && typeof item.value.value === "string") {
        figure.caption = item.value.value;
      } else {
        errors.push(
          issueAt({ code: "wrong-type", message: "figure caption must be a string", path: [...path, "caption"] }, rangeOf(text, item.value)),
        );
      }
    } else {
      diagnostics.push(
        issueAt(
          { code: "unknown-key", message: `unknown figure key: ${JSON.stringify(String(key ?? item.key))}`, path },
          keyRange,
        ),
      );
    }
  }
  if (figure.path === "") {
    errors.push(issueAt({ code: "missing-key", message: "figure requires a path", path }, range));
    return;
  }
  out.push(figure);
}

function decodeGroupLike(
  text: string,
  node: unknown,
  path: (string | number)[],
  container: { content: Block[] },
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
  setTitle: (title?: string, relation?: string) => void,
): void {
  if (!isMap(node)) {
    errors.push(issueAt({ code: "wrong-type", message: "group must be a mapping", path }, rangeOf(text, node)));
    return;
  }
  for (const item of node.items as Pair[]) {
    const key = isScalar(item.key) && typeof item.key.value === "string" ? item.key.value : null;
    const keyRange = rangeOf(text, item.key);
    if (key === "title" || key === "relation") {
      if (isScalar(item.value) && typeof item.value.value === "string") {
        if (key === "title") setTitle(item.value.value, undefined);
        else setTitle(undefined, item.value.value);
      } else {
        errors.push(
          issueAt({ code: "wrong-type", message: `group ${key} must be a string`, path: [...path, key] }, rangeOf(text, item.value)),
        );
      }
    } else if (key === "content") {
      if (!isSeq(item.value)) {
        errors.push(
          issueAt({ code: "wrong-type", message: "group content must be a list", path: [...path, "content"] }, rangeOf(text, item.value)),
        );
        continue;
      }
      item.value.items.forEach((child, index) => {
        decodeBlock(text, child, [...path, "content", index], container.content, errors, diagnostics);
      });
    } else {
      diagnostics.push(
        issueAt(
          { code: "unknown-key", message: `unknown group key: ${JSON.stringify(String(key ?? item.key))}`, path },
          keyRange,
        ),
      );
    }
  }
}

function decodeIncorporation(
  text: string,
  pair: Pair,
  path: (string | number)[],
  out: Block[],
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
): void {
  const node = pair.value as ParsedNode | null;
  const range = rangeOf(text, node);
  if (!isMap(node)) {
    errors.push(
      issueAt(
        { code: "wrong-type", message: "incorporation must be a mapping with id, source and content", path },
        range,
      ),
    );
    return;
  }
  const inc: IncorporationBlock = {
    kind: "incorporation",
    id: "",
    source: { task: "", revision: "" },
    content: [],
    pos: range ?? zeroRange(),
  };
  let sawSource = false;
  let sawContent = false;
  for (const item of node.items as Pair[]) {
    const key = isScalar(item.key) && typeof item.key.value === "string" ? item.key.value : null;
    const keyRange = rangeOf(text, item.key);
    if (key === "id") {
      if (isScalar(item.value) && typeof item.value.value === "string") {
        inc.id = item.value.value;
      } else {
        errors.push(issueAt({ code: "wrong-type", message: "incorporation id must be a string", path: [...path, "id"] }, rangeOf(text, item.value)));
      }
    } else if (key === "source") {
      sawSource = true;
      decodeIncorporationSource(text, item.value, [...path, "source"], inc, errors, diagnostics);
    } else if (key === "content") {
      sawContent = true;
      if (isSeq(item.value)) {
        item.value.items.forEach((child, index) => {
          decodeBlock(text, child, [...path, "content", index], inc.content, errors, diagnostics);
        });
      } else {
        errors.push(
          issueAt({ code: "wrong-type", message: "incorporation content must be a list", path: [...path, "content"] }, rangeOf(text, item.value)),
        );
      }
    } else {
      diagnostics.push(
        issueAt(
          { code: "unknown-key", message: `unknown incorporation key: ${JSON.stringify(String(key ?? item.key))}`, path },
          keyRange,
        ),
      );
    }
  }
  if (!sawSource) {
    errors.push(issueAt({ code: "missing-key", message: "incorporation requires source", path }, range));
  }
  if (!sawContent) {
    errors.push(issueAt({ code: "missing-key", message: "incorporation requires content", path }, range));
  }
  if (inc.id === "") {
    errors.push(issueAt({ code: "missing-key", message: "incorporation requires id", path }, range));
  }
  // The block still enters the model: the viewer renders whatever decoded
  // cleanly and the error list carries the rest; invalid input never
  // silently disappears.
  out.push(inc);
}

function decodeIncorporationSource(
  text: string,
  node: unknown,
  path: (string | number)[],
  inc: IncorporationBlock,
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
): void {
  if (!isMap(node)) {
    errors.push(
      issueAt({ code: "wrong-type", message: "incorporation source must be a mapping", path }, rangeOf(text, node)),
    );
    return;
  }
  for (const item of node.items as Pair[]) {
    const key = isScalar(item.key) && typeof item.key.value === "string" ? item.key.value : null;
    const keyRange = rangeOf(text, item.key);
    if (key === "task" || key === "revision") {
      if (isScalar(item.value) && typeof item.value.value === "string") {
        if (key === "task") inc.source.task = item.value.value;
        else inc.source.revision = item.value.value;
      } else {
        errors.push(
          issueAt({ code: "wrong-type", message: `source ${key} must be a string`, path: [...path, key] }, rangeOf(text, item.value)),
        );
      }
    } else if (key === "select") {
      if (isSeq(item.value)) {
        const parts: string[] = [];
        item.value.items.forEach((part, index) => {
          if (isScalar(part) && typeof part.value === "string") {
            parts.push(part.value);
          } else {
            errors.push(
              issueAt(
                { code: "wrong-type", message: "select entries must be strings", path: [...path, "select", index] },
                rangeOf(text, part),
              ),
            );
          }
        });
        inc.source.select = parts;
      } else {
        errors.push(
          issueAt({ code: "wrong-type", message: "select must be a list of path segments", path: [...path, "select"] }, rangeOf(text, item.value)),
        );
      }
    } else {
      diagnostics.push(
        issueAt(
          { code: "unknown-key", message: `unknown incorporation source key: ${JSON.stringify(String(key ?? item.key))}`, path },
          keyRange,
        ),
      );
    }
  }
}

function zeroRange(): SourceRange {
  const start = { line: 0, column: 0, offset: 0 };
  return { start, end: { ...start } };
}
