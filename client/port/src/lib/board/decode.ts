// The presentation decoder for board grammar 1, mirroring the
// board-grammar crate contract one for one: same five kinds, same
// fields, same refusals, same version handling. Admission itself is the
// crate's job on the Tauri side (the validate_board bridge command);
// this module turns accepted text into positioned render blocks and is
// the only verdict in the browser lane, which has no bridge. One named
// divergence: a component kind this build does not know renders as its
// raw source beside a diagnostic, where the crate refuses; parity over
// the shared fixture corpus is pinned by tests on both sides.

import {
  parseAllDocuments,
  isMap,
  isSeq,
  isScalar,
  isAlias,
  type Document,
  type Pair,
  type ParsedNode,
} from "yaml";
import type { Diagnostic, SourceRange, StructuredIssue } from "./errors.ts";
import { issueAt } from "./errors.ts";
import type {
  Block,
  Board,
  FigureBlock,
  GroupBlock,
  IncorporationBlock,
  UnknownBlock,
} from "./model.ts";

/** The grammar versions this viewer decodes. The crate's
 * SUPPORTED_GRAMMAR_VERSIONS is the authority; the equality of the two
 * ranges is a tested claim on both sides, over shared fixtures. */
export const SUPPORTED_GRAMMAR_VERSIONS: readonly number[] = [1];

/** Deepest group or incorporation nesting decode accepts, matching the
 * crate's MAX_NESTING_DEPTH. */
export const MAX_NESTING_DEPTH = 100;

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

/** The component's authored bytes, for the raw-source rendering of an
 * unknown kind. */
function rawSource(text: string, range: SourceRange | null): string {
  if (!range) return "";
  return text.slice(range.start.offset, range.end.offset).trimEnd();
}

function zeroRange(): SourceRange {
  const start = { line: 0, column: 0, offset: 0 };
  return { start, end: { ...start } };
}

export function decodeBoard(text: string): DecodeResult {
  const errors: StructuredIssue[] = [];
  const diagnostics: Diagnostic[] = [];

  const docs = parseAllDocuments(text, { uniqueKeys: true });
  if (docs.length > 1) {
    return {
      status: "invalid",
      errors: [
        issueAt(
          {
            code: "multiple-documents",
            message: "the file holds several YAML documents; keep exactly one board in the file",
          },
          rangeOf(text, docs[1]?.contents),
        ),
      ],
      diagnostics,
    };
  }
  const doc: Document.Parsed | undefined = docs[0];
  if (!doc) {
    return {
      status: "invalid",
      errors: [
        issueAt({
          code: "yaml-syntax",
          message:
            "the board text holds no YAML document; write the board mapping with grammar, task and content",
        }),
      ],
      diagnostics,
    };
  }
  for (const error of doc.errors) {
    const pos = offsetToPos(text, error.pos[0] ?? 0);
    errors.push({
      code: error.code === "DUPLICATE_KEY" ? "duplicate-key" : "yaml-syntax",
      message: error.message,
      line: pos.line,
      column: pos.column,
    });
  }
  for (const warning of doc.warnings) {
    const pos = offsetToPos(text, warning.pos[0] ?? 0);
    diagnostics.push({
      code: "yaml-syntax",
      message: warning.message,
      line: pos.line,
      column: pos.column,
    });
  }
  if (errors.length > 0) return { status: "invalid", errors, diagnostics };

  // The crate refuses anchors and aliases in the event stream, before
  // the tree is read; the mirror refuses them the same way.
  const anchor = findAnchorOrAlias(doc.contents);
  if (anchor !== null) {
    errors.push(
      issueAt(
        {
          code: "anchor-alias",
          message: `the board uses a YAML ${anchor}; write the content out in full instead`,
        },
        rangeOf(text, doc.contents),
      ),
    );
    return { status: "invalid", errors, diagnostics };
  }

  const root = doc.contents;
  if (!isMap(root)) {
    return {
      status: "invalid",
      errors: [
        issueAt(
          {
            code: "wrong-type",
            message: `the board is a mapping with grammar, task and content at the top; found ${describe(root)}; restructure the file so those three fields sit at the top level`,
          },
          rangeOf(text, root),
        ),
      ],
      diagnostics,
    };
  }

  let grammarPair: Pair | null = null;
  let taskPair: Pair | null = null;
  let contentPair: Pair | null = null;
  const unknown: string[] = [];
  for (const pair of root.items as Pair[]) {
    const name = scalarKey(pair.key);
    if (name === null) {
      errors.push(
        issueAt(
          {
            code: "wrong-type",
            message: `board field names are plain text like 'grammar'; found the key ${describe(pair.key)}; rename it`,
          },
          rangeOf(text, pair.key),
        ),
      );
      continue;
    }
    if (name === "grammar") grammarPair = pair;
    else if (name === "task") taskPair = pair;
    else if (name === "content") contentPair = pair;
    else unknown.push(name);
  }
  if (unknown.length > 0) {
    errors.push(
      issueAt(
        {
          code: "unknown-key",
          message: `unknown field '${unknown.join("', '")}' at the board top; the board mapping takes only grammar, task and content; move that content into a component or drop the field`,
        },
        rangeOf(text, root),
      ),
    );
  }
  if (!grammarPair) {
    errors.push(
      issueAt(
        {
          code: "missing-key",
          message: "every board declares its grammar version; add 'grammar: 1' to the top mapping",
        },
        rangeOf(text, root),
      ),
    );
  } else {
    checkGrammar(text, grammarPair, errors);
  }
  if (!taskPair) {
    errors.push(
      issueAt(
        {
          code: "missing-key",
          message: "the board does not name its task; add 'task: <task name>' to the top mapping",
        },
        rangeOf(text, root),
      ),
    );
  } else if (!isScalar(taskPair.value) || typeof taskPair.value.value !== "string") {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the board's 'task' names the task as t-N; found ${describe(taskPair.value)}`,
          path: ["task"],
        },
        rangeOf(text, taskPair.value),
      ),
    );
  } else if (!/^t\-(0|[1-9][0-9]*)$/.test(taskPair.value.value)) {
    errors.push(
      issueAt(
        {
          code: "bad-task",
          message: `the board's 'task' is '${taskPair.value.value}'; name the task as t-N, the tracker's own address`,
          path: ["task"],
        },
        rangeOf(text, taskPair.value),
      ),
    );
  }
  if (!contentPair) {
    errors.push(
      issueAt(
        {
          code: "missing-key",
          message: "the board has no 'content' list; add one holding the board's components",
        },
        rangeOf(text, root),
      ),
    );
  } else if (!isSeq(contentPair.value)) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the board's 'content' is a list of components; found ${describe(contentPair.value)}; write 'content:' with one component per list item`,
          path: ["content"],
        },
        rangeOf(text, contentPair.value),
      ),
    );
  }
  if (errors.length > 0) return { status: "invalid", errors, diagnostics };

  const grammar = Number((grammarPair!.value as { value: number }).value);
  const task = (taskPair!.value as { value: string }).value;
  const content: Block[] = [];
  (contentPair!.value as { items: ParsedNode[] }).items.forEach((item, index) => {
    decodeComponent(text, item, `/content/${index}`, ["content", index], 1, content, errors, diagnostics);
  });
  if (errors.length > 0) return { status: "invalid", errors, diagnostics };

  return { status: "ok", board: { grammar, task, content, diagnostics } };
}

function versionsText(): string {
  if (SUPPORTED_GRAMMAR_VERSIONS.length === 1) {
    return `version ${SUPPORTED_GRAMMAR_VERSIONS[0]}`;
  }
  return `versions ${SUPPORTED_GRAMMAR_VERSIONS[0]} through ${SUPPORTED_GRAMMAR_VERSIONS[SUPPORTED_GRAMMAR_VERSIONS.length - 1]}`;
}

function checkGrammar(text: string, pair: Pair, errors: StructuredIssue[]): void {
  const raw = scalarRaw(text, pair.value);
  const supported = versionsText();
  if (raw === "1" && SUPPORTED_GRAMMAR_VERSIONS.includes(1)) {
    return;
  }
  const description = describe(pair.value);
  if (isScalar(pair.value) && typeof pair.value.value === "number" && Number.isInteger(pair.value.value)) {
    errors.push(
      issueAt(
        {
          code: "grammar-version",
          message: `grammar ${pair.value.value} is outside the versions this viewer renders (${supported}); write the board in grammar ${SUPPORTED_GRAMMAR_VERSIONS[0]}`,
          path: ["grammar"],
        },
        rangeOf(text, pair.value),
      ),
    );
    return;
  }
  errors.push(
    issueAt(
      {
        code: "grammar-version",
        message: `grammar must be the integer ${SUPPORTED_GRAMMAR_VERSIONS[0]}; found ${description}; change the line to 'grammar: ${SUPPORTED_GRAMMAR_VERSIONS[0]}'`,
        path: ["grammar"],
      },
      rangeOf(text, pair.value),
    ),
  );
}

/** The authored bytes of a scalar, so `grammar: 1.0` cannot pass as 1. */
function scalarRaw(text: string, node: unknown): string {
  const range = rangeOf(text, node);
  if (!range) return "";
  return text.slice(range.start.offset, range.end.offset).trim();
}

// None means no anchor or alias anywhere in the tree.
function findAnchorOrAlias(node: unknown): "alias" | "anchor" | null {
  if (node === null || typeof node !== "object") return null;
  if (isAlias(node)) return "alias";
  if (isScalar(node)) {
    return typeof (node as { anchor?: unknown }).anchor === "string" &&
      ((node as { anchor: string }).anchor.length > 0)
      ? "anchor"
      : null;
  }
  if (isMap(node)) {
    if (typeof (node as { anchor?: unknown }).anchor === "string" && (node as { anchor: string }).anchor.length > 0) {
      return "anchor";
    }
    for (const pair of (node as { items: Pair[] }).items) {
      const found = findAnchorOrAlias(pair.key) ?? findAnchorOrAlias(pair.value);
      if (found) return found;
    }
    return null;
  }
  if (isSeq(node)) {
    if (typeof (node as { anchor?: unknown }).anchor === "string" && (node as { anchor: string }).anchor.length > 0) {
      return "anchor";
    }
    for (const item of (node as { items: ParsedNode[] }).items) {
      const found = findAnchorOrAlias(item);
      if (found) return found;
    }
  }
  return null;
}

function scalarKey(node: unknown): string | null {
  return isScalar(node) && typeof node.value === "string" ? node.value : null;
}

function describe(value: unknown): string {
  if (value === null || value === undefined) return "nothing (an empty value)";
  if (isScalar(value)) {
    const scalar = value as { value: unknown };
    switch (typeof scalar.value) {
      case "string":
        return `the text "${scalar.value}"`;
      case "number":
        return Number.isInteger(scalar.value) ? "a number" : "a decimal number";
      case "boolean":
        return "a boolean";
      case "bigint":
        return "a number";
      default:
        return "an unreadable value";
    }
  }
  if (isMap(value)) return "a mapping";
  if (isSeq(value)) return "a list";
  if (isAlias(value)) return "a YAML alias";
  return "an unreadable value";
}

function decodeComponent(
  text: string,
  node: unknown,
  at: string,
  path: (string | number)[],
  depth: number,
  out: Block[],
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
): void {
  const range = rangeOf(text, node);
  if (depth > MAX_NESTING_DEPTH) {
    errors.push(
      issueAt(
        {
          code: "too-deep",
          message: `the components nest more than ${MAX_NESTING_DEPTH} levels deep at ${at}; flatten the nesting so a reader can follow the board`,
          path,
        },
        range,
      ),
    );
    return;
  }
  if (!isMap(node)) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the component at ${at} is a mapping naming exactly one kind; found ${describe(node)}; write one of text, group, figure, incorporation, annotation`,
          path,
        },
        range,
      ),
    );
    return;
  }
  const named: Array<[string, unknown]> = [];
  for (const pair of (node as { items: Pair[] }).items) {
    const name = scalarKey(pair.key);
    if (name === null) {
      errors.push(
        issueAt(
          {
            code: "wrong-type",
            message: `component field names are the kind names as plain text; found the key ${describe(pair.key)} at ${at}`,
            path,
          },
          rangeOf(text, pair.key),
        ),
      );
      continue;
    }
    named.push([name, pair.value]);
  }
  const kindRange = rangeOf(text, (node as { items: Pair[] }).items[0]?.key) ?? range;
  if (named.length === 0) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the component at ${at} names no kind; write exactly one of text, group, figure, incorporation, annotation`,
          path,
        },
        range,
      ),
    );
    return;
  }
  if (named.length > 1) {
    const listed = named.map(([name]) => `'${name}'`).join(", ");
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the component at ${at} carries several kinds (${listed}); a component is exactly one kind; split them into separate list items`,
          path,
        },
        range,
      ),
    );
    return;
  }
  const [name, body] = named[0]!;
  const bodyPath = [...path, name];
  const bodyRange = rangeOf(text, body);

  if (name === "text" || name === "annotation") {
    if (!isScalar(body) || typeof body.value !== "string") {
      errors.push(
        issueAt(
          {
            code: "wrong-type",
            message: `the ${name} at ${at} is ${name === "text" ? "markdown as text" : "the concern as markdown text"}; found ${describe(body)}`,
            path: bodyPath,
          },
          bodyRange,
        ),
      );
      return;
    }
    out.push(
      name === "text"
        ? { kind: "text", markdown: body.value, pos: bodyRange ?? range ?? zeroRange() }
        : { kind: "annotation", markdown: body.value, pos: bodyRange ?? range ?? zeroRange() },
    );
    return;
  }

  if (name === "group") {
    decodeGroup(text, body, at, bodyPath, depth, out, errors, diagnostics, bodyRange ?? range);
    return;
  }
  if (name === "figure") {
    decodeFigure(text, body, at, bodyPath, out, errors, bodyRange ?? range);
    return;
  }
  if (name === "incorporation") {
    decodeIncorporation(text, body, at, bodyPath, depth, out, errors, diagnostics, bodyRange ?? range);
    return;
  }

  // The one deliberate divergence from the crate: a kind this build does
  // not know (a newer minor grammar's kind) renders as its raw source
  // with a diagnostic instead of refusing the board.
  const unknownBlock: UnknownBlock = {
    kind: "unknown",
    name,
    source: rawSource(text, range),
    pos: range ?? zeroRange(),
  };
  out.push(unknownBlock);
  diagnostics.push(
    issueAt(
      {
        code: "unknown-kind",
        message: `unknown component kind '${name}' at ${at}; the known kinds are text, group, figure, incorporation, annotation; it renders below as its raw source`,
        path,
      },
      kindRange,
    ),
  );
}

function decodeGroup(
  text: string,
  body: unknown,
  at: string,
  path: (string | number)[],
  depth: number,
  out: Block[],
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
  range: SourceRange | null,
): void {
  if (!isMap(body)) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the group at ${at} is a mapping with optional 'title' and a 'content' list; found ${describe(body)}`,
          path,
        },
        rangeOf(text, body),
      ),
    );
    return;
  }
  const group: GroupBlock = { kind: "group", content: [], pos: range ?? zeroRange() };
  let content: ParsedNode[] | null = null;
  const unknown: string[] = [];
  for (const pair of (body as { items: Pair[] }).items) {
    const key = scalarKey(pair.key);
    if (key === "title") {
      if (isScalar(pair.value) && typeof pair.value.value === "string") {
        group.title = pair.value.value;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: `the group title at ${at} is text; found ${describe(pair.value)}`, path: [...path, "title"] },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === "content") {
      if (isSeq(pair.value)) {
        content = (pair.value as { items: ParsedNode[] }).items;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: `the group's 'content' at ${at} is a list of components; found ${describe(pair.value)}`, path: [...path, "content"] },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === null) {
      errors.push(
        issueAt(
          { code: "wrong-type", message: `group field names are plain text; found the key ${describe(pair.key)} at ${at}`, path },
          rangeOf(text, pair.key),
        ),
      );
    } else {
      unknown.push(key);
    }
  }
  if (unknown.length > 0) {
    errors.push(
      issueAt(
        {
          code: "unknown-key",
          message: `the group at ${at} takes optional 'title' and 'content' only; drop the field '${unknown.join("', '")}'`,
          path,
        },
        rangeOf(text, body),
      ),
    );
  }
  if (content === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the group at ${at} needs a 'content' list; add the components it groups`, path },
        rangeOf(text, body),
      ),
    );
    return;
  }
  content.forEach((child, index) => {
    decodeComponent(text, child, `${at}/group/content/${index}`, [...path, "content", index], depth + 1, group.content, errors, diagnostics);
  });
  out.push(group);
}

function decodeFigure(
  text: string,
  body: unknown,
  at: string,
  path: (string | number)[],
  out: Block[],
  errors: StructuredIssue[],
  range: SourceRange | null,
): void {
  if (!isMap(body)) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the figure at ${at} is a mapping with 'input' and 'caption'; found ${describe(body)}`,
          path,
        },
        rangeOf(text, body),
      ),
    );
    return;
  }
  let input: string | null = null;
  let caption: string | null = null;
  const unknown: string[] = [];
  for (const pair of (body as { items: Pair[] }).items) {
    const key = scalarKey(pair.key);
    if (key === "input") {
      if (isScalar(pair.value) && typeof pair.value.value === "string") {
        input = pair.value.value;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: `the figure input at ${at} is a relative path as text; found ${describe(pair.value)}`, path: [...path, "input"] },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === "caption") {
      if (isScalar(pair.value) && typeof pair.value.value === "string") {
        caption = pair.value.value;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: `the figure caption at ${at} is text; found ${describe(pair.value)}`, path: [...path, "caption"] },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === null) {
      errors.push(
        issueAt(
          { code: "wrong-type", message: `figure field names are plain text; found the key ${describe(pair.key)} at ${at}`, path },
          rangeOf(text, pair.key),
        ),
      );
    } else {
      unknown.push(key);
    }
  }
  if (unknown.length > 0) {
    errors.push(
      issueAt(
        {
          code: "unknown-key",
          message: `the figure at ${at} takes only 'input' and 'caption'; drop the field '${unknown.join("', '")}'`,
          path,
        },
        rangeOf(text, body),
      ),
    );
  }
  if (input === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the figure at ${at} needs 'input', the relative path of the file inside the board tree`, path },
        rangeOf(text, body),
      ),
    );
  } else {
    checkFigurePath(text, input, at, path, errors);
  }
  if (caption === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the figure at ${at} needs a 'caption'; write a line saying what the figure shows`, path },
        rangeOf(text, body),
      ),
    );
  }
  if (errors.length > 0) return;
  const figure: FigureBlock = {
    kind: "figure",
    input: input!,
    caption: caption!,
    pos: range ?? zeroRange(),
  };
  out.push(figure);
}

function checkFigurePath(
  text: string,
  input: string,
  at: string,
  path: (string | number)[],
  errors: StructuredIssue[],
): void {
  const bad = (message: string, node?: unknown) =>
    errors.push(
      issueAt({ code: "bad-figure-path", message, path }, rangeOf(text, node) ?? null),
    );
  if (input.length === 0) {
    bad(`the figure at ${at} names no file; give 'input' a relative path inside the board tree, like 'figures/plinth-section.png'`);
  } else if (input.startsWith("/")) {
    bad(`the figure at ${at} points at '${input}'; figure inputs are relative paths inside the board tree, like 'figures/plinth-section.png'`);
  } else if (input.split("/").some((segment) => segment === "..")) {
    bad(`the figure at ${at} walks out of the board tree with '..'; reference a file inside the board tree, like 'figures/plinth-section.png'`);
  }
}

function decodeIncorporation(
  text: string,
  body: unknown,
  at: string,
  path: (string | number)[],
  depth: number,
  out: Block[],
  errors: StructuredIssue[],
  diagnostics: Diagnostic[],
  range: SourceRange | null,
): void {
  if (!isMap(body)) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the incorporation at ${at} is a mapping with 'source' and 'content'; found ${describe(body)}`,
          path,
        },
        rangeOf(text, body),
      ),
    );
    return;
  }
  const incorporation: IncorporationBlock = {
    kind: "incorporation",
    source: { task: "", revision: 0, select: [] },
    content: [],
    pos: range ?? zeroRange(),
  };
  let source: ParsedNode | null = null;
  let content: ParsedNode[] | null = null;
  let sawContent = false;
  const unknown: string[] = [];
  for (const pair of (body as { items: Pair[] }).items) {
    const key = scalarKey(pair.key);
    if (key === "source") {
      source = pair.value as ParsedNode;
    } else if (key === "content") {
      sawContent = true;
      if (isSeq(pair.value)) {
        content = (pair.value as { items: ParsedNode[] }).items;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: `the incorporation's 'content' at ${at} is a list of components; found ${describe(pair.value)}`, path: [...path, "content"] },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === null) {
      errors.push(
        issueAt(
          { code: "wrong-type", message: `incorporation field names are plain text; found the key ${describe(pair.key)} at ${at}`, path },
          rangeOf(text, pair.key),
        ),
      );
    } else {
      unknown.push(key);
    }
  }
  if (unknown.length > 0) {
    errors.push(
      issueAt(
        {
          code: "unknown-key",
          message: `the incorporation at ${at} takes only 'source' and 'content'; drop the field '${unknown.join("', '")}'`,
          path,
        },
        rangeOf(text, body),
      ),
    );
  }
  if (source === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the incorporation at ${at} cites no source; add 'source' with task, revision and select`, path },
        rangeOf(text, body),
      ),
    );
  } else {
    decodeSource(text, source, at, [...path, "source"], incorporation.source, errors);
  }
  if (!sawContent) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the incorporation at ${at} needs a 'content' list; add the text receiving what you take from the source`, path },
        rangeOf(text, body),
      ),
    );
  }
  if (errors.length > 0) return;
  content!.forEach((child, index) => {
    decodeComponent(text, child, `${at}/incorporation/content/${index}`, [...path, "content", index], depth + 1, incorporation.content, errors, diagnostics);
  });
  if (errors.length > 0) return;
  if (!hasSubstantiveText(incorporation.content)) {
    errors.push(
      issueAt(
        {
          code: "no-receiving-text",
          message: `incorporation at ${at} has no text component; add a text explaining what you take from the source`,
          path,
        },
        rangeOf(text, body),
      ),
    );
    return;
  }
  out.push(incorporation);
}

// the receiving text is the author's own prose: content inside a nested
// incorporation belongs to that incorporation's own source and does not count
function hasSubstantiveText(content: Block[]): boolean {
  for (const component of content) {
    if (component.kind === "text" && component.markdown.trim().length > 0) return true;
    if (component.kind === "group" && hasSubstantiveText(component.content)) return true;
  }
  return false;
}

function decodeSource(
  text: string,
  body: unknown,
  at: string,
  path: (string | number)[],
  source: { task: string; revision: number; select: string[] },
  errors: StructuredIssue[],
): void {
  if (!isMap(body)) {
    errors.push(
      issueAt(
        {
          code: "wrong-type",
          message: `the incorporation source at ${at} is a mapping with task, revision and select; found ${describe(body)}`,
          path,
        },
        rangeOf(text, body),
      ),
    );
    return;
  }
  let task: string | null = null;
  let revision: number | null = null;
  let select: ParsedNode[] | null = null;
  const unknown: string[] = [];
  for (const pair of (body as { items: Pair[] }).items) {
    const key = scalarKey(pair.key);
    if (key === "task") {
      if (isScalar(pair.value) && typeof pair.value.value === "string") {
        task = pair.value.value;
      } else {
        errors.push(
          issueAt(
            {
              code: "wrong-type",
              message: `the source 'task' at ${at} is the source task's name as text; found ${describe(pair.value)}`,
              path: [...path, key],
            },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === "revision") {
      // the authored form names the revision by its dense number; the
      // commit door resolves it
      if (
        isScalar(pair.value) &&
        typeof pair.value.value === "number" &&
        Number.isInteger(pair.value.value) &&
        pair.value.value >= 1
      ) {
        revision = pair.value.value;
      } else {
        errors.push(
          issueAt(
            {
              code: "bad-revision",
              message: `the incorporation source at ${at} pins the revision by number; write the source revision's dense number as an integer, like revision: 3`,
              path: [...path, key],
            },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === "select") {
      if (isSeq(pair.value)) {
        select = (pair.value as { items: ParsedNode[] }).items;
      } else {
        errors.push(
          issueAt(
            { code: "wrong-type", message: `the source 'select' at ${at} is a list of paths; found ${describe(pair.value)}`, path: [...path, "select"] },
            rangeOf(text, pair.value),
          ),
        );
      }
    } else if (key === null) {
      errors.push(
        issueAt(
          { code: "wrong-type", message: `source field names are plain text; found the key ${describe(pair.key)} at ${at}`, path },
          rangeOf(text, pair.key),
        ),
      );
    } else {
      unknown.push(key);
    }
  }
  if (unknown.length > 0) {
    errors.push(
      issueAt(
        {
          code: "unknown-key",
          message: `the incorporation source at ${at} takes only task, revision and select; drop the field '${unknown.join("', '")}'`,
          path,
        },
        rangeOf(text, body),
      ),
    );
  }
  if (task === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the incorporation source at ${at} does not name its task; add 'task: <task name>'`, path },
        rangeOf(text, body),
      ),
    );
  } else {
    source.task = task;
  }
  if (revision === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the incorporation source at ${at} pins no revision; add 'revision: <revision number>'`, path },
        rangeOf(text, body),
      ),
    );
  } else {
    source.revision = revision;
  }
  if (select === null) {
    errors.push(
      issueAt(
        { code: "missing-key", message: `the incorporation source at ${at} selects nothing; add 'select' listing the paths it takes, like [/content/0]`, path },
        rangeOf(text, body),
      ),
    );
    return;
  }
  if (select.length === 0) {
    errors.push(
      issueAt(
        { code: "empty-select", message: `the incorporation at ${at} selects nothing; list at least one path into the source board, like [/content/0]`, path: [...path, "select"] },
        rangeOf(text, body),
      ),
    );
    return;
  }
  select.forEach((part, index) => {
    const written = isScalar(part) && typeof part.value === "string" ? part.value : null;
    if (written === null) {
      errors.push(
        issueAt(
          { code: "wrong-type", message: `each 'select' entry is a path as text; found ${describe(part)}`, path: [...path, "select", index] },
          rangeOf(text, part),
        ),
      );
      return;
    }
    const hint = selectPathHint(written);
    if (hint !== null) {
      errors.push(
        issueAt(
          {
            code: "bad-select-path",
            message: `at ${at}: the select path '${written}' is not well formed; ${hint}`,
            path: [...path, "select", index],
          },
          rangeOf(text, part),
        ),
      );
    } else {
      source.select.push(written);
    }
  });
}

// The select path language, mirroring the crate's SelectPath::parse: a
// positional structural path naming whole components, /content/0 or
// /content/1/group/content/0. Returns the refusal hint, or null when the
// path is well formed.
function selectPathHint(input: string): string | null {
  const bad = (hint: string): string => hint;
  if (!input.startsWith("/")) {
    return bad("address the source board from its root, like /content/0");
  }
  const segments = input.slice(1).split("/");
  if (segments.some((segment) => segment.length === 0)) {
    return bad("write every step: 'content', an index, then 'group' or 'incorporation' and 'content', like /content/1/group/content/0");
  }
  if (segments[0] !== "content") {
    return bad("start at the source board's content list, like /content/0");
  }
  if (indexHint(segments[1]) !== null) {
    return bad(indexHint(segments[1])!.length > 0
      ? `after 'content' comes a component index, like /content/0; ${indexHint(segments[1])}`
      : "after 'content' comes a component index, like /content/0");
  }
  let tail = segments.slice(2);
  while (tail.length > 0) {
    if (tail[0] !== "group" && tail[0] !== "incorporation") {
      return bad("enter a component through its kind, 'group' or 'incorporation', like /content/1/group/content/0");
    }
    if (tail[1] !== "content") {
      return bad(`after '${tail[0]}' comes its 'content' list, like /content/1/group/content/0`);
    }
    const hint = indexHint(tail[2]);
    if (hint !== null) {
      return bad(hint.length > 0
        ? `after '${tail[0]}/content' comes an index, like /content/1/group/content/0; ${hint}`
        : `after '${tail[0]}/content' comes an index, like /content/1/group/content/0`);
    }
    tail = tail.slice(3);
  }
  return null;
}

// null when the segment is a plain index; a hint when it is not.
function indexHint(segment: string | undefined): string | null {
  if (segment === undefined) return "";
  if (segment.length === 0 || !/^[0-9]+$/.test(segment)) return "indices are plain digits";
  if (!Number.isSafeInteger(Number(segment))) return "the index is too large to be a component position";
  return null;
}
