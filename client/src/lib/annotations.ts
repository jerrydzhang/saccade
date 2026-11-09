// Task-board annotations (t-14 "The viewer", ratified law): `a` on a
// part drafts in place; `space p` publishes the drafts together as ONE
// commit with ONE message under the human's own identity. The published
// board.yaml is the shown revision's authored text plus annotation
// components and nothing else - byte-level insertions at the parts'
// own positions. Drafts live in the window's state since the
// window store landed (t-24@r20); the interim localStorage store
// left with the window store, its one-time mover living in windows.ts.

// -- the draft and the payload ------------------------------------------

export type Draft = { path: string; text: string };

// -- YAML insertion ----------------------------------------------------

/** A component block in the authored board.yaml: its first line, and
 * the line after its last. */
type BlockSpan = { start: number; end: number };

function dashIndent(line: string): number {
  const match = /^(\s*)- /.exec(line);
  return match?.[1]?.length ?? -1;
}

/** The line spans of the components at a content level, starting from
 * `from` at the given indent: each block runs to the next line at or
 * below its own indent. */
function componentSpans(lines: string[], from: number, indent: number): BlockSpan[] {
  const spans: BlockSpan[] = [];
  let index = from;
  while (index < lines.length) {
    const line = lines[index] ?? "";
    if (line.trim().length === 0) {
      index++;
      continue;
    }
    const at = dashIndent(line);
    if (at !== indent) {
      if (at < indent || at < 0) break;
      index++;
      continue;
    }
    let end = index + 1;
    while (end < lines.length) {
      const candidate = lines[end] ?? "";
      if (candidate.trim().length === 0) {
        end++;
        continue;
      }
      const candidateIndent = dashIndent(candidate);
      if (candidateIndent !== -1 && candidateIndent <= indent) break;
      if (candidateIndent === -1 && candidate.length - candidate.trimStart().length <= indent) break;
      end++;
    }
    spans.push({ start: index, end });
    index = end;
  }
  return spans;
}

function locateBlockSpanDepth(lines: string[], path: string): { span: BlockSpan; indent: number } | null {
  const steps = path.split("/").filter((step) => step.length > 0);
  if (steps[0] !== "content") return null;
  // the board's top-level content list begins after its `content:` key
  let from = lines.findIndex((line) => line.trim() === "content:") + 1;
  if (from <= 0) return null;
  let indent = 2;
  let span: BlockSpan | null = null;
  // the path walks list/index pairs, each optionally entering a named
  // kind's own content list
  let cursor = 0;
  while (cursor < steps.length) {
    const listStep = steps[cursor];
    const indexStep = steps[cursor + 1];
    if (listStep !== "content" || indexStep === undefined) return null;
    const index = Number(indexStep);
    if (!Number.isInteger(index)) return null;
    cursor += 2;
    const spans = componentSpans(lines, from, indent);
    const found = spans[index];
    if (!found) return null;
    span = found;
    const nestedStep = steps[cursor];
    if (nestedStep === "group" || nestedStep === "incorporation") {
      // the following pair rides inside this block's own content list
      cursor += 1;
      let contentLine = -1;
      for (let line = found.start + 1; line < found.end; line++) {
        if ((lines[line] ?? "").trim() === "content:") {
          contentLine = line;
          break;
        }
      }
      if (contentLine < 0) return null;
      const line = lines[contentLine] ?? "";
      from = contentLine + 1;
      indent = line.length - line.trimStart().length + 2;
    } else {
      return { span, indent };
    }
  }
  return span === null ? null : { span, indent };
}

/** The block a select path names, as authored lines. */
export function locateBlockSpan(lines: string[], path: string): BlockSpan | null {
  return locateBlockSpanDepth(lines, path)?.span ?? null;
}

/** The block a select path names, with the indentation its own `- `
 * carries: everything the explainer edit needs to place an annotation
 * beside it. */
export function locateComponent(lines: string[], path: string): { span: BlockSpan; indent: number } | null {
  return locateBlockSpanDepth(lines, path);
}

function insertAnnotation(lines: string[], span: BlockSpan, text: string, indent: number): string[] {
  const pad = " ".repeat(indent);
  const bodyPad = " ".repeat(indent + 4);
  const annotation = [`${pad}- annotation: |`];
  for (const line of text.split("\n")) {
    annotation.push(line.length > 0 ? `${bodyPad}${line}` : "");
  }
  const out = [...lines];
  // trim trailing blank lines of the part, insert, keep the blanks after
  let end = span.end;
  while (end > span.start + 1 && (out[end - 1] ?? "").trim().length === 0) end--;
  out.splice(end, 0, ...annotation);
  return out;
}

/** The published board text: the shown authored text with one
 * annotation component inserted per draft, in path order, and nothing
 * else. Drafts on paths the text cannot locate refuse. */
export function annotatedBoard(boardText: string, drafts: Draft[]): { text: string | null; missing: string[] } {
  const lines = boardText.split("\n");
  const ordered = [...drafts].sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
  const missing: string[] = [];
  const insertions: Array<{ span: BlockSpan; text: string; indent: number }> = [];
  for (const draft of ordered) {
    const located = locateBlockSpanDepth(lines, draft.path);
    if (!located) {
      missing.push(draft.path);
      continue;
    }
    insertions.push({ span: located.span, text: draft.text, indent: located.indent });
  }
  if (missing.length > 0) return { text: null, missing };
  // insert from the bottom up so earlier spans stay valid
  let working = [...lines];
  insertions
    .sort((a, b) => b.span.start - a.span.start)
    .forEach((insertion) => {
      working = insertAnnotation(working, insertion.span, insertion.text, insertion.indent);
    });
  return { text: working.join("\n"), missing: [] };
}

// -- the publish act ---------------------------------------------------

export type PublishIdentity = { actor: string; tier: "human" };

export type PublishRequest = {
  task: number;
  /** The SHOWN revision's authored board text: the base the drafts
   * annotate. */
  boardText: string;
  drafts: Draft[];
  message: string;
  identity: PublishIdentity;
  /** The figure files the board names, fetched fresh: path -> bytes. */
  figures: Array<{ path: string; bytes: Uint8Array }>;
};

export type PublishPayload = {
  context: { actor: string; tier: "human"; scope: null };
  task: number;
  files: { files: Array<{ path: string; content: string }> };
  message: string;
  client: string;
};

function base64Of(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

/** The wire payload: the annotated board plus its unchanged figures,
 * one commit under the human's own identity. */
export function publishPayload(request: PublishRequest): { payload: PublishPayload } | { failure: string } {
  const annotated = annotatedBoard(request.boardText, request.drafts);
  if (annotated.text === null) {
    return { failure: `the drafts name parts this board does not hold: ${annotated.missing.join(", ")}` };
  }
  const files: Array<{ path: string; content: string }> = [
    { path: "board.yaml", content: base64Of(new TextEncoder().encode(annotated.text)) },
    ...request.figures.map((figure) => ({ path: figure.path, content: base64Of(figure.bytes) })),
  ];
  return {
    payload: {
      context: { actor: request.identity.actor, tier: request.identity.tier, scope: null },
      task: request.task,
      files: { files },
      message: request.message,
      client: "board-viewer",
    },
  };
}

/** The publish POST: the suite's one record write. */
export async function publishAnnotations(
  base: URL,
  payload: PublishPayload,
): Promise<{ ok: true; revision: number } | { ok: false; code: string; detail: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15_000);
  let response: Response;
  try {
    response = await fetch(new URL("/api/v1/boards/commit", base).toString(), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);
    return { ok: false, code: "lane-unavailable", detail: `the tracker server did not answer: ${String(error)}` };
  }
  clearTimeout(timer);
  const body: unknown = await response.json().catch(() => null);
  const record = body as { error?: { code?: string; detail?: string }; revision?: number } | null;
  if (!response.ok) {
    return {
      ok: false,
      code: record?.error?.code ?? `status ${response.status}`,
      detail: record?.error?.detail ?? "",
    };
  }
  const revision = record?.revision;
  if (typeof revision !== "number") {
    return { ok: false, code: "core-protocol", detail: "the publish answer carries no revision" };
  }
  return { ok: true, revision };
}
