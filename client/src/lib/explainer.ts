// Explainer-file annotations (t-14 "The viewer", ratified item 3; the
// transport settled on t-24@r21). On an explainer - a bare-path file
// that decodes as board grammar 1 - `a` drafts in place and the
// editor's save IS the submit: one annotation component written
// straight into the file, through the local core's `file.edit`, with
// old = the annotated component's own lines and new = those lines plus
// the annotation. Nothing pooled, nothing recorded: a file write is
// not a record act.
//
// Before anything is sent the client applies the edit to the shown
// text itself and decodes the result as grammar 1; a result that does
// not decode with the annotation where intended sends nothing and the
// editor stays open. Old text that does not occur exactly once refuses
// - that exact-once rule is also the wire's half of the guard, since
// file.edit is all-or-none.
//
// The staleness guard is whole-file byte equality, t-14's literal
// rule: a file that changed since this window read it refuses the
// write, keeps the annotation TEXT on the device (keyed by the path,
// not yet moved into the window state; t-24@r21 A6), drops the
// placement and says so;
// the next `a` on the freshly read file opens the editor with the kept
// text. The only thing ever kept is the text of a refused annotation.

import { decodeBoard } from "./board/decode.ts";
import { locateComponent } from "./annotations.ts";
import type { Board } from "./board/model.ts";

export type ExplainerEdit = { old: string; new: string };

export type ExplainerEditPlan = {
  edit: ExplainerEdit;
  /** The shown text with the annotation applied: what the file should
   * read after the edit lands. */
  annotated: string;
  /** The content index the annotation component takes in the decoded
   * result: right after the part it addresses. */
  annotationIndex: number;
};

/** Walk a select path to the component list it names and the index it
 * names within that list: the same pairing locateComponent walks, so
 * the decoded result can be checked at the exact spot the edit
 * targeted. */
function decodedSlot(
  board: Board,
  selectPath: string,
): { list: Board["content"]; index: number } | null {
  const steps = selectPath.split("/").filter((step) => step.length > 0);
  if (steps[0] !== "content") return null;
  let list: Board["content"] = board.content;
  let cursor = 0;
  while (cursor < steps.length) {
    if (steps[cursor] !== "content") return null;
    const index = Number(steps[cursor + 1]);
    if (!Number.isInteger(index)) return null;
    cursor += 2;
    const nested = steps[cursor];
    if (nested === "group" || nested === "incorporation") {
      const part = list[index];
      if (!part || part.kind !== nested) return null;
      list = part.content;
      cursor += 1;
    } else {
      return { list, index };
    }
  }
  return null;
}

function countOccurrences(text: string, needle: string): number {
  let count = 0;
  let at = text.indexOf(needle);
  while (at >= 0) {
    count++;
    at = text.indexOf(needle, at + needle.length);
  }
  return count;
}

function annotationLines(text: string, indent: number): string[] {
  const pad = " ".repeat(indent);
  const bodyPad = " ".repeat(indent + 4);
  const lines = [`${pad}- annotation: |`];
  for (const line of text.split("\n")) {
    lines.push(line.length > 0 ? `${bodyPad}${line}` : "");
  }
  return lines;
}

/** The edit plan for one annotation on the shown explainer text: the
 * `file.edit` old/new pair, the annotated result, and where the
 * annotation must land. A refusal names the reason and sends nothing. */
export function explainerEdit(
  boardText: string,
  selectPath: string,
  annotationText: string,
): ExplainerEditPlan | { failure: string } {
  const lines = boardText.split("\n");
  const located = locateComponent(lines, selectPath);
  if (!located) {
    return { failure: "the part this annotation addresses is not in the shown text" };
  }
  // the component's own lines, without the blank tail the span may
  // carry: the exact text the edit names as old
  let end = located.span.end;
  while (end > located.span.start + 1 && (lines[end - 1] ?? "").trim().length === 0) end--;
  const old = lines.slice(located.span.start, end).join("\n");
  const occurrences = countOccurrences(boardText, old);
  if (occurrences !== 1) {
    return {
      failure: `the part's own text occurs ${occurrences} times in the file; the write refuses - identical components cannot be told apart`,
    };
  }
  const annotation = annotationLines(annotationText, located.indent);
  const new_ = `${old}\n${annotation.join("\n")}`;
  const working = [...lines];
  working.splice(end, 0, ...annotation);
  const annotated = working.join("\n");
  const decoded = decodeBoard(annotated);
  if (decoded.status !== "ok") {
    return {
      failure: `the annotated file would not decode as a board; nothing was sent: ${
        decoded.errors[0]?.message ?? "decode refused"
      }`,
    };
  }
  const slot = decodedSlot(decoded.board, selectPath);
  const placed = slot ? slot.list[slot.index + 1] : undefined;
  if (!slot || !placed || placed.kind !== "annotation") {
    return {
      failure:
        "the annotated file decodes, but the annotation does not sit beside the part it addresses (its indentation would nest it elsewhere); nothing was sent",
    };
  }
  return { edit: { old, new: new_ }, annotated, annotationIndex: slot.index + 1 };
}

// -- the kept text of a refused annotation -----------------------------

const KEPT_KEY = "saccade.explainer.kept";

function keptKey(path: string): string {
  return `${KEPT_KEY}.${path}`;
}

export function loadKeptExplainer(path: string): string | null {
  try {
    return localStorage.getItem(keptKey(path));
  } catch {
    return null;
  }
}

export function saveKeptExplainer(path: string, text: string): void {
  try {
    localStorage.setItem(keptKey(path), text);
  } catch {
    // a full or unavailable store loses only the convenience: the
    // refused text still rode the refusal note's editor for now
  }
}

export function clearKeptExplainer(path: string): void {
  try {
    localStorage.removeItem(keptKey(path));
  } catch {
    // nothing depends on the removal succeeding
  }
}

// -- the submit over the wire ------------------------------------------

export type EditOutcome = { ok: true; bytes: number } | { ok: false; code: string; detail: string };

export type EditInvoke =
  (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

/** One annotation into the file: the edit command through the bridge,
 * which rides the local core's `file.edit` (all-or-none, exact-once).
 * An edit-failed answer means the anchor text is no longer there
 * exactly once - the file moved under this window. */
export async function submitExplainerEdit(
  invoke: EditInvoke,
  request: { path: string; old: string; new: string; on: string },
): Promise<EditOutcome> {
  let result: unknown;
  try {
    result = await invoke("edit_declaration", {
      path: request.path,
      old: request.old,
      new: request.new,
      on: request.on,
    });
  } catch (error) {
    const candidate = error as { code?: unknown; message?: unknown };
    if (
      typeof candidate === "object" &&
      candidate !== null &&
      typeof candidate.code === "string" &&
      typeof candidate.message === "string"
    ) {
      return { ok: false, code: candidate.code, detail: candidate.message };
    }
    return { ok: false, code: "read-failed", detail: String(error) };
  }
  const bytes = (result as { bytes?: unknown } | null)?.bytes;
  if (typeof bytes !== "number") {
    return { ok: false, code: "bridge-invalid", detail: "the edit answer carries no byte count" };
  }
  return { ok: true, bytes };
}
