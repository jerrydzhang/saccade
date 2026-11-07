// Structured decode outcomes. Malformed input never becomes a board: the
// raw source stays available and every failure is an addressed, positional
// error the viewer can render verbatim.

export type Pos = { line: number; column: number; offset: number };

export type SourceRange = { start: Pos; end: Pos };

// The closed list of load-lane failure codes the host bridge can send.
// Both sides ship from this repository, so an unknown code is a bug, not
// a case to handle - but the renderer never trusts the wire blindly.
export const LANE_CODES = [
  "asset-missing",
  "asset-denied",
  "lane-unavailable",
  "core-unavailable",
  "core-protocol",
  "bridge-invalid",
  "machine-unknown",
  "machine-unavailable",
  "machine-needs-login",
  "machine-unreachable",
  "machine-needs-update",
  "read-failed",
  "exec-timeout",
  "exec-spawn-failed",
  "asset-too-large",
  "text-truncated",
] as const;

export type LaneCode = (typeof LANE_CODES)[number];

// The decode codes mirror the board-grammar crate's ErrorKind families
// one for one, plus the two bridge verdict codes (board-refused for a
// crate refusal, decoder-mismatch for the never-expected case where the
// mirror refuses text the crate accepted).
export type StructuredIssue = {
  code:
    | "yaml-syntax"
    | "multiple-documents"
    | "anchor-alias"
    | "duplicate-key"
    | "missing-key"
    | "unknown-key"
    | "unknown-kind"
    | "grammar-version"
    | "wrong-type"
    | "bad-revision"
    | "bad-task"
    | "bad-figure-path"
    | "bad-select-path"
    | "empty-select"
    | "no-receiving-text"
    | "too-deep"
    | "board-refused"
    | "decoder-mismatch"
    | LaneCode;
  message: string;
  /** Path into the declaration, e.g. ["content", 2, "group", "title"]. */
  path?: (string | number)[];
  line?: number;
  column?: number;
};

export type Diagnostic = StructuredIssue;

export function issueAt(
  base: Omit<StructuredIssue, "line" | "column">,
  range?: SourceRange | null,
): StructuredIssue {
  if (!range) return { ...base };
  return {
    ...base,
    line: range.start.line,
    column: range.start.column,
  };
}
