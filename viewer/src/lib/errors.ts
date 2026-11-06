// Structured decode outcomes. Malformed input never becomes a board: the
// raw source stays available and every failure is an addressed, positional
// error the viewer can render verbatim.

export type Pos = { line: number; column: number; offset: number };

export type SourceRange = { start: Pos; end: Pos };

export type StructuredIssue = {
  code:
    | "yaml-syntax"
    | "duplicate-key"
    | "unknown-key"
    | "unknown-block"
    | "missing-key"
    | "wrong-type"
    | "empty-board"
    | "asset-missing"
    | "asset-denied";
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
