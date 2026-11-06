// Copy semantics for selections that touch math. Pure core here; the DOM
// adapter (App.svelte) feeds it unit descriptors derived from data
// attributes. Source mode emits authored bytes for math; Plain mode emits
// the TeX value. A selection touching part of a formula emits that formula
// once; equal formulas at distinct locations are distinct selections.

export type CopyMode = "source" | "plain";

export type CopyUnit = {
  id: string;
  kind: "prose" | "code" | "math" | "mermaid";
  /** Rendered text content of the unit's element (prose, code). */
  text: string;
  /** Authored bytes (math with delimiters); TeX value otherwise rides text. */
  source: string;
  tex?: string;
  display?: boolean;
  /** True when the selection intersects this unit at all. */
  touched: boolean;
  /** True when the selection covers the whole unit. */
  covered: boolean;
};

export type CopySegment = {
  blockId: string;
  units: CopyUnit[];
};

/**
 * Serialize a selection described by segments of touched units, in document
 * order. Blocks separate by newlines; inline units concatenate directly.
 * Partially-selected prose keeps only its intersecting text (already cut by
 * the DOM adapter before this call); a touched formula always serializes
 * whole and exactly once.
 */
export function serializeSelection(segments: CopySegment[], mode: CopyMode): string {
  const lines: string[] = [];
  for (const segment of segments) {
    const parts: string[] = [];
    let inlineBuffer = "";
    const flushInline = (): void => {
      if (inlineBuffer.length > 0) {
        parts.push(inlineBuffer);
        inlineBuffer = "";
      }
    };
    for (const unit of segment.units) {
      if (!unit.touched) continue;
      if (unit.kind === "math") {
        flushInline();
        // One authored expression: emitted once, whole, per selection.
        parts.push(mode === "source" ? unit.source : (unit.tex ?? unit.text));
        continue;
      }
      if (unit.kind === "code" || unit.kind === "mermaid") {
        flushInline();
        parts.push(unit.covered || mode === "source" ? unit.source : unit.text);
        continue;
      }
      inlineBuffer += unit.text;
    }
    flushInline();
    if (parts.length > 0) {
      lines.push(parts.join(""));
    }
  }
  return lines.join("\n");
}
