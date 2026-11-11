// The board's logical targets, FROM THE MODEL (content parts, never
// DOM-derived): identity is the canonical part path. Svelte-free and
// sources-free - unit-testable without any transport.

import type { Board } from "./model.ts";
import type { TargetKind } from "../select/label-select.ts";

export type BoardTargetKind = "text" | "annotation" | "figure" | "group" | "incorporation" | "unknown";

export type BoardTarget = Readonly<{
  /** The canonical part path - the target's identity (scenario 12:
   * selection follows identity while it exists in the dataset). */
  part: string;
  kind: TargetKind;
  /** Prose for text narrowing: the block's markdown, groups by
   * title, figures by caption. */
  text: string;
}>;

/** The model-derived targets of a decoded board: every content part,
 * nested included (the walk the observation readout already uses). */
/** The canonical part paths strictly enclosing a part path (its
 * ancestor containers, outermost first). `/content/0/content/1` ->
 * [`/content/0`]. */
export function partAncestors(part: string): readonly string[] {
  const out: string[] = [];
  let index = part.lastIndexOf("/content/");
  while (index > 0) {
    out.unshift(part.slice(0, index));
    index = part.lastIndexOf("/content/", index - 1);
  }
  return out;
}

export function boardTargetsOf(board: Board): readonly BoardTarget[] {
  const out: BoardTarget[] = [];
  const walk = (blocks: readonly import("./model.ts").Block[], prefix: string): void => {
    blocks.forEach((block, index) => {
      const part = `${prefix}/content/${index}`;
      let text = "";
      if (block.kind === "text" || block.kind === "annotation") text = block.markdown;
      else if (block.kind === "group") text = block.title ?? "";
      else if (block.kind === "figure") text = block.caption;
      out.push({ part, kind: block.kind as BoardTargetKind, text });
      if (block.kind === "group" || block.kind === "incorporation") {
        walk(block.content, part);
      }
    });
  };
  walk(board.content, "");
  return out;
}
