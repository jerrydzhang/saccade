// The find units of a shown board: one unit per text-bearing block,
// derived FROM THE MODEL (never the DOM). The unit id is the block's
// canonical part path, so an occurrence maps straight to presentation
// geometry. Prose first, authored source as fallback, is the session
// module's rule over these units.

import type { Board } from "../board/model.ts";
import type { FindUnit } from "./session.ts";

export function findUnitsOf(board: Board | null): readonly FindUnit[] {
  if (board === null) return [];
  const out: FindUnit[] = [];
  const walk = (blocks: readonly import("../board/model.ts").Block[], prefix: string): void => {
    blocks.forEach((block, index) => {
      const part = `${prefix}/content/${index}`;
      if (block.kind === "text" || block.kind === "annotation") {
        // text = the shown prose (markdown carries both the rendered
        // reading and the authored source here; the session's own
        // normalize handles the math delimiters)
        out.push({ id: part, text: block.markdown, source: block.markdown });
      }
      if (block.kind === "group" || block.kind === "incorporation") {
        walk(block.content, part);
      }
    });
  };
  walk(board.content, "");
  return out;
}
