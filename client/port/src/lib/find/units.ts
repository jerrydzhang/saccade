// The find units of a shown board: one unit per text-bearing block,
// derived FROM THE MODEL (never the DOM): the shown prose comes
// from the renderer's own parse (shown.ts). The unit id is the block's
// canonical part path, so an occurrence maps straight to presentation
// geometry. Prose first, authored source as fallback, is the session
// module's rule over these units.

import type { Board } from "../board/model.ts";
import type { FindUnit } from "./session.ts";
import { shownText } from "./shown.ts";

export function findUnitsOf(board: Board | null): readonly FindUnit[] {
  if (board === null) return [];
  const out: FindUnit[] = [];
  const walk = (blocks: readonly import("../board/model.ts").Block[], prefix: string): void => {
    blocks.forEach((block, index) => {
      const part = `${prefix}/content/${index}`;
      if (block.kind === "text" || block.kind === "annotation") {
        // text = the shown prose (the renderer's own parse, without
        // markdown syntax), so each occurrence is one a reader sees
        // and the painter marks (marks.ts); source = the authored
        // markdown, the session's fallback (a TeX query, say)
        out.push({ id: part, text: shownText(block.markdown), source: block.markdown });
      }
      if (block.kind === "group" || block.kind === "incorporation") {
        walk(block.content, part);
      }
    });
  };
  walk(board.content, "");
  return out;
}
