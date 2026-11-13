// The board's logical targets, FROM THE MODEL (content parts, never
// DOM-derived): identity is the canonical part path. Svelte-free and
// sources-free - unit-testable without any transport.

import type { Board } from "./model.ts";
import type { TargetKind } from "../select/label-select.ts";
import { parseMarkdown } from "../markdown.ts";
import { refPart, refsIn } from "./refs.ts";

/** A diagram's part path: its block's part plus its ordinal there (the
 * renderer stamps the same path on the rendered diagram). */
export function diagramPart(blockPart: string, ordinal: number): string {
  return `${blockPart}/diagram/${ordinal}`;
}

export type BoardTargetKind = "text" | "annotation" | "figure" | "group" | "incorporation" | "unknown";

export type BoardTarget = Readonly<{
  /** The canonical part path - the target's identity (scenario 12:
   * selection follows identity while it exists in the dataset). */
  part: string;
  kind: TargetKind;
  /** Prose for text narrowing: the block's markdown, groups by
   * title, figures by caption. */
  text: string;
  /** The incorporation's model-authored citation source, for
   * reference targets (t-36@r90); absent for every other kind. */
  source?: Readonly<{ task: string; revision: number; select: readonly string[] }>;
}>;

/** The model-derived targets of a decoded board: every content part,
 * nested included (the walk the observation readout already uses). */
/** A grammar select path (which names block kinds between content
 * segments, e.g. /content/0/group/content/1) to the canonical part
 * path (/content/0/content/1). Non-content segments are dropped. */
export function selectToPart(select: string): string {
  // Grammar select paths alternate the literal segment "content"
  // with indices and kind names between: /content/0/group/content/1.
  // The canonical part keeps the content/index pairs only.
  const segments = select.split("/");
  let out = "";
  for (let index = 0; index < segments.length - 1; index += 1) {
    if (segments[index] === "content" && /^\d+$/.test(segments[index + 1] ?? "")) {
      out += `/content/${segments[index + 1]}`;
      index += 1;
    }
  }
  return out;
}

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
      const base = { part, kind: block.kind as BoardTargetKind, text };
      out.push(
        block.kind === "incorporation"
          ? { ...base, source: { task: block.source.task, revision: block.source.revision, select: block.source.select } }
          : base,
      );
      // every rendered diagram in a block's markdown is a target of its
      // own (the prototype's mermaid figure kind, inst16.html:224)
      if (block.kind === "text" || block.kind === "annotation") {
        parseMarkdown(block.markdown, 0, 1).mermaidFences.forEach((_, ordinal) => {
          out.push({ part: diagramPart(part, ordinal), kind: "diagram", text: "diagram" });
        });
        // every ref in the prose is a target of its own (the prototype's
        // linkify, inst16.html:197); its text is the ref itself
        refsIn(block.markdown).forEach((ref, ordinal) => {
          out.push({ part: refPart(part, ordinal), kind: "ref", text: ref });
        });
      }
      // an incorporation's citation is a target of its own (the
      // prototype's .ref in the cited source line, inst16.html:206)
      if (block.kind === "incorporation") {
        out.push({
          part: `${part}/citation`,
          kind: "reference",
          text: `${block.source.task}@r${block.source.revision}`,
          source: { task: block.source.task, revision: block.source.revision, select: block.source.select },
        });
      }
      if (block.kind === "group" || block.kind === "incorporation") {
        walk(block.content, part);
      }
    });
  };
  walk(board.content, "");
  return out;
}
