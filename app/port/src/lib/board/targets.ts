// The board's logical targets, FROM THE MODEL (content parts, never
// DOM-derived): identity is the canonical part path. Svelte-free and
// sources-free - unit-testable without any transport. The walk feeds
// the publication directly: its output is the contract's one target
// shape (lib/contracts/buffer.ts TargetDefinition), and the acts are
// wired by the controller at publication (resolved when asked).

import type { TargetAct, TargetDefinition } from "../contracts/buffer.ts";
import type { Board } from "./model.ts";
import type { TargetKind } from "../select/label-select.ts";
import { parseMarkdown } from "../markdown.ts";
import { refPart, refsIn } from "./refs.ts";

/** A diagram's part path: its block's part plus its ordinal there (the
 * renderer stamps the same path on the rendered diagram). */
export function diagramPart(blockPart: string, ordinal: number): string {
  return `${blockPart}/diagram/${ordinal}`;
}

/** A math expression's part path: its block's part plus its ordinal
 * there (the renderer stamps the same path on the rendered span). */
export function mathPart(blockPart: string, ordinal: number): string {
  return `${blockPart}/math/${ordinal}`;
}

/** Whether the pointer's click acts on a kind. The clickable kinds
 * wire clicks today (citations, prose refs, figures, diagrams,
 * incorporations - and the task list's rows and rail entries); math's
 * only act is the menu's, and a click on a paragraph or a group must
 * not copy or fold. */
export function targetClickable(kind: string): boolean {
  switch (kind) {
    case "figure":
    case "diagram":
    case "incorporation":
    case "ref":
    case "reference":
      return true;
    default:
      return false;
  }
}

/** A text block's headline: the first line of its prose that says
 * anything, stripped of the private-use markers and heading markup
 * the renderer eats. */
function proseHeadline(markdown: string): string {
  return (
    markdown
      .split("\n")
      .find((line) => line.trim().length > 0)
      ?.replace(/[\uE000-\uE003]/g, "")
      .replace(/^\s*#+\s*/, "")
      .trim() ?? ""
  );
}

/** The headline a kind's target carries: what heads the act menu
 * (citations and refs their address and ref text, text and groups the
 * prose's first line, figures their caption, incorporations their
 * source line, diagrams and math their kind). */
function headlineOf(block: import("./model.ts").Block, part: string): string {
  if (block.kind === "text" || block.kind === "annotation") return proseHeadline(block.markdown);
  if (block.kind === "group") return block.title ?? "";
  if (block.kind === "figure") return block.caption;
  if (block.kind === "incorporation") return `incorporates ${block.source.task}@r${block.source.revision}`;
  return part;
}

/** A grammar select path (which names block kinds between content
 * segments, e.g. /content/0/group/content/1) to the canonical part
 * path (/content/0/content/1). Non-content segments are dropped. */
export function selectToPart(select: string): string {
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

/** The facts of a part's own target, resolved when an act asks: the
 * incorporation's citation source, the math expression's TeX, the
 * ref's text. Null when the part names no target. */
export function targetFactsAt(
  blocks: readonly import("./model.ts").Block[],
  part: string,
): {
  source?: Readonly<{ task: string; revision: number; select: readonly string[] }>;
  tex?: string;
  ref?: string;
} | null {
  for (const target of walkTargets(blocks)) {
    if (target.part !== part) continue;
    return { source: target.source, tex: target.tex, ref: target.ref };
  }
  return null;
}

/** The model-derived targets of a decoded board: every content part,
 * nested included (the walk the observation readout already uses).
 * `acts` wires each target's acts at publication; without it the
 * targets carry the shape alone. */
export function boardTargetsOf(
  blocks: readonly import("./model.ts").Block[],
  acts?: (target: { id: string; kind: string; label: string }) => readonly TargetAct[],
): readonly TargetDefinition[] {
  const out: TargetDefinition[] = [];
  for (const target of walkTargets(blocks)) {
    out.push({
      id: target.part,
      kind: target.kind,
      label: target.label,
      inside: target.inside,
      clickable: targetClickable(target.kind),
      anchor: target.part,
      acts:
        acts === undefined
          ? () => []
          : () => acts({ id: target.part, kind: target.kind, label: target.label }),
    });
  }
  return out;
}

/** The walk every content part of a decoded board, nested included,
 * in document order - diagrams, math expressions and refs of their
 * block's own, an incorporation's citation after its source line -
 * with each kind's headline and the facts its acts resolve. */
function* walkTargets(
  blocks: readonly import("./model.ts").Block[],
  prefix = "",
): Generator<{
  part: string;
  kind: TargetKind;
  label: string;
  inside: string | null;
  source?: Readonly<{ task: string; revision: number; select: readonly string[] }>;
  tex?: string;
  ref?: string;
}> {
  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index]!;
    const part = `${prefix}/content/${index}`;
    const inside = prefix === "" ? null : prefix;
    const label = headlineOf(block, part);
    if (block.kind === "incorporation") {
      const source = { task: block.source.task, revision: block.source.revision, select: block.source.select };
      yield { part, kind: "incorporation", label, inside, source };
      // an incorporation's citation is a target of its own (the
      // prototype's .ref in the cited source line, inst16.html:206)
      yield {
        part: `${part}/citation`,
        kind: "reference",
        label: `${block.source.task}@r${block.source.revision}`,
        inside: part,
        source,
      };
    } else {
      yield { part, kind: block.kind as TargetKind, label, inside };
    }
    // every rendered diagram in a block's markdown is a target of its
    // own (the prototype's mermaid figure kind, inst16.html:224)
    if (block.kind === "text" || block.kind === "annotation") {
      const parsed = parseMarkdown(block.markdown, 0, 1);
      for (let ordinal = 0; ordinal < parsed.mermaidFences.length; ordinal += 1) {
        yield { part: diagramPart(part, ordinal), kind: "diagram", label: "diagram", inside: part };
      }
      // every math expression in the block's markdown is a target of
      // its own, in document order - the same order the renderer
      // walks the parse when it stamps the spans; the TeX rides the
      // facts (the copy act's payload)
      for (let ordinal = 0; ordinal < parsed.mathNodes.length; ordinal += 1) {
        yield {
          part: mathPart(part, ordinal),
          kind: "math",
          label: "math",
          inside: part,
          tex: parsed.mathNodes[ordinal]!.value,
        };
      }
      // every ref in the prose is a target of its own (the
      // prototype's linkify, inst16.html:197); its text is the ref
      let refOrdinal = 0;
      for (const ref of refsIn(block.markdown)) {
        yield { part: refPart(part, refOrdinal), kind: "ref", label: ref, inside: part, ref };
        refOrdinal += 1;
      }
    }
    if (block.kind === "group" || block.kind === "incorporation") {
      yield* walkTargets(block.content, part);
    }
  }
}
