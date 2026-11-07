// The read model for board grammar 1. The shapes here mirror the
// board-grammar crate's Board and Component one for one, plus the source
// positions rendering needs; loading a declaration never establishes a
// tracker task association, and the `task` reference stays authored
// content - the viewer trusts no board's claim of which task it names.

import type { Diagnostic, SourceRange } from "./errors.ts";

export type Block =
  | TextBlock
  | GroupBlock
  | AnnotationBlock
  | IncorporationBlock
  | FigureBlock
  | UnknownBlock;

export type TextBlock = {
  kind: "text";
  markdown: string;
  pos: SourceRange;
};

export type AnnotationBlock = {
  kind: "annotation";
  markdown: string;
  pos: SourceRange;
};

export type GroupBlock = {
  kind: "group";
  title?: string;
  content: Block[];
  pos: SourceRange;
};

export type IncorporationBlock = {
  kind: "incorporation";
  source: IncorporationSource;
  content: Block[];
  pos: SourceRange;
};

/**
 * A declared incorporation claim. Rendering it validates nothing: the
 * task, revision and select paths are displayed exactly as authored.
 */
export type IncorporationSource = {
  task: string;
  revision: string;
  select: string[];
};

export type FigureBlock = {
  kind: "figure";
  /** Board-tree-relative path, resolved on the declaration's machine. */
  input: string;
  caption: string;
  pos: SourceRange;
};

/**
 * A component whose kind this build does not know: a kind admitted by a
 * newer minor revision of the supported grammar. It never refuses the
 * board; it renders as its raw authored source beside a diagnostic.
 */
export type UnknownBlock = {
  kind: "unknown";
  name: string;
  source: string;
  pos: SourceRange;
};

export type Board = {
  /** The grammar version the board declares; decode accepts only the
   * supported versions, so this is always one of them. */
  grammar: number;
  /** Authored metadata only; never a tracker association. */
  task: string;
  content: Block[];
  diagnostics: Diagnostic[];
};

export function walkBlocks(
  blocks: Block[],
  visit: (block: Block) => void,
): void {
  for (const block of blocks) {
    visit(block);
    if (block.kind === "group" || block.kind === "incorporation") {
      walkBlocks(block.content, visit);
    }
  }
}
