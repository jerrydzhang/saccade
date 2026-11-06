// The provisional board read model. The YAML grammar is an explicitly
// provisional input adapter (t-71 scope), not a ratified stored schema:
// loading a declaration never establishes a tracker task association, and
// the `task` string stays authored metadata.

import type { Diagnostic, SourceRange } from "./errors.ts";

export type Block =
  | TextBlock
  | GroupBlock
  | AnnotationBlock
  | IncorporationBlock
  | FigureBlock;

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
  relation?: string;
  content: Block[];
  pos: SourceRange;
};

export type IncorporationBlock = {
  kind: "incorporation";
  id: string;
  source: IncorporationSource;
  content: Block[];
  pos: SourceRange;
};

/**
 * A declared incorporation claim. Rendering it validates nothing: the task,
 * revision and select path are displayed exactly as authored.
 */
export type IncorporationSource = {
  task: string;
  revision: string;
  select?: string[];
};

export type FigureBlock = {
  kind: "figure";
  /** Declaration-relative path, resolved on the declaration's machine. */
  path: string;
  caption?: string;
  pos: SourceRange;
};

export type Board = {
  /** Authored metadata only; never a tracker association. */
  task: string | null;
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
