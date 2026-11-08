// DOM projection of the read model. Plain typed DOM building: the same
// view model feeds the human UI, the machine-readable inspection, and the
// copy handler. Rendered elements carry their source positions and logical
// unit ids as data attributes.

import type { Root, RootContent, PhrasingContent } from "mdast";
import { parseMarkdown, docRange, promotedDisplayMath, authoredSlice } from "./markdown.ts";
import type { Diagnostic } from "./board/errors.ts";
import type { Board, Block, FigureBlock, UnknownBlock } from "./board/model.ts";

export type RenderHooks = {
  /** Document character offset -> projection unit id. */
  unitIndex: Map<number, string>;
  /** Block-local character offset -> exact authored math bytes; built
   * by renderMarkdown from the same parse the renderer walks. */
  authoredMath?: Map<number, string>;
  /** Raw HTML and other render-time findings. */
  onDiagnostic: (diagnostic: Diagnostic) => void;
};

/**
 * The exact authored bytes for a math node: sliced from the block
 * source through the shared authoredMath map the parse stage builds.
 * Reconstruction from the value would violate authored spelling.
 */
export function authoredMathOf(
  hooks: RenderHooks,
  node: { position?: { start?: { offset?: number | null } } },
): string {
  const offset = node.position?.start?.offset;
  if (offset === null || offset === undefined) return "";
  return hooks.authoredMath?.get(offset) ?? "";
}

type Attrs = Record<string, string | boolean | undefined>;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: (Node | string | null | undefined)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === false) continue;
    if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child);
  }
  return node;
}

export function unitIdAt(hooks: RenderHooks, offset: number | null | undefined): string | undefined {
  if (offset === null || offset === undefined) return undefined;
  return hooks.unitIndex.get(offset);
}

export function renderBoard(board: Board, hooks: RenderHooks): HTMLElement {
  const root = el("div", { class: "board" });
  // Block ids continue across nesting levels so they are unique
  // document-wide.
  const counter = { value: 0 };
  const walk = (blocks: Block[], container: HTMLElement): void => {
    for (const block of blocks) {
      container.append(renderBlock(block, `b${counter.value++}`, hooks));
    }
  };
  walk(board.content, root);
  return root;
}

export function renderBlock(block: Block, blockId: string, hooks: RenderHooks): HTMLElement {
  switch (block.kind) {
    case "text": {
      const article = el("article", { class: "block block-text", "data-block-id": blockId });
      article.append(renderMarkdown(block.markdown, block.pos.start.offset, block.pos.start.line, hooks, blockId));
      return article;
    }
    case "annotation": {
      const aside = el(
        "aside",
        { class: "block block-annotation", "data-block-id": blockId, "aria-label": "review annotation" },
        el("p", { class: "annotation-label" }, "review annotation"),
      );
      aside.append(renderMarkdown(block.markdown, block.pos.start.offset, block.pos.start.line, hooks, blockId));
      return aside;
    }
    case "group": {
      const section = el("section", {
        class: "block block-group",
        "data-block-id": blockId,
      });
      if (block.title) {
        section.append(el("h2", { class: "group-title" }, block.title));
      }
      let childCounter = 0;
      for (const child of block.content) {
        section.append(renderBlock(child, `${blockId}.${childCounter++}`, hooks));
      }
      return section;
    }
    case "incorporation": {
      const div = el("div", { class: "block block-incorporation", "data-block-id": blockId });
      const select = block.source.select ? ` select ${block.source.select.join("/")}` : "";
      // The citation names its source as t-N@rK, the revision read
      // straight from the authored number; hydration turns it into a
      // link to that revision.
      div.append(
        el(
          "p",
          { class: "provenance" },
          "incorporates ",
          el(
            "span",
            {
              class: "citation",
              "data-cite-task": block.source.task,
              "data-cite-revision": String(block.source.revision),
            },
            block.source.task,
          ),
          el("span", { class: "provenance-hash" }, ` @r${block.source.revision}`),
          ...(select ? [el("span", {}, select)] : []),
        ),
      );
      let childCounter = 0;
      for (const child of block.content) {
        div.append(renderBlock(child, `${blockId}.${childCounter++}`, hooks));
      }
      return div;
    }
    case "figure": {
      return renderFigureBlock(block, blockId);
    }
    case "unknown": {
      return renderUnknownBlock(block, blockId);
    }
  }
}

export function renderFigureBlock(block: FigureBlock, blockId: string): HTMLElement {
  const figure = el(
    "figure",
    { class: "block block-figure", "data-block-id": blockId, "data-figure-input": block.input },
  );
  const frame = el("div", { class: "figure-frame", "data-pending": "figure" });
  frame.append(el("p", { class: "figure-placeholder" }, `figure: ${block.input}`));
  figure.append(frame);
  figure.append(el("figcaption", {}, block.caption));
  return figure;
}

/** An unknown kind (a newer minor grammar's admission) keeps its place in
 * the document but renders as its raw authored source, never silently
 * dropped and never manufactured into a known shape. */
export function renderUnknownBlock(block: UnknownBlock, blockId: string): HTMLElement {
  const article = el(
    "article",
    {
      class: "block block-unknown",
      "data-block-id": blockId,
      "data-unknown-kind": block.name,
      "aria-label": `unknown component kind ${block.name}, rendered as source`,
    },
    el("p", { class: "unknown-label" }, `unknown component kind '${block.name}' - rendered as its raw source`),
  );
  article.append(
    el(
      "pre",
      {
        class: "md-code",
        "data-copy-unit": "code",
        "data-unit-source": block.source,
        "data-block-id": blockId,
        "data-line": String(block.pos.start.line),
      },
      el("code", {}, block.source),
    ),
  );
  return article;
}

export function renderMarkdown(
  markdown: string,
  blockOffset: number,
  blockLine: number,
  hooks: RenderHooks,
  blockId: string,
): DocumentFragment {
  const parsed = parseMarkdown(markdown, blockOffset, blockLine);
  for (const diagnostic of parsed.diagnostics) {
    hooks.onDiagnostic(diagnostic);
  }
  const authoredMath = new Map<number, string>();
  const collectAuthored = (node: RootContent | void): void => {
    if (!node) return;
    if (node.type === "inlineMath" || node.type === "math") {
      const start = node.position?.start?.offset;
      if (start !== null && start !== undefined) {
        authoredMath.set(start, authoredSlice(markdown, node.position));
      }
      return;
    }
    if ("children" in node && Array.isArray(node.children)) {
      for (const child of node.children as RootContent[]) {
        collectAuthored(child);
      }
    }
  };
  for (const child of parsed.root.children) {
    collectAuthored(child);
  }
  hooks.authoredMath = authoredMath;

  const fragment = document.createDocumentFragment();
  for (const child of parsed.root.children) {
    const onlyChild = child.type === "paragraph" ? child.children[0] : undefined;
    if (onlyChild && onlyChild.type === "inlineMath" && promotedDisplayMath(markdown, child, onlyChild)) {
      fragment.append(renderDisplayMath(onlyChild, blockOffset, blockLine, hooks, blockId));
    } else {
      fragment.append(renderNode(child, blockOffset, blockLine, hooks, blockId));
    }
  }
  return fragment;
}

function renderDisplayMath(
  node: PhrasingContent & { value: string },
  blockOffset: number,
  blockLine: number,
  hooks: RenderHooks,
  blockId: string,
): HTMLElement {
  const pos = docRange(node.position, blockOffset, blockLine);
  const id = unitIdAt(hooks, pos?.start.offset);
  return el(
    "span",
    {
      class: "math-display",
      "data-copy-unit": "math",
      "data-unit-display": "true",
      ...(id ? { "data-unit-id": id } : {}),
      "data-unit-tex": node.value,
      "data-unit-source": authoredMathOf(hooks, node) || `$$${node.value}$$`,
      "data-pending": "math",
      ...(pos ? { "data-line": String(pos.start.line) } : {}),
    },
    `$$${node.value}$$`,
  );
}

function renderNode(
  node: RootContent,
  blockOffset: number,
  blockLine: number,
  hooks: RenderHooks,
  blockId: string,
): HTMLElement {
  const pos = docRange(node.position, blockOffset, blockLine);
  const lineAttr = pos ? { "data-line": String(pos.start.line) } : {};

  switch (node.type) {
    case "paragraph":
      return el("p", { class: "md-paragraph", ...lineAttr }, ...renderChildren(node.children as PhrasingContent[], node, blockOffset, blockLine, hooks, blockId));
    case "heading": {
      const tag = node.depth <= 3 ? ((["h2", "h3", "h4"] as const)[node.depth - 1] ?? "h4") : "h4";
      return el(tag, { class: "md-heading", ...lineAttr }, ...renderChildren(node.children as PhrasingContent[], node, blockOffset, blockLine, hooks, blockId));
    }
    case "thematicBreak":
      return el("hr", { class: "md-rule" });
    case "blockquote": {
      const quote = el("blockquote", { class: "md-quote", ...lineAttr });
      for (const child of node.children) {
        quote.append(renderNode(child, blockOffset, blockLine, hooks, blockId));
      }
      return quote;
    }
    case "list": {
      const list = el(node.ordered ? "ol" : "ul", { class: "md-list", ...(node.start ? { start: String(node.start) } : {}) });
      for (const item of node.children) {
        const li = el("li", { class: "md-list-item", ...(item.checked !== null && item.checked !== undefined ? { "data-checked": String(item.checked) } : {}) });
        for (const child of item.children) {
          li.append(renderNode(child, blockOffset, blockLine, hooks, blockId));
        }
        list.append(li);
      }
      return list;
    }
    case "table": {
      const table = el("table", { class: "md-table", ...lineAttr });
      const head = node.children[0];
      if (head) {
        const thead = el("thead");
        const tr = el("tr");
        head.children.forEach((cell, i) => {
          const align = node.align?.[i] ?? null;
          tr.append(el("th", { ...(align ? { style: `text-align:${align}` } : {}) }, ...renderChildren(cell.children as PhrasingContent[], cell, blockOffset, blockLine, hooks, blockId)));
        });
        thead.append(tr);
        table.append(thead);
      }
      const tbody = el("tbody");
      for (const row of node.children.slice(1)) {
        const tr = el("tr");
        row.children.forEach((cell, i) => {
          const align = node.align?.[i] ?? null;
          tr.append(el("td", { ...(align ? { style: `text-align:${align}` } : {}) }, ...renderChildren(cell.children as PhrasingContent[], cell, blockOffset, blockLine, hooks, blockId)));
        });
        tbody.append(tr);
      }
      table.append(tbody);
      return table;
    }
    case "math": {
      return renderMathBlock(node, blockOffset, blockLine, hooks, blockId);
    }
    case "code": {
      if (node.lang === "mermaid") {
        const range = pos;
        const id = unitIdAt(hooks, range?.start.offset);
        return el(
          "div",
          {
            class: "mermaid-block",
            "data-copy-unit": "mermaid",
            ...(id ? { "data-unit-id": id } : {}),
            "data-unit-source": node.value,
            "data-pending": "mermaid",
            "data-block-id": blockId,
            ...lineAttr,
          },
          el("pre", { class: "md-code", "aria-label": "mermaid source" }, el("code", {}, node.value)),
        );
      }
      const id = unitIdAt(hooks, pos?.start.offset);
      return el(
        "pre",
        {
          class: "md-code",
          "data-copy-unit": "code",
          ...(id ? { "data-unit-id": id } : {}),
          "data-unit-source": node.value,
          "data-block-id": blockId,
          ...lineAttr,
        },
        el("code", {}, node.value),
      );
    }
    case "html": {
      // Never rendered; surfaced as explicit diagnostic by parseMarkdown.
      return el("code", { class: "raw-html", ...lineAttr }, node.value);
    }
    default:
      return renderPhrasing(node as PhrasingContent, blockOffset, blockLine, hooks, blockId);
  }
}

/** Flow math (fenced $$ blocks) renders as the same display owner the
 * promoted paragraphs use, so it is visible and copyable like any
 * other display expression. */
function renderMathBlock(
  node: Extract<RootContent, { type: "math" }>,
  blockOffset: number,
  blockLine: number,
  hooks: RenderHooks,
  blockId: string,
): HTMLElement {
  const pos = docRange(node.position, blockOffset, blockLine);
  const id = unitIdAt(hooks, pos?.start.offset);
  return el(
    "span",
    {
      class: "math-display",
      "data-copy-unit": "math",
      "data-unit-display": "true",
      ...(id ? { "data-unit-id": id } : {}),
      "data-unit-tex": node.value,
      "data-unit-source": authoredMathOf(hooks, node),
      "data-pending": "math",
      "data-block-id": blockId,
      ...(pos ? { "data-line": String(pos.start.line) } : {}),
    },
    node.value,
  );
}

function renderChildren(
  children: PhrasingContent[],
  _parent: RootContent,
  blockOffset: number,
  blockLine: number,
  hooks: RenderHooks,
  blockId: string,
): (Node | string)[] {
  return children.map((child) => renderPhrasing(child, blockOffset, blockLine, hooks, blockId));
}

export function renderPhrasing(
  node: PhrasingContent,
  blockOffset: number,
  blockLine: number,
  hooks: RenderHooks,
  blockId: string,
): HTMLElement {
  const pos = docRange(node.position, blockOffset, blockLine);
  const lineAttr = pos ? { "data-line": String(pos.start.line) } : {};

  switch (node.type) {
    case "text":
      return el("span", { class: "md-text", ...lineAttr }, node.value);
    case "emphasis":
      return el("em", { ...lineAttr }, ...renderChildren(node.children as PhrasingContent[], node, blockOffset, blockLine, hooks, blockId));
    case "strong":
      return el("strong", { ...lineAttr }, ...renderChildren(node.children as PhrasingContent[], node, blockOffset, blockLine, hooks, blockId));
    case "delete":
      return el("del", { ...lineAttr }, ...renderChildren(node.children as PhrasingContent[], node, blockOffset, blockLine, hooks, blockId));
    case "inlineCode": {
      const id = unitIdAt(hooks, pos?.start.offset);
      return el("code", { class: "md-inline-code", "data-copy-unit": "code", ...(id ? { "data-unit-id": id } : {}), "data-unit-source": node.value, ...lineAttr }, node.value);
    }
    case "inlineMath": {
      const id = unitIdAt(hooks, pos?.start.offset);
      return el(
        "span",
        {
          class: "math-inline",
          "data-copy-unit": "math",
          "data-unit-display": "false",
          ...(id ? { "data-unit-id": id } : {}),
          "data-unit-tex": node.value,
          "data-unit-source": authoredMathOf(hooks, node) || `$${node.value}$`,
          "data-pending": "math",
          ...lineAttr,
        },
        `$${node.value}$`,
      );
    }
    case "link": {
      const anchor = el("a", { class: "md-link", href: node.url, title: `${node.url} (navigation is inert in the read-only viewer)`, ...lineAttr }, ...renderChildren(node.children as PhrasingContent[], node, blockOffset, blockLine, hooks, blockId));
      anchor.addEventListener("click", (event) => event.preventDefault());
      return anchor;
    }
    case "image": {
      const span = el("span", { class: "md-image", "data-image-src": node.url, "data-pending": "image", ...lineAttr });
      span.append(el("em", {}, `[image: ${node.alt || node.url}]`));
      return span;
    }
    case "break":
      return el("br", {});
    default: {
      // Unknown phrasing node: render nothing visible, surface it.
      hooks.onDiagnostic({
        code: "unknown-key",
        message: `unrendered markdown node type: ${String((node as { type?: string }).type)}`,
        line: pos?.start.line,
        column: pos?.start.column,
      });
      return el("span", { class: "md-missing" }, "");
    }
  }
}

export type { Root };
