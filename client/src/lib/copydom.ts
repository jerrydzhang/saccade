// DOM adapter for copy semantics: turns a live selection into segments of
// unit descriptors for serializeSelection. Math and code units are atomic;
// prose pieces are assembled only from the actual selected intervals
// BETWEEN unit boundaries, scoped to real block boundaries - a block
// change flushes the group assembled so far, never future selection, and
// the trailing interval after the last unit belongs to that unit's block.
// MathJax internals never leak: sub-ranges stop at unit boundaries.

import { serializeSelection, type CopyMode, type CopySegment, type CopyUnit } from "./copy.ts";

type Point = { node: Node; offset: number };

function textBetween(range: Range, from: Point, to: Point): string {
  const probe = document.createRange();
  try {
    probe.setStart(from.node, from.offset);
    probe.setEnd(to.node, to.offset);
  } catch {
    return "";
  }
  if (probe.collapsed) return "";
  return probe.toString();
}

function pointRange(from: Point, to: Point): Range | null {
  const probe = document.createRange();
  try {
    probe.setStart(from.node, from.offset);
    probe.setEnd(to.node, to.offset);
  } catch {
    return null;
  }
  return probe;
}

export function selectionSegments(): CopySegment[] {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return [];
  const segments: CopySegment[] = [];

  for (let r = 0; r < selection.rangeCount; r++) {
    const range = selection.getRangeAt(r);
    const units = Array.from(document.querySelectorAll<HTMLElement>("[data-copy-unit]")).filter((node) =>
      range.intersectsNode(node),
    );
    if (units.length === 0) {
      const text = range.toString();
      if (text.trim().length > 0) {
        segments.push({
          blockId: "selection",
          units: [{ id: "sel0", kind: "prose", text, source: text, touched: true, covered: true }],
        });
      }
      continue;
    }

    const startPoint: Point = { node: range.startContainer, offset: range.startOffset };
    const endPoint: Point = { node: range.endContainer, offset: range.endOffset };
    let cursor: Point = startPoint;
    let currentBlock: string | null = null;
    let currentUnits: CopyUnit[] = [];

    const flush = (): void => {
      if (currentUnits.length > 0) {
        segments.push({ blockId: currentBlock ?? "selection", units: currentUnits.splice(0) });
      }
      currentBlock = null;
    };

    const blockOf = (unitEl: HTMLElement): string =>
      unitEl.closest("[data-block-id]")?.getAttribute("data-block-id") ?? "selection";

    const pushProse = (text: string): void => {
      if (text.length > 0) {
        currentUnits.push({
          id: `prose${currentUnits.length}`,
          kind: "prose",
          text,
          source: text,
          touched: true,
          covered: true,
        });
      }
    };

    units.forEach((unitEl, index) => {
      const unitStart: Point = { node: unitEl, offset: 0 };
      const unitBlock = blockOf(unitEl);

      // prose strictly between the moving cursor and this unit's start,
      // within the range
      const beforeRange = pointRange(cursor, unitStart);
      const before = beforeRange ? beforeRange.toString() : "";

      // a real block boundary flushes the group assembled so far
      if (currentUnits.length > 0 && unitBlock !== currentBlock) {
        flush();
      }
      if (currentBlock === null) {
        currentBlock = unitBlock;
      }
      pushProse(before);

      const kind = (unitEl.getAttribute("data-copy-unit") ?? "code") as CopyUnit["kind"];
      let covered = false;
      try {
        covered =
          range.comparePoint(unitEl, 0) >= 0 &&
          range.comparePoint(unitEl, unitEl.childNodes.length) <= 0;
      } catch {
        covered = false;
      }
      currentUnits.push({
        id: (unitEl.getAttribute("data-unit-id") ?? `u${index}`) as string,
        kind,
        text: kind === "math" ? (unitEl.getAttribute("data-unit-tex") ?? "") : (unitEl.textContent ?? ""),
        source: unitEl.getAttribute("data-unit-source") ?? unitEl.textContent ?? "",
        tex: unitEl.getAttribute("data-unit-tex") ?? undefined,
        touched: true,
        covered,
      });

      // the cursor moves to just after this unit
      cursor = { node: unitEl.parentNode ?? unitEl, offset: unitIndexInParent(unitEl) + 1 };
    });

    // trailing prose after the last unit, inside the range
    const trailing = textBetween(range, cursor, endPoint);
    pushProse(trailing);
    flush();
  }
  return segments;
}

function unitIndexInParent(el: HTMLElement): number {
  let index = 0;
  let node = el.previousSibling;
  while (node) {
    index += 1;
    node = node.previousSibling;
  }
  return index;
}

export type CopyHandlerOptions = {
  /** When present, Ctrl+C writes the payload through the native shell
   * bridge and suppresses the engine's copy command - the native
   * WebKitGTK never dispatches the copy event (execCommand copy is
   * false there), so the bridge is the only working write path. */
  bridgeWrite?: (text: string) => Promise<boolean>;
  /** Observes every handled copy, in product order. */
  onHandled?: (mode: CopyMode, text: string) => void;
};

export function installCopyHandler(getMode: () => CopyMode, options: CopyHandlerOptions = {}): () => void {
  const payload = (): string => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) return "";
    const segments = selectionSegments();
    if (segments.length === 0) return "";
    return serializeSelection(segments, getMode());
  };
  const handler = (event: ClipboardEvent): void => {
    const text = payload();
    if (text.length === 0) return;
    event.clipboardData?.setData("text/plain", text);
    event.preventDefault();
    options.onHandled?.(getMode(), text);
  };
  document.addEventListener("copy", handler);
  const keydown = (event: KeyboardEvent): void => {
    if (!options.bridgeWrite) return;
    if (!(event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === "c")) return;
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
    const text = payload();
    if (text.length === 0) return;
    event.preventDefault();
    options.onHandled?.(getMode(), text);
    void options.bridgeWrite(text);
  };
  if (options.bridgeWrite) window.addEventListener("keydown", keydown);
  return () => {
    document.removeEventListener("copy", handler);
    if (options.bridgeWrite) window.removeEventListener("keydown", keydown);
  };
}
