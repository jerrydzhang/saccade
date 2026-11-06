// DOM adapter for copy semantics: turns a live selection into segments of
// unit descriptors for serializeSelection. Math and code units are atomic;
// prose keeps only its intersecting text. The serialized text then replaces
// the clipboard payload.

import { serializeSelection, type CopyMode, type CopySegment, type CopyUnit } from "./copy.ts";

export function selectionSegments(mode: CopyMode): CopySegment[] {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return [];
  const segments: CopySegment[] = [];

  for (let r = 0; r < selection.rangeCount; r++) {
    const range = selection.getRangeAt(r);
    const units = Array.from(document.querySelectorAll<HTMLElement>("[data-copy-unit]")).filter((node) => range.intersectsNode(node));
    if (units.length === 0) {
      const text = range.toString();
      if (text.trim().length > 0) {
        segments.push({ blockId: "selection", units: [{ id: "sel0", kind: "prose", text, source: text, touched: true, covered: true }] });
      }
      continue;
    }
    // Document-order walk: prose between unit boundaries comes from
    // sub-ranges, so MathJax internals never leak into copied prose.
    let cursor = range.cloneRange();
    const current: CopyUnit[] = [];
    let blockId = "selection";
    const flush = (proseText: string): void => {
      if (proseText.trim().length > 0) {
        current.push({ id: `sel${current.length}`, kind: "prose", text: proseText, source: proseText, touched: true, covered: true });
      }
    };
    units.forEach((unitEl, index) => {
      const blockHost = unitEl.closest("[data-block-id]");
      const thisBlock = blockHost?.getAttribute("data-block-id") ?? "selection";
      if (thisBlock !== blockId) {
        flush(cursor.toString());
        segments.push({ blockId, units: current.splice(0) });
        blockId = thisBlock;
      }
      const unitStart = document.createRange();
      unitStart.setStart(range.startContainer, range.startOffset);
      unitStart.setEnd(unitEl, 0);
      const before = unitStart.toString();
      flush(before);
      const kind = (unitEl.getAttribute("data-copy-unit") ?? "code") as CopyUnit["kind"];
      const covered = range.comparePoint(unitEl, 0) >= 0 && range.comparePoint(unitEl, unitEl.childNodes.length) <= 0;
      current.push({
        id: (unitEl.getAttribute("data-unit-id") ?? `u${index}`) as string,
        kind,
        text: kind === "math" ? (unitEl.getAttribute("data-unit-tex") ?? "") : (unitEl.textContent ?? ""),
        source: unitEl.getAttribute("data-unit-source") ?? unitEl.textContent ?? "",
        tex: unitEl.getAttribute("data-unit-tex") ?? undefined,
        touched: true,
        covered,
      });
      cursor = document.createRange();
      cursor.setStartAfter(unitEl);
      cursor.setEnd(range.endContainer, range.endOffset);
    });
    flush(cursor.toString());
    if (current.length > 0) {
      segments.push({ blockId, units: current });
    }
  }
  return segments;
}

export function installCopyHandler(getMode: () => CopyMode): () => void {
  const handler = (event: ClipboardEvent): void => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
    const segments = selectionSegments(getMode());
    if (segments.length === 0) return;
    const text = serializeSelection(segments, getMode());
    if (text.length === 0) return;
    event.clipboardData?.setData("text/plain", text);
    event.preventDefault();
  };
  document.addEventListener("copy", handler);
  return () => document.removeEventListener("copy", handler);
}
