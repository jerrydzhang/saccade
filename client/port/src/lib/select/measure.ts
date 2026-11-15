// The label overlay's measurement: a part's box relative to the
// instrument's content host - the overlay's origin (overlay.ts states
// the placement contract). The DOM half of the overlay, shared by
// every instrument that paints labels.

import type { PartMeasure } from "./overlay.ts";

export function partMeasureIn(host: HTMLElement | null): PartMeasure {
  return (part) => {
    if (host === null) return null;
    const target = host.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
    if (!(target instanceof HTMLElement)) return null;
    const hostBox = host.getBoundingClientRect();
    // a target flowing in a line of text (a link or ref) is measured
    // by its first line's fragment: its chip stands before it there
    const inline = getComputedStyle(target).display === "inline";
    const box = (inline ? target.getClientRects()[0] : undefined) ?? target.getBoundingClientRect();
    return {
      left: box.left - hostBox.left,
      top: box.top - hostBox.top,
      width: box.width,
      height: box.height,
      inline,
    };
  };
}
