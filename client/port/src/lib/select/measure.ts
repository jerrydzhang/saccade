// The label overlay's measurement: a part's box relative to the
// instrument's content host - the overlay's origin (overlay.ts states
// the placement contract) - and the room select mode reserves in a
// line for an inline target's chip. The DOM half of the overlay,
// shared by every instrument that paints labels.

import { CHIP_GAP, CHIP_MARGIN_ROOM, type PartMeasure } from "./overlay.ts";

/** A part's own element under `root` (never a chip, which carries
 * the part it names). */
function partElement(root: ParentNode, part: string): HTMLElement | null {
  const found = root.querySelector(`[data-port-part=${JSON.stringify(part)}]:not(.port-label-chip)`);
  return found instanceof HTMLElement ? found : null;
}

function isInline(element: Element): boolean {
  return getComputedStyle(element).display === "inline";
}

/** The element that scrolls (and so clips) what `element` shows. */
function scrollerOf(element: Element): HTMLElement | null {
  for (let node = element.parentElement; node !== null; node = node.parentElement) {
    const style = getComputedStyle(node);
    if (/(auto|scroll|hidden|clip)/.test(`${style.overflowX} ${style.overflowY}`)) return node;
  }
  return null;
}

const NOT_TEXT = new Set(["svg", "img", "canvas", "video", "iframe", "object"]);

/** A block's first line of text (client coordinates), or null when
 * something that is not text (a drawing, an image) comes first. */
function firstLine(block: HTMLElement): DOMRect | null {
  const walker = block.ownerDocument.createTreeWalker(block, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    if (node instanceof Element) {
      if (NOT_TEXT.has(node.localName)) return null;
      continue;
    }
    if ((node.textContent ?? "").trim() === "") continue;
    const range = block.ownerDocument.createRange();
    range.selectNodeContents(node);
    const rect = range.getClientRects()[0];
    if (rect !== undefined && rect.height > 0) return rect;
  }
  return null;
}

export function partMeasureIn(host: HTMLElement | null): PartMeasure {
  return (part) => {
    if (host === null) return null;
    const target = partElement(host, part);
    if (target === null) return null;
    const hostBox = host.getBoundingClientRect();
    const clip = scrollerOf(target)?.getBoundingClientRect().left ?? 0;
    // a target flowing in a line of text (a link or ref) is measured
    // by its first line's fragment, which begins with its chip's room
    if (isInline(target)) {
      const box = target.getClientRects()[0] ?? target.getBoundingClientRect();
      return {
        left: box.left - hostBox.left,
        top: box.top - hostBox.top,
        width: box.width,
        height: box.height,
        inline: true,
        room: box.left - clip,
      };
    }
    // a block's chip stands beside its first line, not its padding
    const box = target.getBoundingClientRect();
    const style = getComputedStyle(target);
    const contentTop = box.top + parseFloat(style.paddingTop || "0") + parseFloat(style.borderTopWidth || "0");
    const line = firstLine(target);
    const lineHeight = parseFloat(style.lineHeight);
    const indent = target.getAttribute("data-port-label-indent");
    return {
      left: box.left - hostBox.left + (indent === null ? 0 : Number(indent)),
      top: (line?.top ?? contentTop) - hostBox.top,
      width: box.width,
      height: line?.height ?? (Number.isFinite(lineHeight) ? lineHeight : 0),
      // an indented block (reserveInlineRoom) holds its chip inside
      room: indent === null ? box.left - clip : 0,
    };
  };
}

/** Keep the reading place while `change` relays the text: the first
 * part at or below the top of the view stays where it was on screen,
 * so lines rewrapping above or around it never move what is read. */
export function keepReadingPlace(root: HTMLElement, change: () => void): void {
  const scroller = scrollerOf(root);
  if (scroller === null) {
    change();
    return;
  }
  const viewTop = scroller.getBoundingClientRect().top;
  let anchor: HTMLElement | null = null;
  for (const element of root.querySelectorAll<HTMLElement>("[data-port-part]:not(.port-label-chip)")) {
    if (isInline(element)) continue;
    if (element.getBoundingClientRect().top >= viewTop) {
      anchor = element;
      break;
    }
  }
  const before = anchor?.getBoundingClientRect().top ?? null;
  change();
  if (anchor === null || before === null) return;
  const moved = anchor.getBoundingClientRect().top - before;
  if (moved !== 0) scroller.scrollTop += moved;
}

/** The painted width of a stamp for `label` (px), measured on a real
 * chip so the room matches the stamp exactly. */
function stampWidth(root: HTMLElement, label: string): number {
  const probe = root.ownerDocument.createElement("span");
  probe.className = "port-label-chip";
  probe.style.visibility = "hidden";
  probe.textContent = label;
  root.appendChild(probe);
  const width = probe.getBoundingClientRect().width;
  probe.remove();
  return width;
}

/** Select mode's room for chips (t-59), so a stamp never covers its
 * target or a neighbour:
 * - each inline target (a link or a ref in prose) gets room just
 *   before it in the line ([data-port-label-room], labels.css);
 * - a block with no margin on screen to its left (a rail entry at the
 *   window's edge) is indented by the stamp's width, its chip standing
 *   where its text began.
 * The reading place holds while the lines relay (keepReadingPlace),
 * and again when the room is released (the returned function). */
export function reserveInlineRoom(
  root: HTMLElement,
  chips: readonly { readonly part: string; readonly label: string }[],
): () => void {
  const rooms: [HTMLElement, string][] = [];
  const indents: [HTMLElement, number, number][] = [];
  for (const chip of chips) {
    const element = partElement(root, chip.part);
    if (element === null) continue;
    if (isInline(element)) {
      rooms.push([element, chip.label]);
      continue;
    }
    const clip = scrollerOf(element)?.getBoundingClientRect().left ?? 0;
    if (element.getBoundingClientRect().left - clip < CHIP_MARGIN_ROOM) {
      const padding = parseFloat(getComputedStyle(element).paddingLeft || "0");
      indents.push([element, padding, stampWidth(root, chip.label)]);
    }
  }
  // every indented block makes the SAME room - the widest stamp's - so
  // rows that shared a column keep sharing it (t-63: each row indented
  // by its own stamp moved the rail's rows unevenly)
  const room = Math.max(0, ...indents.map(([, , width]) => width)) + CHIP_GAP;
  for (const entry of indents) entry[2] = entry[1] + room;
  if (rooms.length === 0 && indents.length === 0) return () => {};
  const previous = new Map<HTMLElement, string>();
  keepReadingPlace(root, () => {
    for (const [element, label] of rooms) element.setAttribute("data-port-label-room", label);
    for (const [element, padding, indented] of indents) {
      previous.set(element, element.style.getPropertyValue("padding-left"));
      // the attribute keeps where the text began: the chip stands there
      element.setAttribute("data-port-label-indent", String(padding));
      element.style.setProperty("padding-left", `${indented}px`);
    }
  });
  return () => {
    keepReadingPlace(root, () => {
      for (const [element] of rooms) element.removeAttribute("data-port-label-room");
      for (const [element] of indents) {
        element.removeAttribute("data-port-label-indent");
        const before = previous.get(element) ?? "";
        if (before === "") element.style.removeProperty("padding-left");
        else element.style.setProperty("padding-left", before);
      }
    });
  };
}
