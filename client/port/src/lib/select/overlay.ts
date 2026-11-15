// The label overlay's placement model (order of work step 3):
// where a chip sits for a measured part. Pure math, testable in
// node; the Svelte component composes over it. The model exists
// because F3 drew every chip a full content-height low - placement
// must be computed against the RIGHT origin (the content host) and
// painted inside a layer that COVERS the host, and that contract
// is checked here and in the lane, not assumed.

/** A part to label: the selection session's entry. */
export interface LabelChip {
  readonly part: string;
  readonly label: string;
}

/** Where a chip stands against its anchor point (left, top):
 * - "margin": a block's chip, in the margin to the block's left - its
 *   right edge at the anchor, its top level with the block's top;
 * - "before": an inline target's chip (a link or ref in prose), just
 *   before the target on its first line - its right edge at the
 *   anchor, centred on the line, so the target's own text stays clear;
 * - "inside": a block with no margin to stand in (its left edge at the
 *   host's): the chip sits over the block's top-left corner, the only
 *   place left on screen. The stylesheet turns the side into the
 *   translation (labels.css). */
export type ChipSide = "margin" | "before" | "inside";

/** Host-relative coordinates for a chip (px, against the content
 * host's top-left, which is also the overlay's origin when the
 * overlay covers the host with inset 0): the anchor point the side
 * names. */
export interface ChipPlacement {
  readonly part: string;
  readonly label: string;
  readonly left: number;
  readonly top: number;
  readonly side: ChipSide;
}

/** A part's measured box, host-relative. `inline` marks a target
 * that flows in a line of text (a link or ref); its box is its first
 * line's fragment. Width and height default to 0 (a point). */
export type PartBox = Readonly<{
  left: number;
  top: number;
  width?: number;
  height?: number;
  inline?: boolean;
}>;

/** Measure a part's host-relative box; null when the part has no
 * measurable element (not rendered, folded away, or unmounted). */
export type PartMeasure = (part: string) => PartBox | null;

/** The gap between a chip and what it names (px). */
export const CHIP_GAP = 6;

/** The room a margin chip needs to the block's left (px): a two-key
 * stamp and its gap. Less than this and the chip would leave the
 * host, so it goes inside. */
export const CHIP_MARGIN_ROOM = 40;

/** A chip's painted size, estimated from its label (px): the stamp is
 * mono at --port-label-size with a little padding (labels.css), so its
 * width grows by a character's advance per key. Used only to keep two
 * chips from stacking on one another. */
export type ChipSize = (label: string) => Readonly<{ width: number; height: number }>;

export const estimatedChipSize: ChipSize = (label) => ({ width: 12 + 9.5 * label.length, height: 21 });

/** Place every measurable chip beside - never over - what it names;
 * unmeasurable parts get NO chip (a label over nothing is a lie).
 * Inline targets are placed first, so a link that opens its block
 * keeps the chip nearest it and the block's own chip steps further
 * into the margin rather than sitting on it. */
export function chipPlacements(
  chips: readonly LabelChip[],
  measure: PartMeasure,
  size: ChipSize = estimatedChipSize,
): ChipPlacement[] {
  const measured: { chip: LabelChip; box: PartBox; order: number }[] = [];
  chips.forEach((chip, order) => {
    const box = measure(chip.part);
    if (box !== null) measured.push({ chip, box, order });
  });
  const placed: { order: number; placement: ChipPlacement; rect: Rect }[] = [];
  const inlineFirst = [...measured].sort(
    (a, b) => Number(b.box.inline === true) - Number(a.box.inline === true) || a.order - b.order,
  );
  for (const { chip, box, order } of inlineFirst) {
    const { width, height } = size(chip.label);
    let at = placeBeside(box);
    let rect = rectOf(at, width, height);
    // step left, clear of every chip already standing there
    for (let guard = 0; at.side !== "inside" && guard < placed.length; guard++) {
      const hit = placed.find((other) => overlaps(other.rect, rect));
      if (hit === undefined) break;
      at = { ...at, left: hit.rect.left - CHIP_GAP / 2 };
      rect = rectOf(at, width, height);
    }
    placed.push({ order, placement: { part: chip.part, label: chip.label, ...at }, rect });
  }
  return placed.sort((a, b) => a.order - b.order).map((entry) => entry.placement);
}

type Rect = Readonly<{ left: number; top: number; right: number; bottom: number }>;

function rectOf(at: { left: number; top: number; side: ChipSide }, width: number, height: number): Rect {
  if (at.side === "inside") return { left: at.left, top: at.top, right: at.left + width, bottom: at.top + height };
  const top = at.side === "before" ? at.top - height / 2 : at.top;
  return { left: at.left - width, top, right: at.left, bottom: top + height };
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

function placeBeside(box: PartBox): { left: number; top: number; side: ChipSide } {
  if (box.inline === true) {
    return { left: box.left - CHIP_GAP / 2, top: box.top + (box.height ?? 0) / 2, side: "before" };
  }
  if (box.left >= CHIP_MARGIN_ROOM) {
    return { left: box.left - CHIP_GAP, top: box.top, side: "margin" };
  }
  return { left: box.left, top: box.top, side: "inside" };
}

/** The visibility contract (the F3 lesson): a placement is visible
 * when its point sits inside the host's content box. The host box
 * here is the host's own size (width/height); a chip outside it
 * would draw off-screen or over the wrong content. */
export function isWithinHost(
  placement: Readonly<{ left: number; top: number }>,
  host: Readonly<{ width: number; height: number }>,
): boolean {
  return (
    placement.left >= 0 &&
    placement.top >= 0 &&
    placement.left < host.width &&
    placement.top < host.height
  );
}
