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
 *   right edge at the anchor, centred on the block's first line;
 * - "inline": an inline target's chip (a link or ref in prose), in the
 *   room the line reserves for it just before the target (labels.css,
 *   [data-port-label-room]) - its left edge at the anchor, centred on
 *   the target's first line; the stamp takes its own space in the
 *   line, so it covers neither its target nor a neighbour;
 * - "inside": a block with no margin to stand in (a rail entry at the
 *   window's edge): select mode indents the block by the stamp's
 *   width (measure.ts reserveInlineRoom) and the chip stands in that
 *   indent, its left edge at the anchor, centred on the first line.
 * The stylesheet turns the side into the translation (labels.css). */
export type ChipSide = "margin" | "inline" | "inside";

/** Host-relative coordinates for a chip (px, against the overlay's
 * origin): the anchor point the side names. */
export interface ChipPlacement {
  readonly part: string;
  readonly label: string;
  readonly left: number;
  readonly top: number;
  readonly side: ChipSide;
}

/** A part's measured box, host-relative.
 * - `inline` marks a target that flows in a line of text (a link or a
 *   ref); its box is its first line's fragment, which begins with the
 *   room reserved for its chip.
 * - For a block, top and height are its FIRST LINE's (a block's
 *   padding is not where its text starts).
 * - `room` is the space to the box's left that is still on screen
 *   (px): the margin a chip can stand in. Defaults to `left`.
 * Width and height default to 0 (a point). */
export type PartBox = Readonly<{
  left: number;
  top: number;
  width?: number;
  height?: number;
  inline?: boolean;
  room?: number;
}>;

/** Measure a part's host-relative box; null when the part has no
 * measurable element (not rendered, folded away, or unmounted). */
export type PartMeasure = (part: string) => PartBox | null;

/** The gap between a margin chip and what it names (px). */
export const CHIP_GAP = 10;

/** The room a margin chip needs to the block's left (px): a two-key
 * stamp and its gap. Less than this and the chip would leave the
 * screen, so it goes inside. */
export const CHIP_MARGIN_ROOM = 44;

/** A chip's painted size, estimated from its label (px): the stamp is
 * mono at --port-label-size with a little padding (labels.css), so its
 * width grows by a character's advance per key. Used only to keep two
 * margin chips from stacking on one another. */
export type ChipSize = (label: string) => Readonly<{ width: number; height: number }>;

export const estimatedChipSize: ChipSize = (label) => ({ width: 12 + 9.5 * label.length, height: 21 });

/** Place every measurable chip beside - never over - what it names;
 * unmeasurable parts get NO chip (a label over nothing is a lie). Two
 * margin chips on one line (a group and its first block) stand side
 * by side, the later one further out. */
export function chipPlacements(
  chips: readonly LabelChip[],
  measure: PartMeasure,
  size: ChipSize = estimatedChipSize,
): ChipPlacement[] {
  const placements: ChipPlacement[] = [];
  const margin: Rect[] = [];
  for (const chip of chips) {
    const box = measure(chip.part);
    if (box === null) continue;
    let at = placeBeside(box);
    if (at.side === "margin") {
      const { width, height } = size(chip.label);
      let rect = marginRect(at, width, height);
      for (let guard = 0; guard < margin.length; guard++) {
        const hit = margin.find((other) => overlaps(other, rect));
        if (hit === undefined) break;
        at = { ...at, left: hit.left - CHIP_GAP / 2 };
        rect = marginRect(at, width, height);
      }
      margin.push(rect);
    }
    placements.push({ part: chip.part, label: chip.label, ...at });
  }
  return placements;
}

function placeBeside(box: PartBox): { left: number; top: number; side: ChipSide } {
  const middle = box.top + (box.height ?? 0) / 2;
  if (box.inline === true) return { left: box.left, top: middle, side: "inline" };
  if ((box.room ?? box.left) >= CHIP_MARGIN_ROOM) {
    return { left: box.left - CHIP_GAP, top: middle, side: "margin" };
  }
  return { left: box.left, top: middle, side: "inside" };
}

type Rect = Readonly<{ left: number; top: number; right: number; bottom: number }>;

function marginRect(at: { left: number; top: number }, width: number, height: number): Rect {
  return { left: at.left - width, top: at.top - height / 2, right: at.left, bottom: at.top + height / 2 };
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
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
