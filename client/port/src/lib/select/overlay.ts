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

/** Host-relative coordinates for a chip (px, against the content
 * host's top-left, which is also the overlay's origin when the
 * overlay covers the host with inset 0). */
export interface ChipPlacement {
  readonly part: string;
  readonly label: string;
  readonly left: number;
  readonly top: number;
}

/** Measure a part's host-relative box; null when the part has no
 * measurable element (not rendered, folded away, or unmounted). */
export type PartMeasure = (part: string) => { left: number; top: number } | null;

/** Place every measurable chip; unmeasurable parts get NO chip
 * (a label over nothing is a lie). */
export function chipPlacements(
  chips: readonly LabelChip[],
  measure: PartMeasure,
): ChipPlacement[] {
  const placements: ChipPlacement[] = [];
  for (const chip of chips) {
    const box = measure(chip.part);
    if (box === null) continue;
    placements.push({ part: chip.part, label: chip.label, left: box.left, top: box.top });
  }
  return placements;
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
