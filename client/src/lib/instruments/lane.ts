// The activity lane's pure half (t-14, the tasks instrument's `e`):
// the task's events on a 6h/24h axis, the last 6 hours taking two
// thirds. The page's history timeline is the source: each entry rides
// its record's moment; the axis holds the last 24 hours, and events
// older than the axis clamp at its left edge (the list below carries
// them whole). Positions are fractions of the axis width, so the
// renderer stays thin and the layout stays testable.

export type LaneEntry = {
  at: number;
  /** The entry's own line: the event, or the message behind it. */
  label: string;
  kind: string;
};

export type LanePosition = {
  entry: LaneEntry;
  /** The fraction of the axis width, 0 to 1. */
  x: number;
  /** True when the entry predates the axis and clamped to its edge. */
  clamped: boolean;
};

export type LaneLayout = {
  positions: LanePosition[];
  /** Where the last six hours begin, as a fraction: one third, the
   * last six hours taking the two thirds to its right. */
  sixHourMark: number;
  spanHours: number;
};

const HOUR = 60 * 60;
const SPAN_HOURS = 24;
const RECENT_HOURS = 6;

/** The layout of one task's events against now: the last 6 hours
 * take the right two thirds of the axis ([1/3, 1]), the preceding 18
 * hours the left third ([0, 1/3]). Events outside the span clamp at
 * the left edge. */
export function laneLayout(entries: LaneEntry[], now: number): LaneLayout {
  const span = SPAN_HOURS * HOUR;
  const recent = RECENT_HOURS * HOUR;
  const start = now - span;
  const recentStart = now - recent;
  const olderSpan = span - recent; // 18h
  const olderFraction = 1 / 3;
  const positions = entries.map((entry) => {
    if (entry.at <= start) {
      return { entry, x: 0, clamped: entry.at < start };
    }
    if (entry.at >= recentStart) {
      const x = olderFraction + ((entry.at - recentStart) / recent) * (1 - olderFraction);
      return { entry, x, clamped: false };
    }
    const x = ((entry.at - start) / olderSpan) * olderFraction;
    return { entry, x, clamped: false };
  });
  return { positions, sixHourMark: olderFraction, spanHours: SPAN_HOURS };
}
