// Return places: pure data + the ports the buffer container implements.
// Shared record types; restoration machinery lives in the container
// (components/Buffer.ts). Content-specific anchor types/adapters belong
// to the instrument's companion module.
//
// Port carry of the ratified contract shapes (t-36 r47-r51). The
// slice-1 container implements the trivially-correct subset: capture
// records default positions only (no anchor machinery yet), restore
// requests the address through the normal refresh pipeline and reports
// restored/defaulted/failed honestly - never a fake fallback. Full
// anchor capture/reveal lands with the Board instrument slice.

import type {
  Bounds,
  BufferId,
  Point,
  RefreshOutcome,
  Unsubscribe,
  Viewport,
} from "./buffer.ts";
type ViewportMetrics = ReturnType<Viewport["metrics"]>;

// Compatibility shape, not a new stored location grammar. A board
// revision is an index within its board location, currently encoded in
// path.
export type Address = Readonly<{ machine: string; path: string }>;
export type AnchorData =
  | string | number | boolean | null
  | readonly AnchorData[]
  | Readonly<{ [key: string]: AnchorData }>;
export type SavedPosition<A extends AnchorData> =
  | Readonly<{ kind: "default" }>
  | Readonly<{ kind: "anchor"; anchor: A; viewportOffset: Point }>;
export type SavedPlace<A extends AnchorData = AnchorData> = Readonly<{
  bufferId: BufferId;
  address: Address;
  position: SavedPosition<A>;
}>;

// Content supplies the displayed address and semantic anchor; geometry
// comes from the buffer's presentation service. All facts must refer to
// the same actual presentation, never a later requested-but-unshown
// payload.
export type ContentCapture<A extends AnchorData> = Readonly<{
  address: Address;
  position:
    | Readonly<{ kind: "default" }>
    | Readonly<{ kind: "anchor"; anchor: A; bounds: Bounds }>;
}>;
export type AnchorLocation =
  | Readonly<{ kind: "pending" }>
  | Readonly<{ kind: "found"; bounds: Bounds }>
  | Readonly<{ kind: "missing" }>;

// Private adapter supplied to Buffer.ts through instrument attachment.
// It is NOT the rendering component's controller interface.
export interface ContentPlaces<A extends AnchorData> {
  captureDisplayed(viewport: ViewportMetrics): ContentCapture<A>;
  // Request the captured addressed place using the normal refresh
  // pipeline. This lower-level operation neither advances a trail nor
  // enters history.
  request(address: Address): Promise<RefreshOutcome>;
  acceptsAnchor(data: AnchorData): data is A;
  // Expose the actual anchor: unfold enclosing visual containers as
  // needed, then report pending until resulting geometry is usable.
  // Never substitute the folded container for the requested passage or
  // reset unrelated state.
  revealAnchor(anchor: A): AnchorLocation;
  defaultPosition(viewport: ViewportMetrics): Point;
  subscribeChanges(observer: () => void): Unsubscribe;
}

export type RestoreStatus =
  | "pending-content"
  | "pending-layout"
  | "restored"
  | "defaulted"
  | "failed"
  | "superseded"
  | "cancelled";
export interface RestoringPlace {
  snapshot(): RestoreStatus;
  subscribe(observer: (status: RestoreStatus) => void): Unsubscribe;
  // Cancels this placement job, not the buffer or retained draft work.
  cancel(): void;
}

// Host-facing, non-domain-specific facade. Restoration checks buffer
// ownership and validates anchor data with the attached adapter before
// using it.
export interface ReturnPlaces {
  capture(): SavedPlace;
  // One operation schedules content request + layout/anchor placement.
  // The App caller must not implement a retry/measurement/polling loop.
  restore(place: SavedPlace): RestoringPlace;
}

export type BoardAnchor = Readonly<{ kind: "board-part"; path: string }>;
export type TaskAnchor = Readonly<{ kind: "task"; task: string }>;
