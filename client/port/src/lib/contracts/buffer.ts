// The buffer container's shared contract: state shapes, viewport,
// diagnostics and the latest-intent publication mechanism. No DOM,
// Svelte or transport imports.
//
// Port carry of the ratified contract shapes: the type
// declarations come from the reviewed contract drafts. The container
// implementation itself lives beside this contract in
// components/Buffer.ts; the host-facing controller interfaces that the
// construction glue supersedes (BufferSession, ConnectInstrument) are
// not restated here - the glue's BufferAccess/HostBuffer shapes are
// the buffer's public faces.

export type BufferId = string;
export type DiagnosticId = string;
export type Unsubscribe = () => void;

// A runtime symbol, not a type-only brand: the constructor below must
// exist at runtime, and strip-only type runners erase declarations.
const knownIssueBrand: unique symbol = Symbol("saccade-port.known-issue");

/** A recognized issue: centrally fixed severity, issue-specific
 * context. The slice-1 catalogue is the minimal one (code + human
 * message); it grows on demand, never per-call attention flags. */
export type KnownIssue = Readonly<
  { [knownIssueBrand]: true } & {
    code: string;
    message: string;
  }
>;

/** The producer-side constructor for recognized issues. */
export function knownIssue(detail: Readonly<{ code: string; message: string }>): KnownIssue {
  return { [knownIssueBrand]: true, code: detail.code, message: detail.message };
}

// Shared input contexts are declared in input.ts. Allowing native
// editing NEVER means dispatching to an inactive buffer underneath.
export interface InputLease {
  // Idempotent; cannot reactivate a deactivated/destroyed buffer.
  // Releases only its own claim, never a newer interaction's claim.
  release(): void;
}
export interface BufferInteractions {
  // An operation offered by the current context may enter another
  // context. This mechanism does not require every combination to be
  // offered. Releasing returns to the previous still-live context of
  // THIS buffer.
  capture(context: import("./input.ts").InputContext): InputLease;
}

export type BufferContext = Readonly<{
  bufferId: BufferId;
  windowId: string;
}>;
export interface BufferDiagnostics {
  report(context: BufferContext, issue: KnownIssue): DiagnosticId;
  unexpected(context: BufferContext, cause: unknown): DiagnosticId;
}

import type { Action, CommandBinding, InputContext } from "./input.ts";
import type { AnchorLocation } from "./places.ts";
import type { SelectSessionView } from "../select/session.ts";
import type { MenuSession, MenuView } from "../menu/session.ts";

export type Point = Readonly<{ left: number; top: number }>;
// Bounds are relative to this buffer's scrollable content origin.
export type Bounds = Readonly<{
  left: number;
  top: number;
  right: number;
  bottom: number;
}>;
export interface Viewport {
  metrics(): Readonly<{ position: Point; width: number; height: number }>;
  scrollLines(amount: number): void;
  scrollPage(direction: -1 | 1, extent: "half" | "full"): void;
  /** Scroll a FRACTION of the client height (the prototype's paging
   * steps: ^f/^b 0.9, ^d/^u and d/u 0.5 - inst16 :495/:505). */
  scrollFraction(direction: -1 | 1, fraction: number): void;
  start(): void;
  end(): void;
  reveal(bounds: Bounds): void;
  moveTo(position: Point): void;
}

// T contains domain display data, not input/viewport state. The
// container retains it opaquely. Content libraries prepare it without
// mutating the currently published payload; publication is the
// buffer's shared mechanism.
export type ContentState<T extends object> =
  | Readonly<{ kind: "empty" }>
  | Readonly<{ kind: "pending"; previous: Readonly<T> | null }>
  | Readonly<{
      kind: "available";
      value: Readonly<T>;
      lastFailure: DiagnosticId | null;
    }>
  | Readonly<{ kind: "unavailable"; failure: DiagnosticId }>;

export type BufferState<T extends object> =
  | Readonly<{
      kind: "alive";
      active: boolean;
      content: ContentState<T>;
    }>
  | Readonly<{ kind: "destroyed" }>;

// Logical content metadata is published with its data. Both types
// relocated here so lib never imports components for a type
// (audit defect e); components/Buffer re-exports them. The target
// shape is the brief's fixed one: one published target carries
// identity, kind, label, nesting, ordered acts and a durable anchor.
export type TargetAct = Readonly<{ key: string; action: Action }>;
export type TargetDefinition = Readonly<{
  /** The part path (what data-port-part observes): identity, the same
   * paths the instruments mark today. */
  id: string;
  /** What select mode narrows on. */
  kind: string;
  /** What heads the act menu. */
  label: string;
  /** The target id this one sits inside, when it has one (a rail
   * agent in its project): its label draws nested. Null at the top
   * level. */
  inside: string | null;
  /** The acts, in order. The FIRST is the primary - what a plain click
   * and Enter run; a modified click runs the act its key names (the
   * modified-click law: shift/middle is the `o` act). The ratified fields (primary,
   * commands, plainClick, modifiedClick) all read off this list. Acts
   * depend on window services and live state: resolved when asked
   * for, never frozen at publication. */
  acts: () => readonly TargetAct[];
  /** Whether the pointer's click acts on this target. True permits a
   * plain click to run the first act and a modified click (shift or
   * middle) to run the o act; false leaves the pointer only the
   * right-click menu - select, the menu and Enter are the same
   * either way. Required, with no default: no kind gets click
   * behaviour by accident. */
  clickable: boolean;
  /** What survives a refresh when the target has one (a row's task):
   * what selection follows and what Back returns to, since a part
   * path changes meaning when rows reorder. Null when none. */
  anchor: string | null;
}>;
/** The reading anchor the frame captures for Back: the published
 * target the reading position sits inside, named by its durable
 * anchor; the default when none holds it. The instrument's places
 * adapter adds the address and feeds the return places. */
export type ReadingAnchor =
  | Readonly<{ kind: "default" }>
  | Readonly<{ kind: "anchor"; anchor: string; bounds: Bounds }>;
// The frame's presentation of a published frame's targets (the ratified
// container contracts, carried beside them so lib implements the
// frame without importing components).
export interface TargetPresentation {
  activatePointer(kind: "plain" | "modified"): boolean;
  showActions(at: Point): void;
  release(): void;
}
export interface TargetPresentations {
  // Scoped to the buffer's published frame: the machinery resolves
  // identity and acts against the CURRENT publication at ask time
  // (the instruments' own sessions read their targets live too), and
  // a lease whose element the gesture cannot measure acts nothing.
  presentTarget(id: string, measure: () => Bounds | null): TargetPresentation;
  // The frame surface's face: what the frame's components
  // render from the leases the instrument's view receives - the select
  // session's labels, the act menu's view, session and input context,
  // the frame's reading context (the shared keys, defined once), the
  // reading anchor Back rides, and the selection's durable anchor.
  selectView(): SelectSessionView | null;
  menuView(): MenuView | null;
  menuSession(): MenuSession;
  subscribe(observer: () => void): Unsubscribe;
  readingContext(extra?: readonly CommandBinding[]): InputContext;
  readingAnchor(): ReadingAnchor;
  /** Where a published anchor sits: found with its bounds, missing
   * when no published target carries it, pending when the target is
   * published but its geometry is not usable yet - the restoration's
   * own three answers, from the frame. */
  locateAnchor(anchor: string): import("./places.ts").AnchorLocation;
  /** The anchor of the label-selected target (a row's task), while a
   * published target still carries it: what selection follows across
   * a refresh. Null when the selection is anchor-less or its anchor
   * has left the published targets. */
  selectedAnchor(): string | null;
}
export interface TargetGeometry {
  bounds(id: string): Bounds | null;
  subscribeChanges(observer: () => void): Unsubscribe;
}
export type ContentFrame<T extends object> = Readonly<{
  value: Readonly<T>;
  targets: readonly TargetDefinition[];
}>;
export type PreparedContent<T extends object> =
  | Readonly<{ kind: "ready"; value: Readonly<T> }>
  | Readonly<{ kind: "failed"; issue: KnownIssue }>;
export type RefreshOutcome = "applied" | "failed" | "superseded" | "destroyed";
export interface ContentUpdates<T extends object> {
  // Update the displayed value (or the previous value while loading).
  // This is synchronous state updating, never queue invalidation. No
  // displayed value means no callback; return the same value for no change.
  updateDisplayed(update: (current: Readonly<T>) => Readonly<T>): void;
  // Load/replace content through the buffer's latest-request queue.
  // Ordinary display changes use updateDisplayed, not this queue.
  // A synchronous ready result needs no pending display frame.
  refresh(
    prepare: () => PreparedContent<T> | Promise<PreparedContent<T>>,
  ): Promise<RefreshOutcome>;
}
