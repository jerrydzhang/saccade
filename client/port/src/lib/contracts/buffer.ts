// The buffer container's shared contract: state shapes, viewport,
// diagnostics and the latest-intent publication mechanism. No DOM,
// Svelte or transport imports.
//
// Port carry of the ratified contract shapes (t-36 r47-r51): the type
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
