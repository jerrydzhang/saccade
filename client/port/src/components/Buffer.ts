// The buffer container: the common container for viewport/scrolling,
// loading and interaction context - not an owner of domain-specific
// operations. This module carries the container's contracts from the
// ratified construction package (t-36 r49) and, below them, the
// slice-1 container implementation.
//
// Slice-1 honesty notes (what is real, what waits):
// - real: state transitions (empty/pending/available/unavailable/
//   destroyed), latest-intent refresh with generation checks,
//   retention of the last usable frame on failure, activation flags,
//   input-lease capture, idempotent destruction with attachment
//   release, default-position return places over the normal refresh
//   pipeline.
// - minimal-but-real: the viewport is the DOM scroller element the
//   window's presentation attaches; a slice-1 named constant
//   SCROLL_LINE_PX stands in for the line height (a tunable default
//   per r49: later tasks change the VALUE mechanically).
// - waiting for later slices: target presentations (no targets exist
//   yet), the geometry service (no measurement machinery yet), key
//   matching (no bindings installed; handleKey answers allow-default),
//   anchor capture/reveal (capture records default positions only).
//   Each reports what it cannot yet do instead of pretending.

import type {
  Bounds,
  BufferContext,
  BufferDiagnostics,
  BufferId,
  BufferState,
  ContentState,
  ContentUpdates,
  DiagnosticId,
  InputLease,
  KnownIssue,
  Point,
  PreparedContent,
  RefreshOutcome,
  Unsubscribe,
  Viewport,
} from "../lib/contracts/buffer.ts";
import { knownIssue } from "../lib/contracts/buffer.ts";
import type {
  Action,
  InputContext,
  KeyDisposition,
  KeyInput,
  ReadCommands,
} from "../lib/contracts/input.ts";
import type {
  AnchorData,
  ContentCapture,
  ContentPlaces,
  RestoreStatus,
  RestoringPlace,
  ReturnPlaces,
  SavedPlace,
} from "../lib/contracts/places.ts";
import type { Component } from "svelte";

// Logical content metadata is published with its data.
export type TargetDefinition = Readonly<{
  id: string;
  primary: Action;
  commands: ReadCommands;
  // Pointer bindings reference actions; absence is not a hidden default act.
  plainClick: Action | null;
  modifiedClick: Action | null;
}>;
export type ContentFrame<T extends object> = Readonly<{
  value: Readonly<T>;
  targets: readonly TargetDefinition[];
}>;
export interface TargetPresentation {
  activatePrimary(): void;
  activatePointer(kind: "plain" | "modified"): boolean;
  showActions(at: Point): void;
  release(): void;
}
export interface TargetPresentations {
  // Scoped to one published frame. Shared machinery resolves identity/actions
  // and schedules measurement together; caller doesn't patch later geometry.
  presentTarget(id: string, measure: () => Bounds | null): TargetPresentation;
}
export interface TargetGeometry {
  bounds(id: string): Bounds | null;
  subscribeChanges(observer: () => void): Unsubscribe;
}
export interface InputCapture {
  capture(context: InputContext): InputLease;
}
export interface BufferAccess<T extends object> {
  // Minted by the window host for this instance, not chosen by an instrument.
  readonly context: BufferContext;
  snapshot(): BufferState<ContentFrame<T>>;
  subscribe(observer: (state: BufferState<ContentFrame<T>>) => void): Unsubscribe;
  readonly viewport: Viewport;
  readonly geometry: TargetGeometry;
  readonly updates: ContentUpdates<ContentFrame<T>>;
  readonly interactions: InputCapture;
  setReadingContext(context: InputContext): void;
}
export interface InstrumentAttachment<UI extends object, A extends AnchorData> {
  readonly presentation: UI;
  readonly input: InputContext;
  readonly places: ContentPlaces<A>;
  release(): void;
}
export interface HostBuffer<T extends object> {
  readonly id: string;
  readonly viewport: Viewport;
  readonly places: ReturnPlaces;
  snapshot(): BufferState<ContentFrame<T>>;
  subscribe(observer: (state: BufferState<ContentFrame<T>>) => void): Unsubscribe;
  activate(): boolean;
  deactivate(): void;
  handleKey(input: KeyInput): KeyDisposition;
  destroy(): void;
  // Same-frame data + target definitions must remain associated. A repeated
  // query is not another registration or content change. Old frame facades
  // cannot publish stale geometry or activate retargeted operations.
  presentationsFor(frame: ContentFrame<T>): TargetPresentations;
}
export type BufferBuild<T extends object, UI extends object, A extends AnchorData> = Readonly<{
  context: BufferContext;
  viewport: Viewport;
  diagnostics: BufferDiagnostics;
  connect: (buffer: BufferAccess<T>) => InstrumentAttachment<UI, A>;
}>;
export type BuiltBuffer<T extends object, UI extends object> = Readonly<{
  host: HostBuffer<T>;
  presentation: UI;
}>;
export interface BufferFactory {
  // Controller construction calls connect synchronously exactly once.
  // Buffer owns attachment input/places/release; caller gets UI + host facade.
  create<T extends object, UI extends object, A extends AnchorData>(
    build: BufferBuild<T, UI, A>,
  ): BuiltBuffer<T, UI>;
}

// components/Buffer.ts: generic read-only projection. Loading behaviour isn't
// reimplemented per instrument; only the payload is projected for rendering.
export function mapBufferState<X extends object, Y extends object>(
  state: BufferState<X>,
  project: (value: Readonly<X>) => Readonly<Y>,
): BufferState<Y> {
  if (state.kind === "destroyed") return state;
  const content = state.content;
  switch (content.kind) {
    case "empty":
    case "unavailable":
      return { ...state, content };
    case "pending":
      return {
        ...state,
        content: {
          kind: "pending",
          previous: content.previous === null ? null : project(content.previous),
        },
      };
    case "available":
      return {
        ...state,
        content: {
          kind: "available",
          value: project(content.value),
          lastFailure: content.lastFailure,
        },
      };
  }
}

// The scroll-line stand-in (r49 tunable default): the VALUE changes
// mechanically when the prototype's feel is measured; the meaning of
// the default is a ratified matter, not a per-call-site guess.
const SCROLL_LINE_PX = 40;

export interface SliceBufferViewport extends Viewport {
  // The window's presentation attaches its scroller element; before
  // that, the viewport reads zeros and scrolls nothing.
  attachElement(element: HTMLElement | null): void;
}

export function createViewport(): SliceBufferViewport {
  let element: HTMLElement | null = null;
  const viewport: SliceBufferViewport = {
    attachElement(next) {
      element = next;
    },
    metrics() {
      if (element === null) return { position: { left: 0, top: 0 }, width: 0, height: 0 };
      return {
        position: { left: element.scrollLeft, top: element.scrollTop },
        width: element.clientWidth,
        height: element.clientHeight,
      };
    },
    scrollLines(amount) {
      if (element !== null) element.scrollBy({ top: amount * SCROLL_LINE_PX });
    },
    scrollPage(direction, extent) {
      if (element === null) return;
      const share = extent === "half" ? 0.5 : 1;
      element.scrollBy({ top: direction * element.clientHeight * share });
    },
    start() {
      if (element !== null) element.scrollTop = 0;
    },
    end() {
      if (element !== null) element.scrollTop = element.scrollHeight;
    },
    reveal(bounds) {
      if (element === null) return;
      const { position } = viewport.metrics();
      if (bounds.top < position.top) {
        element.scrollTop = bounds.top;
      } else if (bounds.bottom > position.top + element.clientHeight) {
        element.scrollTop = bounds.bottom - element.clientHeight;
      }
    },
    moveTo(position) {
      if (element !== null) element.scrollTop = position.top;
    },
  };
  return viewport;
}

interface DiagnosticEntry {
  id: DiagnosticId;
  bufferId: BufferId;
  code: string;
  message: string;
}

export interface SliceDiagnostics extends BufferDiagnostics {
  // The window's observation readout lists the ring; the diagnostics
  // buffer UI arrives with its own slice.
  entries(): readonly DiagnosticEntry[];
}

/** The slice-1 diagnostics reporter: an in-memory ring over the minimal
 * catalogue (code + human message). Severity/notification routing is
 * its own contract; nothing here notifies. */
export function createDiagnostics(): SliceDiagnostics {
  const ring: DiagnosticEntry[] = [];
  let next = 0;
  const record = (bufferId: BufferId, issue: KnownIssue): DiagnosticId => {
    const id = `d-${++next}`;
    ring.push({ id, bufferId, code: issue.code, message: issue.message });
    if (ring.length > 200) ring.splice(0, ring.length - 200);
    return id;
  };
  return {
    report(context, issue) {
      return record(context.bufferId, issue);
    },
    unexpected(context, cause) {
      return record(context.bufferId, knownIssueOf(`unexpected: ${String(cause)}`));
    },
    entries() {
      return [...ring];
    },
  };
}

function knownIssueOf(message: string): KnownIssue {
  return knownIssue({ code: "unexpected", message });
}

const IDLE_RESTORE: RestoreStatus = "pending-content";

// The slice-1 container host. One class per buffer; the factory mints
// one per create call (registration constructed twice must get separate
// controller/input/state).
class SliceBufferHost<
  T extends object,
  UI extends object,
  A extends AnchorData,
> implements HostBuffer<T> {
  readonly id: BufferId;
  // The window mints the viewport and hands it over through the
  // environment; the buffer adopts it (the scroller element is the
  // window presentation's).
  readonly viewport: SliceBufferViewport;
  private readonly context: BufferContext;
  private readonly diagnostics: BufferDiagnostics;
  private readonly connect: (buffer: BufferAccess<T>) => InstrumentAttachment<UI, A>;
  private attachment: InstrumentAttachment<UI, A> | null = null;
  private state: BufferState<ContentFrame<T>> = {
    kind: "alive",
    active: false,
    content: { kind: "empty" },
  };
  private readonly observers = new Set<(state: BufferState<ContentFrame<T>>) => void>();
  private refreshToken = 0;
  private destroyed = false;
  private readingContext: InputContext | null = null;
  private leases: Set<InputLeaseImpl> = new Set();
  private readonly framePresentations = new WeakMap<ContentFrame<T>, TargetPresentations>();
  private restoreJobs: Set<RestoreJobImpl> = new Set();

  private readonly build: BufferBuild<T, UI, A>;

  constructor(build: BufferBuild<T, UI, A>) {
    this.build = build;
    // The window host mints the buffer id in the context; the buffer
    // adopts it, so the window's places and the running instrument
    // name the same buffer.
    this.id = build.context.bufferId;
    this.context = build.context;
    this.viewport = build.viewport as SliceBufferViewport;
    this.diagnostics = build.diagnostics;
    this.connect = build.connect;
  }

  /** The window assembly order: the BufferAccess facade must exist
   * before connect runs, and the attachment lands right after it. */
  wire(): BuiltBuffer<T, UI> {
    const access: BufferAccess<T> = {
      context: this.context,
      snapshot: () => this.snapshot(),
      subscribe: (observer) => this.subscribe(observer),
      viewport: this.viewport,
      geometry: this.geometry(),
      updates: this.updates(),
      interactions: this.interactions(),
      setReadingContext: (context) => {
        this.readingContext = context;
      },
    };
    // Controller construction calls connect synchronously exactly once.
    const attachment = this.connect(access);
    this.attachment = attachment;
    const input = attachment.input;
    if (this.readingContext === null) this.readingContext = input;
    const host: HostBuffer<T> = this;
    return { host, presentation: attachment.presentation };
  }

  private emit(): void {
    for (const observer of [...this.observers]) observer(this.state);
  }

  private set(patch: (state: BufferState<ContentFrame<T>>) => BufferState<ContentFrame<T>>): void {
    if (this.destroyed) return;
    this.state = patch(this.state);
    this.emit();
  }

  snapshot(): BufferState<ContentFrame<T>> {
    return this.state;
  }

  subscribe(observer: (state: BufferState<ContentFrame<T>>) => void): Unsubscribe {
    this.observers.add(observer);
    observer(this.state);
    return () => {
      this.observers.delete(observer);
    };
  }

  activate(): boolean {
    if (this.destroyed) return false;
    this.set((state) => (state.kind === "alive" ? { ...state, active: true } : state));
    return true;
  }

  deactivate(): void {
    this.set((state) => (state.kind === "alive" ? { ...state, active: false } : state));
  }

  // No bindings are installed in slice 1: the matcher arrives with its
  // own ratified contract, and until then keys stay native.
  handleKey(_input: KeyInput): KeyDisposition {
    return "allow-default";
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.refreshToken += 1;
    for (const job of [...this.restoreJobs]) job.cancel();
    this.restoreJobs.clear();
    for (const lease of [...this.leases]) lease.release();
    this.state = { kind: "destroyed" };
    this.emit();
    // Destruction invalidates late results and releases only its own
    // interests; the attachment release is the buffer's own claim.
    this.observers.clear();
    if (this.attachment !== null) {
      const release = this.attachment.release;
      this.attachment = null;
      release();
    }
  }

  presentationsFor(frame: ContentFrame<T>): TargetPresentations {
    const existing = this.framePresentations.get(frame);
    if (existing) return existing;
    const presentations: TargetPresentations = {
      presentTarget: (id, _measure) => {
        // No target machinery exists in slice 1: an instrument that
        // presents a target reports the gap, and the presentation
        // activation stays inert - never a silently wrong act.
        this.diagnostics.report(
          this.context,
          knownIssue({
            code: "target-unavailable",
            message: `no target presentation exists in this build (target '${id}')`,
          }),
        );
        return inertPresentation();
      },
    };
    this.framePresentations.set(frame, presentations);
    return presentations;
  }

  /** The window's return-places face over the attachment's content
   * places. Slice 1: default-position capture; restore requests the
   * address through the normal refresh pipeline and reports honestly. */
  get places(): ReturnPlaces {
    if (this._places === null) this._places = this.buildReturnPlaces();
    return this._places;
  }

  private _places: ReturnPlaces | null = null;

  private buildReturnPlaces(): ReturnPlaces {
    const attachment = () => this.attachment;
    return {
      capture: () => {
        const attached = attachment();
        if (attached === null || this.state.kind !== "alive") {
          throw new Error("capture from a destroyed buffer");
        }
        const capture: ContentCapture<A> = attached.places.captureDisplayed(
          this.viewport.metrics(),
        );
        const offset = this.viewport.metrics().position;
        const position =
          capture.position.kind === "anchor"
            ? {
                kind: "anchor" as const,
                anchor: capture.position.anchor,
                viewportOffset: {
                  left: capture.position.bounds.left - offset.left,
                  top: capture.position.bounds.top - offset.top,
                },
              }
            : { kind: "default" as const };
        return {
          bufferId: this.id,
          address: capture.address,
          position,
        };
      },
      restore: (place: SavedPlace) => {
        const job = new RestoreJobImpl(() => {
          this.restoreJobs.delete(job);
        });
        this.restoreJobs.add(job);
        void this.runRestore(place, job);
        return job;
      },
    };
  }

  private async runRestore(place: SavedPlace, job: RestoreJobImpl): Promise<void> {
    const attached = this.attachment;
    if (attached === null || this.destroyed || place.bufferId !== this.id) {
      job.finish("cancelled");
      return;
    }
    if (place.position.kind === "anchor" && !attached.places.acceptsAnchor(place.position.anchor)) {
      this.diagnostics.report(
        this.context,
        knownIssue({
          code: "anchor-rejected",
          message: "the restored place's anchor does not belong to this buffer's content",
        }),
      );
      job.finish("failed");
      return;
    }
    const outcome = await attached.places.request(place.address);
    if (this.destroyed || job.cancelled) {
      job.finish("cancelled");
      return;
    }
    if (outcome === "applied") {
      // Slice 1 restores at the default position of the requested
      // address; anchor geometry lands with the Board instrument slice.
      job.finish("restored");
    } else if (outcome === "superseded") {
      job.finish("superseded");
    } else if (outcome === "destroyed") {
      job.finish("cancelled");
    } else {
      job.finish("failed");
    }
  }

  private updates(): ContentUpdates<ContentFrame<T>> {
    return {
      refresh: async (
        prepare: () => PreparedContent<ContentFrame<T>> | Promise<PreparedContent<ContentFrame<T>>>,
      ): Promise<RefreshOutcome> => {
        if (this.destroyed) return "destroyed";
        const token = ++this.refreshToken;
        // A synchronous ready result needs no pending display frame:
        // only an async publication shows one.
        let prepared: PreparedContent<ContentFrame<T>> | Promise<PreparedContent<ContentFrame<T>>>;
        try {
          prepared = prepare();
        } catch (cause) {
          prepared = {
            kind: "failed",
            issue: knownIssueOf(`the load failed unexpectedly: ${String(cause)}`),
          };
        }
        if (!isPromise(prepared)) {
          return this.publish(token, prepared);
        }
        this.set((state) => {
          if (state.kind !== "alive") return state;
          const previous = state.content.kind === "available" ? state.content.value : null;
          return { ...state, content: { kind: "pending", previous } };
        });
        try {
          prepared = await prepared;
        } catch (cause) {
          prepared = {
            kind: "failed",
            issue: knownIssueOf(`the load failed unexpectedly: ${String(cause)}`),
          };
        }
        return this.publish(token, prepared);
      },
    };
  }

  private publish(token: number, prepared: PreparedContent<ContentFrame<T>>): RefreshOutcome {
    if (this.destroyed) return "destroyed";
    if (token !== this.refreshToken) return "superseded";
    if (prepared.kind === "ready") {
      this.set((state) => {
        if (state.kind !== "alive") return state;
        return {
          ...state,
          content: {
            kind: "available",
            value: prepared.value,
            lastFailure: null,
          },
        };
      });
      return "applied";
    }
    const failure = this.diagnostics.report(this.context, prepared.issue);
    this.set((state) => {
      if (state.kind !== "alive") return state;
      // Refresh retains the last usable content on failure and
      // reports the diagnostic; no fake revision advance.
      if (state.content.kind === "available") {
        return {
          ...state,
          content: { ...state.content, lastFailure: failure },
        };
      }
      return { ...state, content: { kind: "unavailable", failure } };
    });
    return "failed";
  }

  private geometry(): TargetGeometry {
    return {
      // No measurement machinery exists in slice 1: geometry reads
      // null, and the change subscription stays silent until the
      // presentation service lands.
      bounds: () => null,
      subscribeChanges: () => () => {},
    };
  }

  private interactions(): InputCapture {
    return {
      capture: (context: InputContext) => {
        this.readingContext = context;
        const lease = new InputLeaseImpl(() => {
          this.leases.delete(lease);
        });
        this.leases.add(lease);
        return lease;
      },
    };
  }
}

class InputLeaseImpl implements InputLease {
  private released = false;
  private readonly onRelease: () => void;
  constructor(onRelease: () => void) {
    this.onRelease = onRelease;
  }
  release(): void {
    if (this.released) return;
    this.released = true;
    this.onRelease();
  }
}

class RestoreJobImpl implements RestoringPlace {
  private status: RestoreStatus = IDLE_RESTORE;
  private readonly observers = new Set<(status: RestoreStatus) => void>();
  cancelled = false;
  private done = false;
  private readonly onDone: () => void;
  constructor(onDone: () => void) {
    this.onDone = onDone;
  }
  snapshot(): RestoreStatus {
    return this.status;
  }
  subscribe(observer: (status: RestoreStatus) => void): Unsubscribe {
    this.observers.add(observer);
    observer(this.status);
    return () => {
      this.observers.delete(observer);
    };
  }
  cancel(): void {
    if (this.done) return;
    this.cancelled = true;
    this.finish("cancelled");
  }
  finish(status: RestoreStatus): void {
    if (this.done) return;
    this.done = true;
    this.status = status;
    for (const observer of [...this.observers]) observer(status);
    this.observers.clear();
    this.onDone();
  }
}

function inertPresentation(): TargetPresentation {
  return {
    activatePrimary() {},
    activatePointer() {
      return false;
    },
    showActions() {},
    release() {},
  };
}

function isPromise<T>(value: T | Promise<T>): value is Promise<T> {
  return (
    typeof value === "object" && value !== null && typeof (value as Promise<T>).then === "function"
  );
}

/** The slice-1 factory: one host per create, connect called exactly
 * once, in the window-assembly order the glue expects. The factory is
 * the window's own object; the window owns retention until it closes. */
export function createBufferFactory(_windowId: string): BufferFactory {
  return {
    create<T extends object, UI extends object, A extends AnchorData>(
      build: BufferBuild<T, UI, A>,
    ): BuiltBuffer<T, UI> {
      const host = new SliceBufferHost<T, UI, A>(build);
      return host.wire();
    },
  };
}
