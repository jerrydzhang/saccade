// instruments/index.ts: single reviewed heterogeneous rendering
// boundary, from the ratified construction package (t-36 r49). Svelte
// needs a common component type after registry selection. This cast is
// private here: component and props originate in the SAME generic
// closure. No untyped payload enters the closure. Instrument authors
// need no casts.
//
// Port carry (t-36 slice 1, the composition skeleton): the validated
// glue shapes - registerInstrument, constructInstrument, packComponent
// and the state projection - as ratified, against the port's own
// contract modules under lib/contracts and components/Buffer.ts.

import type { Component } from "svelte";
import type {
  BufferFactory,
  BufferInputState,
  ContentFrame,
  BuiltBuffer,
} from "../../components/Buffer.ts";
import { mapBufferState } from "../../components/Buffer.ts";
import type {
  BufferContext,
  BufferDiagnostics,
  BufferState,
  Unsubscribe,
  Viewport,
} from "../contracts/buffer.ts";
import type { KeyDisposition, KeyInput } from "../contracts/input.ts";
import type {
  AnchorData,
  ReturnPlaces,
} from "../contracts/places.ts";
import type { TargetDefinition } from "../contracts/buffer.ts";
import type {
  InstrumentDefinition,
  InstrumentProps,
} from "./contract.ts";

export type RenderPair = Readonly<{
  Component: Component<Record<string, unknown>>;
  props: Readonly<Record<string, unknown>>;
}>;
function packComponent<P extends Record<string, unknown>>(
  component: Component<P>, props: P,
): RenderPair {
  return {
    Component: component as unknown as Component<Record<string, unknown>>,
    props,
  };
}
import type { WhichKeyView } from "../input/key-matcher.ts";

export interface WindowServices {
  /** The ONE open funnel (trail + anchored restore). */
  open(spec: string, part?: string | null): void;
  /** The elsewhere acts (o/i/w), present when the boot's binding
   * facts have answered. */
  elsewhere?(spec: string, act: "link" | "once" | "new"): void;
  /** The copy observation: the payload recorded, the clipboard
   * attempted. */
  copy(text: string): void;
  /** The frame rail's live view (the rows' agent marks read its
   * words). */
  railWords?(observer: (rail: unknown) => void): () => void;
  [name: string]: unknown;
}

export type WindowEnvironment = Readonly<{
  /** The which-key suspension probe (the r110 tunable): while true,
   * the instrument's own matcher freezes its sequence expiry - the
   * App supplies the popup's state. */
  expirySuspended?: () => boolean;
  context: BufferContext;
  viewport: Viewport;
  diagnostics: BufferDiagnostics;
  buffers: BufferFactory;
  /** The window's services door: the instruments read their slices. */
  services: WindowServices;
  /** The frame's own targets (the rail's), presented beside the
   * publication's: one select session and one act menu span the
   * content and the rail. */
  frameTargets?: {
    targets(): TargetDefinition[];
    bounds(id: string): import("../contracts/buffer.ts").Bounds | null;
  };
}>;
export interface RunningInstrument {
  readonly id: string;
  readonly kind: string;
  readonly places: ReturnPlaces;
  /** The instrument's UI surface (the component's controller), type-
   * erased at the registration boundary; the owning window casts to
   * its own instrument's UI type when it needs a window-service
   * injection (the reference ports) - the one sanctioned App-to-UI
   * channel, never a second connect path. */
  readonly presentation: unknown;
  activate(): boolean;
  deactivate(): void;
  handleKey(input: KeyInput): KeyDisposition;
  /** The instrument's input pending (the which-key chain): the
   * matcher's views and their change notification, or null views
   * when no reading context is installed. */
  inputPending(): WhichKeyView | null;
  inputAll(): WhichKeyView;
  subscribeInputPending(observer: () => void): () => void;
  inputState(): BufferInputState;
  destroy(): void;
  snapshot(): BufferState<RenderPair>;
  subscribe(observer: (state: BufferState<RenderPair>) => void): Unsubscribe;
}
export interface RegisteredInstrument {
  readonly kind: string;
  /** The help column's title. */
  readonly title: string;
  /** The locations this instrument shows (the claims seam). */
  claims(address: import("../contracts/places.ts").Address): boolean;
  /** The location form this instrument offers for a registered
   * project's name, or null. */
  projectView(project: string): string | null;
  construct(environment: WindowEnvironment): RunningInstrument;
}

/** The registered instruments, in registration order: the frame reads
 * this list and never names one. */
const registered: RegisteredInstrument[] = [];

export function registeredInstruments(): readonly RegisteredInstrument[] {
  return registered;
}

/** The instrument registered under a kind, or undefined. */
export function instrumentOfKind(kind: string): RegisteredInstrument | undefined {
  return registered.find((entry) => entry.kind === kind);
}

/** The registered instrument whose claims name the address, in
 * registration order; null when none does. */
export function instrumentForAddress(
  address: import("../contracts/places.ts").Address,
): RegisteredInstrument | null {
  return registered.find((entry) => entry.claims(address)) ?? null;
}

export function registerInstrument<
  T extends object, UI extends object, A extends AnchorData,
>(definition: InstrumentDefinition<T, UI, A>): RegisteredInstrument {
  const registration: RegisteredInstrument = {
    kind: definition.kind,
    title: definition.title,
    claims: definition.claims,
    projectView: definition.projectView,
    construct(environment) {
      const built: BuiltBuffer<T, UI> = environment.buffers.create<T, UI, A>({
        context: environment.context,
        viewport: environment.viewport,
        diagnostics: environment.diagnostics,
        connect: (buffer) => definition.connect(buffer, environment.services),
        ...(environment.expirySuspended !== undefined
          ? { expirySuspended: environment.expirySuspended }
          : {}),
        ...(environment.frameTargets !== undefined
          ? { frameTargets: environment.frameTargets }
          : {}),
      });
      const project = (frame: Readonly<ContentFrame<T>>): RenderPair =>
        packComponent<InstrumentProps<T, UI>>(definition.Component, {
          content: frame.value,
          controller: built.presentation,
          targets: built.host.presentationsFor(frame),
        });
      return {
        id: built.host.id,
        presentation: built.presentation,
        kind: definition.kind,
        places: built.host.places,
        activate: () => built.host.activate(),
        deactivate: () => built.host.deactivate(),
        handleKey: (input) => built.host.handleKey(input),
        inputState: () => built.host.inputState(),
        destroy: () => built.host.destroy(),
        snapshot: () => mapBufferState(built.host.snapshot(), project),
        subscribe: (observer) => built.host.subscribe(
          (state) => observer(mapBufferState(state, project)),
        ),
          inputPending: () => built.host.inputPending(),
    inputAll: () => built.host.inputAll(),
    subscribeInputPending: (observer) => built.host.subscribeInputPending(observer),
};
    },
  };
  registered.push(registration);
  return registration;
}

// App.ts: called only when the window needs a NEW buffer. Revisiting a
// retained buffer activates it; this is not the navigation/open operation.
// No per-instrument constructor branches or second connect path.
export function constructInstrument(
  registration: RegisteredInstrument,
  environment: WindowEnvironment,
): RunningInstrument {
  return registration.construct(environment);
}
