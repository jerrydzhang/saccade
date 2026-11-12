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
export type WindowEnvironment = Readonly<{
  context: BufferContext;
  viewport: Viewport;
  diagnostics: BufferDiagnostics;
  buffers: BufferFactory;
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
  destroy(): void;
  snapshot(): BufferState<RenderPair>;
  subscribe(observer: (state: BufferState<RenderPair>) => void): Unsubscribe;
}
export interface RegisteredInstrument {
  readonly kind: string;
  construct(environment: WindowEnvironment): RunningInstrument;
}
export function registerInstrument<
  T extends object, UI extends object, A extends AnchorData,
>(definition: InstrumentDefinition<T, UI, A>): RegisteredInstrument {
  return {
    kind: definition.kind,
    construct(environment) {
      const built: BuiltBuffer<T, UI> = environment.buffers.create<T, UI, A>({
        context: environment.context,
        viewport: environment.viewport,
        diagnostics: environment.diagnostics,
        connect: definition.connect,
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
        destroy: () => built.host.destroy(),
        snapshot: () => mapBufferState(built.host.snapshot(), project),
        subscribe: (observer) => built.host.subscribe(
          (state) => observer(mapBufferState(state, project)),
        ),
      };
    },
  };
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
