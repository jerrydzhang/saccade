// instruments/contract.ts: public component/controller pairing ABI,
// from the ratified construction package (t-36 r49). New mutable
// controller/UI state per call; the definition itself may be reused
// across windows/buffers; it must never cache a connected controller.

import type { Component } from "svelte";
import type {
  BufferAccess,
  InstrumentAttachment,
  TargetPresentations,
} from "../../components/Buffer.ts";
import type { AnchorData } from "../contracts/places.ts";

export type InstrumentProps<T extends object, UI extends object> = Readonly<{
  content: Readonly<T>;
  controller: UI;
  // Rendering gets presentation leases, not refresh/lifecycle/action-run access.
  targets: TargetPresentations;
}>;
export type InstrumentDefinition<
  T extends object,
  UI extends object,
  A extends AnchorData,
> = Readonly<{
  kind: string;
  Component: Component<InstrumentProps<T, UI>>;
  connect: (buffer: BufferAccess<T>) => InstrumentAttachment<UI, A>;
}>;
