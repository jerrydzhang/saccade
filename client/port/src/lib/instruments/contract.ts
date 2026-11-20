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
  /** The help column's title: the surface's own keys. */
  title: string;
  /** The locations this instrument shows (the frame reads the
   * registered set; the frame never names an instrument). */
  claims(address: import("../contracts/places.ts").Address): boolean;
  /** The location form this instrument offers for a registered
   * project's name (the task list claims `name/tasks`), or null when
   * it offers none. */
  projectView(project: string): string | null;
  /** The targets this instrument publishes beneath a project's rail
   * entry (its `inside` names the project's rail id), or at the rail's
   * foot when the project is null. The instrument's own data, read and
   * polled by the instrument; the frame draws them without knowing
   * their kind. */
  railTargets(project: string | null): import("../contracts/buffer.ts").TargetDefinition[];
  /** The picker's task candidates on a tracker machine: the tasks
   * this instrument's tracker lists, or none (the board offers
   * none). */
  pickerCandidates?(machine: string): Promise<readonly { id: string; title: string }[]>;
  /** The unreachable note under a project's rail entry (its last
   * ask's failure), or null when the entry stands clean. */
  railNote?(project: string): string | null;
  /** The words feed's note (why the agents' words may be stale), or
   * null. */
  railFootnote?(): string | null;
  /** The row component this instrument hands the frame for its rail
   * targets: one component, hosted inside the frame's own entry
   * element, receiving the target and drawing it by kind. Absent, the
   * frame draws the targets as their labels. */
  railRow?: import("svelte").Component<{ target: import("../contracts/buffer.ts").TargetDefinition }>;
  Component: Component<InstrumentProps<T, UI>>;
  connect: (buffer: BufferAccess<T>, services: import("../instruments/registry.ts").WindowServices) => InstrumentAttachment<UI, A>;
}>;
