// The Board instrument definition: the component binding. The
// controller (targets, select session, loading, places) lives in
// boardController.ts - svelte-free and unit-tested; this module is the
// one place the component enters.

import type { Component } from "svelte";
import BoardView from "./BoardView.svelte";
import { connectBoard } from "./boardController.ts";
import type { BoardAnchor, BoardData, BoardUI } from "./boardController.ts";
import type { InstrumentDefinition } from "../lib/instruments/contract.ts";

export * from "./boardController.ts";
export type { SelectSessionView } from "../lib/select/session.ts";

export function boardInstrument(): InstrumentDefinition<BoardData, BoardUI, BoardAnchor> {
  return {
    kind: "board",
    Component: BoardView as Component<InstrumentProps<BoardData, BoardUI>>,
    connect: connectBoard,
  };
}

// re-export the props type for the component binding above
import type { InstrumentProps } from "../lib/instruments/contract.ts";
