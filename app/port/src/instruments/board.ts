// The Board instrument: it registers itself with the instruments'
// registry - the frame reads the registered set and never names an
// instrument. The controller (targets, select session, loading,
// places) lives in boardController.ts - svelte-free and unit-tested;
// this module is the one place the component enters and the window's
// services meet the board (the registry hands them to connect).

import type { Component } from "svelte";
import BoardView from "./BoardView.svelte";
import { connectBoard } from "./boardController.ts";
import type { BoardAnchor, BoardData, BoardServices, BoardUI } from "./boardController.ts";
import type { InstrumentDefinition } from "../lib/instruments/contract.ts";
import { registerInstrument, type RegisteredInstrument } from "../lib/instruments/registry.ts";
import { isTasksPath } from "../lib/location.ts";

export * from "./boardController.ts";

// re-export the props type for the component binding above
import type { InstrumentProps } from "../lib/instruments/contract.ts";

/** The board's registration: called once by the instruments' index
 * (the registry's one list). */
export function register(): RegisteredInstrument {
  return registerInstrument({
    kind: "board",
    title: "this board",
    claims: (address) => !(address.machine.startsWith("tracker:") && isTasksPath(address.path)),
    projectView: () => null,
    railTargets: () => [],
    Component: BoardView as Component<InstrumentProps<BoardData, BoardUI>>,
    connect: (buffer, services) => connectBoard(buffer, services as BoardServices),
  });
}
