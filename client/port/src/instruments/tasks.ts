// The Tasks instrument definition: the component binding. The
// controller (rows, select session, row acts, places) lives in
// tasksController.ts - svelte-free and unit-tested; this module is
// the one place the component enters.

import type { Component } from "svelte";
import TasksView from "./TasksView.svelte";
import { connectTasks } from "./tasksController.ts";
import type { TasksData, TasksUI } from "./tasksController.ts";
import type { InstrumentDefinition, InstrumentProps } from "../lib/instruments/contract.ts";
import type { TaskAnchor } from "../lib/contracts/places.ts";

export * from "./tasksController.ts";
export type { MenuView } from "../lib/menu/session.ts";
export type { SelectSessionView } from "../lib/select/session.ts";

export function tasksInstrument(): InstrumentDefinition<TasksData, TasksUI, TaskAnchor> {
  return {
    kind: "tasks",
    Component: TasksView as Component<InstrumentProps<TasksData, TasksUI>>,
    connect: connectTasks,
  };
}
