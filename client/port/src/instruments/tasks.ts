// The Tasks instrument definition: the component binding and the
// window services' door. The controller (rows, published
// targets, places) lives in tasksController.ts - svelte-free and
// unit-tested; this module is the one place the component enters and
// the one place the window's services (the open funnel, the elsewhere
// surface, the copy observation) are carried INTO the instrument: the
// definition takes them, no cast in the window assembly touches the
// controller's own type.

import type { Component } from "svelte";
import TasksView from "./TasksView.svelte";
import { connectTasks } from "./tasksController.ts";
import type { TasksData, TasksUI } from "./tasksController.ts";
import type { InstrumentDefinition, InstrumentProps } from "../lib/instruments/contract.ts";
import type { TaskAnchor } from "../lib/contracts/places.ts";
import type { TaskPorts } from "../lib/tasks/menu.ts";

export * from "./tasksController.ts";

export function tasksInstrument(ports: TaskPorts): InstrumentDefinition<TasksData, TasksUI, TaskAnchor> {
  return {
    kind: "tasks",
    Component: TasksView as Component<InstrumentProps<TasksData, TasksUI>>,
    connect: (buffer) => connectTasks(buffer, ports),
  };
}
