// The Tasks instrument: it registers itself with the instruments'
// registry - the frame reads the registered set and never names an
// instrument. A project's task list (the name/tasks location) is its
// address; the controller (rows, published targets, places) lives in
// tasksController.ts - svelte-free and unit-tested; this module is the
// one place the component enters and the window's services meet the
// task list.

import type { Component } from "svelte";
import TasksView from "./TasksView.svelte";
import { connectTasks } from "./tasksController.ts";
import type { TasksData, TasksUI } from "./tasksController.ts";
import type { InstrumentProps } from "../lib/instruments/contract.ts";
import type { TaskPorts } from "../lib/tasks/menu.ts";
import { registerInstrument, type RegisteredInstrument } from "../lib/instruments/registry.ts";

export * from "./tasksController.ts";

/** The task list's registration: called once by the instruments'
 * index (the registry's one list). */
export function register(): RegisteredInstrument {
  return registerInstrument({
    kind: "tasks",
    title: "the project's tasks",
    claims: (address) => address.machine.startsWith("tracker:") && address.path === "tasks",
    projectView: (project) => `${project}/tasks`,
    Component: TasksView as Component<InstrumentProps<TasksData, TasksUI>>,
    connect: (buffer, services) =>
      connectTasks(buffer, {
        open: (spec) => services.open(spec),
        elsewhere: (spec, act) => services.elsewhere?.(spec, act),
        copy: (text) => services.copy(text),
        railWords: (observer) => services.railWords?.((rail) => observer(rail as never)) ?? (() => {}),
      }),
  });
}
