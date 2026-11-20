// The Tasks instrument: it registers itself with the instruments'
// registry - the frame reads the registered set and never names an
// instrument. A project's task list (the name/tasks location) is its
// address; the controller (rows, published targets, places) lives in
// tasksController.ts - svelte-free and unit-tested; this module is the
// one place the component enters and the window's services meet the
// task list. The rail publication (what sits beneath the project's
// rail entry - the agents, their words, their acts) is this
// instrument's own data (instruments/tasksRail.ts): it polls it and
// publishes it for the frame to draw.

import type { Component } from "svelte";
import TasksView from "./TasksView.svelte";
import { connectTasks } from "./tasksController.ts";
import { createTasksRail } from "./tasksRail.ts";
import type { TasksData, TasksUI } from "./tasksController.ts";
import type { InstrumentProps } from "../lib/instruments/contract.ts";
import type { TaskPorts } from "../lib/tasks/menu.ts";
import { registerInstrument, type RegisteredInstrument } from "../lib/instruments/registry.ts";
import TasksRailRow from "./TasksRailRow.svelte";
import { serveRailAgents } from "./tasksRail.ts";
import { sourceFor } from "../lib/sources.ts";
import { TrackerSource } from "../lib/tracker.ts";
import { isTasksPath } from "../lib/location.ts";
import { locationTitleOf, narrowedPath, parseNarrowing } from "../lib/tasks/narrowing.ts";
import { tasksPageOf } from "../lib/tasks/page.ts";

export * from "./tasksController.ts";
export * from "./tasksRail.ts";

// the rail publication and the window services, wired at the first
// construct; the frame may ask the publication's targets before any
// construct, and an unbuilt instrument publishes nothing
const railData = createTasksRail({
  ports: () =>
    services ?? {
      open: () => {},
      elsewhere: undefined,
      copy: undefined,
    },
});
let services: TaskPorts | null = null;

// the row component reads the live rail view through the lookup (the
// frame hands it only the target)
serveRailAgents((id) => railData.railAgentFor(id));

/** The task list's registration: called once by the instruments'
 * index (the registry's one list). */
export function register(): RegisteredInstrument {
  return registerInstrument({
    kind: "tasks",
    title: "the project's tasks",
    claims: (address) => address.machine.startsWith("tracker:") && isTasksPath(address.path),
    projectView: (project) => `${project}/tasks`,
    canonicalNarrowing: (opaque) => {
      const verdict = parseNarrowing(opaque);
      return verdict.ok ? { ok: true, canonical: verdict.canonical } : verdict;
    },
    locationTitle: locationTitleOf,
    railTargets: (project) => railData.railTargets(project),
    railNote: (project) => railData.railNote(project),
    railFootnote: () => railData.railFootnote(),
    railRow: TasksRailRow,
    pickerCandidates: async (machine) => {
      const source = sourceFor(machine);
      if (!(source instanceof TrackerSource)) return [];
      const model = tasksPageOf(await source.instrumentPage("tasks"));
      const out: { id: string; title: string }[] = [];
      // the repository's tags: each opens the list narrowed to it
      for (const tag of await source.tags()) {
        out.push({
          id: narrowedPath({ tags: [tag.name.toLowerCase()], untagged: false }),
          title: tag.description === "" ? tag.name : `${tag.name} - ${tag.description}`,
        });
      }
      for (const group of model.groups) {
        for (const entry of group.rows) {
          out.push({ id: entry.task, title: entry.title });
        }
      }
      return out;
    },
    Component: TasksView as Component<InstrumentProps<TasksData, TasksUI>>,
    connect: (buffer, windowServices) => {
      services = windowServices as TaskPorts;
      // the rail publication polls with the instrument alive: the
      // frame reads its targets through the claims seam
      railData.start();
      const ports: TaskPorts = {
        open: (spec) => windowServices.open(spec),
        elsewhere: (spec, act) => windowServices.elsewhere?.(spec, act),
        copy: (text) => windowServices.copy(text),
        // the rows' agent marks read the rail publication's view: the
        // agents joined with their words
        railWords: (observer) => {
          const push = (): void => observer(railData.view());
          push();
          return railData.subscribe(push);
        },
      };
      return connectTasks(buffer, ports);
    },
  });
}

function windowPorts(): TaskPorts {
  return services ?? { open: () => {}, elsewhere: undefined, copy: undefined, railWords: () => () => {} };
}
