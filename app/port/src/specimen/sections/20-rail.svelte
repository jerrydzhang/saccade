<script lang="ts">
  // The frame's rail, through the real component: every word in
  // importance order, a named agent with several tasks, an agent with
  // no word, unregistered agents under the project and at the foot,
  // and the add field open. The beneath rows draw through the
  // instrument's row component (TasksRailRow), handed over the
  // registration the way the tasks instrument hands it; the fixture
  // serves the agents the row resolves by.
  import FrameRail from "../../components/FrameRail.svelte";
  import TasksRailRow from "../../instruments/TasksRailRow.svelte";
  import { serveRailAgents } from "../../instruments/tasksRail.ts";
  import { railAgentPart, railAgentText, projectPart, terminalPart, type RailAgent } from "../../lib/tasks/rail.ts";
  import { tasksRail } from "../tasks-fixture.ts";
  import type { RailPublication, TargetDefinition } from "../../lib/contracts/buffer.ts";

  const noop = () => {};

  // the fixture's publication: the agents beneath their project, the
  // strays at the foot, each carrying the row component
  const byId = new Map<string, RailAgent>();
  for (const entry of tasksRail.projects) {
    for (const group of entry.groups) {
      for (const agent of group.agents) {
        byId.set(railAgentPart(entry.name, agent), agent);
      }
    }
  }
  for (const agent of tasksRail.unregistered) {
    byId.set(terminalPart(agent.address), agent);
  }
  serveRailAgents((id) => byId.get(id) ?? null);

  function beneathTargets(): RailPublication[] {
    const out: RailPublication[] = [];
    for (const entry of tasksRail.projects) {
      for (const group of entry.groups) {
        for (const agent of group.agents) {
          const target: TargetDefinition = {
            id: railAgentPart(entry.name, agent),
            kind: "agent",
            label: railAgentText(agent),
            inside: projectPart(entry.name),
            clickable: true,
            acts: () => [],
            anchor: null,
          };
          out.push({ target, row: TasksRailRow });
        }
      }
    }
    for (const agent of tasksRail.unregistered) {
      const target: TargetDefinition = {
        id: terminalPart(agent.address),
        kind: "agent",
        label: railAgentText(agent),
        inside: null,
        clickable: true,
        acts: () => [],
        anchor: null,
      };
      out.push({ target, row: TasksRailRow });
    }
    return out;
  }

  const projects = tasksRail.projects.map((project) => ({
    name: project.name,
    current: project.current,
    part: projectPart(project.name),
  }));
  const beneath = projects.map((project) => ({
    project,
    targets: beneathTargets().filter((publication) => publication.target.inside === project.part),
  }));
  const foot = beneathTargets().filter((publication) => publication.target.inside === null);
</script>

<div class="rail">
  <FrameRail
    {projects}
    {beneath}
    {foot}
    view={{ visible: true, registered: true, footnote: null, add: { open: true, value: "", problem: null }, pane: null }}
    shown="pyrimel"
    labels={[]}
    menu={null}
    menuSession={null}
    typeAdd={noop}
    submitAdd={noop}
    closeAdd={noop}
    entryActivate={noop}
    entryMenu={noop}
    attachPane={noop}
  />
</div>

<style>
  .rail {
    width: 17rem;
  }
</style>
