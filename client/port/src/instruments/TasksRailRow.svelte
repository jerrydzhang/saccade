<!-- The task list instrument's rail row, handed to the frame through
the registration: one component for this instrument's rail targets,
receiving the published target and drawing it by kind - its agent
resolves through the instrument's own publication (railAgentOf). An
agent reads the state's glyph, the incarnation, the name, every task
it claimed, and Herdr's status line beneath; an unregistered one, its
terminal address and Herdr's label in the name column. The frame owns
the entry element (the marking, select, menu, clicks); this component
owns what is inside. The spans fill the frame's entry grid directly:
the glyph's column, the id and the name on fixed columns, every
claimed task at the right, the status line beneath. -->
<script lang="ts">
  import type { TargetDefinition } from "../lib/contracts/buffer.ts";
  import { railAgentOf } from "./tasksRail.ts";

  let { target }: { target: TargetDefinition } = $props();

  const agent = $derived(railAgentOf(target.id));
  // Herdr's message and progress, the row's tooltip; none when both
  // are absent
  const hint = $derived(
    agent === null ? undefined : [agent.msg, agent.progress].filter((part) => part !== null).join(" - ") || undefined,
  );
</script>

{#if agent !== null}
  <!-- the row's root spans the entry's columns on the entry's own
  tracks (a subgrid), so the tooltip covers the row's content -->
  <span class="port-rail-row" title={hint}>
  <span class="port-state-mark" data-port-word={agent.state} aria-hidden="true"></span>
  {#if agent.incarnation === null}
    <!-- unregistered: the terminal's address stands in the id column,
    Herdr's label in the name column; beneath a project the tasks
    column says so, the foot group's rows carry no tasks column -->
    <span class="port-rail-who">
      <span class="port-rail-incarnation">{agent.address}</span>
      {#if agent.label !== null}<span class="port-rail-agent-name">{agent.label}</span>{/if}
    </span>
    {#if target.inside !== null}<span class="port-rail-tasks port-rail-unregistered">unregistered</span>{/if}
  {:else}
    <span class="port-rail-who">
      <span class="port-rail-incarnation">{agent.incarnation}</span>
      {#if agent.name !== null}<span class="port-rail-agent-name">{agent.name}</span>{/if}
    </span>
    <span class="port-rail-tasks" title={agent.tasks.join(", ")}>{agent.tasks.join(", ")}</span>
  {/if}
  {#if agent.msg !== null}<span class="port-rail-msg">{agent.msg}</span>{/if}
  </span>
{/if}
