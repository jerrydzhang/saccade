<!-- The control plane's rail (t-48): projects, then each project's
running agents in importance order of Herdr's word (t-63: flat - no
word headings; each agent row leads with its state's glyph). A project
switches this window, an agent opens its task here. An agent reads
incarnation, name, every task it claimed, and Herdr's status line
beneath; an unregistered one, its terminal address and Herdr's label.
Presentation only: the clicks and the menu are TasksView's delegation,
the add field's typing is the controller's. -->

<script lang="ts">
  import type { TasksUI } from "./tasksController.ts";
  import { projectPart, railAgentPart, railAgentText, wordRunsOf, type Rail, type RailAgent } from "../lib/tasks/rail.ts";
  import { ADD_CLOSED, type AddView } from "../lib/tasks/add.ts";

  let {
    rail,
    controller,
    hovered = null,
  }: {
    rail: Rail;
    controller: Pick<TasksUI, "subscribeAdd" | "typeAdd">;
    hovered?: string | null;
  } = $props();

  // the add field (t-54): open while the controller's add session is;
  // typing stays native (a text-entry capture), Enter and Escape are
  // the session's own commands
  let add: AddView = $state(ADD_CLOSED);
  let addInput: HTMLInputElement | null = $state(null);
  let addValue = $state("");
  $effect(() => controller.subscribeAdd((view) => {
    add = view;
    if (document.activeElement !== addInput) addValue = view.value;
  }));
  $effect(() => {
    if (add.open) addInput?.focus({ preventScroll: true });
  });

  const titleOf = (agent: RailAgent): string =>
    [agent.msg, agent.progress].filter((part) => part !== null).join(" - ");
  const footRuns = $derived(wordRunsOf(rail.unregistered));
</script>

{#snippet glyph(state: string)}
  <span class="port-state-mark" data-port-word={state} aria-hidden="true"></span>
{/snippet}

{#snippet agentLines(agent: RailAgent)}
  {@render glyph(agent.state)}
  {#if agent.incarnation === null}
    <!-- unregistered: the terminal's address stands in the id column,
    Herdr's label in the name column -->
    <span class="port-plane-who">
      <span class="port-plane-incarnation">{agent.address}</span>
      {#if agent.label !== null}<span class="port-plane-agent-name">{agent.label}</span>{/if}
    </span>
    <span class="port-plane-tasks port-plane-unregistered">unregistered</span>
  {:else}
    <span class="port-plane-who">
      <span class="port-plane-incarnation">{agent.incarnation}</span>
      {#if agent.name !== null}<span class="port-plane-agent-name">{agent.name}</span>{/if}
    </span>
    <span class="port-plane-tasks" title={agent.tasks.join(", ")}>{agent.tasks.join(", ")}</span>
  {/if}
  {#if agent.msg !== null}<span class="port-plane-msg">{agent.msg}</span>{/if}
{/snippet}

<nav class="port-plane-rail" data-port-control="rail" aria-label="projects">
  {#each rail.projects as project (project.name)}
    <section class="port-plane-project" data-port-project={project.name} data-port-current={project.current ? "" : undefined}>
      <button
        class="port-plane-name"
        data-port-control={`project-${project.name}`}
        data-port-part={projectPart(project.name)}
        data-port-kind="project"
        data-port-title={project.name}
        data-port-hover={hovered === projectPart(project.name) ? "" : undefined}
        aria-current={project.current ? "page" : undefined}
      >{project.name}</button>
      {#if project.problem !== null}
        <div class="port-plane-note" title={project.problem}>unreachable</div>
      {/if}
      {#each project.groups as group (group.state)}
        {#each group.agents as agent (agent.address)}
          <!-- every agent carries y (copy its attach command, t-55), so
          none is disabled; an unregistered one is marked so -->
          <button
            class="port-plane-agent"
            data-port-control={`agent-${project.name}-${agent.incarnation ?? agent.address}`}
            data-port-part={railAgentPart(project.name, agent)}
            data-port-kind="agent"
            data-port-title={railAgentText(agent)}
            data-port-unregistered={agent.incarnation === null ? "" : undefined}
            data-port-word={group.state}
            data-port-hover={hovered === railAgentPart(project.name, agent) ? "" : undefined}
            title={titleOf(agent)}
          >
            {@render agentLines(agent)}
          </button>
        {/each}
      {/each}
    </section>
  {/each}
  {#if add.open}
    <div class="port-plane-add">
      <input
        bind:this={addInput}
        bind:value={addValue}
        data-port-control="project-input"
        data-port-captured-input=""
        type="text"
        placeholder="host:port/name"
        aria-label="add a project"
        oninput={(event) => controller.typeAdd(event.currentTarget.value)}
      />
      {#if add.problem !== null}
        <div class="port-plane-note" data-port-control="project-input-problem">{add.problem}</div>
      {/if}
    </div>
  {/if}
  {#if rail.unregistered.length > 0}
    <!-- agents no tracker registered and no project's repo holds
    (t-55): named by the terminal's title and the address, under their words as the core reports them -->
    <section class="port-plane-project" data-port-control="unregistered">
      <div class="port-plane-name port-plane-group">unregistered</div>
      {#each footRuns as run, index (index)}
        {#each run.agents as agent (agent.address)}
          <button
            class="port-plane-agent"
            data-port-control={`unregistered-${agent.address}`}
            data-port-part={railAgentPart(null, agent)}
            data-port-kind="agent"
            data-port-title={railAgentText(agent)}
            data-port-unregistered=""
            data-port-word={agent.state}
            data-port-hover={hovered === railAgentPart(null, agent) ? "" : undefined}
            title={titleOf(agent)}
          >
            {@render glyph(agent.state)}
            <span class="port-plane-who">
              <span class="port-plane-incarnation">{agent.address}</span>
              {#if agent.label !== null}<span class="port-plane-agent-name">{agent.label}</span>{/if}
            </span>
            {#if agent.msg !== null}<span class="port-plane-msg">{agent.msg}</span>{/if}
          </button>
        {/each}
      {/each}
    </section>
  {/if}
  {#if rail.herdr !== null}
    <div class="port-plane-note" data-port-control="herdr-note">{rail.herdr}</div>
  {/if}
</nav>
