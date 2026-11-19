<!-- The frame's rail, painted: the projects this device has registered
and, under each, its running agents grouped by Herdr's word; the add
field rides at the rail's foot. Any window can show the rail; the
frame (App.svelte) hosts this pane beside the content and feeds it the
frame rail's live view. The pane's entries are ordinary targets of the
shown buffer: select mode labels them with the content's (the labels
ride the same session, measured in the pane), a plain click or Enter
runs the primary act through the shown lease, a right-click opens the
act list. With no shown buffer (the empty window), a project's plain
click opens its task list directly. The wheel scrolls the rail and
keeps the shown project in view; the reading keys never scroll it. -->
<script lang="ts">
  import { agentPart, projectPart, terminalPart, type Rail, type RailAgent } from "../lib/tasks/rail.ts";
  import type { AddView } from "../lib/tasks/add.ts";
  import type { MenuView } from "../lib/menu/session.ts";
  import LabelOverlay from "./LabelOverlay.svelte";
  import ActMenu from "./ActMenu.svelte";
  import { partMeasureIn } from "../lib/select/measure.ts";

  let {
    rail,
    view,
    shown,
    labels,
    typed = "",
    menu,
    menuSession,
    typeAdd,
    submitAdd,
    closeAdd,
    entryActivate,
    entryMenu,
    attachPane,
  }: {
    rail: Rail;
    view: { visible: boolean; add: AddView; pane: HTMLElement | null };
    /** The project this window shows: the rail's mark. */
    shown: string | null;
    /** The select session's labels over the presented set (the
     * content's and the rail's): the pane paints its own. */
    labels: readonly { part: string; label: string; nested?: boolean }[];
    /** The select session's typed label, marked on its chips. */
    typed?: string;
    /** The act menu's view, when the menu's target is a rail entry
     * (the content's menu is the surface's own). */
    menu: MenuView | null;
    /** The menu's session (the shown buffer's one menu session; the
     * pane renders it for its own entries). */
    menuSession: unknown;
    typeAdd(value: string): void;
    submitAdd(): void;
    closeAdd(): void;
    /** A plain click or Enter on an entry: through the shown lease
     * when one is present, else a project's entry opens its task
     * list. */
    entryActivate(part: string, event: MouseEvent): void;
    /** A right-click on an entry: the act list at the pointer. */
    entryMenu(part: string, event: MouseEvent): void;
    attachPane(element: HTMLElement | null): void;
  } = $props();

  let pane: HTMLElement | null = $state(null);
  let list: HTMLElement | null = $state(null);
  let addInput: HTMLInputElement | null = $state(null);
  $effect(() => {
    attachPane(pane);
  });
  // the add field opens focused: typing lands in it natively (Enter
  // and Escape are its captured context's own commands)
  $effect(() => {
    if (view.add.open) addInput?.focus({ preventScroll: true });
  });

  const chipLabels = $derived(labels.filter((entry) => entry.part.startsWith("rail/")));
  const measure = $derived(partMeasureIn(pane));

  const titleOf = (agent: RailAgent): string =>
    [agent.msg, agent.progress].filter((part) => part !== null).join(" - ");

  // The wheel scrolls the rail's own list; the shown project stays in
  // view (the rail's clamping rule), whichever project's row the wheel
  // would push out.
  function keepShownInView(): void {
    if (shown === null || list === null) return;
    const marked = list.querySelector(`[data-port-control=${JSON.stringify(`project-${shown}`)}]`);
    marked?.scrollIntoView({ block: "nearest" });
  }

  function partOf(event: Event): string | null {
    return (event.target as Element | null)?.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null;
  }

  function railMenu(): MenuView | null {
    return menu !== null && menu.part.startsWith("rail/") ? menu : null;
  }
</script>

{#if view.visible}
  <nav
    class="port-rail"
    data-port-control="rail"
    data-port-frame-capture=""
    aria-label="projects"
    bind:this={pane}
  >
    <div class="port-rail-list" bind:this={list} onwheel={() => keepShownInView()}>
      {#each rail.projects as project (project.name)}
        <section
          class="port-rail-project"
          data-port-shown={project.current ? "" : undefined}
        >
          <button
            class="port-rail-name"
            data-port-control={`project-${project.name}`}
            data-port-part={projectPart(project.name)}
            data-port-kind="project"
            data-port-title={project.name}
            data-port-current={project.current ? "" : undefined}
            aria-current={project.current ? "page" : undefined}
            onclick={(event) => entryActivate(partOf(event) ?? `rail/project/${project.name}`, event)}
            oncontextmenu={(event) => {
              event.preventDefault();
              entryMenu(partOf(event) ?? `rail/project/${project.name}`, event);
            }}
          >{project.name}</button>
          {#if project.problem !== null}
            <div class="port-rail-note" title={project.problem}>unreachable</div>
          {/if}
          {#each project.groups as group (group.state)}
            {#each group.agents as agent (agent.address)}
              <button
                class="port-rail-agent"
                data-port-control={`agent-${project.name}-${agent.incarnation ?? agent.address}`}
                data-port-part={agent.incarnation === null
                  ? terminalPart(agent.address)
                  : agentPart(project.name, agent.incarnation)}
                data-port-kind="agent"
                data-port-unregistered={agent.incarnation === null ? "" : undefined}
                data-port-word={group.state}
                title={titleOf(agent)}
                onclick={(event) => entryActivate(partOf(event)!, event)}
                oncontextmenu={(event) => {
                  event.preventDefault();
                  entryMenu(partOf(event)!, event);
                }}
              >
                <span class="port-state-mark" data-port-word={agent.state} aria-hidden="true"></span>
                <span class="port-rail-who">
                  <span class="port-rail-incarnation">{agent.incarnation ?? agent.address}</span>
                  {#if agent.name !== null}<span class="port-rail-agent-name">{agent.name}</span>{/if}
                </span>
                <span class="port-rail-tasks" title={agent.tasks.join(", ")}>{agent.tasks.join(", ")}</span>
              </button>
            {/each}
          {/each}
        </section>
      {/each}
      {#if view.add.open}
        <div class="port-rail-add">
          <!-- the field's keys are the frame capture's own (Enter
          submits, Escape leaves - the window routes them here) -->
          <input
            bind:this={addInput}
            data-port-control="project-input"
            data-port-captured-input=""
            type="text"
            placeholder="machine:/path/to/repository"
            aria-label="register a project"
            value={view.add.value}
            oninput={(event) => typeAdd(event.currentTarget.value)}
          />
          {#if view.add.problem !== null}
            <div class="port-rail-note" data-port-control="project-input-problem">{view.add.problem}</div>
          {/if}
        </div>
      {/if}
      {#if rail.unregistered.length > 0}
        <section class="port-rail-project" data-port-control="unregistered">
          <div class="port-rail-name port-rail-group">unregistered</div>
          {#each rail.unregistered as agent (agent.address)}
            <button
              class="port-rail-agent"
              data-port-control={`unregistered-${agent.address}`}
              data-port-part={terminalPart(agent.address)}
              data-port-kind="agent"
              data-port-unregistered=""
              data-port-word={agent.state}
              title={[agent.msg, agent.progress].filter((part) => part !== null).join(" - ")}
              onclick={(event) => entryActivate(partOf(event)!, event)}
              oncontextmenu={(event) => {
                event.preventDefault();
                entryMenu(partOf(event)!, event);
              }}
            >
              <span class="port-state-mark" data-port-word={agent.state} aria-hidden="true"></span>
              <span class="port-rail-who">
                <span class="port-rail-incarnation">{agent.address}</span>
                {#if agent.label !== null}<span class="port-rail-agent-name">{agent.label}</span>{/if}
              </span>
            </button>
          {/each}
        </section>
      {/if}
      {#if rail.herdr !== null}
        <div class="port-rail-note">{rail.herdr}</div>
      {/if}
    </div>
    {#if chipLabels.length > 0}
      <LabelOverlay chips={chipLabels} {measure} {typed} />
    {/if}
    {#if railMenu() !== null}
      <ActMenu menu={railMenu()!} host={pane} session={menuSession as never} />
    {/if}
  </nav>
{/if}
