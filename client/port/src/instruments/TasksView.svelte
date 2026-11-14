<!-- The Tasks instrument's component: renders the published page
model (groups, tables, task rows) and paints the select labels and
the row act menu. Presentation only - it receives the content and its
controller through the instrument props, never refresh, lifecycle or
action-run access. Row identity and the row's link ride the port's
own content-structure scheme: data-port-part names the row's identity,
data-port-kind names the kind, data-port-task names the task (the
return anchor's identity), data-port-open names the row's link. -->

<script lang="ts">
  import type { MenuView, SelectSessionView } from "./tasks.ts";
import LabelOverlay from "../components/LabelOverlay.svelte";
import ActMenu from "../components/ActMenu.svelte";
import { partMeasureIn } from "../lib/select/measure.ts";
import { ageOf } from "../lib/tasks/page.ts";
  import type { TasksData, TasksUI } from "./tasks.ts";
  import { projectPart, railAgentPart, railAgentText, type Rail } from "../lib/tasks/rail.ts";
  import { ADD_CLOSED, type AddView } from "../lib/tasks/add.ts";

  let { content, controller }: { content: TasksData; controller: TasksUI } = $props();

  let host: HTMLElement | null = $state(null);
  let select: SelectSessionView | null = $state(null);
  let menu: MenuView | null = $state(null);
  let rail: Rail = $state({ projects: [], unregistered: [], herdr: null });

  $effect(() => controller.subscribeRail((next) => {
    rail = next;
  }));

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

  $effect(() => {
    const offSelect = controller.subscribeSelect((view) => {
      select = view;
    });
    const offMenu = controller.subscribeMenu((view) => {
      menu = view;
    });
    return () => {
      offSelect();
      offMenu();
    };
  });

  // A plain click on a row runs its first act directly on the CLICKED
  // row (scenario 17's mouse path: the act targets the clicked row,
  // not a selection). Right-click opens the same act list at the
  // pointer (scenario 9's third invocation path).
  // a rail entry is a target like a row (t-48): the same click, the
  // same right-click menu
  function rowFromEvent(event: MouseEvent): HTMLElement | null {
    const element = (event.target as HTMLElement | null)?.closest?.(
      '[data-port-kind="task"], [data-port-kind="project"], [data-port-kind="agent"]',
    );
    return element instanceof HTMLElement ? element : null;
  }

  function onClick(event: MouseEvent): void {
    const row = rowFromEvent(event);
    if (row === null) return;
    const part = row.getAttribute("data-port-part");
    if (part === null) return;
    event.preventDefault();
    controller.rowPrimaryClick(part);
  }

  function onContextMenu(event: MouseEvent): void {
    const row = rowFromEvent(event);
    if (row === null) return;
    const part = row.getAttribute("data-port-part");
    if (part === null) return;
    const kind = row.getAttribute("data-port-kind") ?? "task";
    const task = row.getAttribute("data-port-task") ?? "";
    const title = row.getAttribute("data-port-title") ?? "";
    event.preventDefault();
    const box = host?.getBoundingClientRect();
    const at =
      box === undefined
        ? undefined
        : { x: event.clientX - box.left, y: event.clientY - box.top };
    controller.openMenuFor(
      { part, kind, label: kind === "task" ? `${task} ${title}`.trim() : title },
      at,
    );
  }

  // the ages are read against the moment the shown page arrived: a
  // follow refresh moves them with the rows
  const now = $derived((void content.page, Date.now()));

  const page = $derived(content.page);

  // The row highlight follows the POINTER'S OWN MOVEMENT, never a row
  // that scrolls under a resting pointer (m-547 (4); CSS :hover did
  // exactly that as j/k scrolled). A move that changes the pointer's
  // coordinates marks the row under it; any scroll clears the mark.
  // The look is the prototype's ledger row: ink only, no shading.
  let hovered: string | null = $state(null);
  let lastPointer: { x: number; y: number } | null = null;
  function pointerMoved(event: MouseEvent): void {
    if (lastPointer !== null && lastPointer.x === event.clientX && lastPointer.y === event.clientY) return;
    lastPointer = { x: event.clientX, y: event.clientY };
    const row = (event.target as HTMLElement | null)?.closest?.(".port-tasks-row");
    hovered = row instanceof HTMLElement ? row.getAttribute("data-port-part") : null;
  }
  $effect(() => {
    const clear = (): void => {
      hovered = null;
    };
    document.addEventListener("scroll", clear, { capture: true, passive: true });
    return () => document.removeEventListener("scroll", clear, { capture: true });
  });
</script>

<!-- The container's pointer handling is act delegation, not an
interactive element of its own: the keyboard path to the same acts is
the select session + the act menu (Enter runs the first act), so the
a11y click advisories do not apply. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div
  class="port-tasks-stage"
  role="presentation"
  onmousemove={pointerMoved}
  onmouseleave={() => {
    hovered = null;
  }}
>
<div class="port-plane" bind:this={host} role="document" onclick={onClick} oncontextmenu={onContextMenu}>
<!-- the control plane's rail (t-48): projects, then each project's
running agents by Herdr's word; a project switches this window, an
agent opens its task here -->
<nav class="port-plane-rail" data-port-control="rail" aria-label="projects">
  {#each rail.projects as project (project.name)}
    <section class="port-plane-project" data-port-project={project.name}>
      <button
        class="port-plane-name"
        data-port-control={`project-${project.name}`}
        data-port-part={projectPart(project.name)}
        data-port-kind="project"
        data-port-title={project.name}
        aria-current={project.current ? "page" : undefined}
      >{project.name}</button>
      {#if project.problem !== null}
        <div class="port-plane-note" title={project.problem}>unreachable</div>
      {/if}
      {#each project.groups as group (group.state)}
        <div class="port-plane-word" data-port-word={group.state}>{group.state === "" ? "no word" : group.state}</div>
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
            title={[agent.msg, agent.progress].filter((part) => part !== null).join(" - ")}
          >
            {#if agent.incarnation === null}
              <span class="port-plane-incarnation">{agent.label ?? agent.address}</span>
              <span class="port-plane-task">unregistered</span>
            {:else}
              <span class="port-plane-incarnation">{agent.incarnation}</span>
              <span class="port-plane-task">{agent.tasks[0] ?? ""}</span>
            {/if}
            {#if agent.msg !== null}<span class="port-plane-msg">{agent.msg}</span>{/if}
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
    (t-55): named by Herdr's where label (else the terminal's title)
    and the address, each with its word as the core reports it -->
    <section class="port-plane-project" data-port-control="unregistered">
      <div class="port-plane-name port-plane-group">unregistered</div>
      {#each rail.unregistered as agent (agent.address)}
        <button
          class="port-plane-agent"
          data-port-control={`unregistered-${agent.address}`}
          data-port-part={railAgentPart(null, agent)}
          data-port-kind="agent"
          data-port-title={railAgentText(agent)}
          data-port-unregistered=""
          data-port-word={agent.state}
          title={[agent.msg, agent.progress].filter((part) => part !== null).join(" - ")}
        >
          <span class="port-plane-incarnation">{agent.label ?? ""}</span>
          <span class="port-plane-task">{agent.address}</span>
          <span class="port-plane-msg">{[agent.state === "" ? "no word" : agent.state, agent.msg].filter((part) => part !== null).join(" ")}</span>
        </button>
      {/each}
    </section>
  {/if}
  {#if rail.herdr !== null}
    <div class="port-plane-note" data-port-control="herdr-note">{rail.herdr}</div>
  {/if}
</nav>
<div class="port-tasks-view">
  {#if content.status !== "loading" && content.project === null && rail.projects.length === 0}
    <div class="port-tasks-none" data-port-control="plane-empty">no project yet: press a and type its host:port/name</div>
  {:else if content.status !== "loading" && content.project === null}
    <div class="port-tasks-none" data-port-control="plane-empty">choose a project on the rail, or press a to add one</div>
  {:else if content.problem !== null}
    <div class="port-tasks-none" data-port-control="plane-problem">{content.project}: {content.problem}</div>
  {/if}
  {#if page !== null && content.status === "loaded"}
    {#each page.groups as group (group.part)}
      <section class="port-tasks-group" data-port-part={group.part} data-port-kind="group">
        {#if group.title !== ""}
          <h2 class="port-tasks-title">{group.title}</h2>
        {/if}
        {#if group.rows.length === 0}
          {#if group.title !== ""}
            <div class="port-tasks-none">none</div>
          {/if}
        {:else}
          <div class="port-tasks-columns" aria-hidden="true">
            <span>task</span><span>title</span><span class="port-tasks-agents">agents</span><span class="port-tasks-commit">latest commit</span><span>age</span>
          </div>
          {#each group.rows as row (row.part)}
            <div
              class="port-tasks-row"
              data-port-part={row.part}
              data-port-kind="task"
              data-port-task={row.task}
              data-port-open={row.location}
              data-port-title={row.title}
              data-port-hover={hovered === row.part ? "" : undefined}
            >
              <span class="port-tasks-id">{row.task}</span>
              <span class="port-tasks-title">{row.title}</span>
              <span class="port-tasks-agents">{row.agents}</span>
              <span class="port-tasks-commit">{row.commit}</span>
              <span class="port-tasks-age">{ageOf(row.at, now)}</span>
            </div>
          {/each}
        {/if}
      </section>
    {/each}
  {/if}
</div>
</div>
{#if select !== null}
  <LabelOverlay chips={select.labels} measure={partMeasureIn(host)} typed={select.pendingLabel} />
{/if}
{#if menu !== null}
  <ActMenu {menu} {host} session={controller.menuSession()} />
{/if}
</div>
