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
  import type { TasksData, TasksUI } from "./tasks.ts";

  let { content, controller }: { content: TasksData; controller: TasksUI } = $props();

  let host: HTMLElement | null = $state(null);
  let select: SelectSessionView | null = $state(null);
  let menu: MenuView | null = $state(null);

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
  function rowFromEvent(event: MouseEvent): HTMLElement | null {
    const element = (event.target as HTMLElement | null)?.closest?.('[data-port-kind="task"]');
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
    const task = row.getAttribute("data-port-task") ?? "";
    const title = row.getAttribute("data-port-title") ?? "";
    event.preventDefault();
    const box = host?.getBoundingClientRect();
    const at =
      box === undefined
        ? undefined
        : { x: event.clientX - box.left, y: event.clientY - box.top };
    controller.openMenuFor(
      { part, kind: "task", label: `${task} ${title}`.trim() },
      at,
    );
  }

  function chipBox(part: string): { left: number; top: number } | null {
    const element = host;
    if (element === null) return null;
    const target = element.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
    if (!(target instanceof HTMLElement)) return null;
    const hostBox = element.getBoundingClientRect();
    const box = target.getBoundingClientRect();
    return { left: box.left - hostBox.left, top: box.top - hostBox.top };
  }

  const page = $derived(content.page);
</script>

<!-- The container's pointer handling is act delegation, not an
interactive element of its own: the keyboard path to the same acts is
the select session + the act menu (Enter runs the first act), so the
a11y click advisories do not apply. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div class="port-tasks-stage">
<div class="port-tasks-view" bind:this={host} role="document" onclick={onClick} oncontextmenu={onContextMenu}>
  {#if page !== null && content.status === "loaded"}
    {#each page.groups as group (group.part)}
      <section class="port-tasks-group" data-port-part={group.part} data-port-kind="group">
        {#if group.title !== ""}
          <h2 class="port-tasks-title">{group.title}</h2>
        {/if}
        {#each group.rows as row, rowIndex (row.part)}
          {#if rowIndex === 0}
            <div class="port-tasks-columns" aria-hidden="true">
              <span>task</span><span>title</span><span>agents</span><span>latest commit</span><span>latest revision</span>
            </div>
          {/if}
          <div
            class="port-tasks-row"
            data-port-part={row.part}
            data-port-kind="task"
            data-port-task={row.task}
            data-port-open={row.location}
            data-port-title={row.title}
          >
            <span class="port-tasks-id">{row.task}</span>
            <span class="port-tasks-title">{row.title}</span>
            <span class="port-tasks-agents">{row.agents}</span>
            <span class="port-tasks-commit">{row.commit}</span>
            <span class="port-tasks-revision">{row.revision}</span>
          </div>
        {/each}
      </section>
    {/each}
  {/if}
</div>
{#if select !== null}
  <LabelOverlay chips={select.labels} measure={chipBox} />
{/if}
</div>
{#if menu !== null}
  <div class="port-menu" data-port-menu="">
    <div class="port-menu-label">{menu.label}</div>
    {#each menu.acts as act, index (act.key)}
      <button
        class="port-menu-act"
        data-port-menu-act={act.key}
        onclick={(event) => {
          event.stopPropagation();
          controller.menuSession().choose(index);
        }}>{act.key} {act.label}</button>
    {/each}
  </div>
{/if}
