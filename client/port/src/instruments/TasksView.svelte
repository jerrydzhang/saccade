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
import { placeMenuInStage } from "../lib/menu/geometry.ts";
import { bindMenuDismissal } from "../lib/menu/dismiss.ts";
import { ageOf } from "../lib/tasks/page.ts";
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

  const now = $state(Date.now());
let menuElement: HTMLElement | null = $state(null);
  let menuAt = $state({ x: 4, y: 4 });

  // An open menu closes on a pointer-down or a wheel outside it
  // (lib/menu/dismiss.ts); bound only while the menu is painted.
  $effect(() => {
    const element = menuElement;
    if (element === null) return;
    return bindMenuDismissal(element, controller.menuSession());
  });

  $effect(() => {
    if (menu === null) return;
    const element = menuElement;
    const stageElement = host?.parentElement ?? null;
    if (element === null || stageElement === null) return;
    const hostElement = host;
    if (hostElement === null) return;
    const targetElement =
      menu.part !== null
        ? hostElement.querySelector(`[data-port-part=${JSON.stringify(menu.part)}]`)
        : null;
    menuAt = placeMenuInStage({ menu: element, stage: stageElement, host: hostElement, at: menu.at, target: targetElement });
  });

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
<div class="port-tasks-view" bind:this={host} role="document" onclick={onClick} oncontextmenu={onContextMenu}>
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
            <span>task</span><span>title</span><span>agents</span><span>latest commit</span><span>age</span>
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
{#if select !== null}
  <LabelOverlay chips={select.labels} measure={chipBox} typed={select.pendingLabel} />
{/if}
{#if menu !== null}
  <div
      class="port-menu"
      data-port-menu=""
      data-port-menu-part={menu.part ?? ""}
      bind:this={menuElement}
      style="left: {menuAt.x}px; top: {menuAt.y}px"
    >
    <div class="port-menu-label">{menu.label}</div>
    {#each menu.acts as act, index (act.key)}
      <button
        class="port-menu-act"
        data-port-menu-act={act.key}
        onclick={(event) => {
          event.stopPropagation();
          controller.menuSession().choose(index);
        }}>{act.key} {act.label}{#if index === 0}<span class="port-menu-enter">{"  \u21b5"}</span>{/if}</button>
    {/each}
  </div>
{/if}
</div>
