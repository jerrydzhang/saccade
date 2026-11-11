<!-- The Board instrument's component: renders the published domain
frame and hydrates it. Presentation only - it receives the content and
its controller through the instrument props, never refresh, lifecycle
or action-run access. Rendering and hydration leave authored source
visible on every conversion failure; a stale publication drops itself
at the generation boundary. -->

<script lang="ts">
  import { hydrate } from "../lib/board/hydrate.ts";
  import { renderBoard, type RenderHooks } from "../lib/render.ts";
  import type { BoardData, BoardUI, MenuView, RenderOutcome, SelectSessionView } from "./board.ts";

  let { content, controller }: { content: BoardData; controller: BoardUI } = $props();

  let host: HTMLElement | null = $state(null);
  let renderNotes: string[] = $state([]);
  let select: SelectSessionView | null = $state(null);
  let menu: MenuView | null = $state(null);

  // The select session's live view: chips painted over the labeled
  // parts' rendered boxes. Presentation only - the narrowing logic
  // lives in the controller, behind the shared matcher.
  $effect(() => {
    const off = controller.subscribeSelect((view) => {
      select = view;
    });
    const offMenu = controller.subscribeMenu((view) => {
      menu = view;
    });
    return () => {
      off();
      offMenu();
    };
  });

  // Right-click opens the same act list at the pointer (scenario 9's
  // third invocation path); plain clicks on act-carrying parts run
  // the first act directly (the fourth path).
  function onContextMenu(event: MouseEvent): void {
    const part = partFromEvent(event);
    if (part === null) return;
    const target = content.targets.find((entry) => entry.part === part);
    if (target === undefined) return;
    event.preventDefault();
    const box = host?.getBoundingClientRect();
    const at =
      box === undefined
        ? undefined
        : { x: event.clientX - box.left, y: event.clientY - box.top };
    controller.openMenuFor(
      { part: target.part, kind: target.kind, label: target.text.split("\n")[0] ?? target.text },
      at,
    );
  }

  function partFromEvent(event: MouseEvent): string | null {
    const element = (event.target as HTMLElement | null)?.closest?.("[data-port-part]");
    if (element === null || element === undefined) return null;
    const part = element.getAttribute("data-port-part");
    return part === null ? null : part;
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

  $effect(() => {
    const frame = content;
    const element = host;
    if (element === null) return;
    const location = frame.resolved;
    if (frame.status !== "loaded" || frame.board === null || location === null) {
      element.replaceChildren();
      return;
    }
    // A pending load may still display this older frame. Never stamp
    // its hydration with the newer request's live generation.
    const generation = frame.generation;
    const hooks: RenderHooks = {
      unitIndex: new Map(),
      folded: new Set(content.folded),
      onDiagnostic: (diagnostic) => {
        renderNotes.push(diagnostic.message);
      },
    };
    const container = renderBoard(frame.board, hooks);
    element.replaceChildren(container);
    const collected: RenderOutcome[] = [];
    void hydrate(container, controller, location, generation, collected).then(
      (outcomes) => {
        if (controller.displayedGeneration() !== generation) return;
        controller.reportRenderOutcomes(generation, outcomes);
      },
    );
  });
</script>

<div class="port-board-view" bind:this={host} role="document" oncontextmenu={onContextMenu}>
  </div>
  {#if select !== null}
    <div class="port-select-labels" aria-hidden="true">
      {#each select.labels as entry (entry.part)}
        {@const box = chipBox(entry.part)}
        {#if box !== null && entry.label !== ""}
          <span class="port-select-label" data-port-label={entry.label} data-port-part={entry.part}
            style="left: {box.left}px; top: {box.top}px">{entry.label}</span>
        {/if}
      {/each}
    </div>
  {/if}
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
