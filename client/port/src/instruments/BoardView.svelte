<!-- The Board instrument's component: renders the published domain
frame and hydrates it. Presentation only - it receives the content and
its controller through the instrument props, never refresh, lifecycle
or action-run access. Rendering and hydration leave authored source
visible on every conversion failure; a stale publication drops itself
at the generation boundary. -->

<script lang="ts">
  import { hydrate } from "../lib/board/hydrate.ts";
  import { renderBoard, type RenderHooks } from "../lib/render.ts";
  import type { BoardData, BoardUI, RenderOutcome, SelectSessionView } from "./board.ts";

  let { content, controller }: { content: BoardData; controller: BoardUI } = $props();

  let host: HTMLElement | null = $state(null);
  let renderNotes: string[] = $state([]);
  let select: SelectSessionView | null = $state(null);

  // The select session's live view: chips painted over the labeled
  // parts' rendered boxes. Presentation only - the narrowing logic
  // lives in the controller, behind the shared matcher.
  $effect(() => {
    const off = controller.subscribeSelect((view) => {
      select = view;
    });
    return off;
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

  $effect(() => {
    const frame = content;
    const element = host;
    if (element === null) return;
    const location = frame.resolved;
    if (frame.status !== "loaded" || frame.board === null || location === null) {
      element.replaceChildren();
      return;
    }
    const generation = controller.currentGeneration();
    const hooks: RenderHooks = {
      unitIndex: new Map(),
      onDiagnostic: (diagnostic) => {
        renderNotes.push(diagnostic.message);
      },
    };
    const container = renderBoard(frame.board, hooks);
    element.replaceChildren(container);
    const collected: RenderOutcome[] = [];
    void hydrate(container, controller, location, generation, collected).then(
      (outcomes) => {
        if (controller.currentGeneration() !== generation) return;
        controller.reportRenderOutcomes(generation, outcomes);
      },
    );
  });
</script>

<div class="port-board-view" bind:this={host}>
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
