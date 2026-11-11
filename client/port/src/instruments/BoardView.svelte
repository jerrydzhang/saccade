<!-- The Board instrument's component: renders the published domain
frame and hydrates it. Presentation only - it receives the content and
its controller through the instrument props, never refresh, lifecycle
or action-run access. Rendering and hydration leave authored source
visible on every conversion failure; a stale publication drops itself
at the generation boundary. -->

<script lang="ts">
  import { hydrate } from "../lib/board/hydrate.ts";
  import { renderBoard, type RenderHooks } from "../lib/render.ts";
  import type { BoardData, BoardUI, RenderOutcome } from "./board.ts";

  let { content, controller }: { content: BoardData; controller: BoardUI } = $props();

  let host: HTMLElement | null = $state(null);
  let renderNotes: string[] = $state([]);

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

<div class="port-board-view" bind:this={host}></div>
