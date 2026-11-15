<script lang="ts">
  // A window's content area, as App.svelte lays it out (the shell, the
  // main area, the scroller), at a fixed size, with the real BoardView
  // inside. The frame is a containing block for fixed children, so the
  // find bar stays in its window.
  import BoardView from "../../instruments/BoardView.svelte";
  import type { BoardData, BoardUI } from "../../instruments/board.ts";

  let {
    content,
    controller,
    width = 1400,
    height = 760,
    scrollTo = 0,
  }: { content: BoardData; controller: BoardUI; width?: number; height?: number; scrollTo?: number } = $props();

  let scroller: HTMLElement | null = $state(null);
  $effect(() => {
    const element = scroller;
    if (element === null || scrollTo === 0) return;
    const timer = setTimeout(() => element.scrollTo({ top: scrollTo }), 600);
    return () => clearTimeout(timer);
  });
</script>

<div class="frame" style="width: {width}px; height: {height}px">
  <div class="port-shell">
    <div class="port-main">
      <main class="port-content" bind:this={scroller}>
        <BoardView {content} {controller} />
      </main>
    </div>
  </div>
</div>

<style>
  .frame {
    position: relative;
    transform: translateZ(0);
    overflow: hidden;
    border-radius: 6px;
    box-shadow: 0 0 0 1px var(--port-color-line);
    max-width: 100%;
  }
</style>
