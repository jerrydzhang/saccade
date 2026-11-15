<script lang="ts">
  // Find: the bar on the float surface, every match tinted, the active
  // one as the selection; then the count left when find closes.
  import BoardFrame from "../board/BoardFrame.svelte";
  import { board, controller, findView, frame } from "../board/fixture.ts";
  import { DOCUMENT } from "../board/boards.ts";

  const content = frame(board(DOCUMENT));
  const parts = [["/content/1", 2], ["/content/2", 1]] as const;
</script>

<div class="stack">
  <BoardFrame {content} controller={controller({ find: findView("region", parts, "/content/1") })} height={760} scrollTo={200} />
  <BoardFrame {content} controller={controller({ find: findView("region", parts, "/content/2", false) })} width={900} height={520} scrollTo={200} />
</div>

<style>
  .stack {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }
</style>
