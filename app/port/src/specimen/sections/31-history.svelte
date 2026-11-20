<script lang="ts">
  // History, where light is news: the changed parts lit with their
  // word diff, the rest resting. Wide, the commit reads whole in the
  // margin card; narrow, in the footer at the reading width. Last, the
  // refusal in the card's place.
  import BoardFrame from "../board/BoardFrame.svelte";
  import { board, controller, historyFrame } from "../board/fixture.ts";
  import { HISTORY_AFTER, HISTORY_BEFORE, HISTORY_CHANGES, HISTORY_MESSAGE, DOCUMENT } from "../board/boards.ts";
  import { frame } from "../board/fixture.ts";

  const view = { task: "t-2", revision: 9, latest: 9, message: HISTORY_MESSAGE, tracked: null, trackedLabel: null };
  const wide = historyFrame(board(HISTORY_BEFORE), board(HISTORY_AFTER), HISTORY_CHANGES, view);
  const tracked = historyFrame(board(HISTORY_BEFORE), board(HISTORY_AFTER), HISTORY_CHANGES, {
    ...view,
    tracked: "/content/4",
    trackedLabel: "A table that disagrees with the hand estimates is...",
  });
  const refusal = "History needs sac 0.20.7 or later; this tracker serves 0.20.4.";
</script>

<div class="stack">
  <BoardFrame content={wide} controller={controller()} width={1400} height={980} />
  <BoardFrame content={tracked} controller={controller()} width={760} height={980} scrollTo={240} />
  <BoardFrame content={frame(board(DOCUMENT))} controller={controller({ refusal })} width={1400} height={420} />
  <BoardFrame content={frame(board(DOCUMENT))} controller={controller({ refusal })} width={760} height={420} />
</div>

<style>
  .stack {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }
</style>
