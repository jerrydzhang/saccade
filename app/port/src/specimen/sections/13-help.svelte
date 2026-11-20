<script lang="ts">
  // Help (?): the real surface over a board's reading table and the
  // window's, in columns by the context the keys come from.
  import WhichKey from "../../components/WhichKey.svelte";
  import { boardReadingContext, windowContext } from "../../lib/input/contexts.ts";
  import { helpGroups } from "../../lib/input/help.ts";
  import { createMatcher } from "../../lib/input/key-matcher.ts";

  const none = () => {};
  const board = createMatcher(
    boardReadingContext({
      scrollToTop: none, scrollToBottom: none, scrollDown: none, scrollUp: none,
      scrollFraction: none, enterSelect: none, enterFind: none, nextMatch: none,
      prevMatch: none, clearSearch: none, readingLayout: none,
      historyStep: none, historyTrack: none, historyWalk: none,
    }),
  );
  const windowKeys = createMatcher(
    windowContext({
      openPicker: none, closeOverlays: none, pickerIsOpen: () => false,
      trailBack: () => true, trailForward: () => true, toggleHistory: none,
      linkByLabels: none, unlink: none, showHelp: none,
    }),
  );
  const view = helpGroups([
    { title: "this board", view: board.allView() },
    { title: "the window", view: windowKeys.allView() },
  ]);
</script>

<div class="stage">
  <WhichKey {view} />
</div>

<style>
  .stage {
    position: relative;
    height: 44rem;
    transform: translateZ(0);
  }
</style>
