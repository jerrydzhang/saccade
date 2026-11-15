<script lang="ts">
  // The board document on its sheet: headings, prose, a list, a link
  // and refs, code, math, a figure, an annotation, an incorporation
  // collapsed and one expanded, and a group - the real renderer and
  // hydration, from the authored YAML.
  import BoardFrame from "../board/BoardFrame.svelte";
  import { board, controller, expanded, frame } from "../board/fixture.ts";
  import { CITED, DOCUMENT } from "../board/boards.ts";

  const content = frame(board(DOCUMENT));
  const ui = controller({ expansions: [expanded("/content/6/content/1", board(CITED), ["/content/1"])] });
</script>

<BoardFrame {content} controller={ui} height={2050} />
