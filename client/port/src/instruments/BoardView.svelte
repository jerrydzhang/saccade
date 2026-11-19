<!-- The Board instrument's component: renders the published domain
frame and hydrates it. Presentation only - it receives the content and
its controller through the instrument props, never refresh, lifecycle
or action-run access. Rendering and hydration leave authored source
visible on every conversion failure; a stale publication drops itself
at the generation boundary. -->

<script lang="ts">
  import { untrack } from "svelte";
  import { hydrate } from "../lib/board/hydrate.ts";
  import { paintMarks } from "../lib/find/marks.ts";
import FrameSurface from "../components/FrameSurface.svelte";
import type { TargetPresentations } from "../lib/contracts/buffer.ts";
  import { renderBoard, type RenderHooks } from "../lib/render.ts";
  import { paintDiffMarks } from "../lib/board/history-diff.ts";
  import { paintExpansions } from "../lib/board/expansion-paint.ts";
  import type { BoardData, BoardUI, FindView, RenderOutcome } from "./board.ts";

  let { content, controller, targets = null }: { content: BoardData; controller: BoardUI; targets?: TargetPresentations | null } = $props();

  let host: HTMLElement | null = $state(null);
  let renderNotes: string[] = $state([]);
  let find: FindView = $state({
    open: false,
    query: "",
    count: 0,
    active: -1,
    unitParts: [],
    activeUnit: null,
    marks: new Map(),
  });

  const findOpen = $derived(find.open);
  let findInput: HTMLInputElement | null = $state(null);
  let findValue = $state("");

  $effect(() => {
    const off = controller.subscribeFind((view) => {
      find = view;
      if (document.activeElement !== findInput) findValue = view.query;
    });
    return off;
  });

  $effect(() => {
    // Count/query notifications do not refocus a blurred field.
    if (findOpen) findInput?.focus({ preventScroll: true });
  });

  // The render's own key: a frame republished for anything else (the
  // render outcomes this very render reports, a follow poll answering
  // the SAME revision with equal content as new objects, a find or
  // menu change) leaves the key equal and the board is NOT rebuilt.
  // Rebuilding put every figure back to its placeholder and refetched
  // it - right when the outcomes said the figures were displayed - so
  // s/x or a click in that window found no asset (D3/D4). A pinned
  // revision (repo/t-N@rK) is immutable, so its address names its
  // content; any other source keys on the board object itself.
  let boardSerial = 0;
  let lastBoard: unknown = null;
  const renderKey = $derived.by(() => {
    const path = content.resolved?.path ?? "";
    let identity = path;
    if (!/@r\d+$/.test(path)) {
      if (content.board !== lastBoard) {
        lastBoard = content.board;
        boardSerial += 1;
      }
      identity = `${path}#${boardSerial}`;
    }
    return [
      content.status,
      String(content.generation),
      content.resolved?.machine ?? "",
      identity,
      content.board === null ? "no-board" : "board",
      // history mode renders a diffed board at the same pinned path
      content.history ? `history@r${content.history.revision}` : "",
    ].join("\u0000");
  });
  let renders = $state(0);
  // the units hydrated so far (the readout's presentation.hydrated)
  let hydrated = 0;

  $effect(() => {
    // tracked: the render key and the host element, nothing else (a
    // prop object such as the controller reads as changed on every
    // republication, so the render body runs untracked)
    void renderKey;
    const element = host;
    if (element === null) return;
    untrack(() => renderInto(element));
  });

  function renderInto(element: HTMLElement): void {
    const frame = content;
    const board = frame.board;
    const location = frame.resolved;
    if (frame.status !== "loaded" || board === null || location === null) {
      element.replaceChildren();
      renders += 1;
      return;
    }
    // A pending load may still display this older frame. Never stamp
    // its hydration with the newer request's live generation.
    const generation = frame.generation;
    const hooks: RenderHooks = {
      unitIndex: new Map(),
      folded: new Set(frame.folded),
      onDiagnostic: (diagnostic) => {
        renderNotes.push(diagnostic.message);
      },
    };
    const container = renderBoard(board, hooks);
    if (frame.history) paintDiffMarks(container);
    element.replaceChildren(container);
    renders += 1;
    controller.reportPresentation({ renders, hydrated });
    controller.reportRendered(generation);
    const collected: RenderOutcome[] = [];
    void hydrate(container, controller, location, generation, collected).then(
      (outcomes) => {
        hydrated += outcomes.length;
        controller.reportPresentation({ renders, hydrated });
        if (controller.displayedGeneration() !== generation) return;
        controller.reportRenderOutcomes(generation, outcomes);
      },
    );
  }

  // A fold is an attribute on the rendered group (m-547 (5)): the
  // folded set is applied to the board as it stands, never a rebuild.
  const foldedKey = $derived(content.folded.join("\n"));
  $effect(() => {
    const element = host;
    void renders;
    const folded = new Set(foldedKey.length === 0 ? [] : foldedKey.split("\n"));
    if (element === null) return;
    for (const group of element.querySelectorAll<HTMLElement>(".block-group[data-port-part]")) {
      const part = group.getAttribute("data-port-part") ?? "";
      if (folded.has(part)) group.setAttribute("data-port-folded", "");
      else group.removeAttribute("data-port-folded");
    }
  });

  // t-58: the expanded incorporations, painted onto the board as it
  // stands (as a fold): after every render and every toggle, each
  // fresh cited container hydrated against the cited revision.
  let expansionChanges = $state(0);
  let historyRefusal: string | null = $state(null);
  $effect(() => controller.subscribeHistoryRefusal((next) => {
    historyRefusal = next;
  }));
  $effect(() => controller.subscribeExpansions(() => {
    expansionChanges += 1;
  }));
  $effect(() => {
    const element = host;
    void renders;
    void expansionChanges;
    if (element === null) return;
    untrack(() => {
      const generation = content.generation;
      paintExpansions(
        element,
        controller.expansions(),
        () => ({ unitIndex: new Map(), onDiagnostic: (diagnostic) => renderNotes.push(diagnostic.message) }),
        (container, expansion) => {
          if (expansion.location !== null) void hydrate(container, controller, expansion.location, generation, []);
        },
      );
    });
  });

  // D2: the match marks on the matched TEXT (lib/find/marks.ts): a
  // mark on every occurrence, the active one distinct, repainted on
  // every find-view change (typing, stepping, commit, clear) and
  // after every render (declared after the render effect, so it runs
  // on the fresh DOM). A step brings the active mark into view - the
  // nearest scroll, inside the part the controller already revealed.
  let paintedActive = -1;
  $effect(() => {
    const element = host;
    const view = find;
    void renders;
    if (element === null) return;
    const active = paintMarks(element, { query: view.query, parts: view.marks });
    const stepped = view.active !== paintedActive;
    paintedActive = view.active;
    if (active !== null && stepped && active.tagName === "MARK") {
      active.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  });
</script>

{#if findOpen}
  <div class="port-find">
    <input
      bind:this={findInput}
      bind:value={findValue}
      data-port-control="find-input"
      data-port-captured-input=""
      aria-label="find"
      spellcheck="false"
      oninput={(event) => controller.typeFind(event.currentTarget.value)}
    />
    <span class="port-find-status" data-port-find-status="">
      {find.count > 0 ? `${find.active + 1} / ${find.count}` : find.query.trim().length > 0 ? "no match" : ""}
    </span>
  </div>
{:else if find.count > 0}
  <div class="port-find-indicator" data-find-indicator="">
    {find.active + 1} / {find.count}
  </div>
{/if}
<!-- the board: its sheet, and the margin the commit card reads in (a
footer under the sheet when the window is too narrow for a margin) -->
<div class="port-board" data-port-history={content.history ? "" : undefined}>
  <div class="port-board-sheet">
    <div class="port-board-stage">
      <FrameSurface presentations={targets}>
        <div class="port-board-view" bind:this={host} role="document">
        </div>
      </FrameSurface>
    </div>
  </div>
  {#if historyRefusal !== null && !content.history}
    <aside class="port-history" data-port-control="history-refusal" aria-label="history">
      <div class="port-history-card">
        <div class="port-history-message">{historyRefusal}</div>
      </div>
    </aside>
  {/if}
  {#if content.history}
    <!-- the aside spans the board's height; its card sticks to the top
    of the view, so the commit message reads beside the change wherever
    it sits; narrow, it is the footer, stuck to the view's bottom -->
    <aside class="port-history" data-port-control="history" aria-label="history">
      <div class="port-history-card">
        <div class="port-history-head" data-port-control="history-revision">
          {`r${content.history.revision} of ${content.history.latest} \u00b7 ${content.history.changed.length} changed`}
        </div>
        {#if content.history.tracked !== null}
          <div class="port-history-head" data-port-control="history-tracked">tracking {content.history.trackedLabel}</div>
        {/if}
        <div class="port-history-message" data-port-control="history-message">{content.history.message}</div>
      </div>
    </aside>
  {/if}
</div>
