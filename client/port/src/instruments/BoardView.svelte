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
import LabelOverlay from "../components/LabelOverlay.svelte";
import { placeMenuInStage } from "../lib/menu/geometry.ts";
import { bindMenuDismissal } from "../lib/menu/dismiss.ts";
  import { renderBoard, type RenderHooks } from "../lib/render.ts";
  import type { BoardData, BoardUI, FindView, MenuView, RenderOutcome, SelectSessionView } from "./board.ts";

  let { content, controller }: { content: BoardData; controller: BoardUI } = $props();

  let host: HTMLElement | null = $state(null);
  let renderNotes: string[] = $state([]);
  let select: SelectSessionView | null = $state(null);
  let menu: MenuView | null = $state(null);
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
    const offFind = controller.subscribeFind((view) => {
      find = view;
      if (document.activeElement !== findInput) findValue = view.query;
    });
    return () => {
      off();
      offMenu();
      offFind();
    };
  });

  $effect(() => {
    // Count/query notifications do not refocus a blurred field.
    if (findOpen) findInput?.focus({ preventScroll: true });
  });

  // Right-click opens the same act list at the pointer (scenario 9's
  // third invocation path); plain clicks on act-carrying parts run
  // the first act directly (the fourth path).
  function onContextMenu(event: MouseEvent): void {
    const citation = (event.target as HTMLElement | null)?.closest?.('[data-port-control="citation"]');
    if (citation instanceof HTMLElement) {
      // the citation is its own target (the prototype's ref shape):
      // the menu carries the reference acts, labeled by the address
      const part = citation.getAttribute("data-port-part-of");
      const address = citation.getAttribute("data-port-open") ?? "";
      if (part !== null) {
        event.preventDefault();
        const box = host?.getBoundingClientRect();
        const at =
          box === undefined
            ? undefined
            : { x: event.clientX - box.left, y: event.clientY - box.top };
        controller.openMenuFor({ part, kind: "reference", label: address }, at);
      }
      return;
    }
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
    element.replaceChildren(container);
    renders += 1;
    controller.reportPresentation({ renders, hydrated });
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
      placeholder="find in this revision"
      oninput={(event) => controller.typeFind(event.currentTarget.value)}
    />
    <span class="port-find-status" data-port-find-status="">
      {find.count > 0 ? `${find.active + 1} / ${find.count}` : "type to search"}
    </span>
  </div>
{:else if find.count > 0}
  <div class="port-find-indicator" data-find-indicator="">
    {find.active + 1} / {find.count}
  </div>
{/if}
<div class="port-board-stage">
  <div class="port-board-view" bind:this={host} role="document" oncontextmenu={onContextMenu}>
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
