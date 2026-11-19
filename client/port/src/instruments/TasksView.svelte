<!-- The Tasks instrument's component: renders the published page
model (groups, tables, task rows) inside the frame's surface - the
frame draws the select labels and the act menu over this
content from the TargetPresentations lease the props carry, and
forwards the pointer's gestures on the marked elements. The view keeps
what is about its own data: the rail, the add field, and the row
highlight: it follows the POINTER'S OWN MOVEMENT, never a
row that scrolls under a resting pointer; a move that changes the
pointer's coordinates marks the row under it; any scroll clears the
mark. Row identity and the row's link ride the port's own
content-structure scheme: data-port-part names the row's identity,
data-port-kind names the kind, data-port-task names the task (the
return anchor's identity), data-port-open names the row's link. -->

<script lang="ts">
  import type { TargetPresentations } from "../lib/contracts/buffer.ts";
  import FrameSurface from "../components/FrameSurface.svelte";
  import PlaneRail from "./PlaneRail.svelte";
  import TasksPipeline from "./TasksPipeline.svelte";
  import type { TasksData, TasksUI } from "./tasks.ts";
  import { wordsByIncarnationOf, type Rail } from "../lib/tasks/rail.ts";

  let {
    content,
    controller,
    targets = null,
  }: {
    content: TasksData;
    controller: TasksUI;
    /** The frame's presentation of this instrument's published
     * targets; null outside the registry (the specimen hosts this
     * component bare, and an instrument with no lease gets no
     * surface). */
    targets?: TargetPresentations | null;
  } = $props();

  let rail: Rail = $state({ projects: [], unregistered: [], herdr: null });

  $effect(() => controller.subscribeRail((next) => {
    rail = next;
  }));

  // The row highlight follows the POINTER'S OWN MOVEMENT, never a row
  // that scrolls under a resting pointer; CSS :hover did
  // exactly that as j/k scrolled). A move that changes the pointer's
  // coordinates marks the row under it; any scroll clears the mark.
  // A rail entry is pointed the same way.
  let hovered: string | null = $state(null);
  let lastPointer: { x: number; y: number } | null = null;
  function pointerMoved(event: MouseEvent): void {
    if (lastPointer !== null && lastPointer.x === event.clientX && lastPointer.y === event.clientY) return;
    lastPointer = { x: event.clientX, y: event.clientY };
    const row = rowFromEvent(event);
    hovered = row instanceof HTMLElement ? row.getAttribute("data-port-part") : null;
  }
  function rowFromEvent(event: MouseEvent): HTMLElement | null {
    const element = (event.target as HTMLElement | null)?.closest?.(
      '[data-port-kind="task"], [data-port-kind="project"], [data-port-kind="agent"]',
    );
    return element instanceof HTMLElement ? element : null;
  }
  $effect(() => {
    const clear = (): void => {
      hovered = null;
    };
    document.addEventListener("scroll", clear, { capture: true, passive: true });
    return () => document.removeEventListener("scroll", clear, { capture: true });
  });

  // the ages are read against the moment the shown page arrived: a
  // follow refresh moves them with the rows
  const now = $derived((void content.page, Date.now()));

  const page = $derived(content.page);
  // each row's agents marked by their live words: the rail's join
  const words = $derived(wordsByIncarnationOf(rail));
</script>

<FrameSurface presentations={targets}>
  <!-- the stage's pointer handling is the row highlight only: the
  gestures that run acts are the frame surface's -->
  <div
    class="port-tasks-stage"
    role="presentation"
    onmousemove={pointerMoved}
    onmouseleave={() => {
      hovered = null;
    }}
  >
  <div class="port-plane" role="document">
    <PlaneRail {rail} {controller} {hovered} />
    <div class="port-tasks-view">
      {#if content.status !== "loading" && content.project === null && rail.projects.length === 0}
        <div class="port-tasks-none" data-port-control="plane-empty">no project yet: press a and type machine:/path/to/repository</div>
      {:else if content.status !== "loading" && content.project === null}
        <div class="port-tasks-none" data-port-control="plane-empty">choose a project on the rail, or press a to add one</div>
      {:else if content.problem !== null}
        <div class="port-tasks-none" data-port-control="plane-problem">{content.project}: {content.problem}</div>
      {/if}
      {#if page !== null && content.status === "loaded"}
        <TasksPipeline {page} {words} {now} {hovered} />
      {/if}
    </div>
  </div>
  </div>
</FrameSurface>
