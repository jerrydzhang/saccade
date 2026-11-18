<!-- The frame's surface (t-73): the drawing the frame owns, hosted
above the instrument's content - the select labels over the targets in
view and the act menu - composed from the shared components the
instruments paint through today (LabelOverlay, ActMenu). It renders
from the TargetPresentations lease the instrument's view receives, and
forwards the pointer's gestures on a marked element to that element's
presentation: a plain click runs the primary act, a shifted or middle
click runs the act its key names, a right-click opens the act list at
the pointer. An instrument with no lease (a bare host, the specimen)
or one publishing no targets gets no surface: nothing paints, no
gesture acts, and the instrument's own content is the stage's whole
story. Presentation only - the sessions live behind the presentations. -->
<script lang="ts">
  import type { Bounds, TargetPresentations } from "../lib/contracts/buffer.ts";
  import type { SelectSessionView } from "../lib/select/session.ts";
  import type { MenuView } from "../lib/menu/session.ts";
  import { partMeasureIn } from "../lib/select/measure.ts";
  import LabelOverlay from "./LabelOverlay.svelte";
  import ActMenu from "./ActMenu.svelte";

  let {
    presentations = null,
    children,
  }: {
    presentations?: TargetPresentations | null;
    children?: import("svelte").Snippet;
  } = $props();

  let content: HTMLElement | null = $state(null);
  let select: SelectSessionView | null = $state(null);
  let menu: MenuView | null = $state(null);

  // the surface renders what the presentations hold now; every view
  // change in the frame's sessions re-reads both
  $effect(() => {
    if (presentations === null) return;
    return presentations.subscribe(() => {
      select = presentations.selectView();
      menu = presentations.menuView();
    });
  });

  // The gesture's own measurement rides the lease: an element it
  // cannot measure has left, and a stale lease acts nothing.
  function measurePart(part: string): Bounds | null {
    const found = content?.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
    if (!(found instanceof HTMLElement)) return null;
    const box = found.getBoundingClientRect();
    // a hidden part (inside a folded group) has no geometry
    if (box.width === 0 && box.height === 0) return null;
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
  }

  function partOf(event: Event): string | null {
    const target = event.target as Element | null;
    return target?.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null;
  }

  function atOf(event: MouseEvent): { left: number; top: number } {
    const box = content?.getBoundingClientRect();
    return { left: event.clientX - (box?.left ?? 0), top: event.clientY - (box?.top ?? 0) };
  }

  function onClick(event: MouseEvent): void {
    if (presentations === null) return;
    const part = partOf(event);
    if (part === null) return;
    // a shifted or middle click is the modified click (t-47): the act
    // its key names - `o`, open elsewhere - never a second mode
    const modified = event.shiftKey || event.button === 1;
    presentations.presentTarget(part, () => measurePart(part)).activatePointer(modified ? "modified" : "plain");
  }

  function onContextMenu(event: MouseEvent): void {
    if (presentations === null) return;
    const part = partOf(event);
    if (part === null) return;
    event.preventDefault();
    presentations.presentTarget(part, () => measurePart(part)).showActions(atOf(event));
  }
</script>

<!-- The container's pointer handling is act delegation, not an
interactive element of its own: the keyboard path to the same acts is
the select session + the act menu (Enter runs the first act), so the
a11y click advisories do not apply. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div class="port-frame-stage" onclick={onClick} oncontextmenu={onContextMenu}>
  <div class="port-frame-content" bind:this={content}>
    {@render children?.()}
  </div>
  {#if select !== null}
    <LabelOverlay chips={select.labels} measure={partMeasureIn(content)} typed={select.pendingLabel} />
  {/if}
  {#if menu !== null && presentations !== null}
    <ActMenu {menu} host={content} session={presentations.menuSession()} />
  {/if}
</div>
