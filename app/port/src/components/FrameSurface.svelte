<!-- The frame's surface: the drawing the frame owns, hosted
above the instrument's content - the select labels over the targets in
view and the act menu - composed from the shared components the
instruments paint through today (LabelOverlay, ActMenu). It renders
from the TargetPresentations lease the instrument's view receives, and
forwards the pointer's gestures on a marked element to that element's
presentation: a plain click runs the primary act, a shifted or middle
click runs the act its key names, a right-click opens the act list at
the pointer. The middle button is a press-then-release lifecycle (the
press arms, the release over the element acts, leaving cancels, and
the stray click the release synthesizes is dropped), and a plain click
within a right gesture's window is the WebDriver's synthesized
artifact, dropped - the guards live in lib/frame/pointer-guards.ts.
An instrument with no lease (a bare host, the specimen) or one
publishing no targets gets no surface: nothing paints, no gesture
acts, and the instrument's own content is the stage's whole story.
Presentation only - the sessions live behind the presentations. -->
<script lang="ts">
  import type { Bounds, TargetPresentations } from "../lib/contracts/buffer.ts";
  import type { SelectSessionView } from "../lib/select/session.ts";
  import type { MenuView } from "../lib/menu/session.ts";
  import { partMeasureIn } from "../lib/select/measure.ts";
  import { createPointerGuards, surfaceClick } from "../lib/frame/pointer-guards.ts";
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
  // a menu whose target is a frame-rail entry renders in the rail pane
  // (App.svelte's own ActMenu); the content's surface skips it
  let menu: MenuView | null = $state(null);
  // the part a middle press armed on: the release acts only over it
  let armedPart: string | null = null;

  const guards = createPointerGuards(() => performance.now());

  // the surface renders what the presentations hold now; every view
  // change in the frame's sessions re-reads both
  $effect(() => {
    if (presentations === null) return;
    return presentations.subscribe(() => {
      select = presentations.selectView();
      const view = presentations.menuView();
      menu = view !== null && view.part.startsWith("rail/") ? null : view;
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
    const lease = presentations.presentTarget(part, () => measurePart(part));
    // the default dies inside surfaceClick: a marked element may
    // carry its own (the citation links are anchors with href '#'),
    // and the dropped artifacts die with it too
    surfaceClick(event, lease, part, guards);
  }

  function onContextMenu(event: MouseEvent): void {
    guards.contextmenu();
    if (presentations === null) return;
    const part = partOf(event);
    if (part === null) return;
    event.preventDefault();
    presentations.presentTarget(part, () => measurePart(part)).showActions(atOf(event));
  }

  // the middle button: the press arms (and keeps the platform's
  // middle behaviour out), the release over the same element acts the
  // modified click, leaving the surface cancels
  function onMouseDown(event: MouseEvent): void {
    if (event.button !== 1) return;
    event.preventDefault();
    guards.middlePress();
    armedPart = partOf(event);
  }

  // The release's own button is not trusted - this stack's WebDriver
  // reports a middle release (and the click it synthesizes after it)
  // as button 0 (recorded live in the native lane) - so the press
  // decides and the release over the armed element acts.
  function onMouseUp(event: MouseEvent): void {
    const overArmed = armedPart !== null && armedPart === partOf(event);
    armedPart = null;
    if (!guards.middleRelease() || !overArmed) return;
    const part = partOf(event);
    if (part === null) return;
    presentations?.presentTarget(part, () => measurePart(part)).activatePointer("modified");
  }

  function onStageLeave(): void {
    armedPart = null;
    guards.middleLeave();
  }

  function onAuxClick(event: MouseEvent): void {
    if (guards.middleAuxClick(event.button)) event.preventDefault();
  }
</script>

<!-- The container's pointer handling is act delegation, not an
interactive element of its own: the keyboard path to the same acts is
the select session + the act menu (Enter runs the first act), so the
a11y click advisories do not apply. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div
  class="port-frame-stage"
  onclick={onClick}
  oncontextmenu={onContextMenu}
  onmousedown={onMouseDown}
  onmouseup={onMouseUp}
  onmouseleave={onStageLeave}
  onauxclick={onAuxClick}
>
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
