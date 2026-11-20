<!-- The act menu, painted: one component for every instrument that
opens one (the board, the task list). It places itself in the stage
beside its target or at the pointer (lib/menu/geometry.ts), closes on
an outside mouse-down or wheel (lib/menu/dismiss.ts), and runs a
clicked act through the session. While it is open its target carries
data-port-selected, so selection highlights what it acts on (t-59).
The session owns what the acts are and what a close means; this
component is presentation only. -->

<script lang="ts">
  import { bindMenuDismissal } from "../lib/menu/dismiss.ts";
  import { placeMenuInStage } from "../lib/menu/geometry.ts";
  import type { MenuSession, MenuView } from "../lib/menu/session.ts";

  let {
    menu,
    host,
    session,
  }: {
    menu: MenuView;
    /** The instrument's content element; its parent is the stage the
     * menu is positioned in. */
    host: HTMLElement | null;
    session: MenuSession;
  } = $props();

  let element: HTMLElement | null = $state(null);
  let at = $state({ x: 4, y: 4 });

  $effect(() => {
    const painted = element;
    if (painted === null) return;
    return bindMenuDismissal(painted, session);
  });

  $effect(() => {
    const view = menu;
    const painted = element;
    const hostElement = host;
    const stage = hostElement?.parentElement ?? null;
    if (painted === null || hostElement === null || stage === null) return;
    const target =
      view.part !== null
        ? hostElement.querySelector(`[data-port-part=${JSON.stringify(view.part)}]`)
        : null;
    at = placeMenuInStage({ menu: painted, stage, host: hostElement, at: view.at, target });
  });

  // the target is highlighted while its menu is open, as the pointer
  // highlights a row; the mark goes with the menu or its next target
  $effect(() => {
    const part = menu.part;
    const hostElement = host;
    if (part === null || hostElement === null) return;
    const target = hostElement.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
    if (target === null) return;
    target.setAttribute("data-port-selected", "");
    return () => target.removeAttribute("data-port-selected");
  });
</script>

<div
  class="port-float port-menu"
  data-port-menu=""
  data-port-menu-part={menu.part ?? ""}
  bind:this={element}
  style="left: {at.x}px; top: {at.y}px"
>
  <div class="port-menu-label port-float-head">{menu.label}</div>
  {#each menu.acts as act, index (act.key)}
    <!-- the first act is Enter's: its highlight says so -->
    <button
      class="port-menu-act"
      data-port-menu-act={act.key}
      aria-current={index === 0 ? "true" : undefined}
      onclick={(event) => {
        event.stopPropagation();
        session.choose(index);
      }}><kbd class="port-menu-key">{act.key}</kbd><span class="port-menu-name">{act.label}</span></button>
  {/each}
</div>
