<!-- The act menu, painted: one component for every instrument that
opens one (the board, the task list). It places itself in the stage
beside its target or at the pointer (lib/menu/geometry.ts), closes on
an outside mouse-down or wheel (lib/menu/dismiss.ts), and runs a
clicked act through the session. The session owns what the acts are
and what a close means; this component is presentation only. -->

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
</script>

<div
  class="port-menu"
  data-port-menu=""
  data-port-menu-part={menu.part ?? ""}
  bind:this={element}
  style="left: {at.x}px; top: {at.y}px"
>
  <div class="port-menu-label">{menu.label}</div>
  {#each menu.acts as act, index (act.key)}
    <button
      class="port-menu-act"
      data-port-menu-act={act.key}
      onclick={(event) => {
        event.stopPropagation();
        session.choose(index);
      }}>{act.key} {act.label}{#if index === 0}<span class="port-menu-enter">{"  \u21b5"}</span>{/if}</button>
  {/each}
</div>
