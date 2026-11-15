<script lang="ts">
  // The act menu on a task row: the real ActMenu, placed beside its
  // target by the menu's own geometry; the target lit while it is open.
  import ActMenu from "../../components/ActMenu.svelte";
  import { createMenuSession, type MenuView } from "../../lib/menu/session.ts";

  const rows = [
    ["t-0", "derive a working theory from scratch"],
    ["t-2", "numerically check the hand-estimated atlas walk"],
    ["t-5", "empirical checks for the tube score, run alongside the t-0 theory"],
    ["t-7", "tabulate the three disagreeing regions"],
  ] as const;

  const menu: MenuView = {
    open: true,
    label: "t-2 numerically check the hand-estimated atlas walk",
    acts: [
      { key: "o", label: "open board" },
      { key: "i", label: "open in..." },
      { key: "w", label: "open in a new window" },
      { key: "y", label: "copy reference" },
    ],
    at: null,
    part: "/row/t-2",
  };
  const session = createMenuSession(() => []);
  let host: HTMLElement | null = $state(null);
</script>

<div class="stage">
  <div class="rows" bind:this={host}>
    {#each rows as [id, title] (id)}
      <div class="row" data-port-part="/row/{id}"><span class="id">{id}</span><span>{title}</span></div>
    {/each}
  </div>
  {#if host !== null}
    <ActMenu {menu} {host} {session} />
  {/if}
</div>

<style>
  .stage {
    position: relative;
    height: 22rem;
    max-width: 52rem;
  }
  .row {
    display: grid;
    grid-template-columns: 3.5rem 1fr;
    padding: 0.3rem 0.6rem;
    border-radius: var(--port-radius);
  }
  .id {
    font-family: var(--port-font-mono);
    color: var(--port-color-muted);
    font-size: 0.92em;
  }
</style>
