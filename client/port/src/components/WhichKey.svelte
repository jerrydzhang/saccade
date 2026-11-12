<script lang="ts">
  // The which-key popup (the r110 ruling, generalized): any pending
  // prefix's continuations, read from the active context's own
  // command tables - and the help surface (the same component over
  // the empty prefix, a recorded deliberate difference from the
  // prototype's hand-curated list). Composition only; the views come
  // from the matcher, the tunables from the caller.
  let {
    view,
    title = "",
  }: {
    view: { prefix: readonly { key: string }[]; completions: readonly { key: string; label: string }[] };
    title?: string;
  } = $props();
</script>

<div class="port-whichkey" data-port-whichkey="" role="listbox" aria-label="pending keys">
  {#if title !== ""}
    <div class="port-whichkey-title">{title}</div>
  {/if}
  {#each view.completions as row (row.key)}
    <div class="port-whichkey-row" data-port-whichkey-key={row.key}>
      <span class="port-whichkey-key">{row.key}</span>
      <span class="port-whichkey-label">{row.label}</span>
    </div>
  {/each}
</div>
