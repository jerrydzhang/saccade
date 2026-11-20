<script lang="ts">
  // The vocabulary itself: the type, the window's inks, the accent,
  // the paper, the float surface, the label stamp and the states.
  const swatches = [
    ["background", "--port-color-background"],
    ["ink", "--port-color-ink"],
    ["muted", "--port-color-muted"],
    ["faint", "--port-color-faint"],
    ["line", "--port-color-line"],
    ["accent", "--port-color-accent"],
    ["selection", "--port-color-selection"],
    ["paper", "--port-color-paper"],
    ["float", "--port-float-background"],
    ["label", "--port-label-background"],
    ["working", "--port-state-working"],
    ["blocked", "--port-state-blocked"],
    ["done", "--port-state-done"],
    ["idle", "--port-state-idle"],
    ["error", "--port-state-error"],
  ] as const;
</script>

<div class="tokens">
  <p class="type-text">Serif for the tools and the document, <span class="type-mono">t-2@r9 i-3 77be1f2</span> in mono.</p>
  <div class="swatches">
    {#each swatches as [name, token] (token)}
      <div class="swatch"><span class="chip" style="background: var({token})"></span>{name}</div>
    {/each}
  </div>
</div>

<style>
  .type-text {
    font-family: var(--port-font-text);
    font-size: 1.05rem;
  }
  .type-mono {
    font-family: var(--port-font-mono);
    font-size: 0.9em;
  }
  .swatches {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(9rem, 1fr));
    gap: 0.5rem 1rem;
  }
  .swatch {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--port-color-muted);
    font-size: var(--port-size-small);
  }
  .chip {
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 4px;
    box-shadow: inset 0 0 0 1px var(--port-color-line);
  }
</style>
