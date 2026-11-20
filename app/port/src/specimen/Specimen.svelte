<script lang="ts">
  // Every section under ./sections, in file-name order: a block's
  // owner adds its section beside its block; nothing here lists them.
  import type { Component } from "svelte";

  let { only = null }: { only?: string | null } = $props();

  const modules = import.meta.glob<{ default: Component }>("./sections/*.svelte", { eager: true });
  const sections = Object.entries(modules)
    .map(([path, module]) => ({ name: path.replace(/^.*\/\d*-?/, "").replace(/\.svelte$/, ""), Section: module.default, path }))
    .sort((a, b) => a.path.localeCompare(b.path))
    .filter((section) => only === null || section.name === only);
</script>

<div class="port-specimen">
  {#each sections as section (section.path)}
    <section class="port-specimen-section" data-specimen-section={section.name}>
      <h2 class="port-specimen-title">{section.name}</h2>
      <section.Section />
    </section>
  {/each}
</div>

<style>
  /* the specimen's own frame: placement only, read from tokens */
  .port-specimen {
    min-height: 100vh;
    padding: 1.5rem 2rem 4rem;
    background: var(--port-color-background);
    color: var(--port-color-ink);
    font-family: var(--port-font-ui);
  }
  .port-specimen-section {
    margin-bottom: 3rem;
  }
  .port-specimen-title {
    font-size: var(--port-size-small);
    font-weight: 400;
    color: var(--port-color-faint);
    margin: 0 0 0.75rem;
  }
</style>
