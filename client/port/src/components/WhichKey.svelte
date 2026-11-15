<script lang="ts">
  // The key surfaces (the r110 ruling, t-59's float): the which-key
  // popup - a pending prefix's continuations, its head the prefix -
  // and help, every key set in columns by the context it comes from.
  // One floating surface for both; composition only. The views are
  // projections of the matcher's own (lib/input/help.ts).
  import type { HelpView, PendingKeysView } from "../lib/input/help.ts";

  let { view }: { view: PendingKeysView | HelpView } = $props();
</script>

{#if "groups" in view}
  <div class="port-float port-help" data-port-control="which-key" role="listbox" aria-label="keys">
    {#each view.groups as group (group.title)}
      <section class="port-help-group" data-port-help-group={group.title}>
        <div class="port-float-head">{group.title}</div>
        <div class="port-whichkey-rows">
          {#each group.rows as row (row.label)}
            <div class="port-whichkey-row" data-port-whichkey-key={row.key}>
              <kbd class="port-whichkey-key">{row.key}</kbd>
              <span class="port-whichkey-label">{row.label}</span>
            </div>
          {/each}
        </div>
      </section>
    {/each}
  </div>
{:else}
  <div class="port-float port-whichkey" data-port-control="which-key" role="listbox" aria-label="pending keys">
    <div class="port-float-head"><kbd class="port-whichkey-key">{view.prefix}</kbd> ...</div>
    <div class="port-whichkey-rows">
      {#each view.rows as row (row.key)}
        <div class="port-whichkey-row" data-port-whichkey-key={row.key}>
          <kbd class="port-whichkey-key">{row.key}</kbd>
          <span class="port-whichkey-label">{row.label}</span>
        </div>
      {/each}
    </div>
  </div>
{/if}
