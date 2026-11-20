<script lang="ts">
  // The strip: the real component in its states - loaded and
  // unlinked, linked, loading, failed, and a jump chip shown.
  import Strip from "../../components/Strip.svelte";
  import type { StripView } from "../../App.ts";

  const base = { jumpChip: null, note: null };
  const strips: StripView[] = [
    { ...base, location: "pyrimel/tasks", status: "loaded", link: null },
    { ...base, location: "pyrimel/t-2@r9", status: "loaded", link: "pyrimel/tasks" },
    { ...base, location: "pyrimel/t-2@r9", status: "loading", link: "pyrimel/tasks" },
    { ...base, location: "pyrimel/t-9", status: "unavailable", link: null, note: "not a location: \"t-9x\"" },
    { ...base, location: "pyrimel/t-2@r9", status: "loaded", link: null, jumpChip: "b" },
  ];
</script>

{#each strips as strip, index (index)}
  <div class="frame"><Strip {strip} /></div>
{/each}

<style>
  .frame {
    margin-bottom: 0.75rem;
    background: var(--port-color-background);
    box-shadow: inset 0 0 0 1px var(--port-color-line);
  }
</style>
