<script lang="ts">
  // The shared label overlay (order of work step 3): one component
  // both instruments paint their select labels through. Composition
  // only - the placement math lives in lib/select/overlay.ts, the
  // measurement is injected, and the layer COVERS the host (the
  // parent must be position:relative; the overlay is inset 0), so a
  // chip's host-relative coordinates land beside its part. F3 - the
  // sibling-after-the-host layer that drew every chip a content-
  // height low - is structurally impossible from this component.
  import { chipPlacements, type PartMeasure } from "../lib/select/overlay.ts";

  let {
    chips,
    measure,
  }: {
    chips: readonly { readonly part: string; readonly label: string }[];
    measure: PartMeasure;
  } = $props();

  const placements = $derived(chipPlacements(chips, measure));
</script>

<div class="port-label-overlay" aria-hidden="true">
  {#each placements as entry (entry.part)}
    {#if entry.label !== ""}
      <span
        class="port-label-chip"
        data-port-label={entry.label}
        data-port-part={entry.part}
        style="left: {entry.left}px; top: {entry.top}px">{entry.label}</span>
    {/if}
  {/each}
</div>
