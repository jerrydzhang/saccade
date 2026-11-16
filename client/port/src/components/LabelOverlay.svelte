<script lang="ts">
  // The shared label overlay (order of work step 3): one component
  // both instruments paint their select labels through. Composition
  // only - the placement math lives in lib/select/overlay.ts, the
  // measurement is injected, and the layer covers the host (the
  // parent must be position:relative; the overlay is inset 0), so a
  // chip's host-relative coordinates land beside its part, never over
  // it: a block's in the margin, an inline target's in the room the
  // line reserves for it (lib/select/measure.ts reserveInlineRoom,
  // the reading place held). F3 - the sibling-after-the-host layer
  // that drew every chip a content-height low - is structurally
  // impossible from this component.
  import { untrack } from "svelte";
  import { chipPlacements, type PartMeasure } from "../lib/select/overlay.ts";
  import { reserveInlineRoom } from "../lib/select/measure.ts";

  let {
    chips,
    measure,
    typed = "",
  }: {
    /** `nested` (t-63): the target sits inside another labelled one;
     * its chip is drawn outlined (labels.css). */
    chips: readonly { readonly part: string; readonly label: string; readonly nested?: boolean }[];
    measure: PartMeasure;
    /** The label keystrokes typed so far: each chip shows them marked
     * (the prototype's hintKey span.typed). */
    typed?: string;
  } = $props();

  let overlay: HTMLElement | null = $state(null);
  // bumped once the inline room is in the lines: placement re-measures
  let laidOut = $state(0);
  // the room follows the labelled parts, not the typing: narrowing a
  // label never relays the text
  const parts = $derived(chips.map((chip) => chip.part).join("\n"));

  $effect(() => {
    void parts;
    const root = overlay?.parentElement ?? null;
    if (root === null) return;
    const release = untrack(() => reserveInlineRoom(root, chips));
    untrack(() => {
      laidOut += 1;
    });
    return release;
  });

  const nested = $derived(new Set(chips.filter((chip) => chip.nested === true).map((chip) => chip.part)));

  const placements = $derived.by(() => {
    void laidOut;
    return chipPlacements(chips, measure);
  });
</script>

<div class="port-label-overlay" aria-hidden="true" bind:this={overlay}>
  {#each placements as entry (entry.part)}
    {#if entry.label !== ""}
      <span
        class="port-label-chip"
        data-port-label={entry.label}
        data-port-part={entry.part}
        data-port-label-side={entry.side}
        data-port-label-nested={nested.has(entry.part) ? "" : undefined}
        style="left: {entry.left}px; top: {entry.top}px"
        >{#if typed.length > 0}<span class="port-label-typed">{entry.label.slice(0, typed.length)}</span>{entry.label.slice(typed.length)}{:else}{entry.label}{/if}</span>
    {/if}
  {/each}
</div>
