<script lang="ts">
  // Select labels: the real overlay over fixture blocks and inline
  // refs, measured as the app measures them - a block's stamp in the
  // margin to its left, a ref's just before it, neither over its text.
  import LabelOverlay from "../../components/LabelOverlay.svelte";
  import { partMeasureIn } from "../../lib/select/measure.ts";

  const chips = [
    { part: "/content/0", label: "A" },
    { part: "/content/1", label: "S" },
    { part: "/content/1/ref/0", label: "D" },
    { part: "/content/1/ref/1", label: "F" },
    { part: "/content/2", label: "J" },
    { part: "/content/2/ref/0", label: "K" },
  ];
  let host: HTMLElement | null = $state(null);
  // measure once the faces have landed, as the app does after layout
  let ready = $state(false);
  void document.fonts.ready.then(() => {
    ready = true;
  });
</script>

<div class="stage">
  <div class="paper" bind:this={host}>
    <h2 data-port-part="/content/0">Why this is being asked</h2>
    <p data-port-part="/content/1">
      The parent discussion (<a class="port-ref" data-port-part="/content/1/ref/0">t-0@r9</a>, first group) walked its
      behavioural atlas through a working form of a loss by hand estimate only. Those estimates are now being leaned on;
      see <a class="port-ref" data-port-part="/content/1/ref/1">t-29</a> for the table.
    </p>
    <p data-port-part="/content/2">
      <a class="port-ref" data-port-part="/content/2/ref/0">t-3</a> rescored the earlier candidates. A table that
      disagrees with the hand estimates is a fully successful result.
    </p>
    {#if ready && host !== null}
      <LabelOverlay {chips} measure={partMeasureIn(host)} typed="" />
    {/if}
  </div>
</div>

<style>
  .stage {
    max-width: 52rem;
  }
  .paper {
    position: relative;
    padding: 1.5rem 3rem 1.5rem 5rem;
    background: var(--port-color-paper);
    color: var(--port-color-paper-ink);
    box-shadow: var(--port-paper-shadow);
    border-radius: var(--port-radius);
    line-height: 1.62;
  }
  h2 {
    font-size: 1.15rem;
    margin: 0 0 0.5rem;
  }
  p {
    margin: 0 0 0.8rem;
    max-width: 36em;
  }
  a {
    color: var(--port-color-accent);
  }
</style>
