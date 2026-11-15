<!-- The strip (t-59): the window's one line of chrome - the app, the
location, a note or jump chip, and the link at the right edge. The load
status shows only when it is not simply loaded (lib/strip.ts). Composition
only; the view is the window's (App.ts stripView). -->

<script lang="ts">
  import type { StripView } from "../App.ts";
  import { stripStatus } from "../lib/strip.ts";

  let { strip }: { strip: StripView | null } = $props();

  const shownStatus = $derived(stripStatus(strip?.status ?? "empty"));
</script>

<header class="port-strip">
  <span class="port-strip-name">saccade</span>
  <span class="port-strip-location" data-port-strip="location">
    {strip?.location ?? "no board"}
  </span>
  {#if shownStatus !== null}
    <span
      class="port-strip-status"
      data-port-strip="status"
      data-port-strip-failed={shownStatus.failed ? "" : undefined}>{shownStatus.text}</span>
  {/if}
  {#if strip?.note}
    <span class="port-strip-note">{strip.note}</span>
  {/if}
  {#if strip?.jumpChip}
    <span class="port-strip-chip">{strip.jumpChip}</span>
  {/if}
  <span class="port-strip-link" data-port-strip="link">
    {#if strip?.link != null}linked to <span class="port-strip-link-name">{strip.link}</span>{:else}unlinked{/if}
  </span>
</header>
