<script lang="ts">
  // The strip (round 16, the ledger; t-34@r6 changes 5 and 6): every
  // window's one line. The window's name or location, state only when it
  // matters, then the window's chips and its link. No fields.

  type Props = {
    /** "viewer" shows the location; an instrument shows its name. */
    kind: "viewer" | "tasks" | "agents";
    tracker: string | null;
    location: string | null;
    /** In history on an older revision: "current rN". */
    current: string | null;
    state: string | null;
    note: string | null;
    deckName: string | null;
    deckState: string | null;
    onCopyCue: () => void;
    marginalia: boolean;
    zoom: number;
    jumpChip: string | null;
    link: string | null;
  };

  let { kind, tracker, location, current, state, note, deckName, deckState, onCopyCue, marginalia, zoom, jumpChip, link }: Props = $props();
</script>

<header class="strip" data-control="strip">
  {#if kind !== "viewer"}
    <span class="name">{kind}</span>
    {#if tracker}<span class="chip">{tracker}</span>{/if}
  {:else if location}
    <span class="loc" data-control="board:location">{location}</span>
    {#if current}<span class="dim3">{current}</span>{/if}
  {:else}
    <span class="dim3">no board - space space opens the picker</span>
  {/if}
  {#if state}<span class="chip warn" data-control="strip:state">{state}</span>{/if}
  {#if note}<span class="chip" role="status" data-control="instrument:note">{note}</span>{/if}
  {#if deckName}
    <span class="chip on" data-control="deck:line">deck: {deckName}</span>
    <button type="button" class="chip" data-control="deck:copy" onclick={onCopyCue}>copy cue command</button>
  {/if}
  {#if deckState}
    <span class="chip" data-control="deck:state">{deckState}</span>
  {/if}
  <span class="sp"></span>
  {#if marginalia}<span class="chip on">marginalia</span>{/if}
  {#if zoom !== 1}<span class="dim3">{Math.round(zoom * 100)}%</span>{/if}
  {#if jumpChip !== null}
    <span class="jump-chip-self" data-control="jump:self">{jumpChip}</span>
  {/if}
  {#if link !== null}
    <span class="lnk" data-control="link:target">-&gt; {link}</span>
  {:else if kind !== "viewer"}
    <span class="lnk none">unlinked</span>
  {/if}
</header>
