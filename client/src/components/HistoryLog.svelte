<script lang="ts">
  // History's docked log (round 16's renderLog): every revision, newest
  // first, the shown one expanded with its full message, hash, time,
  // session and how many parts it changed; any entry goes there.
  import type { HistoryEntry } from "../lib/pages.ts";

  type Props = {
    task: string;
    /** The commit entries, newest first. */
    commits: HistoryEntry[];
    at: number;
    /** Parts the shown revision changed; null while it is counted. */
    changes: number | null;
    onGo: (revision: number) => void;
  };

  let { task, commits, at, changes, onGo }: Props = $props();
  let list: HTMLElement | undefined = $state(undefined);

  const when = (seconds: number): string => {
    const date = new Date(seconds * 1000);
    const pad = (n: number): string => String(n).padStart(2, "0");
    return `${pad(date.getMonth() + 1)}/${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  };

  $effect(() => {
    void at;
    list?.querySelector<HTMLElement>(".log-entry.cur")?.scrollIntoView({ block: "nearest" });
  });
</script>

<aside class="history-log" data-control="history:margin" bind:this={list}>
  <p class="sec log-head">history - {task} - {commits.length} revision{commits.length === 1 ? "" : "s"}</p>
  {#each commits as entry (entry.seq)}
    {@const current = entry.revision === at}
    <button
      type="button"
      class="log-entry"
      class:cur={current}
      data-control="history:entry"
      data-revision={entry.revision}
      onclick={() => entry.revision !== null && onGo(entry.revision)}
    >
      <span class="l1"><span class="r">r{entry.revision}</span>{entry.message ?? ""}</span>
      {#if current}
        <span class="meta">
          {(entry.commit ?? "").slice(0, 8)} - {when(entry.at)} - {entry.session?.label ?? entry.actor}{changes !== null ? ` - ${changes} part${changes === 1 ? "" : "s"} changed` : ""}
        </span>
      {/if}
    </button>
  {/each}
  <p class="log-hint">h l step time - n N changes - c current - t tracks, H L walk it - space h leaves</p>
</aside>
