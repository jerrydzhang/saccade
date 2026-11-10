<script lang="ts">
  // The agents window (round 16's glance): one section per live state,
  // empty ones skipped, one ledger row per session - shape, incarnation,
  // detail, task, time.
  import { rowDetail, rowTime, type AgentsSection } from "../lib/instruments/agents.ts";
  import { taskAge } from "../lib/instruments/tasks.ts";

  type Props = {
    sections: AgentsSection[];
    onSelect: (click: MouseEvent) => void;
  };

  let { sections, onSelect }: Props = $props();
  const now = (): number => Math.floor(Date.now() / 1000);
  let shown = $derived(sections.filter((section) => section.rows.length > 0));
</script>

{#if shown.length === 0}
  <p class="sec instrument-label">agents</p>
  <p class="dim3 instrument-empty">none - no session reports to this tracker</p>
{/if}
{#each shown as section, index (section.word ?? "absent")}
  <p class="sec instrument-label">{#if section.word}<span class="st {section.word}"></span>{/if}{section.word ?? "absent"}</p>
  <div class="instrument-table" role="table" aria-label="{section.word ?? "absent"} sessions" data-control="instrument:section">
    {#each section.rows as row (row.incarnation + String(index))}
      <button
        type="button"
        class="row instrument-row"
        role="row"
        data-control="instrument:row"
        data-block-id={`instrument-row-${row.incarnation}`}
        data-instrument="agents"
        onclick={onSelect}
        data-incarnation={row.incarnation}
        data-address={row.address}
        data-harness={row.harness}
        data-tasks={row.tasks.join(" ")}
        data-workspace={row.workspace}
        data-last-commit={row.lastCommit}
        data-open-path={row.tasks[0] ?? ""}
        data-word={row.word?.state ?? ""}
        data-app={row.word?.app ?? ""}
        data-msg={row.word?.msg ?? ""}
        data-progress={row.word?.progress ?? ""}
        data-title={row.word?.title ?? ""}
        data-finished={row.word?.finished ?? ""}
        data-at={rowTime(row) ?? ""}
      >
        <span class="st-cell">{#if row.word?.state}<span class="st {row.word.state}" title={row.word.state}></span>{/if}</span>
        <span class="sid">{row.incarnation}</span>
        <span class="instrument-detail">{rowDetail(row)}</span>
        <span class="tid">{row.tasks[0] ?? ""}</span>
        <span class="num">{rowTime(row) !== null ? taskAge(rowTime(row), now()) : ""}</span>
      </button>
    {:else}
      <p class="dim3 instrument-empty">none</p>
    {/each}
  </div>
{/each}
