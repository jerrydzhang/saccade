<script lang="ts">
  // The tasks window (round 16's pipeline): one section per state, one
  // ledger row per task - id, title, agent shapes, latest commit, age.
  import { taskAge, type TasksSection } from "../lib/instruments/tasks.ts";

  type Props = {
    sections: TasksSection[];
    /** Agent ids to their live state word, for the shapes. */
    shapes: (agents: string) => Array<{ id: string; word: string | null }>;
    onSelect: (click: MouseEvent) => void;
  };

  let { sections, shapes, onSelect }: Props = $props();
  const now = (): number => Math.floor(Date.now() / 1000);
</script>

{#each sections as section (section.title)}
  <p class="sec instrument-label">{section.title}</p>
  <div class="instrument-table" role="table" aria-label="{section.title} tasks" data-control="instrument:section">
    {#each section.rows as row (row.task)}
      <button
        type="button"
        class="row instrument-row"
        role="row"
        data-control="instrument:row"
        data-block-id={`instrument-row-${row.task}`}
        data-instrument="tasks"
        data-task={row.task}
        data-open-path={row.location}
        data-agents={row.agents}
        onclick={onSelect}
      >
        <span class="tid">{row.task}</span>
        <span class="instrument-title">{row.title}</span>
        <span class="instrument-agents">
          {#each shapes(row.agents) as shape (shape.id)}
            <span class="agent-cell">
              {#if shape.word}<span class="st {shape.word}" title={shape.word}></span>{/if}
              <span class="sid">{shape.id}</span>
            </span>
          {/each}
        </span>
        <span class="commit-cell" title={row.latestCommit}>{#if row.latestRevision}<span class="r">{row.latestRevision}</span> {/if}{row.latestCommit.slice(0, 7)}</span>
        <span class="num">{taskAge(row.at, now())}</span>
      </button>
    {:else}
      <p class="dim3 instrument-empty">none</p>
    {/each}
  </div>
{/each}
