<!-- The plane's pipeline: the tasks by stage. A stage's title takes
its state's hue; its rows stay neutral and carry no column header (the
columns explain themselves): id, title, agents (each marked with its
live Herdr word's shape), the latest commit's short hash, age.
Presentation only - the clicks and the menu are TasksView's
delegation. -->

<script lang="ts">
  import { ageOf, rowAgentsOf, shortHash, stageStateOf, type TasksPageModel } from "../lib/tasks/page.ts";

  let {
    page,
    words,
    now,
    hovered = null,
  }: {
    page: TasksPageModel;
    /** The live word of each agent the rail shows, by incarnation. */
    words: ReadonlyMap<string, string>;
    now: number;
    hovered?: string | null;
  } = $props();
</script>

{#each page.groups as group (group.part)}
  <section class="port-tasks-group" data-port-part={group.part} data-port-kind="group">
    {#if group.title !== ""}
      <h2 class="port-tasks-heading" data-port-state={stageStateOf(group.title)}>{group.title}</h2>
    {/if}
    {#if group.rows.length === 0}
      {#if group.title !== ""}
        <div class="port-tasks-none">none</div>
      {/if}
    {:else}
      {#each group.rows as row (row.part)}
        <div
          class="port-tasks-row"
          data-port-part={row.part}
          data-port-kind="task"
          data-port-task={row.task}
          data-port-open={row.location}
          data-port-title={row.title}
          data-port-hover={hovered === row.part ? "" : undefined}
        >
          <span class="port-tasks-id">{row.task}</span>
          <span class="port-tasks-title">{row.title}</span>
          <span class="port-tasks-agents">
            {#each rowAgentsOf(row) as agent (agent)}
              <span class="port-tasks-agent">
                {#if words.has(agent)}<span class="port-state-mark" data-port-word={words.get(agent)} aria-hidden="true"></span>{/if}{agent}
              </span>
            {/each}
          </span>
          <span class="port-tasks-commit">{shortHash(row.commit)}</span>
          <span class="port-tasks-age">{ageOf(row.at, now)}</span>
        </div>
      {/each}
    {/if}
  </section>
{/each}
