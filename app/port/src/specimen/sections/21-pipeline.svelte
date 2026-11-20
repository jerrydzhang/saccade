<script lang="ts">
  // The project's task list, through the real component: every stage (the
  // waiting one empty), each row's agents marked by their live words
  // joined from the rail, one row pointed.
  import TasksPipeline from "../../instruments/TasksPipeline.svelte";
  import { wordsByIncarnationOf } from "../../lib/tasks/rail.ts";
  import { tasksPage, tasksRail } from "../tasks-fixture.ts";

  const words = wordsByIncarnationOf(tasksRail);
  const pointed = tasksPage.groups[1]?.rows[1]?.part ?? null;
</script>

<div class="port-tasks-view">
  <TasksPipeline page={tasksPage} {words} now={Date.now()} hovered={pointed} />
</div>
