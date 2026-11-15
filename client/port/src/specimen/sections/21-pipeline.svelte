<script lang="ts">
  // The plane's pipeline, through the real component: every stage (the
  // waiting one empty), each row's agents marked by their live words
  // joined from the rail, one row pointed.
  import TasksPipeline from "../../instruments/TasksPipeline.svelte";
  import { wordsByIncarnationOf } from "../../lib/tasks/rail.ts";
  import { planePage, planeRail } from "../plane-fixture.ts";

  const words = wordsByIncarnationOf(planeRail);
  const pointed = planePage.groups[1]?.rows[1]?.part ?? null;
</script>

<div class="port-tasks-view">
  <TasksPipeline page={planePage} {words} now={Date.now()} hovered={pointed} />
</div>
