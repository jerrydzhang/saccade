<script lang="ts">
  // The whole plane, as the window shows it: the Tasks instrument's
  // own component over a fixture-backed controller.
  import TasksView from "../../instruments/TasksView.svelte";
  import { fixtureTasksUI, planeContent } from "../plane-fixture.ts";

  const controller = fixtureTasksUI();
</script>

<TasksView content={planeContent} {controller} />
