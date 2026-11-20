<script lang="ts">
  // The whole task list window, as the frame shows it: the Tasks instrument's
  // own component over a fixture-backed controller.
  import TasksView from "../../instruments/TasksView.svelte";
  import { fixtureTasksUI, tasksContent } from "../tasks-fixture.ts";

  const controller = fixtureTasksUI();
</script>

<TasksView content={tasksContent} {controller} />
