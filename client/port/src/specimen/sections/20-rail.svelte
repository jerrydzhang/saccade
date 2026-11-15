<script lang="ts">
  // The plane's rail, through the real component: every word in
  // importance order, a named agent with several tasks, an agent with
  // no word, unregistered agents under the project and at the foot,
  // and the add field open.
  import PlaneRail from "../../instruments/PlaneRail.svelte";
  import { fixtureTasksUI, planeRail } from "../plane-fixture.ts";

  const controller = fixtureTasksUI({ open: true, value: "", problem: null });
</script>

<div class="rail">
  <PlaneRail rail={planeRail} {controller} hovered="rail/agent/pyrimel/i-9" />
</div>

<style>
  .rail {
    width: 16rem;
  }
</style>
