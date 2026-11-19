<script lang="ts">
  // The frame's rail, through the real component: every word in
  // importance order, a named agent with several tasks, an agent with
  // no word, unregistered agents under the project and at the foot,
  // and the add field open. Select labels and the act menu are the
  // shown buffer's; the specimen hosts the pane bare, so none appear.
  import FrameRail from "../../components/FrameRail.svelte";
  import { tasksRail } from "../tasks-fixture.ts";

  const noop = () => {};
</script>

<div class="rail">
  <FrameRail
    rail={tasksRail}
    view={{ visible: true, add: { open: true, value: "", problem: null }, pane: null }}
    shown="pyrimel"
    labels={[]}
    menu={null}
    menuSession={null}
    typeAdd={noop}
    submitAdd={noop}
    closeAdd={noop}
    entryActivate={noop}
    entryMenu={noop}
    attachPane={noop}
  />
</div>

<style>
  .rail {
    width: 16rem;
  }
</style>
