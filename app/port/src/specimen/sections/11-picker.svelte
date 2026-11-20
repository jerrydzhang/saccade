<script lang="ts">
  // The picker: the real Picker over a fixture tracker's tasks, inside
  // a stage that holds its fixed position (a transformed ancestor).
  import Picker from "../../components/Picker.svelte";

  const context = {
    recent: [],
    tasks: [
      { id: "t-3", title: "rescore earlier search candidates with the tube score" },
      { id: "t-30", title: "the suite as built: an explainer" },
      { id: "t-2", title: "numerically check the hand-estimated atlas walk" },
      { id: "t-0", title: "derive a working theory from scratch" },
    ],
    specOf: (id: string) => `127.0.0.1:8811/pyrimel/${id}`,
    openHere: "127.0.0.1:8811/pyrimel/t-2",
  };
</script>

<div class="stage">
  <Picker {context} submit={() => {}} close={() => {}} />
</div>

<style>
  .stage {
    position: relative;
    height: 22rem;
    transform: translateZ(0);
  }
</style>
