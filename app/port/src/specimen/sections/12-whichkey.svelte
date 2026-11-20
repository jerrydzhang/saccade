<script lang="ts">
  // Which-key: the real popup over the window's own table, with space
  // pending - the projection the app paints (lib/input/help.ts).
  import WhichKey from "../../components/WhichKey.svelte";
  import { windowContext } from "../../lib/input/contexts.ts";
  import { pendingKeys } from "../../lib/input/help.ts";
  import { createMatcher } from "../../lib/input/key-matcher.ts";

  const matcher = createMatcher(
    windowContext({
      openPicker: () => {}, closeOverlays: () => {}, pickerIsOpen: () => false,
      trailBack: () => true, trailForward: () => true, toggleHistory: () => {},
      linkByLabels: () => {}, unlink: () => {}, showHelp: () => {},
    }),
  );
  matcher.resolve({ key: " ", ctrl: false, shift: false, alt: false, meta: false });
  const view = pendingKeys(matcher.pendingView()!);
</script>

<div class="stage">
  <WhichKey {view} />
</div>

<style>
  .stage {
    position: relative;
    height: 16rem;
    transform: translateZ(0);
  }
</style>
