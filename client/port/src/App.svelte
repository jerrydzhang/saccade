<!-- The port window's assembly (slice 1): the strip, the buffer's
scroll surface and the published frame's component. The window owns
the scroller element; the buffer's viewport adopts it. Presentation
reads the mapped buffer state only - the component/props pair comes
from the registry's reviewed erasure boundary. -->

<script lang="ts">
  import { onMount } from "svelte";
  import { createPortWindow, type PortWindow, type StripView } from "./App.ts";
  import type { RenderPair } from "./lib/instruments/registry.ts";
  import type { BufferState } from "./lib/contracts/buffer.ts";
  import type { PickerState } from "./lib/input/picker.ts";
  import Picker from "./components/Picker.svelte";

  let port: PortWindow | null = null;
  let scroller: HTMLElement | null = $state(null);
  let frame: BufferState<RenderPair> | null = $state(null);
  let strip: StripView | null = $state(null);
  let picker = $state<PickerState>({ open: false });
  let pickerContext = $state({
    recent: [] as string[],
    tasks: [] as { id: string; title: string }[],
    specOf: (id: string) => id,
    openHere: null as string | null,
  });
  let ask = $state<import("./lib/elsewhere.ts").AskState | null>(null);
  let askInput: HTMLInputElement | null = $state(null);
  let askValue = $state("");
  let inspection = $state<import("./lib/figures/inspection.ts").InspectionState | null>(null);

  onMount(() => {
    const windowPort = createPortWindow();
    port = windowPort;
    const offFrame = windowPort.subscribeFrame((state) => {
      frame = state;
    });
    const offStrip = windowPort.subscribeStrip((view) => {
      strip = view;
    });
    const offPicker = windowPort.subscribePicker((state) => {
      picker = state;
    });
    const offPickerContext = windowPort.subscribePickerContext((context) => {
      pickerContext = context;
    });
    const offAsk = windowPort.subscribeAsk((state) => {
      ask = state;
      if (state !== null) askValue = "";
    });
    const offInspection = windowPort.subscribeInspection((state) => {
      inspection = state;
    });
    void windowPort.boot();
    return () => {
      offFrame();
      offStrip();
      offPicker();
      offPickerContext();
      offAsk();
      offInspection();
      windowPort.destroy();
      port = null;
    };
  });

  $effect(() => {
    if (port !== null) port.attachViewport(scroller);
  });

  // The picker opens focused: typing edits the field natively (the
  // window key seam skips field targets); Enter submits through the
  // one open funnel, Escape closes. Text entry never falls through to
  // the board keymap.
  $effect(() => {

  });

  // The ask opens focused the same way: typing answers a window chip,
  // Enter delivers, Escape cancels (the chips ride the strip).
  $effect(() => {
    if (ask !== null) askInput?.focus();
  });

</script>

<div class="port-shell">
  <header class="port-strip">
    <span class="port-strip-name">saccade (port)</span>
    <span class="port-strip-location" data-port-strip="location">
      {strip?.location ?? "no board"}
    </span>
    <span class="port-strip-status" data-port-strip="status">{strip?.status ?? "empty"}</span>
    {#if strip?.note}
      <span class="port-strip-note">{strip.note}</span>
    {/if}
    {#if strip?.jumpChip}
      <span class="port-strip-chip">{strip.jumpChip}</span>
    {/if}
    <span class="port-strip-link" data-port-strip="link">
      {strip?.link != null ? `-> ${strip.link}` : "unlinked"}
    </span>
  </header>
  {#if inspection !== null}
    <div class="port-inspection" data-port-control="inspection">
      <figure class="port-inspection-figure">
        <img src={inspection.src} alt={inspection.caption || inspection.part} style="transform: scale({inspection.zoom})" />
        {#if inspection.caption}
          <figcaption>{inspection.caption}</figcaption>
        {/if}
      </figure>
    </div>
  {/if}
  {#if ask !== null}
    <div class="port-picker">
      <input
        bind:this={askInput}
        bind:value={askValue}
        data-port-control="ask-input"
        placeholder="deliver to window chip: {ask.chips.map((entry) => entry.chip).join(", ")}"
        onkeydown={(event) => {
          if (event.key === "Enter") {
            port?.answerAsk(askValue);
          } else if (event.key === "Escape") {
            port?.cancelAsk();
          }
        }}
      />
    </div>
  {/if}
  {#if picker.open}
    <Picker
      context={pickerContext}
      submit={(spec) => port?.submitPicker(spec)}
      close={() => port?.closePicker()}
    />
  {/if}
  <main class="port-content" bind:this={scroller}>
    {#if frame !== null && frame.kind === "alive"}
      {#if frame.content.kind === "available"}
        {@const pair = frame.content.value}
        {@const Component = pair.Component}
        <Component {...pair.props} />
      {:else if frame.content.kind === "pending"}
        <div class="port-placeholder">
          {#if frame.content.previous !== null}
            {@const previous = frame.content.previous}
            {@const Previous = previous.Component}
            <Previous {...previous.props} />
          {:else}
            <p class="port-state-line">loading...</p>
          {/if}
        </div>
      {:else if frame.content.kind === "unavailable"}
        <p class="port-state-line">the board did not load; the diagnostic carries the reason</p>
      {/if}
    {/if}
  </main>
</div>
