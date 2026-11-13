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
  import WhichKey from "./components/WhichKey.svelte";

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
  let whichKeyView = $state<unknown>(null);
  let helpView = $state<unknown>(null);
  let ask = $state<import("./lib/elsewhere.ts").AskState | null>(null);
  let askInput: HTMLInputElement | null = $state(null);
  let askValue = $state("");
  let inspection = $state<import("./lib/figures/inspection.ts").InspectionState | null>(null);
  // the viewer's drag (the prototype's lb.onmousedown origin) and the
  // stage's layout size (a transform never changes offsetWidth)
  let drag: { sx: number; sy: number } | null = null;
  let stageWidth = $state(0);
  let stageHeight = $state(0);

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
    const offWhichKey = windowPort.subscribeWhichKey((pending, help) => {
      whichKeyView = pending;
      helpView = help ? windowPort.helpView() : null;
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
      offWhichKey();
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
    <span class="port-strip-name">saccade</span>
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
  {#if whichKeyView !== null}
    <WhichKey view={whichKeyView} title="keys" />
  {/if}
  {#if helpView !== null}
    <WhichKey view={helpView} title="all keys" />
  {/if}
  {#if picker.open}
    <Picker
      context={pickerContext}
      submit={(spec) => port?.submitPicker(spec)}
      close={() => port?.closePicker()}
    />
  {/if}
  <div class="port-main">
  <main class="port-content" data-port-control="content" bind:this={scroller}>
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
  {#if inspection !== null}
    <!-- the figure viewer: the prototype's lightbox (inst16.html:241,
    393-401) - over the content area, the stage centred and transformed
    by translate(x - w*z/2, y - h*z/2) scale(z); the wheel zooms, a drag
    pans, the keys ride the inspection context (+ - = Escape) -->
    <div
      class="port-inspection"
      data-port-control="inspection"
      role="presentation"
      onwheel={(event) => {
        event.preventDefault();
        port?.inspectionWheel(event.deltaY);
      }}
      onmousedown={(event) => {
        if (inspection === null) return;
        drag = { sx: event.clientX - inspection.x, sy: event.clientY - inspection.y };
      }}
      onmousemove={(event) => {
        if (drag !== null) port?.inspectionPan(event.clientX - drag.sx, event.clientY - drag.sy);
      }}
      onmouseup={() => {
        drag = null;
      }}
    >
      <div
        class="port-inspection-stage"
        data-port-control="inspection-stage"
        bind:offsetWidth={stageWidth}
        bind:offsetHeight={stageHeight}
        style="transform: translate({inspection.x - (stageWidth * inspection.zoom) / 2}px, {inspection.y - (stageHeight * inspection.zoom) / 2}px) scale({inspection.zoom})"
      >
        {#if inspection.svg !== undefined}
          {@html inspection.svg}
        {:else if inspection.text !== undefined}
          <pre class="port-inspection-text">{inspection.text}</pre>
        {:else}
          <img src={inspection.src} alt={inspection.caption || inspection.part} draggable="false" />
        {/if}
      </div>
      <div class="port-inspection-hint" data-port-control="inspection-hint">{"wheel or + - zoom \u00b7 = resets \u00b7 drag pans \u00b7 Esc closes"}</div>
    </div>
  {/if}
  </div>
</div>
