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

  let port: PortWindow | null = null;
  let scroller: HTMLElement | null = $state(null);
  let frame: BufferState<RenderPair> | null = $state(null);
  let strip: StripView | null = $state(null);
  let picker = $state<PickerState>({ open: false });
  let pickerInput: HTMLInputElement | null = $state(null);
  let pickerValue = $state("");
  let findView: { open: boolean; query: string; count: number; active: number } | null = $state(null);
  let findInput: HTMLInputElement | null = $state(null);
  let findValue = $state("");

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
      if (state.open) pickerValue = "";
    });
    const offFind = windowPort.subscribeFind((state) => {
      findView = state;
      if (state.open && document.activeElement !== findInput) findValue = state.query;
    });
    void windowPort.boot();
    return () => {
      offFrame();
      offStrip();
      offPicker();
      offFind();
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
    if (picker.open) pickerInput?.focus();
  });

  // Find opens focused the same way; movement scrolls the part into
  // view through the presentation's own DOM (presentation owns DOM
  // adaptation; the session owns the semantics).
  $effect(() => {
    if (findView?.open) findInput?.focus();
  });

  function revealPart(part: string): void {
    scroller?.querySelector(`[data-port-part=${JSON.stringify(part)}]`)?.scrollIntoView({ block: "center" });
  }
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
  </header>
  {#if picker.open}
    <div class="port-picker">
      <input
        bind:this={pickerInput}
        bind:value={pickerValue}
        data-port-control="picker-input"
        placeholder="open: host:8811/repo/t-N, /absolute/path, machine:/absolute/path"
        onkeydown={(event) => {
          if (event.key === "Enter") {
            port?.submitPicker(pickerValue);
          } else if (event.key === "Escape") {
            port?.closePicker();
          }
        }}
      />
    </div>
  {/if}
  {#if findView?.open}
    <div class="port-find">
      <input
        bind:this={findInput}
        bind:value={findValue}
        data-port-control="find-input"
        placeholder="find in this revision"
        oninput={() => {
          port?.findType(findValue);
        }}
        onkeydown={(event) => {
          if (event.key === "Enter") {
            const moved = port?.findMove(event.shiftKey ? -1 : 1);
            if (moved?.part != null) revealPart(moved.part);
          } else if (event.key === "Escape") {
            port?.findEscape();
          }
        }}
      />
      {#if findView !== null}
        <span class="port-find-status" data-port-find-status="">
          {findView.count > 0 ? `${findView.active + 1} / ${findView.count}` : "type to search"}
        </span>
      {/if}
    </div>
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
