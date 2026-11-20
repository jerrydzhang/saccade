<!-- The port window's assembly (slice 1): the strip, the buffer's
scroll surface and the published frame's component. The window owns
the scroller element; the buffer's viewport adopts it. Presentation
reads the mapped buffer state only - the component/props pair comes
from the registry's reviewed erasure boundary. -->

<script lang="ts">
  import { onMount } from "svelte";
  import { createPortWindow, type FrameRailView, type PortWindow, type StripView } from "./App.ts";
  import type { RenderPair } from "./lib/instruments/registry.ts";
  import type { BufferState } from "./lib/contracts/buffer.ts";
  import type { PickerState } from "./lib/input/picker.ts";
  import Picker from "./components/Picker.svelte";
  import WhichKey from "./components/WhichKey.svelte";
  import { pendingKeys, type HelpView, type PendingKeysView } from "./lib/input/help.ts";
  import type { WhichKeyView } from "./lib/input/key-matcher.ts";
  import Strip from "./components/Strip.svelte";
  import FrameRail from "./components/FrameRail.svelte";
  import { railNoteFor } from "./lib/instruments/claims.ts";
  import type { FrameRailProject } from "./App.ts";
  import type { TargetDefinition } from "./lib/contracts/buffer.ts";

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
  let whichKeyView = $state<PendingKeysView | null>(null);
  let helpView = $state<HelpView | null>(null);
  let ask = $state<import("./lib/elsewhere.ts").AskState | null>(null);
  let askInput: HTMLInputElement | null = $state(null);
  let askValue = $state("");
  let inspection = $state<import("./lib/figures/inspection.ts").InspectionState | null>(null);
  let railState = $state<FrameRailView>({
    visible: false,
    footnote: null,
    notes: [],
    add: { open: false, value: "", problem: null },
    pane: null,
  });
  let railProjects = $state<FrameRailProject[]>([]);
  let railBeneath = $state<readonly { project: FrameRailProject; targets: readonly TargetDefinition[] }[]>([]);
  let railFoot = $state<readonly TargetDefinition[]>([]);
  let railNotes = $state<readonly { part: string; note: string }[]>([]);
  let railSelectLabels = $state<readonly { part: string; label: string; nested?: boolean }[]>([]);
  let railSelectTyped = $state("");
  let railMenu = $state<import("./lib/menu/session.ts").MenuView | null>(null);
  let railMenuSession = $state<unknown>(null);
  // the shown project for the rail's mark
  let railShown = $state<string | null>(null);
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
      whichKeyView = pending === null ? null : pendingKeys(pending as WhichKeyView);
      helpView = help ? windowPort.helpView() : null;
    });
    const offAsk = windowPort.subscribeAsk((state) => {
      ask = state;
      if (state !== null) askValue = "";
    });
    const offInspection = windowPort.subscribeInspection((state) => {
      inspection = state;
    });
    const offRailState = windowPort.subscribeRailState((view) => {
      railState = view;
    });
    const offRail = windowPort.subscribeRail((next) => {
      railProjects = next;
      railBeneath = next.map((project) => ({
        project,
        targets: windowPort.railTargets(project.name),
      }));
      railFoot = windowPort.railTargets(null);
      railNotes = next
        .map((project) => ({ part: project.part, note: railNoteFor(project.name) }))
        .filter((entry) => entry.note !== null);
      railShown = next.find((project) => project.current)?.name ?? null;
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
      offRailState();
      offRail();
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

  // The rail's select chips and its act menu ride the presentations
  // the rail's entries ride now (the shown buffer's, or the frame's
  // own in a window that shows none): subscribed here so the pane
  // paints its own subset; a frame state move re-reads them.
  $effect(() => {
    const targets = frame === null ? null : (port?.railLease() ?? null);
    if (targets === null) {
      railSelectLabels = [];
      railMenu = null;
      railMenuSession = null;
      return;
    }
    const read = (): void => {
      const select = targets.selectView();
      railSelectLabels = select?.labels ?? [];
      railSelectTyped = select?.pendingLabel ?? "";
      const menu = targets.menuView();
      railMenu = menu !== null && menu.part.startsWith("rail/") ? menu : null;
      railMenuSession = targets.menuSession();
    };
    read();
    return targets.subscribe(read);
  });

  /** The rail entry's liveness measure, taken in the pane: an entry
   * the pane cannot measure has left, and a stale lease acts nothing. */
  function railMeasure(part: string): import("./lib/contracts/places.ts").Bounds | null {
    const pane = railState.pane;
    if (pane === null) return null;
    const element = pane.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
    if (!(element instanceof HTMLElement)) return null;
    const box = element.getBoundingClientRect();
    if (box.width === 0 && box.height === 0) return null;
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
  }

  /** A rail entry's plain click: through the rail's lease (the entry
   * is an ordinary target - the primary act runs); while a pending
   * open still draws the previous content there is none, and a
   * project's entry opens its task list directly. */
  function railEntryActivate(part: string, event: MouseEvent): void {
    const lease = railLeaseOf();
    if (lease !== null) {
      lease.presentTarget(part, () => railMeasure(part)).activatePointer("plain");
      event.preventDefault();
      return;
    }
    const project = part.startsWith("rail/project/") ? part.slice("rail/project/".length) : null;
    if (project !== null) port?.openSpec(`${project}/tasks`);
  }

  function railEntryMenu(part: string, event: MouseEvent): void {
    const lease = railLeaseOf();
    if (lease === null) return;
    const content = scroller?.getBoundingClientRect();
    lease
      .presentTarget(part, () => railMeasure(part))
      .showActions({ left: event.clientX - (content?.left ?? 0), top: event.clientY - (content?.top ?? 0) });
  }

  function railLeaseOf(): import("./lib/contracts/buffer.ts").TargetPresentations | null {
    return port?.railLease() ?? null;
  }

</script>

<div class="port-shell">
  <Strip {strip} />
  {#if ask !== null}
    <div class="port-float port-picker">
      <input
        class="port-picker-field"
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
    <WhichKey view={whichKeyView} />
  {/if}
  {#if helpView !== null}
    <WhichKey view={helpView} />
  {/if}
  {#if picker.open}
    <Picker
      context={pickerContext}
      submit={(spec, elsewhere) => port?.submitPicker(spec, elsewhere)}
      close={() => port?.closePicker()}
    />
  {/if}
  <div class="port-main">
  <main class="port-content" data-port-control="content" bind:this={scroller}>
    <div class="port-frame-layout" class:port-frame-layout-with-rail={railState.visible}>
      {#if railState.visible}
        <FrameRail
          projects={railProjects}
          beneath={railBeneath}
          foot={railFoot}
          notes={railNotes}
          view={railState}
          shown={railShown}
          labels={railSelectLabels}
          typed={railSelectTyped}
          menu={railMenu}
          menuSession={railMenuSession}
          typeAdd={(value) => port?.typeAdd(value)}
          submitAdd={() => port?.submitAdd()}
          closeAdd={() => port?.closeAdd()}
          entryActivate={railEntryActivate}
          entryMenu={railEntryMenu}
          attachPane={(element) => port?.attachRailPane(element)}
        />
      {/if}
      <div class="port-frame-body">
        {#if frame !== null && frame.kind === "alive"}
          {#if frame.content.kind === "empty"}
            <!-- the empty window: the frame's surface, blank; no instrument
            involved -->
            <div class="port-empty" data-port-control="empty-surface" role="document"></div>
          {:else if frame.content.kind === "available"}
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
      </div>
    </div>
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
