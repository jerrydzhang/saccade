// The port window's assembly: the composition skeleton of slice 1.
// The window owns its one buffer through the ratified glue -
// registration, construction, activation - and the suite doors the
// shared Rust shell evals into every window. A citation activation
// opens through the port's own model, so the receiving window owns the
// act; the shell already guaranteed focus.
//
// The window owns its picker, explicit-open trail, shell doors and
// routing seam. Reading interactions (Find, select, target menus)
// belong to the instrument. Every opener uses the same openSpec path;
// the observation module projects the read-only inspection contract.
//
// Log markers ("[viewer] startup", "[viewer] test open board FAILED")
// are the LANE's boot-evidence gates, parsed from the app log by the
// native lane; they are lane material, not the observation contract.
// The observation contract is __saccadePort.readout().

import {
  constructInstrument,
  registerInstrument,
  type RenderPair,
  type RunningInstrument,
  type WindowEnvironment,
} from "./lib/instruments/registry.ts";
import {
  createBufferFactory,
  createDiagnostics,
  createViewport,
  type SliceBufferViewport,
  type SliceDiagnostics,
} from "./components/Buffer.ts";
import { boardInstrument, parseOpenSpec } from "./instruments/board.ts";
import type { BoardUI } from "./instruments/board.ts";
import {
  PORT_SURFACE,
  buildReadout,
  nativeWindowLabel,
  type DoorDelivery,
  type PortObservation,
} from "./lib/observation.ts";
import { tauriInvoke } from "./lib/sources.ts";
import { createTrail, type TrailView } from "./lib/navigation/trail.ts";
import { createHostMatcher, dispatchStroke, type HostMatcher } from "./lib/input/host.ts";
import { isPinnedWindowChord, windowContext } from "./lib/input/contexts.ts";
import { createPickerController, type PickerState } from "./lib/input/picker.ts";
import {
  assignChips,
  elsewhereDecision,
  type AskState,
  type ElsewhereAct,
  type SuiteWindow,
} from "./lib/elsewhere.ts";
import { WindowStore } from "./lib/windows.ts";
import { serverBase } from "./lib/tracker.ts";
import { SourceUnavailableError } from "./lib/board/errors.ts";
import { knownIssue } from "./lib/contracts/buffer.ts";
import type { BufferState } from "./lib/contracts/buffer.ts";

export interface StripView {
  location: string | null;
  status: string;
  jumpChip: string | null;
  note: string | null;
}

export interface PortWindow {
  attachViewport(element: HTMLElement | null): void;
  openSpec(spec: string, part?: string | null): void;
  boot(): Promise<void>;
  destroy(): void;
  subscribeFrame(observer: (state: BufferState<RenderPair>) => void): () => void;
  subscribeStrip(observer: (strip: StripView) => void): () => void;
  trailBack(): boolean;
  trailForward(): boolean;
  trailView(): TrailView;
  openPicker(): void;
  closePicker(): void;
  submitPicker(spec: string): void;
  subscribePicker(observer: (state: PickerState) => void): () => void;
  answerAsk(chip: string): void;
  cancelAsk(): void;
  subscribeAsk(observer: (state: AskState | null) => void): () => void;
  readout(): PortObservation;
}

export function createPortWindow(windowId = "main"): PortWindow {
  const diagnostics: SliceDiagnostics = createDiagnostics();
  const buffers = createBufferFactory(windowId);
  const viewport: SliceBufferViewport = createViewport();

  const registration = registerInstrument(boardInstrument());
  const context = { bufferId: `${windowId}-board-1`, windowId };
  const environment: WindowEnvironment = {
    context,
    viewport,
    diagnostics,
    buffers,
  };
  const instrument: RunningInstrument = constructInstrument(registration, environment);
  // The window's one buffer starts active; focus switching is a later
  // slice's surface.
  instrument.activate();

  // The open-location picker: the window's open surface, submitting
  // through the ONE open funnel below (openSpec). Text-entry
  // semantics: the key seam skips field targets, so typing inside the
  // picker stays native (scenario 10).
  const picker = createPickerController((spec) => openSpec(spec));

  // -- the elsewhere surface (S18): the window's o/i/w OPEN acts -----
  // The window binds a window-store state on the home tracker (the
  // state carries this window's link - the o act's remembered
  // receiver); binding happens when a tracker is first reached (the
  // store lives there, and at boot none is known yet).
  let windowStore: WindowStore | null = null;
  let windowStateId: string | null = null;
  let linkLabel: string | null = null;
  let ask: AskState | null = null;
  let elsewhereNote: string | null = null;
  const askObservers = new Set<(state: AskState | null) => void>();

  function notifyAsk(): void {
    for (const observer of askObservers) observer(ask);
  }

  async function suiteWindows(): Promise<SuiteWindow[]> {
    const invoke = tauriInvoke();
    return (await invoke("suite_windows")) as SuiteWindow[];
  }

  async function ensureWindowState(): Promise<string | null> {
    if (windowStore !== null) return null;
    let restored: string | null = null;
    try {
      const base = serverBase();
      const invoke = tauriInvoke();
      const binding = (await invoke("window_binding")) as {
        label: string;
        // the shell skips the field when no state is bound (serde
        // skip_serializing_if): absent means unbound, not null
        state_id?: string | null;
      };
      windowStore = new WindowStore({ base });
      if (typeof binding.state_id === "string" && binding.state_id.length > 0) {
        windowStateId = binding.state_id;
        await windowStore.attach(windowStateId);
      } else {
        const fresh =
          typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        await windowStore.create(fresh, {});
        await windowStore.attach(fresh);
        await invoke("bind_window_state", {
          stateId: fresh,
          server: base.toString(),
          device: windowStore.device,
        });
        windowStateId = fresh;
      }
      const state = await windowStore.getState(windowStateId);
      restored = state.location;
      await refreshLink();
      injectReferenceElsewhere();
      notifyStrip();
    } catch (error) {
      // no bridge, or the store refused: the elsewhere acts stay
      // unoffered and the note records why (the honest refusal, never
      // a fake surface)
      elsewhereNote = `elsewhere unavailable: ${String(error instanceof Error ? error.message : error)}`;
      notifyStrip();
    }
    return restored;
  }

  async function refreshLink(): Promise<void> {
    linkLabel = null;
    if (windowStore === null || windowStateId === null) return;
    try {
      const state = await windowStore.getState(windowStateId);
      const linked = state.link;
      if (linked === null || linked.length === 0) return;
      const windows = await suiteWindows();
      linkLabel = windows.find((entry) => entry.state_id === linked)?.label ?? null;
    } catch {
      // a dead store read: the link resolves null until the next
      // refresh; o asks rather than delivering into a guess
    }
  }

  async function openInNewWindow(spec: string, becomeLink: boolean): Promise<void> {
    if (windowStore === null || windowStateId === null) return;
    const invoke = tauriInvoke();
    const fresh =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    // the delivery rides the state's birth location: the receiver
    // restores it at boot through its own model (no eval into a
    // half-loaded webview - the race the old app shipped as its
    // linked-delivery hole)
    await windowStore.create(fresh, { location: spec });
    const label = (await invoke("open_suite_window", { stateId: fresh })) as string;
    if (becomeLink) {
      // the store refuses links between windows not both attached on
      // this device, and the receiver attaches when ITS boot binds -
      // bounded wait for the attach, then the link (never an unbounded
      // or dishonest success)
      await setLinkWhenAttached(windowStateId, fresh);
      await refreshLink();
      notifyStrip();
    }
  }

  async function setLinkWhenAttached(from: string, to: string, tries = 24): Promise<void> {
    if (windowStore === null) return;
    for (let attempt = 0; attempt < tries; attempt++) {
      try {
        await windowStore.setLink(from, to);
        return;
      } catch {
        const state = await windowStore.getState(to).catch(() => null);
        if (state !== null && state.attachedHere) {
          // attached on this device and still refused: not a race -
          // the honest give-up names it
          throw new Error(`the link could not be set even after ${to} attached`);
        }
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }
    throw new Error(`the receiver ${to} never attached on this device`);
  }

  async function openElsewhere(spec: string, act: ElsewhereAct): Promise<void> {
    if (windowStore === null || windowStateId === null) {
      await ensureWindowState();
      if (windowStore === null || windowStateId === null) {
        note = elsewhereNote ?? "the window store is not serving this window";
        notifyStrip();
        return;
      }
    }
    const invoke = tauriInvoke();
    const windows = await suiteWindows();
    const decision = elsewhereDecision(act, { linkLabel, windows });
    if (decision.kind === "deliver") {
      await invoke("open_in_window", { label: decision.label, location: spec });
      return;
    }
    if (decision.kind === "new-window") {
      await openInNewWindow(spec, decision.becomeLink);
      return;
    }
    // the ask: chips on every window, the answer typed here
    await invoke("broadcast_jump_labels", {
      labels: assignChips(windows.map((entry) => entry.label)).map((entry) => ({
        window: entry.window,
        chip: entry.chip,
      })),
    });
    ask = { act, spec, chips: assignChips(windows.map((entry) => entry.label)) };
    notifyAsk();
    notifyStrip();
  }

  async function answerAsk(chip: string): Promise<void> {
    const asking = ask;
    if (asking === null) return;
    ask = null;
    const invoke = tauriInvoke();
    await invoke("broadcast_jump_labels_clear");
    const match = asking.chips.find((entry) => entry.chip === chip.trim());
    if (match === undefined) {
      notifyAsk();
      notifyStrip();
      return;
    }
    await invoke("open_in_window", { label: match.window, location: asking.spec });
    if (asking.act === "link" && windowStore !== null && windowStateId !== null) {
      const windows = await suiteWindows();
      const target = windows.find((entry) => entry.label === match.window);
      if (target?.state_id !== undefined && target.state_id !== null) {
        await windowStore.setLink(windowStateId, target.state_id);
        await refreshLink();
      }
    }
    notifyAsk();
    notifyStrip();
  }

  async function cancelAsk(): Promise<void> {
    if (ask === null) return;
    ask = null;
    try {
      const invoke = tauriInvoke();
      await invoke("broadcast_jump_labels_clear");
    } catch {
      // the broadcast already cleared or the bridge left; the ask is
      // closed either way
    }
    notifyAsk();
    notifyStrip();
  }

  /** The reference acts grow the OPEN acts once the window state is
   * bound (the ports object is injected once at build; this re-taps
   * the controller's ports with the elsewhere member present). */
  function injectReferenceElsewhere(): void {
    const ui = instrument.presentation as BoardUI;
    ui.setReferencePorts({
      open: (spec, part) => openSpec(spec, part),
      copy: (payload) => {
        lastCopy = { payload, at: Date.now() };
        try {
          void navigator.clipboard.writeText(payload);
        } catch {
          // the lane's webview may refuse the clipboard API; the
          // recorded payload is the act's truth either way
        }
        notifyStrip();
      },
      elsewhere: (spec, act) => {
        void openElsewhere(spec, act);
      },
    });
  }

  // Window commands own the picker. Reading interactions and their
  // captured field commands belong to the active instrument.
  const windowHost: HostMatcher = createHostMatcher(
    windowContext({
      openPicker: () => picker.open(),
      closeOverlays: () => picker.close(),
      pickerIsOpen: () => picker.snapshot().open,
      trailBack: () => trail.back(),
      trailForward: () => trail.forward(),
    }),
  );

  // -- observation state ------------------------------------------------
  let doors: { jumpChip: string | null; delivered: DoorDelivery[] } = {
    jumpChip: null,
    delivered: [],
  };
  let note: string | null = null;

  const frameObservers = new Set<(state: BufferState<RenderPair>) => void>();
  const stripObservers = new Set<(strip: StripView) => void>();
  let lastFrame: BufferState<RenderPair> = instrument.snapshot();

  // The window's trail (scenario 14): explicit openings only; the
  // departure capture lands on the entry being left; Back/Forward
  // restore recorded places - never folds, zoom or drafts. Key
  // bindings for Back/Forward are pinned when the S14 body lands
  // (prototype parity through the adapter); the surface is real now.
  const trail = createTrail({
    capture: () => instrument.places.capture(),
    restore: (place) => instrument.places.restore(place),
  });

  const unsubscribeFrame = instrument.subscribe((state) => {
    lastFrame = state;
    for (const observer of [...frameObservers]) observer(state);
    notifyStrip();
  });

  function stripView(): StripView {
    const state = lastFrame;
    let location: string | null = null;
    let status = "empty";
    if (state.kind === "alive") {
      const content = state.content;
      if (content.kind === "available") {
        const frame = content.value.props["content"] as
          { location?: string; status?: string } | undefined;
        location = frame && typeof frame.location === "string" ? frame.location : null;
        status = frame && typeof frame.status === "string" ? frame.status : "loaded";
      } else if (content.kind === "pending") {
        status = "loading";
      } else if (content.kind === "unavailable") {
        status = "unavailable";
      } else {
        status = "empty";
      }
    } else {
      status = "destroyed";
    }
    return { location, status, jumpChip: doors.jumpChip, note };
  }

  function notifyStrip(): void {
    const view = stripView();
    for (const observer of [...stripObservers]) observer(view);
  }

  function recordDelivery(spec: string, opened: boolean, detail: string | null): void {
    doors = { ...doors, delivered: [...doors.delivered, { spec, opened, detail }] };
    if (doors.delivered.length > 50) doors.delivered = doors.delivered.slice(-50);
  }

  // -- the suite doors (the shared shell's eval messenger writes these) --
  const shellWindow = window as unknown as Record<string, unknown>;
  shellWindow.__suiteJumpShow = (chip: string) => {
    doors = { ...doors, jumpChip: chip };
    notifyStrip();
  };
  shellWindow.__suiteJumpHide = () => {
    doors = { ...doors, jumpChip: null };
    notifyStrip();
  };
  shellWindow.__suiteOpenAt = (spec: string) => {
    // The receiving window owns the open: the delivered spec opens
    // through this window's own model, exactly like a citation
    // followed here. Focus never moves; the shell already set it.
    recordDelivery(spec, true, null);
    openSpec(spec);
  };

  // -- opens: every surface funnels through one path ---------------------
  function openSpec(spec: string, part?: string | null): void {
    const address = parseOpenSpec(spec);
    if (address === null) {
      note = `not a location: ${JSON.stringify(spec)}`;
      diagnostics.report(context, knownIssue({ code: "asset-denied", message: note }));
      notifyStrip();
      return;
    }
    note = null;
    // every tracker reach funnels through here, and the parse above is
    // what registers the tracker the store lives on: the window state
    // (and with it the elsewhere acts) binds on the first reach (S18)
    void ensureWindowState();
    // One operation schedules the content request; the refresh
    // pipeline publishes. A citation naming a part anchors the open
    // there (fold reveal rides the placement, scenario 8). The open
    // is an EXPLICIT opening: the trail records it.
    trail.opened(spec);
    const place = {
      bufferId: instrument.id,
      address,
      position:
        typeof part === "string" && part.length > 0
          ? {
              kind: "anchor" as const,
              anchor: { kind: "board-part" as const, path: part },
              viewportOffset: { left: 0, top: 0 },
            }
          : { kind: "default" as const },
    };
    instrument.places.restore(place);
    notifyStrip();
  }

  // -- reference ports: the citation acts ride the window's one open
  // funnel and the copy observation (scenario 9/13). The controller
  // offers no reference act until these are injected; the follow
  // composes its spec itself (the controller holds the resolved
  // machine - formatLocation, never a colon join) and the open lands
  // anchored exactly as a typed location would.
  let lastCopy: { payload: string; at: number } | null = null;
  (instrument.presentation as BoardUI).setReferencePorts({
    open: (spec, part) => openSpec(spec, part),
    copy: (payload) => {
      lastCopy = { payload, at: Date.now() };
      // the real attempt, failure swallowed (the prototype's copy():
      // try clipboard, toast what copied); the readout carries the
      // payload the ACTION produced - the observation surface
      try {
        void navigator.clipboard.writeText(payload);
      } catch {
        // the lane's webview may refuse the clipboard API; the
        // recorded payload is the act's truth either way
      }
      notifyStrip();
    },
  });

  // -- the key routing seam: window context, then the active buffer ---
  window.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) return;
    const target = event.target as HTMLElement | null;
    const field = target !== null &&
      (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
    // Declared captured fields use their active buffer's text-entry
    // context. Other fields (including the window picker) stay native.
    if (field && !target?.closest("[data-port-captured-input]")) return;
    // Modifier chords stay native until the IME/native-editing
    // boundary set is pinned (draft-open, grouped decision): the port
    // never eats an OS or browser chord while the boundary is unset.
    // The ONE carve-out is the oracle-pinned window chords (Ctrl-o,
    // Ctrl-i - the app's own back/forward bindings): the window
    // context owns exactly those two, so they reach the matcher;
    // every other modifier chord still passes native.
    if (
      (event.ctrlKey || event.altKey || event.metaKey) &&
      !isPinnedWindowChord({
        key: event.key,
        ctrl: event.ctrlKey,
        shift: event.shiftKey,
        alt: event.altKey,
        meta: event.metaKey,
      })
    )
      return;
    const stroke = {
      key: event.key,
      ctrl: event.ctrlKey,
      shift: event.shiftKey,
      alt: event.altKey,
      meta: event.metaKey,
    };
    const disposition = field
      ? instrument.handleKey(stroke)
      : dispatchStroke(windowHost, instrument, stroke);
    if (disposition === "consume") event.preventDefault();
  });

  // -- the lanes' test-open channel: the one startup open ----------------
  async function openTestBoard(): Promise<boolean> {
    let invoke: ReturnType<typeof tauriInvoke>;
    try {
      invoke = tauriInvoke();
    } catch (error) {
      if (error instanceof SourceUnavailableError) {
        console.log("[viewer] startup: no bridge; explicit controls only");
        return false;
      }
      throw error;
    }
    try {
      console.log("[viewer] startup: bridge present");
      const restored = await ensureWindowState();
      if (restored !== null && restored.length > 0) {
        console.log(`[viewer] startup: restoring the bound state's location: ${restored}`);
        openSpec(restored);
        return true;
      }
      const spec = (await invoke("test_open")) as string | null;
      if (!spec) {
        console.log("[viewer] no test open; starting empty");
        return false;
      }
      console.log(`[viewer] test open requested: ${spec}`);
      openSpec(spec);
      return true;
    } catch (error) {
      console.error(`[viewer] test open board FAILED: ${String(error)}`);
      return true;
    }
  }

  async function boot(): Promise<void> {
    const requested = await openTestBoard();
    // The probe: the slice-1 temporary readout ride. The observation
    // contract itself (__saccadePort.readout) is product surface; this
    // console echo exists only until the fresh harness adapter reads
    // the contract through its own door (slice 2, coordinated via the
    // orchestrator). It prints already-existing state, never sets it.
    const settle = (ready: () => boolean) =>
      new Promise<void>((resolve) => {
        const deadline = Date.now() + 20_000;
        const timer = setInterval(() => {
          if (ready() || Date.now() > deadline) {
            clearInterval(timer);
            resolve();
          }
        }, 200);
      });
    await settle(() => {
      const state = lastFrame;
      if (state.kind !== "alive") return true;
      return state.content.kind === "available" || state.content.kind === "unavailable";
    });
    const state = lastFrame;
    if (state.kind === "alive" && state.content.kind === "available") {
      const frame = state.content.value.props["content"] as
        { board?: { content?: unknown[] } | null } | undefined;
      const blocks = frame?.board?.content?.length ?? 0;
      console.log(`[viewer] test open board opened: ${blocks} blocks`);
    } else if (state.kind === "alive" && state.content.kind === "unavailable") {
      const failure = diagnostics.entries().at(-1);
      console.error(`[viewer] test open board FAILED: ${failure ? failure.message : "no board"}`);
    } else if (requested) {
      console.error("[viewer] test open board FAILED: no board");
    }
    console.log(`[port] observation: ${JSON.stringify(readout())}`);
  }

  function readout(): PortObservation {
    const metrics = viewport.metrics();
    return buildReadout({
      windowLabel: nativeWindowLabel(),
      link: linkLabel,
      ask: ask === null
        ? null
        : { act: ask.act, spec: ask.spec, chips: ask.chips },
      state: lastFrame,
      diagnostics: diagnostics.entries(),
      doors,
      trail: trail.view(),
      viewport: {
        left: metrics.position.left,
        top: metrics.position.top,
        width: metrics.width,
        height: metrics.height,
      },
      copy: lastCopy,
    });
  }

  // the observation surface: the port's own product door, read-only
  shellWindow[PORT_SURFACE] = {
    readout,
  };

  return {
    attachViewport(element: HTMLElement | null) {
      viewport.attachElement(element);
    },
    openSpec,
    boot,
    subscribeFrame(observer) {
      frameObservers.add(observer);
      observer(lastFrame);
      return () => {
        frameObservers.delete(observer);
      };
    },
    subscribeStrip(observer) {
      stripObservers.add(observer);
      observer(stripView());
      return () => {
        stripObservers.delete(observer);
      };
    },
    readout,
    trailBack: () => trail.back(),
    trailForward: () => trail.forward(),
    trailView: () => trail.view(),
    openPicker: () => picker.open(),
    closePicker: () => picker.close(),
    submitPicker: (spec) => picker.submit(spec),
    subscribePicker(observer) {
      return picker.subscribe(observer);
    },
    answerAsk: (chip) => {
      void answerAsk(chip);
    },
    cancelAsk: () => {
      void cancelAsk();
    },
    subscribeAsk(observer) {
      askObservers.add(observer);
      observer(ask);
      return () => {
        askObservers.delete(observer);
      };
    },
    destroy() {
      windowHost.dispose();
      unsubscribeFrame();
      instrument.destroy();
    },
  };
}
