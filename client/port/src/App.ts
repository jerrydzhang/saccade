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
import {
  PORT_SURFACE,
  buildReadout,
  nativeWindowLabel,
  type DoorDelivery,
  type PortObservation,
} from "./lib/observation.ts";
import { tauriInvoke } from "./lib/sources.ts";
import { formatLocation } from "./lib/location.ts";
import { selectToPart } from "./lib/board/targets.ts";
import { createTrail, type TrailView } from "./lib/navigation/trail.ts";
import { createHostMatcher, dispatchStroke, type HostMatcher } from "./lib/input/host.ts";
import { windowContext } from "./lib/input/contexts.ts";
import { createPickerController, type PickerState } from "./lib/input/picker.ts";
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

  // Window commands own the picker. Reading interactions and their
  // captured field commands belong to the active instrument.
  const windowHost: HostMatcher = createHostMatcher(
    windowContext({
      openPicker: () => picker.open(),
      closeOverlays: () => picker.close(),
      pickerIsOpen: () => picker.snapshot().open,
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

  // citation activation: the hydration-created links hand off to the
  // window through this event; a citation naming a select path opens
  // anchored there (the grammar path mapped to the canonical part);
  // shifted opens elsewhere defer to the suite slice - reported,
  // never silently flattened
  window.addEventListener("suite-open-citation", (event) => {
    const detail = (event as CustomEvent<{ machine: string; path: string; select?: string | null; shifted: boolean }>)
      .detail;
    if (!detail || typeof detail.path !== "string") return;
    if (detail.shifted) {
      console.log("[port] shifted citation open deferred (suite slice)");
      return;
    }
    const part = typeof detail.select === "string" ? selectToPart(detail.select) : null;
    // The open takes the parseable location form, never a hand-rolled
    // machine:path join: a tracker machine ("tracker:seedtracker") with
    // a colon join parses as nothing ("not a location" - the S08 live
    // finding). formatLocation builds the form the picker and the
    // header field take back; the tracker's host resolves through the
    // registry the first hosted open registered.
    openSpec(formatLocation({ machine: detail.machine, path: detail.path }), part === "" ? null : part);
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
    if (event.ctrlKey || event.altKey || event.metaKey) return;
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
    destroy() {
      windowHost.dispose();
      unsubscribeFrame();
      instrument.destroy();
    },
  };
}
