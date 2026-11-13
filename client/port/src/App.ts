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
import { tasksInstrument } from "./instruments/tasks.ts";
import type { TasksUI } from "./instruments/tasks.ts";
import { instrumentPath } from "./lib/location.ts";
import type { Address } from "./lib/contracts/places.ts";
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
import { SEQUENCE_TIMEOUT_MS } from "./lib/input/key-matcher.ts";
import { whichKeyTunables } from "./lib/input/which-key.ts";
import { isPinnedChord, MODIFIER_KEYS, windowContext } from "./lib/input/contexts.ts";
import { createPickerController, type PickerState } from "./lib/input/picker.ts";
import {
  assignChips,
  elsewhereDecision,
  type AskState,
  type ElsewhereAct,
  type SuiteWindow,
} from "./lib/elsewhere.ts";
import { WindowStore } from "./lib/windows.ts";
import { serverBase, TrackerSource } from "./lib/tracker.ts";
import { formatLocation } from "./lib/location.ts";
import { sourceFor } from "./lib/sources.ts";
import { tasksPageOf } from "./lib/tasks/page.ts";
import type { PickContext } from "./lib/picker/model.ts";
import {
  zoomIn as inspectionZoomIn,
  zoomOut as inspectionZoomOut,
  zoomReset as inspectionZoomReset,
  zoomWheel as inspectionZoomWheel,
  type InspectionState,
} from "./lib/figures/inspection.ts";
import { inspectionContext } from "./lib/input/contexts.ts";
import { SourceUnavailableError } from "./lib/board/errors.ts";
import { knownIssue } from "./lib/contracts/buffer.ts";
import type { BufferState } from "./lib/contracts/buffer.ts";
import { createKeyTrace } from "./lib/input/key-trace.ts";
import {
  changeReadingLayout,
  DEFAULT_READING_LAYOUT,
  readingLayoutOf,
  readingLayoutVars,
  type ReadingLayout,
  type ReadingLayoutChange,
} from "./lib/board/reading-layout.ts";

/** The which-key popup delay in ms (the r110 tunable, nvim's
 * neighborhood - the human tunes). */
export const WHICH_KEY_DELAY_MS = 400;

export interface StripView {
  location: string | null;
  status: string;
  jumpChip: string | null;
  note: string | null;
  /** The link state as information (the floor's law: form open):
   * the linked window's label, or null when this window is
   * unlinked. The o act's target, made visible. */
  link: string | null;
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
  /** The picker's ranked-list context, refreshed on open (the
   * tracker's task list arrives asynchronously). */
  pickerContext(): PickContext;
  subscribePickerContext(observer: (context: PickContext) => void): () => void;
  /** The which-key surface (r110): the live popup view (null when
   * hidden) and the help variant; the observation carries both. */
  whichKeyView(): { pending: boolean; view: unknown } | null;
  helpView(): unknown;
  subscribeWhichKey(observer: (pending: unknown, help: boolean) => void): () => void;
  closePicker(): void;
  submitPicker(spec: string): void;
  subscribePicker(observer: (state: PickerState) => void): () => void;
  answerAsk(chip: string): void;
  cancelAsk(): void;
  subscribeAsk(observer: (state: AskState | null) => void): () => void;
  closeInspection(): void;
  /** The viewer's pointer acts (the prototype's lightbox): a wheel
   * notch zooms, a drag pans to an absolute offset. */
  inspectionWheel(deltaY: number): void;
  inspectionPan(x: number, y: number): void;
  subscribeInspection(observer: (state: InspectionState | null) => void): () => void;
  readout(): PortObservation;
}

export function createPortWindow(windowId = "main"): PortWindow {
  const diagnostics: SliceDiagnostics = createDiagnostics();
  const buffers = createBufferFactory(windowId);
  const viewport: SliceBufferViewport = createViewport();

  // The window's buffers, one per surface kind, retained: the board
  // constructs at assembly, the tasks buffer constructs at its first
  // open (the glue's "called only when the window needs a NEW
  // buffer"). Exactly one surface is ACTIVE - keys, the trail and the
  // readout route to it; the other keeps its state (the trail's
  // return lands back on it).
  type Surface = "board" | "tasks";
  const instruments: Partial<Record<Surface, RunningInstrument>> = {};
  let activeSurface: Surface = "board";

  function surfaceOfAddress(address: Address): Surface {
    return instrumentPath(address.path) !== null ? "tasks" : "board";
  }

  function instrumentFor(surface: Surface): RunningInstrument {
    const existing = instruments[surface];
    if (existing !== undefined) return existing;
    if (surface === "tasks") {
      const registration = registerInstrument(tasksInstrument());
      const instrument = constructInstrument(registration, {
        context: { bufferId: `${windowId}-tasks-1`, windowId },
        viewport,
        diagnostics,
        buffers,
        expirySuspended: () => whichKey !== null,
      });
      instrument.activate();
      instrument.subscribeInputPending(() => {
        const view = livePending();
        whichKeyPending = view !== null;
        whichKey = view === null ? null : view;
        notifyWhichKey();
      });
      instruments.tasks = instrument;
      injectPorts();
      unsubscribeTasksFrame = subscribeSurface(instrument, "tasks");
      watchSelect("tasks");
      return instrument;
    }
    const registration = registerInstrument(boardInstrument());
    const instrument = constructInstrument(registration, {
      context: { bufferId: `${windowId}-board-1`, windowId },
      viewport,
      diagnostics,
      buffers,
      expirySuspended: () => whichKey !== null,
    });
    instrument.activate();
      instrument.subscribeInputPending(() => {
        const view = livePending();
        whichKeyPending = view !== null;
        whichKey = view === null ? null : view;
        notifyWhichKey();
      });
    instruments.board = instrument;
    return instrument;
  }

  function activeInstrument(): RunningInstrument {
    const instrument = instruments[activeSurface];
    if (instrument === undefined) throw new Error("the active surface carries no buffer");
    return instrument;
  }

  function setActive(surface: Surface): void {
    if (activeSurface === surface) return;
    instruments[activeSurface]?.deactivate();
    activeSurface = surface;
    instruments[activeSurface]?.activate();
    notifyStrip();
  }

  // The board buffer constructs at assembly and starts active (the
  // tasks buffer joins at its first open).
  instrumentFor("board");

  // The open-location picker: the window's open surface, submitting
  // through the ONE open funnel below (openSpec). Text-entry
  // semantics: the key seam skips field targets, so typing inside the
  // picker stays native (scenario 10).
  const picker = createPickerController((spec) => openSpec(spec));

  // The picker's context (F1, the scope rule: the window's ONE
  // tracker): recents are this window's own trail (newest first),
  // the task list loads from the tracker the window last reached,
  // and unopened tasks compose their specs from that same base.
  let pickerContextValue: PickContext = {
    recent: [],
    tasks: [],
    specOf: (id) => id,
    openHere: null,
  };
  const pickerContextObservers = new Set<(context: PickContext) => void>();

  async function refreshPickerContext(): Promise<void> {
    try {
      await refreshPickerContextInner();
    } catch (error) {
      diagnostics.report(
        refusalContext,
        knownIssue({ code: "asset-denied", message: `picker context refresh failed: ${String(error)}` }),
      );
    }
  }

  async function refreshPickerContextInner(): Promise<void> {
    const view = trail.view();
    const recent = [...view.entries].reverse().map((entry) => entry.location);
    const current = displayedFrame();
    let openHere: string | null = null;
    if (current.kind === "alive") {
      const frame = current.content.kind === "available" ? current.content.value.props["content"] as
        { location?: string } | undefined
        : undefined;
      openHere = frame?.location ?? null;
    }
    // the tracker machine rides the parser's own mapping (a host form
    // becomes a tracker: machine) - never a raw split of the string
    let trackerMachine: string | null = null;
    for (const candidate of [openHere, ...recent]) {
      if (typeof candidate !== "string") continue;
      const address = parseOpenSpec(candidate);
      if (address !== null && address.machine.startsWith("tracker:")) {
        trackerMachine = address.machine;
        break;
      }
    }
    const tasks: { id: string; title: string }[] = [];
    if (trackerMachine !== null) {
      try {
        const source = sourceFor(trackerMachine);
        if (source instanceof TrackerSource) {
          const model = tasksPageOf(await source.instrumentPage("tasks"));
          for (const row of model.groups) {
            for (const entry of row.rows) {
              tasks.push({ id: entry.task, title: entry.title });
            }
          }
        }
      } catch {
        // a dead tracker read: the picker still lists recents
      }
    }
    pickerContextValue = {
      recent,
      tasks,
      specOf: (id) =>
        trackerMachine !== null
          ? formatLocation({ machine: trackerMachine, path: id })
          : id,
      openHere,
    };
    for (const observer of [...pickerContextObservers]) observer(pickerContextValue);
  }


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

  // -- the which-key surface (r110): the pending prefix's popup and
  // the help variant ---------------------------------------------
  let whichKey: import("./lib/input/key-matcher.ts").WhichKeyView | null = null;
  let whichKeyPending = false;
  let helpOpen = false;
  const whichKeyObservers = new Set<(pending: unknown, help: boolean) => void>();

  function notifyWhichKey(): void {
    for (const observer of [...whichKeyObservers]) observer(whichKey, helpOpen);
  }



  async function beginLinkAsk(): Promise<void> {
    // space-l: the prototype's D1 modal - labels over windows, a key
    // sets the link; the ask machinery's link act is the same shape
    if (windowStore === null || windowStateId === null) {
      await ensureWindowState();
      if (windowStore === null || windowStateId === null) return;
    }
    const windows = await suiteWindows();
    const others = windows.filter((entry) => entry.label !== nativeWindowLabel());
    if (others.length === 0) {
      note = "no other window to link";
      notifyStrip();
      return;
    }
    await openAsk("link", "this window");
  }

  async function clearTheLink(): Promise<void> {
    if (windowStore === null || windowStateId === null) return;
    await windowStore.clearLink(windowStateId).catch(() => {});
    await refreshLink();
  }

  function notifyAsk(): void {
    for (const observer of askObservers) observer(ask);
  }

  let attachedViewport: HTMLElement | null = null;

  // -- the board's reading layout (lib/board/reading-layout.ts) -------
  // The window holds it beside the rest of its state: applied as CSS
  // custom properties on the content scroller (the board's stylesheet
  // reads them), and written to the window state once the steps
  // settle, never per key press. A bound state's stored layout applies
  // at boot unless this window already changed its own.
  const READING_LAYOUT_SETTLE_MS = 400;
  let readingLayout: ReadingLayout = DEFAULT_READING_LAYOUT;
  let readingLayoutChanged = false;
  let readingLayoutTimer: ReturnType<typeof setTimeout> | null = null;

  function applyReadingLayout(layout: ReadingLayout): void {
    readingLayout = layout;
    const element = attachedViewport;
    if (element === null) return;
    for (const [name, value] of Object.entries(readingLayoutVars(layout))) {
      element.style.setProperty(name, value);
    }
  }

  function changeLayout(change: ReadingLayoutChange): void {
    readingLayoutChanged = true;
    applyReadingLayout(changeReadingLayout(readingLayout, change));
    scheduleReadingLayoutWrite();
  }

  function scheduleReadingLayoutWrite(): void {
    if (readingLayoutTimer !== null) clearTimeout(readingLayoutTimer);
    readingLayoutTimer = setTimeout(() => {
      readingLayoutTimer = null;
      if (windowStore === null || windowStateId === null) return;
      const { zoom, width } = readingLayout;
      void windowStore.patch(windowStateId, { zoom, width }).catch(() => {
        // an unreachable store: the layout stays this session's; the
        // next change writes it again
      });
    }, READING_LAYOUT_SETTLE_MS);
  }

  // -- figure inspection (S04): a TEMPORARY window surface ----------
  // It dies on close, never enters the trail, and while open it is
  // modal for keys: the inspector's context answers every stroke
  // (its own + - = and Escape act; anything else consumed inert), so
  // nothing falls through to the board beneath - which keeps its
  // updates the whole time (deactivation decouples input from
  // content).
  let inspection: InspectionState | null = null;
  const inspectionObservers = new Set<(state: InspectionState | null) => void>();

  function notifyInspection(): void {
    for (const observer of inspectionObservers) observer(inspection);
  }

  const inspectionHost: HostMatcher = createHostMatcher(
    inspectionContext({
      zoomIn: () => {
        if (inspection !== null) inspection = { ...inspection, zoom: inspectionZoomIn(inspection.zoom) };
        notifyInspection();
        notifyStrip();
      },
      zoomOut: () => {
        if (inspection !== null) inspection = { ...inspection, zoom: inspectionZoomOut(inspection.zoom) };
        notifyInspection();
        notifyStrip();
      },
      zoomReset: () => {
        // = resets the zoom AND the pan (the prototype's reset)
        if (inspection !== null) inspection = { ...inspection, zoom: inspectionZoomReset(), x: 0, y: 0 };
        notifyInspection();
        notifyStrip();
      },
      close: () => {
        inspection = null;
        notifyInspection();
        notifyStrip();
      },
      pan: (dx, dy) => {
        if (inspection !== null) inspection = { ...inspection, x: inspection.x + dx, y: inspection.y + dy };
        notifyInspection();
      },
    }),
  );

  /** The shown diagram's SVG, cloned with an explicit size from its
   * viewBox (the rendered one sizes itself to its column). */
  function diagramSvg(svg: SVGSVGElement): string {
    const clone = svg.cloneNode(true) as SVGSVGElement;
    const box = svg.viewBox?.baseVal;
    if (box !== undefined && box !== null && box.width > 0 && box.height > 0) {
      clone.setAttribute("width", String(box.width));
      clone.setAttribute("height", String(box.height));
    }
    clone.removeAttribute("style");
    return new XMLSerializer().serializeToString(clone);
  }

  function openInspection(part: string): void {
    // enlarge what is SHOWN: the displayed figure's own asset (the
    // hydration's data: URL) and its caption, read at open time
    // the part element may BE the figure block (figure parts carry
    // data-port-part on the block itself) or contain one
    const partElement = (attachedViewport ?? document).querySelector(
      `[data-port-part="${part}"]`,
    );
    const element =
      partElement instanceof HTMLElement && partElement.classList.contains("block-figure")
        ? partElement
        : (partElement?.querySelector(".block-figure") ?? null);
    // a rendered diagram (its own part, data-port-kind diagram): the
    // shown SVG, enlarged as an image
    if (partElement instanceof HTMLElement && partElement.getAttribute("data-port-kind") === "diagram") {
      const svg = partElement.querySelector("svg");
      if (svg === null) {
        note = `the diagram at ${part} is not rendered (nothing to enlarge)`;
        notifyStrip();
        return;
      }
      inspection = { part, src: "", svg: diagramSvg(svg), caption: "diagram", zoom: inspectionZoomReset(), x: 0, y: 0 };
      notifyInspection();
      notifyStrip();
      return;
    }
    const img = (element?.querySelector("img.figure-img") as HTMLImageElement | null) ?? null;
    const text = element?.querySelector("pre.figure-text")?.textContent ?? null;
    if (element === null || (img === null && text === null)) {
      note = `the figure at ${part} is not displayed (no asset to enlarge)`;
      notifyStrip();
      return;
    }
    const caption = element.querySelector("figcaption")?.textContent ?? "";
    inspection =
      img !== null
        ? { part, src: img.src, caption, zoom: inspectionZoomReset(), x: 0, y: 0 }
        : { part, src: "", text: text ?? "", caption, zoom: inspectionZoomReset(), x: 0, y: 0 };
    notifyInspection();
    notifyStrip();
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
      windowStore = new WindowStore({ base: () => serverBase().toString() });
      const bind = async (id: string): Promise<void> => {
        await invoke("bind_window_state", {
          stateId: id,
          server: base.toString(),
          device: windowStore!.device,
        });
      };
      if (typeof binding.state_id === "string" && binding.state_id.length > 0) {
        windowStateId = binding.state_id;
        await windowStore.attach(windowStateId).catch(() => {
          // the store may be unreachable (a relaunch's persisted host is
          // dead until a location re-tells it): the shell's seed still
          // binds this window; the store facts catch up on the next
          // successful reach
        });
        await bind(windowStateId);
      } else if (binding.label === "main") {
        // the launch restore: the SHELL's memory names the states open
        // at the last quit; the first is claimed here, each of the rest
        // opens as its own window (the seed binds it). The windows come
        // back unconditionally - the memory is the persistence - even
        // when the store is not yet reachable (their locations restore
        // through their own boots when it is)
        const last = (await invoke("last_session")) as string[];
        let claimed = false;
        for (const id of last) {
          if (!claimed) {
            windowStateId = id;
            claimed = true;
          } else {
            await invoke("open_suite_window", { stateId: id }).catch(() => {});
          }
        }
        if (!claimed) {
          const fresh =
            typeof crypto.randomUUID === "function"
              ? crypto.randomUUID()
              : `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
          await windowStore.create(fresh, {});
          windowStateId = fresh;
        }
        const claimedId = windowStateId;
        if (claimedId !== null) {
          await windowStore.attach(claimedId).catch(() => {});
          await bind(claimedId);
        }
      } else {
        // a suite window the shell did not seed: its own fresh state
        const fresh =
          typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        await windowStore.create(fresh, {});
        windowStateId = fresh;
        await windowStore.attach(fresh).catch(() => {});
        await bind(fresh);
      }
      const boundId = windowStateId;
      const state =
        boundId === null ? null : await windowStore.getState(boundId).catch(() => null);
      restored = state === null ? null : state.location;
      if (readingLayoutChanged) scheduleReadingLayoutWrite();
      else if (state !== null) applyReadingLayout(readingLayoutOf(state));
      await refreshLink();
      injectPorts();
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
      // a state that no window carries is a dead link: clear it, so
      // the strip never names a ghost and o never delivers into one
      if (linkLabel === null) await windowStore.clearLink(windowStateId).catch(() => {});
    } catch {
      // a dead store read: the link resolves null until the next
      // refresh; o asks rather than delivering into a guess
    } finally {
      notifyStrip();
    }
  }

  // Opening through the link FOCUSES the receiver (human ruling m-547
  // (3), reversing t-14's "a new window never asks for focus" for the
  // elsewhere acts): the platform's activation request, the window
  // jump's own door. A refused request is a note, never a failed open.
  async function focusReceiver(label: string): Promise<void> {
    try {
      await tauriInvoke()("focus_suite_window", { label });
    } catch (error) {
      note = `the receiver ${label} could not be focused: ${String(error).slice(0, 80)}`;
      notifyStrip();
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
    await focusReceiver(label);
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
      // A JUST-OPENED window's state can still be binding when the
      // human's o arrives (the suite worker's base defect 1: the act
      // silently dropped, 'nothing happened'). WAIT for the binding,
      // bounded - the act runs when the store lands; only a genuinely
      // unbound window refuses, with the note.
      const deadline = Date.now() + 4000;
      while ((windowStore === null || windowStateId === null) && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 120));
        await ensureWindowState();
      }
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
      await focusReceiver(decision.label);
      return;
    }
    if (decision.kind === "new-window") {
      await openInNewWindow(spec, decision.becomeLink);
      return;
    }
    await openAsk(act, spec);
  }

  async function openAsk(act: ElsewhereAct, spec: string): Promise<void> {
    // the ask: chips on every window, the answer typed here
    const invoke = tauriInvoke();
    const windows = await suiteWindows();
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
    // the space-l link ask sets a link and opens nothing ("this window"
    // is its spec): focus moves only for an open
    if (asking.spec !== "this window") await focusReceiver(match.window);
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

  /** The controller ports over every constructed surface: the open
   * funnel, the copy observation, the elsewhere acts. The closures
   * read the live world at call time, so a rebinding needs only a
   * re-tap, and the acts appear with the window state (S18). */
  function injectPorts(): void {
    const board = instruments.board;
    if (board !== undefined) {
      (board.presentation as BoardUI).setReferencePorts({
        open: (spec: string, part: string | null) => openSpec(spec, part),
        figureEnlarge: (part: string) => {
          openInspection(part);
        },
        copy: (payload: string) => {
          lastCopy = { payload, at: Date.now() };
          try {
            void navigator.clipboard.writeText(payload);
          } catch {
            // the lane's webview may refuse the clipboard API; the
            // recorded payload is the act's truth either way
          }
          notifyStrip();
        },
        elsewhere: (spec: string, act: ElsewhereAct) => {
          openElsewhere(spec, act).catch((error: unknown) => {
            diagnostics.report(
              refusalContext,
              knownIssue({ code: "read-failed", message: `the open act failed: ${String(error)}` }),
            );
            note = `the open act failed: ${String(error).slice(0, 120)}`;
            notifyStrip();
          });
        },
      });
    }
    if (board !== undefined) {
      (board.presentation as BoardUI).setReadingLayoutPort(changeLayout);
    }
    const tasks = instruments.tasks;
    if (tasks !== undefined) {
      (tasks.presentation as TasksUI).setTaskPorts({
        open: (spec: string) => openSpec(spec),
        elsewhere: (spec: string, act: ElsewhereAct) => {
          openElsewhere(spec, act).catch((error: unknown) => {
            diagnostics.report(
              refusalContext,
              knownIssue({ code: "read-failed", message: `the open act failed: ${String(error)}` }),
            );
            note = `the open act failed: ${String(error).slice(0, 120)}`;
            notifyStrip();
          });
        },
        copy: (text: string) => {
          lastCopy = { payload: text, at: Date.now() };
          try {
            void navigator.clipboard.writeText(text);
          } catch {
            // the recorded payload is the act's truth either way
          }
          notifyStrip();
        },
      });
    }
  }

  // Window commands own the picker. Reading interactions and their
  // captured field commands belong to the active instrument.
  const windowHost: HostMatcher = createHostMatcher(
    windowContext({
      openPicker: () => {
        void refreshPickerContext();
        picker.open();
      },
      closeOverlays: () => {
        picker.close();
        whichKey = null;
        helpOpen = false;
        notifyWhichKey();
      },
      enterFind: () => {
        if (activeSurface === "board" && instruments.board !== undefined) {
          (instruments.board.presentation as BoardUI).enterFind();
        }
      },
      linkByLabels: () => {
        void beginLinkAsk();
      },
      unlink: () => {
        void clearTheLink();
      },
      showHelp: () => {
        helpOpen = true;
        notifyWhichKey();
        notifyStrip();
      },
      pickerIsOpen: () => picker.snapshot().open,
      trailBack: () => trail.back(),
      trailForward: () => trail.forward(),
    }),
    { expirySuspended: () => whichKey !== null },
  );

  // the popup's own clock: the delay after the prefix began; the
  // expiry suspension reads whichKey !== null
  let whichKeyDelayTimer: number | null = null;
  // the routed pending: the ACTIVE INSTRUMENT's first (its reading
  // context owns the board keys), the window's when the instrument
  // holds none - the which-key property holds at every level
  function livePending(): import("./lib/input/key-matcher.ts").WhichKeyView | null {
    const instrument = instruments[activeSurface];
    if (instrument !== undefined) {
      const view = instrument.inputPending();
      if (view !== null) return view;
    }
    return windowHost.pendingView();
  }

  windowHost.subscribePending(() => {
    const view = livePending();
    whichKeyPending = view !== null;
    if (view === null) {
      whichKey = null;
      if (whichKeyDelayTimer !== null) {
        clearTimeout(whichKeyDelayTimer);
        whichKeyDelayTimer = null;
      }
    } else if (whichKey === null && whichKeyDelayTimer === null) {
      const since = windowHost.pendingSince();
      if (since !== null) {
        const elapsed = Date.now() - since;
        const wait = Math.max(0, WHICH_KEY_DELAY_MS - elapsed);
        whichKeyDelayTimer = setTimeout(() => {
          whichKeyDelayTimer = null;
          const live = livePending();
          if (live !== null) whichKey = live;
          notifyWhichKey();
        }, wait) as unknown as number;
      }
    } else {
      whichKey = view;
    }
    notifyWhichKey();
  });


  // -- observation state ------------------------------------------------
  let doors: { jumpChip: string | null; delivered: DoorDelivery[] } = {
    jumpChip: null,
    delivered: [],
  };
  let note: string | null = null;

  const frameObservers = new Set<(state: BufferState<RenderPair>) => void>();
  const stripObservers = new Set<(strip: StripView) => void>();
  const lastFrames: Partial<Record<Surface, BufferState<RenderPair>>> = {};

  // The window's trail (scenario 14): explicit openings only; the
  // departure capture lands on the entry being left; Back/Forward
  // restore recorded places - never folds, zoom or drafts. Key
  // bindings for Back/Forward are pinned when the S14 body lands
  // (prototype parity through the adapter); the surface is real now.
  const trail = createTrail({
    capture: () => activeInstrument().places.capture(),
    restore: (place) => {
      // the restored address names its own surface: the window
      // activates that surface's retained buffer and restores there
      const surface = surfaceOfAddress(place.address);
      const target = instrumentFor(surface);
      setActive(surface);
      target.places.restore(place);
    },
  });

  let unsubscribeTasksFrame: (() => void) | null = null;

  function subscribeSurface(instrument: RunningInstrument, surface: Surface): () => void {
    return instrument.subscribe((state) => {
      lastFrames[surface] = state;
      if (surface !== activeSurface) return;
      for (const observer of [...frameObservers]) observer(state);
      notifyStrip();
    });
  }
  const unsubscribeBoardFrame = subscribeSurface(activeInstrument(), "board");

  function displayedFrame(): BufferState<RenderPair> {
    const state = lastFrames[activeSurface];
    if (state !== undefined) return state;
    const instrument = instruments[activeSurface];
    return instrument !== undefined ? instrument.snapshot() : { kind: "destroyed" };
  }

  function stripView(): StripView {
    const state = displayedFrame();
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
    return { location, status, jumpChip: doors.jumpChip, note, link: linkLabel };
  }

  /** The readout's input field (batch 3's contract): the pending
   * strokes in help labels, the popup's content while shown, and
   * the NAMED tunables. v1 reads the WINDOW host's pending; the
   * instrument chain (g's continuations) extends this when the
   * instruments expose their views. */
  function inputView(): Readonly<{
    pending: readonly string[];
    whichKey: Readonly<{
      prefix: string;
      entries: readonly { keys: string; label: string; actionId: string }[];
    }> | null;
    tunables: Readonly<{
      whichKeyDelayMs: number;
      whichKeySuspendsTimeout: boolean;
      sequenceTimeoutMs: number;
    }>;
  }> {
    const pendingView = livePending();
    const pending = pendingView === null
      ? []
      : pendingView.prefix.map((stroke) =>
          [
            stroke.ctrl ? "Ctrl-" : "",
            stroke.alt ? "Alt-" : "",
            stroke.meta ? "Meta-" : "",
            stroke.key === " " ? "spc" : stroke.key,
          ].join(""),
        );
    let popup: ReturnType<typeof inputView>["whichKey"] = null;
    if (helpOpen) {
      // the help surface IS the which-key view over the empty prefix:
      // the UNION of the routed tables - the window's commands and
      // the active instrument's own
      const merged: { keys: string; label: string; actionId: string }[] = [
        ...windowHost.allView().entries,
      ];
      const instrument = instruments[activeSurface];
      if (instrument !== undefined) merged.push(...instrument.inputAll().entries);
      popup = { prefix: "", entries: merged };
    } else if (whichKey !== null) {
      popup = {
        prefix: whichKey.prefix
          .map((stroke) => (stroke.key === " " ? "spc" : stroke.key))
          .join(" "),
        entries: whichKey.entries,
      };
    }
    return {
      pending,
      whichKey: popup,
      tunables: {
        whichKeyDelayMs: whichKeyTunables.delayMs,
        whichKeySuspendsTimeout: whichKeyTunables.suspendsTimeout,
        sequenceTimeoutMs: SEQUENCE_TIMEOUT_MS,
      },
    };
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
  // the open-refusal diagnostic's buffer context: a parse refusal
  // names no buffer yet, so it rides the board buffer (the assembly
  // surface)
  const refusalContext = { bufferId: `${windowId}-board-1`, windowId };
  function openSpec(spec: string, part?: string | null): void {
    const address = parseOpenSpec(spec);
    if (address === null) {
      note = `not a location: ${JSON.stringify(spec)}`;
      diagnostics.report(refusalContext, knownIssue({ code: "asset-denied", message: note }));
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
    // the departure capture rides the CURRENT surface (the entry
    // being left), then the opened address names its own
    trail.opened(spec);
    const surface = surfaceOfAddress(address);
    const target = instrumentFor(surface);
    setActive(surface);
    const place = {
      bufferId: target.id,
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
    target.places.restore(place);
    notifyStrip();
  }

  // -- controller ports: the citation acts and the task-row acts ride
  // the window's one open funnel, the copy observation and the
  // elsewhere surface (scenario 9/13/17). No controller offers an act
  // before these are injected; the follow composes its spec itself
  // (formatLocation, never a colon join) and the open lands anchored
  // exactly as a typed location would.
  let lastCopy: { payload: string; at: number } | null = null;
  injectPorts();

  // -- the key routing seam: inspection (modal), then the window
  // context, then the active buffer ---
  // the live select views (the trace's labels and typed buffer while
  // a select session holds the keys)
  const selectViews: Partial<Record<Surface, import("./lib/select/session.ts").SelectSessionView | null>> = {};
  function watchSelect(surface: Surface): void {
    const presentation = instruments[surface]?.presentation as
      | { subscribeSelect?: (observer: (view: import("./lib/select/session.ts").SelectSessionView | null) => void) => () => void }
      | undefined;
    presentation?.subscribeSelect?.((view) => {
      selectViews[surface] = view;
    });
  }
  watchSelect("board");
  function selectFacts(): Record<string, string> {
    const view = selectViews[activeSurface];
    if (view === null || view === undefined) return {};
    return {
      "select.labels": view.labels.slice(0, 40).map((entry) => `${entry.label}=${entry.part}`).join(",") +
        (view.labels.length > 40 ? `,+${view.labels.length - 40}` : ""),
      "select.typed": view.pendingLabel,
    };
  }

  // the key trace (SACCADE_KEY_TRACE, lib/input/key-trace.ts): off
  // unless the shell says so; it reads, never routes
  const keyTrace = createKeyTrace({
    scroller: () => attachedViewport,
    facts: () => {
      const state = activeInstrument().inputState();
      return {
        surface: activeSurface,
        "buf.active": state.active,
        "buf.ctx": `${state.context}/${state.kind ?? "none"}`,
        "buf.pending": state.pending.map((stroke) => stroke.key).join(","),
        ...selectFacts(),
        "win.pending": (windowHost.pendingView()?.prefix ?? []).map((stroke) => stroke.key).join(","),
        picker: picker.snapshot().open,
        ask: ask !== null,
        inspection: inspection !== null,
        whichKey: whichKey !== null,
        help: helpOpen,
      };
    },
  });

  const windowKeyDown = (event: KeyboardEvent): void => {
    const trace = keyTrace.begin(event);
    try {
      routeKeyDown(event, trace);
    } finally {
      trace?.end();
    }
  };

  const routeKeyDown = (
    event: KeyboardEvent,
    trace: ReturnType<typeof keyTrace.begin>,
  ): void => {
    if (event.defaultPrevented) {
      trace?.set("route", "already-prevented");
      return;
    }
    // a bare modifier is never a command: it passes untouched, before
    // any surface (the inspection's modal consume included) sees it
    if (MODIFIER_KEYS.has(event.key)) {
      trace?.set("route", "modifier");
      return;
    }
    // the inspection surface owns every stroke while it is open: its
    // own keys act, everything else is consumed inert - the board
    // beneath never sees a key (scenario 4)
    if (inspection !== null) {
      const stroke = {
        key: event.key,
        ctrl: event.ctrlKey,
        shift: event.shiftKey,
        alt: event.altKey,
        meta: event.metaKey,
      };
      const resolution = inspectionHost.stroke(stroke);
      inspectionHost.apply(resolution);
      event.preventDefault();
      trace?.set("route", "inspection");
      trace?.set("answer", resolution.kind);
      return;
    }
    // the help surface is a modal too: every stroke closes it (the
    // inspection's own-deactivation semantics), so the reading
    // context's Escape-clears-search keeps its stroke when no help
    // is open
    if (helpOpen) {
      helpOpen = false;
      notifyWhichKey();
      event.preventDefault();
      return;
    }
    const target = event.target as HTMLElement | null;
    const field = target !== null &&
      (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
    // Declared captured fields use their active buffer's text-entry
    // context. Other fields (including the window picker) stay native.
    if (field && !target?.closest("[data-port-captured-input]")) {
      trace?.set("route", "native-field");
      return;
    }
    // Modifier chords stay native until the IME/native-editing
    // boundary set is pinned (draft-open, grouped decision): the port
    // never eats an OS or browser chord while the boundary is unset.
    // The carve-out is the pinned chords (contexts.ts PINNED_CHORDS:
    // the prototype's own Ctrl-o/i trail and Ctrl-f/b/d/u paging):
    // those reach the matcher; every other modifier chord still
    // passes native.
    if (
      (event.ctrlKey || event.altKey || event.metaKey) &&
      !isPinnedChord({
        key: event.key,
        ctrl: event.ctrlKey,
        shift: event.shiftKey,
        alt: event.altKey,
        meta: event.metaKey,
      })
    ) {
      trace?.set("route", "native-chord");
      return;
    }
    const stroke = {
      key: event.key,
      ctrl: event.ctrlKey,
      shift: event.shiftKey,
      alt: event.altKey,
      meta: event.metaKey,
    };
    trace?.set("route", field ? "captured-field" : "window-then-buffer");
    let windowAnswer: string | null = null;
    const disposition = field
      ? activeInstrument().handleKey(stroke)
      : dispatchStroke(
          windowHost,
          activeInstrument(),
          stroke,
          trace === null
            ? undefined
            : (owner, answer) => {
                windowAnswer = answer;
                trace.set(owner, answer);
              },
        );
    if (trace !== null && (field || !/^invoke:|^prefix$/.test(String(windowAnswer)))) {
      trace.set("buffer", activeInstrument().inputState().lastAnswer);
    }
    trace?.set("disposition", disposition);
    if (disposition === "consume") event.preventDefault();
  };
  window.addEventListener("keydown", windowKeyDown);

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
        // a restored location never falls through to the test-open
        // channel; the window SET restored through the launch restore
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
      const state = displayedFrame();
      if (state.kind !== "alive") return true;
      return state.content.kind === "available" || state.content.kind === "unavailable";
    });
    const state = displayedFrame();
    if (state.kind === "alive" && state.content.kind === "available") {
      const frame = state.content.value.props["content"] as
        { board?: { content?: unknown[] } | null } | undefined;
      const blocks = frame?.board?.content?.length ?? 0;
      console.log(`[viewer] test open board opened: ${blocks} blocks`);
    } else if (state.kind === "alive" && state.content.kind === "unavailable") {
      const failure = diagnostics.entries().at(-1);
      console.error(`[viewer] test open board FAILED: ${failure ? failure.message : "no board"}`);
    }
    // E4 (t-34@r6 change 4, the ratified floor): a COLD profile - a
    // genuinely first launch with no opened content - shows the TASKS
    // window, never a blank page. With no tracker reached yet the
    // tasks surface carries its honest empty state; the picker (F1)
    // supplies the first tracker and the list loads through it. Any
    // open - the lane's test open, a restore, a delivery - leaves the
    // surface to its own routing.
    const after = displayedFrame();
    if (after.kind !== "alive" || (after.content.kind !== "available" && after.content.kind !== "unavailable")) {
      // the tasks instrument constructs through its own door (the
      // buffer joins here, empty until a tracker is reached)
      void instrumentFor("tasks");
      setActive("tasks");
    } else if (requested) {
      console.error("[viewer] test open board FAILED: no board");
    }
    console.log(`[port] observation: ${JSON.stringify(readout())}`);
  }

  // GTK's focus facts for this window (m-537): refreshed in the
  // background - at boot, on focus and blur, and on each readout -
  // since the readout itself is synchronous
  let focusState: PortObservation["window"]["focus"] = null;
  let focusRefreshing = false;
  function refreshFocusState(): void {
    if (focusRefreshing) return;
    let invoke: ReturnType<typeof tauriInvoke>;
    try {
      invoke = tauriInvoke();
    } catch {
      return;
    }
    focusRefreshing = true;
    void invoke("window_focus_state")
      .then((answer) => {
        const state = answer as { accepts_focus: boolean; focus_on_map: boolean; is_active: boolean; focus_widget?: string | null };
        focusState = {
          acceptsFocus: state.accepts_focus,
          focusOnMap: state.focus_on_map,
          active: state.is_active,
          focusWidget: state.focus_widget ?? null,
        };
      })
      .catch(() => {})
      .finally(() => {
        focusRefreshing = false;
      });
  }
  refreshFocusState();
  window.addEventListener("focus", refreshFocusState);
  window.addEventListener("blur", refreshFocusState);

  function readout(): PortObservation {
    refreshFocusState();
    const metrics = viewport.metrics();
    return buildReadout({
      surface: activeSurface,
      windowLabel: nativeWindowLabel(),
      link: linkLabel,
      bound: windowStore !== null && windowStateId !== null,
      focus: focusState,
      ask: ask === null
        ? null
        : { act: ask.act, spec: ask.spec, chips: ask.chips },
      state: displayedFrame(),
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
      inspection:
        inspection === null
          ? null
          : { part: inspection.part, zoom: inspection.zoom, x: inspection.x, y: inspection.y },
      follow:
        activeSurface === "board" && instruments.board !== undefined
          ? ((instruments.board.presentation as BoardUI | null)?.followState() ?? null)
          : null,
      presentation:
        activeSurface === "board" && instruments.board !== undefined
          ? ((instruments.board.presentation as BoardUI | null)?.presentationStats() ?? null)
          : null,
      input: inputView(),
      reading: readingLayout,
    });
  }

  // the observation surface: the port's own product door, read-only
  shellWindow[PORT_SURFACE] = {
    readout,
  };

  return {
    attachViewport(element: HTMLElement | null) {
      viewport.attachElement(element);
      attachedViewport = element;
      applyReadingLayout(readingLayout);
    },
    openSpec,
    boot,
    subscribeFrame(observer) {
      frameObservers.add(observer);
      observer(displayedFrame());
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
    openPicker: () => {
      void refreshPickerContext();
      picker.open();
    },
    closePicker: () => picker.close(),
    pickerContext: () => pickerContextValue,
    whichKeyView: () => (whichKey !== null ? { pending: whichKeyPending, view: whichKey } : null),
    helpView: () => {
      if (!helpOpen) return null;
      const instrument = instruments[activeSurface];
      const view = instrument !== undefined ? instrument.inputAll() : null;
      if (view !== null && view.entries.length > 0) return view;
      return windowHost.allView();
    },
    subscribeWhichKey(observer) {
      whichKeyObservers.add(observer);
      observer(whichKey, helpOpen);
      return () => {
        whichKeyObservers.delete(observer);
      };
    },
    subscribePickerContext(observer) {
      pickerContextObservers.add(observer);
      observer(pickerContextValue);
      return () => {
        pickerContextObservers.delete(observer);
      };
    },
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
    closeInspection: () => {
      inspection = null;
      notifyInspection();
      notifyStrip();
    },
    inspectionWheel: (deltaY) => {
      if (inspection === null) return;
      inspection = { ...inspection, zoom: inspectionZoomWheel(inspection.zoom, deltaY) };
      notifyInspection();
    },
    inspectionPan: (x, y) => {
      if (inspection === null) return;
      inspection = { ...inspection, x, y };
      notifyInspection();
    },
    subscribeInspection(observer) {
      inspectionObservers.add(observer);
      observer(inspection);
      return () => {
        inspectionObservers.delete(observer);
      };
    },
    destroy() {
      if (readingLayoutTimer !== null) clearTimeout(readingLayoutTimer);
      window.removeEventListener("keydown", windowKeyDown);
      windowHost.dispose();
      unsubscribeBoardFrame();
      unsubscribeTasksFrame?.();
      for (const instrument of Object.values(instruments)) instrument.destroy();
    },
  };
}
