<script lang="ts">
  import { ViewerModel } from "./lib/board/viewmodel.ts";
  import {
    tauriInvoke,
    SourceUnavailableError,
    sourceFor,
    type Machine,
    type Location,
  } from "./lib/sources.ts";
  import { projectBoard, searchUnits, type SearchOccurrence } from "./lib/board/projection.ts";
  import { renderBoard } from "./lib/render.ts";
  import { hydrate } from "./lib/board/hydrate.ts";
  import { bindDiagramExpansion } from "./lib/overlay.ts";
  import { installCopyHandler } from "./lib/copydom.ts";
  import {
    collectTargets,
    highlight,
    inTextField,
    labelsFor,
    scrollBottom,
    scrollByLines,
    scrollByPages,
    scrollTop,
    showLabels,
    type SelectableTarget,
  } from "./lib/selectdom.ts";
  import {
    assignLabels,
    isTextKind,
    narrowStep,
    startNarrow,
    type NarrowState,
    type TargetKind,
  } from "./lib/select.ts";
  import { openContextMenu, type MenuAct } from "./lib/menu.ts";
  import { helpKeys as windowHelpKeys, type WindowKind } from "./lib/commands.ts";
  import Strip from "./components/Strip.svelte";
  import Picker from "./components/Picker.svelte";
  import Help from "./components/Help.svelte";
  import TasksWindow from "./components/TasksWindow.svelte";
  import AgentsWindow from "./components/AgentsWindow.svelte";
  import HistoryLog from "./components/HistoryLog.svelte";
  import { formatLocation, parseLocation } from "./lib/location.ts";
  import {
    annotatedBoard,
    publishPayload,
    publishAnnotations,
    type Draft,
  } from "./lib/annotations.ts";
  import {
    explainerEdit,
    loadKeptExplainer,
    saveKeptExplainer,
    clearKeptExplainer,
    submitExplainerEdit,
  } from "./lib/explainer.ts";
  import {
    applyDocument,
    cueCommand,
    deckNameFromBytes,
    jumpDeck,
    newDeck,
    stepDeck,
    waitingSlide,
    type DeckState,
  } from "./lib/deck.ts";
  import { trackerCoreMachine } from "./lib/sources.ts";
  import {
    WindowsClient,
    fieldsDiffer,
    type WindowState,
  } from "./lib/windows.ts";
  import {
    assignChips,
    entryMatches,
    jumpPickerEntries,
    shortLocation,
    type JumpEntry,
  } from "./lib/suite.ts";
  import { citationSpec, VIEWER_ROUTING, type ElsewhereAct } from "./lib/acts.ts";
  import { serverBase } from "./lib/tracker.ts";
  import { resolveRelative } from "./lib/resolve.ts";
  import { pickerProject, projectCandidates, rankedPicker, uniqueCandidates, type PickerItem } from "./lib/picker.ts";
  import { HistoryMode, marginaliaRuns, type MarginaliaRun } from "./lib/historymode.ts";
  import { ADD_CLOSE, ADD_OPEN, DEL_CLOSE, DEL_OPEN } from "./lib/history.ts";
  import { pageFact, type HistoryEntry } from "./lib/pages.ts";
  import { parseTaskSpec } from "./lib/location.ts";
  import { inspect } from "./lib/board/inspection.ts";
  import type { CopyMode } from "./lib/copy.ts";
  import { startWords, onWords, setWords, type WordsTable } from "./lib/words.ts";
  import {
    agentsAddressMap,
    agentsSections,
    attachCommand,
    mergeAgentSections,
    type AgentRow,
  } from "./lib/instruments/agents.ts";
  import { tasksSections } from "./lib/instruments/tasks.ts";
  import { laneLayout, type LaneEntry, type LaneLayout } from "./lib/instruments/lane.ts";
  import type { HistoryEntry as TimelineEntry } from "./lib/pages.ts";
  import type { RenderOutcome, ViewerState } from "./lib/board/viewmodel.ts";
  import type { Board } from "./lib/board/model.ts";
  import type { Diagnostic } from "./lib/board/errors.ts";
  import { homeTracker, knownTrackers, TrackerSource } from "./lib/tracker.ts";
  import type { ViewPage } from "./lib/pages.ts";
  import { formatLocation as stripFormat } from "./lib/location.ts";
  import {
    applyLook,
    lookSummary,
    resolveLook,
    setLookOverride,
    type LookSource,
  } from "./lib/look/store.ts";
  import { startLook } from "./lib/look/bridge.ts";
  import type { LookWeight } from "./lib/look/lookVars.ts";

  const model = new ViewerModel();

  // The full state feeds the chrome (header, search, diagnostics). The
  // content mount deliberately depends only on the board/location slices:
  // a copy-mode or theme toggle must not rebuild the document or lose
  // selection and scroll position.
  let view = $state.raw(model.snapshot);
  let board: Board | null = $state.raw(model.snapshot.board);
  let location: Location | null = $state.raw(model.snapshot.location);
  $effect(() => {
    const unsubscribe = model.subscribe((s) => {
      view = s;
      board = s.board;
      location = s.location;
    });
    return unsubscribe;
  });

  // The instruments' live half: the app's ONE feed lands its words in
  // the shell's table; windows read it here (one subscription per
  // process by the client's own design; the core allows more,
  // t-24@r23). A window with
  // no host starts it anyway - it resolves false and the lanes inject
  // through the inspection setter, test-only machinery.
  let wordsTable: WordsTable = $state.raw({});
  void startWords();
  $effect(() => {
    const unsubscribe = onWords((words) => {
      wordsTable = words;
    });
    return unsubscribe;
  });

  // The tasks instrument's live join: the rows name agent ids; their
  // terminal addresses come from the agents page (the record's own
  // join), read beside the pipeline page and on every fresh load.
  let agentAddresses: Record<string, string> = $state.raw({});
  $effect(() => {
    const current = view;
    if (current.instrument?.kind !== "tasks" || !current.location) return;
    const source = sourceFor(current.location.machine);
    if (!(source instanceof TrackerSource)) return;
    let stale = false;
    void source
      .agentsPage()
      .then((page) => {
        if (!stale) agentAddresses = agentsAddressMap(page);
      })
      .catch(() => {
        if (!stale) agentAddresses = {};
      });
    return () => {
      stale = true;
    };
  });

  // The `e` overlay: the task's activity (the lane and its events) or
  // the session's details, one panel, Escape leaves.
  let overlay: {
    title: string;
    facts: Array<{ label: string; value: string }>;
    lane: LaneLayout | null;
    events: LaneEntry[];
  } | null = $state.raw(null);

  // The lanes' test-open channel: the one startup open, riding the
  // same model.openInput the controls use. Test-only machinery
  // (t-24@r11) - the launch argument and the local-device path form
  // left the human surface with them, and a human run starts empty.
  // Each step logs a marker so the native evidence can tell "webview
  // never loaded" (no startup line), "invoke failed", and "open
  // failed" apart.
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
      await model.openInput(spec);
      const snap = model.snapshot;
      if (snap.board) {
        console.log(`[viewer] test open board opened: ${snap.board.content.length} blocks`);
      } else {
        const detail = snap.laneMessage ?? snap.errors.map((e) => e.message).join("; ");
        console.error(`[viewer] test open board FAILED: ${detail || "no board"}`);
      }
      return true;
    } catch (error) {
      console.error(`[viewer] test open board FAILED: ${String(error)}`);
      return true;
    }
  }

  // -- the window and its stored state (t-14 "The suite", t-24@r20) --------------
  //
  // One window, one window state: what this window shows, the part at
  // the reading height, its zoom, its drafts - on the tracker's
  // server, outside the record. Writes settle before sending; the
  // watcher applies every answer that is not this client's own echo.
  // A window with no tracker server runs unbacked, like every read
  // face without one.
  let windowStore: WindowsClient | null = null;
  let windowStateId: string | null = null;
  let windowWatch: { stop: () => void } | null = null;
  let windowLabel: string | null = null;
  let windowRetryTimer: ReturnType<typeof setTimeout> | null = null;
  // the mirrors this window applies and writes: a value already here
  // is never written again, so no-op settles bump no sequences
  let appliedLocation: string | null = null;
  let knownPart: string | null = null;
  let knownZoom: number | null = null;
  let knownDrafts: Draft[] = [];
  let windowDiscarded = false;
  let zoomWriteTimer: ReturnType<typeof setTimeout> | null = null;
  let partWriteTimer: ReturnType<typeof setTimeout> | null = null;

  /** The part at the reading height, as the window store names it: the select
   * path of the top-level block nearest the reading line. */
  function windowReadingPart(): string | null {
    if (!contentHost || !view.board) return null;
    const readingLine = contentHost.getBoundingClientRect().top + contentHost.clientHeight / 3;
    const blocks = Array.from(contentHost.querySelectorAll<HTMLElement>(":scope > .block"));
    let best: HTMLElement | null = null;
    let bestDistance = Number.MAX_SAFE_INTEGER;
    for (const block of blocks) {
      const rect = block.getBoundingClientRect();
      const distance = rect.top > readingLine ? rect.top - readingLine : readingLine - rect.bottom;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = block;
      }
    }
    if (!best) return null;
    const index = blocks.indexOf(best);
    return index >= 0 ? `/content/${index}` : null;
  }

  /** Scroll so the named part sits at the reading height. The block
   * may not exist for a few frames after an open; a handful of frames
   * are tried before giving up quietly. */
  function scrollToStatePart(part: string, tries = 20): void {
    if (!contentHost) return;
    const match = /^\/content\/(\d+)$/.exec(part);
    if (!match) return;
    const index = Number(match[1]);
    const block = contentHost.querySelectorAll<HTMLElement>(":scope > .block")[index];
    if (!block) {
      if (tries > 0) window.requestAnimationFrame(() => scrollToStatePart(part, tries - 1));
      return;
    }
    const readingLine = contentHost.getBoundingClientRect().top + contentHost.clientHeight / 3;
    const delta = block.getBoundingClientRect().top - readingLine;
    contentHost.scrollTop += delta;
  }

  function applyZoom(value: number): void {
    zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value));
    if (contentHost) contentHost.style.zoom = String(zoom);
  }

  /** The state as the window store carries it, applied locally: zoom, drafts,
   * the location to open, the part to scroll to. A discarded state
   * (changed 0, nothing in it) leaves the window empty - the human
   * opens anew or closes. */
  async function applyWindowState(state: WindowState): Promise<void> {
    if (state.changed === 0 && state.location === null) {
      windowDiscarded = true;
      console.log("[viewer] windows state discarded elsewhere; the window sits empty");
      return;
    }
    const nextZoom = state.zoom ?? 1;
    if (nextZoom !== knownZoom) {
      knownZoom = nextZoom;
      applyZoom(nextZoom);
    }
    const nextDrafts = state.drafts ?? [];
    if (JSON.stringify(nextDrafts) !== JSON.stringify(knownDrafts)) {
      knownDrafts = nextDrafts;
      drafts = nextDrafts;
    }
    if (state.location !== null && state.location !== appliedLocation) {
      appliedLocation = state.location;
      await model.openInput(state.location);
    }
    if (state.part !== knownPart) {
      knownPart = state.part;
      if (state.part !== null) scrollToStatePart(state.part);
    }
    // the link: a new target refreshes the HUD's name and watches it
    if (state.link !== linkTargetId) {
      void refreshLink(state.link);
    }
  }

  /** The window's state: the shell names the binding (the state
   * survives a reload); a first window creates its state, attaches
   * this device, applies what the state carries, and watches. The
   * one-time interim-draft move runs on the first start with the window store. */
  /** The retry for a window that started before any server existed:
   * one quiet timer, gone the moment the window binds. */
  function scheduleWindowRetry(): void {
    if (windowRetryTimer !== null || windowStateId !== null) return;
    windowRetryTimer = setTimeout(() => {
      windowRetryTimer = null;
      if (windowStateId !== null) return;
      void startStateWindow();
    }, 2000);
  }

  async function startStateWindow(): Promise<void> {
    let invoke: ReturnType<typeof tauriInvoke>;
    try {
      invoke = tauriInvoke();
    } catch (error) {
      if (error instanceof SourceUnavailableError) {
        console.log("[viewer] windows: no bridge; the window runs unbacked");
        return;
      }
      throw error;
    }
    windowStore = WindowsClient.open();
    if (windowStore === null) {
      // no server YET: the server setting appears once, typed in the
      // header - a window that started before it retries, and the
      // window store arrives without a restart
      console.log("[viewer] windows: no tracker reached yet; the window retries when one is");
      scheduleWindowRetry();
      void launchDefault(null);
      return;
    }
    try {
      console.log("[viewer] windows: bootstrap starts");
      const binding = (await invoke("window_binding")) as {
        label: string;
        state_id?: string | null;
      };
      windowLabel = binding.label;
      console.log(`[viewer] windows: binding ${binding.label} ${binding.state_id ?? "(none)"}`);
      let stateId = binding.state_id ?? null;
      if (stateId === null) {
        const fresh =
          typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : `w-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const created = await windowStore.create(fresh);
        stateId = created.id;
      }
      windowStateId = stateId;
      console.log(`[viewer] windows: state ${stateId}`);
      const attached = await windowStore.attach(stateId);
      console.log("[viewer] windows: attached");
      // the shell's binding carries the window store's address and this
      // device's id: the close detach needs them after the webview is
      // gone
      await invoke("bind_window_state", {
        stateId,
        server: windowStore.base.toString(),
        device: windowStore.device,
      });
      console.log("[viewer] windows: shell bound");
      knownZoom = attached.zoom;
      knownPart = attached.part;
      knownDrafts = attached.drafts ?? [];
      await applyWindowState(attached);
      console.log("[viewer] windows: state applied");
      void launchDefault(binding.state_id ?? null);
      windowWatch = windowStore.watch(stateId, (state) => {
        void applyWindowState(state);
      });
      console.log("[viewer] windows: watching");
      void windowStore.moveInterimDrafts();
      console.log(`[viewer] windows window bound: ${stateId}`);
    } catch (error) {
      console.error(`[viewer] windows window failed: ${String(error)}`);
      if (error instanceof Error && error.stack) {
        console.error(`[viewer] windows window stack: ${error.stack}`);
      }
      // a transient failure retries on the same quiet timer
      scheduleWindowRetry();
    }
  }
  const testOpening = openTestBoard();
  void startStateWindow();

  /** What a window with nothing to show opens (t-34@r6 change 4): the
   * launch window of a fresh start opens the home tracker's tasks; any
   * other empty window - or one with no tracker reached yet - opens
   * the picker, never a blank page or a field. The lanes' test open
   * wins over both. */
  let launchDefaultDone = false;
  async function launchDefault(boundStateId: string | null): Promise<void> {
    // once per window: the no-tracker retry must not reopen a picker
    // the human closed
    if (launchDefaultDone) return;
    launchDefaultDone = true;
    if (await testOpening) return;
    if (view.location !== null || windowDiscarded || pickerOpen) return;
    const home = homeTracker();
    if (windowLabel === "main" && boundStateId === null && home !== null) {
      await model.openInput(`${home}/tasks`);
      return;
    }
    openPicker();
  }

  // -- the window jump (t-14 "The suite") --------------------------------
  //
  // Labels on every visible suite window, and the picker listing
  // windows on this device, then detached ones: Enter focuses an
  // attached window, resumes a detached state here in a new window,
  // and ^x twice discards a state on every device. Focus moves only
  // here: every other act leaves focus alone. The keys ' and " are
  // placeholders (t-14, "Open and deferred").
  type JumpMode =
    | { kind: "labels"; chips: { window: string; chip: string }[]; typed: string }
    | { kind: "link"; chips: { window: string; chip: string }[]; typed: string }
    | {
        kind: "picker";
        entries: JumpEntry[];
        query: string;
        selected: number;
        armed: string | null;
      };
  let jumpMode = $state<JumpMode | null>(null);
  let jumpChipShown = $state<string | null>(null);
  let jumpNote = $state<string | null>(null);

  // the shell's broadcast door: the acting window assigns the chips,
  // every suite window shows its own
  (window as unknown as Record<string, unknown>).__suiteJumpShow = (chip: string) => {
    jumpChipShown = chip;
  };
  (window as unknown as Record<string, unknown>).__suiteJumpHide = () => {
    jumpChipShown = null;
  };

  function blurForJump(): void {
    (document.activeElement as HTMLElement | null)?.blur();
  }

  async function shellInvoke(): Promise<ReturnType<typeof tauriInvoke>> {
    const invoke = tauriInvoke();
    return invoke;
  }

  async function enterJumpLabels(): Promise<void> {
    blurForJump();
    jumpNote = null;
    try {
      const invoke = await shellInvoke();
      const windows = (await invoke("suite_windows")) as { label: string }[];
      const chips = assignChips(windows.map((window) => window.label));
      const list = [...chips].map(([windowLabel, chip]) => ({ window: windowLabel, chip }));
      if (list.length === 0) {
        jumpNote = "no suite windows to jump between";
        return;
      }
      await invoke("broadcast_jump_labels", { labels: list });
      jumpMode = { kind: "labels", chips: list, typed: "" };
    } catch (error) {
      jumpNote = `the labels failed: ${String(error)}`;
    }
  }

  async function enterJumpPicker(): Promise<void> {
    blurForJump();
    jumpNote = null;
    if (windowStore === null) {
      jumpNote = "no tracker server; the window store is not serving this window";
      return;
    }
    try {
      const invoke = await shellInvoke();
      const windows = (await invoke("suite_windows")) as {
        label: string;
        state_id?: string | null;
      }[];
      const states = await windowStore.list();
      const entries = jumpPickerEntries(
        windows.map((window) => ({ label: window.label, stateId: window.state_id ?? null })),
        states.map((state) => ({
          id: state.id,
          location: state.location,
          attachedHere: state.attachedHere,
        })),
      );
      jumpMode = { kind: "picker", entries, query: "", selected: 0, armed: null };
    } catch (error) {
      jumpNote = `the picker failed: ${String(error)}`;
    }
  }

  async function clearJumpBroadcast(): Promise<void> {
    try {
      const invoke = await shellInvoke();
      await invoke("broadcast_jump_labels_clear");
    } catch {
      // the labels clear with the mode; a failed broadcast is quiet
    }
  }

  function exitJump(): void {
    jumpMode = null;
    void clearJumpBroadcast();
  }

  async function jumpFocus(label: string): Promise<void> {
    const invoke = await shellInvoke();
    await invoke("focus_suite_window", { label });
  }

  async function refreshJumpPicker(mode: Extract<JumpMode, { kind: "picker" }>): Promise<void> {
    if (windowStore === null) return;
    const invoke = await shellInvoke();
    const windows = (await invoke("suite_windows")) as {
      label: string;
      state_id?: string | null;
    }[];
    const states = await windowStore.list();
    mode.entries = jumpPickerEntries(
      windows.map((window) => ({ label: window.label, stateId: window.state_id ?? null })),
      states.map((state) => ({
        id: state.id,
        location: state.location,
        attachedHere: state.attachedHere,
      })),
    );
    if (mode.selected >= mode.entries.length) mode.selected = Math.max(0, mode.entries.length - 1);
  }

  async function jumpAct(entry: JumpEntry): Promise<void> {
    const target = entry;
    exitJump();
    try {
      if (target.kind === "window" && target.windowLabel !== null) {
        // the jump: focus moves because the human asked
        await jumpFocus(target.windowLabel);
      } else {
        // resume a detached state here: a window opens, the human
        // asked for this one, so the activation request goes out with
        // it
        const invoke = await shellInvoke();
        const label = (await invoke("open_suite_window", { stateId: target.stateId })) as string;
        await jumpFocus(label);
      }
    } catch (error) {
      jumpNote = `the jump failed: ${String(error)}`;
    }
  }

  async function onJumpKey(event: KeyboardEvent): Promise<void> {
    const mode = jumpMode;
    if (mode === null) return;
    if (mode.kind === "labels" || mode.kind === "link") {
      const binding = mode.kind === "link";
      if (event.key === "Escape") {
        exitJump();
        return;
      }
      if (event.key === "Backspace") {
        mode.typed = mode.typed.slice(0, -1);
        return;
      }
      if (/^[a-z]$/.test(event.key)) {
        mode.typed += event.key;
        const matches = mode.chips.filter((entry) => entry.chip.startsWith(mode.typed));
        const landed = matches.length === 1 ? matches[0] : undefined;
        if (matches.length === 0) {
          exitJump();
          return;
        }
        if (matches.length === 1 && landed && landed.chip === mode.typed) {
          const target = landed.window;
          exitJump();
          try {
            if (binding) {
              await linkToWindow(target);
            } else {
              // the jump: focus moves because the human asked
              await jumpFocus(target);
            }
          } catch (error) {
            jumpNote = `${binding ? "the link" : "the jump"} failed: ${String(error)}`;
          }
        }
      }
      return;
    }
    // the picker
    if (event.key === "Escape") {
      exitJump();
      return;
    }
    if (event.ctrlKey && event.key === "x") {
      const entry = mode.entries[mode.selected];
      if (!entry) return;
      if (mode.armed === entry.stateId) {
        if (windowStore !== null) {
          await windowStore.discard(entry.stateId);
          await refreshJumpPicker(mode);
        }
        mode.armed = null;
      } else {
        mode.armed = entry.stateId;
      }
      return;
    }
    if (event.key === "ArrowUp" || (event.key === "k" && !event.ctrlKey)) {
      event.preventDefault();
      mode.armed = null;
      mode.selected = Math.max(0, mode.selected - 1);
      return;
    }
    if (event.key === "ArrowDown" || (event.key === "j" && !event.ctrlKey)) {
      event.preventDefault();
      mode.armed = null;
      mode.selected = Math.min(mode.entries.length - 1, mode.selected + 1);
      return;
    }
    if (/^[a-z0-9/]$/.test(event.key) && !event.ctrlKey) {
      mode.query += event.key;
      return;
    }
    if (event.key === "Backspace") {
      mode.query = mode.query.slice(0, -1);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const alive = mode.entries.filter((entry) => entryMatches(entry, mode.query));
      const entry = alive[mode.selected];
      if (entry) await jumpAct(entry);
    }
  }

  function visibleJumpEntries(mode: Extract<JumpMode, { kind: "picker" }>): JumpEntry[] {
    return mode.entries.filter((entry) => entryMatches(entry, mode.query));
  }

  // -- the link (t-14 "The suite") ---------------------------------------
  //
  // Per device, one per window, many windows may link to one target,
  // only between windows attached on this device - the window store enforces
  // the shape and lapses the link when its target detaches. A linked
  // window shows its target; space l links by labels, space u
  // unlinks; links never travel devices.
  let linkTargetId = $state<string | null>(null);
  let linkTargetName = $state<string | null>(null);
  let linkTargetWindowLabel = $state<string | null>(null);
  let linkWatch: { stop: () => void } | null = null;

  async function refreshLink(targetId: string | null): Promise<void> {
    linkTargetId = targetId;
    linkWatch?.stop();
    linkWatch = null;
    if (targetId === null || windowStore === null) {
      linkTargetName = null;
      linkTargetWindowLabel = null;
      return;
    }
    try {
      const target = await windowStore.getState(targetId);
      linkTargetName = shortLocation(target.location);
      const invoke = await shellInvoke();
      const windows = (await invoke("suite_windows")) as {
        label: string;
        state_id?: string | null;
      }[];
      linkTargetWindowLabel =
        windows.find((window) => window.state_id === targetId)?.label ?? null;
      // the target's own changes follow: its name in this window's
      // HUD moves when the target moves, without any polling
      linkWatch = windowStore.watch(targetId, (targetState) => {
        linkTargetName = shortLocation(targetState.location);
      });
    } catch {
      // the target left the window store: the link lapses server-side; this
      // window shows no target until a new link is set
      linkTargetName = null;
      linkTargetWindowLabel = null;
    }
  }

  async function enterLinkLabels(): Promise<void> {
    blurForJump();
    jumpNote = null;
    if (windowStore === null || windowStateId === null || windowDiscarded) {
      jumpNote = "no tracker server; the window store is not serving this window";
      return;
    }
    try {
      const invoke = await shellInvoke();
      const windows = (await invoke("suite_windows")) as {
        label: string;
        state_id?: string | null;
      }[];
      const chips = assignChips(windows.map((window) => window.label));
      const list = [...chips].map(([windowLabel, chip]) => ({ window: windowLabel, chip }));
      await invoke("broadcast_jump_labels", { labels: list });
      jumpMode = { kind: "link", chips: list, typed: "" };
    } catch (error) {
      jumpNote = `the labels failed: ${String(error)}`;
    }
  }

  async function linkToWindow(windowLabel: string): Promise<void> {
    if (windowStore === null || windowStateId === null) return;
    const invoke = await shellInvoke();
    const windows = (await invoke("suite_windows")) as {
      label: string;
      state_id?: string | null;
    }[];
    const target = windows.find((window) => window.label === windowLabel);
    if (!target?.state_id) {
      jumpNote = `no state bound to ${windowLabel} yet`;
      return;
    }
    try {
      await windowStore.setLink(windowStateId, target.state_id);
      exitJump();
    } catch (error) {
      jumpNote = `the link refused: ${String(error)}`;
      exitJump();
    }
  }

  async function unlinkHere(): Promise<void> {
    if (windowStore === null || windowStateId === null || windowDiscarded) return;
    try {
      await windowStore.clearLink(windowStateId);
    } catch (error) {
      jumpNote = `the unlink refused: ${String(error)}`;
    }
  }

  // -- the two acts (t-14 "The suite") -----------------------------------
  //
  // Enter (or a click) is the primary act; o (or shift- or
  // middle-click) opens elsewhere through the link. With no link,
  // opening elsewhere asks once - labels on visible windows, n new,
  // / a detached one - and the answer becomes the link; i opens
  // elsewhere once without changing the link; w always opens a new
  // window. The viewer's Enter follows here (the declaration is the
  // instrument's: acts.ts); the tasks and agents instruments (t-18)
  // declare their own and arrive without rework.
  type AskOnce = {
    act: ElsewhereAct;
    spec: string;
    chips: { window: string; chip: string }[];
    typed: string;
    detached: JumpEntry[];
    detachedSelected: number;
    showDetached: boolean;
  };
  let askOnce = $state<AskOnce | null>(null);

  async function openElsewhere(spec: string, act: ElsewhereAct): Promise<void> {
    if (windowStore === null || windowStateId === null || windowDiscarded) {
      jumpNote = "no tracker server; the window store is not serving this window";
      return;
    }
    if (act === "link" && linkTargetWindowLabel !== null) {
      // through the link: the target opens it here, focus never moves
      const target = linkTargetWindowLabel;
      await openInWindow(target, spec);
      return;
    }
    if (act === "new") {
      await openInNewWindow(spec, false);
      return;
    }
    // no link (or a once act): ask once - unless this is the only
    // window, where there is nothing to ask: a new window opens and,
    // for o, becomes the link (t-34@r6 change 3)
    blurForJump();
    jumpNote = null;
    try {
      const invoke = await shellInvoke();
      const windows = (await invoke("suite_windows")) as { label: string }[];
      if (windows.length <= 1) {
        await openInNewWindow(spec, act === "link");
        return;
      }
      const chips = assignChips(windows.map((window) => window.label));
      const list = [...chips].map(([windowLabel, chip]) => ({ window: windowLabel, chip }));
      await invoke("broadcast_jump_labels", { labels: list });
      const states = await windowStore.list();
      const detached = states
        .filter((state) => !state.attachedHere)
        .map((state) => ({
          kind: "detached" as const,
          chip: null,
          windowLabel: null,
          stateId: state.id,
          name: shortLocation(state.location),
        }));
      askOnce = {
        act,
        spec,
        chips: list,
        typed: "",
        detached,
        detachedSelected: 0,
        showDetached: false,
      };
    } catch (error) {
      jumpNote = `the ask failed: ${String(error)}`;
    }
  }

  async function openInWindow(label: string, spec: string): Promise<void> {
    const invoke = await shellInvoke();
    await invoke("open_in_window", { label, location: spec });
  }

  /** A new window shows a new window state: the caller creates the
   * state (so it can link to it) and the window's bootstrap attaches
   * what it finds. The optional answer of an ask-once becomes the
   * link when the act was o. */
  async function openInNewWindow(spec: string, becomeLink: boolean): Promise<void> {
    if (windowStore === null || windowStateId === null) return;
    const fresh =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `w-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    await windowStore.create(fresh);
    const invoke = await shellInvoke();
    const label = (await invoke("open_suite_window", { stateId: fresh })) as string;
    await invoke("open_in_window", { label, location: spec });
    if (becomeLink) {
      await windowStore.setLink(windowStateId, fresh);
    }
  }

  async function askOnceAnswer(location: "window" | "new" | "detached", pick?: JumpEntry): Promise<void> {
    const asking = askOnce;
    if (asking === null) return;
    askOnce = null;
    void clearJumpBroadcast();
    const becomeLink = asking.act === "link";
    try {
      if (location === "window") {
        const chip = asking.typed;
        const match = asking.chips.find((entry) => entry.chip === chip);
        if (!match) return;
        await openInWindow(match.window, asking.spec);
        if (becomeLink && windowStore !== null && windowStateId !== null) {
          const invoke = await shellInvoke();
          const windows = (await invoke("suite_windows")) as {
            label: string;
            state_id?: string | null;
          }[];
          const target = windows.find((window) => window.label === match.window);
          if (target?.state_id) await windowStore.setLink(windowStateId, target.state_id);
        }
        return;
      }
      if (location === "new") {
        await openInNewWindow(asking.spec, becomeLink);
        return;
      }
      if (pick) {
        // a detached state: it resumes here in a new window and opens
        // the citation there
        const invoke = await shellInvoke();
        const label = (await invoke("open_suite_window", { stateId: pick.stateId })) as string;
        await invoke("open_in_window", { label, location: asking.spec });
        if (becomeLink && windowStore !== null && windowStateId !== null) {
          await windowStore.setLink(windowStateId, pick.stateId);
        }
      }
    } catch (error) {
      jumpNote = `the open failed: ${String(error)}`;
    }
  }

  async function onAskOnceKey(event: KeyboardEvent): Promise<void> {
    const asking = askOnce;
    if (asking === null) return;
    if (asking.showDetached) {
      if (event.key === "Escape") {
        askOnce = null;
        void clearJumpBroadcast();
        return;
      }
      if (event.key === "ArrowUp" || event.key === "k") {
        asking.detachedSelected = Math.max(0, asking.detachedSelected - 1);
        return;
      }
      if (event.key === "ArrowDown" || event.key === "j") {
        asking.detachedSelected = Math.min(asking.detached.length - 1, asking.detachedSelected + 1);
        return;
      }
      if (event.key === "Enter") {
        const alive = asking.detached;
        await askOnceAnswer("detached", alive[asking.detachedSelected]);
        return;
      }
      return;
    }
    if (event.key === "Escape") {
      askOnce = null;
      void clearJumpBroadcast();
      return;
    }
    if (event.key === "/") {
      asking.showDetached = true;
      return;
    }
    if (event.key === "n") {
      await askOnceAnswer("new");
      return;
    }
    if (event.key === "Backspace") {
      asking.typed = asking.typed.slice(0, -1);
      return;
    }
    if (/^[a-z]$/.test(event.key)) {
      asking.typed += event.key;
      const matches = asking.chips.filter((entry) => entry.chip.startsWith(asking.typed));
      const landed = matches.length === 1 ? matches[0] : undefined;
      if (matches.length === 0) {
        askOnce = null;
        void clearJumpBroadcast();
        return;
      }
      if (matches.length === 1 && landed && landed.chip === asking.typed) {
        await askOnceAnswer("window");
      }
    }
  }

  /** The elsewhere half of a citation act: the spec the target window
   * opens, routed per the two acts. */
  async function openSpecElsewhere(spec: string, act: ElsewhereAct): Promise<void> {
    await openElsewhere(spec, act);
  }

  // the citation click's routing: hydrate signals every citation
  // activation; the shell routes it per the instrument's declaration -
  // the viewer follows here, a shifted or middle click opens
  // elsewhere through the link (the answer becoming the link when
  // there is none)
  window.addEventListener("suite-open-citation", ((event: CustomEvent<{ machine: string; path: string; shifted: boolean }>) => {
    const spec = citationSpec(event.detail, event.detail.path);
    if (spec === null) return;
    if (event.detail.shifted || VIEWER_ROUTING.primary === "link") {
      void openElsewhere(spec, "link");
      return;
    }
    void model.openInput(spec);
  }) as EventListener);

  /** A settled write: the location this window opened, with the
   * reading position reset - new content, no part yet. */
  $effect(() => {
    const current = location;
    if (!current || windowStore === null || windowStateId === null || windowDiscarded) return;
    const spec = formatLocation(current);
    if (spec === appliedLocation) return;
    appliedLocation = spec;
    knownPart = null;
    void windowStore.patch(windowStateId, { location: spec, part: null }).catch(() => {
      // an unreachable window store leaves the window readable; the next
      // settled write retries
    });
  });

  /** The reading height settles while reading: written when it stops
   * moving, never per scroll frame (t-24@r20). */
  function schedulePartWrite(): void {
    if (windowStore === null || windowStateId === null || windowDiscarded) return;
    if (partWriteTimer !== null) clearTimeout(partWriteTimer);
    partWriteTimer = setTimeout(() => {
      partWriteTimer = null;
      if (windowStore === null || windowStateId === null || windowDiscarded) return;
      const part = windowReadingPart();
      if (part === null || part === knownPart) return;
      knownPart = part;
      void windowStore.patch(windowStateId, { part }).catch(() => {});
    }, 600);
  }

  $effect(() => {
    const host = contentHost;
    if (!host) return;
    const onScroll = (): void => schedulePartWrite();
    host.addEventListener("scroll", onScroll);
    return () => host.removeEventListener("scroll", onScroll);
  });

  let query = $state("");
  let activeMatch = $state(0);
  let searchOpen = $state(false);
  let searchInput: HTMLInputElement | undefined = $state(undefined);
  let contentHost: HTMLElement | undefined = $state(undefined);
  let instrumentHost: HTMLElement | undefined = $state(undefined);

  /** The element the motion keys scroll: the instrument when one is
   * open, the document otherwise. */
  function scroller(): HTMLElement {
    return (view.instrument ? instrumentHost : contentHost) ?? contentHost!;
  }

  function openSearch(): void {
    searchOpen = true;
    window.requestAnimationFrame(() => searchInput?.focus());
  }

  function closeSearch(): void {
    searchOpen = false;
    returnKeys();
  }

  /** Focus goes back to the document whenever a temporary field closes
   * or a button has been clicked, so the window's keys work at once
   * (t-34@r6 change 7: the only text fields are temporary). */
  function returnKeys(): void {
    const active = document.activeElement;
    if (active instanceof HTMLElement && active !== document.body) active.blur();
  }

  function onButtonClick(event: MouseEvent): void {
    const target = event.target;
    if (!(target instanceof Element)) return;
    // only the window's own buttons (the strip, the document, the
    // rows): an overlay or dialog keeps its focus, it owns its keys
    const button = target.closest("button");
    if (button && button.closest(".strip, main.content") && !button.closest(".draft-editor")) button.blur();
  }

  // Key-driven chrome: the select-first keymap (t-14 "Interaction")
  // plus the standing search and copy-mode keys. No visible mode
  // buttons: every mode is a key, and ? lists them on demand.
  type KeyMode = "normal" | "labels" | "gg";
  let keyMode = $state<KeyMode>("normal");
  let selected: SelectableTarget | null = $state(null);
  let targets: SelectableTarget[] = [];
  let labels: string[] = [];
  let narrow: NarrowState | null = null;
  let hideLabels: (() => void) | null = null;
  let closeMenu: (() => void) | null = null;
  let helpOpen = $state(false);

  function exitLabelMode(): void {
    if (hideLabels) hideLabels();
    hideLabels = null;
    narrow = null;
    keyMode = "normal";
  }

  function enterLabelMode(): void {
    if (keyMode === "labels") return;
    targets = collectTargets(document.body);
    if (targets.length === 0) return;
    labels = labelsFor(targets);
    narrow = startNarrow(targets.length);
    if (hideLabels) hideLabels();
    hideLabels = showLabels(targets, labels);
    keyMode = "labels";
  }

  function selectTarget(index: number | null): void {
    const previous = selected;
    if (index === null || targets[index] === undefined) {
      selected = null;
      highlight(null, previous);
      return;
    }
    selected = targets[index] ?? null;
    highlight(selected, previous);
  }

  /** The tasks row's live shape: the agents' word joined through the
   * agents page's addresses. No word, no shape - colour is spent only
   * on state. */
  function taskRowShapes(agents: string): Array<{ id: string; word: string | null }> {
    return agents
      .split(/\s+/)
      .filter((id) => id.length > 0)
      .map((id) => {
        const address = agentAddresses[id];
        const state = address ? (wordsTable[address]?.state ?? null) : null;
        return { id, word: state };
      });
  }

  /** The agents window spans trackers (t-34@r6 change 5): every other
   * tracker this device has reached contributes its agents page. */
  let otherAgentPages = $state.raw<Array<{ project: string; page: ViewPage }>>([]);

  $effect(() => {
    const current = view;
    if (current.instrument?.kind !== "agents" || !current.location) {
      otherAgentPages = [];
      return;
    }
    const here = current.location.machine.startsWith("tracker:") ? current.location.machine.slice("tracker:".length) : null;
    const others = Object.keys(knownTrackers()).filter((name) => name !== here);
    void Promise.all(
      others.map(async (project) => {
        const source = sourceFor(`tracker:${project}`);
        if (!(source instanceof TrackerSource)) return null;
        try {
          return { project, page: await source.instrumentPage("agents") };
        } catch {
          return null;
        }
      }),
    ).then((pages) => {
      otherAgentPages = pages.filter((entry): entry is { project: string; page: ViewPage } => entry !== null);
    });
  });

  /** The agents window's sections: this tracker's and every other
   * reached tracker's, merged by word, each row naming its project. */
  let agentSections = $derived.by(() => {
    const current = view;
    if (current.instrument?.kind !== "agents") return [];
    const here = current.location?.machine.startsWith("tracker:") ? current.location.machine.slice("tracker:".length) : "";
    return mergeAgentSections([
      { project: here, sections: agentsSections(current.instrument.page, wordsTable) },
      ...otherAgentPages.map(({ project, page }) => ({ project, sections: agentsSections(page, wordsTable) })),
    ]);
  });

  /** The tasks instrument's sections. */
  let taskSections = $derived.by(() => {
    const current = view;
    if (current.instrument?.kind !== "tasks") return [];
    return tasksSections(current.instrument.page);
  });

  function copyText(text: string | null, what: string): void {
    if (!text) return;
    if (bridgePresent) void bridgeWrite(text);
    else void navigator.clipboard?.writeText(text).catch(() => {});
    copyNote = `${what} copied`;
    window.setTimeout(() => (copyNote = null), 1200);
  }
  let copyNote = $state<string | null>(null);

  /** The branch a workspace sits on, read from its own HEAD through
   * the tracker's machine - a plain file read, for diffing from the
   * primary checkout (t-14). Null when the read refuses: no invented
   * branch. */
  async function workspaceBranch(workspace: string | null): Promise<string | null> {
    if (!workspace) return null;
    try {
      const loaded = await sourceFor("tracker-machine").loadText({
        machine: "tracker-machine",
        path: `${workspace}/.git/HEAD`,
      });
      const match = /ref: refs\/heads\/(\S+)/.exec(loaded.text.trim());
      return match?.[1] ?? null;
    } catch {
      return null;
    }
  }

  function showActivityOverlay(task: string): void {
    const current = view;
    if (!current.location) return;
    const source = sourceFor(current.location.machine);
    if (!(source instanceof TrackerSource)) return;
    overlay = { title: task, facts: [], lane: null, events: [] };
    void source
      .history(task)
      .then((page) => {
        const entries: LaneEntry[] = (page.entries ?? []).map((entry: TimelineEntry) => ({
          at: entry.at,
          kind: entry.event,
          label:
            entry.message ??
            entry.grounds ??
            entry.explanation ??
            entry.event,
        }));
        const now = Math.floor(Date.now() / 1000);
        overlay = {
          title: task,
          facts: [],
          lane: laneLayout(entries, now),
          events: [...entries].reverse(),
        };
      })
      .catch((error) => {
        overlay = {
          title: task,
          facts: [],
          lane: null,
          events: [],
        };
        copyNote = String(error instanceof SourceUnavailableError ? error.message : error);
      });
  }

  function showDetailsOverlay(row: AgentRow): void {
    const facts: Array<{ label: string; value: string }> = [
      { label: "incarnation", value: row.incarnation },
    ];
    if (row.harness) facts.push({ label: "harness", value: row.harness });
    if (row.workspace) facts.push({ label: "workspace", value: row.workspace });
    if (row.lastCommit) facts.push({ label: "last commit", value: row.lastCommit });
    if (row.tasks.length > 0) facts.push({ label: "tasks", value: row.tasks.join(" ") });
    overlay = { title: row.incarnation, facts, lane: null, events: [] };
  }

  function instrumentRowActs(target: SelectableTarget, open: () => void): MenuAct[] {
    const kind = target.element.getAttribute("data-instrument");
    const task = target.element.getAttribute("data-task");
    const acts: MenuAct[] = [];
    if (kind === "tasks") {
      acts.push(...openActs(target.element.getAttribute("data-open-path"), "board"));
      if (task) {
        acts.push({
          key: "e",
          label: "Show activity (the lane and the events)",
          onAct: () => showActivityOverlay(task),
        });
        acts.push(copyAct("y", "Copy reference", () => task, "the reference"));
      }
      return acts;
    }
    // the agents row: open, details, the copies (t-14: the attach
    // command, the workspace path, the branch)
    acts.push({
      key: "Enter",
      label: task ? "open the task's board through the link" : "no task owned - nothing to open",
      onAct: task ? open : () => {},
    });
    acts.push({ key: "e", label: "the session's details", onAct: () => {
      const row = rowFromElement(target.element);
      if (row) showDetailsOverlay(row);
    } });
    const attach = attachCommand(rowFromElement(target.element) ?? {
      incarnation: "",
      address: target.element.getAttribute("data-address"),
      harness: null,
      tasks: [],
      workspace: null,
      lastCommit: null,
      word: null,
    });
    const workspace = target.element.getAttribute("data-workspace");
    acts.push({
      key: "a",
      label: "copy the attach command",
      onAct: () => copyText(attach, "the attach command"),
    });
    acts.push({
      key: "w",
      label: "copy the workspace path",
      onAct: () => copyText(workspace, "the workspace path"),
    });
    acts.push({
      key: "b",
      label: "copy the branch",
      onAct: () => {
        const row = rowFromElement(target.element);
        if (!row) return;
        void workspaceBranch(row.workspace).then((branch) => {
          if (branch) copyText(branch, "the branch");
          else copyNote = "no branch readable for this workspace";
        });
      },
    });
    return acts;
  }

  /** The row's data, back from its element: the acts read what the
   * render wrote. */
  function rowFromElement(element: HTMLElement): AgentRow | null {
    const get = (name: string) => element.getAttribute(name);
    const incarnation = get("data-incarnation");
    if (!incarnation) return null;
    const tasks = (get("data-tasks") ?? "").split(/\s+/).filter((t) => t.length > 0);
    const at = get("data-at");
    const state = get("data-word");
    return {
      incarnation,
      address: get("data-address"),
      harness: get("data-harness"),
      tasks,
      workspace: get("data-workspace"),
      lastCommit: get("data-last-commit"),
      word:
        at === null
          ? null
          : {
              state: state ?? null,
              app: get("data-app"),
              msg: get("data-msg"),
              progress: get("data-progress"),
              title: get("data-title"),
              finished: get("data-finished"),
              at: Number(at),
            },
    };
  }

  /** A row selects on click: the pointer's own path to the primary
   * act, beside the label flow. */
  function selectInstrumentRow(click: MouseEvent): void {
    const eventTarget = click.target as HTMLElement | null;
    const element = eventTarget?.closest<HTMLElement>(".instrument-row");
    if (!element) return;
    targets = collectTargets(document.body);
    const index = targets.findIndex((candidate) => candidate.element === element);
    if (index >= 0) selectTarget(index);
  }

  /** Enter in an instrument goes through the link (t-14: tasks and
   * agents have no here). */
  /** Enter on a row (t-34@r6 change 3): the board opens through the
   * link - in a new, linked window when this is the only one. A window
   * with no shell or window store is the only window there is, so the
   * board opens in it. With no row chosen, the first row is chosen. */
  function instrumentEnter(): void {
    const target = selected;
    if (!target || !target.element.classList.contains("instrument-row")) {
      moveRowCursor(1);
      return;
    }
    const path = target.element.getAttribute("data-open-path");
    const current = view.location;
    if (!path || !current) return;
    // a row from another project names its full location
    const elsewhere = target.element.getAttribute("data-open-location");
    const spec = elsewhere ?? citationSpec(current, path);
    if (windowStore === null || windowStateId === null || windowDiscarded || spec === null) {
      if (elsewhere) void model.openInput(elsewhere);
      else void model.open({ machine: current.machine, path });
      return;
    }
    void openElsewhere(spec, "link");
  }

  /** j k in a list window: a row cursor (t-34@r6 change 2), the view
   * following it; the first press chooses the first row. */
  function moveRowCursor(delta: number): void {
    const rows = Array.from(instrumentHost?.querySelectorAll<HTMLElement>(".instrument-row") ?? []);
    if (rows.length === 0) return;
    const at = selected ? rows.indexOf(selected.element) : -1;
    const next = at < 0 ? (delta > 0 ? 0 : rows.length - 1) : Math.min(rows.length - 1, Math.max(0, at + delta));
    const row = rows[next];
    if (!row) return;
    targets = collectTargets(document.body);
    labels = labelsFor(targets);
    const index = targets.findIndex((candidate) => candidate.element === row);
    if (index >= 0) selectTarget(index);
    row.scrollIntoView({ block: "nearest" });
  }

  /** The select path of a rendered part from its block id: b3 is
   * /content/3, b3.0 is /content/3/group/content/0. */
  function partPath(element: HTMLElement): string | null {
    const id = element.getAttribute("data-block-id");
    const match = id ? /^b(\d+(?:\.\d+)*)$/.exec(id) : null;
    if (!match || match[1] === undefined) return null;
    return match[1].split(".").map((index, depth) => (depth === 0 ? `/content/${index}` : `/group/content/${index}`)).join("");
  }

  /** The citation of a part: the window's location, then its path. */
  function partCitation(element: HTMLElement): string {
    const where = view.location ? formatLocation(view.location) : "";
    const path = partPath(element);
    return path ? `${where}:${path}` : where;
  }

  function copyAct(key: string, label: string, text: () => string | null, what: string): MenuAct {
    return { key, label, onAct: () => copyText(text(), what) };
  }

  /** o i w on a location: through the link, once, a new window. */
  function openActs(path: string | null, noun: string): MenuAct[] {
    const go = (how: "link" | "once" | "new"): void => {
      const current = view.location;
      if (!current || !path) return;
      const spec = citationSpec(current, path);
      if (spec !== null) void openSpecElsewhere(spec, how);
    };
    return [
      { key: "o", label: `Open ${noun} (through the link)`, onAct: () => go("link") },
      { key: "i", label: `Open ${noun} elsewhere once`, onAct: () => go("once") },
      { key: "w", label: `Open ${noun} in a new window`, onAct: () => go("new") },
    ];
  }

  function annotateAct(element: HTMLElement): MenuAct {
    return {
      key: "a",
      label: "Annotate after this part",
      onAct: () => {
        actElement = element;
        beginDraft();
      },
    };
  }

  function historyAct(element: HTMLElement): MenuAct {
    return { key: "t", label: "History of this part", onAct: () => void historyOfPart(element) };
  }

  /** t on a part: history, tracking that part (H L walk its changes). */
  async function historyOfPart(element: HTMLElement): Promise<void> {
    if (!view.board) return;
    const blocks = contentBlocks();
    const top = blocks.find((block) => block === element || block.contains(element));
    const part = top ? view.board.content[blocks.indexOf(top)] : undefined;
    if (!part) return;
    if (!historyMode) await enterHistory();
    if (!historyMode) return;
    historyMode.track(partKeyForAnchor(part));
    top?.scrollIntoView({ block: "center", behavior: "auto" });
  }

  /** The menu of a selected part, by kind (round 16's KINDS). */
  function actsFor(target: SelectableTarget): MenuAct[] {
    const element = target.element;
    const open = (): void => {
      const path = element.getAttribute("data-open-path");
      const location = view.location;
      if (path && location) void model.open({ machine: location.machine, path });
    };
    if (element.classList.contains("instrument-row")) {
      return instrumentRowActs(target, open);
    }
    const onTaskBoard = view.location?.machine.startsWith("tracker:") === true && Boolean(view.board?.task);
    const canAnnotate = Boolean(view.board) && (onTaskBoard || view.location?.machine === "tracker-machine");
    const cite = copyAct("y", "Copy citation", () => partCitation(element), "the citation");
    switch (target.kind) {
      case "citation": {
        const path = element.getAttribute("data-open-path");
        return [
          { key: "f", label: "Follow here", onAct: open },
          ...openActs(path, "the cited revision"),
          copyAct("y", "Copy reference", () => path, "the reference"),
        ];
      }
      case "incorporation": {
        const path = element.querySelector<HTMLElement>("[data-open-path]")?.getAttribute("data-open-path") ?? null;
        const acts: MenuAct[] = [];
        if (canAnnotate) acts.push(annotateAct(element));
        acts.push({ key: "e", label: "Show cited components", onAct: () => void expandIncorporation(element) });
        if (path && view.location) {
          const location = view.location;
          acts.push({ key: "f", label: "Follow the source here", onAct: () => void model.open({ machine: location.machine, path }) });
          acts.push(openActs(path, "the source")[0] as MenuAct);
        }
        if (onTaskBoard) acts.push(historyAct(element));
        acts.push(cite);
        return acts;
      }
      case "figure":
        return [
          { key: "x", label: "Enlarge", onAct: () => element.querySelector<HTMLElement>("img, .figure-frame")?.click() },
          ...(canAnnotate ? [annotateAct(element)] : []),
          copyAct("y", "Copy location", () => element.getAttribute("data-figure-input"), "the figure's location"),
        ];
      case "diagram":
        return [{ key: "x", label: "Enlarge", onAct: () => element.click() }];
      case "group":
        return [
          { key: "z", label: element.classList.contains("closed") ? "Unfold" : "Fold", onAct: () => element.classList.toggle("closed") },
          ...(canAnnotate ? [annotateAct(element)] : []),
          ...(onTaskBoard ? [historyAct(element)] : []),
          cite,
        ];
      case "text":
      case "annotation":
      case "unknown":
        return [
          ...(canAnnotate ? [annotateAct(element)] : []),
          ...(onTaskBoard ? [historyAct(element)] : []),
          cite,
          copyAct("c", "Copy the text", () => target.text, "the text"),
        ];
      default:
        return [{ key: "Enter", label: "Activate", onAct: () => element.click() }];
    }
  }

  function documentActs(): MenuAct[] {
    const acts: MenuAct[] = [
      { key: "s", label: "labels on what can be selected", onAct: () => enterLabelMode() },
      { key: "/", label: "search", onAct: () => openSearch() },
      { key: "?", label: "help", onAct: () => (helpOpen = true) },
      { key: "+", label: "zoom in", onAct: () => zoomBy(0.1) },
      { key: "-", label: "zoom out", onAct: () => zoomBy(-0.1) },
      { key: "=", label: "zoom reset", onAct: () => zoomReset() },
      {
        key: "m",
        label: view.copyMode === "plain" ? "copy plain (now source)" : "copy source (now plain)",
        onAct: () => model.setCopyMode(view.copyMode === "plain" ? "source" : "plain"),
      },
    ];
    return acts;
  }

  function openTargetMenu(target: SelectableTarget, at: { x: number; y: number }): void {
    if (closeMenu) closeMenu();
    const selection = selected;
    const head =
      target.element.classList.contains("instrument-row")
        ? (target.element.getAttribute("data-task") ?? target.element.getAttribute("data-incarnation") ?? "")
        : target.kind === "citation"
          ? (target.element.getAttribute("data-open-path") ?? "citation")
          : `${partPath(target.element) ?? target.kind}  ${target.text.slice(0, 48)}`;
    closeMenu = openContextMenu({
      ...at,
      acts: actsFor(target),
      title: head.trim(),
      enterFirst: true,
      onClose: () => {
        closeMenu = null;
        selectTarget(null);
        void selection;
      },
    });
  }

  function openDocumentMenu(at: { x: number; y: number }): void {
    if (closeMenu) closeMenu();
    closeMenu = openContextMenu({
      ...at,
      acts: documentActs(),
      onClose: () => {
        closeMenu = null;
      },
    });
  }

  function moveSelection(delta: number): void {
    targets = collectTargets(document.body);
    labels = labelsFor(targets);
    const current = selected ? targets.findIndex((t) => t.element === selected?.element) : -1;
    const next = Math.min(Math.max(current + delta, 0), targets.length - 1);
    if (targets[next] !== undefined) {
      selected = targets[next] ?? null;
      highlight(selected, current >= 0 ? (targets[current] ?? null) : null);
    }
  }

  // The picker (t-14): space space opens it; match quality before
  // recency, the open item last; a result shows only kind and name;
  // a typed location always resolves even when nothing enumerated it.
  let pickerOpen = $state(false);
  let pickerQuery = $state("");
  let pickerActive = $state(0);

  // the project the picker is about, and that project's boards: loaded
  // from its tasks page when the picker opens or the typed project
  // changes (one project at a time, t-34@r6 change 5)
  let pickerProjectItems = $state.raw<PickerItem[]>([]);
  let pickerLoadedProject: string | null = null;
  let pickerProjectName = $derived(
    pickerProject(
      pickerQuery,
      view.location?.machine.startsWith("tracker:") ? view.location.machine.slice("tracker:".length) : homeTracker(),
      Object.keys(knownTrackers()),
    ),
  );

  $effect(() => {
    const project = pickerProjectName;
    if (!pickerOpen || project === null) return;
    if (project === pickerLoadedProject && pickerProjectItems.length > 0) return;
    pickerLoadedProject = project;
    const source = sourceFor(`tracker:${project}`);
    if (!(source instanceof TrackerSource)) return;
    void source
      .instrumentPage("tasks")
      .then((page) => {
        if (pickerLoadedProject !== project) return;
        const rows = tasksSections(page).flatMap((section) => section.rows);
        pickerProjectItems = projectCandidates(project, rows);
      })
      .catch(() => {
        if (pickerLoadedProject === project) pickerProjectItems = projectCandidates(project, []);
      });
  });

  function pickerCandidates(): PickerItem[] {
    const items: PickerItem[] = [...pickerProjectItems];
    // recent places: files anywhere, boards of this project
    const project = pickerProjectName;
    for (const location of model.recent) {
      const spec = formatLocation(location);
      const tracker = location.machine.startsWith("tracker:") ? location.machine.slice("tracker:".length) : null;
      if (tracker !== null && tracker !== project) continue;
      items.push({ location: spec, kind: tracker !== null ? "task" : "file", name: spec });
    }
    // the locations the shown page references: the citations carry
    // their target in data-open-path
    for (const element of document.querySelectorAll<HTMLElement>("[data-open-path]")) {
      const path = element.getAttribute("data-open-path");
      const location = view.location;
      if (path && location && location.machine.startsWith("tracker:")) {
        items.push({ location: `${location.machine.slice("tracker:".length)}/${path}`, kind: "board", name: path });
      }
    }
    return uniqueCandidates(items);
  }

  let pickerResults = $derived.by(() => {
    const openKey = view.location ? formatLocation(view.location) : null;
    const candidates = pickerCandidates();
    const query = pickerQuery.trim();
    // a typed location always resolves even when nothing enumerated it
    if (query.length > 0 && parseLocation(query)) {
      const known = candidates.some((candidate) => candidate.location === query);
      if (!known) {
        const typed = { location: query, kind: "board" as const, name: "" };
        return rankedPicker(query, [typed, ...candidates], model.recent.map(formatLocation), openKey);
      }
    }
    return rankedPicker(query, candidates, model.recent.map(formatLocation), openKey);
  });

  // -- this window's commands (round 16's commands(), t-34@r6) ----------
  //
  // One table per window: the space menu (which-key), the help sheet
  // and nothing else read it, so they cannot drift apart. Each entry
  // shows only where it applies.
  function spaceCommands(): MenuAct[] {
    const cmds: MenuAct[] = [{ key: " ", label: "Find a board or file (picker)", onAct: () => openPicker() }];
    if (!view.instrument && view.board) {
      const taskBoard = view.location?.machine.startsWith("tracker:") === true && view.board.task !== undefined && view.board.task !== null;
      if (taskBoard) {
        cmds.push({ key: "h", label: historyMode ? "Leave history" : "History", onAct: () => void toggleHistory() });
        cmds.push({ key: "m", label: marginaliaMode ? "Leave marginalia" : "Marginalia", onAct: () => void toggleMarginalia() });
      }
      if (historyMode) {
        cmds.push({ key: "n", label: "Next change", onAct: () => jumpToChange(1) });
        cmds.push({ key: "N", label: "Previous change", onAct: () => jumpToChange(-1) });
        cmds.push({ key: "c", label: "Current revision", onAct: () => void historyCurrent() });
      }
      const publishLabel =
        drafts.length > 0
          ? `Publish ${drafts.length} draft annotation${drafts.length === 1 ? "" : "s"} (one commit)`
          : "Publish drafts (none yet)";
      cmds.push({ key: "p", label: publishLabel, onAct: () => openPublish() });
      cmds.push({ key: "/", label: "Find text", onAct: () => openSearch() });
    }
    cmds.push({ key: "l", label: "Link to a window (labels)", onAct: () => void enterLinkLabels() });
    if (linkTargetId !== null) cmds.push({ key: "u", label: "Unlink this window", onAct: () => void unlinkHere() });
    return cmds;
  }

  /** What this window is, for the menu and help heads. */
  function windowKind(): WindowKind {
    return view.instrument ? view.instrument.kind : "viewer";
  }

  function helpKeys(): Array<[string, string]> {
    return windowHelpKeys({ kind: windowKind(), history: historyMode !== null, deck: deck !== null });
  }

  /** space: the which-key menu of this window's commands, at the foot of
   * the window; a key that is no command closes it. */
  function openLeader(): void {
    if (closeMenu) closeMenu();
    const cmds = spaceCommands();
    closeMenu = openContextMenu({
      x: Math.max(8, window.innerWidth / 2 - 140),
      y: window.innerHeight - 60 - cmds.length * 22,
      acts: cmds,
      title: `space ...  ${windowKind()} (this window only)`,
      closeOnOtherKey: true,
      control: "leader:menu",
      onClose: () => {
        closeMenu = null;
      },
    });
  }

  /** c in history: back to the current revision, the reading part kept. */
  async function historyCurrent(): Promise<void> {
    if (!historyMode) return;
    const anchor = readingAnchorKey();
    historyMode.at = historyMode.newest;
    await paintHistoryRevision();
    restoreReadingAnchor(anchor);
  }

  function openPicker(): void {
    pickerOpen = true;
    pickerQuery = "";
    pickerActive = 0;
  }

  function closePicker(): void {
    pickerOpen = false;
    returnKeys();
  }

  function pickerMove(delta: number): void {
    const count = pickerResults.length;
    if (count === 0) return;
    pickerActive = (pickerActive + delta + count) % count;
  }

  async function pickerOpenResult(item: PickerItem): Promise<void> {
    closePicker();
    void ensureDeck();
    await model.openInput(item.location);
  }

  function onPickerKey(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      event.preventDefault();
      closePicker();
      return;
    }
    // every letter types (t-31, t-34@r6 change 2): the picker moves on
    // arrows and Ctrl-n Ctrl-p (the mockup's keys), Ctrl-j Ctrl-k too
    const ctrlOnly = event.ctrlKey && !event.altKey && !event.metaKey;
    if (event.key === "ArrowDown" || (ctrlOnly && (event.key === "n" || event.key === "j"))) {
      event.preventDefault();
      pickerMove(1);
      return;
    }
    if (event.key === "ArrowUp" || (ctrlOnly && (event.key === "p" || event.key === "k"))) {
      event.preventDefault();
      pickerMove(-1);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const typed = pickerQuery.trim();
      const item = pickerResults[pickerActive] ?? (typed.length > 0 && parseLocation(typed) ? { location: typed, kind: "board" as const, name: "" } : undefined);
      if (item) void pickerOpenResult(item);
    }
  }

  // The history mode (t-14): a mode that engages with the board at a
  // revision. The mode object sits outside the reactive graph and the
  // shell repaints through a version counter; the margin (the full
  // commit), the change ruler and the reading tick are its chrome.
  // raw: assignments trigger, the mode object's internals stay
  // non-reactive - repaints flow through the paint functions
  let historyMode = $state.raw<HistoryMode | null>(null);
  let historyCommit = $state<HistoryEntry | null>(null);
  let historyCommits = $state.raw<HistoryEntry[]>([]);
  let historyChanges = $state<number | null>(null);

  /** A log entry: go to that revision, the reading part kept. */
  async function historyGo(revision: number): Promise<void> {
    if (!historyMode) return;
    const anchor = readingAnchorKey();
    historyMode.at = revision;
    await paintHistoryRevision();
    restoreReadingAnchor(anchor);
  }
  let historyPainted = false;

  function historyTracker(): TrackerSource | null {
    const location = view.location;
    if (!location || !location.machine.startsWith("tracker:")) return null;
    const source = sourceFor(location.machine);
    return source instanceof TrackerSource ? source : null;
  }

  /** The block at the reading line, keyed by its top-level part: the
   * anchor the mode keeps across its steps. */
  function readingAnchorKey(): { key: string; offset: number } | null {
    if (!contentHost) return null;
    const readingLine = contentHost.getBoundingClientRect().top + contentHost.clientHeight / 3;
    const blocks = contentBlocks();
    let best: HTMLElement | null = null;
    let bestDistance = Number.MAX_SAFE_INTEGER;
    for (const block of blocks) {
      const rect = block.getBoundingClientRect();
      const distance = rect.top > readingLine ? rect.top - readingLine : readingLine - rect.bottom;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = block;
      }
    }
    if (!best || !view.board) return null;
    const index = blocks.indexOf(best);
    const part = view.board.content[index];
    if (!part) return null;
    const key = partKeyForAnchor(part);
    const rect = best.getBoundingClientRect();
    return { key, offset: rect.top - readingLine };
  }

  function partKeyForAnchor(part: Board["content"][number]): string {
    if (part.kind === "text" || part.kind === "annotation") return `t:${part.markdown.trim().slice(0, 200)}`;
    if (part.kind === "group") return `g:${part.title ?? ""}`;
    if (part.kind === "incorporation") return `i:${part.source.task}@${part.source.revision}`;
    if (part.kind === "figure") return `f:${part.input}`;
    return `u:${part.name}`;
  }

  /** Restore the reading anchor: the keyed part, or its nearest
   * surviving neighbour, back at its viewport offset. */
  function restoreReadingAnchor(anchor: { key: string; offset: number } | null): void {
    if (!anchor || !contentHost) return;
    const blocks = contentBlocks();
    if (!view.board) return;
    let target: HTMLElement | null = null;
    const keys = view.board.content.map(partKeyForAnchor);
    const keyed = keys.indexOf(anchor.key);
    if (keyed >= 0) {
      target = blocks[keyed] ?? null;
    } else {
      // the nearest surviving neighbour by key order
      let bestAt = -1;
      keys.forEach((key, index) => {
        void key;
        if (index <= keyed) bestAt = index;
      });
      if (bestAt < 0) bestAt = 0;
      target = blocks[bestAt] ?? null;
    }
    if (!target) return;
    const readingLine = contentHost.getBoundingClientRect().top + contentHost.clientHeight / 3;
    const current = target.getBoundingClientRect().top - readingLine;
    contentHost.scrollBy({ top: current - anchor.offset, behavior: "auto" });
  }

  /** Wrap the diff sentinels the marked markdown carries into real
   * add/del spans; a mark never covers a line break because the
   * pure half keeps the breaks outside. */
  function wrapDiffSentinels(root: HTMLElement): void {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const touched: Text[] = [];
    let node = walker.nextNode();
    while (node) {
      const text = node as Text;
      if (text.data.includes(ADD_OPEN) || text.data.includes(DEL_OPEN)) touched.push(text);
      node = walker.nextNode();
    }
    for (const text of touched) {
      const fragment = document.createDocumentFragment();
      let buffer = "";
      let mode: "same" | "add" | "del" = "same";
      const flush = (): void => {
        if (buffer.length === 0) return;
        if (mode === "same") fragment.append(document.createTextNode(buffer));
        else fragment.append(Object.assign(document.createElement("span"), { className: mode === "add" ? "diff-add" : "diff-del", textContent: buffer }));
        buffer = "";
      };
      for (const character of text.data) {
        if (character === ADD_OPEN) {
          flush();
          mode = "add";
        } else if (character === ADD_CLOSE || character === DEL_CLOSE) {
          flush();
          mode = "same";
        } else if (character === DEL_OPEN) {
          flush();
          mode = "del";
        } else {
          buffer += character;
        }
      }
      flush();
      text.replaceWith(fragment);
    }
  }

  async function paintHistoryRevision(): Promise<void> {
    if (!historyMode || !contentHost) return;
    const mode = historyMode;
    const painted = await mode.boardAtPosition();
    if (historyMode !== mode) return;
    const generation = model.currentGeneration;
    contentHost.textContent = "";
    const units = projectBoard(painted.board);
    const unitIndex = new Map<number, string>();
    for (const unit of units) unitIndex.set(unit.pos.start.offset, unit.id);
    const outcomes: RenderOutcome[] = [];
    const hooks = {
      unitIndex,
      onDiagnostic: (diagnostic: Diagnostic) => {
        outcomes.push({ kind: "html", status: "reported", message: diagnostic.message, line: diagnostic.line });
      },
    };
    const rendered = renderBoard(painted.board, hooks);
    wrapDiffSentinels(rendered);
    contentHost.append(rendered);
    paintHistoryRuler();
    historyCommit = mode.commitOf(mode.at);
    historyChanges = painted.changes;
    historyCommits = mode.entries
      .filter((entry) => entry.event === "task_committed" && entry.revision !== null)
      .sort((a, b) => (b.revision ?? 0) - (a.revision ?? 0));
    const historyLocation = view.location;
    if (!historyLocation) return;
    void hydrate(contentHost, model, historyLocation, generation, outcomes).then((all) => {
      model.setRenderOutcomes(generation, all);
    });
    historyPainted = true;
  }

  /** The change ruler: one tick per changed part or word, at its
   * height in the document, coloured by the change. */
  function paintHistoryRuler(): void {
    if (!contentHost) return;
    let ruler = contentHost.querySelector(":scope > .history-ruler");
    if (!ruler) {
      ruler = document.createElement("div");
      ruler.className = "history-ruler";
      contentHost.append(ruler);
    }
    ruler.textContent = "";
    const hostTop = contentHost.getBoundingClientRect().top - contentHost.scrollTop;
    for (const mark of contentHost.querySelectorAll<HTMLElement>(CHANGE_MARKS)) {
      if (mark.closest(".history-ruler")) continue;
      const rect = mark.getBoundingClientRect();
      const tick = document.createElement("span");
      const kind = mark.classList.contains("chg-add") || mark.classList.contains("diff-add") ? "add" : mark.classList.contains("chg-del") || mark.classList.contains("diff-del") ? "del" : "mod";
      tick.className = `history-ruler-tick ${kind}`;
      tick.style.top = `${rect.top - hostTop}px`;
      tick.style.height = `${rect.height}px`;
      ruler.append(tick);
    }
  }

  async function enterHistory(): Promise<void> {
    const location = view.location;
    const source = historyTracker();
    if (!location || !source) {
      // history is a tracker board's mode; a file location has none
      return;
    }
    const spec = parseTaskSpec(location.path);
    if (!spec) return;
    const mode = new HistoryMode(source, `t-${spec.task}`);
    try {
      await mode.load(spec.revision);
    } catch {
      historyMode = null;
      return;
    }
    historyMode = mode;
    const anchor = readingAnchorKey();
    await paintHistoryRevision();
    restoreReadingAnchor(anchor);
  }

  async function leaveHistory(): Promise<void> {
    const mode = historyMode;
    if (!mode) return;
    historyMode = null;
    historyCommit = null;
    const location = view.location;
    if (location && location.machine.startsWith("tracker:")) {
      const anchor = readingAnchorKey();
      await model.open({ machine: location.machine, path: mode.exitPath() }, { navigate: true });
      restoreReadingAnchor(anchor);
    }
  }

  async function toggleHistory(): Promise<void> {
    if (historyMode) await leaveHistory();
    else await enterHistory();
  }

  async function historyStep(direction: "older" | "newer"): Promise<void> {
    if (!historyMode) return;
    const anchor = readingAnchorKey();
    historyMode.step(direction);
    await paintHistoryRevision();
    restoreReadingAnchor(anchor);
  }

  /** Every change in the painted revision at any depth: word marks and
   * whole parts added, removed or moved (a modified group is its
   * children's changes). */
  const CHANGE_MARKS = ".diff-add, .diff-del, .chg-add, .chg-del, .chg-mod:not(.block-group)";

  function jumpToChange(delta: number): void {
    if (!contentHost) return;
    const marks = Array.from(contentHost.querySelectorAll<HTMLElement>(CHANGE_MARKS));
    if (marks.length === 0) return;
    const readingLine = contentHost.getBoundingClientRect().top + contentHost.clientHeight / 3;
    const target =
      delta > 0
        ? marks.find((mark) => mark.getBoundingClientRect().top > readingLine)
        : [...marks].reverse().find((mark) => mark.getBoundingClientRect().top < readingLine);
    if (target) target.scrollIntoView({ block: "center", behavior: "auto" });
  }

  async function walkTracked(direction: "older" | "newer"): Promise<void> {
    if (!historyMode) return;
    if (historyMode.trackedKey === null) {
      const anchor = readingAnchorKey();
      if (anchor) historyMode.track(anchor.key);
    }
    const target = await historyMode.walkTracked(direction);
    if (target === null) return;
    const anchor = readingAnchorKey();
    historyMode.at = target;
    await paintHistoryRevision();
    restoreReadingAnchor(anchor);
  }

  function trackReadingPart(): void {
    if (!historyMode) return;
    const anchor = readingAnchorKey();
    if (anchor) historyMode.track(anchor.key);
  }

  // Zoom (+ - =, t-14): the document's scale, carried by the window's
  // window state since the window store landed (t-24@r20; it was memory-only
  // before) - written when it settles, applied from the state on
  // resume and on another device's change.
  let zoom = $state(1);
  const ZOOM_MIN = 0.7;
  const ZOOM_MAX = 2;
  function zoomBy(delta: number): void {
    applyZoom(zoom + delta);
    scheduleZoomWrite();
  }
  function zoomReset(): void {
    applyZoom(1);
    scheduleZoomWrite();
  }

  /** A settled zoom write: sent when the steps stop, never per key
   * press (t-24@r20). */
  function scheduleZoomWrite(): void {
    if (windowStore === null || windowStateId === null || windowDiscarded) return;
    if (zoomWriteTimer !== null) clearTimeout(zoomWriteTimer);
    zoomWriteTimer = setTimeout(() => {
      zoomWriteTimer = null;
      if (windowStore === null || windowStateId === null || windowDiscarded) return;
      if (zoom === knownZoom) return;
      knownZoom = zoom;
      void windowStore.patch(windowStateId, { zoom }).catch(() => {});
    }, 400);
  }

  $effect(() => {
    void zoom;
    if (contentHost && contentHost.style.zoom === "") contentHost.style.zoom = String(zoom);
  });

  /** The primary act of a figure: enlarge, in the document's own
   * overlay - the image at its natural size on the app background. */
  function enlargeFigure(frame: HTMLElement): void {
    const image = frame.querySelector("img");
    if (!image) return;
    closeFigureOverlay();
    const overlay = document.createElement("div");
    overlay.className = "figure-overlay";
    overlay.dataset.control = "figure:overlay";
    const clone = image.cloneNode(true) as HTMLImageElement;
    overlay.append(clone);
    const close = (): void => {
      overlay.remove();
      window.removeEventListener("keydown", onKey, true);
    };
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    };
    overlay.addEventListener("click", close);
    window.addEventListener("keydown", onKey, true);
    document.body.append(overlay);
  }

  function closeFigureOverlay(): void {
    document.querySelectorAll(".figure-overlay").forEach((element) => element.remove());
  }

  /** Expand an incorporation: only the cited components, fetched from
   * the cited revision's component page. The provenance stays; a
   * refusal keeps the collapse. */
  async function expandIncorporation(block: HTMLElement): Promise<void> {
    const existing = block.nextElementSibling;
    if (existing?.classList.contains("incorporation-expanded")) {
      existing.remove();
      return;
    }
    const location = view.location;
    const source = historyTracker();
    if (!location || !source) return;
    const spec = parseTaskSpec(location.path);
    if (!spec || spec.revision === null) return;
    const blockIndex = contentBlocks().indexOf(block);
    if (blockIndex < 0 || !view.board) return;
    const part = view.board.content[blockIndex];
    if (!part || part.kind !== "incorporation") return;
    const selectPaths = part.source.select;
    const container = document.createElement("div");
    container.className = "incorporation-expanded";
    container.dataset.control = "incorporation:expanded";
    const provenance = document.createElement("p");
    provenance.className = "incorporation-provenance";
    provenance.textContent = `${part.source.task}@r${part.source.revision}, the cited parts only`;
    container.append(provenance);
    for (const path of selectPaths) {
      try {
        const page = await source.citedComponents(`${part.source.task}@r${part.source.revision}`, path);
        for (const citedPart of page.content ?? []) {
          if (citedPart.kind !== "text") continue;
          const el = document.createElement("div");
          el.className = "block incorporation-cited";
          const rendered = renderBoard(
            { grammar: 1, task: "", content: [{ kind: "text", markdown: citedPart.text, pos: { start: { line: 0, column: 0, offset: 0 }, end: { line: 0, column: 0, offset: 0 } } }], diagnostics: [] },
            { unitIndex: new Map(), onDiagnostic: () => {} },
          );
          el.append(rendered);
          container.append(el);
        }
      } catch (error) {
        const note = document.createElement("p");
        note.className = "incorporation-provenance";
        note.textContent = `the cited part refused: ${String(error)}`;
        container.append(note);
      }
    }
    block.insertAdjacentElement("afterend", container);
  }

  /** The click and primary acts the document carries: figures enlarge,
   * incorporations expand, citations open - pointer complete. */
  function onContentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement | null;
    if (!target) return;
    const figure = target.closest<HTMLElement>(".block-figure .figure-frame, .block-figure img");
    if (figure) {
      const frame = figure.closest<HTMLElement>(".block-figure") ?? figure;
      enlargeFigure(frame);
      return;
    }
    const incorporation = target.closest<HTMLElement>(".block-incorporation");
    if (incorporation && !target.closest("a")) {
      void expandIncorporation(incorporation);
    }
  }

  $effect(() => {
    contentHost?.addEventListener("click", onContentClick);
    return () => contentHost?.removeEventListener("click", onContentClick);
  });

  // Draft annotations (t-14, ratified): `a` on the part at the reading
  // height drafts in place; drafts ride the window's state since
  // the window store landed (t-24@r20) - two devices showing one state share
  // them, and the window's own drafts never lose on a refused
  // publish. `space p` publishes them together as ONE commit with ONE
  // message under the human's own identity; the current revision of a
  // task board only. A window with no window store keeps its drafts in
  // memory, which is still never a lost draft.
  // On an explainer - a bare-path file that decodes as a board - the
  // editor's save IS the submit: one annotation written straight into
  // the file (t-24@r21), nothing pooled.
  let drafts = $state<Draft[]>([]);
  let draftEditor = $state<{ path: string; text: string } | null>(null);
  let draftEditorIndex = $state<number | null>(null);
  let draftKind = $state<"task" | "explainer">("task");
  let publishOpen = $state(false);
  let publishMessage = $state("");
  let publishBusy = $state(false);
  let publishNote = $state<string | null>(null);

  function currentTaskSpec(): { tracker: string; task: string; taskNumber: number } | null {
    const location = view.location;
    if (!location || !location.machine.startsWith("tracker:")) return null;
    const spec = parseTaskSpec(location.path);
    if (!spec) return null;
    return {
      tracker: location.machine.slice("tracker:".length),
      task: `t-${spec.task}`,
      taskNumber: spec.task,
    };
  }

  function updateDrafts(next: Draft[]): void {
    drafts = next;
    knownDrafts = next;
    if (windowStore === null || windowStateId === null || windowDiscarded) return;
    void windowStore.patch(windowStateId, { drafts: next }).catch(() => {
      // an unreachable window store keeps the drafts in memory; the next
      // settled draft write carries them
    });
  }

  /** The part the act lands on: the selected block, else the one at
   * the reading height - a top-level part, as the select path names. */
  // the element a menu act names: the menu clears the selection before
  // its act runs, so the act hands its own part over here
  let actElement: HTMLElement | null = null;

  function actPartIndex(): number | null {
    if (!contentHost || !view.board) return null;
    const blocks = contentBlocks();
    let element: HTMLElement | null = actElement ?? selected?.element ?? null;
    actElement = null;
    // a part inside a group acts through its top-level part
    if (element) element = blocks.find((block) => block === element || block.contains(element)) ?? element;
    if (!element) {
      const readingLine = contentHost.getBoundingClientRect().top + contentHost.clientHeight / 3;
      let best = Number.MAX_SAFE_INTEGER;
      for (const block of blocks) {
        const rect = block.getBoundingClientRect();
        const distance = rect.top > readingLine ? rect.top - readingLine : readingLine - rect.bottom;
        if (distance < best) {
          best = distance;
          element = block;
        }
      }
    }
    if (!element) return null;
    const index = blocks.indexOf(element);
    return index >= 0 && index < view.board.content.length ? index : null;
  }

  function beginDraft(): void {
    if (!view.board) return;
    const location = view.location;
    if (!location) return;
    if (location.machine === "tracker-machine") {
      // the explainer write (t-24@r21): bare paths only - the file
      // this window decoded as a board. The text of a refused
      // annotation is kept on the device keyed by the path; the next
      // `a` opens the editor with it, placement dropped.
      draftKind = "explainer";
      publishNote = null;
      const index = actPartIndex();
      if (index === null) return;
      draftEditorIndex = index;
      draftEditor = { path: `/content/${index}`, text: loadKeptExplainer(location.path) ?? "" };
      return;
    }
    // the revision gate reads the ASKED location: a bare open is the
    // human asking for the current board, and the view naming its
    // resolved revision does not make it pinned (t-24@r11 refuses a
    // revision the human actually pinned)
    const spec = parseTaskSpec((view.askedLocation ?? location).path);
    if (!spec) {
      publishNote =
        "annotations draft on a task board or on a bare path on the tracker's machine; this location is neither - bare paths only for the explainer write (t-24@r21)";
      return;
    }
    draftKind = "task";
    if (spec.revision !== null) {
      // t-24@r11: only the current revision drafts; a pinned revision
      // refuses, saying where annotation is possible
      publishNote = "only the current revision of a task board drafts; open the task bare (tracker/t-N) and annotate there";
      return;
    }
    const index = actPartIndex();
    if (index === null) return;
    const existing = drafts.find((draft) => draft.path === `/content/${index}`);
    draftEditorIndex = index;
    draftEditor = { path: `/content/${index}`, text: existing?.text ?? "" };
  }

  function saveDraft(): void {
    const editor = draftEditor;
    if (!editor) return;
    const text = editor.text;
    if (draftKind === "explainer") {
      // the submit IS the save (t-24@r21 A5): one annotation into the
      // file, nothing pooled; an empty save just closes
      if (text.trim().length === 0) {
        cancelDraft();
        return;
      }
      void submitExplainer(text);
      return;
    }
    const next = drafts.filter((draft) => draft.path !== editor.path);
    if (text.trim().length > 0) next.push({ path: editor.path, text: text.replace(/\s+$/, "") });
    updateDrafts(next);
    draftEditor = null;
    draftEditorIndex = null;
  }

  /** The explainer annotation submit: the edit computed against the
   * shown text and decoded before anything is sent (t-24@r21 A1), the
   * whole file re-read and byte-compared with what this window shows
   * (A4), then one file.edit through the bridge. A refusal keeps the
   * annotation TEXT, drops the placement, and says where to place it
   * again; the wire's exact-once answer is the same refusal reached
   * through the other half of the guard. */
  async function submitExplainer(text: string): Promise<void> {
    const location = view.location;
    const shown = model.snapshot.rawText ?? "";
    if (!location || location.machine !== "tracker-machine" || shown.length === 0) return;
    const editorPath = draftEditor?.path;
    if (!editorPath) return;
    const generation = model.currentGeneration;
    const stale = (): boolean => {
      // the human navigated while the act awaited: the submit is void,
      // nothing is written, nothing is noted
      return model.currentGeneration !== generation;
    };
    const plan = explainerEdit(shown, editorPath, text);
    if ("failure" in plan) {
      // nothing was sent; the editor stays open with the reason
      publishNote = plan.failure;
      return;
    }
    let machine: string;
    try {
      machine = await trackerCoreMachine();
    } catch (error) {
      publishNote = error instanceof Error ? error.message : String(error);
      return;
    }
    if (stale()) return;
    let current: string;
    try {
      const source = sourceFor("tracker-machine");
      current = (await source.loadText(location)).text;
    } catch (error) {
      publishNote = error instanceof Error ? error.message : String(error);
      return;
    }
    if (stale()) return;
    if (current !== shown) {
      saveKeptExplainer(location.path, text);
      draftEditor = null;
      draftEditorIndex = null;
      publishNote =
        "the file changed since this window read it; the annotation text is kept - reopen the file and press a on the part you want, the editor opens with the kept text";
      return;
    }
    const outcome = await submitExplainerEdit(tauriInvoke(), {
      path: location.path,
      old: plan.edit.old,
      new: plan.edit.new,
      on: machine,
    });
    if (stale()) return;
    if (!outcome.ok) {
      if (outcome.code === "edit-failed") {
        saveKeptExplainer(location.path, text);
        draftEditor = null;
        draftEditorIndex = null;
        publishNote =
          "the file moved under this window - the edit's exact-once guard refused and the file is untouched; the annotation text is kept - reopen the file and press a on the part you want";
        return;
      }
      publishNote = `the write refused: ${outcome.code} ${outcome.detail}`;
      return;
    }
    clearKeptExplainer(location.path);
    draftEditor = null;
    draftEditorIndex = null;
    publishNote = `annotation written into ${location.path}`;
    await model.open({ machine: location.machine, path: location.path });
  }

  function cancelDraft(): void {
    draftEditor = null;
    draftEditorIndex = null;
  }

  let publishPanel: { open: boolean } | null = null;

  function openPublish(): void {
    if (!view.board) return;
    if (view.location?.machine === "tracker-machine" || draftKind === "explainer") {
      publishNote =
        "an explainer has no pooled publish; each annotation is written straight into its file on save (t-24@r21)";
      return;
    }
    if (drafts.length === 0) {
      publishNote = "no drafts to publish; press a on a part first";
      return;
    }
    publishMessage = `annotations on ${view.board.task ?? "the board"}: ${drafts.length} draft${drafts.length === 1 ? "" : "s"}`;
    publishNote = null;
    publishOpen = true;
    publishPanel = { open: true };
    void publishPanel;
  }

  function cancelPublish(): void {
    publishOpen = false;
    publishPanel = null;
  }

  async function publishDrafts(): Promise<void> {
    const current = currentTaskSpec();
    if (!current || publishBusy) return;
    publishBusy = true;
    publishNote = null;
    try {
      const location = view.location;
      const source = historyTracker();
      if (!location || !source) {
        publishNote = "annotations publish on a task board served by a tracker";
        return;
      }
      // the revision gate reads the ASKED location, exactly as the
      // draft gate does: a bare open is the current board
      const spec = parseTaskSpec((view.askedLocation ?? location).path);
      if (!spec) return;
      if (spec.revision !== null) {
        publishNote = "only the current revision of a task board publishes; open the task bare (tracker/t-N)";
        return;
      }
      // the board page carries the state: an offered board refuses,
      // naming the CLI step
      const page = await source.currentBoardPage(spec.task);
      const state = pageFact(page, "state");
      if (state !== null && state !== "open") {
        publishNote = `the board is ${state}; publishing refuses - reject it with grounds first (sac reject t-${spec.task} --grounds "..."), then publish`;
        return;
      }
      // the concurrency guard: the shown text must still be current
      // (t-24@r11: a draft is never lost on a refused publish)
      const currentBoard = await source.currentBoardText(spec.task);
      const shown = model.snapshot.rawText ?? "";
      if (shown.length > 0 && shown !== currentBoard.text) {
        publishNote = `the board moved (a revision landed after this window read r${spec.revision ?? "current"}); the drafts are kept - reopen the task and review, then publish`;
        return;
      }
      // the figures ride unchanged, fetched fresh at their pinned paths
      const pinned = currentBoard.pinned;
      const figures: Array<{ path: string; bytes: Uint8Array }> = [];
      const shownBoard = model.snapshot.board;
      if (!shownBoard) return;
      for (const block of shownBoard.content) {
        if (block.kind !== "figure") continue;
        const resolved = resolveRelative("", block.input);
        if (!resolved.ok) continue;
        const bytes = await source.loadAssetBytes(pinned, resolved.joined);
        figures.push({ path: resolved.joined, bytes });
      }
      const identity = await bridgePublishIdentity();
      const payload = publishPayload({
        task: spec.task,
        boardText: shown,
        drafts,
        message: publishMessage,
        identity,
        figures,
      });
      if ("failure" in payload) {
        publishNote = payload.failure;
        return;
      }
      const base = serverBase();
      const result = await publishAnnotations(base, payload.payload);
      if (!result.ok) {
        if (result.code === "invalid_state_transition") {
          publishNote = `the board is offered; publishing refuses - reject it with grounds first (sac reject t-${spec.task} --grounds "..."), then publish`;
        } else {
          publishNote = `the publish refused: ${result.code} ${result.detail}`;
        }
        return;
      }
      // published: the drafts landed with the new revision; the
      // window's state carries none now (t-24@r20: drafts ride
      // the state)
      knownDrafts = [];
      drafts = [];
      if (windowStore !== null && windowStateId !== null && !windowDiscarded) {
        void windowStore.patch(windowStateId, { drafts: [] }).catch(() => {});
      }
      publishOpen = false;
      publishPanel = null;
      publishNote = `published as r${result.revision} under ${identity.actor}/human`;
      await model.open({ machine: location.machine, path: `t-${spec.task}` });
    } finally {
      publishBusy = false;
    }
  }

  async function bridgePublishIdentity(): Promise<{ actor: string; tier: "human" }> {
    try {
      const invoke = tauriInvoke();
      const identity = (await invoke("publish_identity")) as { actor: string; tier: string };
      return { actor: identity.actor, tier: "human" };
    } catch {
      // no host: the fallback sac itself uses, never a refusal
      return { actor: "human", tier: "human" };
    }
  }

  let mainWrap: HTMLElement | undefined = $state(undefined);

  /** The rendered top-level blocks: one render root (div.board) sits
   * between the content host and the blocks renderBoard appended. */
  function contentBlocks(): HTMLElement[] {
    // a part the revision removed is shown, not part of the board
    return Array.from(contentHost?.querySelectorAll<HTMLElement>(":scope > .board > .block:not(.chg-del)") ?? []);
  }

  // -- the deck (t-21, t-24@r21) -----------------------------------------
  // The address is a herdr-core document per window: deck- plus 96
  // random bits, created on the machine the bare-path rule resolves
  // (the hub's own core there, the declared tracker machine
  // elsewhere). Holding the address is the permission; the document
  // holds locations only, never the human's position; the shell
  // deletes it when the window closes. The cue feed is the wake-driven
  // hold (deck_wait), never a timer: every answer re-enters the hold
  // while this window's deck stands.
  let deck = $state.raw<DeckState | null>(null);
  let deckMachine = "";
  const deckSnapshots = new Map<number, string>();
  let deckFeedLost = $state<string | null>(null);

  function deckInvoke(): ReturnType<typeof tauriInvoke> {
    return tauriInvoke();
  }

  async function ensureDeck(): Promise<void> {
    if (deck || !bridgePresent) return;
    let machine: string;
    try {
      machine = await trackerCoreMachine();
    } catch (error) {
      // no core, no server set, or not the hub without a declaration:
      // the deck is absent, the reason on the app log for the lanes
      console.log(
        `[viewer] deck absent: ${error instanceof Error ? error.message : String(error)}`,
      );
      return;
    }
    const bytes = new Uint8Array(12);
    crypto.getRandomValues(bytes);
    const name = deckNameFromBytes(bytes);
    try {
      await deckInvoke()("deck_create", { name, on: machine });
    } catch (error) {
      console.log(`[viewer] deck refused: ${String(error)}`);
      return;
    }
    deckMachine = machine;
    deck = newDeck(name);
    console.log(`[viewer] deck offered: ${name}`);
    void watchDeck();
  }

  async function watchDeck(): Promise<void> {
    const current = deck;
    if (!current) return;
    try {
      await deckInvoke()("deck_watch", { name: current.name, on: deckMachine });
    } catch (error) {
      deckFeedLost = String(error);
      return;
    }
    let quietLogged = false;
    // the hold loops on the ADDRESS, not on a deck object: every
    // landing reassigns the deck state, and an identity loop would
    // stop watching after the first cue
    while (deck !== null && deck.name === current.name) {
      let message: { kind?: string; version?: number; reason?: string };
      try {
        message = (await deckInvoke()("deck_wait", { name: current.name })) as typeof message;
      } catch (error) {
        if (deck === null || deck.name !== current.name) return;
        deckFeedLost = String(error);
        console.log(`[viewer] deck feed invoke failed: ${deckFeedLost}`);
        return;
      }
      if (deck === null || deck.name !== current.name) return;
      if (message.kind === "doc") {
        await deckDocument();
        continue;
      }
      if (message.kind === "lost") {
        deckFeedLost = message.reason ?? "the feed ended";
        console.log(`[viewer] deck feed lost: ${deckFeedLost}`);
        return;
      }
      // Quiet: the hold expired with nothing cued; re-enter it. One
      // line for the first cycle is the lanes' evidence that the feed
      // cycles at all; the rest stay silent.
      if (!quietLogged) {
        quietLogged = true;
        console.log("[viewer] deck feed cycling quiet (hold expired, re-entering)");
      }
    }
  }

  async function deckDocument(): Promise<void> {
    const before = deck;
    if (!before) return;
    let doc: { value: unknown; version: number };
    try {
      doc = (await deckInvoke()("deck_read", {
        name: before.name,
        on: deckMachine,
      })) as typeof doc;
    } catch (error) {
      console.log(`[viewer] deck read failed: ${String(error)}`);
      return;
    }
    if (deck !== before) return;
    const onLatest = view.slideIndex !== null && view.slideIndex === before.position;
    const event = applyDocument(before, onLatest, doc.value, doc.version);
    if (event.dead) {
      // the address is dead: the document was deleted from outside
      deck = null;
      deckSnapshots.clear();
      publishNote = "the deck's document was deleted; the address is dead";
      return;
    }
    deck = event.state;
    console.log(
      `[viewer] cue landing: arrived=${event.arrived.length} show=${event.show} slides=${event.state.slides.length}`,
    );
    // file slides snapshot when the cue lands; boards pin by revision
    // when shown (their location already carries @rK if the agent
    // cued it, else the open pins what answers)
    for (const index of event.arrived) {
      const spec = event.state.slides[index];
      if (spec === undefined) continue;
      const location = parseLocation(spec);
      if (!location || location.machine.startsWith("tracker:")) continue;
      void snapshotFor(index, location);
    }
    if (event.show) {
      await showSlide(event.state.slides.length - 1);
    }
  }

  async function snapshotFor(index: number, location: Location): Promise<void> {
    const current = deck;
    if (!current) return;
    try {
      const loaded = await sourceFor(location.machine).loadText(location);
      if (deck !== current) return;
      deckSnapshots.set(index, loaded.text);
    } catch {
      // no snapshot held: showing the slide answers with the live
      // read's own structured refusal
    }
  }

  async function showSlide(index: number): Promise<void> {
    const current = deck;
    if (!current) return;
    const spec = current.slides[index];
    if (spec === undefined) return;
    const location = parseLocation(spec);
    if (!location) {
      publishNote = `slide ${index + 1} is a location this viewer cannot parse: ${spec}`;
      return;
    }
    console.log(`[viewer] show slide ${index + 1}: ${spec} via ${(new Error().stack ?? "").split("\n")[2]?.trim() ?? "?"}`);
    if (location.machine.startsWith("tracker:")) {
      await model.open(location, { deck: true, slideIndex: index });
    } else {
      const snapshot = deckSnapshots.get(index);
      if (typeof snapshot === "string") {
        await model.openSnapshot(location, snapshot, index);
      } else {
        await model.open(location, { deck: true, slideIndex: index });
      }
    }
    console.log(
      `[viewer] slide shown: index=${index} view=${model.snapshot.location?.path ?? "none"} status=${model.snapshot.status}`,
    );
    if (deck !== current) return;
    deck = { ...current, position: index, entered: true };
  }

  async function deckStep(direction: "older" | "newer"): Promise<void> {
    const current = deck;
    if (!current) return;
    const step = stepDeck(current, direction);
    deck = step.state;
    if (step.target !== null) await showSlide(step.target);
  }

  async function deckJump(digit: number): Promise<void> {
    const current = deck;
    if (!current) return;
    const jump = jumpDeck(current, digit);
    deck = jump.state;
    if (jump.target !== null) await showSlide(jump.target);
  }

  function copyCueCommand(): void {
    const current = deck;
    if (!current) return;
    const command = cueCommand(current.name, "<location>");
    if (bridgePresent) void bridgeWrite(command);
    else void navigator.clipboard?.writeText(command).catch(() => {});
  }

  let deckOnLatest = $derived(
    deck !== null && view.slideIndex !== null && view.slideIndex === deck.position,
  );
  let deckWaiting = $derived(deck === null ? null : waitingSlide(deck, deckOnLatest));
  let deckStateText = $derived.by(() => {
    const current = deck;
    if (!current) return "";
    if (view.slideIndex !== null) {
      const shown = `slide ${view.slideIndex + 1}/${current.slides.length}`;
      return deckWaiting !== null ? `${shown}, slide ${deckWaiting} waiting` : shown;
    }
    return deckWaiting !== null ? `slide ${deckWaiting} waiting` : "";
  });

  // The deck offers itself once the shell is up: at mount, and again
  // on the next open after a failed reach. Not at the script's top -
  // the deck state and the bridge marker are declared later, and a
  // call there reads them before initialization (the remote lane's
  // first run caught exactly that).

  let historyActive = $derived(historyMode !== null);

  // Marginalia (space m): rows, each run of parts beside the full
  // message of the commit that last changed it. The rows carry the
  // revision their commit landed; the full message rides the margin.
  let marginaliaMode = $state.raw<HistoryMode | null>(null);
  let marginaliaRows = $state<MarginaliaRun[]>([]);
  let marginaliaCommit = $state<HistoryEntry | null>(null);
  let marginaliaAt = $state(0);

  let marginaliaActive = $derived(marginaliaMode !== null);

  async function toggleMarginalia(): Promise<void> {
    if (marginaliaMode) {
      marginaliaMode = null;
      marginaliaRows = [];
      marginaliaCommit = null;
      return;
    }
    const location = view.location;
    const source = historyTracker();
    if (!location || !source) return;
    const spec = parseTaskSpec(location.path);
    if (!spec) return;
    const mode = new HistoryMode(source, `t-${spec.task}`);
    try {
      await mode.load(spec.revision);
      const runs = await marginaliaRuns(mode);
      if (!view.board) return;
      // a run's commit: the entry that landed its lastChanged revision
      const first = runs[0];
      marginaliaMode = mode;
      marginaliaRows = runs;
      marginaliaAt = mode.newest;
      marginaliaCommit = first ? mode.commitOf(first.lastChanged) : null;
    } catch {
      marginaliaMode = null;
    }
  }

  function marginaliaCommitFor(run: MarginaliaRun): HistoryEntry | null {
    return marginaliaMode?.commitOf(run.lastChanged) ?? null;
  }

  function scrollToRun(run: MarginaliaRun): void {
    if (!contentHost || !view.board) return;
    const blocks = contentBlocks();
    const target = blocks[run.indexes[0] ?? -1];
    if (target) target.scrollIntoView({ block: "start", behavior: "auto" });
  }

  $effect(() => {
    if (mainWrap && historyMode) {
      const onScroll = (): void => {
        // the reading tick rides the content's scroll
        mainWrap?.style.setProperty("--tick-visible", "1");
      };
      contentHost?.addEventListener("scroll", onScroll);
      return () => contentHost?.removeEventListener("scroll", onScroll);
    }
    return undefined;
  });

  function onWindowKey(event: KeyboardEvent): void {
    if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "c") {
      event.preventDefault();
      model.setCopyMode(view.copyMode === "plain" ? "source" : "plain");
      return;
    }
    if (event.key === "Escape") {
      if (pickerOpen) {
        event.preventDefault();
        closePicker();
        return;
      }
      if (searchOpen) {
        event.preventDefault();
        closeSearch();
        return;
      }
      if (helpOpen) {
        event.preventDefault();
        helpOpen = false;
        return;
      }
      if (overlay) {
        event.preventDefault();
        overlay = null;
        return;
      }
    }
    if (closeMenu || inTextField(event)) return;
    if (helpOpen || pickerOpen) return;

    if (event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === "o") {
      event.preventDefault();
      void model.back();
      return;
    }
    if (event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === "i") {
      event.preventDefault();
      void model.forward();
      return;
    }
    if (jumpMode !== null) {
      // the window jump owns the keyboard while it is on
      event.preventDefault();
      void onJumpKey(event);
      return;
    }
    if (askOnce !== null) {
      // the ask-once owns the keyboard while it is on
      event.preventDefault();
      void onAskOnceKey(event);
      return;
    }

    switch (keyMode) {
      case "labels": {
        if (event.key === "Escape") {
          event.preventDefault();
          exitLabelMode();
          return;
        }
        if (event.key === "Enter" && narrow) {
          event.preventDefault();
          const alive = narrow.alive;
          exitLabelMode();
          if (alive.length === 1) {
            const index = alive[0];
            if (index !== undefined) {
              selectTarget(index);
              const target = targets[index];
              if (target) {
                const rect = target.element.getBoundingClientRect();
                openTargetMenu(target, { x: rect.right + 8, y: rect.top });
              }
            }
          }
          return;
        }
        if (event.key.length !== 1) return;
        event.preventDefault();
        if (!narrow) return;
        const texts = targets.map((t) => t.text);
        const kinds = targets.map((t) => t.kind);
        const step = narrowStep(narrow, kinds, texts, labels, event.key);
        if (step.selected !== null) {
          const index = step.selected;
          exitLabelMode();
          selectTarget(index);
          const target = targets[index];
          if (target) {
            const rect = target.element.getBoundingClientRect();
            openTargetMenu(target, { x: rect.right + 8, y: rect.top });
          }
          return;
        }
        narrow = step;
        // repaint the chips for what is still alive
        const aliveTargets = step.alive.map((index) => targets[index]).filter((t) => t !== undefined);
        if (hideLabels) hideLabels();
        hideLabels = showLabels(aliveTargets, aliveTargets.map((t) => labels[targets.indexOf(t)] ?? ""));
        return;
      }
      case "gg": {
        event.preventDefault();
        keyMode = "normal";
        if (event.key === "g") scrollTop(scroller());
        return;
      }
      case "normal":
        break;
    }

    switch (event.key) {
      case "h":
        if (historyMode) {
          event.preventDefault();
          void historyStep("older");
          return;
        }
        if (deck && !marginaliaMode) {
          // h l move between slides; in history they step time (t-14)
          event.preventDefault();
          console.log(`[viewer] deck key h (target=${(event.target as HTMLElement)?.tagName})`);
          void deckStep("older");
          return;
        }
        break;
      case "l":
        if (historyMode) {
          event.preventDefault();
          void historyStep("newer");
          return;
        }
        if (deck && !marginaliaMode) {
          event.preventDefault();
          console.log(`[viewer] deck key l (target=${(event.target as HTMLElement)?.tagName})`);
          void deckStep("newer");
          return;
        }
        break;
      case "n":
        if (historyMode) {
          event.preventDefault();
          jumpToChange(1);
          return;
        }
        break;
      case "N":
        if (historyMode) {
          event.preventDefault();
          jumpToChange(-1);
          return;
        }
        break;
      case "t":
        if (historyMode) {
          event.preventDefault();
          trackReadingPart();
          return;
        }
        break;
      case "H":
        if (historyMode) {
          event.preventDefault();
          void walkTracked("older");
          return;
        }
        break;
      case "L":
        if (historyMode) {
          event.preventDefault();
          void walkTracked("newer");
          return;
        }
        break;
      case "j":
        event.preventDefault();
        if (view.instrument) moveRowCursor(1);
        else if (selected) moveSelection(1);
        else scrollByLines(scroller(), 2);
        return;
      case "k":
        event.preventDefault();
        if (view.instrument) moveRowCursor(-1);
        else if (selected) moveSelection(-1);
        else scrollByLines(scroller(), -2);
        return;
      case "d":
        event.preventDefault();
        scrollByPages(scroller(), 0.5);
        return;
      case "u":
        event.preventDefault();
        scrollByPages(scroller(), -0.5);
        return;
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
        // digits jump to slide numbers in normal mode (t-14, t-24@r21:
        // label mode consumes them first - this switch is normal mode)
        if (deck) {
          event.preventDefault();
          console.log(`[viewer] deck digit ${event.key} (target=${(event.target as HTMLElement)?.tagName})`);
          void deckJump(Number(event.key));
          return;
        }
        break;
      case "g":
        event.preventDefault();
        keyMode = "gg";
        return;
      case "G":
        event.preventDefault();
        scrollBottom(scroller());
        return;
      case "s":
        event.preventDefault();
        enterLabelMode();
        return;
      case "a":
        event.preventDefault();
        beginDraft();
        return;
      case "+":
        event.preventDefault();
        zoomBy(0.1);
        return;
      case "-":
        event.preventDefault();
        zoomBy(-0.1);
        return;
      case "=":
        event.preventDefault();
        zoomReset();
        return;
      case "?":
        event.preventDefault();
        helpOpen = true;
        return;
      case "'":
        // the window jump's labels (the key is a placeholder, t-14)
        event.preventDefault();
        void enterJumpLabels();
        return;
      case '"':
        // the window jump's picker (the key is a placeholder, t-14)
        event.preventDefault();
        void enterJumpPicker();
        return;
      case "Enter": {
        event.preventDefault();
        const target = selected;
        if (view.instrument) {
          // the instruments have no here: Enter goes through the link
          instrumentEnter();
          return;
        }
        if (target) {
          const rect = target.element.getBoundingClientRect();
          openTargetMenu(target, { x: rect.right + 8, y: rect.top });
        }
        return;
      }
      case "o":
      case "i":
      case "w": {
        // the elsewhere acts on the selected citation (t-14): o opens
        // through the link, i elsewhere once, w always a new window
        const target = selected;
        const path = target?.element.getAttribute("data-open-path");
        const current = view.location;
        if (!target || !path || !current) return;
        const spec = citationSpec(current, path);
        if (spec === null) return;
        event.preventDefault();
        if (event.key === "o") void openSpecElsewhere(spec, "link");
        else if (event.key === "i") void openSpecElsewhere(spec, "once");
        else void openSpecElsewhere(spec, "new");
        return;
      }
      case " ":
        event.preventDefault();
        openLeader();
        return;
    }
    if (event.ctrlKey && !event.shiftKey) {
      if (event.key === "f") {
        // page down
        event.preventDefault();
        scrollByPages(scroller(), 1);
        return;
      }
      if (event.key === "b") {
        event.preventDefault();
        scrollByPages(scroller(), -1);
        return;
      }
      if (event.key === "d") {
        event.preventDefault();
        scrollByPages(scroller(), 0.5);
        return;
      }
      if (event.key === "u") {
        event.preventDefault();
        scrollByPages(scroller(), -0.5);
        return;
      }
    }
  }

  function onContextMenu(event: MouseEvent): void {
    event.preventDefault();
    const element = (event.target as HTMLElement | null)?.closest<HTMLElement>(
      ".block, .diagram-frame, .citation-link, .instrument-row, [data-control]",
    );
    if (element) {
      targets = collectTargets(document.body);
      const target = targets.find((candidate) => candidate.element === element);
      if (target) {
        selectTarget(targets.indexOf(target));
        openTargetMenu(target, { x: event.clientX, y: event.clientY });
        return;
      }
    }
    selectTarget(null);
    openDocumentMenu({ x: event.clientX, y: event.clientY });
  }

  $effect(() => {
    window.addEventListener("keydown", onWindowKey);
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("click", onButtonClick);
    return () => {
      window.removeEventListener("keydown", onWindowKey);
      window.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("click", onButtonClick);
    };
  });

  async function bridgeWrite(text: string): Promise<boolean> {
    try {
      const invoke = tauriInvoke();
      await invoke("write_clipboard", { text });
      return true;
    } catch {
      return false;
    }
  }

  const bridgePresent = (() => {
    try {
      tauriInvoke();
      return true;
    } catch (error) {
      return error instanceof SourceUnavailableError ? false : true;
    }
  })();

  let occurrences: SearchOccurrence[] = $derived.by(() => {
    if (!view.board || query.trim().length === 0) return [];
    return searchUnits(projectBoard(view.board), query);
  });

  $effect(() => {
    if (!contentHost) return;
    // the history mode paints its own revisions; the plain render
    // waits until the mode leaves
    if (historyMode) return;
    const currentBoard = board;
    const currentLocation = location;
    contentHost.textContent = "";
    if (!currentBoard || !currentLocation) return;
    const generation = model.currentGeneration;
    const units = projectBoard(currentBoard);
    const unitIndex = new Map<number, string>();
    for (const unit of units) unitIndex.set(unit.pos.start.offset, unit.id);
    const outcomes: RenderOutcome[] = [];
    const hooks = {
      unitIndex,
      onDiagnostic: (diagnostic: Diagnostic) => {
        outcomes.push({ kind: "html", status: "reported", message: diagnostic.message, line: diagnostic.line });
      },
    };
    const rendered = renderBoard(currentBoard, hooks);
    contentHost.append(rendered);
    void hydrate(contentHost, model, currentLocation, generation, outcomes).then((all) => {
      model.setRenderOutcomes(generation, all);
    });
  });

  $effect(() => {
    document.documentElement.dataset.theme = view.theme;
  });

  // The look: configuration from the platform config file (t-24@r3),
  // applied to the document root as the colour token map. The theme
  // marker above stays for the inspection contract only; light or dark
  // is the scheme's own. Any change to the source or the test override
  // re-resolves and re-applies.
  let lookVersion = $state(0);
  let lookNotice = $state<string | null>(null);
  let lookNoticeKind = $state<"absent" | "problem" | null>(null);
  let lastSource: LookSource | null = null;
  function lookChanged(source: LookSource | null): void {
    lastSource = source;
    const look = resolveLook(source);
    applyLook(document.documentElement, look);
    lookNotice = look.notice;
    lookNoticeKind = look.noticeKind;
    lookVersion += 1;
  }
  $effect(() => {
    void lookVersion; // re-apply when the override changes
    const look = resolveLook(lastSource);
    applyLook(document.documentElement, look);
    lookNotice = look.notice;
    lookNoticeKind = look.noticeKind;
  });
  $effect(() => {
    void startLook(lookChanged);
  });

  // Inspection test surfaces (t-24@r3): in-memory, never presented as
  // human functionality.
  function setInspectedLook(o: { schemeText?: string; weight?: LookWeight } | null): void {
    setLookOverride(o);
    lookVersion += 1;
  }

  $effect(() => {
    if (contentHost) bindDiagramExpansion(contentHost);
  });

  // The drafts and the draft editor paint in place, after the parts
  // they annotate; a drafts change repaints only the draft blocks,
  // never the document.
  function paintDrafts(): void {
    if (!contentHost) return;
    contentHost.querySelectorAll(":scope > .draft-annotation").forEach((element) => element.remove());
    // a cleared editor (cancel, save) leaves its element behind unless
    // the paint removes it: the element tracks the state exactly. The
    // removal runs only when the state holds no editor - a paint that
    // rides a render-outcome landing mid-typing must not replace the
    // mounted textarea under the typing hand. The editor sits in the
    // render root beside the blocks, one level below the content host.
    if (!draftEditor || draftEditorIndex === null) {
      contentHost.querySelectorAll(":scope > .board > .draft-editor").forEach((element) => element.remove());
    }
    if (!view.board) return;
    const blocks = contentBlocks();
    for (const draft of drafts) {
      const match = /^\/content\/(\d+)$/.exec(draft.path);
      const index = match?.[1] === undefined ? -1 : Number(match[1]);
      const block = blocks[index];
      if (!block) continue;
      const el = document.createElement("div");
      el.className = "block draft-annotation";
      el.setAttribute("data-draft-path", draft.path);
      const text = document.createElement("p");
      text.className = "draft-text";
      text.textContent = draft.text;
      el.append(text);
      block.insertAdjacentElement("afterend", el);
    }
    if (draftEditor && draftEditorIndex !== null) {
      const block = blocks[draftEditorIndex];
      // the mounted editor for THIS path stays exactly as it is: it
      // self-syncs draftEditor.text through its own input listener, and
      // rebuilding it per keystroke replaces the textarea under the
      // typing hand. Only a genuinely different editor (a new path)
      // re-renders.
      const existingEditor = contentHost.querySelector(":scope > .board > .draft-editor");
      if (existingEditor && existingEditor.getAttribute("data-editor-path") === draftEditor.path) {
        return;
      }
      existingEditor?.remove();
      if (block) {
        const el = document.createElement("div");
        el.className = "draft-editor";
        el.dataset.editorPath = draftEditor.path;
        const area = document.createElement("textarea");
        area.className = "draft-editor-input";
        area.value = draftEditor.text;
        area.setAttribute("aria-label", "annotation draft");
        area.dataset.control = "draft:input";
        area.addEventListener("input", () => {
          if (draftEditor) draftEditor.text = area.value;
        });
        area.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            cancelDraft();
          } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            saveDraft();
          }
          event.stopPropagation();
        });
        const actions = document.createElement("div");
        actions.className = "draft-editor-actions";
        const save = document.createElement("button");
        save.type = "button";
        save.textContent = "save";
        save.dataset.control = "draft:save";
        save.addEventListener("click", () => saveDraft());
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.textContent = "cancel";
        cancel.dataset.control = "draft:cancel";
        cancel.addEventListener("click", () => cancelDraft());
        actions.append(save, cancel);
        el.append(area, actions);
        block.insertAdjacentElement("afterend", el);
        window.requestAnimationFrame(() => area.focus());
      }
    }
  }

  $effect(() => {
    void drafts;
    void draftEditor;
    void draftEditorIndex;
    void view.renderGeneration;
    paintDrafts();
  });

  $effect(() =>
    installCopyHandler(() => view.copyMode, {
      // the bridge write exists only inside the native shell; the
      // browser keeps the engine's copy event path
      bridgeWrite: bridgePresent ? (text) => bridgeWrite(text) : undefined,
      onHandled: (mode, text) => {
        (window as unknown as { __lastCopy?: { mode: CopyMode; text: string } }).__lastCopy = { mode, text };
      },
    }),
  );

  $effect(() => {
    const w = window as unknown as Record<string, unknown>;
    w.__saccadeApp = {
      inspection: () => inspect(model),
      open: (machine: Machine, path: string) => void model.open({ machine, path }),
      setCopyMode: (mode: CopyMode) => model.setCopyMode(mode),
      setTheme: (theme: ViewerState["theme"]) => model.setTheme(theme),
      // the look's test surfaces: in-memory overrides and a summary
      setLook: setInspectedLook,
      look: () => lookSummary(resolveLook(lastSource)),
      // the words' test surface: the lanes' injection, named as such
      setWords: (next: WordsTable) => setWords(next),
      // the deck's test surface (t-21): the standing deck state for
      // HUD and key checks in the browser lane, where no core exists.
      // Test-only machinery, named as such, never human functionality.
      deck: () =>
        deck === null
          ? null
          : {
              name: deck.name,
              slides: [...deck.slides],
              position: deck.position,
              entered: deck.entered,
              feedLost: deckFeedLost,
            },
      deckTest: (patch: {
        slides?: string[];
        position?: number | null;
        entered?: boolean;
      }) => {
        const current = deck ?? newDeck("deck-" + "0".repeat(24));
        deck = {
          ...current,
          slides: patch.slides ?? current.slides,
          position: patch.position ?? current.position,
          entered: patch.entered ?? current.entered,
        };
      },
      deckShow: (index: number) => void showSlide(index),
    };
  });

  // the deck's mount-time offer (see the note at deckStateText)
  $effect(() => {
    void ensureDeck();
  });

  // -- the strip (round 16, t-34@r6 changes 5 and 6) -----------------------
  //
  // One line: the window's location (in history, the revision shown,
  // with the current one beside it), then state only when it matters,
  // then the link. No fields: a tracker is reached by typing its
  // location in the picker.
  let stripLocation = $derived.by(() => {
    if (!view.location) return null;
    const shown = stripFormat(view.location);
    const revision = historyCommit?.revision ?? null;
    return revision === null ? shown : `${shown.replace(/@r\d+$/, "")}@r${revision}`;
  });
  let stripCurrent = $derived.by(() => {
    const revision = historyCommit?.revision ?? null;
    if (revision === null || historyMode === null) return null;
    return revision < historyMode.newest ? `current r${historyMode.newest}` : null;
  });
  let stripTracker = $derived(
    view.location && view.location.machine.startsWith("tracker:")
      ? view.location.machine.slice("tracker:".length)
      : null,
  );
  let stripState = $derived.by(() => {
    if (view.status === "loading") return "loading";
    if (view.status === "invalid") return "refused - see below";
    if (lookNoticeKind === "problem") return "look configuration problem - see help";
    return null;
  });

  function jumpToMatch(occurrence: SearchOccurrence): void {
    const elById = document.querySelector<HTMLElement>(`[data-unit-id="${occurrence.unit.id}"]`);
    if (elById) {
      elById.scrollIntoView({ block: "center", behavior: "auto" });
      flash(elById);
      return;
    }
    // Prose units have no dedicated element: find the nearest rendered
    // block at or above the unit's source line.
    const candidates = Array.from(document.querySelectorAll<HTMLElement>("[data-line]"));
    let best: HTMLElement | null = null;
    let bestLine = -1;
    for (const candidate of candidates) {
      const line = Number(candidate.getAttribute("data-line"));
      if (line <= occurrence.unit.pos.start.line && line > bestLine) {
        best = candidate;
        bestLine = line;
      }
    }
    if (best) {
      best.scrollIntoView({ block: "center", behavior: "auto" });
      flash(best);
    }
  }

  function flash(target: HTMLElement): void {
    target.classList.add("flash");
    window.setTimeout(() => target.classList.remove("flash"), 900);
  }

  function nextMatch(delta: number): void {
    if (occurrences.length === 0) return;
    activeMatch = (activeMatch + delta + occurrences.length) % occurrences.length;
    const occurrence = occurrences[activeMatch];
    if (occurrence) jumpToMatch(occurrence);
  }
</script>

<div class="app">
  <Strip
    kind={view.instrument ? view.instrument.kind : "viewer"}
    tracker={stripTracker}
    location={stripLocation}
    current={stripCurrent}
    state={stripState}
    note={copyNote}
    deckName={deck ? deck.name : null}
    deckState={deckStateText || null}
    onCopyCue={copyCueCommand}
    marginalia={marginaliaActive}
    {zoom}
    jumpChip={jumpChipShown}
    link={linkTargetName}
  />

  {#if searchOpen}
  <div class="search-bar" role="search">
    <input
      type="search"
      placeholder="search prose and math"
      aria-label="search"
      data-control="search"
      bind:this={searchInput}
      bind:value={query}
      onkeydown={(event) => {
        if (event.key === "Enter") nextMatch(event.shiftKey ? -1 : 1);
      }}
    />
    <span class="search-state" data-control="search:state">
      {#if query.trim().length === 0}
        type to search
      {:else if occurrences.length === 0}
        no matches
      {:else}
        {activeMatch + 1} / {occurrences.length}
      {/if}
    </span>
    <button type="button" data-control="search:prev" onclick={() => nextMatch(-1)} aria-label="previous match">prev</button>
    <button type="button" data-control="search:next" onclick={() => nextMatch(1)} aria-label="next match">next</button>
  </div>
  {/if}

  {#if publishOpen}
  <div class="publish-overlay" role="dialog" aria-label="publish annotations" data-control="publish:overlay">
    <div class="publish-panel">
      <p class="history-position">publish {drafts.length} draft{drafts.length === 1 ? "" : "s"} as one commit, under your own identity</p>
      <input
        type="text"
        class="picker-input"
        placeholder="the commit's one message"
        aria-label="publish message"
        data-control="publish:message"
        bind:value={publishMessage}
      />
      {#if publishNote}
        <p class="publish-note" data-control="publish:note">{publishNote}</p>
      {/if}
      <div class="draft-editor-actions">
        <button type="button" data-control="publish:go" disabled={publishBusy} onclick={() => void publishDrafts()}>publish</button>
        <button type="button" data-control="publish:cancel" onclick={cancelPublish}>cancel</button>
      </div>
    </div>
  </div>
  {/if}

  {#if helpOpen}
    <Help kind={windowKind()} keys={helpKeys()} commands={spaceCommands()} {lookNotice} />
  {/if}

  {#if pickerOpen}
    <Picker
      bind:query={pickerQuery}
      bind:active={pickerActive}
      results={pickerResults}
      placeholder={homeTracker() === null ? "no tracker yet - type its location with the host once: host:8811/saccade/tasks" : "tasks, boards, files - a typed location always resolves"}
      onKey={onPickerKey}
      onChoose={(item) => void pickerOpenResult(item)}
    />
  {/if}

  {#if overlay}
  <div class="overlay-backdrop" role="dialog" aria-label="{overlay.title} details" data-control="instrument:overlay">
    <div class="overlay-panel">
      <p class="sec">{overlay.title}</p>
      {#if overlay.lane}
        <div class="activity-lane" data-control="instrument:lane">
          <div class="activity-axis">
            {#each overlay.lane.positions as pos, i (i)}
              {#if pos.x >= 0}
                <span
                  class="activity-event"
                  class:activity-clamped={pos.clamped}
                  style:left={`${pos.x * 100}%`}
                  title={`${pos.entry.label} (${pos.entry.kind})`}
                ></span>
              {/if}
            {/each}
            <span class="activity-mark" style:left={`${overlay.lane.sixHourMark * 100}%`} title="6h"></span>
          </div>
          <p class="dim3 activity-scale">24h <span>6h mark</span> now</p>
        </div>
      {/if}
      {#if overlay.facts.length > 0}
        <div class="overlay-facts">
          {#each overlay.facts as fact (fact.label)}
            <p class="overlay-fact"><span class="dim3">{fact.label}</span> <span>{fact.value}</span></p>
          {/each}
        </div>
      {/if}
      {#if overlay.events.length > 0}
        <div class="overlay-events" data-control="instrument:events">
          {#each overlay.events as entry, i (i)}
            <p class="overlay-event"><span class="dim3">{entry.kind}</span> {entry.label}</p>
          {/each}
        </div>
      {:else if !overlay.lane}
        <p class="dim3 instrument-empty">nothing yet</p>
      {/if}
      <p class="dim3 overlay-hint">Escape leaves</p>
    </div>
  </div>
  {/if}


  {#if publishNote && !publishOpen}
    <p class="look-notice" data-control="action:note" aria-live="polite">{publishNote}</p>
  {/if}

  {#if jumpMode !== null}
    <div class="jump-overlay" role="dialog" aria-label="window jump" data-control="jump:overlay">
      {#if jumpMode.kind === "labels"}
        <p class="jump-title">window jump: type a label</p>
        <p class="jump-typed" data-control="jump:typed">{jumpMode.typed.length > 0 ? jumpMode.typed : "..."}</p>
      {:else if jumpMode.kind === "link"}
        <p class="jump-title">link this window to: type a label</p>
        <p class="jump-typed" data-control="link:typed">{jumpMode.typed.length > 0 ? jumpMode.typed : "..."}</p>
      {:else}
        <p class="jump-title">windows here, then detached</p>
        <div class="jump-list" data-control="jump:list">
          {#each visibleJumpEntries(jumpMode) as entry, index}
            {@const selected = index === jumpMode.selected}
            <button
              type="button"
              class="jump-row"
              class:jump-active={selected}
              data-control="jump:row"
              onclick={() => void jumpAct(entry)}
            >
              <span class="jump-entry-chip">{entry.chip ?? ""}</span>
              <span class="jump-entry-kind">{entry.kind === "window" ? "window" : "detached"}</span>
              <span class="jump-entry-name">{entry.name}</span>
              {#if selected && jumpMode.armed === entry.stateId}
                <span class="jump-armed" data-control="jump:armed">^x again discards the state on every device</span>
              {/if}
            </button>
          {:else}
            <p class="picker-empty">no windows and no detached states</p>
          {/each}
        </div>
        <p class="jump-hint">enter jumps there or resumes here | ^x ^x discards | escape leaves</p>
      {/if}
      {#if jumpNote}
        <p class="jump-note" data-control="jump:note">{jumpNote}</p>
      {/if}
    </div>
  {/if}

  {#if askOnce !== null}
    <div class="jump-overlay" role="dialog" aria-label="open elsewhere" data-control="ask:overlay">
      {#if askOnce.showDetached}
        <p class="jump-title">open in a detached state</p>
        <div class="jump-list" data-control="ask:detached">
          {#each askOnce.detached as entry, index}
            <button
              type="button"
              class="jump-row"
              class:jump-active={index === askOnce.detachedSelected}
              data-control="ask:row"
              onclick={() => void askOnceAnswer("detached", entry)}
            >
              <span class="jump-entry-kind">detached</span>
              <span class="jump-entry-name">{entry.name}</span>
            </button>
          {:else}
            <p class="picker-empty">no detached states</p>
          {/each}
        </div>
        <p class="jump-hint">enter opens there{askOnce.act === "link" ? " and becomes the link" : ""} | escape leaves</p>
      {:else}
        <p class="jump-title">open elsewhere: a label, n new, / a detached one</p>
        <p class="jump-typed" data-control="ask:typed">{askOnce.typed.length > 0 ? askOnce.typed : "..."}</p>
        <p class="jump-hint">the answer {askOnce.act === "link" ? "becomes the link" : "changes no link"} | escape leaves</p>
      {/if}
      {#if jumpNote}
        <p class="jump-note" data-control="ask:note">{jumpNote}</p>
      {/if}
    </div>
  {/if}

  {#if view.errors.length > 0}
    <section class="errors" aria-label="load errors">
      {#each view.errors as issue}
        <p class="render-error"><span class="err-body"><strong>{issue.code}</strong> {issue.message}{#if issue.line !== undefined} (line {issue.line}, column {issue.column ?? 0}){/if}</span></p>
      {/each}
      {#if view.laneMessage}
        <p class="render-error"><span class="err-body">{view.laneMessage}</span></p>
      {/if}
    </section>
  {/if}

  <div class="board-wrap" class:history-active={historyActive} class:marginalia-active={marginaliaActive} class:instrument-active={view.instrument !== null} bind:this={mainWrap}>
    <main class="content" bind:this={contentHost} aria-label="board content"></main>
    {#if view.instrument}
      <main class="content instrument" bind:this={instrumentHost} aria-label="{view.instrument.kind} instrument" data-instrument={view.instrument.kind}>
        {#if view.instrument.kind === "tasks"}
          <TasksWindow sections={taskSections} shapes={taskRowShapes} onSelect={selectInstrumentRow} />
        {:else}
          <AgentsWindow sections={agentSections} here={stripTracker ?? ""} onSelect={selectInstrumentRow} />
        {/if}
      </main>
    {/if}
    {#if historyActive && historyCommit}
      <HistoryLog
        task={view.board?.task ?? ""}
        commits={historyCommits}
        at={historyCommit.revision ?? 0}
        changes={historyChanges}
        onGo={(revision) => void historyGo(revision)}
      />
    {/if}
    {#if marginaliaActive}
      <aside class="marginalia-margin" data-control="marginalia:margin">
        {#each marginaliaRows as run, index (run.indexes[0] ?? index)}
          {@const commit = marginaliaCommitFor(run)}
          <button type="button" class="marginalia-row" data-control="marginalia:row" onclick={() => scrollToRun(run)}>
            <p class="marginalia-parts">parts {run.indexes[0] !== undefined ? run.indexes[0] + 1 : ""}{run.indexes.length > 1 ? `-${(run.indexes[run.indexes.length - 1] ?? 0) + 1}` : ""} | r{run.lastChanged}</p>
            <p class="marginalia-commit">{commit?.message ?? commit?.grounds ?? ""}</p>
          </button>
        {/each}
        <p class="history-hint">space m leaves</p>
      </aside>
    {/if}
  </div>
</div>
