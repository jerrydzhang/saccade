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
  import { formatLocation, parseLocation } from "./lib/location.ts";
  import {
    annotatedBoard,
    publishPayload,
    publishAnnotations,
    type Draft,
  } from "./lib/annotations.ts";
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
  import { pickerLabelKind, rankedPicker, type PickerItem } from "./lib/picker.ts";
  import { trackerSourcesForPicker } from "./lib/sources.ts";
  import { HistoryMode, marginaliaRuns, type MarginaliaRun } from "./lib/historymode.ts";
  import { ADD_CLOSE, ADD_OPEN, DEL_CLOSE, DEL_OPEN } from "./lib/history.ts";
  import { pageFact, type HistoryEntry } from "./lib/pages.ts";
  import { parseTaskSpec } from "./lib/location.ts";
  import { inspect } from "./lib/board/inspection.ts";
  import type { CopyMode } from "./lib/copy.ts";
  import type { RenderOutcome, ViewerState } from "./lib/board/viewmodel.ts";
  import type { Board } from "./lib/board/model.ts";
  import type { Diagnostic } from "./lib/board/errors.ts";
  import { serverHost, setServerHost, TrackerSource } from "./lib/tracker.ts";
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

  // The lanes' test-open channel: the one startup open, riding the
  // same model.openInput the controls use. Test-only machinery
  // (t-24@r11) - the launch argument and the local-device path form
  // left the human surface with them, and a human run starts empty.
  // Each step logs a marker so the native evidence can tell "webview
  // never loaded" (no startup line), "invoke failed", and "open
  // failed" apart.
  async function openTestBoard(): Promise<void> {
    let invoke: ReturnType<typeof tauriInvoke>;
    try {
      invoke = tauriInvoke();
    } catch (error) {
      if (error instanceof SourceUnavailableError) {
        console.log("[viewer] startup: no bridge; explicit controls only");
        return;
      }
      throw error;
    }
    try {
      console.log("[viewer] startup: bridge present");
      const spec = (await invoke("test_open")) as string | null;
      if (!spec) {
        console.log("[viewer] no test open; starting empty");
        return;
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
    } catch (error) {
      console.error(`[viewer] test open board FAILED: ${String(error)}`);
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
      console.log("[viewer] windows: no tracker server set; the window retries when one appears");
      scheduleWindowRetry();
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
  void openTestBoard();
  void startStateWindow();

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
    // no link (or a once act): ask once
    blurForJump();
    jumpNote = null;
    try {
      const invoke = await shellInvoke();
      const windows = (await invoke("suite_windows")) as { label: string }[];
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

  let hostPath = $state("");
  let query = $state("");
  let activeMatch = $state(0);
  let searchOpen = $state(false);
  let searchInput: HTMLInputElement | undefined = $state(undefined);
  let contentHost: HTMLElement | undefined = $state(undefined);

  function openSearch(): void {
    searchOpen = true;
    window.requestAnimationFrame(() => searchInput?.focus());
  }

  function closeSearch(): void {
    searchOpen = false;
  }

  // Key-driven chrome: the select-first keymap (t-14 "Interaction")
  // plus the standing search and copy-mode keys. No visible mode
  // buttons: every mode is a key, and ? lists them on demand.
  type KeyMode = "normal" | "labels" | "leader" | "gg";
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

  function actsFor(target: SelectableTarget): MenuAct[] {
    const acts: MenuAct[] = [];
    const open = (): void => {
      const path = target.element.getAttribute("data-open-path");
      const location = view.location;
      if (path && location) void model.open({ machine: location.machine, path });
    };
    const enlarge = (): void => target.element.click();
    const copy = (): void => {
      // the bridge write exists only inside the native shell; the
      // browser keeps the engine's clipboard
      if (bridgePresent) void bridgeWrite(target.text);
      else void navigator.clipboard?.writeText(target.text).catch(() => {});
    };
    if (target.kind === "citation" || target.kind === "incorporation") {
      acts.push({ key: "Enter", label: "open the cited revision", onAct: open });
      // the two acts, pointer-reachable (t-14: every act is reachable
      // by the pointer too)
      acts.push({
        key: "o",
        label: "open elsewhere through the link",
        onAct: () => {
          const current = view.location;
          const path = target.element.getAttribute("data-open-path");
          if (current && path) {
            const spec = citationSpec(current, path);
            if (spec !== null) void openSpecElsewhere(spec, "link");
          }
        },
      });
      acts.push({
        key: "i",
        label: "open elsewhere once, the link unchanged",
        onAct: () => {
          const current = view.location;
          const path = target.element.getAttribute("data-open-path");
          if (current && path) {
            const spec = citationSpec(current, path);
            if (spec !== null) void openSpecElsewhere(spec, "once");
          }
        },
      });
      acts.push({
        key: "w",
        label: "open in a new window",
        onAct: () => {
          const current = view.location;
          const path = target.element.getAttribute("data-open-path");
          if (current && path) {
            const spec = citationSpec(current, path);
            if (spec !== null) void openSpecElsewhere(spec, "new");
          }
        },
      });
    }
    if (target.kind === "diagram") {
      acts.push({ key: "Enter", label: "enlarge", onAct: enlarge });
    }
    if (target.kind === "control") {
      acts.push({ key: "Enter", label: "activate", onAct: enlarge });
    }
    if (isTextKind(target.kind)) {
      acts.push({ key: "c", label: "copy the part", onAct: copy });
    }
    if (acts.length === 0) {
      acts.push({ key: "Enter", label: "open the menu", onAct: () => {} });
    }
    return acts;
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
    closeMenu = openContextMenu({
      ...at,
      acts: actsFor(target),
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
  let pickerInput: HTMLInputElement | undefined = $state(undefined);

  function pickerCandidates(): PickerItem[] {
    const items: PickerItem[] = [];
    for (const source of trackerSourcesForPicker()) {
      for (const fact of source.seenTasks) {
        items.push({ location: fact.spec.startsWith("t-") ? `${source.tracker}/${fact.spec}` : fact.spec, kind: "task", name: fact.name ?? fact.spec });
      }
    }
    // the locations the shown page references: the citations carry
    // their target in data-open-path
    for (const element of document.querySelectorAll<HTMLElement>("[data-open-path]")) {
      const path = element.getAttribute("data-open-path");
      const location = view.location;
      if (path && location) {
        items.push({ location: `${location.machine.startsWith("tracker:") ? location.machine.slice("tracker:".length) : location.machine}/${path}`, kind: "board", name: path });
      }
    }
    return items;
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

  function openPicker(): void {
    pickerOpen = true;
    pickerQuery = "";
    pickerActive = 0;
    window.requestAnimationFrame(() => pickerInput?.focus());
  }

  function closePicker(): void {
    pickerOpen = false;
  }

  function pickerMove(delta: number): void {
    const count = pickerResults.length;
    if (count === 0) return;
    pickerActive = (pickerActive + delta + count) % count;
  }

  async function pickerOpenResult(item: PickerItem): Promise<void> {
    closePicker();
    await model.openInput(item.location);
  }

  function onPickerKey(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      event.preventDefault();
      closePicker();
      return;
    }
    if (event.key === "ArrowDown" || (event.key === "j" && !event.ctrlKey)) {
      event.preventDefault();
      pickerMove(1);
      return;
    }
    if (event.key === "ArrowUp" || (event.key === "k" && !event.ctrlKey)) {
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
    const blocks = Array.from(contentHost.querySelectorAll<HTMLElement>(":scope > .block"));
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
    const historyLocation = view.location;
    if (!historyLocation) return;
    void hydrate(contentHost, model, historyLocation, generation, outcomes).then((all) => {
      model.setRenderOutcomes(generation, all);
    });
    historyPainted = true;
  }

  function paintHistoryRuler(): void {
    if (!contentHost) return;
    let ruler = contentHost.querySelector(":scope > .history-ruler");
    if (!ruler) {
      ruler = document.createElement("div");
      ruler.className = "history-ruler";
      contentHost.append(ruler);
    }
    ruler.textContent = "";
    for (const block of Array.from(contentHost.querySelectorAll<HTMLElement>(":scope > .block"))) {
      if (block.querySelector(".diff-add, .diff-del") === null) continue;
      const tick = document.createElement("span");
      tick.className = "history-ruler-tick";
      tick.style.top = `${block.offsetTop}px`;
      tick.style.height = `${block.offsetHeight}px`;
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

  function jumpToChange(delta: number): void {
    if (!contentHost) return;
    const marks = Array.from(contentHost.querySelectorAll<HTMLElement>(".diff-add, .diff-del"));
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
    const blockIndex = Array.from(contentHost?.querySelectorAll<HTMLElement>(":scope > .block") ?? []).indexOf(block);
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
  let drafts = $state<Draft[]>([]);
  let draftEditor = $state<{ path: string; text: string } | null>(null);
  let draftEditorIndex = $state<number | null>(null);
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
  function actPartIndex(): number | null {
    if (!contentHost || !view.board) return null;
    const blocks = Array.from(contentHost.querySelectorAll<HTMLElement>(":scope > .block"));
    let element: HTMLElement | null = selected?.element ?? null;
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
    const spec = parseTaskSpec(location?.path ?? "");
    if (!location || !spec) {
      publishNote = "annotations draft on a task board; this location is not one";
      return;
    }
    if (spec.revision !== null) {
      // t-24@r11: only the current revision drafts; a pinned older
      // revision refuses, saying where annotation is possible
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
    const next = drafts.filter((draft) => draft.path !== editor.path);
    if (text.trim().length > 0) next.push({ path: editor.path, text: text.replace(/\s+$/, "") });
    updateDrafts(next);
    draftEditor = null;
    draftEditorIndex = null;
  }

  function cancelDraft(): void {
    draftEditor = null;
    draftEditorIndex = null;
  }

  let publishPanel: { open: boolean } | null = null;

  function openPublish(): void {
    if (!view.board) return;
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
      const spec = parseTaskSpec(location.path);
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
    const blocks = Array.from(contentHost.querySelectorAll<HTMLElement>(":scope > .block"));
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
    if (event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === "f") {
      event.preventDefault();
      openSearch();
      return;
    }
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
      case "leader": {
        event.preventDefault();
        keyMode = "normal";
        // space leads this instrument's own commands only (t-14)
        if (event.key === " " && !pickerOpen) {
          openPicker();
        } else if (event.key === "h") {
          void toggleHistory();
        } else if (event.key === "m") {
          void toggleMarginalia();
        } else if (event.key === "p") {
          openPublish();
        } else if (event.key === "l") {
          // space l: link this window by labels (t-14)
          void enterLinkLabels();
        } else if (event.key === "u") {
          // space u: unlink this window
          void unlinkHere();
        }
        // the other leader commands (marginalia, publish) land with
        // their own chapters
        return;
      }
      case "gg": {
        event.preventDefault();
        keyMode = "normal";
        if (event.key === "g") scrollTop(contentHost);
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
        break;
      case "l":
        if (historyMode) {
          event.preventDefault();
          void historyStep("newer");
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
        if (selected) moveSelection(1);
        else scrollByLines(contentHost, 2);
        return;
      case "k":
        event.preventDefault();
        if (selected) moveSelection(-1);
        else scrollByLines(contentHost, -2);
        return;
      case "d":
        event.preventDefault();
        scrollByPages(contentHost, 0.5);
        return;
      case "u":
        event.preventDefault();
        scrollByPages(contentHost, -0.5);
        return;
      case "g":
        event.preventDefault();
        keyMode = "gg";
        return;
      case "G":
        event.preventDefault();
        scrollBottom(contentHost);
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
        keyMode = "leader";
        return;
    }
    if (event.ctrlKey && !event.shiftKey) {
      if (event.key === "f") {
        // page down
        event.preventDefault();
        scrollByPages(contentHost, 1);
        return;
      }
      if (event.key === "b") {
        event.preventDefault();
        scrollByPages(contentHost, -1);
        return;
      }
      if (event.key === "d") {
        event.preventDefault();
        scrollByPages(contentHost, 0.5);
        return;
      }
      if (event.key === "u") {
        event.preventDefault();
        scrollByPages(contentHost, -0.5);
        return;
      }
    }
  }

  function onContextMenu(event: MouseEvent): void {
    event.preventDefault();
    const element = (event.target as HTMLElement | null)?.closest<HTMLElement>(
      ".block, .diagram-frame, .citation-link, [data-control]",
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
    return () => {
      window.removeEventListener("keydown", onWindowKey);
      window.removeEventListener("contextmenu", onContextMenu);
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
  let lastSource: LookSource | null = null;
  function lookChanged(source: LookSource | null): void {
    lastSource = source;
    const look = resolveLook(source);
    applyLook(document.documentElement, look);
    lookNotice = look.notice;
    lookVersion += 1;
  }
  $effect(() => {
    void lookVersion; // re-apply when the override changes
    const look = resolveLook(lastSource);
    applyLook(document.documentElement, look);
    lookNotice = look.notice;
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
    if (!view.board) return;
    const blocks = Array.from(contentHost.querySelectorAll<HTMLElement>(":scope > .block"));
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
      if (block) {
        const existing = contentHost.querySelector(":scope > .draft-editor");
        existing?.remove();
        const el = document.createElement("div");
        el.className = "draft-editor";
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
    w.__boardViewer = {
      inspection: () => inspect(model),
      open: (machine: Machine, path: string) => void model.open({ machine, path }),
      setCopyMode: (mode: CopyMode) => model.setCopyMode(mode),
      setTheme: (theme: ViewerState["theme"]) => model.setTheme(theme),
      // the look's test surfaces: in-memory overrides and a summary
      setLook: setInspectedLook,
      look: () => lookSummary(resolveLook(lastSource)),
    };
  });

  async function openHost(): Promise<void> {
    if (hostPath) await model.openInput(hostPath);
  }

  // The one tracker setting: the server host, the address the tracker
  // publishes beside its database. Set once; every tracker location
  // rides it.
  let serverInput = $state(serverHost() ?? "");
  function saveServer(): void {
    setServerHost(serverInput);
  }

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
  <header class="app-header">
    {#if jumpChipShown !== null}
      <span class="jump-chip-self" data-control="jump:self">{jumpChipShown}</span>
    {/if}
    {#if linkTargetName !== null}
      <span class="link-target" data-control="link:target" title="this window's link">-&gt; {linkTargetName}</span>
    {/if}
    <span class="identity">
      {#if view.location}
        <strong>{view.location.machine}</strong>
        <code class="location-path" data-control="board:location">{view.location.path}</code>
        <span class="status status-{view.status}">{view.status}</span>
        {#if view.board?.task}
          <span class="task-meta">task: <code>{view.board.task}</code> (authored reference)</span>
        {/if}
      {:else}
        <span class="status status-empty">no board open</span>
      {/if}
    </span>
    <span class="controls">
      <label class="control">location
        <input type="text" bind:value={hostPath} placeholder="saccade/t-N, /path on the tracker's machine, or machine:/path" aria-label="board location" data-control="open:host-path" />
      </label>
      <button type="button" data-control="open:host" onclick={() => void openHost()}>open</button>
      <label class="control">tracker server
        <input type="text" bind:value={serverInput} placeholder="127.0.0.1:8811" aria-label="tracker server host" data-control="server:host" />
      </label>
      <button type="button" data-control="server:save" onclick={saveServer}>save</button>
    </span>
  </header>

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
  <div class="help-overlay" role="dialog" aria-label="keys and commands" data-control="help:overlay">
    <div class="help-panel">
      <div class="help-section">
        <p class="help-title">motion</p>
        <p class="help-row"><span class="ctx-menu-key">j k</span> line down / up</p>
        <p class="help-row"><span class="ctx-menu-key">Ctrl-f Ctrl-b</span> page down / up</p>
        <p class="help-row"><span class="ctx-menu-key">Ctrl-d Ctrl-u</span> half page down / up (d u)</p>
        <p class="help-row"><span class="ctx-menu-key">gg G</span> top / bottom</p>
        <p class="help-row"><span class="ctx-menu-key">Ctrl-o Ctrl-i</span> back / forward</p>
      </div>
      <div class="help-section">
        <p class="help-title">select</p>
        <p class="help-row"><span class="ctx-menu-key">s</span> labels on what can be selected</p>
        <p class="help-row"><span class="ctx-menu-key">label</span> select it and open its menu</p>
        <p class="help-row"><span class="ctx-menu-key">text</span> narrows the text targets</p>
        <p class="help-row"><span class="ctx-menu-key">Enter</span> the selected target's acts</p>
        <p class="help-row"><span class="ctx-menu-key">right-click</span> the same menu at the pointer</p>
        <p class="help-row"><span class="ctx-menu-key">Escape</span> leaves the mode</p>
      </div>
      <div class="help-section">
        <p class="help-title">viewer</p>
        <p class="help-row"><span class="ctx-menu-key">space</span> this instrument's commands</p>
        <p class="help-row"><span class="ctx-menu-key">Ctrl+f</span> search (Escape closes)</p>
        <p class="help-row"><span class="ctx-menu-key">Ctrl+Shift+C</span> copy source / plain</p>
        <p class="help-row"><span class="ctx-menu-key">?</span> this list</p>
      </div>
      <div class="help-section">
        <p class="help-title">windows</p>
        <p class="help-row"><span class="ctx-menu-key">'</span> labels on every suite window; type a label to jump</p>
        <p class="help-row"><span class="ctx-menu-key">"</span> windows here, then detached; enter resumes; ^x twice discards</p>
        <p class="help-row"><span class="ctx-menu-key">space l</span> link this window by labels</p>
        <p class="help-row"><span class="ctx-menu-key">space u</span> unlink this window</p>
      </div>
      <div class="help-section">
        <p class="help-title">two acts</p>
        <p class="help-row"><span class="ctx-menu-key">Enter</span> the primary act, here</p>
        <p class="help-row"><span class="ctx-menu-key">o</span> open elsewhere through the link (shift- or middle-click too)</p>
        <p class="help-row"><span class="ctx-menu-key">i</span> open elsewhere once, the link unchanged</p>
        <p class="help-row"><span class="ctx-menu-key">w</span> open in a new window</p>
      </div>
    </div>
  </div>
  {/if}

  {#if pickerOpen}
  <div class="picker-overlay" role="dialog" aria-label="picker" data-control="picker:overlay">
    <div class="picker-panel">
      <input
        type="text"
        class="picker-input"
        placeholder="tasks, boards, files - a typed location always resolves"
        aria-label="picker query"
        data-control="picker:input"
        bind:this={pickerInput}
        bind:value={pickerQuery}
        onkeydown={onPickerKey}
      />
      <div class="picker-list" data-control="picker:list">
        {#each pickerResults as item, index (item.location)}
          <button
            type="button"
            class="picker-row"
            class:picker-active={index === pickerActive}
            data-control="picker:result"
            onclick={() => void pickerOpenResult(item)}
            onmouseenter={() => (pickerActive = index)}
          >
            <span class="picker-kind">{pickerLabelKind(item).kind}</span>
            <span class="picker-name">{pickerLabelKind(item).label}</span>
          </button>
        {:else}
          <p class="picker-empty">nothing matches</p>
        {/each}
      </div>
    </div>
  </div>
  {/if}

  {#if lookNotice}
    <p class="look-notice" aria-live="polite">{lookNotice}</p>
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

  <div class="board-wrap" class:history-active={historyActive} class:marginalia-active={marginaliaActive} bind:this={mainWrap}>
    <main class="content" bind:this={contentHost} aria-label="board content"></main>
    {#if historyActive && historyCommit}
      <aside class="history-margin" data-control="history:margin">
        <p class="history-position">revision r{historyMode?.at ?? ""} of r{historyMode?.newest ?? ""}</p>
        <p class="history-commit">{historyCommit.message ?? historyCommit.grounds ?? historyCommit.explanation ?? ""}</p>
        <p class="history-hint">h l step time | n N changes | t tracks | H L the tracked part | space h leaves</p>
      </aside>
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
