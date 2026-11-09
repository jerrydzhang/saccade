<script lang="ts">
  import { ViewerModel } from "./lib/board/viewmodel.ts";
  import { tauriInvoke, SourceUnavailableError, type Machine, type Location } from "./lib/sources.ts";
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
  import { inspect } from "./lib/board/inspection.ts";
  import type { CopyMode } from "./lib/copy.ts";
  import type { RenderOutcome, ViewerState } from "./lib/board/viewmodel.ts";
  import type { Board } from "./lib/board/model.ts";
  import type { Diagnostic } from "./lib/board/errors.ts";
  import { serverHost, setServerHost } from "./lib/tracker.ts";
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
  void openTestBoard();

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
    if (helpOpen) return;

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
        // space leads the viewer's own commands; each command lands
        // with its own chapter (picker, history, marginalia, publish)
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
      case "?":
        event.preventDefault();
        helpOpen = true;
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
    </div>
  </div>
  {/if}

  {#if lookNotice}
    <p class="look-notice" aria-live="polite">{lookNotice}</p>
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

  <main class="content" bind:this={contentHost} aria-label="board content"></main>
</div>
