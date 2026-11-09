<script lang="ts">
  import { ViewerModel } from "./lib/board/viewmodel.ts";
  import { tauriInvoke, SourceUnavailableError, type Machine, type Location } from "./lib/sources.ts";
  import { projectBoard, searchUnits, type SearchOccurrence } from "./lib/board/projection.ts";
  import { renderBoard } from "./lib/render.ts";
  import { hydrate } from "./lib/board/hydrate.ts";
  import { bindDiagramExpansion } from "./lib/overlay.ts";
  import { installCopyHandler } from "./lib/copydom.ts";
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

  // Key-driven chrome: Ctrl+F opens the search, Escape closes it,
  // Ctrl+Shift+C switches between Source and Plain copy. No visible
  // mode buttons: Source is the default and Plain is one keystroke.
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
    if (event.key === "Escape" && searchOpen) {
      event.preventDefault();
      closeSearch();
    }
  }

  $effect(() => {
    window.addEventListener("keydown", onWindowKey);
    return () => window.removeEventListener("keydown", onWindowKey);
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
