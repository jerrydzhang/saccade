<script lang="ts">
  import { ViewerModel } from "./lib/viewmodel.ts";
  import { fixtureNames, tauriInvoke, SourceUnavailableError, type Machine, type Location } from "./lib/sources.ts";
  import { projectBoard, searchUnits, type SearchOccurrence } from "./lib/projection.ts";
  import { renderBoard } from "./lib/render.ts";
  import { hydrate } from "./lib/hydrate.ts";
  import { bindDiagramExpansion } from "./lib/overlay.ts";
  import { installCopyHandler } from "./lib/copydom.ts";
  import { inspect } from "./lib/inspection.ts";
  import type { CopyMode } from "./lib/copy.ts";
  import type { RenderOutcome, ViewerState } from "./lib/viewmodel.ts";
  import type { Board } from "./lib/model.ts";
  import type { Diagnostic } from "./lib/errors.ts";

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

  // Foreground explicit-board open: the shell hands the launch argument
  // over the bridge and the app opens it through the host lane - the
  // same handlers the controls use. Each step logs a marker so the
  // native evidence can tell "webview never loaded" (no startup line),
  // "invoke failed", and "open failed" apart.
  async function openInitialBoard(): Promise<void> {
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
      const laneConsole = (await invoke("lane_console_enabled")) as boolean;
      if (laneConsole) {
        (window as unknown as { __BOARD_VIEWER_DIAG__?: boolean }).__BOARD_VIEWER_DIAG__ = true;
      }
      const path = (await invoke("initial_board")) as string | null;
      if (!path) {
        console.log("[viewer] no initial board; starting empty");
        return;
      }
      console.log(`[viewer] initial board requested: ${path}`);
      await model.open({ machine: "host", path });
      const snap = model.snapshot;
      if (snap.board) {
        console.log(`[viewer] initial board opened: ${snap.board.content.length} blocks`);
      } else {
        const detail = snap.laneMessage ?? snap.errors.map((e) => e.message).join("; ");
        console.error(`[viewer] initial board FAILED: ${detail || "no board"}`);
      }
    } catch (error) {
      console.error(`[viewer] initial board FAILED: ${String(error)}`);
    }
  }
  void openInitialBoard();

  const fixtures = fixtureNames();
  let fixturePath = $state(fixtures[0] ?? "");
  let hostPath = $state("");
  let query = $state("");
  let activeMatch = $state(0);
  let showDiagnostics = $state(false);
  let contentHost: HTMLElement | undefined = $state(undefined);

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

  $effect(() => {
    if (contentHost) bindDiagramExpansion(contentHost);
  });

  $effect(() => installCopyHandler(() => view.copyMode));

  $effect(() => {
    const w = window as unknown as Record<string, unknown>;
    w.__boardViewer = {
      inspection: () => inspect(model),
      open: (machine: Machine, path: string) => void model.open({ machine, path }),
      setCopyMode: (mode: CopyMode) => model.setCopyMode(mode),
      setTheme: (theme: ViewerState["theme"]) => model.setTheme(theme),
    };
  });

  async function openFixture(): Promise<void> {
    if (fixturePath) await model.open({ machine: "fixture", path: fixturePath });
  }

  async function openHost(): Promise<void> {
    if (hostPath) await model.open({ machine: "host", path: hostPath });
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
        <code class="location-path">{view.location.path}</code>
        <span class="status status-{view.status}">{view.status}</span>
        {#if view.board?.task}
          <span class="task-meta">task: <code>{view.board.task}</code> (authored metadata)</span>
        {/if}
      {:else}
        <span class="status status-empty">no board open</span>
      {/if}
    </span>
    <span class="controls">
      <label class="control">fixture
        <select bind:value={fixturePath} aria-label="fixture board">
          {#each fixtures as name}
            <option value={name}>{name}</option>
          {/each}
        </select>
      </label>
      <button type="button" onclick={() => void openFixture()}>open fixture</button>
      <label class="control">host path
        <input type="text" bind:value={hostPath} placeholder="/path/to/board.yaml" aria-label="host declaration path" />
      </label>
      <button type="button" onclick={() => void openHost()}>open host path</button>
      <span class="control-sep" aria-hidden="true"></span>
      <span class="copy-toggle" role="group" aria-label="copy mode">
        <button type="button" class:active={view.copyMode === "source"} onclick={() => model.setCopyMode("source")}>source</button>
        <button type="button" class:active={view.copyMode === "plain"} onclick={() => model.setCopyMode("plain")}>plain</button>
      </span>
      <span class="theme-toggle" role="group" aria-label="theme">
        {#each ["system", "light", "dark"] as theme}
          <button type="button" class:active={view.theme === theme} onclick={() => model.setTheme(theme as "system" | "light" | "dark")}>{theme}</button>
        {/each}
      </span>
      <button type="button" aria-pressed={showDiagnostics} onclick={() => (showDiagnostics = !showDiagnostics)}>diagnostics</button>
    </span>
  </header>

  <div class="search-bar" role="search">
    <input
      type="search"
      placeholder="search prose and math"
      aria-label="search"
      bind:value={query}
      onkeydown={(event) => {
        if (event.key === "Enter") nextMatch(event.shiftKey ? -1 : 1);
      }}
    />
    <span class="search-state">
      {#if query.trim().length === 0}
        type to search
      {:else if occurrences.length === 0}
        no matches
      {:else}
        {activeMatch + 1} / {occurrences.length}
      {/if}
    </span>
    <button type="button" onclick={() => nextMatch(-1)} aria-label="previous match">prev</button>
    <button type="button" onclick={() => nextMatch(1)} aria-label="next match">next</button>
  </div>

  {#if view.errors.length > 0}
    <section class="errors" aria-label="load errors">
      {#each view.errors as issue}
        <p class="render-error"><strong>{issue.code}</strong> {issue.message}{#if issue.line !== undefined} (line {issue.line}, column {issue.column ?? 0}){/if}</p>
      {/each}
      {#if view.laneMessage}
        <p class="render-error">{view.laneMessage}</p>
      {/if}
    </section>
  {/if}

  {#if showDiagnostics}
    <section class="diagnostics" aria-label="diagnostics">
      <h3>diagnostics</h3>
      {#if view.diagnostics.length === 0}
        <p>none</p>
      {:else}
        {#each view.diagnostics as issue}
          <p><strong>{issue.code}</strong> {issue.message}{#if issue.line !== undefined} (line {issue.line}){/if}</p>
        {/each}
      {/if}
    </section>
  {/if}

  <main class="content" bind:this={contentHost} aria-label="board content"></main>
</div>
