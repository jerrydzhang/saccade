// The Board instrument controller (svelte-free): the minimal canonical instrument that
// proves the composition skeleton. Its controller loads board text
// through the source/transport layer, admits it through the shared
// grammar crate on the bridge (the mirror decoder alone without a
// bridge), and publishes the domain frame; the component renders and
// hydrates it. The FULL Board instrument - targets, selection, input
// bindings, return anchors, annotation acts - is the next slice; every
// gap below is reported, never papered over.
//
// Instrument placement (r47-r51): a component meaningful on its own,
// connected once per buffer, staying connected while inactive,
// releasing on destruction.

import type { BufferAccess, ContentFrame, InstrumentAttachment } from "../components/Buffer.ts";
import { createLoadScope, failureIssue, issueOf, readyFrame } from "../lib/instruments/scaffold.ts";
import { knownIssue } from "../lib/contracts/buffer.ts";
import type {
  KnownIssue,
  Point,
  PreparedContent,
  RefreshOutcome,
  Viewport,
} from "../lib/contracts/buffer.ts";
import type { InputContext } from "../lib/contracts/input.ts";
import { boardReadingContext } from "../lib/input/contexts.ts";
import { createSelectSession, type SelectSessionView } from "../lib/select/session.ts";
import { createBoardMenu } from "../lib/menu/acts.ts";
import type { MenuView } from "../lib/menu/session.ts";
export type { MenuView };
import { boardTargetsOf, partAncestors, type BoardTarget } from "../lib/board/targets.ts";
export { boardTargetsOf } from "../lib/board/targets.ts";
export type { BoardTarget, BoardTargetKind } from "../lib/board/targets.ts";
import type {
  Address,
  BoardAnchor,
  ContentCapture,
  ContentPlaces,
} from "../lib/contracts/places.ts";
import type { ViewportMetrics } from "../lib/contracts/places.ts";
export type { BoardAnchor };
import type { AnchorData, AnchorLocation } from "../lib/contracts/places.ts";
import type { InstrumentDefinition, InstrumentProps } from "../lib/instruments/contract.ts";
import type { StructuredIssue } from "../lib/board/errors.ts";
import { SourceUnavailableError } from "../lib/board/errors.ts";
import type { Board, FigureBlock } from "../lib/board/model.ts";
import { decodeBoard } from "../lib/board/decode.ts";
import { blockPartOf, diffBoards, placeIn, shownAt, touches, type BoardDiff } from "../lib/board/history-diff.ts";
import { crateVerdict } from "../lib/board/validate.ts";
import { createFollow } from "../lib/board/follow.ts";
import type { ReadingLayoutChange } from "../lib/board/reading-layout.ts";
import { createFindInteraction, type FindView } from "../lib/find/interaction.ts";
import { findUnitsOf } from "../lib/find/units.ts";
import { createExpansions, type Expansion } from "../lib/board/expansion.ts";
export type { Expansion } from "../lib/board/expansion.ts";
export type { FindView } from "../lib/find/interaction.ts";
import type { HydrationHost } from "../lib/board/hydrate.ts";
import { formatLocation, instrumentPath } from "../lib/location.ts";
import { hostedTrackerLocation, parseLocation } from "../lib/location.ts";
import { registerTracker } from "../lib/tracker.ts";
import { sourceFor, type Location } from "../lib/sources.ts";
import { TrackerSource } from "../lib/tracker.ts";

export type RenderOutcome = {
  kind: "math" | "mermaid" | "figure" | "image" | "html" | "node";
  status: "ok" | "error" | "denied" | "missing" | "reported";
  message: string;
  line?: number;
};

/** The board frame's published domain data: what the window's
 * observation readout projects and the component renders. */
export type BoardData = Readonly<{
  /** The load that produced THIS frame; retained frames keep their
   * own token while a newer request is pending. */
  generation: number;
  /** The location the open ASKED for, formatted. */
  asked: string;
  /** The location that answered (a bare tracker location resolves to
   * its pinned revision); equals asked when nothing resolved. */
  location: string;
  /** The raw location the shown frame was read at: figure and image
   * loads ride the same machine + path. Null when nothing resolved. */
  resolved: import("../lib/sources.ts").Location | null;
  status: "loading" | "loaded" | "invalid";
  board: Board | null;
  errors: readonly StructuredIssue[];
  renderOutcomes: readonly RenderOutcome[];
  /** The logical targets of the shown frame, FROM THE MODEL (content
   * parts, never DOM-derived): identity is the canonical part path.
   * Labels exist only for visible presentations (the select
   * session); target existence never depends on geometry. */
  targets: readonly BoardTarget[];
  /** Folded group part paths: the header stays rendered, the content
   * does not. Model existence is untouched (scenario 12: folded-in
   * children keep existing, lose presentation). */
  folded: readonly string[];
  /** History mode (t-49): the revision shown against the one before
   * it; absent outside history. */
  history?: HistoryView | null;
}>;

/** What history mode shows: the revision, the newest one, the commit
 * that landed it (docked beside the board), and the changed parts in
 * reading order (n and N jump between them). */
export type HistoryView = Readonly<{
  task: string;
  revision: number;
  latest: number;
  message: string;
  changed: readonly string[];
  /** The change n or N last reached, -1 before any. */
  at: number;
  /** The tracked part (t), a shown part path, and its first line;
   * null while nothing is tracked. */
  tracked: string | null;
  trackedLabel: string | null;
}>;

/** The board controller's face to its own component: rendering needs
 * generation-scoped figure loads and outcome publication, plus the
 * select session's live view (label chips) - nothing else. Lifecycle
 * lives on the buffer, not here. */
export interface BoardUI extends HydrationHost {
  findView(): FindView;
  subscribeFind(observer: (view: FindView) => void): () => void;
  typeFind(query: string): void;
  /** The select session's live label view; null when inactive. */
  subscribeSelect(observer: (view: SelectSessionView | null) => void): () => void;
  /** The last label-selected target (identity = part path), if any. */
  getSelection(): BoardTarget | null;
  /** The menu's live view; null when closed. */
  subscribeMenu(observer: (view: MenuView | null) => void): () => void;
  /** Enter find from the window command (space-/): the same entry
   * the reading context's bare / runs. */
  enterFind(): void;
  /** space h: enter or leave history mode (t-49). */
  toggleHistory(): void;
  /** The follow state as information (the S15 diagnosis): the
   * followed address and whether the poll is armed. */
  /** The view's work counts (the readout's presentation field). */
  reportPresentation(stats: Readonly<{ renders: number; hydrated: number }>): void;
  presentationStats(): Readonly<{ renders: number; hydrated: number }> | null;
  followState(): {
    armed: boolean;
    task: number | null;
    address: string | null;
    requests: readonly string[];
  };
  /** The menu session (invocation paths for the component). */
  menuSession(): import("../lib/menu/session.ts").MenuSession;
  /** Fold or unfold a group part path (republishes the frame). */
  toggleFold(part: string): void;
  /** Whether a group part path is folded. */
  isFolded(part: string): boolean;
  /** Expand or collapse an incorporation part (t-58). */
  toggleExpansion(part: string): void;
  /** The shown board's expanded incorporations. */
  expansions(): readonly Expansion[];
  subscribeExpansions(observer: (expansions: readonly Expansion[]) => void): () => void;
  /** Why history mode would not open, shown in its card's place until
   * the next open or toggle; null when there is nothing to say. */
  subscribeHistoryRefusal(observer: (refusal: string | null) => void): () => void;
  /** A plain click on an incorporation's source line: its first act
   * (expand or collapse) through the shared menu path. */
  incorporationPrimaryClick(part: string): void;
  /** Open the act menu for a target (selection or right-click). */
  openMenuFor(target: Readonly<{ part: string; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** A plain click on a citation runs its FIRST act directly - the
   * same action object every invocation path runs (scenario 9). The
   * part names the citation's incorporation block. */
  referencePrimaryClick(part: string): void;
  /** A shifted or middle click on the citation: its `o` act, open at
   * the link (t-47). */
  referenceElsewhereClick(part: string): void;
  /** A shifted or middle click on a prose ref: its `o` act. */
  refElsewhereClick(part: string): void;
  /** A plain click on a figure runs its first act (enlarge) through
   * the shared menu path. */
  figurePrimaryClick(part: string, kind?: "figure" | "diagram"): void;
  /** A plain click on a prose ref: its first act (follow here). */
  refPrimaryClick(part: string): void;
  /** The window services the reference acts ride (the one open
   * funnel and the copy observation); injected by the window, and no
   * reference act is offered before then. */
  setReferencePorts(ports: {
    open(spec: string, part: string | null): void;
    copy(payload: string): void;
    elsewhere?(spec: string, act: "link" | "once" | "new"): void;
    /** The figure's enlarge act (S04): the window opens its
     * inspection surface for the part. */
    figureEnlarge?(part: string): void;
  }): void;
  /** The window's reading layout (lib/board/reading-layout.ts): the
   * zoom and column-width keys hand their change to the window, which
   * holds the layout with the rest of its state. */
  setReadingLayoutPort(port: (change: ReadingLayoutChange) => void): void;
  /** Update outcomes for the actual shown frame's generation;
   * rendering never enters or invalidates the loading queue. Also the
   * render's settle: its hydration is done. */
  reportRenderOutcomes(generation: number, outcomes: readonly RenderOutcome[]): void;
  /** A frame's board is in the DOM (before hydration): history mode
   * places the part it holds (t-57). */
  reportRendered(generation: number): void;
}

/** The live select-session view the component paints chips from. */
const EMPTY_FRAME: BoardData = {
  generation: 0,
  asked: "",
  location: "",
  resolved: null,
  status: "loading",
  board: null,
  errors: [],
  renderOutcomes: [],
  targets: [],
  folded: [],
};

export function connectBoard(buffer: BufferAccess<BoardData>): InstrumentAttachment<BoardUI, BoardAnchor> {
  // Controller state: the open generation, the resolved location the
  // shown frame was read at, and the outcomes its hydration reported.
  const scope = createLoadScope<BoardData>(EMPTY_FRAME);
  const requestLog: string[] = [];
  let resolved: Location | null = null;
  // The address the shown frame was ASKED as (a bare t-N stays bare):
  // what a capture names, so a trail return or a relaunch follows
  // again rather than pinning the revision it resolved to (t-43).
  let shownAsked: Address | null = null;
  // Folded group part paths: view state, survives refreshes of this
  // buffer (folds are never rewound by anything but their own act).
  const foldedSet = new Set<string>();
  // The follow intent (scenario 15): a normally opened board follows
  // current through a bounded poll; a pinned open never follows.
  let followAddress: Address | null = null;
  let openIntent = 0;
  const follow = createFollow({
    async currentRevision(task) {
      if (followAddress === null) return null;
      const source = sourceFor(followAddress.machine);
      if (!(source instanceof TrackerSource)) return null;
      try {
        const page = await source.currentBoardPage(task);
        return page.location ?? null;
      } catch {
        return null; // a failing poll retries silently; refresh
        // failures retain content and report through the pipeline
      }
    },
    shownRevision() {
      const resolved = scope.frame().resolved;
      return resolved !== null ? resolved.path : null;
    },
    async refresh() {
      const address = followAddress;
      if (address === null) return;
      return buffer.updates.refresh(async () => {
        const prepared = await load(address);
        return prepared.kind === "ready" ? readyFrame(prepared.value) : prepared;
      });
    },
    // the DOM timer id and Node's Timeout meet across lib mixes; the
    // id is opaque to the follow module
    schedule: (ms, run) => setTimeout(run, ms) as unknown as number,
    cancel: (id) => clearTimeout(id as unknown as ReturnType<typeof setTimeout>),
  });
  // The outcomes key of the currently published frame: re-rendering
  // the same outcomes must not republish a new frame (the frame
  // publication re-runs the component's render effect, which reports
  // again - only a real outcome change may republish).
  let publishedOutcomesKey = "";
  let presentation: Readonly<{ renders: number; hydrated: number }> | null = null;

  function displayedData(): Readonly<BoardData> | null {
    const state = buffer.snapshot();
    if (scope.isDestroyed() || state.kind !== "alive") return null;
    const frame = state.content.kind === "available" ? state.content.value
      : state.content.kind === "pending" ? state.content.previous : null;
    return frame?.value ?? null;
  }
  const displayedGeneration = (): number | null => displayedData()?.generation ?? null;
  const find = createFindInteraction({
    units: () => findUnitsOf(displayedData()?.board ?? null),
    capture: (context) => buffer.interactions.capture(context),
    reveal(occurrence) {
      const bounds = buffer.geometry.bounds(occurrence.unit.id);
      if (bounds !== null) buffer.viewport.reveal(bounds);
    },
  });

  // Incorporation expansions (t-58): buffer view state beside the
  // folds, keyed by part and source; the cited revision is read on
  // the shown board's machine and decoded like any board.
  const expansionObservers = new Set<(expansions: readonly Expansion[]) => void>();
  // history's refusal, shown where its card would be: a console line
  // alone read as a flicker (the human, 2026-11-14)
  const refusalObservers = new Set<(refusal: string | null) => void>();
  let historyRefusal: string | null = null;
  function setHistoryRefusal(next: string | null): void {
    if (historyRefusal === next) return;
    historyRefusal = next;
    for (const observer of [...refusalObservers]) observer(next);
  }
  const expansions = createExpansions({
    async load(machine, path) {
      const loaded = await sourceFor(machine).loadText({ machine, path });
      const decoded = decodeBoard(loaded.text);
      if (decoded.status !== "ok") throw new Error(`${path} does not decode`);
      return { board: decoded.board, location: loaded.location };
    },
    machine: () => scope.frame().resolved?.machine ?? null,
    sourceAt(part) {
      const target = scope.frame().targets.find((entry) => entry.part === part && entry.kind === "incorporation");
      return target?.source ?? null;
    },
    onChange() {
      const view = expansions.view();
      for (const observer of [...expansionObservers]) observer(view);
    },
  });

  const host: BoardUI = {
    displayedGeneration,
    findView: find.view,
    subscribeFind: find.subscribe,
    typeFind: find.type,
    async loadFigureAsset(location: Location, block: FigureBlock) {
      if (scope.isDestroyed()) return "stale";
      const token = displayedGeneration();
      if (token === null) return "stale";
      try {
        const asset = await sourceFor(location.machine).loadAsset(location, block.input);
        if (token !== displayedGeneration()) return "stale";
        return asset.bytes;
      } catch (error) {
        if (token !== displayedGeneration()) return "stale";
        if (error instanceof SourceUnavailableError) {
          if (error.code === "asset-denied") return `denied: ${error.message}`;
          return `missing: ${error.message}`;
        }
        return `missing: ${String(error)}`;
      }
    },
    subscribeMenu(observer) {
      menuObservers.add(observer);
      observer(menuView());
      return () => {
        menuObservers.delete(observer);
      };
    },
    menuSession() {
      return boardMenu.session;
    },
    toggleFold(part) {
      if (foldedSet.has(part)) foldedSet.delete(part);
      else foldedSet.add(part);
      buffer.updates.updateDisplayed((frame) => ({
        ...frame, value: { ...frame.value, folded: [...foldedSet] },
      }));
    },
    isFolded(part) {
      return foldedSet.has(part);
    },
    toggleExpansion(part) {
      expansions.toggle(part);
    },
    expansions() {
      return expansions.view();
    },
    subscribeHistoryRefusal(observer) {
      refusalObservers.add(observer);
      observer(historyRefusal);
      return () => {
        refusalObservers.delete(observer);
      };
    },
    subscribeExpansions(observer) {
      expansionObservers.add(observer);
      observer(expansions.view());
      return () => {
        expansionObservers.delete(observer);
      };
    },
    incorporationPrimaryClick(part) {
      const target = scope.frame().targets.find((entry) => entry.part === part && entry.kind === "incorporation");
      if (target === undefined) return;
      boardMenu.session.primaryClick({ kind: "incorporation", label: target.text, part });
      notifyMenu();
    },
    openMenuFor(target, at) {
      boardMenu.openFor(target, at);
    },
    referencePrimaryClick(part) {
      // the same action object every path runs: the citation's target
      // built from the MODEL (label = the address, as the prototype's
      // menu head reads el.dataset.loc)
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined || target.source === undefined) return;
      const label = `${target.source.task}@r${target.source.revision}`;
      boardMenu.session.primaryClick({ kind: "reference", label, part });
      notifyMenu();
    },
    referenceElsewhereClick(part) {
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined || target.source === undefined) return;
      const label = `${target.source.task}@r${target.source.revision}`;
      boardMenu.session.keyClick({ kind: "reference", label, part }, "o");
      notifyMenu();
    },
    refElsewhereClick(part: string) {
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined) return;
      boardMenu.session.keyClick({ kind: "ref", label: target.text, part }, "o");
      notifyMenu();
    },
    refPrimaryClick(part: string) {
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined) return;
      boardMenu.session.primaryClick({ kind: "ref", label: target.text, part });
      notifyMenu();
    },
    figurePrimaryClick(part: string, kind: "figure" | "diagram" = "figure") {
      // the figure's own target kind; the primary act is enlarge
      // (actsFor's figure branch), run through the SAME session path
      // every invocation uses
      // the label is the figure's own citation (the menu head), the
      // same composition the copy act uses
      const frame = scope.frame();
      const match = /@r(\d+)$/.exec(frame.location);
      const citation =
        match !== null && frame.board !== null && frame.board.task.length > 0
          ? `${frame.board.task}@r${match[1]}:${part}`
          : part;
      boardMenu.session.primaryClick({ kind, label: citation ?? part, part });
      notifyMenu();
    },
    setReadingLayoutPort(port) {
      readingLayoutPort = port;
    },
    setReferencePorts(ports) {
      referencePorts = ports;
    },
    reportRendered(generation) {
      if (!scope.isDestroyed()) placeHold(generation, false);
    },
    reportRenderOutcomes(reported, outcomes) {
      if (!scope.isDestroyed()) placeHold(reported, true);
      if (scope.isDestroyed() || reported !== displayedGeneration()) return;
      const key = JSON.stringify([reported, outcomes]);
      if (key === publishedOutcomesKey) return;
      buffer.updates.updateDisplayed((frame) => {
        if (frame.value.generation !== reported) return frame;
        publishedOutcomesKey = key;
        return { ...frame, value: { ...frame.value, renderOutcomes: [...outcomes] } };
      });
    },
    subscribeSelect(observer) {
      return selectSession.subscribe(observer);
    },
    getSelection() {
      return selection;
    },
    enterFind() {
      if (scope.frame().board != null) find.enter();
    },
    toggleHistory() {
      void toggleHistory();
    },
    reportPresentation(stats) {
      presentation = stats;
    },
    presentationStats() {
      return presentation;
    },
    followState() {
      return {
        armed: follow.running().armed,
        task: follow.running().task,
        address: followAddress === null ? null : formatLocation(followAddress),
        requests: [...requestLog],
      };
    },
  };

  async function load(address: Address): Promise<PreparedContent<BoardData>> {
    if (scope.isDestroyed())
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "the buffer was destroyed"),
      };
    const token = scope.begin();
    // a new open starts with no published outcomes
    publishedOutcomesKey = "";
    const asked = formatLocation({
      machine: address.machine,
      path: address.path,
    });
    scope.setAsked(asked);
    const location: Location = { machine: address.machine, path: address.path };
    const page = instrumentPath(location.path);
    if (page !== null) {
      // The entry pages ride a tracker location; their instruments are
      // later slices. The gap is reported, never guessed.
      return {
        kind: "failed",
        issue: failureIssue(
          "asset-denied",
          `the ${page} page has no instrument in this build yet (slices 2+)`,
        ),
      };
    }
    let text: string;
    let resolvedLocation: Location = location;
    try {
      const loaded = await sourceFor(location.machine).loadText(location);
      if (scope.isStale(token))
        return {
          kind: "failed",
          issue: failureIssue("lane-unavailable", "superseded"),
        };
      text = loaded.text;
      // a bare tracker location resolved: the board on screen is the
      // revision that answered, and its figures read the same tree
      resolvedLocation = loaded.location;
    } catch (error) {
      if (scope.isStale(token))
        return {
          kind: "failed",
          issue: failureIssue("lane-unavailable", "superseded"),
        };
      resolved = null;
      return { kind: "failed", issue: issueOf(error) };
    }
    resolved = resolvedLocation;
    shownAsked = { machine: address.machine, path: address.path };
    // The native gate: admission through the shared board-grammar
    // crate. Null without a bridge, where the mirror decoder alone
    // decides; a bridge that fails to answer refuses loudly.
    let verdict: Awaited<ReturnType<typeof crateVerdict>>;
    try {
      verdict = await crateVerdict(text);
    } catch (error) {
      verdict = {
        verdict: "refused",
        kind: "bridge-invalid",
        message: error instanceof Error ? error.message : String(error),
      };
    }
    if (scope.isStale(token))
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "superseded"),
      };
    const result = decodeBoard(text);
    if (scope.isStale(token))
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "superseded"),
      };

    if (verdict && verdict.verdict === "refused") {
      const issue: StructuredIssue = {
        code: "board-refused",
        message: verdict.message,
      };
      if (verdict.line !== undefined) {
        issue.line = verdict.line;
        issue.column = verdict.column ?? 0;
      }
      return {
        kind: "ready",
        value: {
          asked: scope.asked(),
          location: formatLocation(resolvedLocation),
          resolved: resolvedLocation,
          status: "invalid",
          board: null,
          errors: [issue],
          generation: token,
          renderOutcomes: [],
          targets: [],
          folded: [...foldedSet],
        },
      };
    }
    if (result.status === "invalid") {
      if (verdict && verdict.verdict === "ok") {
        // The mirror refused text the crate accepted: a decoder bug.
        // Never render an unvalidated board; refuse with both accounts.
        const first = result.errors[0];
        const message = `decoder disagreement: the grammar validator accepted the board, the presentation decoder refused it (${first?.code}: ${first?.message})`;
        return {
          kind: "ready",
          value: {
            asked: scope.asked(),
            location: formatLocation(resolvedLocation),
            resolved: resolvedLocation,
            status: "invalid",
            board: null,
            errors: [{ code: "decoder-mismatch", message }, ...result.errors],
            generation: token,
            renderOutcomes: [],
            targets: [],
            folded: [...foldedSet],
          },
        };
      }
      return {
        kind: "ready",
        value: {
          asked: scope.asked(),
          location: formatLocation(resolvedLocation),
          resolved: resolvedLocation,
          status: "invalid",
          board: null,
          errors: result.errors,
          generation: token,
          renderOutcomes: [],
          targets: [],
          folded: [...foldedSet],
        },
      };
    }
    scope.adopt({
      generation: token,
      asked: scope.asked(),
      location: formatLocation(resolvedLocation),
      resolved: resolvedLocation,
      status: "loaded",
      board: result.board,
      errors: [],
      renderOutcomes: [],
      targets: boardTargetsOf(result.board),
      folded: [...foldedSet],
    });
    return { kind: "ready", value: scope.frame() };
  }

  // -- history mode (t-49, t-57; t-14 "The viewer", History) -----------
  // A mode over the shown board, not a navigation: the window's trail
  // never sees it. Entering shows the shown revision against the one
  // before it; h and l step time; leaving shows the revision reached,
  // plain (a bare open again when that is the newest one). The follow
  // pauses while history holds the window.
  //
  // The part being read keeps its screen position: entering, every
  // step and leaving hold one part (the tracked one, else the deepest
  // part at the reading height - the view's top, where the trail's own
  // capture reads) at the offset it had, mapped across revisions by
  // the diff's places (its nearest surviving neighbour when it is
  // gone). t tracks the part at the reading height; H and L walk to
  // the revisions that changed it.
  type HistoryState = {
    machine: string;
    task: number;
    revision: number;
    latest: number;
    messages: Map<number, string>;
    /** The shown revision's diff; null until the first one shows. */
    diff: BoardDiff | null;
    at: number;
    /** The tracked part, as a shown path of `diff`; null: none. */
    tracked: string | null;
    /** Each revision's diff against the one before it, read once. */
    diffs: Map<number, Promise<BoardDiff>>;
    /** What leaving opens at the newest revision: the asked address. */
    returnTo: Address;
  };
  let history: HistoryState | null = null;
  let historyIntent = 0;

  function historyView(state: HistoryState, targets: readonly BoardTarget[] = scope.frame().targets): HistoryView {
    const tracked = state.tracked;
    const label =
      tracked === null
        ? null
        : (targets.find((target) => target.part === tracked)?.text ?? "")
            .split("\n")
            .find((line) => line.trim().length > 0)
            ?.replace(/[\uE000-\uE003]/g, "")
            .replace(/^\s*#+\s*/, "")
            .trim() ?? "";
    return {
      task: `t-${state.task}`,
      revision: state.revision,
      latest: state.latest,
      message: state.messages.get(state.revision) ?? "",
      changed: state.diff?.changed ?? [],
      at: state.at,
      tracked,
      trackedLabel: label === null ? null : label.length > 48 ? `${label.slice(0, 47)}...` : label,
    };
  }

  /** A held part: where it must stand on screen once the frame of its
   * generation renders (null: the next one rendered after `after`). */
  type Hold = { shown: string; offset: number; generation: number | null; after: number };
  let hold: Hold | null = null;

  /** Place the held part, if this render is the one it waits for; the
   * render placed again once hydration settled its layout (figures,
   * math and diagrams change heights), then the hold ends. */
  function placeHold(generation: number, settled: boolean): void {
    const held = hold;
    if (held === null) return;
    if (held.generation === null && generation > held.after) held.generation = generation;
    if (held.generation !== generation) return;
    const bounds = buffer.geometry.bounds(held.shown);
    if (bounds !== null) {
      const { position } = buffer.viewport.metrics();
      buffer.viewport.moveTo({ left: position.left, top: Math.max(0, Math.ceil(bounds.top - held.offset)) });
    }
    if (settled) hold = null;
  }

  /** The deepest block part at the reading height (the view's top) and
   * its offset from it; the first part below when the height falls
   * between parts. */
  function readingPart(): { part: string; offset: number } | null {
    const top = buffer.viewport.metrics().position.top;
    let inside: { part: string; top: number } | null = null;
    let below: { part: string; top: number } | null = null;
    for (const target of scope.frame().targets) {
      if (blockPartOf(target.part) !== target.part) continue;
      const bounds = buffer.geometry.bounds(target.part);
      if (bounds === null) continue;
      if (top >= bounds.top && top < bounds.bottom) {
        if (inside === null || target.part.length > inside.part.length) inside = { part: target.part, top: bounds.top };
      } else if (bounds.top >= top && (below === null || bounds.top < below.top)) {
        below = { part: target.part, top: bounds.top };
      }
    }
    const chosen = inside ?? below;
    return chosen === null ? null : { part: chosen.part, offset: chosen.top - top };
  }

  /** The part a step holds: the tracked one where it shows, else the
   * one at the reading height. */
  function heldPart(state: HistoryState): { part: string; offset: number } | null {
    if (state.tracked !== null) {
      const bounds = buffer.geometry.bounds(state.tracked);
      if (bounds !== null) return { part: state.tracked, offset: bounds.top - buffer.viewport.metrics().position.top };
    }
    return readingPart();
  }

  function diffAt(state: HistoryState, revision: number): Promise<BoardDiff> {
    const cached = state.diffs.get(revision);
    if (cached !== undefined) return cached;
    const read = (async () => {
      const source = sourceFor(state.machine);
      if (!(source instanceof TrackerSource)) throw new Error("history reads a tracker board's revisions");
      const pinned = `t-${state.task}@r${revision}`;
      const [text, before, changes] = await Promise.all([
        source.boardTextAt(pinned),
        revision > 1 ? source.boardTextAt(`t-${state.task}@r${revision - 1}`) : Promise.resolve(null),
        revision > 1 ? source.revisionChanges(state.task, revision - 1, revision) : Promise.resolve([]),
      ]);
      const current = decodeBoard(text);
      if (current.status !== "ok") throw new SourceUnavailableError("core-protocol", `${pinned} does not decode`);
      const previous = before === null ? null : decodeBoard(before);
      return diffBoards(previous !== null && previous.status === "ok" ? previous.board : null, current.board, changes);
    })();
    state.diffs.set(revision, read);
    void read.catch(() => {
      if (state.diffs.get(revision) === read) state.diffs.delete(revision);
    });
    return read;
  }

  async function toggleHistory(): Promise<void> {
    setHistoryRefusal(null);
    if (history !== null) {
      await leaveHistory();
      return;
    }
    const frame = scope.frame();
    const location = frame.resolved;
    if (frame.status !== "loaded" || frame.board === null || location === null) return;
    const spec = /^t-(\d+)@r(\d+)$/.exec(location.path);
    if (spec === null || !location.machine.startsWith("tracker:")) {
      note("history reads a tracker board's revisions");
      return;
    }
    const source = sourceFor(location.machine);
    if (!(source instanceof TrackerSource)) return;
    const reading = readingPart();
    const intent = ++historyIntent;
    let page;
    try {
      page = await source.history(`t-${spec[1]}`);
    } catch (error) {
      note(`the history could not be read: ${String(error)}`);
      return;
    }
    if (intent !== historyIntent) return;
    // the tracker must serve component changes (sac 0.20.7): ask once
    // before entering, so an older tracker refuses in words, never by
    // opening and falling back
    if (Number(spec[2]) > 1) {
      try {
        await source.revisionChanges(Number(spec[1]), Number(spec[2]) - 1, Number(spec[2]));
      } catch (error) {
        if (intent !== historyIntent) return;
        const message = error instanceof Error ? error.message : String(error);
        note(message);
        setHistoryRefusal(
          /component changes/.test(message)
            ? "history needs this project's tracker at sac 0.20.7 or later (its diff page serves no component changes)"
            : `history could not open: ${message}`,
        );
        return;
      }
      if (intent !== historyIntent) return;
    }
    const messages = new Map<number, string>();
    let latest = Number(spec[2]);
    for (const entry of page.entries ?? []) {
      if (entry.revision === null) continue;
      latest = Math.max(latest, entry.revision);
      if (entry.message !== null && entry.event === "task_committed") messages.set(entry.revision, entry.message);
    }
    follow.stop();
    followAddress = null;
    const asked = parseLocation(frame.asked);
    history = {
      machine: location.machine,
      task: Number(spec[1]),
      revision: Number(spec[2]),
      latest,
      messages,
      diff: null,
      at: -1,
      tracked: null,
      diffs: new Map(),
      returnTo: asked ?? { machine: location.machine, path: location.path },
    };
    // the plain board's part is the shown revision's own path
    await showRevision(Number(spec[2]), (diff) => {
      const shown = reading === null ? null : shownAt(diff, "after", reading.part);
      return shown === null || reading === null ? null : { shown: shown.shown, offset: reading.offset };
    });
  }

  /** Show a revision against its predecessor; `place` answers, from
   * the new diff, the part to hold and its offset. */
  async function showRevision(
    revision: number,
    place: (diff: BoardDiff) => { shown: string; offset: number } | null,
  ): Promise<void> {
    const state = history;
    if (state === null) return;
    const intent = ++historyIntent;
    const pinned = `t-${state.task}@r${revision}`;
    await buffer.updates.refresh(async () => {
      try {
        const diff = await diffAt(state, revision);
        if (intent !== historyIntent || history !== state) {
          return { kind: "failed", issue: failureIssue("lane-unavailable", "superseded") };
        }
        state.revision = revision;
        state.diff = diff;
        state.at = -1;
        const held = place(diff);
        const location: Location = { machine: state.machine, path: pinned };
        resolved = location;
        const token = scope.begin();
        hold = held === null ? null : { ...held, generation: token, after: token };
        const targets = boardTargetsOf(diff.board);
        scope.adopt({
          generation: token,
          asked: scope.asked(),
          location: formatLocation(location),
          resolved: location,
          status: "loaded",
          board: diff.board,
          errors: [],
          renderOutcomes: [],
          targets,
          folded: [...foldedSet],
          history: historyView(state, targets),
        });
        return readyFrame(scope.frame());
      } catch (error) {
        return { kind: "failed", issue: issueOf(error) };
      }
    });
  }

  async function leaveHistory(): Promise<void> {
    const state = history;
    if (state === null) return;
    // the plain revision's part: the held one's place after
    const held = heldPart(state);
    const place = held === null || state.diff === null ? null : placeIn(state.diff, held.part, "after");
    history = null;
    historyIntent++;
    hold =
      place === null || held === null
        ? null
        : { shown: place.path, offset: held.offset, generation: null, after: scope.frame().generation };
    const address =
      state.revision === state.latest
        ? state.returnTo
        : { machine: state.machine, path: `t-${state.task}@r${state.revision}` };
    const outcome = await places.request(address);
    if (outcome !== "applied" && hold !== null && hold.generation === null) hold = null;
  }

  function stepHistory(direction: -1 | 1): void {
    const state = history;
    if (state === null || state.diff === null) return;
    const next = state.revision + direction;
    if (next < 1 || next > state.latest) return;
    // stepping back, the revision before is this diff's earlier side
    // and the next diff's later one; stepping on, the other way round
    const held = heldPart(state);
    const place = held === null ? null : placeIn(state.diff, held.part, direction < 0 ? "before" : "after");
    const tracking = state.tracked !== null;
    void showRevision(next, (diff) => {
      const shown = place === null ? null : shownAt(diff, direction < 0 ? "after" : "before", place.path);
      // a tracked part gone in the revision reached ends the tracking
      if (tracking) state.tracked = place !== null && place.exact && shown !== null && shown.exact ? shown.shown : null;
      return shown === null || held === null ? null : { shown: shown.shown, offset: held.offset };
    });
  }

  /** t: track the part at the reading height, or stop tracking. */
  function toggleTrack(): void {
    const state = history;
    if (state === null || state.diff === null) return;
    state.tracked = state.tracked !== null ? null : (readingPart()?.part ?? null);
    publishHistory(state);
  }

  /** H and L: the revision before or after this one that changed the
   * tracked part (it, or anything inside it), the part held still. */
  async function walkTracked(direction: -1 | 1): Promise<void> {
    const state = history;
    if (state === null || state.diff === null) return;
    if (state.tracked === null) {
      note("H and L walk a tracked part's changes; t tracks the part at the reading height");
      return;
    }
    const held = heldPart(state);
    const intent = historyIntent;
    const later = direction > 0;
    const start = placeIn(state.diff, state.tracked, later ? "after" : "before");
    if (start === null || !start.exact) {
      note(later ? `the tracked part is gone after r${state.revision}` : `r${state.revision} added the tracked part`);
      return;
    }
    let path = start.path;
    let found: { revision: number; shown: string } | null = null;
    try {
      for (let k = state.revision + direction; k >= 1 && k <= state.latest; k += direction) {
        const diff = await diffAt(state, k);
        if (intent !== historyIntent || history !== state) return;
        const shown = shownAt(diff, later ? "before" : "after", path);
        if (shown === null || !shown.exact) break;
        if (touches(diff, shown.shown)) {
          found = { revision: k, shown: shown.shown };
          break;
        }
        const on = placeIn(diff, shown.shown, later ? "after" : "before");
        if (on === null || !on.exact) break;
        path = on.path;
      }
    } catch (error) {
      note(`the revisions could not be read: ${String(error)}`);
      return;
    }
    if (found === null) {
      note(`no ${later ? "later" : "earlier"} revision changed the tracked part`);
      return;
    }
    const target = found;
    await showRevision(target.revision, () => {
      state.tracked = target.shown;
      return held === null ? null : { shown: target.shown, offset: held.offset };
    });
  }

  function publishHistory(state: HistoryState): void {
    buffer.updates.updateDisplayed((frame) => ({
      ...frame,
      value: { ...frame.value, history: historyView(state) },
    }));
  }

  function jumpChange(direction: -1 | 1): boolean {
    const state = history;
    if (state === null) return false;
    const changed = state.diff?.changed ?? [];
    if (changed.length === 0) return true;
    const count = changed.length;
    state.at = state.at === -1 ? (direction === 1 ? 0 : count - 1) : (state.at + direction + count) % count;
    const part = changed[state.at]!;
    const bounds = buffer.geometry.bounds(part);
    if (bounds !== null) buffer.viewport.reveal(bounds);
    publishHistory(state);
    return true;
  }

  function note(message: string): void {
    console.warn(`[port] history: ${message}`);
  }

  const places: ContentPlaces<BoardAnchor> = {
    captureDisplayed(viewport: ViewportMetrics): ContentCapture<BoardAnchor> {
      // The place names the location as it was asked: a bare board
      // returns bare and keeps following, a pinned one stays pinned
      // (t-43). History mode is the exception: a revision reached by
      // stepping time is the pinned one shown, as leaving history
      // would open it (the newest revision opens as asked).
      const inPast = history !== null && history.revision !== history.latest;
      const shown = inPast ? resolved : (shownAsked ?? resolved);
      const address: Address = shown
        ? { machine: shown.machine, path: shown.path }
        : { machine: "", path: "" };
      // The reading anchor (hybrid default, r49): the target identity
      // when the reading position sits inside one, else the default
      // position. Geometry is live measurement; a target with no
      // measurable presentation is skipped, never guessed.
      const viewTop = viewport.position.top;
      for (const target of scope.frame().targets) {
        const bounds = buffer.geometry.bounds(target.part);
        if (bounds === null) continue;
        if (viewTop >= bounds.top && viewTop < bounds.bottom) {
          return {
            address,
            position: { kind: "anchor", anchor: { kind: "board-part", path: target.part }, bounds },
          };
        }
      }
      return { address, position: { kind: "default" } };
    },
    async request(address: Address): Promise<RefreshOutcome> {
      // Explicit opens AND trail restores arrive here. Follow refreshes
      // call load directly and do not end the interaction.
      requestLog.push(formatLocation(address));
      if (requestLog.length > 6) requestLog.shift();
      // an open from anywhere ends history mode, and any refusal note
      history = null;
      setHistoryRefusal(null);
      historyIntent++;
      find.navigateAway();
      // Follow the asked bare address, never the pinned resolution.
      // An explicit open invalidates any in-flight poll immediately.
      const intent = ++openIntent;
      follow.stop();
      followAddress = null;
      try {
        return await buffer.updates.refresh(async () => {
          const prepared = await load(address);
          return prepared.kind === "ready" ? readyFrame(prepared.value) : prepared;
        });
      } finally {
        if (!scope.isDestroyed() && intent === openIntent) {
          const task = /^t-(\d+)$/.exec(address.path);
          if (task !== null && address.machine.startsWith("tracker:")) {
            followAddress = { ...address };
            follow.start(Number(task[1]));
          }
        }
      }
    },
    acceptsAnchor(data: AnchorData): data is BoardAnchor {
      return (
        typeof data === "object" &&
        data !== null &&
        !Array.isArray(data) &&
        (data as { kind?: unknown }).kind === "board-part" &&
        typeof (data as { path?: unknown }).path === "string" &&
        // the anchor must name a part of THIS frame's model
        scope.frame().targets.some((target) => target.part === (data as { path: string }).path)
      );
    },
    revealAnchor(anchor: BoardAnchor): AnchorLocation {
      // Fold reveal (scenario 8): unfold exactly the enclosing visual
      // containers needed to expose the actual anchor - never a
      // substituted container, never unrelated state. Unfolding
      // republishes the frame; geometry becomes usable on the next
      // call, after the re-render.
      const enclosing = partAncestors(anchor.path);
      let unfolded = false;
      for (const ancestor of enclosing) {
        if (foldedSet.has(ancestor)) {
          foldedSet.delete(ancestor);
          unfolded = true;
        }
      }
      if (unfolded) {
        buffer.updates.updateDisplayed((frame) => ({
          ...frame, value: { ...frame.value, folded: [...foldedSet] },
        }));
        return { kind: "pending" };
      }
      const bounds = buffer.geometry.bounds(anchor.path);
      if (bounds === null) return { kind: "pending" };
      return { kind: "found", bounds };
    },
    defaultPosition(_viewport: ReturnType<Viewport["metrics"]>): Point {
      return { left: 0, top: 0 };
    },
    subscribeChanges(_observer: () => void) {
      return () => {};
    },
  };

  // -- the select session (label selection; scenario 12) --------------
  // 's' enters; labels go to VISIBLE targets only (geometry decides
  // visibility; existence never depends on geometry); exact-single
  // selects and exits; Escape exits; selection follows identity (the
  // part path). The session logic is the shared, unit-tested module;
  // the controller supplies model targets, geometry-decided
  // visibility, and the buffer's own capture/restore.
  let selection: BoardTarget | null = null;
  const reading: InputContext = boardReadingContext({
    scrollToTop: () => buffer.viewport.start(),
    scrollToBottom: () => buffer.viewport.end(),
    scrollDown: () => buffer.viewport.scrollLines(1),
    scrollUp: () => buffer.viewport.scrollLines(-1),
    scrollFraction: (direction, fraction) => buffer.viewport.scrollFraction(direction, fraction),
    enterSelect: () => selectSession.enter(),
    enterFind: () => {
      if (displayedData()?.board != null) find.enter();
    },
    nextMatch: () => {
      if (!jumpChange(1)) find.step(1);
    },
    prevMatch: () => {
      if (!jumpChange(-1)) find.step(-1);
    },
    clearSearch: () => {
      if (find.hasSearch()) find.clearSearch();
      else if (history !== null && history.tracked !== null) toggleTrack();
    },
    historyStep: (direction) => stepHistory(direction),
    historyTrack: () => toggleTrack(),
    historyWalk: (direction) => void walkTracked(direction),
    readingLayout: (change) => readingLayoutPort?.(change),
  });

  const selectSession = createSelectSession({
    targets: () => scope.frame().targets,
    visible(target) {
      const metrics = buffer.viewport.metrics();
      const viewTop = metrics.position.top;
      const viewBottom = metrics.position.top + metrics.height;
      const bounds = buffer.geometry.bounds(target.part);
      if (bounds === null) return false;
      return bounds.top < viewBottom && bounds.bottom > viewTop;
    },
    capture: (context) => buffer.interactions.capture(context),
    onSelected(target) {
      selection = target;
      // scenario 9: selection opens the act menu (a no-op for
      // act-less kinds)
      boardMenu.openFor({ part: target.part, kind: target.kind, label: target.text.split("\n")[0] ?? target.text });
    },
  });

  // -- the menu (scenario 9): selection opens the target's acts ------
  const menuObservers = new Set<(view: MenuView | null) => void>();
  let referencePorts: import("../lib/menu/acts.ts").ReferencePorts | null = null;
  let readingLayoutPort: ((change: ReadingLayoutChange) => void) | null = null;
  const boardMenu = createBoardMenu({
    isFolded: (part) => foldedSet.has(part),
    toggleFold: (part) => host.toggleFold(part),
    isExpanded: (part) => expansions.isExpanded(part),
    toggleExpansion: (part) => expansions.toggle(part),
    referenceOf: (part) => {
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined || target.source === undefined) return null;
      return { task: target.source.task, revision: target.source.revision, select: target.source.select };
    },
    referencePorts: () => referencePorts,
    referenceMachine: () => scope.frame().resolved?.machine ?? null,
    refOf: (part) => {
      const target = scope.frame().targets.find((entry) => entry.part === part);
      return target !== undefined && target.kind === "ref" ? target.text : null;
    },
    partCitation: (part) => {
      // the SHOWN revision (a bare open displays its resolved pin)
      // plus the part's full canonical path - never the parent
      const frame = scope.frame();
      const match = /@r(\d+)$/.exec(frame.location);
      if (match === null || frame.board === null || frame.board.task.length === 0) return null;
      return `${frame.board.task}@r${match[1]}:${part}`;
    },
    capture: (context) => buffer.interactions.capture(context),
  });
  function menuView(): MenuView | null {
    if (!boardMenu.session.isOpen()) return null;
    const target = boardMenu.session.target();
    if (target === null) return null;
    const acts = boardMenu.session.acts();
    return {
      open: true,
      label: target.label,
      acts: acts.map((act) => ({ key: act.key, label: act.action.label })),
      at: boardMenu.session.at(),
      part: target.part,
    };
  }
  function notifyMenu(): void {
    const view = menuView();
    for (const observer of [...menuObservers]) observer(view);
  }
  boardMenu.setOnNotify(() => notifyMenu());


  // Live publication changes matches without rewinding the interaction
  // or treating background refresh as explicit navigation.
  const offFindContent = buffer.subscribe(() => find.contentChanged());

  return {
    presentation: host,
    input: reading,
    places,
    release() {
      scope.destroy();
      follow.stop();
      offFindContent();
      find.close();
      selectSession.exit();
      boardMenu.session.outsidePointerDown();
    },
  };
}

export function parseOpenSpec(spec: string): Address | null {
  // a location typed with its host reaches that tracker and remembers it
  const hosted = hostedTrackerLocation(spec);
  if (hosted) registerTracker(hosted.tracker, hosted.host);
  return parseLocation(spec);
}
