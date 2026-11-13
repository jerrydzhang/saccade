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
import { crateVerdict } from "../lib/board/validate.ts";
import { createFollow } from "../lib/board/follow.ts";
import { createFindInteraction, type FindView } from "../lib/find/interaction.ts";
import { findUnitsOf } from "../lib/find/units.ts";
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
  /** The follow state as information (the S15 diagnosis): the
   * followed address and whether the poll is armed. */
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
  /** Open the act menu for a target (selection or right-click). */
  openMenuFor(target: Readonly<{ part: string; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** A plain click on a citation runs its FIRST act directly - the
   * same action object every invocation path runs (scenario 9). The
   * part names the citation's incorporation block. */
  referencePrimaryClick(part: string): void;
  /** A plain click on a figure runs its first act (enlarge) through
   * the shared menu path. */
  figurePrimaryClick(part: string): void;
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
  /** Update outcomes for the actual shown frame's generation;
   * rendering never enters or invalidates the loading queue. */
  reportRenderOutcomes(generation: number, outcomes: readonly RenderOutcome[]): void;
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
    figurePrimaryClick(part: string) {
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
      boardMenu.session.primaryClick({ kind: "figure", label: citation ?? part, part });
      notifyMenu();
    },
    setReferencePorts(ports) {
      referencePorts = ports;
    },
    reportRenderOutcomes(reported, outcomes) {
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

  const places: ContentPlaces<BoardAnchor> = {
    captureDisplayed(viewport: ViewportMetrics): ContentCapture<BoardAnchor> {
      const address: Address = resolved
        ? { machine: resolved.machine, path: resolved.path }
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
    nextMatch: () => find.step(1),
    prevMatch: () => find.step(-1),
    clearSearch: () => {
      if (find.hasSearch()) find.clearSearch();
    },
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
  const boardMenu = createBoardMenu({
    isFolded: (part) => foldedSet.has(part),
    toggleFold: (part) => host.toggleFold(part),
    referenceOf: (part) => {
      const target = scope.frame().targets.find((entry) => entry.part === part);
      if (target === undefined || target.source === undefined) return null;
      return { task: target.source.task, revision: target.source.revision, select: target.source.select };
    },
    referencePorts: () => referencePorts,
    referenceMachine: () => scope.frame().resolved?.machine ?? null,
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
