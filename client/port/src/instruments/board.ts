// The Board instrument, slice 1: the minimal canonical instrument that
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
import { knownIssue } from "../lib/contracts/buffer.ts";
import type {
  KnownIssue,
  Point,
  PreparedContent,
  RefreshOutcome,
  Viewport,
} from "../lib/contracts/buffer.ts";
import { commandContext } from "../lib/contracts/input.ts";
import type { InputContext } from "../lib/contracts/input.ts";
import type {
  Address,
  BoardAnchor,
  ContentCapture,
  ContentPlaces,
} from "../lib/contracts/places.ts";
import type { AnchorData, AnchorLocation } from "../lib/contracts/places.ts";
import type { InstrumentDefinition, InstrumentProps } from "../lib/instruments/contract.ts";
import type { StructuredIssue } from "../lib/board/errors.ts";
import { SourceUnavailableError } from "../lib/board/errors.ts";
import type { Board, FigureBlock } from "../lib/board/model.ts";
import { decodeBoard } from "../lib/board/decode.ts";
import { crateVerdict } from "../lib/board/validate.ts";
import type { HydrationHost } from "../lib/board/hydrate.ts";
import { formatLocation, instrumentPath } from "../lib/location.ts";
import { hostedTrackerLocation, parseLocation } from "../lib/location.ts";
import { registerTracker } from "../lib/tracker.ts";
import { sourceFor, type Location } from "../lib/sources.ts";
import type { Component } from "svelte";
import BoardView from "./BoardView.svelte";

export type RenderOutcome = {
  kind: "math" | "mermaid" | "figure" | "image" | "html" | "node";
  status: "ok" | "error" | "denied" | "missing" | "reported";
  message: string;
  line?: number;
};

/** The board frame's published domain data: what the window's
 * observation readout projects and the component renders. */
export type BoardData = Readonly<{
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
}>;

/** The board controller's face to its own component: rendering needs
 * generation-scoped figure loads and outcome publication, nothing
 * else. Lifecycle lives on the buffer, not here. */
export interface BoardUI extends HydrationHost {
  /** Publish rendering outcomes for the shown generation: content and
   * outcomes publish consistently, through the normal refresh
   * pipeline. */
  reportRenderOutcomes(generation: number, outcomes: readonly RenderOutcome[]): void;
}

export function boardInstrument(): InstrumentDefinition<BoardData, BoardUI, BoardAnchor> {
  return {
    kind: "board",
    Component: BoardView as Component<InstrumentProps<BoardData, BoardUI>>,
    connect: connectBoard,
  };
}

/** Wrap a prepared domain frame as the buffer's published content:
 * the logical content metadata rides with its data. */
function readyFrame(value: Readonly<BoardData>): PreparedContent<ContentFrame<BoardData>> {
  return { kind: "ready", value: { value, targets: [] } };
}

const EMPTY_FRAME: BoardData = {
  asked: "",
  location: "",
  resolved: null,
  status: "loading",
  board: null,
  errors: [],
  renderOutcomes: [],
};

function failureIssue(code: string, message: string): KnownIssue {
  return knownIssue({ code, message });
}

function issueOf(error: unknown): KnownIssue {
  if (error instanceof SourceUnavailableError) {
    return failureIssue(error.code, error.message);
  }
  return failureIssue("read-failed", String(error));
}

function connectBoard(buffer: BufferAccess<BoardData>): InstrumentAttachment<BoardUI, BoardAnchor> {
  // Controller state: the open generation, the resolved location the
  // shown frame was read at, and the outcomes its hydration reported.
  let generation = 0;
  let resolved: Location | null = null;
  let askedSpec = "";
  let lastFrame: BoardData = EMPTY_FRAME;
  let released = false;

  const host: BoardUI = {
    currentGeneration: () => generation,
    async loadFigureAsset(location: Location, block: FigureBlock) {
      if (released) return "stale";
      const token = generation;
      try {
        const asset = await sourceFor(location.machine).loadAsset(location, block.input);
        if (token !== generation) return "stale";
        return asset.bytes;
      } catch (error) {
        if (token !== generation) return "stale";
        if (error instanceof SourceUnavailableError) {
          if (error.code === "asset-denied") return `denied: ${error.message}`;
          return `missing: ${error.message}`;
        }
        return `missing: ${String(error)}`;
      }
    },
    reportRenderOutcomes(reported, outcomes) {
      if (released || reported !== generation) return;
      const frame: BoardData = { ...lastFrame, renderOutcomes: [...outcomes] };
      void buffer.updates.refresh(() => readyFrame(frame));
    },
  };

  async function load(address: Address): Promise<PreparedContent<BoardData>> {
    if (released)
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "the buffer was destroyed"),
      };
    const token = ++generation;
    const asked = formatLocation({
      machine: address.machine,
      path: address.path,
    });
    askedSpec = asked;
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
      if (token !== generation)
        return {
          kind: "failed",
          issue: failureIssue("lane-unavailable", "superseded"),
        };
      text = loaded.text;
      // a bare tracker location resolved: the board on screen is the
      // revision that answered, and its figures read the same tree
      resolvedLocation = loaded.location;
    } catch (error) {
      if (token !== generation)
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
    if (token !== generation)
      return {
        kind: "failed",
        issue: failureIssue("lane-unavailable", "superseded"),
      };
    const result = decodeBoard(text);
    if (token !== generation)
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
          asked: askedSpec,
          location: formatLocation(resolvedLocation),
          resolved: resolvedLocation,
          status: "invalid",
          board: null,
          errors: [issue],
          renderOutcomes: [],
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
            asked: askedSpec,
            location: formatLocation(resolvedLocation),
            resolved: resolvedLocation,
            status: "invalid",
            board: null,
            errors: [{ code: "decoder-mismatch", message }, ...result.errors],
            renderOutcomes: [],
          },
        };
      }
      return {
        kind: "ready",
        value: {
          asked: askedSpec,
          location: formatLocation(resolvedLocation),
          resolved: resolvedLocation,
          status: "invalid",
          board: null,
          errors: result.errors,
          renderOutcomes: [],
        },
      };
    }
    lastFrame = {
      asked: askedSpec,
      location: formatLocation(resolvedLocation),
      resolved: resolvedLocation,
      status: "loaded",
      board: result.board,
      errors: [],
      renderOutcomes: [],
    };
    return { kind: "ready", value: lastFrame };
  }

  const places: ContentPlaces<BoardAnchor> = {
    captureDisplayed(): ContentCapture<BoardAnchor> {
      // Slice 1 captures default positions only; anchor capture lands
      // with the Board instrument slice.
      const address: Address = resolved
        ? { machine: resolved.machine, path: resolved.path }
        : { machine: "", path: "" };
      return { address, position: { kind: "default" } };
    },
    async request(address: Address): Promise<RefreshOutcome> {
      return buffer.updates.refresh(async () => {
        const prepared = await load(address);
        return prepared.kind === "ready" ? readyFrame(prepared.value) : prepared;
      });
    },
    acceptsAnchor(data: AnchorData): data is BoardAnchor {
      return (
        typeof data === "object" &&
        data !== null &&
        !Array.isArray(data) &&
        (data as { kind?: unknown }).kind === "board-part"
      );
    },
    revealAnchor(_anchor: BoardAnchor): AnchorLocation {
      // No anchor geometry in slice 1: reported missing, never a
      // substituted container or a guessed position.
      return { kind: "missing" };
    },
    defaultPosition(_viewport: ReturnType<Viewport["metrics"]>): Point {
      return { left: 0, top: 0 };
    },
    subscribeChanges(_observer: () => void) {
      return () => {};
    },
  };

  const reading: InputContext = commandContext(() => []);

  return {
    presentation: host,
    input: reading,
    places,
    release() {
      released = true;
      generation += 1;
    },
  };
}

export function parseOpenSpec(spec: string): Address | null {
  // a location typed with its host reaches that tracker and remembers it
  const hosted = hostedTrackerLocation(spec);
  if (hosted) registerTracker(hosted.tracker, hosted.host);
  return parseLocation(spec);
}
