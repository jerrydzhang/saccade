// The deck (t-14 "The viewer", t-3 "Agent-owned viewer", transport
// settled on t-24@r21): when an agent holds the window's address it
// may cue slides. The address is a herdr-core document's name -
// `deck-` plus 96 random bits, on the hub's core; holding it is the
// permission, and nothing lists documents, so an untold agent can
// never reach the deck. The agent cues by reading the document and
// setting it with that version, appending a slide: the document is a
// JSON array of location strings, a slide's number is its position in
// the list, and reading the document is the read-back. The document
// holds locations only, never the human's position; the viewer
// deletes it when the window closes, and the address is dead from
// then on.
//
// A cue waits unless the human is on the latest slide - "on the
// latest" only while the window shows that slide as cued; after
// following a ref away, a cue waits, stated plainly in the HUD. The
// deck's first contact is the one exception the confirmed semantics
// carry: a deck the human never entered starts with its first cue
// (handing out the address is asking for the slideshow; reading is
// never pulled away once the human has stepped inside). Slides are
// ordinary views; boards are pinned by revision when the cue lands, a
// file is its cued snapshot for the life of the deck. h l move
// between slides - in history they keep stepping time; digits jump.
// Deck motion never touches the window's back/forward stacks.

/** The address: `deck-` plus 24 lowercase hex characters - 96 bits of
 * randomness inside the core's 32-character name rule (probed against
 * the pinned core 0.1.17: 1-32 of a-z 0-9 - _ .). */
export function deckNameFromBytes(random: Uint8Array): string {
  let hex = "";
  for (const byte of random) hex += byte.toString(16).padStart(2, "0");
  return `deck-${hex}`;
}

export function isValidDeckName(name: string): boolean {
  return /^deck-[0-9a-f]{24}$/.test(name);
}

export type DeckState = {
  name: string;
  /** The document's list: location strings, slide N at index N-1. */
  slides: string[];
  /** The deck's own pointer: the index of the slide it last showed
   * or moved to; null until the deck is entered. It survives following
   * a ref away - the human can resume where the deck stood. */
  position: number | null;
  /** The human has entered the deck: a slide was shown or motion was
   * asked. Until then a cue shows without waiting. */
  entered: boolean;
  /** The document version the slides came from. */
  version: number;
};

export function newDeck(name: string): DeckState {
  return { name, slides: [], position: null, entered: false, version: 0 };
}

export type DeckEvent = {
  state: DeckState;
  /** 0-based indexes of the slides this document landing appended. */
  arrived: number[];
  /** The newest arrived slide should show now: the human was on the
   * latest slide, or this is the deck's first contact. */
  show: boolean;
  /** The document was deleted: the address is dead. */
  dead: boolean;
};

/** One document landing (a doc event's read-back): follow the list,
 * decide show or wait. `onLatestView` says the window is showing the
 * slide the pointer names - showing anything else (a ref, the human's
 * own open) means a cue waits. A deleted document ends the deck. */
export function applyDocument(
  state: DeckState,
  onLatestView: boolean,
  value: unknown,
  version: number,
): DeckEvent {
  if (value === null || value === undefined) {
    return { state: { ...state, version }, arrived: [], show: false, dead: true };
  }
  if (
    !Array.isArray(value) ||
    value.some((entry) => typeof entry !== "string")
  ) {
    // not the shape the cue protocol names: ignore the landing, keep
    // the deck as it stood
    return { state, arrived: [], show: false, dead: false };
  }
  const slides = value as string[];
  const arrived: number[] = [];
  for (let index = state.slides.length; index < slides.length; index++) {
    arrived.push(index);
  }
  const wasOnLatest =
    state.position !== null && state.position === state.slides.length - 1 && onLatestView;
  // the one exception the confirmed semantics carry: the deck's FIRST
  // cue lands on whatever the window shows (handing out the address is
  // asking for the slideshow). After that the rule is strict: only a
  // human on the latest slide sees the next one arrive.
  const show = arrived.length > 0 && (state.slides.length === 0 || wasOnLatest);
  const latest = slides.length - 1;
  let position = state.position;
  if (show) position = latest;
  else if (position !== null && position > latest) position = Math.max(latest, 0);
  return {
    state: { ...state, slides, position, version },
    arrived,
    show,
    dead: false,
  };
}

/** `h` in the deck steps back; `l` steps forward. Outside the deck,
 * `l` enters at the latest slide (next means newest) and `h` does
 * nothing. Inside, both stop at the ends. The target is an index, or
 * null for no motion. */
export function stepDeck(
  state: DeckState,
  direction: "older" | "newer",
): { state: DeckState; target: number | null } {
  const latest = state.slides.length - 1;
  if (latest < 0) return { state, target: null };
  if (direction === "older") {
    if (!state.entered || state.position === null) return { state, target: null };
    const target = Math.max(state.position - 1, 0);
    return { state: { ...state, position: target, entered: true }, target };
  }
  if (!state.entered) {
    return { state: { ...state, position: latest, entered: true }, target: latest };
  }
  const base = state.position === null ? -1 : state.position;
  const target = Math.min(base + 1, latest);
  return { state: { ...state, position: target, entered: true }, target };
}

/** Digits jump to slide numbers, 1-based. A number past the last
 * slide does nothing (beyond-9 decks step with h l - single-press
 * digits only, no count state to terminate them). */
export function jumpDeck(
  state: DeckState,
  digit: number,
): { state: DeckState; target: number | null } {
  const index = digit - 1;
  if (index < 0 || index >= state.slides.length) return { state, target: null };
  return { state: { ...state, position: index, entered: true }, target: index };
}

/** The HUD's waiting text: the slide the next forward step will land
 * on, when one waits. "slide 4 waiting" (t-24@r21 B4). */
export function waitingSlide(state: DeckState, onLatestView: boolean): number | null {
  const latest = state.slides.length - 1;
  if (latest < 0) return null;
  if (onLatestView && state.position === latest) return null;
  const next = (state.position ?? -1) + 1;
  return next <= latest ? next + 1 : null;
}

/** The exact cue command an agent runs, the string the viewer's copy
 * act carries and the README writes out: read the document, append
 * the location, set it with that version, retry on a lost race, and
 * name the slide number back. The agent runs it verbatim with the
 * slide's location as the argument. */
export function cueCommand(name: string, location: string): string {
  return `python3 - '${location}' <<'DECK'
import json, subprocess, sys
name = "${name}"
def call(method, params):
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    out = subprocess.run(["hc", "connect"], input=line + "\\n", capture_output=True, text=True)
    return json.loads(out.stdout)
doc = call("doc.get", {"name": name})["result"]
while True:
    slides = (doc.get("value") or [])
    slides.append(sys.argv[1])
    done = call("doc.set", {"name": name, "version": doc["version"], "value": slides})
    if "error" not in done:
        print("slide", len(slides))
        break
    doc = call("doc.get", {"name": name})["result"]
DECK`;
}
