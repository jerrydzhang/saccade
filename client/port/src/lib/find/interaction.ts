// Find's input handoff: the pure search session plus declared field
// commands. The host lease owns return to the CURRENT base; Find never
// reinstalls a saved context, restores content, or writes the trail.
import { createFindSession } from "./session.ts";
import type { FindOccurrence, FindUnit } from "./session.ts";
import { key, sequence, shift, textEntryContext } from "../contracts/input.ts";
import type { InputContext } from "../contracts/input.ts";

export type FindView = Readonly<{
  open: boolean;
  query: string;
  count: number;
  active: number;
  /** The occurrence parts (their unit ids), for match marks. */
  unitParts: readonly string[];
  /** The ACTIVE occurrence's part (D2's mark target), or null when
   * no occurrence is active. */
  activeUnit: string | null;
}>;
export interface FindInteractionDeps {
  units(): readonly FindUnit[];
  capture(context: InputContext): { release(): void };
  reveal(occurrence: FindOccurrence): void;
}
export function createFindInteraction(deps: FindInteractionDeps) {
  const session = createFindSession(deps.units);
  const observers = new Set<(view: FindView) => void>();
  let lease: { release(): void } | null = null;
  const view = (): FindView => {
    const occurrences = session.occurrences();
    const count = occurrences.length;
    const active = count > 0 ? session.active() : -1;
    return {
      open: session.isOpen(),
      query: session.query(),
      count,
      active,
      unitParts: occurrences.map((occurrence) => occurrence.unit.id),
      activeUnit: active >= 0 && active < count ? occurrences[active]!.unit.id : null,
    };
  };
  const notify = () => { for (const observer of [...observers]) observer(view()); };
  function close(): void {
    session.escape();
    lease?.release();
    lease = null;
    notify();
  }
  function move(delta: number): void {
    const occurrence = session.move(delta);
    if (occurrence !== null) deps.reveal(occurrence);
    notify();
  }
  // The n/N ruling (t-36 r110): Enter COMMITS the search - the field
  // closes, keys return to the document, and the matches stay (the
  // session's query and active index survive the field's close);
  // n / N step them in the reading context until the search is
  // cleared (Escape, or a new search's typing).
  function commit(): void {
    session.escape();
    lease?.release();
    lease = null;
    notify();
  }
  const commands = [
    { action: { id: "find.commit", label: "commit the search", run: commit }, keys: [sequence(key("Enter"))] },
    { action: { id: "find.close", label: "leave find", run: close }, keys: [sequence(key("Escape"))] },
  ];
  const context = textEntryContext(() => commands);
  return {
    view,
    contentChanged: notify,
    enter(): void {
      if (session.isOpen()) return;
      session.enter();
      lease = deps.capture(context);
      notify();
    },
    type(query: string): void {
      if (!session.isOpen()) return;
      session.type(query);
      notify();
    },
    // Explicit openings and trail restores end Find. Ordinary source
    // refreshes must not call this: live content is not navigation.
    navigateAway: close,
    close,
    // n/N in the reading context: step the committed search's
    // matches; a no-op when none exist (nvim's n with no search)
    step(delta: number): void {
      if (session.occurrences().length === 0) return;
      move(delta);
    },
    // whether a committed search is live (n/N and the clearing
    // Escape consult this)
    hasSearch(): boolean {
      return session.query().length > 0;
    },
    clearSearch(): void {
      session.clearSearch();
      notify();
    },
    /** The current view, for mark painting on changes the session
     * itself does not notify (commit, clear). */
    viewNow(): FindView {
      return view();
    },
    subscribe(observer: (view: FindView) => void): () => void {
      observers.add(observer);
      observer(view());
      return () => { observers.delete(observer); };
    },
  };
}
export type FindInteraction = ReturnType<typeof createFindInteraction>;
