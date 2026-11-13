// The elsewhere surface's pure half (S18): which move an OPEN act
// makes, the ask's chip assignment, and the ask state's shape. The
// window store, the shell invokes and the delivery ride App.ts - what
// is here is only what a test can pin without either.

/** The three OPEN acts (the r90 KINDS order: f follow-here FIRST, then
 * the OPEN acts, then y copy): o delivers at the link (asking when no
 * link is remembered), i delivers once WITHOUT touching the link, w
 * always opens a new viewer. */
export type ElsewhereAct = "link" | "once" | "new";

/** One window the shell can see: its label and, when it carries one,
 * the window-store state it is showing. The shell serializes the
 * state field snake_case (serde default, no rename). */
export type SuiteWindow = { label: string; state_id?: string | null };

/** The decision an OPEN act makes given the live world: deliver at the
 * remembered link, open (and possibly become) a new window, or ask
 * which window receives. One window never asks - there is nothing to
 * choose: a new window opens and, for o, becomes the link (the
 * t-34@r6 change 3 semantics). */
export type ElsewhereDecision =
  | { kind: "deliver"; label: string }
  | { kind: "new-window"; becomeLink: boolean }
  | { kind: "ask" };

export function elsewhereDecision(
  act: ElsewhereAct,
  world: { linkLabel: string | null; windows: readonly SuiteWindow[] },
): ElsewhereDecision {
  if (act === "new") return { kind: "new-window", becomeLink: false };
  // a link only delivers while its window LIVES: the label must name
  // a current window (a dead link - the window closed - is no link;
  // the one-window/new/ask rules take over, never a silent delivery
  // into a dead state)
  if (
    act === "link" &&
    world.linkLabel !== null &&
    world.windows.some((entry) => entry.label === world.linkLabel)
  ) {
    return { kind: "deliver", label: world.linkLabel };
  }
  if (world.windows.length <= 1) {
    return { kind: "new-window", becomeLink: act === "link" };
  }
  return { kind: "ask" };
}

/** The ask's chips: one per visible suite window, a, b, c ... in the
 * window list's stable order; beyond one alphabet the chip grows a
 * second character, exactly as the select-first labels do. */
export function assignChips(windowLabels: readonly string[]): { window: string; chip: string }[] {
  return windowLabels.map((label, index) => ({ window: label, chip: chipFor(index) }));
}

function chipFor(index: number): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz";
  let chip = "";
  let value = index;
  do {
    chip = alphabet[value % 26] + chip;
    value = Math.floor(value / 26) - 1;
  } while (value >= 0);
  return chip;
}

/** A live ask, as the readout exposes it: the act that asked, the spec
 * it carries, the chips a window answers with. */
export type AskState = Readonly<{
  act: ElsewhereAct;
  spec: string;
  chips: readonly { window: string; chip: string }[];
}>;
