// The select session: label selection over the visible targets
// (scenario 12). Pure logic - no DOM, no sources: the host supplies
// visibility, capture and restore. Labels go to visible targets only;
// typing narrows through the shared matcher's onUnmatched hook (one
// dispatch chain); exact-single selects and exits; Escape exits;
// selection follows identity (the part path).

import { commandContext, key, sequence } from "../contracts/input.ts";
import type { InputContext, KeyInput, KeyResolution } from "../contracts/input.ts";
import { assignLabels, narrowStep, startNarrow, type NarrowState, type TargetKind } from "./label-select.ts";

export type SelectableTarget = Readonly<{
  part: string;
  kind: TargetKind;
  text: string;
}>;

export type SelectSessionView = Readonly<{
  active: true;
  labels: readonly { part: string; label: string }[];
  /** The label keystrokes typed so far (the chips show it). */
  pendingLabel: string;
}>;

export interface SelectSessionDeps {
  /** The current frame's targets, FROM THE MODEL (never DOM-derived). */
  targets(): readonly SelectableTarget[];
  /** Whether a target's presentation is viewport-visible now. A
   * target with no geometry is simply not visible yet - never proof
   * it does not exist. */
  visible(target: SelectableTarget): boolean;
  /** Capture input. Releasing the lease returns to the CURRENT base;
   * the session must not reinstall a saved context itself. */
  capture(context: InputContext): { release(): void };
  /** The exact-single selection landed; identity is the part path. */
  onSelected(target: SelectableTarget): void;
}

export interface SelectSession {
  /** Enter if there is anything visible to label; a no-op otherwise. */
  enter(): void;
  /** Exit (Escape, teardown, or after a selection). */
  exit(): void;
  /** The session's input context, installed while active. */
  context(): InputContext | null;
  view(): SelectSessionView | null;
  subscribe(observer: (view: SelectSessionView | null) => void): () => void;
}

export function createSelectSession(deps: SelectSessionDeps): SelectSession {
  const observers = new Set<(view: SelectSessionView | null) => void>();
  let view: SelectSessionView | null = null;
  let lease: { release(): void } | null = null;
  let active: InputContext | null = null;

  function notify(): void {
    for (const observer of [...observers]) observer(view);
  }

  function exit(): void {
    if (lease !== null) {
      lease.release();
      lease = null;
    }
    active = null;
    if (view !== null) {
      view = null;
      notify();
    }
  }

  function enter(): void {
    if (active !== null) return;
    const visible = deps.targets().filter((target) => deps.visible(target));
    if (visible.length === 0) return;
    const kinds: TargetKind[] = visible.map((target) => target.kind);
    const labels = assignLabels(kinds);
    const byLabel = visible.map((target, index) => ({ part: target.part, label: labels[index] ?? "" }));
    let narrow: NarrowState = startNarrow(visible.length);

    const session: InputContext = {
      ...commandContext(() => [
        {
          action: { id: "select.exit", label: "leave selection", run: () => exit() },
          keys: [sequence(key("Escape"))],
        },
      ]),
      onUnmatched: (stroke: KeyInput): KeyResolution | undefined => {
        if (active === null) return undefined;
        // the prototype's hintKey (inst16.html:106): a non-letter is
        // ignored (consumed inert, the session stays)
        if (!/^[a-zA-Z]$/.test(stroke.key)) return { kind: "consume" };
        const step = narrowStep(narrow, labels, stroke.key);
        narrow = { alive: step.alive, pendingLabel: step.pendingLabel };
        // ZERO remaining matches clears the labels and exits (hintKey),
        // never a session alive with nothing to select that eats every
        // later key (the human's traces, m-536)
        if (step.exit) {
          exit();
          return { kind: "consume" };
        }
        if (step.selected !== null) {
          const chosen = visible[step.selected];
          if (chosen !== undefined) {
            exit();
            deps.onSelected(chosen);
            return { kind: "consume" };
          }
        }
        // the pending prefix filters the chips: only the labels it still
        // matches stay painted (hintKey hides the rest)
        view = {
          active: true,
          labels: byLabel.filter((_, index) => step.alive.includes(index)),
          pendingLabel: step.pendingLabel,
        };
        notify();
        return { kind: "consume" };
      },
    };
    active = session;
    view = { active: true, labels: byLabel, pendingLabel: "" };
    lease = deps.capture(session);
    notify();
  }

  return {
    enter,
    exit,
    context: () => active,
    view: () => view,
    subscribe(observer) {
      observers.add(observer);
      observer(view);
      return () => {
        observers.delete(observer);
      };
    },
  };
}
