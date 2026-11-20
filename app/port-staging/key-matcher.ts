// Module 1 of the t-36 slice-2 extraction: the key matcher.
//
// Implements the shared matcher over the ratified input-context
// contract (input-context.ts, from t-36@r46
// reference/input-context-contract.ts). Scenario clauses (t-36@r50
// reference/acceptance-scenarios.txt):
//   - scenario 10 "Context decides what keys mean; sequences go
//     through one matcher" (lines 221-249): unmatched application keys
//     are consumed in the command context; text entry matches its
//     declared commands first, other typing edits the field natively
//     and never reaches the board keymap; help and execution read the
//     same binding declarations; sequences run through the shared
//     matcher; a partial sequence never fires early and never survives
//     a handoff; sequence timeout is a named tweakable default, 1000 ms
//     (vim timeoutlen, ratified t-36@r49), pending sequence abandoned
//     at expiry, received keys then match their own bindings if
//     declared, else are consumed.
// Executed prototype source (reference of record, /tmp reference copy
// of build round 16):
//   - inst16.html:458-516: the keydown chain - every branch ends in
//     preventDefault/return, so unmatched application keys are consumed
//     (nothing falls through to content underneath); :510 lets
//     modifier chords fall to the native layer (the IME/native-editing
//     boundary set is draft-open, staged here as consume).
//   - inst16.html:499: the prototype's only sequence, gg, is a
//     gPending flag abandoned by a silent 700 ms drop - the ratified
//     difference (adapter A1, harness t36_adapters.py:45-59) replaces
//     that with SEQUENCE_TIMEOUT_MS = 1000 and the own-bindings-or-
//     consumed expiry drain.
//   - inst16.html:71-97: commands() is the one per-context table; the
//     runner (runCmd :97) and help (showHelp :90) both read it fresh,
//     which describeCommands + createMatcher preserve by reading
//     commandsOf(context) on every query.
// Staged decisions on draft-open points are recorded in README.md.

import {
  commandsOf,
  strokeMatches,
  commandContext,
  textEntryContext,
  type Action,
  type CommandBinding,
  type ContextMatcher,
  type InputContext,
  type KeyInput,
  type KeyResolution,
  type KeySequence,
  type StrokePattern,
} from "./input-context.ts";

// The ratified value (t-36@r49; scenario 10, open decision 1: the
// value is a tweakable default - changing it is a mechanical task,
// changing what expiry means is a ratification).
export const SEQUENCE_TIMEOUT_MS = 1000;

export type MatcherOptions = Readonly<{
  // Injectable clock for deterministic tests; Date.now by default.
  now?: () => number;
}>;

// One declared sequence with its action, flattened from the current
// binding table. Read fresh on every resolve/expire: queries read
// current commands, never a frozen copy (contract).
type Declared = Readonly<{ sequence: KeySequence; action: Action }>;

function declaredOf(context: InputContext): readonly Declared[] {
  const out: Declared[] = [];
  for (const binding of commandsOf(context) as readonly CommandBinding[]) {
    for (const sequence of binding.keys) {
      out.push({ sequence, action: binding.action });
    }
  }
  return out;
}

// All declared strokes consumed by `strokes` as a prefix (the
// sequence could still continue).
function prefixMatches(declared: Declared, strokes: readonly KeyInput[]): boolean {
  if (declared.sequence.length < strokes.length) return false;
  return strokes.every((stroke, index) => strokeMatches(declared.sequence[index] as StrokePattern, stroke));
}

// The declared sequence is exactly the typed strokes.
function exactMatches(declared: Declared, strokes: readonly KeyInput[]): boolean {
  return declared.sequence.length === strokes.length && prefixMatches(declared, strokes);
}

function firstExact(declared: readonly Declared[], strokes: readonly KeyInput[]): Declared | null {
  // First declared wins. Collision-as-validation-error is a draft
  // invariant, not ratified (contract tail; scenario 10 keeps
  // collision handling draft), so the staged fallback is order.
  return declared.find((candidate) => exactMatches(candidate, strokes)) ?? null;
}

function anyPrefix(declared: readonly Declared[], strokes: readonly KeyInput[]): boolean {
  return declared.some((candidate) => candidate.sequence.length > strokes.length && prefixMatches(candidate, strokes));
}

export interface ContextMatcherWithTimeout extends ContextMatcher {
  // The ratified expiry (scenario 10): abandon the pending sequence;
  // the received keys then match their own bindings if declared, else
  // are consumed (per the context's own fallback rule). Returns the
  // drained resolutions in stroke order; [] when nothing is pending or
  // the timeout has not run out. The host schedules the check (the
  // timer is presentation-side); this method carries the semantics.
  expire(now?: number): KeyResolution[];
}

export function createMatcher(context: InputContext, options: MatcherOptions = {}): ContextMatcherWithTimeout {
  const now = options.now ?? Date.now;
  // The pending buffer: strokes typed so far that at least one
  // declared sequence still matches as a prefix (the generalization of
  // the prototype's single gPending, inst16.html:499).
  let pending: KeyInput[] = [];
  let pendingAt: number | null = null;

  // A stroke that matches no pending continuation resolves fresh, as
  // if no sequence were pending: its own single-stroke completion
  // invokes, a stroke that starts or continues a sequence pends.
  // In the executed prototype an unrelated key does not disturb the
  // pending gg (inst16.html:499: non-g keys fall through their own
  // branches while gPending stays), so the pending buffer survives.
  function resolveFresh(stroke: KeyInput): KeyResolution {
    const declared = declaredOf(context);
    const lone = [stroke];
    if (anyPrefix(declared, lone)) {
      pending = lone;
      pendingAt = now();
      return { kind: "prefix" };
    }
    const exact = firstExact(declared, lone);
    if (exact !== null) return { kind: "invoke", action: exact.action };
    return context.kind === "text-entry" ? { kind: "native-editing" } : { kind: "consume" };
  }

  return {
    resolve(stroke: KeyInput): KeyResolution {
      const declared = declaredOf(context);
      if (pending.length === 0) return resolveFresh(stroke);
      const next = [...pending, stroke];
      // A partial sequence never fires early: while some declared
      // sequence could still continue, the stroke only extends the
      // pending buffer (vim timeout semantics; the prototype's gg
      // waits for its second g, inst16.html:499).
      if (anyPrefix(declared, next)) {
        pending = next;
        pendingAt = now();
        return { kind: "prefix" };
      }
      const exact = firstExact(declared, next);
      if (exact !== null) {
        pending = [];
        pendingAt = null;
        return { kind: "invoke", action: exact.action };
      }
      // Unrelated to the pending sequence: resolved on its own, the
      // pending buffer untouched (executed prototype behavior above).
      return resolveFresh(stroke);
    },

    expire(nowMs?: number): KeyResolution[] {
      const at = nowMs ?? now();
      if (pending.length === 0 || pendingAt === null) return [];
      if (at - pendingAt < SEQUENCE_TIMEOUT_MS) return [];
      const strokes = pending;
      const declared = declaredOf(context);
      pending = [];
      pendingAt = null;
      // "received keys then match their own bindings if declared, else
      // are consumed": the completed spelling of the buffer, if
      // declared, is its own binding (the completed-but-still-
      // extendable case, vim's longest match on timeout); otherwise
      // each stroke falls back per the context's own unmatched rule.
      const exact = declared.find((candidate) => exactMatches(candidate, strokes));
      if (exact !== undefined) return [{ kind: "invoke", action: exact.action }];
      // The sequence is abandoned: a drained stroke cannot start a new
      // pending. It resolves only against its own single-stroke
      // binding, else falls to the context's unmatched rule.
      return strokes.map((stroke) => {
        const own = declared.find((candidate) => exactMatches(candidate, [stroke]));
        if (own !== undefined) return { kind: "invoke", action: own.action } as KeyResolution;
        return context.kind === "text-entry" ? { kind: "native-editing" } : { kind: "consume" };
      });
    },

    reset(): void {
      // A partial sequence never survives a handoff: dropped without
      // firing (the own-bindings drain is ratified for expiry only).
      pending = [];
      pendingAt = null;
    },
  };
}

// Re-exported so a staging consumer constructs contexts and matcher
// from one module family.
export { commandContext, textEntryContext };
