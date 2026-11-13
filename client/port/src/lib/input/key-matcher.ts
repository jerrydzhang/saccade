// The shared key matcher over the ratified input-context contract.
// Lifted into the port from client/port-staging/key-matcher.ts
// (t-36@r65) with ONE ratified change: collision detection at load
// (validateNoCollisions, t-36@r49 standing policy "collisions = error
// at load", confirmed by the r65 disposition). The staged first-wins
// fallback stays only as dead-code safety behind the validation.
//
// Scenario clauses (t-36@r50 acceptance-scenarios.txt, scenario 10):
// unmatched application keys are consumed in the command context;
// text entry matches declared commands first, other typing edits the
// field natively and never reaches the board keymap; help and
// execution read the same binding declarations; sequences run through
// the one matcher; a partial sequence never fires early and never
// survives a handoff; SEQUENCE_TIMEOUT_MS = 1000 (vim timeoutlen,
// t-36@r49), pending abandoned at expiry, received keys then match
// their own bindings if declared, else are consumed (adapter A1).

import {
  commandsOf,
  strokeMatches,
  commandContext,
  textEntryContext,
  formatStroke,
  type Action,
  type CommandBinding,
  type ContextMatcher,
  type InputContext,
  type KeyInput,
  type KeyResolution,
  type KeySequence,
  type StrokePattern,
} from "./input-context.ts";

// The ratified value (t-36@r49; the value is a tweakable default -
// changing it is a mechanical task; changing what expiry means is a
// ratification).
export const SEQUENCE_TIMEOUT_MS = 1000;

export type MatcherOptions = Readonly<{
  // Injectable clock for deterministic tests; Date.now by default.
  now?: () => number;
}>;

// COLLISIONS ARE ERRORS AT LOAD (t-36@r49 standing policy, wired by
// r65). Two bindings of DIFFERENT actions claiming the same stroke
// spelling refuse the whole context: silent shadowing is exactly the
// reliability bug t-36 exists to kill. Alternatives for one action
// stay legal.
export function validateNoCollisions(context: InputContext): void {
  const owner = new Map<string, string>();
  const clashes: string[] = [];
  for (const binding of commandsOf(context)) {
    for (const sequence of binding.keys) {
      const spelling = sequence.map(formatStroke).join(" ");
      const held = owner.get(spelling);
      if (held !== undefined && held !== binding.action.id) {
        clashes.push(`'${spelling}' declared by both ${held} and ${binding.action.id}`);
      }
      owner.set(spelling, binding.action.id);
    }
  }
  if (clashes.length > 0) {
    throw new Error(`input collision at load: ${clashes.join("; ")}`);
  }
}

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

function prefixMatches(declared: Declared, strokes: readonly KeyInput[]): boolean {
  if (declared.sequence.length < strokes.length) return false;
  return strokes.every((stroke, index) => strokeMatches(declared.sequence[index] as StrokePattern, stroke));
}

function exactMatches(declared: Declared, strokes: readonly KeyInput[]): boolean {
  return declared.sequence.length === strokes.length && prefixMatches(declared, strokes);
}

function firstExact(declared: readonly Declared[], strokes: readonly KeyInput[]): Declared | null {
  // Unreachable behind validateNoCollisions; order fallback retained
  // as dead-code safety for unvalidated foreign contexts.
  return declared.find((candidate) => exactMatches(candidate, strokes)) ?? null;
}

function anyPrefix(declared: readonly Declared[], strokes: readonly KeyInput[]): boolean {
  return declared.some((candidate) => candidate.sequence.length > strokes.length && prefixMatches(candidate, strokes));
}

export interface ContextMatcherWithTimeout extends ContextMatcher {
  // The ratified expiry: abandon the pending sequence; the received
  // keys then match their own bindings if declared, else fall to the
  // context's own fallback rule. Returns drained resolutions in stroke
  // order; [] when nothing is pending or the timeout has not run out.
  // The host schedules the check (the timer is presentation-side);
  // this method carries the semantics.
  expire(now?: number): KeyResolution[];
  /** The which-key view (the r110 ruling, generalized): the pending
   * prefix and the continuations the ACTIVE context's own declared
   * commands offer for it - each row names the NEXT stroke and the
   * action it would run. Null when nothing is pending. The popup's
   * delay and its timeout suspension are the caller's tunables. */
  pendingView(): WhichKeyView | null;
  /** The same view over the EMPTY prefix: every declared sequence's
   * first stroke - the help surface (a deliberate difference from
   * the prototype's hand-curated list, recorded t-36 r125). */
  allView(): WhichKeyView;
  /** When the current pending began (the popup delay's clock), or
   * null when nothing is pending. */
  pendingSince(): number | null;
  /** Restart the pending's expiry window (the popup's suspension
   * tunable): the frozen clock resumes from now. */
  freezePending(): void;
}

export type WhichKeyView = Readonly<{
  prefix: readonly KeyInput[];
  completions: readonly { key: string; label: string }[];
  /** The readout contract's shape (batch 3): every continuing
   * command with its FULL sequence label and action id - the
   * module's own projection, so help and which-key cannot drift. */
  entries: readonly { keys: string; label: string; actionId: string }[];
}>;

import { continuations as whichKeyContinuations } from "./which-key.ts";

function keyLabel(pattern: StrokePattern): string {
  const parts: string[] = [];
  if (pattern.ctrl) parts.push("Ctrl");
  if (pattern.alt) parts.push("Alt");
  if (pattern.meta) parts.push("Meta");
  if (pattern.shift && pattern.key.length === 1) parts.push("Shift");
  parts.push(pattern.key.length === 1 ? pattern.key.toUpperCase() : pattern.key);
  return parts.join("-");
}

function completionsFor(
  declared: readonly Declared[],
  strokes: readonly KeyInput[],
): readonly { key: string; label: string }[] {
  const rows = new Map<string, { key: string; label: string }>();
  for (const candidate of declared) {
    if (!prefixMatches(candidate, strokes)) continue;
    if (candidate.sequence.length <= strokes.length) continue;
    const next = candidate.sequence[strokes.length];
    if (next === undefined) continue;
    const key = keyLabel(next);
    if (!rows.has(key)) rows.set(key, { key, label: candidate.action.label });
  }
  return [...rows.values()];
}

export function createMatcher(context: InputContext, options: MatcherOptions = {}): ContextMatcherWithTimeout {
  const now = options.now ?? Date.now;
  // The pending buffer: strokes typed so far that at least one
  // declared sequence still matches as a prefix.
  let pending: KeyInput[] = [];
  let pendingAt: number | null = null;

  // A stroke that matches no pending continuation resolves fresh, as
  // if no sequence were pending (the executed prototype behavior: an
  // unrelated key resolves on its own and the pending buffer
  // survives, inst16.html:499).
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
    // A session with a dynamic alphabet (the context's own
    // declaration surface) claims the stroke before the fallback.
    const session = context.onUnmatched?.(stroke);
    if (session !== undefined) return session;
    return context.kind === "text-entry" ? { kind: "native-editing" } : { kind: "consume" };
  }

  return {
    resolve(stroke: KeyInput): KeyResolution {
      const declared = declaredOf(context);
      if (pending.length === 0) return resolveFresh(stroke);
      const next = [...pending, stroke];
      // A partial sequence never fires early: while some declared
      // sequence could still continue, the stroke only extends the
      // pending buffer (vim timeout semantics).
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
      // Unrelated to the pending sequence: resolved on its own.
      return resolveFresh(stroke);
    },

    pendingView(): WhichKeyView | null {
      if (pending.length === 0) return null;
      return {
        prefix: [...pending],
        completions: completionsFor(declaredOf(context), pending),
        entries: whichKeyContinuations(context, pending),
      };
    },

    allView(): WhichKeyView {
      return {
        prefix: [],
        completions: completionsFor(declaredOf(context), []),
        entries: whichKeyContinuations(context, []),
      };
    },

    pendingSince(): number | null {
      return pending.length > 0 ? pendingAt : null;
    },

    freezePending(): void {
      // the which-key suspension tunable: an open popup restarts the
      // expiry window (nvim's frozen clock)
      if (pending.length > 0) pendingAt = now();
    },

    expire(nowMs?: number): KeyResolution[] {
      const at = nowMs ?? now();
      if (pending.length === 0 || pendingAt === null) return [];
      if (at - pendingAt < SEQUENCE_TIMEOUT_MS) return [];
      const strokes = pending;
      const declared = declaredOf(context);
      pending = [];
      pendingAt = null;
      // "Received keys then match their own bindings if declared, else
      // are consumed": the completed spelling of the buffer, if
      // declared, is its own binding (the completed-but-still-
      // extendable case, vim's longest match on timeout); otherwise
      // each stroke falls back per the context's own unmatched rule.
      const exact = declared.find((candidate) => exactMatches(candidate, strokes));
      if (exact !== undefined) return [{ kind: "invoke", action: exact.action }];
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

export { commandContext, textEntryContext };
