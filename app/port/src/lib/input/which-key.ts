// The which-key projection (the r110 ruling, batch 3's contract):
// a PURE function over an input context's declared command tables -
// the continuations a pending prefix offers, in describeCommands'
// own labels so help and which-key cannot drift from the table. The
// tunables are named and read (the human tunes), never pinned.
import type { InputContext, KeyInput } from "./input-context.ts";
import { commandsOf, formatSequence, strokeMatches } from "./input-context.ts";
import type { KeySequence, StrokePattern } from "./input-context.ts";

export type WhichKeyEntry = Readonly<{
  /** The FULL sequence's help label ("spc l", "g g", "Ctrl-f"). */
  keys: string;
  label: string;
  actionId: string;
}>;

/** The named tunables (the r110 pair): the popup's delay and the
 * timeout suspension while it is open (nvim's default: suspend). */
export const whichKeyTunables: Readonly<{ delayMs: number; suspendsTimeout: boolean }> = {
  delayMs: 400,
  suspendsTimeout: true,
};

function sequenceStartsWith(sequence: KeySequence, prefix: readonly KeyInput[]): boolean {
  if (sequence.length <= prefix.length) return false;
  return prefix.every((stroke, index) => {
    const pattern = sequence[index] as StrokePattern;
    return strokeMatches(pattern, stroke);
  });
}

/** Every declared command whose sequences CONTINUE the prefix: one
 * entry per alternative sequence, keyed by the full sequence's help
 * label. The empty prefix lists every command (the help surface). */
export function continuations(
  context: InputContext,
  prefix: readonly KeyInput[],
): readonly WhichKeyEntry[] {
  const out: WhichKeyEntry[] = [];
  for (const binding of commandsOf(context)) {
    for (const sequence of binding.keys) {
      if (sequenceStartsWith(sequence, prefix)) {
        out.push({
          keys: formatSequence(sequence),
          label: binding.action.label,
          actionId: binding.action.id,
        });
      }
    }
  }
  return out;
}
