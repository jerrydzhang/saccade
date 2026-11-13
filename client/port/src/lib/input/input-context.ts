// The ratified input-context contract, implemented. Lifted into the
// port from client/port-staging/input-context.ts (the t-36 slice-2
// extraction, t-36@r65) unchanged: it implements t-36@r46's
// reference/input-context-contract.ts. Pure logic only: no DOM, no
// Svelte, no instrument or buffer imports.
//
// The shapes the ratified draft declares opaque are implemented here
// with the same names; StrokePattern carries its fields visibly so the
// shared matcher compares without parsing, and contexts expose their
// command accessor through commandsOf (both contract-family code,
// recorded in the r65 dispositions).

export type KeyInput = Readonly<{
  key: string;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}>;
export type KeyDisposition = "consume" | "allow-default";

// One semantic operation. Its keyboard bindings belong to the current
// context, not to a separate help table or an instrument's event
// handler (contract: Action; scenario 9's "one action, every
// invocation agrees").
export type Action = Readonly<{
  id: string;
  label: string;
  run: () => void | Promise<void>;
}>;

// Patterns are made only by the constructors below and formatted only
// by formatStroke/formatSequence (the draft's "shared formatter").
const strokePatternBrand: unique symbol = Symbol("input-context.strokePattern");
export type StrokePattern = KeyInput & Readonly<{ [strokePatternBrand]: true }>;
export type KeySequence = readonly [StrokePattern, ...StrokePattern[]];

export function key(symbol: string): StrokePattern {
  return { key: symbol, ctrl: false, shift: false, alt: false, meta: false, [strokePatternBrand]: true };
}
export function ctrl(symbol: string): StrokePattern {
  return { key: symbol, ctrl: true, shift: false, alt: false, meta: false, [strokePatternBrand]: true };
}
export function shift(symbol: string): StrokePattern {
  return { key: symbol, ctrl: false, shift: true, alt: false, meta: false, [strokePatternBrand]: true };
}
export function sequence(first: StrokePattern, ...rest: StrokePattern[]): KeySequence {
  return [first, ...rest];
}

// Field equality: a stroke matches a pattern when the key and every
// modifier flag are equal, except Shift on a printable character
// (below). Other modifier normalization stays draft-open (the
// IME/native-editing boundary set, grouped open).
//
// SHIFT AND PRINTABLE CHARACTERS (m-552, the class behind N and '+'): a
// single printable character already encodes Shift - '+' is Shift+= on
// a US layout and plain on the keypad, '?' is Shift+/, 'N' is Shift+n -
// so for those the Shift flag is not compared: the character decides,
// letters by their case. Named keys (Enter, Escape, ArrowDown) keep Shift
// exact, and Ctrl, Alt and Meta are always exact.
export function strokeMatches(pattern: StrokePattern, stroke: KeyInput): boolean {
  const printable = [...pattern.key].length === 1;
  return (
    pattern.key === stroke.key &&
    pattern.ctrl === stroke.ctrl &&
    (printable || pattern.shift === stroke.shift) &&
    pattern.alt === stroke.alt &&
    pattern.meta === stroke.meta
  );
}

export type CommandBinding = Readonly<{
  action: Action;
  // Alternatives for the SAME action, for example d and Ctrl-d.
  keys: readonly KeySequence[];
}>;
export type ReadCommands = () => readonly CommandBinding[];

const contextBrand: unique symbol = Symbol("input-context.context");
export type InputContext = Readonly<{
  [contextBrand]: true;
  kind: "commands" | "text-entry";
  commands: ReadCommands;
  // CONTRACT AMENDMENT (slice 2): sessions with a dynamic alphabet
  // (label narrowing) cannot declare their letters as bindings. The
  // SHARED matcher consults this hook for a stroke no declared
  // binding claims, before its own fallback. It is part of the
  // context's declaration surface - one dispatch chain, never a
  // competing keyboard parser; returning nothing falls through to
  // the default unmatched rule.
  onUnmatched?: (stroke: KeyInput) => KeyResolution | undefined;
}>;

// Neither constructor captures input or enters a mode; the owning
// controller installs the context through its handoff capability.
export function commandContext(commands: ReadCommands): InputContext {
  return { [contextBrand]: true, kind: "commands", commands };
}
export function textEntryContext(commands: ReadCommands): InputContext {
  return { [contextBrand]: true, kind: "text-entry", commands };
}

// The contract-family accessor for the matcher and help. Queries read
// CURRENT commands, never a frozen copy of an earlier view state; the
// prototype reads the live table every time (inst16.html:71-97).
export function commandsOf(context: InputContext): readonly CommandBinding[] {
  return context.commands();
}

// A stroke's help label. Space reads "spc" (inst16 openLeader; the
// app's keyLabel). A staging default the slice may pin later.
export function formatStroke(pattern: StrokePattern): string {
  let out = "";
  if (pattern.ctrl) out += "Ctrl-";
  if (pattern.shift) out += "Shift-";
  if (pattern.alt) out += "Alt-";
  if (pattern.meta) out += "Meta-";
  out += pattern.key === " " ? "spc" : pattern.key;
  return out;
}

// A sequence's help label: strokes in order, space-separated ("g g"),
// one label per alternative sequence.
export function formatSequence(sequence: KeySequence): string {
  return sequence.map(formatStroke).join(" ");
}

// Help/menu UI reads this projection from the context's command
// definitions; no independently authored key/label list.
export type CommandDescription = Readonly<{
  actionId: string;
  label: string;
  keyLabels: readonly string[];
}>;
export function describeCommands(context: InputContext): readonly CommandDescription[] {
  return commandsOf(context).map((binding) => ({
    actionId: binding.action.id,
    label: binding.action.label,
    keyLabels: binding.keys.map(formatSequence),
  }));
}

// Resolves keys only. It does not activate buffers, change history
// facilities or swallow asynchronous errors. Owner-specific completion
// (for example close a menu before invoking its action) stays in that
// controller's approved actions.
export type KeyResolution =
  | Readonly<{ kind: "invoke"; action: Action }>
  | Readonly<{ kind: "prefix" }>
  | Readonly<{ kind: "consume" }>
  | Readonly<{ kind: "native-editing" }>;

// A matcher has per-context sequence state, not global cross-window
// state; replacing or releasing a context must not leak a pending
// prefix into another. See key-matcher.ts for the implemented
// semantics together with the ratified sequence timeout
// (SEQUENCE_TIMEOUT_MS = 1000, t-36@r49).
export interface ContextMatcher {
  resolve(key: KeyInput): KeyResolution;
  reset(): void;
}
