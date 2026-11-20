// The ratified input-context contract, implemented (t-36@r46
// reference/input-context-contract.ts is the ratified draft this file
// implements; the extraction mandate is t-36 slice 2, scenario groups
// 9, 10 and 16 of reference/acceptance-scenarios.txt, t-36@r50).
// Pure logic only: no DOM, no Svelte, no instrument or buffer imports.
//
// What the draft declares as opaque/unique-symbol shapes is implemented
// here with the same names; the two staged deviations are recorded in
// README.md (the StrokePattern carries its fields visibly so the shared
// matcher can compare without parsing, and contexts expose their
// command accessor through commandsOf, both contract-family code).

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

// Exact field equality: a stroke matches a pattern when the key and
// every modifier flag are equal. Modifier normalization values are
// explicitly draft-open (contract tail: "exact modifier normalization
// ... must be pinned before delegating a matcher"), so the matcher
// invents none; the host normalizes KeyboardEvents before calling.
export function strokeMatches(pattern: StrokePattern, stroke: KeyInput): boolean {
  return (
    pattern.key === stroke.key &&
    pattern.ctrl === stroke.ctrl &&
    pattern.shift === stroke.shift &&
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
// CURRENT commands, never a frozen copy of an earlier view state
// (contract on describeCommands; the prototype reads the live table
// every time: commands() is called fresh in runCmd and showHelp,
// inst16.html:71-97).
export function commandsOf(context: InputContext): readonly CommandBinding[] {
  return context.commands();
}

// A stroke's help label. Space reads "spc" (inst16.html openLeader,
// :86 `k === " " ? "spc"`; the app's keyLabel, menu.ts@b8240b5:38-40).
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
// definitions; no independently authored key/label list. The prototype's
// help sheet reads the same table the runner does (showHelp and runCmd
// both call commands(), inst16.html:90-97).
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
// prefix into another. Implemented in key-matcher.ts together with the
// ratified sequence-timeout semantics (SEQUENCE_TIMEOUT_MS = 1000,
// t-36@r49; scenario 10).
export interface ContextMatcher {
  resolve(key: KeyInput): KeyResolution;
  reset(): void;
}
