// The shared input contract: key data, semantic actions and input
// contexts. No DOM, Svelte, instrument, buffer or native-window imports.
// This defines context data, not router installation or context
// lifetimes.
//
// Port carry of the ratified contract shapes (t-36 r47-r51): the type
// declarations come from the reviewed contract drafts; the slice-1
// implementation is deliberately minimal - commandContext is the data
// constructor only. The matcher profile (normalization, sequence
// timing, prefix behavior) is its own ratified decision and lands with
// the input/context slice; nothing here invents it.

export type KeyInput = Readonly<{
  key: string;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}>;
export type KeyDisposition = "consume" | "allow-default";

// One semantic operation. Its keyboard bindings belong to the current
// context, not to a separate help table or an instrument's event handler.
export type Action = Readonly<{
  id: string;
  label: string;
  run: () => void | Promise<void>;
}>;

// Opaque patterns are made by shared constructors and formatted by the
// shared formatter. Exact normalization/sequence timing policy still
// needs approval; a component must not invent its own KeyboardEvent
// parser. The constructors stay declared: no implementation exists
// before the matcher contract is ratified.
declare const strokePattern: unique symbol;
export type StrokePattern = Readonly<{ [strokePattern]: true }>;
export type KeySequence = readonly [StrokePattern, ...StrokePattern[]];
export declare function key(symbol: string): StrokePattern;
export declare function ctrl(symbol: string): StrokePattern;
export declare function shift(symbol: string): StrokePattern;
export declare function sequence(
  first: StrokePattern,
  ...rest: StrokePattern[]
): KeySequence;

export type CommandBinding = Readonly<{
  action: Action;
  // Alternatives for the SAME action. For example d and Ctrl-d.
  keys: readonly KeySequence[];
}>;
export type ReadCommands = () => readonly CommandBinding[];

// A runtime symbol, not a type-only brand: the constructor below must
// exist at runtime, and strip-only type runners erase declarations.
const contextBrand: unique symbol = Symbol("saccade-port.input-context");
export type InputContext = Readonly<{
  [contextBrand]: true;
  kind: "commands" | "text-entry";
}>;

// The slice-1 data constructor: it builds the branded context value and
// captures nothing - the owning controller installs the resulting
// context through its restricted handoff capability. It resolves no
// keys; the matcher that will read these tables arrives with its own
// ratified contract.
export function commandContext(commands: ReadCommands): InputContext {
  void commands;
  return { [contextBrand]: true, kind: "commands" } as InputContext;
}
export declare function textEntryContext(commands: ReadCommands): InputContext;
