// The shared input contract: key data, semantic actions and input
// contexts. No DOM, Svelte, instrument, buffer or native-window imports.
//
// ONE SOURCE OF TRUTH: the ratified contract is implemented in
// ../input/input-context.ts (lifted from the t-36 slice-2 extraction,
// t-36@r65). This path re-exports it so contract consumers keep one
// import surface; the slice-1 placeholder shapes are retired.

export {
  key,
  ctrl,
  shift,
  sequence,
  strokeMatches,
  commandContext,
  textEntryContext,
  commandsOf,
  formatStroke,
  formatSequence,
  describeCommands,
} from "../input/input-context.ts";
export type {
  KeyInput,
  KeyDisposition,
  Action,
  StrokePattern,
  KeySequence,
  CommandBinding,
  ReadCommands,
  InputContext,
  CommandDescription,
  KeyResolution,
  ContextMatcher,
} from "../input/input-context.ts";
