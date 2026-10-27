// Saccade's composition preload: registered only by the saccade-pi
// wrapper, it teaches the pinned runtime's sendUserMessage to forward
// one receipt option to prompt, so a Saccade delivery can observe its
// own native acceptance instead of inferring it. Ordinary pinned Pi
// never loads this file; the registration is argument-scoped, so
// child processes never inherit it.
import { register } from "node:module";

register("./saccade-composition-hooks.mjs", import.meta.url);
