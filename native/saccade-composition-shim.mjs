// Saccade's composition preload: registered only by the saccade-pi
// wrapper, it teaches the pinned runtime's sendUserMessage to forward
// one receipt option to prompt, so a Saccade delivery can observe its
// own native acceptance (queued at steer insertion, started at run
// start, handled on interception) instead of inferring it. Ordinary
// pinned Pi never loads this file. The registration is
// argument-scoped: the hook is installed only in the process the
// wrapper launches, and child processes the session spawns run their
// own Node invocations without the flag, so they never load this file
// at all.
import { register } from "node:module";

register("./saccade-composition-hooks.mjs", import.meta.url);
