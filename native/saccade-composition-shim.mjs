// Saccade's composition preload: registered only by the saccade-pi
// wrapper, it teaches the pinned runtime's sendUserMessage to forward
// one receipt option to prompt, so a Saccade delivery can observe its
// own native acceptance (queued at steer insertion, started at run
// start, handled on interception) instead of inferring it. Ordinary
// pinned Pi never loads this file; other processes that inherit the
// registration run a path-guarded hook that transforms nothing.
import { register } from "node:module";
import { pathToFileURL } from "node:url";

register("./saccade-composition-hooks.mjs", import.meta.url);
