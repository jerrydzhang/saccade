// The host-owned half of the input contract: one place schedules the
// sequence-expiry timer, applies resolutions, and chains the window's
// own context to the active buffer. The matcher module owns the
// semantics; this module owns the timer and the disposition (the
// contract's split: "the host schedules the check, the method carries
// the semantics").

import { createMatcher, SEQUENCE_TIMEOUT_MS } from "./key-matcher.ts";
import type { ContextMatcherWithTimeout } from "./key-matcher.ts";
import type { InputContext, KeyDisposition, KeyInput, KeyResolution } from "./input-context.ts";

export interface HostScheduler {
  schedule(ms: number, run: () => void): number;
  cancel(id: number): void;
}

const domScheduler: HostScheduler = {
  // The DOM timer id and Node's Timeout meet here across lib mixes;
  // the id is opaque to every consumer.
  schedule: (ms, run) => setTimeout(run, ms) as unknown as number,
  cancel: (id) => clearTimeout(id as unknown as ReturnType<typeof setTimeout>),
};

export interface HostMatcher {
  // Resolve a stroke; a prefix schedules the expiry drain (cleared on
  // completion or reset). Nothing is applied here.
  stroke(input: KeyInput): KeyResolution;
  // Apply a resolution: invoke runs the action; the return value is
  // the disposition the caller owes the platform (preventDefault on
  // consume).
  apply(resolution: KeyResolution): KeyDisposition;
  // A handoff drops pending state without firing.
  reset(): void;
  dispose(): void;
}

export function createHostMatcher(
  context: InputContext,
  options: Readonly<{ scheduler?: HostScheduler }> = {},
): HostMatcher {
  const scheduler = options.scheduler ?? domScheduler;
  const matcher: ContextMatcherWithTimeout = createMatcher(context);
  let timer: number | null = null;

  function drain(): void {
    timer = null;
    for (const resolution of matcher.expire()) {
      // Invoke resolutions run; consume/native-editing resolutions
      // from the drain carry no further action (the keys were already
      // received; their disposition was consumed at receipt).
      if (resolution.kind === "invoke") {
        void resolution.action.run();
      }
    }
  }

  function clearTimer(): void {
    if (timer !== null) {
      scheduler.cancel(timer);
      timer = null;
    }
  }

  return {
    stroke(input: KeyInput): KeyResolution {
      const resolution = matcher.resolve(input);
      if (resolution.kind === "prefix") {
        clearTimer();
        timer = scheduler.schedule(SEQUENCE_TIMEOUT_MS, drain);
      }
      return resolution;
    },
    apply(resolution: KeyResolution): KeyDisposition {
      if (resolution.kind === "invoke") {
        clearTimer();
        void resolution.action.run();
        return "consume";
      }
      if (resolution.kind === "prefix") return "consume";
      if (resolution.kind === "consume") return "consume";
      return "allow-default";
    },
    reset(): void {
      clearTimer();
      matcher.reset();
    },
    dispose(): void {
      clearTimer();
      matcher.reset();
    },
  };
}

// The window chain: the window's own context resolves first (window
// surfaces: the picker, overlays); anything it does not claim flows to
// the ACTIVE buffer's reading context - never to an inactive one. The
// buffer's answer is final: consume means the platform default is
// suppressed; allow-default passes through.
export function dispatchStroke(
  window: HostMatcher,
  buffer: Readonly<{ handleKey(input: KeyInput): KeyDisposition }>,
  input: KeyInput,
): KeyDisposition {
  const resolution = window.stroke(input);
  if (resolution.kind === "invoke") return window.apply(resolution);
  if (resolution.kind === "prefix") return window.apply(resolution);
  // The window context did not claim the stroke (or is a commands
  // context answering consume for unmatched): the active buffer
  // decides.
  return buffer.handleKey(input);
}
