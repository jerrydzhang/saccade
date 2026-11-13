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
  /** The matcher's which-key views (the r110 ruling), pass-through:
   * the pending prefix's continuations and the empty-prefix help
   * view, from this host's own context's declared commands. */
  pendingView(): import("./key-matcher.ts").WhichKeyView | null;
  allView(): import("./key-matcher.ts").WhichKeyView;
  pendingSince(): number | null;
  /** Notified whenever the pending state may have changed (a stroke,
   * a drain, a reset) - the popup's reactive source. */
  subscribePending(observer: () => void): () => void;
}

export function createHostMatcher(
  context: InputContext,
  options: Readonly<{
    scheduler?: HostScheduler;
    /** The which-key timeout suspension (the r110 tunable, nvim's
     * default): while true, an open popup freezes the sequence
     * expiry clock - the drain re-arms instead of firing. */
    expirySuspended?: () => boolean;
  }> = {},
): HostMatcher {
  const scheduler = options.scheduler ?? domScheduler;
  const matcher: ContextMatcherWithTimeout = createMatcher(context);
  let timer: number | null = null;
  const pendingObservers = new Set<() => void>();
  const notifyPending = (): void => {
    for (const observer of [...pendingObservers]) observer();
  };

  function drain(): void {
    timer = null;
    if (options.expirySuspended !== undefined && options.expirySuspended()) {
      // the popup holds the sequence: freeze the clock (the pending
      // restarts its window) and re-arm
      matcher.freezePending();
      timer = scheduler.schedule(SEQUENCE_TIMEOUT_MS, drain);
      return;
    }
    for (const resolution of matcher.expire()) {
      // Invoke resolutions run; consume/native-editing resolutions
      // from the drain carry no further action (the keys were already
      // received; their disposition was consumed at receipt).
      if (resolution.kind === "invoke") {
        void resolution.action.run();
      }
    }
    notifyPending();
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
        notifyPending();
      } else {
        notifyPending();
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
      notifyPending();
    },
    pendingView: () => matcher.pendingView(),
    allView: () => matcher.allView(),
    pendingSince: () => matcher.pendingSince(),
    subscribePending(observer: () => void): () => void {
      pendingObservers.add(observer);
      return () => {
        pendingObservers.delete(observer);
      };
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
  /** Read-only: told who answered and how (the key trace). */
  observe?: (owner: "window" | "buffer", answer: string) => void,
): KeyDisposition {
  const resolution = window.stroke(input);
  observe?.("window", resolution.kind === "invoke" ? `invoke:${resolution.action.id}` : resolution.kind);
  if (resolution.kind === "invoke") return window.apply(resolution);
  if (resolution.kind === "prefix") return window.apply(resolution);
  // The window context did not claim the stroke (or is a commands
  // context answering consume for unmatched): the active buffer
  // decides.
  return buffer.handleKey(input);
}
