// The instrument scaffold (t-36 r105, order of work step 1): the
// load-loop scaffolding every instrument was rebuilding by hand.
// An instrument supplies fetch, model, and acts; the scaffold owns
// the rest. This module starts with the pure helpers both
// controllers copied; the stateful load loop (generation, released,
// askedSpec adoption) extracts here next, one controller at a time.

import type { ContentFrame, PreparedContent } from "../contracts/buffer.ts";
import { knownIssue, type KnownIssue } from "../contracts/buffer.ts";
import { SourceUnavailableError } from "../sources.ts";

/** Wrap a prepared domain frame as the buffer's published content:
 * the logical content metadata rides with its data. */
export function readyFrame<D extends object>(
  value: Readonly<D>,
): PreparedContent<ContentFrame<D>> {
  return { kind: "ready", value: { value, targets: [] } };
}

export function failureIssue(code: string, message: string): KnownIssue {
  return knownIssue({ code, message });
}

/** Classify a load failure: a source's own unavailable verdict is
 * carried verbatim; everything else reads as a generic failure. */
export function issueOf(error: unknown): KnownIssue {
  if (error instanceof SourceUnavailableError) {
    return failureIssue(error.code, error.message);
  }
  return failureIssue("read-failed", String(error));
}

/** The load loop's shared state (order of work step 1, stateful
 * half): the open generation that invalidates in-flight loads, the
 * last adopted frame, the asked spec, and the release flag. An
 * instrument's load function supplies the domain work; this scope
 * owns the bookkeeping both controllers copied. */
export interface LoadScope<D> {
  /** Start a load: bumps the generation and returns its token. */
  begin(): number;
  /** True when the token's load is no longer the current one
   * (superseded by a newer begin, or the scope was destroyed). */
  isStale(token: number): boolean;
  /** Destroy the scope: every current and future load is stale. */
  destroy(): void;
  /** Whether the scope is destroyed. */
  isDestroyed(): boolean;
  /** The most recent adopted frame (the instrument's empty until
   * the first adoption). */
  frame(): D;
  /** Adopt a frame as the current one. */
  adopt(frame: D): void;
  /** The location spec the current load was asked for. */
  asked(): string;
  setAsked(spec: string): void;
}

export function createLoadScope<D>(empty: D): LoadScope<D> {
  let generation = 0;
  let released = false;
  let current: D = empty;
  let askedSpec = "";
  return {
    begin(): number {
      return ++generation;
    },
    isStale(token: number): boolean {
      return released || token !== generation;
    },
    destroy(): void {
      released = true;
    },
    isDestroyed(): boolean {
      return released;
    },
    frame(): D {
      return current;
    },
    adopt(frame: D): void {
      current = frame;
    },
    asked(): string {
      return askedSpec;
    },
    setAsked(spec: string): void {
      askedSpec = spec;
    },
  };
}
