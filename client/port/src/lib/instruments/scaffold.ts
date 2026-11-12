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
