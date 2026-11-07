// The bridge verdict: admission is the shared board-grammar crate's job
// on the Tauri side, through the validate_board command. Null means no
// bridge exists (the browser lane), where the presentation decoder alone
// produces the verdict; its contract mirrors the crate's and the parity
// is pinned over the shared fixtures by tests on both sides.

import { tauriInvoke, SourceUnavailableError } from "../sources.ts";

export type CrateVerdict =
  | { verdict: "ok" }
  /** The crate refuses only because the board names a kind this build
   * does not know; the viewer renders those as raw source. */
  | { verdict: "unknown-kind"; message: string }
  /** A refusal: unknown major version, malformed board, anything the
   * crate rejects with its final word. */
  | { verdict: "refused"; kind: string; message: string; line?: number; column?: number };

export async function crateVerdict(text: string): Promise<CrateVerdict | null> {
  let invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
  try {
    invoke = tauriInvoke();
  } catch (error) {
    if (error instanceof SourceUnavailableError) return null;
    throw error;
  }
  try {
    return (await invoke("validate_board", { text })) as CrateVerdict;
  } catch (error) {
    throw new SourceUnavailableError(
      "bridge-invalid",
      `the bridge's grammar validator failed: ${String(error)}`,
    );
  }
}
