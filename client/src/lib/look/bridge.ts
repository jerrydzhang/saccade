// The look's host bridge: reading and following the local
// configuration through the narrow Tauri commands. Split from the pure
// resolution in store.ts so tests can hold the pipeline without the
// renderer-only sources module.

import { tauriInvoke, SourceUnavailableError } from "../sources.ts";
import type { LookSource } from "./store.ts";

/** Read the local configuration through the host bridge. Refuses with
 * SourceUnavailableError outside the native shell. */
export async function readLookSource(): Promise<LookSource> {
  const invoke = tauriInvoke();
  return (await invoke("look_config")) as LookSource;
}

/** Follow the configuration: resolves once, then re-resolves whenever
 * the host reloads (config file or named scheme file changed). Runs for
 * the app's lifetime; each callback sees the newest source. The wait
 * command answers after its bounded hold even when nothing changed;
 * the generation comparison decides. */
export async function followLookSource(onSource: (source: LookSource) => void): Promise<void> {
  let current = await readLookSource();
  onSource(current);
  for (;;) {
    const invoke = tauriInvoke();
    const next = (await invoke("wait_look_config", { since: current.generation })) as LookSource;
    if (next.generation !== current.generation) {
      current = next;
      onSource(current);
    }
  }
}

/** Start following the configuration where a host exists; resolves to
 * false (and the caller falls back to the built-in default with its
 * notice) in a window with no host, such as the browser lane. */
export async function startLook(onSource: (source: LookSource) => void): Promise<boolean> {
  try {
    await followLookSource(onSource);
    return true;
  } catch (error) {
    if (error instanceof SourceUnavailableError) return false;
    throw error;
  }
}
