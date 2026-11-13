// The browser project's mocked native shell: Tauri's OFFICIAL IPC
// mocks (@tauri-apps/api/mocks mockIPC and mockWindows) answering the
// shell commands the port invokes, each written from the Rust
// signature and return shape cited beside it. It models ONE webview,
// the app under test, plus the shell's registry of other windows as
// plain data; it never routes anything between pages - whatever needs
// a second live webview is the native project's.
//
// It runs INSIDE the app's iframe (kit/shell-app.html), because the
// mocks patch the realm's own window.__TAURI_INTERNALS__. The world it
// answers from, and the log every call lands in, belong to the test
// page (window.parent), so a scenario reads which command fired with
// which arguments, and changes the shell's world (a window closing).
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";

/** One live suite window as the shell registry knows it. */
export type ShellWindow = { label: string; stateId?: string };

/** One shell command the app invoked, with its arguments. */
export type ShellCall = { cmd: string; args: Record<string, unknown> };

export type ShellWorld = {
  /** This webview's label (the app under test). */
  label: string;
  /** Every live suite window, this one included. */
  windows: ShellWindow[];
  /** The launch restore's memory: taken once, then empty. */
  lastSession: string[];
  /** The lanes' test-open channel (SAC_APP_TEST_OPEN). */
  testOpen: string | null;
  /** Every invoke, in order. */
  calls: ShellCall[];
  /** Whether SACCADE_KEY_TRACE is set for this shell (off unless a
   * scenario asks). */
  keyTrace?: boolean;
};

export const SHELL_WORLD_KEY = "__portSuiteShell";

type PortGlobals = Window & {
  __suiteOpenAt?: (spec: string) => unknown;
  __suiteJumpShow?: (chip: string) => unknown;
  __suiteJumpHide?: () => unknown;
};

// client/src-tauri/src/suite.rs fresh_label: "w-<unix seconds><hex counter>"
function freshLabel(world: ShellWorld): string {
  const seconds = Math.floor(Date.now() / 1000);
  for (let counter = 0; ; counter++) {
    const candidate = `w-${seconds}${counter.toString(16)}`;
    if (!world.windows.some((entry) => entry.label === candidate)) return candidate;
  }
}

function own(world: ShellWorld): ShellWindow {
  const found = world.windows.find((entry) => entry.label === world.label);
  if (found === undefined) throw new Error(`the shell world does not carry this window ${world.label}`);
  return found;
}

// serde's Option skip_serializing_if = "Option::is_none": absent, never null
function withState(label: string, stateId: string | undefined): { label: string; state_id?: string } {
  return stateId === undefined ? { label } : { label, state_id: stateId };
}

export function installShell(world: ShellWorld): void {
  const self = window as PortGlobals;
  mockWindows(world.label);
  mockIPC((cmd, payload) => {
    const args = (payload ?? {}) as Record<string, unknown>;
    world.calls.push({ cmd, args: JSON.parse(JSON.stringify(args)) as Record<string, unknown> });
    switch (cmd) {
      // commands.rs test_open() -> Option<String>
      case "test_open":
        return world.testOpen;
      // window_registry.rs window_binding(window) -> Binding { label, state_id: Option<String> }
      case "window_binding":
        return withState(world.label, own(world).stateId);
      // window_registry.rs bind_window_state(window, state_id, server, device) -> ()
      case "bind_window_state":
        own(world).stateId = String(args.stateId);
        return null;
      // session.rs last_session() -> Vec<String>, taken once
      case "last_session": {
        const last = world.lastSession;
        world.lastSession = [];
        return last;
      }
      // suite.rs suite_windows(app, registry) -> Vec<SuiteWindow { label, state_id }>, label order
      case "suite_windows":
        return [...world.windows]
          .sort((a, b) => (a.label < b.label ? -1 : a.label > b.label ? 1 : 0))
          .map((entry) => withState(entry.label, entry.stateId));
      // suite.rs open_suite_window(app, registry, state_id: Option<String>) -> Result<String, String>
      case "open_suite_window": {
        const label = freshLabel(world);
        const stateId = typeof args.stateId === "string" ? args.stateId : undefined;
        world.windows.push(stateId === undefined ? { label } : { label, stateId });
        return label;
      }
      // suite.rs focus_suite_window(app, label) -> Result<(), String>
      case "focus_suite_window":
        if (!world.windows.some((entry) => entry.label === args.label)) throw `no suite window ${String(args.label)}`;
        return null;
      // suite.rs open_in_window(app, label, location) -> Result<(), String>: evals
      // window.__suiteOpenAt(location) into that webview
      case "open_in_window":
        if (!world.windows.some((entry) => entry.label === args.label)) throw `no suite window ${String(args.label)}`;
        if (args.label === world.label) self.__suiteOpenAt?.(String(args.location));
        return null;
      // suite.rs broadcast_jump_labels(app, labels: Vec<JumpLabel { window, chip }>) -> ():
      // evals window.__suiteJumpShow(chip) into each listed live webview
      case "broadcast_jump_labels":
        for (const entry of (args.labels ?? []) as { window: string; chip: string }[]) {
          if (entry.window === world.label) self.__suiteJumpShow?.(entry.chip);
        }
        return null;
      // suite.rs broadcast_jump_labels_clear(app) -> (): evals __suiteJumpHide() everywhere
      case "broadcast_jump_labels_clear":
        self.__suiteJumpHide?.();
        return null;
      // commands.rs key_trace(window, lines: Vec<String>) -> bool: whether
      // SACCADE_KEY_TRACE is set; when it is, the lines go to stderr
      // (here: the call log carries them)
      case "key_trace":
        return world.keyTrace === true;
      // commands.rs machine_id() -> Option<String> (/etc/machine-id): no machine id here,
      // so no tracker is ever proven to be this machine
      case "machine_id":
        return null;
      // commands.rs tracker_machine() -> Option<String>
      case "tracker_machine":
        return null;
      // board_validate.rs validate_board(text) -> BoardVerdict, serde tag "verdict" kebab-case.
      // The crate itself is not in the browser: "ok" stands for the valid seed boards only;
      // the crate/decoder admission parity is pinned by the unit tests (lib/board/validate.ts)
      case "validate_board":
        return { verdict: "ok" };
      default:
        throw `the shell mock does not carry ${cmd} (kit/shell-mock.ts)`;
    }
  });
}
