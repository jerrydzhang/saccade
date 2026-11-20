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

/** One live suite window as the shell registry knows it: the label is
 * the window's identity, the link names the window it receives `o`
 * into, opensAt is the place it was born to show. */
export type ShellWindow = { label: string; link?: string; opensAt?: string };

/** One shell command the app invoked, with its arguments. */
export type ShellCall = { cmd: string; args: Record<string, unknown> };

export type ShellWorld = {
  /** This webview's label (the app under test). */
  label: string;
  /** Every live suite window, this one included. */
  windows: ShellWindow[];
  /** The lanes' test-open channel (SAC_APP_TEST_OPEN). */
  testOpen: string | null;
  /** Every invoke, in order. */
  calls: ShellCall[];
  /** Whether SACCADE_KEY_TRACE is set for this shell (off unless a
   * scenario asks). */
  keyTrace?: boolean;
  /** Herdr's words feed as the shell holds it (words.rs WordsSnapshot,
   * its wire names); absent, the feed is connected and holds none. */
  words?: { generation: number; words: Record<string, Record<string, unknown>>; connected: boolean };
};

export const SHELL_WORLD_KEY = "__portSuiteShell";

type PortGlobals = Window & {
  __suiteOpenAt?: (spec: string) => unknown;
  __suiteJumpShow?: (chip: string) => unknown;
  __suiteJumpHide?: () => unknown;
};

// app/src-tauri/src/suite.rs fresh_label: "w-<unix seconds><hex counter>"
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
function withLink(label: string, link: string | undefined): { label: string; link?: string } {
  return link === undefined ? { label } : { label, link: link };
}

function withOpensAt(label: string, opensAt: string | undefined): { label: string; opens_at?: string } {
  return opensAt === undefined ? { label } : { label, opens_at: opensAt };
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
      // window_registry.rs window_binding(window) -> Binding { label, opens_at: Option<String> }
      case "window_binding":
        return withOpensAt(world.label, own(world).opensAt);
      // window_registry.rs set_window_link(registry, from, to) -> Result<(), String>
      case "set_window_link": {
        const from = String(args.from);
        const to = String(args.to);
        if (from === to) throw `a window does not link to itself (${from})`;
        if (!world.windows.some((entry) => entry.label === from)) throw `no live window carries ${from}`;
        if (!world.windows.some((entry) => entry.label === to)) throw `no live window carries ${to}`;
        const source = world.windows.find((entry) => entry.label === from)!;
        source.link = to;
        return null;
      }
      // window_registry.rs clear_window_link(registry, from) -> ()
      case "clear_window_link": {
        const source = world.windows.find((entry) => entry.label === String(args.from));
        if (source !== undefined) delete source.link;
        return null;
      }
      // suite.rs suite_windows(app, registry) -> Vec<SuiteWindow { label, link }>, label order
      case "suite_windows":
        return [...world.windows]
          .sort((a, b) => (a.label < b.label ? -1 : a.label > b.label ? 1 : 0))
          .map((entry) => withLink(entry.label, entry.link));
      // suite.rs open_suite_window(app, registry, location: Option<String>) -> Result<String, String>
      case "open_suite_window": {
        const label = freshLabel(world);
        const location = typeof args.location === "string" ? args.location : undefined;
        world.windows.push(location === undefined ? { label } : { label, opensAt: location });
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
      // suite.rs window_focus_state(window) -> FocusState { accepts_focus,
      // focus_on_map, is_active, focus_widget: Option<String> }: GTK's facts; the mocked webview is the
      // one focused window
      case "window_focus_state":
        return { accepts_focus: true, focus_on_map: true, is_active: true, focus_widget: "WebKitWebView" };
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
      // words.rs words_wait(since: u64, timeout: Option<u64>) -> WordsSnapshot:
      // a newer generation answers at once; otherwise the hold ends with
      // the same snapshot (held one second here, never the full timeout)
      case "words_wait": {
        const words = world.words ?? { generation: 0, words: {}, connected: true };
        if (Number(args.since) < words.generation) return words;
        return new Promise((resolve) => setTimeout(() => resolve(world.words ?? words), 1000));
      }
      default:
        throw `the shell mock does not carry ${cmd} (kit/shell-mock.ts)`;
    }
  });
}
