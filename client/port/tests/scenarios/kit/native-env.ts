// The native project's sandbox check, free of vitest imports so the
// globalSetup can refuse before any tracker starts (kit/tracker-setup.ts).

// the session lane's own env names (client/harness/entry.py
// session_env); WAYLAND_DISPLAY names the lane's private compositor
const REQUIRED = ["TAURI_DRIVER_BIN", "WEBKIT_DRIVER_BIN", "APP_BIN", "WAYLAND_DISPLAY", "SAC_APP_SESSION_TOKEN"] as const;
type Pinned = Record<(typeof REQUIRED)[number], string>;

/** Refuse outside the session lane's sandbox, naming what is missing
 * (the setup calls it before any tracker starts). */
export function pinned(): Pinned {
  const missing = REQUIRED.filter((name) => !process.env[name]);
  if (missing.length > 0) {
    throw new Error(
      `the native project runs only inside the session lane's sandbox (missing ${missing.join(", ")});`
        + " see tests/scenarios/README.md, native",
    );
  }
  return Object.fromEntries(REQUIRED.map((name) => [name, process.env[name]!])) as Pinned;
}
