// The shell feasibility probe (t-19): test-only machinery, named as
// such, never presented as human functionality. When the shell probe
// is active (the native lane sets SAC_APP_SHELL_PROBE and the
// shell's probe module created the probe-second window), each window
// writes its probe key into localStorage, reads both probe keys, and
// reports what it sees through the bridge; the lane's app log answers
// whether the windows share one origin-scoped store. A human run has
// no probe window and the main window writes nothing (the active
// check asks the bridge first).

type TauriInvoke = (cmd: string, args?: unknown) => Promise<unknown>;

type TauriInternals = {
  invoke: TauriInvoke;
  metadata?: { currentWindow?: { label?: string } };
};

function probeWindowLabel(): string | null {
  const w = window as unknown as { __TAURI_INTERNALS__?: TauriInternals };
  return w.__TAURI_INTERNALS__?.metadata?.currentWindow?.label ?? null;
}

function readKey(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function report(label: string, invoke: TauriInvoke): void {
  const payload = {
    report: {
      label,
      sees_main: readKey("probe.main"),
      sees_second: readKey("probe.second"),
    },
  };
  void invoke("shell_probe_report", payload).catch(() => {
    // the probe never disturbs the app: a failed report is silent
  });
}

export function startShellProbe(): void {
  const w = window as unknown as { __TAURI_INTERNALS__?: TauriInternals };
  const tauri = w.__TAURI_INTERNALS__;
  if (!tauri) return;
  const label = probeWindowLabel();
  if (label === null) return;

  // the probe window exists only when the shell probe is active
  if (label === "probe-second") {
    window.setTimeout(() => {
      try {
        localStorage.setItem("probe.second", "written-by-second");
      } catch {
        // report what is readable anyway
      }
      report(label, tauri.invoke);
    }, 2000);
    return;
  }

  // the main window probes only under an active shell probe
  void (async () => {
    let active = false;
    try {
      active = (await tauri.invoke("shell_probe_active")) === true;
    } catch {
      active = false;
    }
    if (!active) return;
    try {
      localStorage.setItem("probe.main", "written-by-main");
    } catch {
      // report what is readable anyway
    }
    // the probe window spawns at 6s and writes at ~8s; 14s sees it
    window.setTimeout(() => report(label, tauri.invoke), 14000);
  })();
}
