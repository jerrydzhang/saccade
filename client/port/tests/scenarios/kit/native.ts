// The native driver: the built port binary through tauri-driver
// (--native-driver WebKitWebDriver) with WebdriverIO as the WebDriver
// client. Each test gets a FRESH app: its own tauri-driver process
// with fresh XDG data/config/cache/state dirs under the run root (no
// launch restore, one window, empty trail), stopped by its own handle
// after the test. It must run where tauri-driver can reach a display:
// inside the session lane's sandbox (client/harness/entry.py
// session), which supplies the pinned binaries and the compositor
// through the env names its payload already reads. Outside that
// sandbox it refuses; it never reaches for the person's display.
import { spawn, type ChildProcess } from "node:child_process";
import { mkdirSync, mkdtempSync, openSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test as base, inject } from "vitest";
import { Key, remote } from "webdriverio";

import type { AppDriver, PortObservation, Tracker } from "./driver.ts";
import { freePort } from "./tracker.ts";
import { pinned } from "./native-env.ts";
import { trackerDoor } from "./tracker-wire.ts";

const NAMED: Record<string, string> = {
  Escape: Key.Escape,
  Enter: Key.Enter,
  Backspace: Key.Backspace,
  Tab: Key.Tab,
  ArrowDown: Key.ArrowDown,
  ArrowUp: Key.ArrowUp,
  ArrowLeft: Key.ArrowLeft,
  ArrowRight: Key.ArrowRight,
};

// shifted symbols ride explicit Shift chords on their US-layout base
// key: on this WebKitWebDriver stack a bare shifted codepoint loses
// its shift after a pointer action (client/harness/t36_port.py
// type_text records the evidence); uppercase letters likewise
const CHORDS: Record<string, string> = { ":": ";", "@": "2" };

type Browser = Awaited<ReturnType<typeof remote>>;

async function stroke(browser: Browser, key: string): Promise<void> {
  const named = NAMED[key];
  const shifted = CHORDS[key] ?? (/^[A-Z]$/.test(key) ? key.toLowerCase() : null);
  const action = browser.action("key");
  const chord = /^Ctrl-(.)$/.exec(key);
  if (chord) {
    await action.down(Key.Control).down(chord[1]!).up(chord[1]!).up(Key.Control).perform();
  } else if (named !== undefined) {
    await action.down(named).up(named).perform();
  } else if (shifted !== null) {
    await action.down(Key.Shift).down(shifted).up(shifted).up(Key.Shift).perform();
  } else {
    await action.down(key).up(key).perform();
  }
}

async function waitListening(port: number, driver: ChildProcess, log: string): Promise<void> {
  await expect
    .poll(
      async () => {
        if (driver.exitCode !== null) throw new Error(`tauri-driver exited (log: ${log})`);
        try {
          return (await fetch(`http://127.0.0.1:${port}/status`)).ok;
        } catch {
          return false;
        }
      },
      { timeout: 20_000, interval: 250 },
    )
    .toBe(true);
}

async function launch(runRoot: string): Promise<{ app: AppDriver; close: () => Promise<void> }> {
  const env = pinned();
  const home = mkdtempSync(join(runRoot, "app-"));
  const xdg: Record<string, string> = {};
  for (const [name, dir] of [["XDG_DATA_HOME", "data"], ["XDG_CONFIG_HOME", "config"], ["XDG_CACHE_HOME", "cache"], ["XDG_STATE_HOME", "state"]] as const) {
    mkdirSync(join(home, dir));
    xdg[name] = join(home, dir);
  }
  const port = await freePort();
  const nativePort = await freePort();
  const log = join(home, "tauri-driver.log");
  const out = openSync(log, "a");
  // the app is spawned by WebKitWebDriver: its display and fresh data
  // dirs ride the chain env (launcher inheritance)
  const driver = spawn(
    env.TAURI_DRIVER_BIN,
    ["--port", String(port), "--native-port", String(nativePort), "--native-host", "127.0.0.1", "--native-driver", env.WEBKIT_DRIVER_BIN],
    { env: { ...process.env, ...xdg }, stdio: ["ignore", out, out] },
  );
  const stopDriver = async (): Promise<void> => {
    if (driver.exitCode !== null || driver.signalCode !== null) return;
    const exited = new Promise((resolve) => driver.once("exit", resolve));
    driver.kill("SIGTERM");
    const timer = setTimeout(() => driver.kill("SIGKILL"), 5_000);
    await exited;
    clearTimeout(timer);
  };
  let browser: Browser;
  try {
    await waitListening(port, driver, log);
    browser = await remote({
      hostname: "127.0.0.1",
      port,
      logLevel: "warn",
      capabilities: {
        browserName: "wry",
        "tauri:options": { application: env.APP_BIN, args: [] },
        // WebKitWebDriver speaks classic WebDriver only: no BiDi socket
        "wdio:enforceWebDriverClassic": true,
      } as WebdriverIO.Capabilities,
    });
  } catch (error) {
    await stopDriver();
    throw error;
  }
  const readoutHere = async (): Promise<PortObservation> => {
    const text = await browser.execute(() =>
      JSON.stringify((window as unknown as { __saccadePort: { readout(): unknown } }).__saccadePort.readout()));
    return JSON.parse(text) as PortObservation;
  };
  const handles = async (): Promise<Map<string, string>> => {
    const original = await browser.getWindowHandle();
    const labels = new Map<string, string>();
    try {
      for (const handle of await browser.getWindowHandles()) {
        await browser.switchToWindow(handle);
        const label = (await readoutHere().catch(() => null))?.window.label;
        if (typeof label === "string") labels.set(label, handle);
      }
    } finally {
      await browser.switchToWindow(original);
    }
    return labels;
  };
  const control = (id: string) => browser.$(`[data-port-control=${JSON.stringify(id)}]`);
  const app: AppDriver = {
    async readout(window) {
      if (window === undefined) return readoutHere();
      const handle = (await handles()).get(window);
      if (handle === undefined) throw new Error(`no window carries the label ${window}`);
      const original = await browser.getWindowHandle();
      await browser.switchToWindow(handle);
      try {
        return await readoutHere();
      } finally {
        await browser.switchToWindow(original);
      }
    },
    async press(...keys) {
      for (const key of keys) await stroke(browser, key);
    },
    async type(text) {
      for (const ch of text) await stroke(browser, ch);
    },
    async click(id, options) {
      await control(id).click({ button: options?.button ?? "left" });
    },
    async clickPart(part, options) {
      const element = browser.$(`[data-port-part=${JSON.stringify(part)}]`);
      // WebdriverIO's click offsets are from the element's center
      const { height } = await element.getSize();
      const y = options?.at === "header" ? -height / 2 + Math.min(6, height / 4) : 0;
      await element.click({ button: options?.button ?? "left", x: 0, y });
    },
    async shows(id) {
      return control(id).isExisting();
    },
    async menuOpen() {
      return browser.$("[data-port-menu]").isExisting();
    },
    async labels() {
      return browser.execute(() =>
        [...document.querySelectorAll("[data-port-label]")].map((element) => ({
          label: element.getAttribute("data-port-label") ?? "",
          part: element.getAttribute("data-port-part") ?? "",
        })));
    },
    async strip() {
      const field = async (name: string): Promise<string> => {
        const element = browser.$(`[data-port-strip=${JSON.stringify(name)}]`);
        return (await element.isExisting()) ? (await element.getText()).trim() : "";
      };
      return { location: await field("location"), status: await field("status"), link: await field("link") };
    },
    async activeMatch() {
      return browser.execute(() =>
        document.querySelector("[data-port-find-active]")?.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null);
    },
    async windows() {
      const labels = [...(await handles()).keys()];
      return labels.sort((a, b) => (a === "main" ? -1 : b === "main" ? 1 : a.localeCompare(b)));
    },
    async focusWindow(label) {
      // the DRIVER's context: input and the default readout follow it;
      // the native window focus stays the compositor's
      const handle = (await handles()).get(label);
      if (handle === undefined) throw new Error(`no window carries the label ${label}`);
      await browser.switchToWindow(handle);
    },
  };
  // the app's own boot is the readiness: its readout door answers
  await expect
    .poll(async () => (await readoutHere().catch(() => null))?.app.name ?? null, { timeout: 30_000 })
    .toBe("saccade-app-port");
  return {
    app,
    close: async () => {
      try {
        await browser.deleteSession();
      } finally {
        await stopDriver();
      }
    },
  };
}

// the lane's run root when inside the sandbox (SHOTS_DIR is its
// shots/ directory), else a fresh temp root; kept after the run
const runRoot = (): string =>
  process.env.SHOTS_DIR ? join(process.env.SHOTS_DIR, "..") : mkdtempSync(join(tmpdir(), "saccade-port-native-"));

export const test = base.extend<{ app: AppDriver; tracker: Tracker }>({
  // eslint-disable-next-line no-empty-pattern
  app: async ({}, use) => {
    const { app, close } = await launch(runRoot());
    try {
      await use(app);
    } finally {
      await close();
    }
  },
  // eslint-disable-next-line no-empty-pattern
  tracker: async ({}, use) => {
    const seeded = inject("tracker");
    await use({ ...seeded, ...trackerDoor(seeded.base) });
  },
});

export { expect };
