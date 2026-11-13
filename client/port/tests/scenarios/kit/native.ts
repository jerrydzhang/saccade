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
  NumpadAdd: Key.Add,
};

// shifted symbols ride explicit Shift chords on their US-layout base
// key: on this WebKitWebDriver stack a bare shifted codepoint loses
// its shift after a pointer action (the retired t36 lane recorded the
// evidence); uppercase letters likewise
const CHORDS: Record<string, string> = {
  "~": "`", "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8",
  "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\", ":": ";", "\"": "'",
  "<": ",", ">": ".", "?": "/",
};

type Browser = Awaited<ReturnType<typeof remote>>;

async function stroke(browser: Browser, key: string): Promise<void> {
  const named = NAMED[key];
  const shifted = CHORDS[key] ?? (/^[A-Z]$/.test(key) ? key.toLowerCase() : null);
  const action = browser.action("key");
  const chord = /^Ctrl-(.)$/.exec(key);
  const chordBase = chord ? CHORDS[chord[1]!] : undefined;
  if (chord && chordBase !== undefined) {
    // a shifted symbol under Ctrl holds Shift too (Ctrl-+ is Ctrl+Shift+=)
    await action.down(Key.Control).down(Key.Shift).down(chordBase).up(chordBase).up(Key.Shift).up(Key.Control).perform();
  } else if (chord) {
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

// a launch on `reuse` (an earlier launch's home) keeps that app's data
// directory: the relaunch journey (S20)
async function launch(runRoot: string, reuse?: string): Promise<{ app: AppDriver; close: () => Promise<void>; home: string }> {
  const env = pinned();
  const home = reuse ?? mkdtempSync(join(runRoot, "app-"));
  const xdg: Record<string, string> = {};
  for (const [name, dir] of [["XDG_DATA_HOME", "data"], ["XDG_CONFIG_HOME", "config"], ["XDG_CACHE_HOME", "cache"], ["XDG_STATE_HOME", "state"]] as const) {
    mkdirSync(join(home, dir), { recursive: true });
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
      // WebDriver takes integer offsets: a fractional height made the
      // header offset fractional and the click refused
      const y = options?.at === "header" ? Math.round(-height / 2 + Math.min(6, height / 4)) : 0;
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
    async partBox(part) {
      return browser.execute((path: string) => {
        const element = document.querySelector(`[data-port-part=${JSON.stringify(path)}]`);
        if (element === null) return null;
        const rect = element.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return null;
        return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
      }, part);
    },
    async menuActs() {
      return browser.execute(() =>
        [...document.querySelectorAll("[data-port-menu-act]")].map((element) => element.getAttribute("data-port-menu-act") ?? ""));
    },
    async clickMenuAct(key) {
      await browser.$(`[data-port-menu-act=${JSON.stringify(key)}]`).click();
    },
    async relaunch() {
      throw new Error("relaunch is the fixture's (kit/native.ts test)");
    },
    async pointAt(part) {
      const box = await browser.execute((path: string) => {
        const rect = document.querySelector(`[data-port-part=${JSON.stringify(path)}]`)!.getBoundingClientRect();
        return { x: Math.round(rect.left + rect.width / 2), y: Math.round(rect.top + rect.height / 2) };
      }, part);
      await browser.action("pointer").move({ x: box.x, y: box.y, origin: "viewport" }).perform();
    },
    async partLook(part) {
      return browser.execute((path: string) => {
        const element = document.querySelector(`[data-port-part=${JSON.stringify(path)}]`);
        if (element === null) return null;
        const style = getComputedStyle(element);
        return { background: style.backgroundColor, color: style.color };
      }, part);
    },
    async drag(control, dx, dy) {
      const box = await browser.execute((id: string) => {
        const rect = document.querySelector(`[data-port-control=${JSON.stringify(id)}]`)!.getBoundingClientRect();
        return { x: Math.round(rect.left + rect.width / 2), y: Math.round(rect.top + rect.height / 2) };
      }, control);
      await browser
        .action("pointer")
        .move({ x: box.x, y: box.y, origin: "viewport" })
        .down()
        .move({ x: box.x + dx, y: box.y + dy, origin: "viewport", duration: 100 })
        .up()
        .perform();
    },
    async wheel() {
      throw new Error("the native stack delivers no wheel events (WebKitWebDriver): wheel scenarios are browser-only");
    },
    async controlBox(control) {
      return browser.execute((id: string) => {
        const element = document.querySelector(`[data-port-control=${JSON.stringify(id)}]`);
        if (element === null) return null;
        const rect = element.getBoundingClientRect();
        return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
      }, control);
    },
    async controlText(control) {
      return browser.execute((id: string) => {
        const element = document.querySelector(`[data-port-control=${JSON.stringify(id)}]`) as HTMLElement | null;
        return element === null ? null : element.innerText;
      }, control);
    },
    async menuGeometry() {
      return browser.execute(() => {
      const menu = document.querySelector("[data-port-menu]") as HTMLElement | null;
        if (menu === null) return null;
        const box = (element: Element) => {
          const rect = element.getBoundingClientRect();
          return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
        };
        const part = menu.getAttribute("data-port-menu-part") ?? "";
        const target = part === "" ? null : document.querySelector(`.port-board-view [data-port-part=${JSON.stringify(part)}], .port-tasks-stage [data-port-part=${JSON.stringify(part)}]`);
        return { menu: box(menu), target: target === null ? null : box(target), part, viewport: { width: window.innerWidth, height: window.innerHeight } };
      });
    },
    async partText(part) {
      return browser.execute((path: string) => {
        const element = document.querySelector(`[data-port-part=${JSON.stringify(path)}]`);
        return element instanceof HTMLElement ? element.innerText : null;
      }, part);
    },
    async findMarks() {
      return browser.execute(() => {
      const out: { part: string | null; text: string; active: boolean }[] = [];
        for (const mark of document.querySelectorAll("[data-port-find-match]")) {
          const part = mark.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null;
          out.push({ part, text: mark.textContent ?? "", active: mark.hasAttribute("data-port-find-active") });
        }
        return out;
      });
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
    .toBe("saccade-app");
  return {
    app,
    home,
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
    const root = runRoot();
    let current = await launch(root);
    // the test holds a delegate, so relaunch can swap the session (its
    // own driver, stopped by its own handle) under the same app object
    const app = new Proxy({} as AppDriver, {
      get(_target, key: string) {
        if (key === "relaunch") {
          return async () => {
            const home = current.home;
            await current.close();
            current = await launch(root, home);
          };
        }
        return (current.app as unknown as Record<string, unknown>)[key];
      },
    });
    try {
      await use(app);
    } finally {
      await current.close();
    }
  },
  // eslint-disable-next-line no-empty-pattern
  tracker: async ({}, use) => {
    const seeded = inject("tracker");
    await use({ ...seeded, ...trackerDoor(seeded.base) });
  },
});

export { expect };
