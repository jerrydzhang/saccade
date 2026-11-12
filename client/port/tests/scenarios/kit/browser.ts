// The browser driver: the port's own index.html, served by this vite
// config, in a fresh iframe per test (a fresh window object, so no
// listener or state survives between tests), driven by vitest browser
// mode's real Playwright input in WebKit. There is no Tauri bridge
// here: the port starts empty and reaches the tracker through the
// host a typed location carries.
import { expect, test as base, inject } from "vitest";
import { locators, page, userEvent, type Locator } from "vitest/browser";

import type { AppDriver, PortObservation, Tracker } from "./driver.ts";
import { SHELL_WORLD_KEY, type ShellCall, type ShellWindow, type ShellWorld } from "./shell-mock.ts";
import { trackerDoor } from "./tracker-wire.ts";

declare module "vitest/browser" {
  interface LocatorSelectors {
    getByPart(part: string): Locator;
  }
}

// content parts are addressed by their data-port-part mirror of the
// model's canonical content path
locators.extend({
  getByPart(part: string) {
    return `css=[data-port-part=${JSON.stringify(part)}]`;
  },
});

type PortWindow = Window & { __saccadePort?: { readout(): PortObservation } };

// user-event keyboard syntax: named keys in braces, the syntax's own
// brace and bracket characters doubled, and an uppercase letter typed
// as a person types it - with Shift held (the keymap binds N as
// shift+n)
function keySyntax(key: string): string {
  const chord = /^Ctrl-(.)$/.exec(key);
  if (chord) return `{Control>}${chord[1]}{/Control}`;
  if (key.length > 1) return `{${key}}`;
  if (key === "{" || key === "[") return key + key;
  if (/^[A-Z]$/.test(key)) return `{Shift>}${key}{/Shift}`;
  return key;
}

// `bound` answers windowBound: with no shell there is no binding to
// wait for; a bridged app reads its shell mock's own record
async function launch(
  src: string,
  bound: (label: string) => boolean = () => true,
): Promise<{ app: AppDriver; close: () => void }> {
  const frame = document.createElement("iframe");
  frame.src = src;
  frame.setAttribute("data-port-suite", "app");
  frame.style.cssText = "position:fixed;inset:0;width:100%;height:100%;border:0";
  document.body.append(frame);
  const portWindow = (): PortWindow => frame.contentWindow as PortWindow;
  // the app's own boot is the readiness: its readout door exists
  await expect.poll(() => typeof portWindow()?.__saccadePort?.readout, { timeout: 20_000 }).toBe("function");
  const inside = page.frameLocator(page.elementLocator(frame));
  // keyboard input lands in the focused frame: a real click on the
  // app's strip gives the app window focus before any key
  await inside.getByText("saccade (port)").click();
  const app: AppDriver = {
    async readout() {
      const door = portWindow().__saccadePort;
      if (!door) throw new Error("the port readout door is gone");
      return JSON.parse(JSON.stringify(door.readout())) as PortObservation;
    },
    async press(...keys) {
      for (const key of keys) await userEvent.keyboard(keySyntax(key));
    },
    async type(text) {
      for (const ch of text) await userEvent.keyboard(keySyntax(ch));
    },
    async click(control, options) {
      await inside.getByTestId(control).click({ button: options?.button ?? "left" });
    },
    async clickPart(part, options) {
      const element = portWindow().document.querySelector(`[data-port-part=${JSON.stringify(part)}]`);
      if (element === null) throw new Error(`the part ${part} is not rendered`);
      const box = element.getBoundingClientRect();
      const position = options?.at === "header"
        ? { x: box.width / 2, y: Math.min(6, box.height / 4) }
        : { x: box.width / 2, y: box.height / 2 };
      await inside.getByPart(part).click({ button: options?.button ?? "left", position });
    },
    async shows(control) {
      return portWindow().document.querySelector(`[data-port-control=${JSON.stringify(control)}]`) !== null;
    },
    async menuOpen() {
      return portWindow().document.querySelector("[data-port-menu]") !== null;
    },
    async labels() {
      return [...portWindow().document.querySelectorAll("[data-port-label]")].map((element) => ({
        label: element.getAttribute("data-port-label") ?? "",
        part: element.getAttribute("data-port-part") ?? "",
      }));
    },
    async strip() {
      const field = (name: string): string =>
        portWindow().document.querySelector(`[data-port-strip=${JSON.stringify(name)}]`)?.textContent?.trim() ?? "";
      return { location: field("location"), status: field("status"), link: field("link") };
    },
    async activeMatch() {
      const element = portWindow().document.querySelector("[data-port-find-active]");
      return element?.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null;
    },
    async windowBound(label = "main") {
      return bound(label);
    },
    async windows() {
      return ["main"];
    },
    async focusWindow(label) {
      if (label !== "main") throw new Error(`the browser has one window; no ${label}`);
    },
  };
  return { app, close: () => frame.remove() };
}

/** The mocked native shell a bridged app runs against (kit/shell-mock.ts). */
export type Shell = {
  /** Every shell command the app invoked, in order. */
  calls: ShellCall[];
  /** The calls of one command, their arguments in order. */
  argsOf(cmd: string): Record<string, unknown>[];
  /** The shell's live windows, this one first. */
  windows(): ShellWindow[];
  /** A window closes: the shell stops listing it. */
  close(label: string): void;
  /** This app's device id on the window store. */
  device: string;
};

export type BridgedOptions = {
  /** Other live windows besides "main": each is born and attached on
   * this device's window store, the way its own boot would do it. */
  others?: string[];
};

export const test = base.extend<{
  app: AppDriver;
  tracker: Tracker;
  bridged: (options?: BridgedOptions) => Promise<{ app: AppDriver; shell: Shell }>;
}>({
  // eslint-disable-next-line no-empty-pattern
  app: async ({}, use) => {
    const { app, close } = await launch("/index.html");
    await use(app);
    close();
  },
  // eslint-disable-next-line no-empty-pattern
  tracker: async ({}, use) => {
    const seeded = inject("tracker");
    await use({ ...seeded, ...trackerDoor(seeded.base) });
  },
  // the app with a mocked shell: one webview labelled main, a fresh
  // device id, and the other windows the scenario names
  bridged: async ({ tracker }, use) => {
    const closers: (() => void)[] = [];
    await use(async (options = {}) => {
      const device = `suite-device-${crypto.randomUUID()}`;
      // the app's device id (client/port/src/lib/windows.ts DEVICE_KEY),
      // fresh per test so no window state crosses tests
      localStorage.setItem("saccade.device.id", device);
      const windows: ShellWindow[] = [{ label: "main" }];
      for (const label of options.others ?? []) {
        const stateId = `suite-state-${crypto.randomUUID()}`;
        await tracker.attachedWindow(stateId, device);
        windows.push({ label, stateId });
      }
      const world: ShellWorld = { label: "main", windows, lastSession: [], testOpen: null, calls: [] };
      (window as unknown as Record<string, ShellWorld>)[SHELL_WORLD_KEY] = world;
      // the binding is done when the shell answered bind_window_state
      // for the window and its registry entry now carries that state
      // (what window_binding returns from then on); the native
      // stopgap reads the real shell's bind_window_state line
      const bound = (label: string): boolean =>
        world.calls.some((call) => call.cmd === "bind_window_state")
        && typeof world.windows.find((entry) => entry.label === label)?.stateId === "string";
      const { app, close } = await launch("/tests/scenarios/kit/shell-app.html", bound);
      closers.push(close);
      const shell: Shell = {
        calls: world.calls,
        argsOf: (cmd) => world.calls.filter((call) => call.cmd === cmd).map((call) => call.args),
        windows: () => world.windows.map((entry) => ({ ...entry })),
        close: (label) => {
          world.windows = world.windows.filter((entry) => entry.label !== label);
        },
        device,
      };
      return { app, shell };
    });
    for (const close of closers) close();
  },
});

export { expect };
