// The browser driver: the port's own index.html, served by this vite
// config, in a fresh iframe per test (a fresh window object, so no
// listener or state survives between tests), driven by vitest browser
// mode's real Playwright input in WebKit. There is no Tauri bridge
// here: the port starts empty and reaches the tracker through the
// host a typed location carries.
import { expect, test as base, inject } from "vitest";
import { commands, locators, page, userEvent, type Locator } from "vitest/browser";

import type { AppDriver, PortObservation, Tracker } from "./driver.ts";

declare module "vitest/browser" {
  interface BrowserCommands {
    trackerCommit(task: string, board: string, message: string): Promise<{ revision: number }>;
    trackerCreateTask(name: string): Promise<string>;
  }
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
  if (key.length > 1) return `{${key}}`;
  if (key === "{" || key === "[") return key + key;
  if (/^[A-Z]$/.test(key)) return `{Shift>}${key}{/Shift}`;
  return key;
}

async function launch(): Promise<{ app: AppDriver; close: () => void }> {
  const frame = document.createElement("iframe");
  frame.src = "/index.html";
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
    async activeMatch() {
      const element = portWindow().document.querySelector("[data-port-find-active]");
      return element?.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null;
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

export const test = base.extend<{ app: AppDriver; tracker: Tracker }>({
  // eslint-disable-next-line no-empty-pattern
  app: async ({}, use) => {
    const { app, close } = await launch();
    await use(app);
    close();
  },
  // eslint-disable-next-line no-empty-pattern
  tracker: async ({}, use) => {
    await use({
      ...inject("tracker"),
      createTask: (name) => commands.trackerCreateTask(name),
      commit: (task, board, message) => commands.trackerCommit(task, board, message),
    });
  },
});

export { expect };
