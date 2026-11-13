// The human's new-window j/k (m-537/538): a window born by o showed as
// focused on niri while no key reached its webview. The shell builds
// suite windows with focused(false), and tao 0.37.2 creates the GTK
// window with accept_focus=false, restoring it only on the toplevel's
// first draw - which may never come. Read from GTK itself (the
// readout's window.focus, the shell's window_focus_state), never from
// focus events: a born window accepts keyboard focus, and still does
// not take focus when mapped (a new window never steals focus).
import { expect, test } from "#scenario";
import { citationAct, openLocation, spec } from "./kit/helpers.ts";

test("a window born by o accepts keyboard focus and does not take it on map", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await expect.poll(async () => (await app.readout("main")).window.focus?.acceptsFocus ?? null).toBe(true);
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const born = (await app.windows()).find((label) => label !== "main")!;
  await expect.poll(async () => (await app.readout(born)).window.focus, { timeout: 10_000 }).not.toBeNull();
  // the restore-on-draw has had its chance by now; GTK's own answer
  await new Promise((resolve) => setTimeout(resolve, 1_500)); // a hold: the state must STAY right
  expect((await app.readout(born)).window.focus).toMatchObject({ acceptsFocus: true, focusOnMap: false });
});

// Trace 2 (m-540): on niri the born window was active in GTK and still
// dropped every key until a click inside it - the WebKit view was not
// the toplevel's focus widget. WebDriver's own keys go to the page
// directly, so a key press here cannot stand in for the compositor's
// activation: the focus-widget fact is asserted, read from GTK. Born
// and never clicked, the window's focus widget is its webview.
test("a window born by o has its webview as GTK focus widget, before any click", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const born = (await app.windows()).find((label) => label !== "main")!;
  await expect.poll(async () => (await app.readout(born)).window.focus?.focusWidget ?? null, { timeout: 10_000 }).toBe("WebKitWebView");
});
