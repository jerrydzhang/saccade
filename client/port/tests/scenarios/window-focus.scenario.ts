// The human's new-window j/k (m-537/538): a window born by o showed as
// focused on niri while no key reached its webview. The shell builds
// suite windows with focused(false), and tao 0.37.2 creates the GTK
// window with accept_focus=false, restoring it only on the toplevel's
// first draw - which may never come. Read from GTK itself (the
// readout's window.focus, the shell's window_focus_state), never from
// focus events: a born window accepts keyboard focus and does not take
// focus merely by being mapped; the elsewhere act that opened it then
// requests focus for it (human ruling m-547 (3), below).
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

// Human ruling m-547 (3): opening through the link FOCUSES the receiver
// (t-14's "never asks for focus" no longer holds for o, i, w and the
// task row's Enter). The activation request is the platform's; GTK's
// own is-active answers whether the compositor granted it. A guard,
// not the proof: Weston activates a new window by itself (this passed
// before the request existed); see the next test.
test("o makes the born window the active one, and main no longer active", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const born = (await app.windows()).find((label) => label !== "main")!;
  await expect.poll(async () => (await app.readout(born)).window.focus?.active ?? null, { timeout: 10_000 }).toBe(true);
  await expect.poll(async () => (await app.readout("main")).window.focus?.active ?? null, { timeout: 10_000 }).toBe(false);
});

// The same ruling for a receiver that already exists: a compositor may
// activate a NEW window by itself (Weston does; the test above passed
// on Weston before the focus request existed), never an existing one,
// so this is the activation request's own proof. SKIPPED IN THIS LANE:
// GTK 3.24.52 activates a Wayland window only through xdg_activation_v1,
// and the lane's Weston 16 does not implement it (no libweston object
// names it), so no request can be granted here; it failed identically
// with and without the request. The human's niri implements it; their
// key trace (gtk.active on each window's focus line) is the proof until
// a lane compositor offers the protocol.
test.skip("o delivering into an existing receiver activates it over the active window", { tags: ["multi-window"] }, async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.citingChild));
  await citationAct(app, "o");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(2);
  const receiver = (await app.windows()).find((label) => label !== "main")!;
  await expect.poll(async () => (await app.readout("main")).window.link, { timeout: 20_000 }).toBe(receiver);
  // w: a third window, now the active one
  await citationAct(app, "w");
  await expect.poll(async () => (await app.windows()).length, { timeout: 20_000 }).toBe(3);
  const third = (await app.windows()).find((label) => label !== "main" && label !== receiver)!;
  await expect.poll(async () => (await app.readout(third)).window.focus?.active ?? null, { timeout: 10_000 }).toBe(true);
  // o from main: the link delivers into the receiver, which activates
  await citationAct(app, "o");
  await expect.poll(async () => (await app.readout(receiver)).window.focus?.active ?? null, { timeout: 10_000 }).toBe(true);
  expect((await app.readout(third)).window.focus?.active).toBe(false);
});
