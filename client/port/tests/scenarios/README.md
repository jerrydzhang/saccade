# The port's scenarios

One runner (vitest), three projects (`client/port/vitest.config.ts`), and one
way to write a scenario. The Python harness in `client/harness/` stays the
authoritative gate until scenarios cut over; nothing here replaces it yet.

| project | what | command | runtime |
| --- | --- | --- | --- |
| unit | `tests/*.test.ts`, pure and component tests, in node | `just port-test` | about 1s |
| browser | `tests/scenarios/*.scenario.ts` and `*.browser.ts` in headless WebKit | `just port-test-browser` | about 10s |
| native | the scenarios tagged `native` or `multi-window`, in the built app, in the session sandbox | `just port-test-native` (gated, below) | about 8s in the sandbox |

`just test` runs the unit and browser projects after the cargo suite, and the
`port-unit` prek hook runs the unit project on every commit. One file or test
name: `just port-test-browser s14` (vitest's own filters pass through).

## Writing a scenario

Copy `s16-find.scenario.ts`. A scenario is a file `<name>.scenario.ts` here:

```ts
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

test("what the person sees happen", async ({ app, tracker }) => {
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await app.press("/");
  await expect.poll(async () => (await app.readout()).find.open).toBe(true);
});
```

The rules:

- Import `test` and `expect` from `#scenario`, never from `vitest`. Each
  project points that alias at its own driver, so the same file runs in
  WebKit and in the native app.
- `app` is a fresh app for this test alone: one window, no launch restore,
  empty trail. `tracker` is the throwaway seeded tracker (the t36 seeds:
  `tracker.tasks.cited`, `citing`, `citingChild`, `landing`). A scenario that
  commits creates its own task first (`tracker.createTask`, then
  `tracker.commit`), so the seeds stay as seeded for everyone else.
- Read state only through the readout: `app.readout()` returns
  `window.__saccadePort.readout()` (`src/lib/observation.ts`).
- Drive only through real input: `app.press`, `app.type`, `app.click`,
  `app.clickPart`. No test-only setters, no open door, no `evaluate`.
- No selectors in a scenario. Controls are named by their
  `data-port-control` id and content by its readout part path. A journey
  used twice belongs in `kit/helpers.ts`; a new kind of input or observation
  belongs in `kit/driver.ts`, implemented in both drivers.
- Wait with `expect.poll`, never a sleep loop. A
  hold, something staying unchanged, is the one place for a fixed wait;
  say so in a comment (see `s15-follow.scenario.ts`).
- Tag the test `{ tags: ["native"] }` when it should also run in the native
  app, or `{ tags: ["multi-window"] }` when it needs real windows. The browser
  project skips `multi-window`.
- A known defect is `test.fails("KNOWN DEFECT: ...")` with a comment naming
  the file and line. When the fix lands the test "fails"; drop `fails` then.
- If the readout lacks what a scenario needs, do not patch the app from
  here: record the missing field and skip or mark the test.
- A browser-only scenario that needs the mocked shell is `<name>.browser.ts`
  and imports `test` from `./kit/browser.ts` for its `bridged` fixture (see
  `elsewhere.browser.ts`); the native project never collects those files.

## The kit

Every module is a test-infrastructure module; each line says why it exists.

- `kit/driver.ts`: the one `AppDriver` interface both drivers implement, so
  scenarios and helpers are driver-blind.
- `kit/helpers.ts`: the shared journeys (open the picker, open a location,
  wait for it, find), written once against `AppDriver`.
- `kit/browser.ts`: the browser driver. The port's own `index.html`, served
  by vitest's vite server, in a fresh iframe per test (a fresh window, so no
  listener or state crosses tests); input is vitest's Playwright `userEvent`.
  Needed because vitest isolates per file, not per test, and the app's window
  listeners do not detach.
- `kit/native.ts`: the native driver. Per test, its own `tauri-driver` with
  fresh XDG dirs, and a WebdriverIO session against the built app.
- `kit/shell-mock.ts` (with `shell-app.html`, `shell-entry.ts`,
  `shell-install.ts`): the browser's mocked native shell. Tauri's official
  `@tauri-apps/api/mocks` (mockIPC, mockWindows) answering each shell command
  the port invokes, written from the Rust signature it cites. It models one
  webview plus the shell's window registry as data and never routes between
  pages; the entry page installs it inside the app's iframe before the port's
  `main.ts` boots, since the mocks patch their own realm. Needed so the app's
  half of multi-window (the link, the strip, the o/i/w decision, the shell
  calls) runs in the browser.
- `kit/tracker-wire.ts`: the fetch-only door into the running tracker
  (commits, window-state reads), usable from both realms.
- `kit/tracker.ts`, `kit/tracker-setup.ts`: the throwaway tracker, a real
  `sac serve` on a fresh temp db and repo on a free loopback port, seeded by
  `client/harness/t36_seeds.py` itself (one source for the seed boards until
  the harness cuts over). It is never the production tracker; the run root is
  printed and kept.

## Browser

WebKit is the nix-pinned Playwright WebKit (`PLAYWRIGHT_BROWSERS_PATH` from
the `saccade-app` devshell; npm `playwright` is pinned to the same 1.63.0, so
nothing is downloaded). It runs display-less on the devshell's pinned mesa
with software EGL, the browser lane's own recipe, so the recipe enters
`nix develop .#saccade-app`. There is no Tauri bridge in a browser, so the
port starts empty and reaches the tracker through the host a typed location
carries; the tracker allows exactly the vitest server's origin.

## Native

The native project runs the scenarios tagged `native` or `multi-window` in
the built port app, inside the session lane's enforced sandbox, through its
suite mode (`client/harness/entry.py session suite`, payload
`client/harness/suite_payload.py`): the containment canary, then the lane's
own private Weston, then `vitest --project native` in the read-only bound
`client/` tree. Each test's fixture (`kit/native.ts`) starts its own
`tauri-driver` with fresh XDG dirs under the run root, so every test gets a
fresh app: one window, no launch restore. WebdriverIO is the client, in
classic WebDriver mode (WebKitWebDriver has no BiDi).

```sh
SAC_APP_RUNTIME_AUTHORIZED=3030 SACCADE_DB=<the caller-side tracker db> just port-test-native
```

`SAC_APP_BIN` defaults to the nix-built port app (`.#saccade-app-port`; its
sources include `port/tests`, so a scenario edit rebuilds it, about 2
minutes); pass a built one to skip that. The run root is printed and kept:
`logs/suite.log` (the vitest output), `logs/weston.log`, and per test
`app-*/tauri-driver.log` (the app's own console and shell lines).

Inside the sandbox there is no `/etc/hosts` (the config binds vite to
`127.0.0.1`) and the client tree is read-only (the payload uses vitest's
runner config loader and points vite's cache at the run root).

Native holds real-window scenarios only: S18 (window birth, delivery landing
in the other webview, the link). To add later: no focus steal on window birth
(needs the compositor's focus fact), launch restore across a relaunch on the
same data dir (S20: a fixture that relaunches on one XDG root), and an `i`
delivery into an already-live receiver that has scrolled away.
