// The launch restore's fresh start (t-71): the shell's memory names
// the states open at the last quit, and a memory id the app's own
// store does not know (a tracker-era id an old open-windows.json
// carries) is skipped, not reopened - the ratified first start after
// the change opens fresh windows. Against the mocked shell; the
// memory's happy path over real relaunches is the native project's
// (S20).
import { expect, test } from "./kit/browser.ts";
import { appWindowStore } from "./kit/app-store.ts";

test("a launch whose last_session names a state the store does not know opens fresh", async ({ bridged }) => {
  const { app, shell } = await bridged({ lastSession: ["tracker-era-id"] });
  // main bound a state of its own, never the unknown one
  const bound = shell.argsOf("bind_window_state").at(-1)?.stateId;
  expect(typeof bound).toBe("string");
  expect(bound).not.toBe("tracker-era-id");
  // the bound state exists in the app's store; the unknown id does not
  const store = appWindowStore([String(bound)]);
  await expect(store.getState(String(bound))).resolves.toMatchObject({ id: bound, location: null });
  await expect(store.getState("tracker-era-id")).rejects.toThrow(/holds no state tracker-era-id/);
  // no ghost window: the unknown id opened nothing
  expect(shell.argsOf("open_suite_window")).toEqual([]);
  expect(shell.windows().map((entry) => entry.label)).toEqual(["main"]);
  // no crash: the window is alive, bound, and opens fresh - the
  // control plane, never a blank page
  await expect.poll(async () => (await app.readout()).window).toMatchObject({ label: "main", bound: true });
  await expect.poll(async () => (await app.readout()).surface).toBe("tasks");
});
