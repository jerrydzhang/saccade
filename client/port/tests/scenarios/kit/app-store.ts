// The browser project's read into the app's OWN window store (t-71):
// the document the shell mock keeps in this origin's localStorage
// (kit/shell-mock.ts), read through the app's WindowStore - the same
// code the app runs. The tracker's window wire is gone; nothing here
// fetches.
import { WindowStore, WINDOW_STORE_KEY } from "../../../src/lib/window-store.ts";

/** A store view over the mock's localStorage document, fresh at every
 * call: the states in `live` answer attachedHere (the live windows the
 * scenario's shell world lists, when a read needs them). */
export function appWindowStore(live: readonly string[] = []): WindowStore {
  return new WindowStore({
    backend: {
      load: () => Promise.resolve(localStorage.getItem(WINDOW_STORE_KEY)),
      save: (text) => {
        localStorage.setItem(WINDOW_STORE_KEY, text);
        return Promise.resolve();
      },
    },
    liveStates: () => Promise.resolve(new Set(live)),
  });
}
