// The key-trace diagnostic (SACCADE_KEY_TRACE): one line per keydown
// on the launching terminal's stderr, through the shell's key_trace
// command. It exists to see a key failure on the person's own machine
// that no lane reproduces: what arrived, where it was routed, what the
// matcher answered, and what the scroller did - at once, a frame
// later, and 150ms later (a scroll that lands and is taken back shows
// as a difference between the three).
//
// Off unless the shell says the variable is set: nothing is listened
// to, measured or sent. It only reads; it never changes routing.

import { tauriInvoke } from "../sources.ts";

export interface KeyTraceRecord {
  /** Fields in arrival order; the line is `name=value` pairs. */
  set(name: string, value: string | number | boolean | null): void;
  /** Close the record: the scroller is measured now, a frame later
   * and 150ms later, then the line is sent. */
  end(): void;
}

export interface KeyTrace {
  /** A record for this keydown, or null while the trace is off. */
  begin(event: KeyboardEvent): KeyTraceRecord | null;
}

type Probe = Readonly<{
  /** The window's scroller, read at call time. */
  scroller: () => HTMLElement | null;
  /** Window facts worth a field on every line. */
  facts: () => Record<string, string | number | boolean | null>;
}>;

function describe(node: Element | EventTarget | null): string {
  if (!(node instanceof Element)) return node === null ? "null" : "non-element";
  const control = node.getAttribute("data-port-control");
  const part = node.getAttribute("data-port-part");
  let out = node.tagName;
  if (node.classList.length > 0) out += `.${[...node.classList].join(".")}`;
  if (control !== null) out += `[control=${control}]`;
  if (part !== null) out += `[part=${part}]`;
  return out;
}

function metrics(element: HTMLElement | null): string {
  if (element === null) return "none";
  return `${element.scrollTop}/${element.scrollHeight}/${element.clientHeight}`;
}

function field(value: string | number | boolean | null): string {
  if (typeof value === "boolean") return value ? "1" : "0";
  if (typeof value === "string") return /[\s="]/.test(value) || value === "" ? JSON.stringify(value) : value;
  return String(value);
}

export function createKeyTrace(probe: Probe): KeyTrace {
  let enabled = false;
  let sequence = 0;
  let queue: string[] = [];
  let flushing = false;

  function send(line: string): void {
    queue.push(line);
    if (flushing) return;
    flushing = true;
    queueMicrotask(() => {
      const lines = queue;
      queue = [];
      flushing = false;
      try {
        void tauriInvoke()("key_trace", { lines }).catch(() => {});
      } catch {
        // the bridge left: the trace has nowhere to write
      }
    });
  }

  function line(kind: string, fields: Record<string, string | number | boolean | null>): string {
    const pairs = Object.entries(fields).map(([name, value]) => `${name}=${field(value)}`);
    return `${kind} #${++sequence} t=${Math.round(performance.now())} ${pairs.join(" ")}`;
  }

  function windowFacts(): Record<string, string | number | boolean | null> {
    const doc = document.scrollingElement as HTMLElement | null;
    return {
      focus: document.hasFocus(),
      vis: document.visibilityState,
      active: describe(document.activeElement),
      inner: `${window.innerWidth}x${window.innerHeight}`,
      dpr: window.devicePixelRatio,
      doc: metrics(doc),
    };
  }

  function install(): void {
    const element = probe.scroller();
    const style = element === null ? null : getComputedStyle(element);
    send(
      line("init", {
        ua: navigator.userAgent,
        ...windowFacts(),
        scroller: describe(element),
        connected: element?.isConnected ?? false,
        "s(top/height/client)": metrics(element),
        overflowY: style?.overflowY ?? null,
        scrollBehavior: style?.scrollBehavior ?? null,
        ...probe.facts(),
      }),
    );
    // what the window itself goes through between keys: focus, and
    // every scroll of the scroller whoever caused it
    for (const name of ["focus", "blur"] as const) {
      window.addEventListener(name, () => send(line(name, windowFacts())));
    }
    document.addEventListener("visibilitychange", () => send(line("visibility", windowFacts())));
    document.addEventListener(
      "scroll",
      (event) => {
        const target = event.target instanceof HTMLElement ? event.target : (document.scrollingElement as HTMLElement | null);
        send(line("scroll", { on: describe(target), "s(top/height/client)": metrics(target), isScroller: target === probe.scroller() }));
      },
      { capture: true, passive: true },
    );
    // every keydown as the window first sees it (capture): a key that
    // shows here and has no `key` line below it never reached the
    // port's handler
    window.addEventListener(
      "keydown",
      (event) => {
        send(line("seen", { key: event.key, code: event.code, target: describe(event.target), prevented: event.defaultPrevented }));
      },
      { capture: true },
    );
  }

  try {
    void tauriInvoke()("key_trace", { lines: [] })
      .then((on) => {
        if (on !== true) return;
        enabled = true;
        install();
      })
      .catch(() => {});
  } catch {
    // no bridge (a browser): the trace stays off
  }

  return {
    begin(event) {
      if (!enabled) return null;
      const fields: Record<string, string | number | boolean | null> = {
        key: event.key,
        code: event.code,
        mods: `${event.ctrlKey ? "C" : "-"}${event.shiftKey ? "S" : "-"}${event.altKey ? "A" : "-"}${event.metaKey ? "M" : "-"}`,
        repeat: event.repeat,
        composing: event.isComposing,
        prevented: event.defaultPrevented,
        target: describe(event.target),
        ...windowFacts(),
        ...probe.facts(),
      };
      const scroller = probe.scroller();
      fields["scroller"] = describe(scroller);
      fields["connected"] = scroller?.isConnected ?? false;
      const before = metrics(scroller);
      let ended = false;
      return {
        set(name, value) {
          fields[name] = value;
        },
        end() {
          if (ended) return;
          ended = true;
          const sync = metrics(probe.scroller());
          let frame = "never";
          requestAnimationFrame(() => {
            frame = metrics(probe.scroller());
          });
          setTimeout(() => {
            send(
              line("key", {
                ...fields,
                "s(top/height/client)": before,
                sync,
                frame,
                later: metrics(probe.scroller()),
                preventedAfter: event.defaultPrevented,
              }),
            );
          }, 150);
        },
      };
    },
  };
}
