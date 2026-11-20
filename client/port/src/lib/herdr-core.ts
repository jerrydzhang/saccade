// The one module that names herdr-core: the rest of the app asks it
// for three things on a machine and never spells the core's own
// vocabulary - its verbs, its socket, its bridge commands - anywhere
// else. Reads ride the native shell's bridge, the local core's
// read-only face; the HTTP ask names a machine, a loopback port and a
// route, and the shell composes and bounds the carrying argv itself -
// its callers never see the tool that carries it. Tests build an
// implementation that answers the asks directly.

import { tauriInvoke, type TauriInvoke } from "./sources.ts";
import { LANE_CODES, RequestRefused, SourceUnavailableError, type LaneCode } from "./board/errors.ts";

export type Machine = string;

/** A loopback port on a machine - the port its declaration gives. The
 * host is not a parameter anywhere: the request always binds
 * 127.0.0.1. */
export type LocalPort = number;

export interface MachineAsks {
  /** A file's text on the machine. */
  readFile(machine: Machine, path: string): Promise<string>;
  /** A machine read's bytes: an asset beside the declaration the read
   * names. */
  readAsset(machine: Machine, base: string, relative: string): Promise<Uint8Array<ArrayBuffer>>;
  /** One HTTP GET against a loopback port on the machine (the
   * project's own services answering there); the route starts with
   * "/". A machine that cannot carry the request says so by name. */
  requestLocal(machine: Machine, port: LocalPort, route: string): Promise<Uint8Array<ArrayBuffer>>;
}

// The bridge rejects with its structured {code, message}; keep the
// code and the human message instead of flattening them into a string.
// An unknown code is a bridge bug, surfaced as one.
function bridgeRefusal(error: unknown): SourceUnavailableError {
  if (typeof error === "object" && error !== null) {
    const candidate = error as { code?: unknown; message?: unknown; body?: unknown };
    if (typeof candidate.code === "string" && typeof candidate.message === "string") {
      const known = (LANE_CODES as readonly string[]).includes(candidate.code);
      const code = known ? (candidate.code as LaneCode) : "bridge-invalid";
      return typeof candidate.body === "string"
        ? new RequestRefused(code, candidate.message, candidate.body)
        : new SourceUnavailableError(code, candidate.message);
    }
  }
  return new SourceUnavailableError("read-failed", String(error));
}

/** The asks answered through the native shell's bridge - the local
 * core. Only tests pass the invoke. The machine named "host" is the
 * shell's own test-only file reader: it rides no core, so the bridge's
 * `on` stays unset for it. The window's one bridge-backed instance is
 * held here: the registry holds its asks, and a fresh instance every
 * ask would defeat the registry's per-session caches. Tests pass
 * their own invoke and get their own instance. */
let bridgeAsks: MachineAsks | null = null;

export function coreAsks(invoke?: TauriInvoke): MachineAsks {
  if (invoke !== undefined) return buildCoreAsks(invoke);
  if (bridgeAsks === null) bridgeAsks = buildCoreAsks(tauriInvoke());
  return bridgeAsks;
}

function buildCoreAsks(invoke: TauriInvoke): MachineAsks {
  const on = (machine: Machine): Record<string, unknown> =>
    machine === "host" ? {} : { on: machine };
  return {
    async readFile(machine, path) {
      const result = (await invoke("read_declaration", { path, ...on(machine) })) as { text: string };
      return result.text;
    },
    async readAsset(machine, base, relative) {
      const result = (await invoke("read_asset", { base, relative, ...on(machine) })) as {
        bytes: number[];
      };
      return new Uint8Array(result.bytes);
    },
    async requestLocal(machine, port, route) {
      if (!route.startsWith("/")) {
        throw new SourceUnavailableError("read-failed", `the route must start with '/': ${route}`);
      }
      try {
        const bytes = (await invoke("local_request", { on: machine, port, route })) as number[];
        return new Uint8Array(bytes);
      } catch (error) {
        throw bridgeRefusal(error);
      }
    },
  };
}

/** The asks answered directly, no core between: tests hand the three
 * workers and the implementation carries the same contract. */
export function directAsks(handlers: {
  readFile(machine: Machine, path: string): Promise<string>;
  readAsset(machine: Machine, base: string, relative: string): Promise<Uint8Array<ArrayBuffer>>;
  requestLocal(machine: Machine, port: LocalPort, route: string): Promise<Uint8Array<ArrayBuffer>>;
}): MachineAsks {
  return { ...handlers };
}
