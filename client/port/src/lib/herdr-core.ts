// The one module that names herdr-core: the rest of the app asks it
// for three things on a machine and never spells the core's own
// vocabulary - its verbs, its socket, its bridge commands - anywhere
// else. Reads ride the native shell's bridge, the local core's
// read-only face; the HTTP ask rides a fixed argv this module
// composes, so its callers never see the tool that carries it. Tests
// build an implementation that performs the asks directly.

import { tauriInvoke, type TauriInvoke } from "./sources.ts";
import { LANE_CODES, SourceUnavailableError, type LaneCode } from "./board/errors.ts";

export type Machine = string;

/** One finished command on a machine: its exit code (null when it
 * never reached one), what it wrote, and whether its bound cut it
 * short. */
export type CommandOutcome = {
  code: number | null;
  stdout: Uint8Array<ArrayBuffer>;
  stderr: Uint8Array<ArrayBuffer>;
  truncated: boolean;
  timedOut: boolean;
};

/** A loopback address on a machine - the port its declaration gives. */
export type LocalAddress = { host: string; port: number };

export interface MachineAsks {
  /** A file's text on the machine. */
  readFile(machine: Machine, path: string): Promise<string>;
  /** One fixed argv on the machine, no shell, bounded. */
  runCommand(machine: Machine, argv: readonly string[]): Promise<CommandOutcome>;
  /** One HTTP GET against a loopback address on the machine (the
   * project's own services answering there); the route starts with
   * "/". A machine that cannot carry the request says so by name. */
  requestLocal(machine: Machine, address: LocalAddress, route: string): Promise<Uint8Array<ArrayBuffer>>;
}

/** The bound one local request may take; the carrying command dies
 * before the exec budget under it. */
const LOCAL_REQUEST_BOUND_SECONDS = 20;

// The bridge rejects with its structured {code, message}; keep the
// code and the human message instead of flattening them into a
// string. An unknown code is a bridge bug, surfaced as one.
function bridgeRefusal(error: unknown): SourceUnavailableError {
  if (typeof error === "object" && error !== null) {
    const candidate = error as { code?: unknown; message?: unknown };
    if (typeof candidate.code === "string" && typeof candidate.message === "string") {
      const known = (LANE_CODES as readonly string[]).includes(candidate.code);
      const code = known ? (candidate.code as LaneCode) : "bridge-invalid";
      return new SourceUnavailableError(code, candidate.message);
    }
  }
  return new SourceUnavailableError("read-failed", String(error));
}

/** The asks answered through the native shell's bridge - the local
 * core. Only tests pass the invoke. */
export function coreAsks(invoke: TauriInvoke = tauriInvoke()): MachineAsks {
  const runCommand = async (machine: Machine, argv: readonly string[]): Promise<CommandOutcome> => {
    const result = (await invoke("exec_on", { on: machine, argv: [...argv] })) as {
      code: number | null;
      stdout: number[];
      stderr: number[];
      truncated: boolean;
      timed_out: boolean;
    };
    return {
      code: result.code,
      stdout: new Uint8Array(result.stdout),
      stderr: new Uint8Array(result.stderr),
      truncated: result.truncated,
      timedOut: result.timed_out,
    };
  };
  return {
    async readFile(machine, path) {
      const result = (await invoke("read_declaration", { path, on: machine })) as { text: string };
      return result.text;
    },
    runCommand,
    async requestLocal(machine, address, route) {
      if (!route.startsWith("/")) {
        throw new SourceUnavailableError("read-failed", `the route must start with '/': ${route}`);
      }
      const url = `http://${address.host}:${address.port}${route}`;
      const argv = [
        "curl",
        "-fsS",
        "--max-time",
        String(LOCAL_REQUEST_BOUND_SECONDS),
        url,
      ];
      let outcome: CommandOutcome;
      try {
        outcome = await runCommand(machine, argv);
      } catch (error) {
        // the carrying command could not even start: the ask names the
        // machine and keeps the core's own refusal code
        const refusal = bridgeRefusal(error);
        throw new SourceUnavailableError(
          refusal.code,
          `machine '${machine}' cannot carry the request to ${url}: ${refusal.message}`,
        );
      }
      if (outcome.code !== 0) {
        const detail = new TextDecoder().decode(outcome.stderr).trim();
        throw new SourceUnavailableError(
          outcome.timedOut ? "exec-timeout" : "machine-unavailable",
          `the request to ${url} on machine '${machine}' failed${detail === "" ? "" : `: ${detail}`}`,
        );
      }
      return outcome.stdout;
    },
  };
}

/** The asks answered directly, no core between: tests hand the three
 * workers and the implementation carries the same contract. */
export function directAsks(handlers: {
  readFile(machine: Machine, path: string): Promise<string>;
  runCommand(machine: Machine, argv: readonly string[]): Promise<CommandOutcome>;
  requestLocal(machine: Machine, address: LocalAddress, route: string): Promise<Uint8Array<ArrayBuffer>>;
}): MachineAsks {
  return { ...handlers };
}
