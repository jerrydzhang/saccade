// The one core module's contract: the asks' shapes, the bridge
// mapping behind them, and the direct implementation tests ride -
// which performs the asks itself, no core between.

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

import { test } from "vitest";

import { coreAsks, directAsks, type CommandOutcome } from "../src/lib/herdr-core.ts";
import { SourceUnavailableError } from "../src/lib/board/errors.ts";

test("a read lands on the bridge's read with the machine riding", async () => {
  const calls: Array<{ cmd: string; args: Record<string, unknown> }> = [];
  const asks = coreAsks(async (cmd, args) => {
    calls.push({ cmd, args: args ?? {} });
    return { text: "the declaration" };
  });
  const text = await asks.readFile("rohan", "/srv/workbench.toml");
  assert.equal(text, "the declaration");
  assert.deepEqual(calls, [
    { cmd: "read_declaration", args: { path: "/srv/workbench.toml", on: "rohan" } },
  ]);
});

test("a run lands on the bridge's exec and its bytes come back bytes", async () => {
  const calls: Array<{ cmd: string; args: Record<string, unknown> }> = [];
  const asks = coreAsks(async (cmd, args) => {
    calls.push({ cmd, args: args ?? {} });
    return {
      code: 0,
      stdout: Array.from(new TextEncoder().encode("out")),
      stderr: Array.from(new TextEncoder().encode("err")),
      truncated: true,
      timed_out: false,
    };
  });
  const outcome = await asks.runCommand("rohan", ["realpath", "-e", "--", "/b", "/b/c"]);
  assert.deepEqual(calls, [
    {
      cmd: "exec_on",
      args: { on: "rohan", argv: ["realpath", "-e", "--", "/b", "/b/c"] },
    },
  ]);
  assert.equal(outcome.code, 0);
  assert.equal(new TextDecoder().decode(outcome.stdout), "out");
  assert.equal(new TextDecoder().decode(outcome.stderr), "err");
  assert.equal(outcome.truncated, true);
  assert.equal(outcome.timedOut, false);
});

test("a local request composes its carrying argv inside the module", async () => {
  const calls: Array<{ cmd: string; args: Record<string, unknown> }> = [];
  const asks = coreAsks(async (cmd, args) => {
    calls.push({ cmd, args: args ?? {} });
    return { code: 0, stdout: [104, 105], stderr: [], truncated: false, timed_out: false };
  });
  const body = await asks.requestLocal("rohan", { host: "127.0.0.1", port: 8811 }, "/view/v1");
  assert.equal(new TextDecoder().decode(body), "hi");
  assert.deepEqual(calls, [
    {
      cmd: "exec_on",
      args: {
        on: "rohan",
        argv: ["curl", "-fsS", "--max-time", "20", "http://127.0.0.1:8811/view/v1"],
      },
    },
  ]);
});

test("a failed local request names the machine and the address", async () => {
  const asks = coreAsks(async () => ({
    code: 7,
    stdout: [],
    stderr: Array.from(new TextEncoder().encode("connection refused")),
    truncated: false,
    timed_out: false,
  }));
  await assert.rejects(
    asks.requestLocal("rohan", { host: "127.0.0.1", port: 8811 }, "/view/v1"),
    (error: unknown) => {
      assert.ok(error instanceof SourceUnavailableError);
      assert.equal(error.code, "machine-unavailable");
      assert.match(error.message, /rohan/);
      assert.match(error.message, /127\.0\.0\.1:8811/);
      assert.match(error.message, /connection refused/);
      return true;
    },
  );
});

test("a timed local request carries its own code", async () => {
  const asks = coreAsks(async () => ({
    code: 28,
    stdout: [],
    stderr: [],
    truncated: false,
    timed_out: true,
  }));
  await assert.rejects(
    asks.requestLocal("sif", { host: "127.0.0.1", port: 9 }, "/"),
    (error: unknown) => {
      assert.ok(error instanceof SourceUnavailableError);
      assert.equal(error.code, "exec-timeout");
      assert.match(error.message, /sif/);
      return true;
    },
  );
});

test("a machine that cannot carry the request keeps the core's refusal code", async () => {
  const asks = coreAsks(async () => {
    throw { code: "exec-spawn-failed", message: "the core could not start the program" };
  });
  await assert.rejects(
    asks.requestLocal("rohan", { host: "127.0.0.1", port: 8811 }, "/view/v1"),
    (error: unknown) => {
      assert.ok(error instanceof SourceUnavailableError);
      assert.equal(error.code, "exec-spawn-failed");
      assert.match(error.message, /rohan/);
      assert.match(error.message, /could not start the program/);
      return true;
    },
  );
});

test("a route that is not a path refuses before anything rides", async () => {
  const asks = coreAsks(async () => {
    throw new Error("nothing may ride");
  });
  await assert.rejects(
    asks.requestLocal("rohan", { host: "127.0.0.1", port: 8811 }, "view/v1"),
    (error: unknown) => {
      assert.ok(error instanceof SourceUnavailableError);
      assert.equal(error.code, "read-failed");
      assert.match(error.message, /must start with/);
      return true;
    },
  );
});

test("the direct implementation performs the asks itself", async () => {
  const root = mkdtempSync(join(tmpdir(), "saccade-asks-"));
  try {
    const path = join(root, "workbench.toml");
    writeFileSync(path, "name = \"saccade\"\n");
    const run = async (
      _machine: string,
      argv: readonly string[],
    ): Promise<CommandOutcome> => {
      const result = await promisify(execFile)(argv[0]!, argv.slice(1), {
        encoding: "buffer",
        timeout: 10_000,
      });
      return {
        code: 0,
        stdout: new Uint8Array(result.stdout),
        stderr: new Uint8Array(result.stderr),
        truncated: false,
        timedOut: false,
      };
    };
    const server = createServer((_request, response) => {
      response.end("served here");
    });
    const address = await new Promise<{ host: string; port: number }>((resolve) => {
      server.listen(0, "127.0.0.1", () => {
        const detected = server.address();
        assert.ok(detected !== null && typeof detected === "object");
        resolve({ host: detected.address, port: detected.port });
      });
    });
    const asks = directAsks({
      readFile: async (_machine, read) => readFileSync(read, "utf8"),
      runCommand: run,
      requestLocal: async (_machine, local, route) => {
        const reply = await fetch(`http://${local.host}:${local.port}${route}`);
        return new Uint8Array(await reply.arrayBuffer());
      },
    });
    assert.equal(await asks.readFile("here", path), "name = \"saccade\"\n");
    const outcome = await asks.runCommand("here", ["cat", path]);
    assert.equal(outcome.code, 0);
    assert.match(new TextDecoder().decode(outcome.stdout), /saccade/);
    const body = await asks.requestLocal("here", address, "/the-route");
    assert.equal(new TextDecoder().decode(body), "served here");
    await new Promise<void>((resolve) => server.close(() => resolve()));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
