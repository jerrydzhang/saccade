// The one core module's contract: the asks' shapes, the bridge
// mapping behind them, and the direct implementation tests ride -
// which answers the asks itself, no core between. The module names
// three asks: read a file, read an asset beside it, request a local
// address. It composes no argv of its own - the carrying command is
// the shell's.

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { test } from "vitest";

import { coreAsks, directAsks } from "../src/lib/herdr-core.ts";
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

test("the host reader rides no core: the bridge's on stays unset", async () => {
  const calls: Array<{ cmd: string; args: Record<string, unknown> }> = [];
  const asks = coreAsks(async (cmd, args) => {
    calls.push({ cmd, args: args ?? {} });
    return { text: "host file" };
  });
  const text = await asks.readFile("host", "/srv/workbench.toml");
  assert.equal(text, "host file");
  assert.deepEqual(calls, [{ cmd: "read_declaration", args: { path: "/srv/workbench.toml" } }]);
});

test("an asset read lands on the bridge's read with the machine riding", async () => {
  const calls: Array<{ cmd: string; args: Record<string, unknown> }> = [];
  const asks = coreAsks(async (cmd, args) => {
    calls.push({ cmd, args: args ?? {} });
    return { bytes: [104, 105] };
  });
  const bytes = await asks.readAsset("rohan", "/srv/repo", "figures/a.png");
  assert.equal(new TextDecoder().decode(bytes), "hi");
  assert.deepEqual(calls, [
    { cmd: "read_asset", args: { base: "/srv/repo", relative: "figures/a.png", on: "rohan" } },
  ]);
});

test("a local request names the machine, the port and the route - the argv is the shell's", async () => {
  const calls: Array<{ cmd: string; args: Record<string, unknown> }> = [];
  const asks = coreAsks(async (cmd, args) => {
    calls.push({ cmd, args: args ?? {} });
    return [104, 105];
  });
  const body = await asks.requestLocal("rohan", 8811, "/view/v1");
  assert.equal(new TextDecoder().decode(body), "hi");
  assert.deepEqual(calls, [
    { cmd: "local_request", args: { on: "rohan", port: 8811, route: "/view/v1" } },
  ]);
});

test("a refused local request keeps the bridge's structured shape", async () => {
  const asks = coreAsks(async () => {
    throw { code: "machine-unavailable", message: "the request to 127.0.0.1:8811/view/v1 on rohan: failed" };
  });
  await assert.rejects(
    asks.requestLocal("rohan", 8811, "/view/v1"),
    (error: unknown) => {
      assert.ok(error instanceof SourceUnavailableError);
      assert.equal(error.code, "machine-unavailable");
      assert.match(error.message, /rohan/);
      assert.match(error.message, /127\.0\.0\.1:8811/);
      return true;
    },
  );
});

test("an unknown refusal code surfaces as a bridge bug, named", async () => {
  const asks = coreAsks(async () => {
    throw { code: "something-new", message: "the core said a new thing" };
  });
  await assert.rejects(
    asks.requestLocal("rohan", 8811, "/view/v1"),
    (error: unknown) => {
      assert.ok(error instanceof SourceUnavailableError);
      assert.equal(error.code, "bridge-invalid");
      assert.match(error.message, /a new thing/);
      return true;
    },
  );
});

test("a route that is not a path refuses before anything rides", async () => {
  const asks = coreAsks(async () => {
    throw new Error("nothing may ride");
  });
  await assert.rejects(
    asks.requestLocal("rohan", 8811, "view/v1"),
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
    const server = createServer((_request, response) => {
      response.end("served here");
    });
    const port = await new Promise<number>((resolve) => {
      server.listen(0, "127.0.0.1", () => {
        const detected = server.address();
        assert.ok(detected !== null && typeof detected === "object");
        resolve(detected.port);
      });
    });
    const asks = directAsks({
      readFile: async (_machine, read) => readFileSync(read, "utf8"),
      readAsset: async (_machine, base, relative) =>
        new Uint8Array(readFileSync(join(base, relative))),
      requestLocal: async (_machine, local, route) => {
        const reply = await fetch(`http://127.0.0.1:${local}${route}`);
        return new Uint8Array(await reply.arrayBuffer());
      },
    });
    assert.equal(await asks.readFile("here", path), "name = \"saccade\"\n");
    assert.equal(new TextDecoder().decode(await asks.readAsset("here", root, "workbench.toml")), "name = \"saccade\"\n");
    const body = await asks.requestLocal("here", port, "/the-route");
    assert.equal(new TextDecoder().decode(body), "served here");
    await new Promise<void>((resolve) => server.close(() => resolve()));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
