import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  baseFromHost,
  parseLocation,
  parseTaskSpec,
  trackerLocation,
  viewUrl,
} from "../src/lib/location.ts";

test("a bare absolute path is a file on the tracker's machine", () => {
  assert.deepEqual(parseLocation("/srv/boards/first.yaml"), {
    machine: "tracker-machine",
    path: "/srv/boards/first.yaml",
  });
});

test("machine:/absolute/path names the remote machine", () => {
  assert.deepEqual(parseLocation("tired:/srv/boards/first.yaml"), {
    machine: "tired",
    path: "/srv/boards/first.yaml",
  });
});

test("the separator is the first colon whose remainder starts with a slash", () => {
  // an ssh-style machine name carries its own colon
  assert.deepEqual(parseLocation("user@host:22:/srv/b.yaml"), {
    machine: "user@host:22",
    path: "/srv/b.yaml",
  });
  // a colon inside the path stays in the path
  assert.deepEqual(parseLocation("tired:/srv/a:b/c.yaml"), {
    machine: "tired",
    path: "/srv/a:b/c.yaml",
  });
  // a tracker-machine path with colons never becomes a machine name
  assert.deepEqual(parseLocation("/srv/weird:colon/b.yaml"), {
    machine: "tracker-machine",
    path: "/srv/weird:colon/b.yaml",
  });
});

test("a machine-prefixed path must be absolute", () => {
  assert.equal(parseLocation("tired:relative.yaml"), null);
  assert.equal(parseLocation("tired:"), null);
});

test("an empty machine name is refused", () => {
  assert.equal(parseLocation(":/srv/b.yaml"), null);
});

test("anything that is not a location is refused, never guessed", () => {
  for (const attempt of ["", "   ", "relative.yaml", "./x.yaml", "~/b.yaml"]) {
    assert.equal(parseLocation(attempt), null, JSON.stringify(attempt));
  }
});

test("surrounding whitespace is not part of the location", () => {
  assert.deepEqual(parseLocation("  /srv/b.yaml  "), {
    machine: "tracker-machine",
    path: "/srv/b.yaml",
  });
});

test("the parser never names the host reader; the explicit form is the inspection surface's", () => {
  // t-24@r11: local-device paths left the human surface; the host
  // reader is the lanes' test-only machinery. The parser produces the
  // tracker-machine form for bare paths and never invents "host";
  // the explicit host: form keeps its name only for the inspection
  // surface's own calls.
  assert.equal(parseLocation("/srv/b.yaml")?.machine, "tracker-machine");
  assert.equal(parseLocation("host:/srv/b.yaml")?.machine, "host");
});

test("a tracker location names the tracker and the task spec", () => {
  assert.deepEqual(trackerLocation("saccade/t-56"), {
    machine: "tracker:saccade",
    path: "t-56",
  });
  assert.deepEqual(trackerLocation("saccade/t-56@r3"), {
    machine: "tracker:saccade",
    path: "t-56@r3",
  });
  assert.equal(
    trackerLocation("/saccade/t-56"),
    null,
    "a leading slash is a tracker-machine path",
  );
  assert.equal(
    trackerLocation("host:/saccade/t-56"),
    null,
    "a machine path keeps its colon form",
  );
  assert.equal(trackerLocation("saccade/t-56/extra"), null);
  assert.deepEqual(
    parseLocation("saccade/t-56"),
    trackerLocation("saccade/t-56"),
  );
  assert.equal(
    parseLocation("saccade/t-56")?.machine,
    "tracker:saccade",
    "the folded parser answers the tracker form",
  );
  assert.equal(
    parseLocation("machine:/path.yaml")?.machine,
    "machine",
    "the machine form survives",
  );
});

test("a task spec parses the bare and revisioned forms", () => {
  assert.deepEqual(parseTaskSpec("t-5"), { task: 5, revision: null });
  assert.deepEqual(parseTaskSpec("t-5@r3"), { task: 5, revision: 3 });
  assert.equal(parseTaskSpec("t-"), null);
  assert.equal(parseTaskSpec("5"), null);
  assert.equal(parseTaskSpec("t-5/r2"), null);
});

test("view routes resolve against the server host with one slash", () => {
  for (const host of [
    "192.168.0.51:8811",
    "127.0.0.1:8811/",
    "http://127.0.0.1:8811",
  ]) {
    const base = baseFromHost(host);
    assert.ok(base !== null, host);
    const url = viewUrl(base, "t-7/file/board.yaml");
    assert.equal(new URL(url).pathname, "/view/v1/t-7/file/board.yaml", url);
    const root = viewUrl(base, "");
    assert.equal(new URL(root).pathname, "/view/v1/", root);
    const nested = viewUrl(
      base,
      "/t-7@r2/file/figures/deep/nested history.png",
    );
    assert.equal(
      new URL(nested).pathname,
      "/view/v1/t-7@r2/file/figures/deep/nested%20history.png",
      nested,
    );
  }
});

test("the native shell's policy lets the webview reach a tracker server", () => {
  const conf = JSON.parse(
    readFileSync(
      new URL("../src-tauri/tauri.conf.json", import.meta.url),
      "utf8",
    ),
  );
  const connect = String(conf.app.security.csp)
    .split(";")
    .map((part: string) => part.trim())
    .find((part: string) => part.startsWith("connect-src"));
  assert.ok(connect, "the policy names connect-src");
  const sources = connect.split(/\s+/);
  assert.ok(sources.includes("http:") && sources.includes("https:"), connect);
});
