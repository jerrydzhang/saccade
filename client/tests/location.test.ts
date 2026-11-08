import test from "node:test";
import assert from "node:assert/strict";
import { apiUrl, baseFromHost, parseLocation, parseTaskSpec, trackerLocation } from "../src/lib/location.ts";

test("a plain absolute path is the machine this viewer runs on", () => {
  assert.deepEqual(parseLocation("/srv/boards/first.yaml"), {
    machine: "host",
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
  // an absolute local path with colons never becomes a machine name
  assert.deepEqual(parseLocation("/srv/weird:colon/b.yaml"), {
    machine: "host",
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
    machine: "host",
    path: "/srv/b.yaml",
  });
});

test("the composed launch form round-trips", () => {
  // the shell composes "--on <machine> <path>" the same way the
  // bridge does: machine, colon, path
  const composed = "user@host:22:/srv/b.yaml";
  const parsed = parseLocation(composed);
  assert.ok(parsed);
  assert.equal(`${parsed.machine}:${parsed.path}`, composed);
});

test("a tracker location names the tracker and the task spec", () => {
  assert.deepEqual(trackerLocation("saccade/t-56"), { machine: "tracker:saccade", path: "t-56" });
  assert.deepEqual(trackerLocation("saccade/t-56@r3"), { machine: "tracker:saccade", path: "t-56@r3" });
  assert.equal(trackerLocation("/saccade/t-56"), null, "a leading slash is a host path");
  assert.equal(trackerLocation("host:/saccade/t-56"), null, "a machine path keeps its colon form");
  assert.equal(trackerLocation("saccade/t-56/extra"), null);
  assert.deepEqual(parseLocation("saccade/t-56"), trackerLocation("saccade/t-56"));
  assert.equal(
    parseLocation("saccade/t-56")?.machine,
    "tracker:saccade",
    "the folded parser answers the tracker form",
  );
  assert.equal(parseLocation("machine:/path.yaml")?.machine, "machine", "the machine form survives");
});

test("a task spec parses the bare and revisioned forms", () => {
  assert.deepEqual(parseTaskSpec("t-5"), { task: 5, revision: null });
  assert.deepEqual(parseTaskSpec("t-5@r3"), { task: 5, revision: 3 });
  assert.equal(parseTaskSpec("t-"), null);
  assert.equal(parseTaskSpec("5"), null);
  assert.equal(parseTaskSpec("t-5/r2"), null);
});

test("tracker routes resolve against the server host with one slash", () => {
  for (const host of ["192.168.0.51:8811", "127.0.0.1:8811/", "http://127.0.0.1:8811"]) {
    const base = baseFromHost(host);
    assert.ok(base !== null, host);
    const url = apiUrl(base, "/api/v1/version");
    assert.equal(new URL(url).pathname, "/api/v1/version", url);
  }
});
