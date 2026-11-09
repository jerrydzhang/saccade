#!/usr/bin/env python3
"""The subscription pin (t-24@r22's first ask): is herdr-core 0.1.17's
events.subscribe limited per connection, per caller, or core-global?
The t-20 companion and the viewer's deck feed both hold subscriptions;
whether they can coexist decides a t-20 follow-up or a README law line.

Methodology per i-9's input (i-12's E2E was inconclusive because
silence cannot distinguish a hang from an idle feed): every emission is
a REAL observable fact - a doc.set whose landing is verified by
doc.get before anyone is judged - and every subscriber proves liveness
by its snapshot reply before the test asks anything of it. A missing
event with a live, emitting core is a real answer, never idle noise.

Variants: a plain second connection; a hello-identified second
connection (calls_you, the caller name the core-to-core handshake
carries); and the second subscriber alone after the first closes -
"one at a time" versus "the first wins forever". One private core, one
session, owned start/stop; the log is retained under /var/tmp.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time

KEEP = pathlib.Path("/var/tmp/saccade-t21")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


class Rpc:
    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect(path)
        self.reader = self.sock.makefile("rb")
        self.writer = self.sock.makefile("wb")
        self.next_id = 0

    def send(self, message: dict) -> None:
        self.writer.write((json.dumps(message) + "\n").encode())
        self.writer.flush()

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        request: dict = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            request["params"] = params
        self.send(request)
        return json.loads(self.reader.readline())

    def read_line(self, timeout: float) -> str | None:
        self.sock.settimeout(timeout)
        try:
            raw = self.reader.readline()
        except (TimeoutError, OSError):
            return None
        return raw.decode() if raw else None

    def read_until_doc(self, name: str, timeout: float) -> tuple[bool, str | None]:
        """Wait for a doc event naming `name`; everything else
        (heartbeat, other lines) is printed as evidence. A heartbeat
        received on the way proves the stream is connected and being
        read."""
        end = time.monotonic() + timeout
        last = "nothing in the window"
        while time.monotonic() < end:
            line = self.read_line(2.0)
            if line is None:
                continue
            if '"heartbeat"' in line:
                last = "heartbeat (stream alive, no event yet)"
                continue
            if '"doc"' in line and name in line:
                return True, line.strip()[:240]
            last = line.strip()[:200]
        return False, last

    def close(self) -> None:
        self.sock.close()


def start_core(core_bin: str, runtime: pathlib.Path, logs: pathlib.Path) -> subprocess.Popen:
    log = (logs / "core.log").open("wb")
    core = subprocess.Popen(
        [core_bin, "serve"],
        env={
            **os.environ,
            "HC_SESSION": "p",
            "XDG_RUNTIME_DIR": str(runtime),
            "HOME": str(runtime.parent / "home"),
        },
        stdout=log,
        stderr=log,
    )
    sock = runtime / "herdr-core" / "p.sock"
    end = time.monotonic() + 15
    while time.monotonic() < end:
        if core.poll() is not None:
            raise SystemExit("the core exited during startup")
        if sock.exists():
            return core
        time.sleep(0.1)
    raise SystemExit("the core socket never appeared")


def main() -> int:
    core_bin = os.environ.get("DECK_PROBE_CORE", "herdr-core")
    core_bin_abs = (
        core_bin
        if core_bin.startswith("/")
        else subprocess.run(["sh", "-c", f"command -v {core_bin}"], capture_output=True, text=True)
        .stdout.strip()
    )
    KEEP.mkdir(parents=True, exist_ok=True)
    probe_dir = pathlib.Path(KEEP) / f"sub-pin-{time.strftime('%Y%m%d-%H%M%S')}"
    runtime = probe_dir / "runtime"
    (probe_dir / "home").mkdir(parents=True)
    runtime.mkdir(parents=True)
    logs = probe_dir / "logs"
    logs.mkdir()
    print(f"probe dir: {probe_dir}", flush=True)
    sock_path = str(runtime / "herdr-core" / "p.sock")

    core = start_core(core_bin_abs, runtime, logs)
    try:
        name = "deck-" + "ef" * 12
        writer = Rpc(sock_path)

        # subscriber 1: the plain shape every current client uses
        sub1 = Rpc(sock_path)
        sub1.send({"jsonrpc": "2.0", "id": 1, "method": "events.subscribe", "params": {}})
        reply1 = sub1.read_line(5.0)
        check(
            "subscriber 1's snapshot reply arrives",
            reply1 is not None and '"result"' in reply1,
            (reply1 or "NONE")[:160],
        )

        # emission 1, verified landed before anyone is judged
        landed = writer.call("doc.set", {"name": name, "value": ["one"], "version": 0})
        got = writer.call("doc.get", {"name": name})
        version_now = got.get("result", {}).get("version")
        check(
            "emission 1 provably landed",
            "result" in landed and version_now == 1,
            json.dumps(landed)[:120],
        )
        got1, detail1 = sub1.read_until_doc(name, 8.0)
        check("subscriber 1 receives the landing (positive control)", got1, detail1)

        # subscriber 2: a plain second connection, no hello
        sub2 = Rpc(sock_path)
        sub2.send({"jsonrpc": "2.0", "id": 1, "method": "events.subscribe", "params": {}})
        reply2 = sub2.read_line(8.0)
        check(
            "subscriber 2 (plain) snapshot reply arrives",
            reply2 is not None and '"result"' in reply2,
            (reply2 or "NONE - no answer within 8s while subscriber 1 holds its feed")[:160],
        )

        # emission 2, verified landed; who sees it?
        writer.call("doc.set", {"name": name, "value": ["one", "two"], "version": version_now})
        got = writer.call("doc.get", {"name": name})
        version_now = got.get("result", {}).get("version")
        check("emission 2 provably landed", version_now == 2, str(version_now))
        got1, detail1 = sub1.read_until_doc(name, 8.0)
        got2, detail2 = sub2.read_until_doc(name, 8.0) if reply2 is not None else (False, "no subscription reply")
        check("subscriber 1 still receives (second subscriber open)", got1, detail1)
        check("subscriber 2 (plain) receives", got2, detail2)

        # subscriber 3: hello-identified, the caller name the
        # core-to-core handshake carries
        sub3 = Rpc(sock_path)
        hello = sub3.call("hello", {"generation": 1, "calls_you": "companion"})
        identified = "result" in hello
        check("subscriber 3's hello (calls_you) answers", identified, json.dumps(hello)[:160])
        if identified:
            sub3.send({"jsonrpc": "2.0", "id": 9, "method": "events.subscribe", "params": {}})
            reply3 = sub3.read_line(8.0)
            check(
                "subscriber 3 (identified) snapshot reply arrives",
                reply3 is not None and '"result"' in reply3,
                (reply3 or "NONE")[:160],
            )
            writer.call("doc.set", {"name": name, "value": ["one", "two", "three"], "version": version_now})
            got = writer.call("doc.get", {"name": name})
            version_now = got.get("result", {}).get("version")
            check("emission 3 provably landed", version_now == 3, str(version_now))
            got1, detail1 = sub1.read_until_doc(name, 8.0)
            got3, detail3 = sub3.read_until_doc(name, 8.0) if reply3 is not None else (False, "no subscription reply")
            check("subscriber 1 still receives (identified subscriber open)", got1, detail1)
            check("subscriber 3 (identified) receives", got3, detail3)

        # the second subscriber alone, after the first closes: does the
        # feed EVER reach a second subscriber, or does the first win
        # for its whole life?
        sub1.close()
        time.sleep(0.5)
        if reply2 is not None:
            writer.call("doc.set", {"name": name, "value": ["one", "two", "three", "four"], "version": version_now})
            got = writer.call("doc.get", {"name": name})
            version_now = got.get("result", {}).get("version")
            check("emission 4 provably landed", version_now == 4, str(version_now))
            got2, detail2 = sub2.read_until_doc(name, 8.0)
            check(
                "subscriber 2 receives once subscriber 1 is gone",
                got2,
                detail2,
            )
        else:
            # subscriber 2 never subscribed at all: a fresh plain
            # connection takes the feed after subscriber 1 closes?
            sub2b = Rpc(sock_path)
            sub2b.send({"jsonrpc": "2.0", "id": 1, "method": "events.subscribe", "params": {}})
            reply2b = sub2b.read_line(8.0)
            check(
                "a fresh plain subscriber's reply arrives after subscriber 1 closes",
                reply2b is not None and '"result"' in reply2b,
                (reply2b or "NONE")[:160],
            )
            if reply2b is not None and '"result"' in reply2b:
                writer.call(
                    "doc.set",
                    {"name": name, "value": ["one", "two", "three", "four"], "version": version_now},
                )
                got = writer.call("doc.get", {"name": name})
                version_now = got.get("result", {}).get("version")
                check("emission 4 provably landed", version_now == 4, str(version_now))
                got2b, detail2b = sub2b.read_until_doc(name, 8.0)
                check("the fresh plain subscriber receives", got2b, detail2b)

        writer.close()
        return 1 if failures else 0
    finally:
        core.terminate()
        try:
            core.wait(timeout=5)
        except subprocess.TimeoutExpired:
            core.kill()
            core.wait(timeout=5)
        print("core stopped", flush=True)
        if failures:
            print(f"FAILURES: {failures}", flush=True)


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:  # noqa: BLE001 - a crashed probe keeps its artifacts
        import traceback

        traceback.print_exc()
        print(f"probe crashed: {error!r}", file=sys.stderr)
        code = 1
    sys.exit(code)
