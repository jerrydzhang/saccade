#!/usr/bin/env python3
"""The relay probe: two private herdr-core instances linked the remote
mesh lane's way (machine add --via), then the questions the deck's
feed still has after the lane run:

- what does a subscription on the hub core with `on` = the remote
  machine actually deliver when a document lands on that machine -
  the event's exact name form (bare, machine-qualified, or nothing);
- whether doc events cross the via pipe at all.

Everything runs on this script's own pair of cores in its own runtime
directory; owned start/stop by pid; the log is retained.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KEEP = pathlib.Path("/var/tmp/saccade-t21")

_core_bin = os.environ.get("DECK_PROBE_CORE", "herdr-core")
_core_bin_abs = (
    _core_bin
    if _core_bin.startswith("/")
    else subprocess.run(["sh", "-c", f"command -v {_core_bin}"], capture_output=True, text=True)
    .stdout.strip()
)
_coreutils_bin = str(
    pathlib.Path(
        subprocess.run(["sh", "-c", "command -v realpath"], capture_output=True, text=True)
        .stdout.strip()
    ).parent
)


class Rpc:
    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect(path)
        self.reader = self.sock.makefile("rb")
        self.writer = self.sock.makefile("wb")
        self.next_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        request: dict = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            request["params"] = params
        self.writer.write((json.dumps(request) + "\n").encode())
        self.writer.flush()
        reply = json.loads(self.reader.readline())
        return reply

    def line(self, timeout: float = 10.0) -> str | None:
        self.sock.settimeout(timeout)
        try:
            raw = self.reader.readline()
        except (TimeoutError, OSError):
            return None
        if not raw:
            return None
        return raw.decode()

    def close(self) -> None:
        self.sock.close()


def start_core(core_bin: str, session: str, runtime: pathlib.Path, logs: pathlib.Path) -> subprocess.Popen:
    core_log = (logs / f"core-{session}.log").open("wb")
    core = subprocess.Popen(
        [core_bin, "serve"],
        env={
            **os.environ,
            "HC_SESSION": session,
            "XDG_RUNTIME_DIR": str(runtime),
            "HOME": str(runtime.parent / "home"),
            # the via program resolves the tested bytes itself (its own
            # pinned-path guard, the lane's shape)
            "BOARD_VIEWER_HERDR_CORE_BIN": _core_bin_abs,
            "BOARD_VIEWER_COREUTILS_BIN": _coreutils_bin,
            "PATH": f"{os.path.dirname(_core_bin_abs)}:{_coreutils_bin}:{os.environ.get('PATH', '')}",
        },
        stdout=core_log,
        stderr=core_log,
    )
    socket_path = runtime / "herdr-core" / f"{session}.sock"
    end = time.monotonic() + 15
    while time.monotonic() < end:
        if core.poll() is not None:
            raise SystemExit(f"core {session} exited during startup")
        if socket_path.exists():
            return core
        time.sleep(0.1)
    raise SystemExit(f"core {session} socket never appeared")


def main() -> int:
    core_bin = _core_bin_abs
    KEEP.mkdir(parents=True, exist_ok=True)
    probe_dir = pathlib.Path(KEEP) / f"relay-probe-{time.strftime('%Y%m%d-%H%M%S')}"
    runtime = probe_dir / "runtime"
    (probe_dir / "home").mkdir(parents=True)
    runtime.mkdir(parents=True)
    logs = probe_dir / "logs"
    logs.mkdir()
    print(f"probe dir: {probe_dir}", flush=True)

    core_a = start_core(core_bin, "a", runtime, logs)
    core_b = start_core(core_bin, "b", runtime, logs)
    try:
        via_py = str(HERE / "via_b.py")
        python = sys.executable
        add = subprocess.run(
            [
                core_bin,
                "machine",
                "add",
                "--via",
                f"{python} {via_py}",
                "--name",
                "b",
                "--session",
                "b",
                "--json",
            ],
            env={
                **os.environ,
                "HC_SESSION": "a",
                "XDG_RUNTIME_DIR": str(runtime),
                "PATH": f"{os.path.dirname(core_bin)}:{os.environ.get('PATH', '')}",
            },
            capture_output=True,
            timeout=30,
        )
        print(
            f"machine add rc={add.returncode} out={add.stdout.decode()[:200]} err={add.stderr.decode()[:200]}",
            flush=True,
        )
        if add.returncode != 0:
            return 1

        name = "deck-" + "cd" * 12
        rpc_b = Rpc(str(runtime / "herdr-core" / "b.sock"))
        # the LANE's creation shape: the app's deck_create goes through
        # the relay (doc.set on=b via A), so the document is born on the
        # hub's relayed write, not on B's own socket
        rpc_a0 = Rpc(str(runtime / "herdr-core" / "a.sock"))
        created = rpc_a0.call("doc.set", {"name": name, "value": [], "version": 0, "on": "b"})
        print(f"doc created via the relay: {json.dumps(created)[:120]}", flush=True)
        # B's own subscription exists BEFORE the landing: if a meshed
        # core stops emitting doc events locally, this shows it
        sub_b0 = Rpc(str(runtime / "herdr-core" / "b.sock"))
        sub_b0.next_id += 1
        sub_b0.writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": sub_b0.next_id,
                        "method": "events.subscribe",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        print(f"subscribe-b0 reply: {(sub_b0.line() or 'NONE')[:160]}", flush=True)
        created = rpc_b.call("doc.set", {"name": name, "value": [], "version": 0})
        print(f"doc on b created: {json.dumps(created)[:120]}", flush=True)
        line = sub_b0.line(timeout=5.0)
        print(f"b0 after create: {(line or 'NONE')[:300]}", flush=True)

        # subscribe on A's socket with on=b: the app's exact shape
        sub = Rpc(str(runtime / "herdr-core" / "a.sock"))
        sub.next_id += 1
        sub.writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": sub.next_id,
                        "method": "events.subscribe",
                        "params": {"on": "b"},
                    }
                )
                + "\n"
            ).encode()
        )
        sub.writer.flush()
        reply = sub.line()
        print(f"subscribe reply: {(reply or 'NONE')[:200]}", flush=True)

        # land a doc on b, then read A's stream raw for 10 s - with the
        # lane's timing first: the cue lands ~35s after the subscription
        # opened, with nothing in between; idleness must not matter
        print("idle 35s (the lane's subscribe-to-cue gap)...", flush=True)
        idle_end = time.monotonic() + 35
        while time.monotonic() < idle_end:
            idle_line = sub.line(timeout=2.0)
            if idle_line is not None:
                print(f"A-stream line (idle window): {idle_line.strip()[:200]}", flush=True)
        written = rpc_b.call(
            "doc.set",
            {"name": name, "value": ["b:/board/deck-slide.yaml"], "version": 1},
        )
        print(f"doc on b written: {json.dumps(written)[:120]}", flush=True)

        end = time.monotonic() + 10
        seen = 0
        while time.monotonic() < end:
            line = sub.line(timeout=2.0)
            if line is None:
                continue
            seen += 1
            print(f"A-stream line: {line.strip()[:300]}", flush=True)
            if seen >= 12:
                break

        # and the same landing read directly on the pre-existing B feed
        sub_b = sub_b0
        written_b = rpc_b.call(
            "doc.set",
            {"name": name, "value": ["b:/board/deck-slide.yaml", "second"], "version": 2},
        )
        print(f"doc on b written again: {json.dumps(written_b)[:120]}", flush=True)
        end = time.monotonic() + 15
        while time.monotonic() < end:
            line = sub_b.line(timeout=2.0)
            if line is None:
                continue
            print(f"B-stream line: {line.strip()[:300]}", flush=True)
            if '"doc"' in line:
                break
        # the lane's exact shape, whole: the document was CREATED
        # through the relay (doc.set on=b, the app's deck_create), the
        # viewer's subscription is on the LOCAL stream (no on - the
        # stream that carries machine news), and the cue rides the
        # relay again
        rpc_a = Rpc(str(runtime / "herdr-core" / "a.sock"))
        local_sub = Rpc(str(runtime / "herdr-core" / "a.sock"))
        local_sub.next_id += 1
        local_sub.writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": local_sub.next_id,
                        "method": "events.subscribe",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        print(f"local subscribe reply: {(local_sub.line() or 'NONE')[:160]}", flush=True)
        read = rpc_a.call("doc.get", {"name": name, "on": "b"})
        version = read.get("result", {}).get("version")
        written = rpc_a.call(
            "doc.set",
            {
                "name": name,
                "on": "b",
                "version": version,
                "value": ["b:/board/deck-slide.yaml", "third"],
            },
        )
        print(f"doc on b written via A: {json.dumps(written)[:160]}", flush=True)
        end = time.monotonic() + 12
        seen_local = False
        while time.monotonic() < end:
            line = local_sub.line(timeout=2.0)
            if line is None:
                continue
            print(f"local-stream line (relayed write): {line.strip()[:300]}", flush=True)
            if '"doc"' in line:
                seen_local = True
                break
        print(f"local stream saw the relayed cue: {seen_local}", flush=True)
        rpc_a.close()
        local_sub.close()
        sub.close()
        sub_b.close()
        sub_b0.close()
        rpc_b.close()
        return 0
    finally:
        core_a.terminate()
        core_b.terminate()
        try:
            core_a.wait(timeout=5)
            core_b.wait(timeout=5)
        except subprocess.TimeoutExpired:
            core_a.kill()
            core_b.kill()
        print("cores stopped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
