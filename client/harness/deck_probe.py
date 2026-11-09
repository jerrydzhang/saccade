#!/usr/bin/env python3
"""The deck probe: one private herdr-core, started and stopped by this
script's own handle, answering the questions t-24@r21 (B2) sends to the
sandbox before the deck is built:

- the name rules admit ``deck-`` plus at least 96 bits, and no method
  lists documents (an address that can be listed or guessed is not a
  permission, so either failure sends the transport back to i-3);
- the doc and file.edit call shapes the client will speak, captured from
  the real core 0.1.17, including the events.subscribe notification a
  cue arrives as and the exact-once/all-or-none edit semantics.

Everything runs against the probe's own session socket in its own
runtime directory; the live hub core is never touched. The log is
retained under the probe directory.
"""

from __future__ import annotations

import json
import os
import pathlib
import secrets
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEEP = pathlib.Path("/var/tmp/saccade-t21")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


class Rpc:
    """One JSON-RPC connection, one message per line (core 0.1.17)."""

    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect(path)
        self.reader = self.sock.makefile("rb")
        self.writer = self.sock.makefile("wb")
        self.next_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        request = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            request["params"] = params
        self.writer.write((json.dumps(request) + "\n").encode())
        self.writer.flush()
        line = self.reader.readline()
        if not line:
            raise RuntimeError(f"{method}: the core closed the connection")
        reply = json.loads(line)
        if reply.get("id") != self.next_id:
            raise RuntimeError(f"{method}: reply id {reply.get('id')} != {self.next_id}")
        return reply

    def read_notification(self, timeout: float = 5.0) -> dict | None:
        """One inbound line that is not a response to a sent id."""
        self.sock.settimeout(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.reader.readline()
            if not line:
                return None
            message = json.loads(line)
            if "id" not in message:
                return message
        return None

    def close(self) -> None:
        try:
            self.reader.close()
            self.writer.close()
        finally:
            self.sock.close()


def main() -> int:
    core_bin = os.environ.get("DECK_PROBE_CORE", "herdr-core")
    KEEP.mkdir(parents=True, exist_ok=True)
    probe_dir = pathlib.Path(KEEP) / f"deck-probe-{time.strftime('%Y%m%d-%H%M%S')}"
    runtime = probe_dir / "runtime"
    home = probe_dir / "home"
    runtime.mkdir(parents=True)
    home.mkdir(parents=True)
    log_path = probe_dir / "probe.log"
    session = "deckprobe"
    socket_path = runtime / "herdr-core" / f"{session}.sock"

    def tee(text: str) -> None:
        with log_path.open("a") as log:
            log.write(text + "\n")

    env = {
        "PATH": os.environ.get("PATH", "/bin:/usr/bin"),
        "HOME": str(home),
        "XDG_RUNTIME_DIR": str(runtime),
        "HC_SESSION": session,
        "HC_NOTES": "0",
    }
    core_log_path = probe_dir / "core.log"
    print(f"probe dir: {probe_dir}", flush=True)
    tee(f"core: {core_bin} serve; env: {env}")
    core_log = core_log_path.open("wb")
    core = subprocess.Popen(
        [core_bin, "serve"],
        env=env,
        stdout=core_log,
        stderr=core_log,
    )
    try:
        end = time.monotonic() + 15.0
        while time.monotonic() < end:
            if core.poll() is not None:
                check("core stayed up", False, f"exited during startup ({core_log_path})")
                return 1
            if socket_path.exists():
                break
            time.sleep(0.1)
        else:
            check("core socket up", False, f"never appeared ({core_log_path})")
            return 1
        check("core socket up", True, str(socket_path))

        rpc = Rpc(str(socket_path))
        hello = rpc.call("hello", {"generation": 1})
        methods = hello.get("result", {}).get("methods", [])
        check(
            "hello answers with methods",
            len(methods) > 0,
            f"{len(methods)} methods",
        )
        doc_methods = sorted(name for name in methods if "doc" in name)
        check(
            "no method lists documents",
            doc_methods == ["doc.get", "doc.set"],
            f"doc-carrying methods: {doc_methods}",
        )

        # -- names --------------------------------------------------------
        deck_name = "deck-" + secrets.token_hex(12)  # 24 hex chars = 96 bits
        created = rpc.call("doc.set", {"name": deck_name, "value": [], "version": 0})
        check(
            "doc.set accepts deck-<24 hex> (96 bits)",
            "result" in created and "error" not in created,
            json.dumps(created)[:200],
        )
        long_name = "deck-" + secrets.token_hex(32)  # 64 hex chars = 256 bits
        long_created = rpc.call("doc.set", {"name": long_name, "value": [], "version": 0})
        print(
            "[NOTE] deck-<64 hex> (256 bits): "
            + ("accepted" if "result" in long_created else "refused by the name cap - 96 bits via 24 hex fits, which is what the deck needs"),
            flush=True,
        )
        weird_name = "deck-" + secrets.token_hex(6) + "/../probe"
        refused = rpc.call("doc.set", {"name": weird_name, "value": [], "version": 0})
        check(
            "doc.set refuses a name with a separator (no traversal)",
            "error" in refused,
            json.dumps(refused)[:200],
        )

        # -- the doc is the read-back ------------------------------------
        read_back = rpc.call("doc.get", {"name": deck_name})
        value = read_back.get("result", {}).get("value")
        version = read_back.get("result", {}).get("version")
        check(
            "doc.get reads back the deck document",
            value == [] and isinstance(version, int) and version >= 1,
            f"value={value!r} version={version!r}",
        )
        never = rpc.call("doc.get", {"name": "deck-" + secrets.token_hex(12)})
        check(
            "doc.get on a name never written is value null version 0",
            never.get("result", {}).get("value") is None
            and never.get("result", {}).get("version") == 0,
            json.dumps(never)[:200],
        )

        # -- the agent's cue race: version conflict -----------------------
        stale = rpc.call(
            "doc.set",
            {"name": deck_name, "value": ["tracker/x/t-1"], "version": 0},
        )
        check(
            "doc.set with a stale version refuses (conflict)",
            "error" in stale and stale.get("error", {}).get("data", {}).get("kind") == "conflict",
            json.dumps(stale)[:200],
        )

        # -- the notification a cue arrives as ----------------------------
        sub_socket = Rpc(str(socket_path))
        subscribed = sub_socket.call("events.subscribe", {})
        snapshot = subscribed.get("result")
        snapshot_text = json.dumps(snapshot)[:400] if snapshot is not None else "none"
        snapshot_names = json.dumps(snapshot)
        check(
            "the subscription snapshot lists no documents",
            deck_name not in snapshot_names and long_name not in snapshot_names,
            snapshot_text,
        )
        appended = rpc.call(
            "doc.set",
            {"name": deck_name, "value": ["tracker/x/t-1"], "version": version},
        )
        check(
            "doc.set appends a slide at the current version",
            "result" in appended,
            json.dumps(appended)[:200],
        )
        notification = sub_socket.read_notification(timeout=5.0)
        check(
            "a cue arrives as a subscription notification",
            notification is not None and deck_name in json.dumps(notification),
            json.dumps(notification)[:300] if notification else "nothing arrived in 5s",
        )
        if notification is not None:
            tee(f"doc notification shape: {json.dumps(notification)}")
            print(f"[SHAPE] doc notification: {json.dumps(notification)}", flush=True)
        sub_socket.close()

        # -- delete semantics: the address dies with the deck -------------
        current = rpc.call("doc.get", {"name": deck_name})
        current_version = current.get("result", {}).get("version")
        deleted = rpc.call(
            "doc.set", {"name": deck_name, "value": None, "version": current_version}
        )
        after_delete = rpc.call("doc.get", {"name": deck_name})
        check(
            "doc.set null deletes the value and retains the version",
            "result" in deleted
            and after_delete.get("result", {}).get("value") is None
            and isinstance(after_delete.get("result", {}).get("version"), int)
            and after_delete.get("result", {}).get("version", 0) >= 1,
            f"delete={json.dumps(deleted)[:120]} after={json.dumps(after_delete)[:120]}",
        )

        # -- terminal listing cannot see documents ------------------------
        listing = subprocess.run(
            [core_bin, "ls", "--all", "--json"],
            env=env,
            capture_output=True,
            timeout=10,
        )
        listing_text = listing.stdout.decode()
        check(
            "terminal listing carries no document names",
            deck_name not in listing_text and long_name not in listing_text,
            f"{len(listing_text)} bytes" if listing_text else "empty",
        )

        # -- file.edit: the write the annotation submit rides -------------
        target = probe_dir / "explainer.yaml"
        original = "content:\n  - text: |\n      first part\n  - text: |\n      second part\n"
        target.write_text(original)
        old = "  - text: |\n      first part"
        new = old + "\n  - annotation: |\n      note on the first part"
        edit = rpc.call("file.edit", {"path": str(target), "edits": [{"old": old, "new": new}]})
        check(
            "file.edit inserts one annotation block",
            "result" in edit and "annotation: |" in target.read_text(),
            json.dumps(edit)[:200],
        )
        # a still-unique old occurring twice refuses, all-or-none
        target.write_text(original + original)
        dup_edit = rpc.call(
            "file.edit",
            {"path": str(target), "edits": [{"old": old, "new": new}]},
        )
        check(
            "file.edit refuses old text that occurs twice (exact-once)",
            "error" in dup_edit
            and target.read_text() == original + original
            and dup_edit.get("error", {}).get("data", {}).get("kind") == "edit_failed",
            json.dumps(dup_edit)[:200],
        )
        check(
            "the refused duplicate edit touched nothing (all-or-none)",
            target.read_text() == original + original,
            target.read_text()[:80],
        )
        target.write_text(original)
        two_edits = rpc.call(
            "file.edit",
            {
                "path": str(target),
                "edits": [
                    {"old": "first part", "new": "first part, edited"},
                    {"old": "text that is not there", "new": "x"},
                ],
            },
        )
        check(
            "file.edit applies all edits or none",
            "error" in two_edits and "first part\n" in target.read_text(),
            json.dumps(two_edits)[:200],
        )
        read = rpc.call("file.read", {"path": str(target)})
        check(
            "file.read answers the edited file",
            "result" in read and "first part" in json.dumps(read),
            json.dumps(read)[:150],
        )

        # -- the exact cue command the copy act carries -------------------
        # The string the viewer copies and the README writes out: an
        # agent runs it verbatim, appending its location as the argument.
        cue_name = "deck-" + secrets.token_hex(12)
        rpc.call("doc.set", {"name": cue_name, "value": [], "version": 0})
        cue_command = f"""python3 - tracker/saccade/t-14@r3 <<'DECK'
import json, subprocess, sys
name = "{cue_name}"
def call(method, params):
    line = json.dumps({{"jsonrpc": "2.0", "id": 1, "method": method, "params": params}})
    out = subprocess.run(["hc", "connect"], input=line + "\\n", capture_output=True, text=True)
    return json.loads(out.stdout)
doc = call("doc.get", {{"name": name}})["result"]
while True:
    slides = (doc.get("value") or [])
    slides.append(sys.argv[1])
    done = call("doc.set", {{"name": name, "version": doc["version"], "value": slides}})
    if "error" not in done:
        print("slide", len(slides))
        break
    doc = call("doc.get", {{"name": name}})["result"]
DECK"""
        ran = subprocess.run(
            ["bash", "-c", cue_command],
            env={**env, "PATH": os.environ.get("PATH", env["PATH"])},
            capture_output=True,
            text=True,
            timeout=30,
        )
        check(
            "the exact cue command appends and names the slide",
            ran.returncode == 0 and ran.stdout.strip() == "slide 1",
            f"exit={ran.returncode} out={ran.stdout.strip()!r} err={ran.stderr.strip()[:120]!r}",
        )
        read_back = rpc.call("doc.get", {"name": cue_name})
        check(
            "the cued slide is in the deck document",
            read_back.get("result", {}).get("value") == ["tracker/saccade/t-14@r3"],
            json.dumps(read_back)[:200],
        )

        rpc.close()
        return 1 if failures else 0
    finally:
        core.terminate()
        try:
            core.wait(timeout=5)
        except subprocess.TimeoutExpired:
            core.kill()
            core.wait(timeout=5)
        core_log.close()
        tee(f"core exit: {core.returncode}")
        print(f"core stopped (exit {core.returncode}); log: {core_log_path}", flush=True)
        if failures:
            print(f"FAILURES: {failures}", flush=True)


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:  # a crashed probe keeps its artifacts
        import traceback

        traceback.print_exc()
        print(f"probe crashed: {error!r}", file=sys.stderr)
        code = 1
    sys.exit(code)
