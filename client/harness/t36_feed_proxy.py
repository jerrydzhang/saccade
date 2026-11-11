#!/usr/bin/env python3
"""The lane-supplied Herdr core proxy (t-36 phase-2): a controlled
port for feed-specific checks, speaking the REAL wire protocol.

The app's words feed dials $XDG_RUNTIME_DIR/herdr-core/<HC_SESSION or
"default">.sock itself and sends `events.subscribe` (core-client/src/
events.rs: jsonrpc request line, id 1). This proxy binds that path
BEFORE the app spawns, answers every subscription with the real
snapshot shape, and then delivers exactly the lines a check asks for:
status events, or an unknown event type. It records every subscribe
line it receives and keeps its connection count readable.

It is observation and control of the feed's TRANSPORT, never a
fabricated publisher of board content: the lines it carries are
terminal words events (the feed's own domain), the same dialect the
pinned core 0.1.17 speaks (`hc events --json`). Board-content checks
use the real core plus the real seeded tracker, never this proxy.

What the protocol pins (all read from core-client/src/events.rs at
4d59c35):
  subscribe request : {"jsonrpc": "2.0", "id": 1, "method":
                       "events.subscribe", "params": {}}
  snapshot answer   : {"jsonrpc": "2.0", "id": 1, "result":
                       {"terminals": [{"address", "state"}, ...]}}
  status event      : {"type": "status", "terminal": T,
                       "data": {"status": {state, app, msg, progress,
                                           title}}}
  finished event    : {"type": "finished", "terminal": T,
                       "data": {"reason": R}}
  unknown event     : any other "type" - the pinned app decodes it to
                       CoreEvent::Other and (at 4d59c35, words.rs) the
                       feed treats it as the stream's end: that is the
                       behavior the S-feed check exercises.
Notifications may ride a jsonrpc envelope or bare params; the client
accepts both, so the proxy sends the bare payload shape.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import threading


class FeedProxy:
    """One Unix socket, one accept loop, one thread per connection.
    Owned handles only; stop() closes the listener and every live
    connection."""

    def __init__(self, socket_path: str, terminals: list[dict] | None = None) -> None:
        self.socket_path = socket_path
        self.terminals = terminals if terminals is not None else [
            {"address": "t-1", "state": "working"},
        ]
        self._lock = threading.Lock()
        self._live: list[socket.socket] = []
        self._subscribes: list[str] = []
        self._queued: list[str] = []
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._started = False

    def start(self) -> None:
        path = pathlib.Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()  # a stale socket from an earlier run is ours to replace
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen(4)
        listener.settimeout(0.5)
        self._listener = listener
        self._stopping = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="t36-feed-proxy")
        self._thread.start()
        self._started = True

    def _accept_loop(self) -> None:
        while not self._stopping:
            listener = self._listener
            if listener is None:
                return
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(30)
        try:
            # the subscribe request line: read it, record it verbatim
            line = b""
            while not line.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    return
                line += chunk
            subscribe = line.decode("utf-8", errors="replace").strip()
            with self._lock:
                self._subscribes.append(subscribe)
                queued, self._queued = self._queued[:], []
            # the snapshot answers first, in the pinned shape
            snapshot = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"terminals": self.terminals},
            }) + "\n"
            conn.sendall(snapshot.encode("utf-8"))
            with self._lock:
                self._live.append(conn)
                pending = queued
            for event in pending:
                try:
                    conn.sendall(event.encode("utf-8"))
                except OSError:
                    return
            # hold the connection open, pruning it when the app hangs up
            while not self._stopping:
                with self._lock:
                    alive = conn in self._live
                if not alive:
                    return
                try:
                    conn.settimeout(0.5)
                    if conn.recv(4096):
                        return  # the client speaks nothing after subscribe
                except TimeoutError:
                    continue
                except OSError:
                    return
        finally:
            with self._lock:
                if conn in self._live:
                    self._live.remove(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _deliver(self, payload: dict) -> int:
        """One event line to every live connection, queued for the
        next one too (the app reconnects at baseline; the follow-up
        event must reach that connection as well)."""
        line = json.dumps(payload) + "\n"
        with self._lock:
            self._queued.append(line)
            targets = self._live[:]
        delivered = 0
        for conn in targets:
            try:
                conn.sendall(line.encode("utf-8"))
                delivered += 1
            except OSError:
                with self._lock:
                    if conn in self._live:
                        self._live.remove(conn)
        return delivered

    # -- the check surface -------------------------------------------------

    def inject_status(
        self,
        terminal: str,
        state: str = "working",
        app: str = "pi",
        msg: str = "",
        progress: str | None = None,
        title: str | None = None,
    ) -> int:
        status: dict = {"state": state, "app": app, "msg": msg}
        if progress is not None:
            status["progress"] = progress
        if title is not None:
            status["title"] = title
        return self._deliver({
            "type": "status",
            "terminal": terminal,
            "data": {"status": status},
        })

    def inject_finished(self, terminal: str, reason: str = "exited") -> int:
        return self._deliver({
            "type": "finished",
            "terminal": terminal,
            "data": {"reason": reason},
        })

    def inject_unknown(self, event_type: str = "deck-cue", extra: dict | None = None) -> int:
        """One event type this app build does not interpret: at
        4d59c35 the feed treats it as the stream's end."""
        payload: dict = {"type": event_type}
        if extra:
            payload.update(extra)
        return self._deliver(payload)

    def recordings(self) -> dict:
        with self._lock:
            return {
                "subscribes": list(self._subscribes),
                "connections": len(self._live),
                "total_subscribes": len(self._subscribes),
                "queued": len(self._queued),
            }

    @property
    def started(self) -> bool:
        return self._started

    def stop(self) -> None:
        self._stopping = True
        with self._lock:
            live = self._live[:]
            self._live = []
        for conn in live:
            try:
                conn.close()
            except OSError:
                pass
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        try:
            pathlib.Path(self.socket_path).unlink()
        except OSError:
            pass
        self._started = False


def default_socket_path() -> str:
    """The path the app's feed resolves at 4d59c35:
    $XDG_RUNTIME_DIR/herdr-core/<HC_SESSION or "default">.sock."""
    runtime = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    session = os.environ.get("HC_SESSION") or "default"
    return str(pathlib.Path(runtime) / "herdr-core" / f"{session}.sock")
