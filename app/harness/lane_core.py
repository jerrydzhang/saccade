"""The suite lane's core facts: the pinned herdr-core's socket path,
its readiness, and the machine name it answers to. Pure module, so the
harness's own tests hold these without a sandbox. The payload
(suite_payload.py) owns the spawn and the stop; the facts here are what
it waits on, checks, and logs."""

from __future__ import annotations

import json
import pathlib
import time


class CoreRefused(Exception):
    """The core never came up, or its facts name nothing. The lane
    refuses, never guesses."""


def core_socket_path(runtime_dir: str, session: str) -> pathlib.Path:
    """The socket the app's shell bridge dials: the transport's own
    default, $XDG_RUNTIME_DIR/herdr-core/<session>.sock. A session that
    is empty or carries a path separator is refused before it can name
    another directory - the same rule the transport applies."""
    if not session or "/" in session or "\0" in session:
        raise CoreRefused(f"session {session!r} names no socket")
    if not runtime_dir:
        raise CoreRefused("XDG_RUNTIME_DIR is not set; the core has nowhere to bind")
    return pathlib.Path(runtime_dir) / "herdr-core" / f"{session}.sock"


def core_facts(status_line: str) -> tuple[str, str]:
    """(version, machine) from the core's own status json (the
    `server start` answer carries the same document): the core's name
    is the machine its local execs answer to. A status without both is
    a refusal."""
    try:
        status = json.loads(status_line)
    except json.JSONDecodeError as error:
        raise CoreRefused(f"the core's status is not json: {error}") from error
    if not isinstance(status, dict):
        raise CoreRefused("the core's status is not an object")
    version = status.get("version")
    core = status.get("core")
    machine = core.get("name") if isinstance(core, dict) else None
    if not isinstance(version, str) or not version:
        raise CoreRefused("the core's status carries no version")
    if not isinstance(machine, str) or not machine:
        raise CoreRefused("the core's status carries no machine name")
    return version, machine


def wait_for_socket(socket: pathlib.Path, deadline_seconds: float) -> None:
    """Bounded readiness: the socket exists before the deadline. A
    bounded wait, never an unbounded one; a timeout is a refusal that
    names the socket."""
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if socket.exists():
            return
        time.sleep(0.2)
    raise CoreRefused(f"the core's socket never appeared: {socket}")
