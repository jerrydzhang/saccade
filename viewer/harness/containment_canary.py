#!/usr/bin/env python3
"""Containment canary: the first payload inside an enforced bwrap sandbox.

It runs read-only enumeration only: it reads /proc, /proc/mounts, /sys and
its own environment, and compares what it finds against explicit baseline
values the harness recorded on the host before the sandbox existed. It
never opens listeners or connections, never signals anything, and never
touches anything outside its own output.

Fail-closed input contract: the harness must supply ALL six host namespace
ids (pid, net, uts, ipc, mnt, user) plus the host HOME, UID and tracker DB
paths. Missing or partial baselines are a refusal, not a softer check.

Any failed assertion exits nonzero with a "containment refusal:" line
naming every failure. No lane runs a test payload, browser, core or
network activity before this canary passes in the same sandbox.
"""

from __future__ import annotations

import os
import pathlib
import sys

REQUIRED_NAMESPACES = ("ipc", "mnt", "net", "pid", "user", "uts")


def is_tmpfs(path: str) -> bool:
    try:
        mounts = pathlib.Path("/proc/mounts").read_text()
    except OSError:
        return False
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == path and parts[2] == "tmpfs":
            return True
    return False


def parse_baselines() -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    raw = os.environ.get("CONTAINMENT_HOST_NS_IDS", "")
    expected: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        kind, value = pair.split("=", 1)
        # An empty value is a missing baseline, not a comparable one:
        # pid= would otherwise silently skip its own namespace check.
        if not kind or not value:
            failures.append(f"baseline malformed or empty: {pair!r} is refused")
            continue
        expected[kind] = value
    for kind in REQUIRED_NAMESPACES:
        if kind not in expected:
            failures.append(f"baseline missing for namespace {kind}: refusing to compare partially")
    for name, var in [
        ("host HOME", "CONTAINMENT_HOST_HOME"),
        ("host UID", "CONTAINMENT_HOST_UID"),
        ("tracker DB", "CONTAINMENT_HOST_TRACKER_DB"),
    ]:
        value = os.environ.get(var)
        if not value:
            failures.append(f"baseline missing: {var} ({name}) is required")
        elif not value.startswith("/"):
            failures.append(f"baseline must be an absolute path: {var}={value!r}")
    return expected, failures


def main() -> int:
    failures: list[str] = []
    expected, input_failures = parse_baselines()
    failures.extend(input_failures)

    host_home = os.environ.get("CONTAINMENT_HOST_HOME", "")
    host_uid = os.environ.get("CONTAINMENT_HOST_UID", "")
    host_db = os.environ.get("CONTAINMENT_HOST_TRACKER_DB", "")

    # Namespace containment: every required namespace must differ from the
    # host's recorded id.
    for kind in REQUIRED_NAMESPACES:
        host_id = expected.get(kind)
        if host_id is None:
            continue  # already reported as missing input
        try:
            mine = os.readlink(f"/proc/self/ns/{kind}")
        except OSError as error:
            failures.append(f"ns/{kind}: unreadable ({error})")
            continue
        if mine == host_id:
            failures.append(f"ns/{kind}: shares the host namespace")

    # PID namespace: the sandbox init is here with us, not the host's.
    try:
        pid1_comm = pathlib.Path("/proc/1/comm").read_text().strip()
        if pid1_comm in ("systemd", "init"):
            failures.append(f"pid namespace: PID 1 looks like the host init: {pid1_comm}")
    except OSError:
        failures.append("pid namespace: /proc/1 unreadable")

    # Mount and visibility containment: host paths must be invisible.
    for label, path in [
        ("host HOME", host_home),
        ("host runtime dir", f"/run/user/{host_uid}" if host_uid else ""),
        ("tracker db", host_db),
    ]:
        if path and os.path.exists(path):
            failures.append(f"{label} visible inside sandbox: {path}")

    # Private fresh tmpfs for /tmp and the runtime dir.
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not is_tmpfs("/tmp"):
        failures.append("/tmp is not a private tmpfs mount")
    if not xdg_runtime or not is_tmpfs(xdg_runtime):
        failures.append(f"XDG_RUNTIME_DIR is not a private tmpfs mount: {xdg_runtime!r}")

    # Environment scrubbing: nothing host-session-shaped may survive. The
    # allowlist builds a fresh dict, so any hit means the chain leaked.
    for var in [
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "DBUS_SESSION_BUS_ADDRESS",
        "XAUTHORITY",
        "WAYLAND_DISPLAY",
        "DISPLAY",
    ]:
        if os.environ.get(var):
            failures.append(f"environment leaked {var}")
    for var in list(os.environ):
        if var.startswith("SACCADE_") or var.startswith("HERDR_"):
            failures.append(f"environment leaked {var}")

    home = os.environ.get("HOME", "")
    if not home or home.startswith("/home/jerry"):
        failures.append(f"HOME is not a private directory: {home!r}")

    # Network containment: loopback only, static enumeration, no packets.
    try:
        import socket

        interfaces = socket.if_nameindex()
        names = {name for _index, name in interfaces}
        if names - {"lo"}:
            failures.append(f"network: non-loopback interfaces present: {sorted(names - {'lo'})}")
    except OSError as error:
        failures.append(f"network: cannot enumerate interfaces ({error})")

    if failures:
        print("containment refusal: the sandbox is not effectively isolated", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("containment canary passed: namespaces, mounts, visibility, env and network scoped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
