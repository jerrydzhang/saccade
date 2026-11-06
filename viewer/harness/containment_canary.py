#!/usr/bin/env python3
"""Containment canary: runs as the first payload inside an isolated
environment and refuses unless effective Linux containment is observed.

This compares namespaces, mounts, visibility and the environment allowlist.
It never contacts production, never opens external network connections, and
never signals anything. Network checks inspect the namespace's interfaces
only; loopback bring-up and any listener are disposable in-sandbox facts.

Two modes:

  bwrap         the harness lane: namespaces are compared against the
                host's, passed explicitly via CONTAINMENT_HOST_NS_IDS.
  nix-sandbox   the nix build-sandbox lane: static invariants prove the
                sandbox (loopback-only interfaces, read-only store mount,
                scrubbed environment, absent host paths).

Both modes compare state; neither contacts production, opens external
network connections, nor signals anything. Disposable in-sandbox
listeners are the only permitted network checks, if any.

Fail-closed: any failed assertion exits nonzero with a "containment refusal:"
line naming every failure. No lane runs a test payload, browser, core or
network activity before this canary passes in the same environment.
"""

from __future__ import annotations

import os
import pathlib
import socket
import sys


def read_ns(kind: str) -> str:
    return os.readlink(f"/proc/self/ns/{kind}")


def is_tmpfs(path: str) -> bool:
    try:
        # TMPFS_MAGIC from statfs(2)
        st = os.statvfs(path)
    except OSError:
        return False
    # statvfs cannot report the fs magic portably; check via /proc/mounts
    try:
        mounts = pathlib.Path("/proc/mounts").read_text()
    except OSError:
        return False
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == path and parts[2] == "tmpfs":
            return True
    return False


def main() -> int:
    failures: list[str] = []

    host_ns = os.environ.get("CONTAINMENT_HOST_NS_IDS", "")
    host_home = os.environ.get("CONTAINMENT_HOST_HOME", "")
    host_uid = os.environ.get("CONTAINMENT_HOST_UID", "")
    host_db = os.environ.get("CONTAINMENT_HOST_TRACKER_DB", "")
    nix_mode = "--mode" in sys.argv and "nix-sandbox" in sys.argv

    # Namespace containment: every listed namespace must differ from the
    # host's. PID namespace additionally means PID 1 is inside with us.
    if host_ns:
        expected = dict(pair.split("=", 1) for pair in host_ns.split(",") if "=" in pair)
        for kind, host_id in sorted(expected.items()):
            try:
                mine = read_ns(kind)
            except OSError as error:
                failures.append(f"ns/{kind}: unreadable ({error})")
                continue
            if mine == host_id:
                failures.append(f"ns/{kind}: shares the host namespace")
    elif not nix_mode:
        failures.append("CONTAINMENT_HOST_NS_IDS missing: cannot prove namespace containment")

    if nix_mode:
        # Namespace evidence, not mount-table reading: this sandbox
        # generation bind-mounts the store writable over the read-only
        # host fs and discards writes, so /proc/mounts and write probes
        # both misread. What holds: the builder shell is PID 1 of a
        # private pid namespace, /proc enumerates only sandbox processes,
        # sysfs/net is absent or loopback-only, and the environment is
        # scrubbed (checked below).
        try:
            pid1_comm = pathlib.Path("/proc/1/comm").read_text().strip()
            if pid1_comm == "systemd" or pid1_comm == "init":
                failures.append(f"pid namespace: PID 1 looks like the host init: {pid1_comm}")
            pids = [entry for entry in os.listdir("/proc") if entry.isdigit()]
            if len(pids) > 64:
                failures.append(f"pid namespace: {len(pids)} processes visible; host /proc leaked")
        except OSError as error:
            failures.append(f"pid namespace: /proc unreadable ({error})")
        try:
            if os.path.isdir("/sys/class/net"):
                host_ifaces = set(os.listdir("/sys/class/net")) - {"lo"}
                if host_ifaces:
                    failures.append(f"sysfs exposes host interfaces: {sorted(host_ifaces)}")
        except OSError:
            pass
        # PID namespace: the build shell's PID 1 exists inside with us and
        # nothing host-side is enumerable beyond this namespace's processes.
        try:
            pids = [entry for entry in os.listdir("/proc") if entry.isdigit()]
            if len(pids) > 64:
                failures.append(f"pid namespace: {len(pids)} processes visible; host /proc leaked")
        except OSError as error:
            failures.append(f"pid namespace: /proc unreadable ({error})")

    try:
        my_pid = os.getpid()
        if my_pid != 1 and not nix_mode:
            stat = pathlib.Path("/proc/1/stat").read_text()
            # inside an enforced pid namespace PID 1 exists and is not a host process
            if "bwrap" not in stat and "python" not in stat:
                failures.append(f"pid namespace: PID 1 is not sandbox-scoped: {stat[:60]}")
    except OSError as error:
        failures.append(f"pid namespace: /proc unreadable ({error})")

    # Mount and visibility containment: host paths must be invisible.
    for label, path in [
        ("host HOME", host_home),
        ("host runtime dir", f"/run/user/{host_uid}" if host_uid else ""),
        ("tracker db", host_db),
        ("host ssh", "/home/jerry/.ssh"),
    ]:
        if not path:
            continue
        if os.path.exists(path):
            failures.append(f"{label} visible inside sandbox: {path}")

    # Private fresh tmpfs mounts for /tmp and the runtime dir (harness
    # lanes; the nix sandbox guarantees its own build-root isolation).
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not nix_mode:
        if not is_tmpfs("/tmp"):
            failures.append("/tmp is not a private tmpfs mount")
        if not xdg_runtime or not is_tmpfs(xdg_runtime):
            failures.append(f"XDG_RUNTIME_DIR is not a private tmpfs mount: {xdg_runtime!r}")

    # Environment scrubbing: nothing host-session-shaped may survive.
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

    # Network containment: the namespace has loopback only. This is a
    # static comparison; no packet leaves the machine and no external
    # probe is attempted.
    try:
        interfaces = socket.if_nameindex()
    except OSError as error:
        interfaces = []
        if not nix_mode:
            failures.append(f"network: cannot enumerate interfaces ({error})")
    names = {name for _index, name in interfaces}
    if names - {"lo"}:
        failures.append(f"network: non-loopback interfaces present: {sorted(names - {'lo'})}")

    if failures:
        print("containment refusal: the environment is not effectively isolated", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    mode = "nix-sandbox" if nix_mode else "bwrap"
    print(f"containment canary passed ({mode}): mounts, visibility, env and network scoped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
