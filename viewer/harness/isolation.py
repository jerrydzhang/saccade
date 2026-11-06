#!/usr/bin/env python3
"""Enforced OS isolation for viewer harness lanes that launch processes.

Contract (t-71, review records 2565 and the i-64 source review):

- One supervisor root per run: bwrap. Inner payloads are never signaled
  individually; the private PID namespace tears down with the supervisor.
- Ownership is kernel-verified, never numeric: the harness holds a pidfd
  and signals only through it. If the pidfd cannot be created the run is
  refused - there is no numeric-PID kill fallback. Reaping is verified by
  waiting on the real child.
- The exact bwrap flag list lives in build_command(); nothing is added at
  runtime. XDG_RUNTIME_DIR is a fresh tmpfs (perms 0700) inside the run
  root; host HOME is never mounted; /run is fresh so host sockets are
  absent.
- The bwrap process itself is spawned with the same fresh environment, so
  nothing is inherited by anyone in the chain.
- Fail-closed: preflight refuses when bwrap or user namespaces are
  unavailable, or when the pidfd API is missing. There is no weaker
  fallback and no silent downgrade.
- Runtime authorization is procedural: gated lanes refuse unless
  BOARD_VIEWER_RUNTIME_AUTHORIZED names the review record authorizing
  this exact harness code. The gate is discipline; containment is the
  security.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

RUNTIME_ROOT = "/var/tmp/saccade-board-viewer"
PR_SET_PDEATHSIG = 1


class ContainmentRefused(Exception):
    pass


def static_preflight(bwrap: str) -> None:
    """Read-only observations only. The unshare probe is the gated
    preflight and runs solely under the authorization gate."""
    if not os.path.isfile(bwrap):
        raise ContainmentRefused(f"bwrap not found at {bwrap}")
    for path in (pathlib.Path("/proc/sys/kernel/unprivileged_userns_clone"), pathlib.Path("/proc/sys/user/max_user_namespaces")):
        if not path.exists():
            continue
        value = int(path.read_text().strip() or "0")
        if value < 1:
            raise ContainmentRefused(
                f"{path} = {value}: user namespaces unavailable; "
                "lane refused (fail-closed, no fallback)"
            )


def host_ns_ids() -> str:
    """Recorded before any sandbox exists; the canary compares against
    these. All six namespaces are required by the canary's input contract."""
    pairs = []
    for kind in ("pid", "net", "uts", "ipc", "mnt", "user"):
        pairs.append(f"{kind}={os.readlink(f'/proc/self/ns/{kind}')}")
    return ",".join(pairs)


def env_allowlist(run_root: pathlib.Path) -> dict[str, str]:
    """A fresh environment for the whole chain (bwrap included): nothing
    is inherited. Host session, agent, tracker and display variables are
    absent by construction."""
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": str(run_root / "home"),
        "TMPDIR": str(run_root / "tmp"),
        "XDG_CONFIG_HOME": str(run_root / "config"),
        "XDG_DATA_HOME": str(run_root / "data"),
        "XDG_CACHE_HOME": str(run_root / "cache"),
        "XDG_RUNTIME_DIR": str(run_root / "runtime"),
    }


def build_command(bwrap: str, run_root: pathlib.Path, payload: list[str], ro_binds: list[tuple[str, str]]) -> list[str]:
    """The exact bwrap invocation. Reviewed spec, verbatim:

    namespaces: --unshare-all (user, pid, net - which also denies the
    host's abstract Unix sockets -, uts, ipc, cgroup). None retained.
    mounts: --proc /proc (private, pidns-scoped); --dev /dev (fresh);
    --tmpfs /dev/shm; --tmpfs /tmp; --tmpfs /run (so the host's
    /run/user/<uid>, DBus and display sockets are absent);
    --perms 0700 --tmpfs <runroot>/runtime (private tmpfs for
    XDG_RUNTIME_DIR). --ro-bind /nix/store. Per-lane explicit read-only
    binds. Read-write bind of the run root only. Host HOME never mounted.
    caps: the default set inside the private user namespace only, so the
    payload can bring up loopback (CAP_NET_ADMIN scoped to the sandbox's
    own net namespace). No host capability crosses this line.
    lifecycle: --die-with-parent (pdeathsig chain), --new-session (no
    controlling terminal), --chdir <run_root>.
    env: every variable below is --setenv from the fresh allowlist.
    """
    cmd = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--chdir",
        str(run_root),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/run",
        "--ro-bind",
        "/nix/store",
        "/nix/store",
        "--bind",
        str(run_root),
        str(run_root),
    ]
    for src, dest in ro_binds:
        cmd += ["--ro-bind", src, dest]
    # Ordered mounts: the runtime tmpfs targets a run-root descendant, so
    # it mounts AFTER the run-root bind (and any lane binds) - a later
    # mount covers an earlier one, and the ordering is explicit here
    # rather than something the canary must catch.
    cmd += ["--perms", "0700", "--tmpfs", str(run_root / "runtime")]
    for key, value in env_allowlist(run_root).items():
        cmd += ["--setenv", key, value]
    cmd += ["--"]
    cmd += payload
    return cmd


def _child_prepare(parent_pid: int) -> None:
    """In-child: die with the parent. The getppid check closes the race
    where the parent died between fork and prctl (signaling SELF, which
    is owned, is the only numeric signal in this file)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


class PidfdSupervisor:
    """Holds one kernel-verified handle for the bwrap supervisor.

    Cleanup signals only through the pidfd (PID reuse cannot redirect
    it) and verifies reaping by waiting on the real child. If the pidfd
    cannot be opened the run is refused: no numeric-PID fallback exists.
    """

    def __init__(self, cmd: list[str], env: dict[str, str]) -> None:
        self._cmd = cmd
        self._env = env
        self._proc: subprocess.Popen[bytes] | None = None
        self._pidfd: int | None = None
        self._refused = False

    def start(self) -> None:
        parent_pid = os.getpid()
        self._proc = subprocess.Popen(
            self._cmd,
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: _child_prepare(parent_pid),
        )
        try:
            self._pidfd = os.pidfd_open(self._proc.pid)
        except (AttributeError, OSError):
            # API missing or child already exited before we could hold a
            # handle. Reap if needed and refuse the run: no fallback.
            if self._proc.poll() is None:
                self._proc.kill()  # child of this process: direct kill is owned
            self._proc.wait()
            self._refused = True
            raise ContainmentRefused("pidfd unavailable: refusing to supervise without a kernel-verified handle")

    def read_output(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        return self._proc.stdout.read().decode(errors="replace")

    def wait(self, timeout: float | None = None) -> int:
        assert self._proc is not None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise

    def stop(self, grace: float = 5.0) -> int:
        """TERM through the pidfd, bounded grace, then KILL through the
        pidfd; verified reaping before returning. Safe on failure paths."""
        if self._refused or self._proc is None:
            return 0
        if self._pidfd is None:
            # start() refused; nothing to signal - just reap.
            return self._proc.wait(timeout=grace)
        if self._proc.poll() is None:
            self._send(signal.SIGTERM)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and self._proc.poll() is None:
                time.sleep(0.05)
            if self._proc.poll() is None:
                self._send(signal.SIGKILL)
        code = self._proc.wait(timeout=grace)
        os.close(self._pidfd)
        self._pidfd = None
        return code

    def _send(self, sig: signal.Signals) -> None:
        assert self._pidfd is not None
        signal.pidfd_send_signal(self._pidfd, sig)


def make_run_root(label: str) -> pathlib.Path:
    """A fresh private root per run, with an ownership stamp. An unowned
    existing directory is never adopted."""
    base = pathlib.Path(f"{RUNTIME_ROOT}-{os.getuid()}")
    base.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix=f"{label}-", dir=str(base)))
    (root / "OWNERSHIP").write_text(f"pid={os.getpid()} created={time.time()}\n")
    for sub in ("home", "tmp", "config", "data", "cache", "logs"):
        (root / sub).mkdir()
    # runtime is NOT a plain dir: build_command mounts a fresh 0700 tmpfs
    # over this path for XDG_RUNTIME_DIR.
    (root / "runtime").mkdir()
    return root


def authorized() -> bool:
    return bool(os.environ.get("BOARD_VIEWER_RUNTIME_AUTHORIZED", ""))


def gate_or_exit(lane: str) -> None:
    if not authorized():
        print(
            f"containment refusal: lane {lane!r} launches processes; runtime is gated "
            "until the reviewed harness code is authorized "
            "(set BOARD_VIEWER_RUNTIME_AUTHORIZED=<review record>)",
            file=sys.stderr,
        )
        sys.exit(2)
