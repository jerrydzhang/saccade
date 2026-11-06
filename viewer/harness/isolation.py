#!/usr/bin/env python3
"""Enforced OS isolation for viewer harness lanes that launch processes.

Design contract (ratified direction, t-71):

- One supervisor root per run: bwrap. The supervisor is the single handle
  everything hangs from. Inner payloads are never signaled individually;
  killing the supervisor tears down the whole private PID namespace.
- Ownership is kernel-verified, never a numeric PID list: the harness holds
  a pidfd for the supervisor and signals only through it, so PID reuse
  cannot redirect a kill. Reaping is verified by waiting on the actual
  child before declaring a run stopped.
- The exact bwrap flag list lives in build_command(); every flag is listed
  and reviewed, nothing is added at runtime.
- Fail-closed: preflight refuses to run any lane when user namespaces or
  bubblewrap are unavailable. There is no weaker fallback. If a future
  weaker wrapper is ever considered, that is a human decision made on a
  disclosed design, never a silent downgrade.

Runtime authorization is procedural: every lane that launches services
refuses unless BOARD_VIEWER_RUNTIME_AUTHORIZED names the review record
that authorized this exact harness code. The gate is discipline, not
security; the security is the containment itself.
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

RUNTIME_ROOT = "/var/tmp/saccade-board-viewer"


class ContainmentRefused(Exception):
    pass


def static_preflight(bwrap: str) -> None:
    """Read-only observations. These do not execute unshare; the unshare
    probe itself is the gated preflight and runs only after review."""
    if not os.path.isfile(bwrap):
        raise ContainmentRefused(f"bwrap not found at {bwrap}")
    userns_clone = pathlib.Path("/proc/sys/kernel/unprivileged_userns_clone")
    max_user_ns = pathlib.Path("/proc/sys/user/max_user_namespaces")
    for path, minimum in ((userns_clone, 1), (max_user_ns, 1)):
        if not path.exists():
            continue
        value = int(path.read_text().strip() or "0")
        if value < minimum:
            raise ContainmentRefused(
                f"{path} = {value}: user namespaces unavailable; "
                "lane refused (fail-closed, no fallback)"
            )


def host_ns_ids() -> str:
    kinds = ["pid", "net", "uts", "ipc", "mnt", "user"]
    pairs = []
    for kind in kinds:
        try:
            pairs.append(f"{kind}={os.readlink(f'/proc/self/ns/{kind}')}")
        except OSError:
            continue
    return ",".join(pairs)


def build_command(bwrap: str, run_root: pathlib.Path, payload: list[str], ro_binds: list[str]) -> list[str]:
    """The exact bwrap invocation. C3 spec, reviewed before first run:

    namespaces: --unshare-all (user, pid, net (+ abstract unix), uts, ipc,
    cgroup). No host namespace is retained.
    mounts: --proc /proc (private, pidns-scoped), --dev /dev (fresh),
    --tmpfs /dev/shm, --tmpfs /tmp, --tmpfs /run. The host's /run/user/<uid>
    is never mounted (fresh /run tmpfs covers it); host HOME is never
    mounted; display sockets live only in the private XDG_RUNTIME_DIR.
    binds: /nix/store read-only; each lane's explicit inputs read-only;
    the run root read-write. Nothing else is mounted.
    caps: default capability set inside the private user namespace only,
    so the payload can bring up loopback (CAP_NET_ADMIN scoped to the
    sandbox's own net namespace). No host capabilities cross this line.
    lifecycle: --die-with-parent (SIGKILL pdeathsig chain), --new-session
    (no controlling terminal), --chdir <run_root>.
    """
    home = run_root / "home"
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
    env = env_allowlist(run_root)
    for key, value in env.items():
        cmd += ["--setenv", key, value]
    cmd += ["--"]
    cmd += payload
    return cmd


def env_allowlist(run_root: pathlib.Path) -> dict[str, str]:
    """A fresh environment: nothing is inherited. Host session, agent,
    tracker and display variables are absent by construction."""
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


class PidfdSupervisor:
    """Holds one kernel-verified handle for the bwrap supervisor.

    Cleanup signals only through the pidfd (immune to PID reuse), then
    verifies reaping by waiting on the child. Inner processes are never
    signaled: the pid namespace dies with the supervisor (pdeathsig chain),
    so no pattern scans and no PID-list arithmetic ever happen.
    """

    def __init__(self, cmd: list[str], env: dict[str, str]) -> None:
        self._cmd = cmd
        self._env = env
        self._proc: subprocess.Popen[bytes] | None = None
        self._pidfd: int | None = None

    def start(self) -> None:
        def parent_death() -> None:
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

        self._proc = subprocess.Popen(
            self._cmd,
            env=self._env,
            stdout=None,
            stderr=None,
            preexec_fn=parent_death,
        )
        self._pidfd = os.pidfd_open(self._proc.pid)

    def wait(self, timeout: float | None = None) -> int:
        assert self._proc is not None
        return self._proc.wait(timeout=timeout)

    def stop(self, grace: float = 5.0) -> None:
        """TERM through the pidfd, bounded grace, then KILL through the
        pidfd; verify reaping before returning. Safe on failure paths."""
        if self._pidfd is None or self._proc is None:
            return
        if self._proc.poll() is None:
            self._send(signal.SIGTERM)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and self._proc.poll() is None:
                time.sleep(0.05)
            if self._proc.poll() is None:
                self._send(signal.SIGKILL)
        self._proc.wait(timeout=grace)
        os.close(self._pidfd)
        self._pidfd = None
        if self._proc.poll() is None:
            raise RuntimeError("supervisor did not terminate: refusing to continue")

    def _send(self, sig: signal.Signals) -> None:
        assert self._pidfd is not None
        os.pidfd_send_signal(self._pidfd, sig)


def make_run_root(label: str) -> pathlib.Path:
    """A fresh private root per run, with an ownership stamp. An unowned
    existing directory is never adopted."""
    base = pathlib.Path(f"{RUNTIME_ROOT}-{os.getuid()}")
    base.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix=f"{label}-", dir=str(base)))
    (root / "OWNERSHIP").write_text(f"pid={os.getpid()} created={time.time()}\n")
    for sub in ("home", "tmp", "config", "data", "cache", "runtime", "logs"):
        (root / sub).mkdir()
    os.chmod(str(root / "runtime"), 0o700)
    return root


def authorized() -> bool:
    marker = os.environ.get("BOARD_VIEWER_RUNTIME_AUTHORIZED", "")
    return bool(marker)


def gate_or_exit(lane: str) -> None:
    if not authorized():
        print(
            f"containment refusal: lane {lane!r} launches processes; runtime is gated "
            "until the reviewed harness code is authorized "
            "(set BOARD_VIEWER_RUNTIME_AUTHORIZED=<review record>)",
            file=sys.stderr,
        )
        sys.exit(2)
