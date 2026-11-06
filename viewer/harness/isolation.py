#!/usr/bin/env python3
"""Enforced OS isolation for viewer harness lanes that launch processes.

Contract (t-71, review records 2565, 2613 and the i-64 pinned-source
review of f53330d):

- One supervisor root per run: bwrap. Inner payloads are never signaled
  individually; the private PID namespace tears down with the supervisor.
- Ownership is kernel-verified, never numeric: supervision APIs are
  checked BEFORE the spawn; a child is released into exec only after the
  parent holds its pidfd, through a one-byte pipe gate. If the handle
  cannot be acquired, the child exits itself and nothing runs
  unsupervised. There is no numeric-PID signal fallback anywhere.
- The child's setup (prctl result checked; parent-death re-check that
  signals only itself) happens before it waits on the release gate.
- The exact bwrap flag list lives in build_command(); nothing is added
  at runtime. XDG_RUNTIME_DIR is a fresh 0700 tmpfs mounted AFTER the
  run-root bind (ordered mounts); host HOME is never mounted; /run is
  fresh so host sockets are absent.
- The bwrap process itself is spawned with the same fresh environment,
  so nothing is inherited by anyone in the chain.
- Fail-closed: preflight refuses when bwrap or user namespaces are
  unavailable, or when the pidfd API is missing. There is no weaker
  fallback and no silent downgrade.
- Runtime authorization is procedural: gated lanes refuse unless
  BOARD_VIEWER_RUNTIME_AUTHORIZED names the review record authorizing
  this exact harness code.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

RUNTIME_ROOT = "/var/tmp/saccade-board-viewer"
PR_SET_PDEATHSIG = 1
# One byte: the gate's os.read(fd, 1) can only ever return one byte.
RELEASE_BYTE = b"R"
RELEASE_EXIT = 125

NS_VALUE = re.compile(r"^[a-z_]+:\[\d+\]$")


class ContainmentRefused(Exception):
    pass


def supervision_apis_usable() -> tuple[bool, str]:
    """Refuse unavailable APIs before any spawn. No numeric fallback."""
    if not hasattr(os, "pidfd_open"):
        return False, "os.pidfd_open is unavailable on this interpreter"
    if not hasattr(signal, "pidfd_send_signal"):
        return False, "signal.pidfd_send_signal is unavailable on this interpreter"
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if not hasattr(libc, "prctl"):
            return False, "libc prctl is unavailable"
    except OSError as error:
        return False, f"libc is unavailable: {error}"
    return True, ""


def static_preflight(bwrap: str) -> None:
    """Read-only observations only. The unshare probe is the gated
    preflight and runs solely under the authorization gate."""
    if not os.path.isfile(bwrap):
        raise ContainmentRefused(f"bwrap not found at {bwrap}")
    usable, reason = supervision_apis_usable()
    if not usable:
        raise ContainmentRefused(f"supervision APIs unavailable: {reason}")
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
    these. All six namespaces are required, in kernel readlink form."""
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


def build_command(
    bwrap: str,
    run_root: pathlib.Path,
    payload: list[str],
    ro_binds: list[tuple[str, str]],
) -> list[str]:
    """The exact bwrap invocation. Reviewed spec, verbatim:

    namespaces: --unshare-all (user, pid, net - which also denies the
    host's abstract Unix sockets -, uts, ipc, cgroup). None retained.
    mounts: --proc /proc (private, pidns-scoped); --dev /dev (fresh);
    --tmpfs /dev/shm; --tmpfs /tmp; --tmpfs /run (so the host's
    /run/user/<uid>, DBus and display sockets are absent);
    --ro-bind /nix/store; per-lane explicit read-only binds; read-write
    bind of the run root only; then, AFTER the run-root bind, the
    ordered mount --perms 0700 --tmpfs <runroot>/runtime for
    XDG_RUNTIME_DIR (a later mount covers an earlier one, so the tmpfs
    is never obscured by a descendant bind). Host HOME never mounted.
    caps: the default set inside the private user namespace only, so the
    payload can bring up loopback (CAP_NET_ADMIN scoped to the sandbox's
    own net namespace). No host capability crosses this line.
    lifecycle: --die-with-parent (pdeathsig chain), --new-session (no
    controlling terminal), --chdir <run_root>.
    env: every allowlist variable is --setenv from the fresh dict.
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
    cmd += ["--perms", "0700", "--tmpfs", str(run_root / "runtime")]
    for key, value in env_allowlist(run_root).items():
        cmd += ["--setenv", key, value]
    cmd += ["--"]
    cmd += payload
    return cmd


def _set_pdeathsig() -> bool:
    """Checked prctl: True only when the kernel accepted the setting."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) == 0


def internal_gate(argv: list[str]) -> None:
    """The EXECUTED gate wrapper: the pinned python runs this as its
    program (dispatch at module bottom), so Popen returns the moment
    this starts. Setup, then a blocking one-byte read - only a correct
    token from the ORIGINAL launcher parent (passed in argv, never
    sampled here: a launcher death before setup would otherwise be
    invisible to an adopted child) execs the real command, in this same
    process (same PID), keeping the parent's pidfd valid across the
    exec. EOF or a wrong byte self-exits, and the parent closing the
    write end is what produces that EOF: no numeric signal against
    anyone else, nothing unsupervised, no preexec blocking.

    argv: parent_pid, release_fd, token_hex, "--", command...
    """
    if len(argv) < 5 or argv[3] != "--":
        print("gate usage: internal-gate <parent-pid> <fd> <token-hex> -- <command...>", file=sys.stderr)
        os._exit(RELEASE_EXIT)
    expected_parent = int(argv[0])
    release_fd = int(argv[1])
    token = bytes.fromhex(argv[2])
    command = argv[4:]
    if not command or not _set_pdeathsig():
        os._exit(RELEASE_EXIT)
    if os.getppid() != expected_parent:
        # The launcher died before we could arm pdeathsig: we were
        # adopted, so nothing it promised holds. Signal only ourselves.
        os.kill(os.getpid(), signal.SIGKILL)
        os._exit(RELEASE_EXIT)
    try:
        data = os.read(release_fd, 1)
    except OSError:
        os._exit(RELEASE_EXIT)
    if data != token:
        os._exit(RELEASE_EXIT)
    os.close(release_fd)
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError as error:
        print(f"gate: exec failed: {error}", file=sys.stderr)
        os._exit(RELEASE_EXIT)


class PidfdSupervisor:
    """Holds one kernel-verified handle for the bwrap supervisor.

    The release gate guarantees the child cannot exec before the parent
    holds its pidfd: acquisition failure therefore cannot strand a
    running payload, and no numeric-PID signal exists in this class.
    Reaping is verified by waiting on the real child.
    """

    def __init__(self, cmd: list[str], env: dict[str, str]) -> None:
        self._cmd = cmd
        self._env = env
        self._proc: subprocess.Popen[bytes] | None = None
        self._pidfd: int | None = None
        self._released = False

    def start(self) -> None:
        usable, reason = supervision_apis_usable()
        if not usable:
            raise ContainmentRefused(f"supervision APIs unavailable: {reason}")
        release_r, release_w = os.pipe()
        parent_pid = os.getpid()  # sampled HERE, in the launcher
        # The gate wrapper is EXECUTED (pass_fds carries only the read
        # end), so Popen returns right after exec; the wrapper then
        # blocks on the release read while this parent acquires the
        # pidfd and releases exactly one byte.
        gate_cmd = [
            sys.executable,
            "-I",
            "-S",
            str(pathlib.Path(__file__).resolve()),
            "internal-gate",
            str(parent_pid),
            str(release_r),
            RELEASE_BYTE.hex(),
            "--",
        ] + self._cmd
        try:
            self._proc = subprocess.Popen(
                gate_cmd,
                env=self._env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                pass_fds=(release_r,),
            )
        finally:
            os.close(release_r)  # the child holds the only read end now
        try:
            self._pidfd = os.pidfd_open(self._proc.pid)
        except OSError as error:
            # Closing the write end EOFs the gate and it self-exits:
            # nothing runs unsupervised and no signal is used.
            os.close(release_w)
            self._proc.wait()
            raise ContainmentRefused(f"pidfd acquisition failed ({error}): nothing was allowed to run unsupervised") from error
        os.write(release_w, RELEASE_BYTE)
        os.close(release_w)
        self._released = True

    def read_output(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        return self._proc.stdout.read().decode(errors="replace")

    def wait(self, timeout: float | None = None) -> int:
        assert self._proc is not None
        return self._proc.wait(timeout=timeout)

    def stop(self, grace: float = 5.0) -> int:
        """TERM through the pidfd, bounded grace, then KILL through the
        pidfd; verified reaping before returning. Safe on failure paths:
        without a pidfd there is nothing this class may signal."""
        if self._proc is None:
            return 0
        if self._pidfd is None:
            # Never released (start failed): the child exited itself on
            # the gate; just reap.
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
    # over this path for XDG_RUNTIME_DIR, after the run-root bind.
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


def dispatch(argv: list[str]) -> int:
    """Fail-closed CLI dispatch: this module is a library and its
    internal gate, nothing else. Unknown invocations refuse."""
    if len(argv) >= 2 and argv[1] == "internal-gate":
        internal_gate(argv[2:])
        return 0
    print("isolation.py: unknown invocation; expected 'internal-gate'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(dispatch(sys.argv))
