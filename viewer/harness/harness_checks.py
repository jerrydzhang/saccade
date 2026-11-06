#!/usr/bin/env python3
"""Pure, no-spawn contract checks for the harness itself: command
assembly, baseline validation, preflight argument resolution, and
supervision API availability. Run as part of check-fast; spawns nothing.

These exist so imports and integration mistakes are caught by the
ordinary fast loop instead of the next review round.
"""

from __future__ import annotations

import io
import os
import pathlib
import signal
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import containment_canary  # noqa: E402
import entry  # noqa: E402
import isolation  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"[harness-checks] ok: {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"[harness-checks] FAIL: {name} {detail}", file=sys.stderr)


def with_env(patch: dict[str, str]) -> None:
    for key, value in patch.items():
        os.environ[key] = value


def main() -> int:
    # --- command assembly -------------------------------------------------
    run_root = pathlib.Path(tempfile.mkdtemp(prefix="harness-checks-"))
    cmd = isolation.build_command("/abs/bwrap", run_root, ["/abs/python", "/canary/c.py"], [("/abs/input", "/input")])
    text = " ".join(cmd)
    check("command starts with absolute bwrap", cmd[0] == "/abs/bwrap", text)
    check("command unshares everything", "--unshare-all" in cmd, text)
    check("command dies with parent", "--die-with-parent" in cmd, text)
    check("command has private proc", cmd.index("--proc") >= 0 and cmd[cmd.index("--proc") + 1] == "/proc", text)
    check("command mounts fresh run", cmd.index("--tmpfs") >= 0, text)
    check("store is read-only bound", cmd.index("--ro-bind", cmd.index("--ro-bind") + 1) > 0, text)
    # ordered mounts: the runtime tmpfs must come after the run-root bind
    check(
        "runtime tmpfs mounts after the run-root bind",
        cmd.index("--perms", cmd.index("--bind")) > cmd.index("--bind")
        and cmd[cmd.index("--perms") + 1 : cmd.index("--perms") + 3] == ["0700", "--tmpfs"],
        text,
    )
    check(
        "runtime tmpfs targets the run-root descendant",
        cmd[cmd.index("--perms") + 3] == str(run_root / "runtime"),
        text,
    )
    check("lane inputs bind read-only", ["--ro-bind", "/abs/input", "/input"] == cmd[cmd.index("--ro-bind", cmd.index("--ro-bind") + 1) :][:3], text)
    # exact payload tail: what the preflight lane assembles, verbatim
    payload_cmd = isolation.build_command(
        "/abs/bwrap",
        run_root,
        ["/abs/python", "/canary/containment_canary.py"],
        [("/abs/canary", "/canary/containment_canary.py")],
    )
    check(
        "preflight payload tail is exactly the pinned interpreter and the bound canary",
        payload_cmd[payload_cmd.index("--") + 1 :] == ["/abs/python", "/canary/containment_canary.py"],
        " ".join(payload_cmd),
    )
    check(
        "the canary bind is the only lane ro-bind",
        payload_cmd.count("--ro-bind") == 2,
        " ".join(payload_cmd),
    )
    env = isolation.env_allowlist(run_root)
    expected_env = {"PATH", "LANG", "HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR"}
    check("env allowlist is exactly the private roots", set(env) == expected_env, str(sorted(env)))
    check("env home is inside the run root", env["HOME"].startswith(str(run_root)), str(env))

    # --- baseline validation ---------------------------------------------
    good_ns = ",".join(
        f"{kind}={kind}:[4026531836]" for kind in ("pid", "net", "uts", "ipc", "mnt", "user")
    )
    with_env(
        {
            "CONTAINMENT_HOST_NS_IDS": good_ns,
            "CONTAINMENT_HOST_HOME": "/home/someone",
            "CONTAINMENT_HOST_UID": "1000",
            "CONTAINMENT_HOST_TRACKER_DB": "/repos/x/saccade.db",
        }
    )
    expected, failures = containment_canary.parse_baselines()
    check("a complete numeric-uid baseline set parses", not failures and len(expected) == 6, str(failures))

    with_env({"CONTAINMENT_HOST_NS_IDS": good_ns.replace("pid=pid:[4026531836]", "pid=")})
    expected, failures = containment_canary.parse_baselines()
    check("empty namespace value is refused", any("pid=" in failure for failure in failures), str(failures))
    check("empty namespace value leaves the kind missing", "pid" not in expected, str(expected))

    with_env({"CONTAINMENT_HOST_NS_IDS": good_ns.replace("pid=pid:[4026531836]", "pid=garbage")})
    expected, failures = containment_canary.parse_baselines()
    check("garbage namespace value is refused", any("kernel readlink form" in failure for failure in failures), str(failures))
    check("garbage namespace value leaves the kind missing", "pid" not in expected, str(expected))

    with_env(
        {
            "CONTAINMENT_HOST_NS_IDS": "pid=pid:[1],net=net:[2]",
            "CONTAINMENT_HOST_HOME": "/home/someone",
            "CONTAINMENT_HOST_UID": "1000",
            "CONTAINMENT_HOST_TRACKER_DB": "/repos/x/saccade.db",
        }
    )
    expected, failures = containment_canary.parse_baselines()
    check("partial namespace baselines are refused", any("refusing to compare partially" in failure for failure in failures), str(failures))

    with_env(
        {
            "CONTAINMENT_HOST_NS_IDS": good_ns,
            "CONTAINMENT_HOST_HOME": "/home/someone",
            "CONTAINMENT_HOST_UID": "jerry",
            "CONTAINMENT_HOST_TRACKER_DB": "/repos/x/saccade.db",
        }
    )
    expected, failures = containment_canary.parse_baselines()
    check("non-numeric uid is refused", any("numeric uid" in failure for failure in failures), str(failures))

    with_env(
        {
            "CONTAINMENT_HOST_NS_IDS": good_ns,
            "CONTAINMENT_HOST_HOME": "relative/home",
            "CONTAINMENT_HOST_UID": "1000",
            "CONTAINMENT_HOST_TRACKER_DB": "/repos/x/saccade.db",
        }
    )
    expected, failures = containment_canary.parse_baselines()
    check("relative HOME baseline is refused", any("absolute path" in failure for failure in failures), str(failures))

    with_env({"CONTAINMENT_HOST_NS_IDS": "", "CONTAINMENT_HOST_HOME": "", "CONTAINMENT_HOST_UID": "", "CONTAINMENT_HOST_TRACKER_DB": ""})
    expected, failures = containment_canary.parse_baselines()
    check("all-empty baselines refuse every required entry", len(failures) >= 9, str(failures))

    # --- preflight argument resolution (pure) -----------------------------
    path, refusal = entry.caller_tracker_db("relative/board.db")
    check("relative tracker-db candidate is refused as given", path is None and "absolute path as given" in refusal, refusal)
    path, refusal = entry.caller_tracker_db("relative/a.db", "/abs/real/board.db")
    check("a relative candidate refuses outright, never falls through", path is None and "absolute path as given" in refusal, refusal)
    path, refusal = entry.caller_tracker_db("", "")
    check("no candidates is a refusal", path is None and refusal != "", refusal)

    # --- supervision API availability (no spawn) --------------------------
    usable, reason = isolation.supervision_apis_usable()
    check("supervision APIs are usable on this host", usable, reason)
    try:
        isolation.static_preflight("/definitely/not/bwrap")
        check("static preflight refuses a missing bwrap", False, "no refusal raised")
    except isolation.ContainmentRefused:
        check("static preflight refuses a missing bwrap", True)

    # --- mocked supervisor call flow (no spawn) ---------------------------
    class FakePopen:
        instances: list["FakePopen"] = []

        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.kwargs = kwargs
            self.pid = 4242
            self.stdout = io.StringIO("")
            self.wait_calls = 0
            self.poll_results = [None, 0]
            self.always_running = False
            FakePopen.instances.append(self)

        def poll(self):
            if self.always_running:
                return None
            return self.poll_results.pop(0) if self.poll_results else 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

    real_popen = subprocess.Popen
    real_pidfd_open = os.pidfd_open
    real_write = os.write
    real_close = os.close
    real_send = signal.pidfd_send_signal
    subprocess.Popen = FakePopen  # type: ignore[assignment]
    pidfd_calls: list[int] = []
    writes: list[bytes] = []
    closes: list[int] = []
    sends: list[tuple[int, int]] = []
    os.pidfd_open = lambda pid, flags=0: pidfd_calls.append(pid) or 999  # type: ignore[assignment]
    os.write = lambda fd, data: writes.append(data)  # type: ignore[assignment]
    os.close = lambda fd: closes.append(fd)  # type: ignore[assignment]
    signal.pidfd_send_signal = lambda fd, sig: sends.append((fd, int(sig)))  # type: ignore[assignment]
    try:
        fresh_env = isolation.env_allowlist(run_root)
        supervisor = isolation.PidfdSupervisor(["/abs/bwrap", "--version"], dict(fresh_env))
        supervisor.start()
        spawned = FakePopen.instances[-1]
        check("gate argv wraps the sandbox command via the pinned interpreter", spawned.argv[:4] == [sys.executable, "-I", "-S", str(pathlib.Path(isolation.__file__).resolve())], " ".join(map(str, spawned.argv[:6])))
        check("gate carries internal-gate, fd, token and separator", spawned.argv[4:8] == ["internal-gate", str(spawned.kwargs["pass_fds"][0]), isolation.RELEASE_BYTE.hex(), "--"], str(spawned.argv[4:9]))
        check("gate tail is exactly the sandbox command", spawned.argv[8:] == ["/abs/bwrap", "--version"], str(spawned.argv[8:]))
        check("only the read end is inherited (pass_fds)", len(spawned.kwargs["pass_fds"]) == 1, str(spawned.kwargs.get("pass_fds")))
        check("no preexec_fn blocks the exec", "preexec_fn" not in spawned.kwargs, str(sorted(spawned.kwargs)))
        check("pidfd acquired once for the spawned pid", pidfd_calls == [4242], str(pidfd_calls))
        check("exactly one release byte written", writes == [isolation.RELEASE_BYTE], str(writes))
        check("release write end closed after release", any(True for _ in closes), str(closes))
        env_keys = set(spawned.kwargs["env"])
        check(
            "the chain env is exactly the fresh allowlist: no inherited host keys",
            env_keys == set(fresh_env)
            and not any(key.startswith(("SACCADE_", "HERDR_", "SSH_")) for key in env_keys),
            str(sorted(env_keys)),
        )

        check("an already-exited supervisor gets no signals", sends == [], str(sends))
        # a still-running supervisor gets the escalation: fresh supervisor,
        # poll always running
        sends.clear()
        supervisor_running = isolation.PidfdSupervisor(["/abs/bwrap", "--version"], dict(fresh_env))
        supervisor_running.start()
        FakePopen.instances[-1].always_running = True
        supervisor_running.stop(grace=0.05)
        check("stop closes the pidfd after verified reaping", 999 in closes, str(closes))

        # acquisition failure: nothing released, nothing written, refused
        FakePopen.instances.clear()
        writes.clear()
        os.pidfd_open = lambda pid, flags=0: (_ for _ in ()).throw(OSError(3, "no such process"))  # type: ignore[assignment]
        supervisor2 = isolation.PidfdSupervisor(["/abs/bwrap", "--version"], {})
        try:
            supervisor2.start()
            check("pidfd acquisition failure raises a refusal", False, "no refusal raised")
        except isolation.ContainmentRefused:
            check("pidfd acquisition failure raises a refusal", True)
        check("acquisition failure writes no release byte", writes == [], str(writes))
        check("acquisition failure reaps the gated child", FakePopen.instances[-1].wait_calls == 1, str(FakePopen.instances[-1].wait_calls))
    finally:
        subprocess.Popen = real_popen  # type: ignore[assignment]
        os.pidfd_open = real_pidfd_open  # type: ignore[assignment]
        os.write = real_write  # type: ignore[assignment]
        os.close = real_close  # type: ignore[assignment]
        signal.pidfd_send_signal = real_send  # type: ignore[assignment]

    if FAILURES:
        print(f"[harness-checks] {len(FAILURES)} failure(s)", file=sys.stderr)
        return 1
    print("[harness-checks] all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
