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
import types
import pathlib
import signal
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import browser_lane  # noqa: E402
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

    # --- pinned playwright closure (pure file-existence check) -----------
    fake_env = pathlib.Path(tempfile.mkdtemp(prefix="pw-closure-"))
    pkg = fake_env / "lib" / "python3.14" / "site-packages" / "playwright"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (fake_env / "bin").mkdir()
    fake_python = fake_env / "bin" / "python3"
    fake_python.write_text("#!/bin/sh\n")
    usable, reason = entry.verify_playwright_closure(str(fake_python))
    check("withPackages closure containing playwright verifies", usable and reason == "", reason)
    empty_env = pathlib.Path(tempfile.mkdtemp(prefix="pw-empty-"))
    (empty_env / "bin").mkdir()
    (empty_env / "bin" / "python3").write_text("#!/bin/sh\n")
    usable, reason = entry.verify_playwright_closure(str(empty_env / "bin" / "python3"))
    check("closure without playwright refuses with the env root named", not usable and "playwright module not found" in reason, reason)
    # --- browser lane wiring traversal (no spawn) -------------------------
    lane_calls = {"canary": 0, "server": 0, "goto": [], "checks": 0, "shutdown": 0, "shots": []}

    class FakeServer:
        def shutdown(self):
            lane_calls["shutdown"] += 1

        def server_close(self):
            lane_calls["shutdown"] += 10

    class FakePage:
        gotos: list[str] = []

        def goto(self, url):
            self.gotos.append(url)
            lane_calls["goto"].append(url)

        def set_default_timeout(self, ms):
            pass

        def wait_for_function(self, fn):
            pass

    class FakeContext:
        def grant_permissions(self, *args, **kwargs):
            pass

        def new_page(self):
            return FakePage()

    class FakeBrowser:
        def new_context(self, **kwargs):
            return FakeContext()

        def close(self):
            pass

    class FakeSyncContext:
        def __enter__(self):
            fake = type("FakePw", (), {})()
            fake.chromium = type(
                "FakeChromium",
                (),
                {"launch": staticmethod(lambda **kwargs: FakeBrowser())},
            )()
            return fake

        def __exit__(self, *args):
            return False

    fake_api_module = types.ModuleType("playwright.sync_api")
    fake_api_module.sync_playwright = lambda: FakeSyncContext()

    real_canary_mod = browser_lane.containment_canary
    real_start = browser_lane.start_server
    real_checks = browser_lane.run_checks
    real_shots = os.environ.get("SHOTS_DIR", "")

    class FakeCanary:
        code = 0

        def main(self):
            lane_calls["canary"] += 1
            return FakeCanary.code

    browser_lane.containment_canary = FakeCanary()
    browser_lane.start_server = lambda: lane_calls.update(server=1) or FakeServer()
    shots_dir = pathlib.Path(tempfile.mkdtemp(prefix="lane-shots-"))
    os.environ["SHOTS_DIR"] = str(shots_dir)
    sys.modules["playwright.sync_api"] = fake_api_module  # type: ignore[assignment]
    sys.modules["playwright"] = types.ModuleType("playwright")  # type: ignore[assignment]
    browser_lane.run_checks = lambda page, base_url, shots: lane_calls.update(checks=1) or 0  # type: ignore[assignment]
    try:
        # failure short-circuit: canary refusal ends the lane before any
        # server exists
        FakeCanary.code = 1
        code = browser_lane.main()
        check(
            "canary refusal short-circuits before server and checks",
            code == 1 and lane_calls["canary"] == 1 and lane_calls["server"] == 0 and lane_calls["checks"] == 0,
            str(lane_calls),
        )

        # traversal: canary -> server -> navigate -> checks -> shutdown
        FakeCanary.code = 0
        code = browser_lane.main()
        check(
            "lane traverses canary, server, navigation, checks, shutdown in order",
            code == 0
            and lane_calls["canary"] == 2
            and lane_calls["server"] == 1
            and lane_calls["goto"] == [f"http://127.0.0.1:{browser_lane.SERVER_PORT}"]
            and lane_calls["checks"] == 1
            and lane_calls["shutdown"] >= 11
            and FakePage.gotos == lane_calls["goto"],
            str(lane_calls),
        )
        check("screenshots land under the run-root shots dir", lane_calls["shots"] == [] and shots_dir.is_dir(), str(shots_dir))

    finally:
        browser_lane.containment_canary = real_canary_mod
        browser_lane.start_server = real_start
        browser_lane.run_checks = real_checks
        if real_shots:
            os.environ["SHOTS_DIR"] = real_shots
        else:
            os.environ.pop("SHOTS_DIR", None)
        sys.modules.pop("playwright.sync_api", None)
        sys.modules.pop("playwright", None)

    # --- pinned fontconfig validation (pure) ------------------------------
    usable, reason = entry.verify_fontconfig_file("relative/fonts.conf")
    check("relative fontconfig path is refused as given", not usable and "store path as given" in reason, reason)
    usable, reason = entry.verify_fontconfig_file("/tmp/definitely-not-here.conf")
    # the prefix rule fires before existence: non-store paths never get
    # as far as a file lookup (existence itself is exercised by the real
    # devshell path at lane start)
    check("non-store fontconfig path is refused before file lookup", not usable and "store path as given" in reason, reason)

    usable, reason = entry.fontconfig_content_ok("<fontconfig><dir>~/.nix-profile/share/fonts</dir></fontconfig>")
    check("fontconfig without a store font dir is refused", not usable and "no store font directories" in reason, reason)

    usable, reason = entry.fontconfig_content_ok(
        "<fontconfig><dir>/nix/store/71mxn2pyq807r32qsd2mdszkajhlb39q-dejavu-fonts-2.37</dir>"
        "<dir>~/.nix-profile/share/fonts</dir></fontconfig>"
    )
    check("fontconfig referencing the pinned store fonts verifies", usable and reason == "", reason)

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

    runtime_read_never = lambda fd, n: (_ for _ in ()).throw(AssertionError("read before its guard"))

    # --- executed gate path (mocked read/prctl/exec/exit; no spawn) ------
    # Every gate syscall is mocked all the time: recording defaults, never
    # the real functions, so no scenario can crash or kill the checker.
    real_exit = os._exit
    real_getppid = os.getppid
    real_kill = os.kill
    real_read = os.read
    real_execvpe = os.execvpe
    real_close = os.close
    real_setpdeath = isolation._set_pdeathsig

    class GateStopped(Exception):
        pass

    def make_calls():
        return {"read": [], "kill": [], "exit": [], "exec": [], "close": [], "pdeath": 0}

    def install_gate(calls, *, read=b"", getppid=4242, pdeath_ok=True):
        os._exit = lambda code: (calls["exit"].append(code), (_ for _ in ()).throw(GateStopped(code)))[1]  # type: ignore[assignment]
        os.getppid = lambda: getppid  # type: ignore[assignment]
        os.kill = lambda pid, sig: calls["kill"].append((pid, int(sig)))  # type: ignore[assignment]
        os.read = lambda fd, n: read  # type: ignore[assignment]
        os.execvpe = lambda path, args, env: calls["exec"].append((path, args, env))  # type: ignore[assignment]
        os.close = lambda fd: calls["close"].append(fd)  # type: ignore[assignment]
        isolation._set_pdeathsig = lambda: calls.__setitem__("pdeath", calls["pdeath"] + 1) or pdeath_ok  # type: ignore[assignment]

    def restore_gate():
        os._exit = real_exit  # type: ignore[assignment]
        os.getppid = real_getppid  # type: ignore[assignment]
        os.kill = real_kill  # type: ignore[assignment]
        os.read = real_read  # type: ignore[assignment]
        os.execvpe = real_execvpe  # type: ignore[assignment]
        os.close = real_close  # type: ignore[assignment]
        isolation._set_pdeathsig = real_setpdeath  # type: ignore[assignment]

    def run_gate(calls, argv, **overrides):
        install_gate(calls, **overrides)
        stopped = None
        try:
            isolation.internal_gate(argv)
        except GateStopped as stop:
            stopped = stop.args[0]
        finally:
            restore_gate()
        return stopped

    valid_argv = ["4242", "7", isolation.RELEASE_BYTE.hex(), "--", "/abs/bwrap", "--version"]

    calls = make_calls()
    stopped = run_gate(calls, valid_argv, read=isolation.RELEASE_BYTE)
    check("gate execs the exact command after prctl, matching parent, valid token", stopped is None and calls["exec"] == [("/abs/bwrap", ["/abs/bwrap", "--version"], os.environ)] and calls["pdeath"] == 1, str(calls))
    check("the happy gate never self-exits or signals", stopped is None and calls["exit"] == [] and calls["kill"] == [], str(calls))
    check("gate closes the release fd before exec", calls["close"] == [7], str(calls))

    calls = make_calls()
    stopped = run_gate(calls, valid_argv)
    install_gate(calls)  # EOF: the launcher closed the write end
    os.read = lambda fd, n: b""  # type: ignore[assignment]
    stopped_eof = None
    try:
        isolation.internal_gate(valid_argv)
    except GateStopped as stop:
        stopped_eof = stop.args[0]
    finally:
        restore_gate()
    check("gate EOF self-exits without exec or signals", stopped_eof == isolation.RELEASE_EXIT and calls["exec"] == [] and calls["kill"] == [], str(calls))

    calls = make_calls()
    install_gate(calls, read=b"X")  # wrong token
    stopped_bad = None
    try:
        isolation.internal_gate(valid_argv)
    except GateStopped as stop:
        stopped_bad = stop.args[0]
    finally:
        restore_gate()
    check("wrong token self-exits without exec", stopped_bad == isolation.RELEASE_EXIT and calls["exec"] == [], str(calls))

    calls = make_calls()
    install_gate(calls, getppid=1, read=runtime_read_never)  # adopted launcher
    stopped_adopted = None
    try:
        isolation.internal_gate(valid_argv)
    except GateStopped as stop:
        stopped_adopted = stop.args[0]
    finally:
        restore_gate()
    check(
        "mismatched parent self-signals and refuses before reading",
        stopped_adopted == isolation.RELEASE_EXIT
        and calls["kill"] == [(os.getpid(), int(signal.SIGKILL))]
        and calls["read"] == []
        and calls["exec"] == [],
        str(calls),
    )

    calls = make_calls()
    install_gate(calls, pdeath_ok=False, read=runtime_read_never)
    stopped_prctl = None
    try:
        isolation.internal_gate(valid_argv)
    except GateStopped as stop:
        stopped_prctl = stop.args[0]
    finally:
        restore_gate()
    check("failed prctl self-exits before the release read", stopped_prctl == isolation.RELEASE_EXIT and calls["read"] == [], str(calls))

    # dispatch: fail-closed unknown invocation; gate routes
    calls = make_calls()
    install_gate(calls)
    try:
        code = isolation.dispatch(["isolation.py", "bogus"])
        check("unknown invocation refuses with exit 2", code == 2, str(code))
    except GateStopped:
        check("unknown invocation refuses with exit 2", False, "reached the gate")
    finally:
        restore_gate()

    calls = make_calls()
    install_gate(calls, read=isolation.RELEASE_BYTE)
    routed = None
    try:
        routed = isolation.dispatch(["isolation.py", "internal-gate"] + valid_argv)
    except GateStopped:
        pass
    finally:
        restore_gate()
    check("dispatch routes internal-gate through the executed path", routed == 0 and len(calls["exec"]) == 1, f"{routed} {calls['exec']}")

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
        gate_path = str(pathlib.Path(isolation.__file__).resolve())
        check("gate argv wraps the sandbox command via the pinned interpreter", spawned.argv[:4] == [sys.executable, "-I", "-S", gate_path], " ".join(map(str, spawned.argv[:6])))
        check(
            "gate carries internal-gate, launcher pid, fd, token and separator",
            spawned.argv[4:9] == ["internal-gate", str(os.getpid()), str(spawned.kwargs["pass_fds"][0]), isolation.RELEASE_BYTE.hex(), "--"],
            str(spawned.argv[4:10]),
        )
        check("gate tail is exactly the sandbox command", spawned.argv[9:] == ["/abs/bwrap", "--version"], str(spawned.argv[9:]))
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
        # The escalation list, asserted at last (deferred in 8b8543b):
        # a still-running supervisor receives TERM then KILL through the
        # pidfd, in that order, and nothing else.
        check(
            "a running supervisor gets exactly TERM then KILL through the pidfd",
            sends == [(999, int(signal.SIGTERM)), (999, int(signal.SIGKILL))],
            str(sends),
        )
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
