#!/usr/bin/env python3
"""Pure, no-spawn contract checks for the harness itself: command
assembly, baseline validation, preflight argument resolution, and
supervision API availability. Run as part of check-fast; spawns nothing.

These exist so imports and integration mistakes are caught by the
ordinary fast loop instead of the next review round.
"""

from __future__ import annotations

import io
import json
import os
import socket
import time
import pathlib
import signal
import subprocess
import sys
import tempfile
import urllib.error

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import containment_canary  # noqa: E402
import entry  # noqa: E402
import isolation  # noqa: E402
import webdriver_client  # noqa: E402
import webdriver_session  # noqa: E402

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

    # --- webdriver session tool: pure contracts ---------------------------
    # capabilities: the packaged wrapper is the application and the board
    # path its single argument, verbatim
    caps = webdriver_client.session_capabilities("/nix/store/app/bin/wrapped", ["/board/real.yaml"])
    always = caps["capabilities"]["alwaysMatch"]
    check(
        "session capabilities carry the application and board args verbatim",
        always["tauri:options"] == {"application": "/nix/store/app/bin/wrapped", "args": ["/board/real.yaml"]},
        str(caps),
    )

    # the W3C envelope: value unwrapping and error mapping over a fake
    # transport, never a socket
    real_urlopen = webdriver_client.urllib.request.urlopen

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls: list[tuple[str, str]] = []

    def fake_urlopen_success(req, timeout=None):
        calls.append((req.get_method(), req.full_url))
        if req.get_method() == "POST" and req.full_url.endswith("/session"):
            return FakeResponse({"value": {"sessionId": "sid-1", "capabilities": {}}})
        return FakeResponse({"value": {"ok": True}})

    webdriver_client.urllib.request.urlopen = fake_urlopen_success
    try:
        client = webdriver_client.WebDriver(4444)
        session_id = client.new_session("/app", ["/board/x.yaml"])
        check("session create unwraps the W3C sessionId", session_id == "sid-1" and client.session == "sid-1", str(calls))
        check("session create posts the tauri capabilities", calls[-1] == ("POST", "http://127.0.0.1:4444/session"), str(calls))
    finally:
        webdriver_client.urllib.request.urlopen = real_urlopen

    def fake_urlopen_error(req, timeout=None):
        import io

        error = urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(json.dumps({"value": {"error": "session not created", "message": "no browser"}}).encode("utf-8")))
        raise error

    webdriver_client.urllib.request.urlopen = fake_urlopen_error
    try:
        client = webdriver_client.WebDriver(4444)
        try:
            client.new_session("/app", [])
            check("an error value raises WebDriverError", False, "no refusal raised")
        except webdriver_client.WebDriverError as error:
            check(
                "an error value raises WebDriverError with the W3C fields",
                error.error == "session not created" and error.message == "no browser",
                str(error),
            )
    finally:
        webdriver_client.urllib.request.urlopen = real_urlopen

    # input builders: real codepoints on the wire, pointer and wheel
    # sequences shaped for the actions endpoint
    check("named keys map to the W3C codepoints", webdriver_client.key_codepoint("escape") == chr(0xE00C) and webdriver_client.key_codepoint("return") == chr(0xE006), repr(webdriver_client.key_codepoint("escape")))
    check("single characters pass through as themselves", webdriver_client.key_codepoint("a") == "a", "")
    try:
        webdriver_client.key_codepoint("notakey")
        check("unknown key names refuse", False, "no refusal raised")
    except webdriver_client.WebDriverError:
        check("unknown key names refuse", True)
    keys_actions = webdriver_client.key_press_actions(["Escape", "a"])
    check(
        "key actions are keyDown/keyUp pairs in order",
        keys_actions[0]["type"] == "key"
        and [step["type"] for step in keys_actions[0]["actions"]] == ["keyDown", "keyUp", "keyDown", "keyUp"]
        and keys_actions[0]["actions"][0]["value"] == chr(0xE00C)
        and keys_actions[0]["actions"][2]["value"] == "a",
        str(keys_actions),
    )
    drag = webdriver_client.pointer_drag_actions(100.0, 100.0, 40.0, 20.0, steps=2)
    steps = drag[0]["actions"]
    check(
        "drag actions press at the element and release at the offset",
        steps[1] == {"type": "pointerDown", "button": 0}
        and steps[-1] == {"type": "pointerUp", "button": 0}
        and steps[2]["x"] == 120
        and steps[2]["y"] == 110
        and steps[2]["origin"] == "viewport"
        and steps[3]["x"] == 140
        and steps[3]["y"] == 120,
        str(steps),
    )
    wheel = webdriver_client.wheel_scroll_actions(5.0, 6.0, 0.0, -120.0)
    check(
        "wheel actions scroll at the resolved point",
        wheel[0]["type"] == "wheel" and wheel[0]["actions"][0] == {"type": "scroll", "x": 5, "y": 6, "deltaX": 0, "deltaY": -120, "duration": 80, "origin": "viewport"},
        str(wheel),
    )
    send = webdriver_client.send_keys_value("\\eta")
    check("send keys carry text plus the per-character array", send == {"text": "\\eta", "value": ["\\", "e", "t", "a"]}, str(send))

    # session verb translation: the whole CLI surface compiles to one
    # request vocabulary, refused loudly on junk
    fields, bare = entry.parse_verb_fields(["target=search", "0 < x < 2"], {"target", "text"})
    check("verb fields split keys from bare values", fields == {"target": "search"} and bare == ["0 < x < 2"], f"{fields} {bare}")
    request = entry.build_input_request(["type", "target=search", "0 < \\eta < 2/b"])
    check("type input targets the search control with the text", request == {"cmd": "input", "kind": "type", "text": "0 < \\eta < 2/b", "target": "search"}, str(request))
    request = entry.build_input_request(["click", "target=overlay:zoom-in"])
    check("click input carries the stable target id", request == {"cmd": "input", "kind": "click", "target": "overlay:zoom-in"}, str(request))
    request = entry.build_input_request(["key", "Escape"])
    check("key input carries the named keys", request == {"cmd": "input", "kind": "key", "keys": ["Escape"]}, str(request))
    request = entry.build_input_request(["drag", "target=diagram-stage", "dx=140", "dy=40"])
    check("drag input carries target and offsets", request == {"cmd": "input", "kind": "drag", "target": "diagram-stage", "dx": 140.0, "dy": 40.0}, str(request))
    request = entry.build_input_request(["wheel", "target=diagram-stage", "dy=-120"])
    check("wheel input carries target and deltas", request == {"cmd": "input", "kind": "wheel", "target": "diagram-stage", "dy": -120.0}, str(request))
    request = entry.build_input_request(["coords", "10", "20"])
    check("raw coords is the explicit escape hatch", request == {"cmd": "input", "kind": "coords", "x": 10.0, "y": 20.0}, str(request))

    # session record lifecycle: token-guarded clear, stale records are
    # never alive, foreign tokens never delete
    record_path = pathlib.Path(tempfile.mkdtemp(prefix="session-record-")) / "session.json"
    real_state_path = entry.session_state_path
    entry.session_state_path = lambda: record_path
    real_environ_reader = entry.process_env_value
    try:
        entry.write_session_record({"token": "tok-1", "daemon_pid": 1})
        loaded = entry.load_session_record()
        check("a written session record loads back", loaded is not None and loaded["token"] == "tok-1", str(loaded))
        entry.process_env_value = lambda pid, name: None
        check("a record with dead pids is not alive", entry.record_alive(loaded) is False)
        check("clearing with a foreign token keeps the record", entry.load_session_record() is not None)
        check("clearing with the own token deletes", entry.clear_session_record("tok-1") is True and entry.load_session_record() is None)
        check("clearing an absent record is a clean no", entry.clear_session_record("tok-1") is False)
        entry.process_env_value = lambda pid, name: "tok-1" if name == "SAC_APP_SESSION_TOKEN" else None
        check("a record whose pid carries the token is alive", entry.record_alive({"token": "tok-1", "daemon_pid": 7}) is True)
        check("a record with a foreign token is not alive", entry.record_alive({"token": "zzz", "daemon_pid": 7}) is False)
    finally:
        entry.session_state_path = real_state_path
        entry.process_env_value = real_environ_reader

    # pidfd ownership ordering, exercised against this checker process:
    # the fd is pinned before the token is read, and the alive probe
    # runs on the same fd the signal would use - no PID-reuse window.
    # The environ reader is mocked because /proc/self/environ shows the
    # exec-time block, never later putenv writes.
    real_environ_reader = entry.process_env_value
    entry.process_env_value = (
        lambda pid, name: "tok-self"
        if pid == os.getpid() and name == "SAC_APP_SESSION_TOKEN"
        else None
    )
    try:
        fd = entry.open_verified_pidfd(os.getpid(), "tok-self")
        check("a pinned pidfd verifies its own token", fd is not None)
        if fd is not None:
            os.close(fd)
        check("a foreign token never verifies", entry.open_verified_pidfd(os.getpid(), "other") is None)
        check(
            "a verified signal goes through the held pidfd",
            entry.signal_verified_pid(os.getpid(), "tok-self", signal.SIGCHLD) is True,
        )
        check(
            "a foreign-token signal is refused outright",
            entry.signal_verified_pid(os.getpid(), "other", signal.SIGCHLD) is False,
        )
        check("a gone pid refuses verification outright", entry.signal_verified_pid(4194304 + 100, "tok-self", signal.SIGCHLD) is False)
        check("waiting on a live self through the pidfd times out", entry.wait_pid_gone(os.getpid(), 0.2) is False)
        check("waiting on a gone pid is immediately true", entry.wait_pid_gone(4194304 + 100, 0.2) is True)
    finally:
        entry.process_env_value = real_environ_reader

    # the payload's target addressing: stable data-control ids only
    check(
        "target ids compile to a data-control selector",
        webdriver_session.target_selector("diagram:u12") == '[data-control="diagram:u12"]',
        "",
    )
    try:
        webdriver_session.target_selector('x"[data-leak]')
        check("target ids with selector metacharacters refuse", False, "no refusal raised")
    except webdriver_client.WebDriverError:
        check("target ids with selector metacharacters refuse", True)

    # payload traversal over fakes: canary -> weston -> driver chain ->
    # one session -> serve -> down -> teardown in owned order, with no
    # real process, port or driver anywhere
    real_canary = webdriver_session.containment_canary
    real_start_weston = webdriver_session.start_weston
    real_popen = webdriver_session.subprocess.Popen
    real_wait_listening = webdriver_session.wait_listening
    real_client = webdriver_session.WebDriver
    traversal_env = {
        "WESTON_BIN": "/traversal/weston",
        "SCREENSHOOTER_BIN": "/traversal/shooter",
        "TAURI_DRIVER_BIN": "/traversal/tauri-driver",
        "WEBKIT_DRIVER_BIN": "/traversal/WebKitWebDriver",
        "APP_BIN": "/traversal/app",
        "BOARD_FILE": "/traversal/board.yaml",
        "SHOTS_DIR": str(pathlib.Path(tempfile.mkdtemp(prefix="traversal-shots-"))),
        "SESSION_CONTROL_SOCKET": str(pathlib.Path(tempfile.mkdtemp(prefix="traversal-ctl-")) / "control.sock"),
        "SESSION_READY_FILE": str(pathlib.Path(tempfile.mkdtemp(prefix="traversal-ready-")) / "ready.json"),
        "SESSION_LIFETIME_SECONDS": "10",
    }
    events: list[str] = []

    class TraversalCanary:
        def main(self):
            events.append("canary")
            return 0

    class TraversalHandle:
        def __init__(self, role):
            self.role = role
            self.terminated = False
            self.killed = False

        def poll(self):
            return None if not (self.terminated or self.killed) else 0

        def terminate(self):
            self.terminated = True
            events.append(f"terminate:{self.role}")

        def kill(self):
            self.killed = True
            events.append(f"kill:{self.role}")

        def wait(self, timeout=None):
            return 0

    weston_handle = TraversalHandle("weston")

    def fake_start_weston(_bin, _name, _logs):
        events.append("weston")
        return weston_handle

    def fake_popen(argv, **_kwargs):
        events.append("spawn:" + pathlib.Path(argv[0]).name)
        return TraversalHandle("tauri-driver")

    class TraversalClient:
        def __init__(self, port, timeout=None):
            self.session = ""

        def new_session(self, application, args):
            events.append("session")
            self.session = "traversal-session"
            return self.session

        def delete_session(self):
            events.append("delete-session")
            self.session = ""

        def execute(self, script, args=None):
            return {"resource": {"status": "loaded"}}

    webdriver_session.containment_canary = TraversalCanary()
    webdriver_session.start_weston = fake_start_weston
    webdriver_session.subprocess.Popen = fake_popen
    webdriver_session.wait_listening = lambda port, timeout: events.append("listen") or True
    webdriver_session.WebDriver = TraversalClient
    saved_env = {key: os.environ.get(key) for key in traversal_env}
    os.environ.update(traversal_env)
    try:
        import threading

        outcome: dict = {}

        def run_payload():
            outcome["code"] = webdriver_session.main()

        payload_thread = threading.Thread(target=run_payload, daemon=True)
        payload_thread.start()
        # wait for the control socket, then take the session down the way
        # the host verb does: one JSON request, one response line
        control_path = traversal_env["SESSION_CONTROL_SOCKET"]
        deadline = time.monotonic() + 20.0
        response = None
        while time.monotonic() < deadline and response is None:
            if pathlib.Path(control_path).exists():
                try:
                    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    conn.settimeout(5.0)
                    conn.connect(control_path)
                    conn.sendall(b'{"cmd": "down"}\n')
                    response = conn.makefile("rb").readline()
                    conn.close()
                except OSError:
                    time.sleep(0.2)
            else:
                time.sleep(0.2)
        payload_thread.join(timeout=30.0)
        check("the down request is acknowledged", response is not None and b'"stopping": true' in response, str(response))
        check("payload traversal exits clean on down", outcome.get("code") == 0, str(outcome))
        check(
            "payload traversal order: canary, weston, driver, listen, session",
            events[:5] == ["canary", "weston", "spawn:tauri-driver", "listen", "session"],
            str(events),
        )
        check(
            "teardown owns the chain in dependency order",
            "delete-session" in events
            and events.index("delete-session") < events.index("terminate:tauri-driver")
            and events.index("terminate:tauri-driver") < events.index("terminate:weston"),
            str(events),
        )
        check(
            "payload session checks all pass on the clean traversal",
            webdriver_session.RESULTS and all(ok for ok, _name, _detail in webdriver_session.RESULTS),
            str(webdriver_session.RESULTS),
        )
        control = pathlib.Path(traversal_env["SESSION_CONTROL_SOCKET"])
        check("the control socket is removed at teardown", not control.exists(), str(control))
    finally:
        webdriver_session.containment_canary = real_canary
        webdriver_session.start_weston = real_start_weston
        webdriver_session.subprocess.Popen = real_popen
        webdriver_session.wait_listening = real_wait_listening
        webdriver_session.WebDriver = real_client
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if FAILURES:
        print(f"[harness-checks] {len(FAILURES)} failure(s)", file=sys.stderr)
        return 1
    print("[harness-checks] all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
