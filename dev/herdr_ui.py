import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


OWNER = "SACCADE_HERDR_UI_OWNER"


class Failure(Exception):
    pass


def wait_for(probe, description, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = probe()
        if result:
            return result
        time.sleep(0.05)
    raise Failure(f"Timed out waiting for {description}; inspect the sandbox logs")


def managed_file(path):
    if path.is_symlink():
        raise Failure(f"Refusing symlink for a managed file: {path}")
    return path


def atomic_json(path, value):
    managed_file(path)
    temporary = managed_file(path.with_suffix(".new"))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def private_directory(path):
    if path.is_symlink():
        raise Failure(f"Refusing symlink directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise Failure(f"Expected an owned, private directory (mode 0700): {path}")


def process_token(pid):
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return next((entry.split(b"=", 1)[1].decode() for entry in data
                     if entry.startswith((OWNER + "=").encode())), None)
    except (OSError, UnicodeError):
        return None


def signal_owned(pid, token, sig):
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if process_token(pid) == token:
            signal.pidfd_send_signal(descriptor, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


class Sandbox:
    def __init__(self, root, app, core, weston, xwayland):
        self.root = root.absolute()
        self.app, self.core, self.weston, self.xwayland = map(str, (app, core, weston, xwayland))
        self.record = self.root / "sandbox.json"
        self.children = []

    @contextlib.contextmanager
    def locked(self):
        private_directory(self.root)
        managed_file(self.record)
        with managed_file(self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.record.exists():
                self.state = json.loads(self.record.read_text())
                if not isinstance(self.state, dict) or self.state.get("root") != str(self.root):
                    raise Failure("Sandbox ownership record does not match its directory")
                token = self.state.get("token")
                if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
                    raise Failure("Invalid sandbox ownership token; refusing process operations")
            else:
                unexpected = set(self.root.iterdir()) - {self.root / "lock"}
                if unexpected:
                    names = ", ".join(sorted(p.name for p in unexpected))
                    raise Failure(f"Refusing to adopt a nonempty directory without a sandbox ownership record: {names}")
                self.state = {"root": str(self.root), "token": uuid.uuid4().hex}
                self.save()
            yield

    def save(self):
        atomic_json(self.record, self.state)

    def environment(self, display=False):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("HC_", "SACCADE_", "HERDR_", "DBUS_"))
               and key not in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "XAUTHORITY",
                               "SSH_AUTH_SOCK", "SSH_AGENT_PID", "SESSION_MANAGER", "GDK_BACKEND")}
        env.update({OWNER: self.state["token"], "HERDR_APP_HOME": str(self.root / "app"),
                    "HC_SESSION": "ui", "HERDR_APP_NO_SOUND": "1", "HERDR_TELEMETRY": "0",
                    "HERDR_UI_OPEN_LOG": str(self.root / "opened.log"), "HOME": str(self.root / "home")})
        for kind in ("CONFIG", "STATE", "CACHE", "DATA", "RUNTIME"):
            env[f"XDG_{kind}_HOME" if kind != "RUNTIME" else "XDG_RUNTIME_DIR"] = str(self.root / kind.lower())
        env["PATH"] = str(self.root / "bin") + os.pathsep + str(Path(self.xwayland).parent) + os.pathsep + env.get("PATH", "")
        if display:
            if not self.state.get("display"):
                raise Failure("Sandbox has no display; run up first")
            env.update(DISPLAY=self.state["display"], WAYLAND_DISPLAY="")
        return env

    def run(self, arguments, display=False, check=True, timeout=30):
        result = subprocess.run(list(map(str, arguments)), env=self.environment(display),
                                text=True, capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise Failure(result.stderr.strip() or result.stdout.strip() or f"Command failed: {arguments[0]}")
        return result

    def owned(self):
        token = self.state["token"]
        return [int(p.name) for p in Path("/proc").iterdir()
                if p.name.isdigit() and process_token(p.name) == token]

    def alive(self, role):
        pid = self.state.get(role)
        return bool(pid and process_token(pid) == self.state["token"])

    def spawn(self, role, arguments, display=False):
        with managed_file(self.root / f"{role}.log").open("w") as output:
            process = subprocess.Popen(list(map(str, arguments)), env=self.environment(display),
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.children.append(process)
        self.state[role] = process.pid
        self.save()
        return process

    def prepare(self):
        for folder in ("app", "home", "config", "state", "cache", "data", "runtime", "bin"):
            private_directory(self.root / folder)
        config = self.root / "config/herdr-core"
        private_directory(config)
        core_config = managed_file(config / "config.toml")
        if not core_config.exists():
            core_config.write_text("notes = false\n")
        link = self.root / "bin/hc"
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise Failure("Sandbox bin/hc is not a managed symlink")
        link.symlink_to(self.core)
        for name in ("xdg-open", "open"):
            opener = managed_file(self.root / "bin" / name)
            opener.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$HERDR_UI_OPEN_LOG"\n')
            opener.chmod(0o700)
        settings_file = managed_file(self.root / "app/settings.json")
        settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
        settings["core"] = self.core
        settings["telemetry"] = False
        settings["sounds"] = {"enabled": False}
        atomic_json(settings_file, settings)
        shared = managed_file(self.root / "app/shared.json")
        if not shared.exists():
            atomic_json(shared, {"frame.welcomed": "skipped"})

    def up(self):
        if all(self.alive(role) for role in ("app_pid", "display_pid", "core_pid")):
            return self.info()
        if self.owned():
            raise Failure("Partial sandbox is still running; run down before up")
        self.prepare()
        self.state.pop("display", None)
        self.save()
        try:
            self.run([self.core, "server", "start"])
            core_status = json.loads(self.run([self.core, "server", "status"]).stdout)
            pid = core_status.get("pid")
            if not pid or process_token(pid) != self.state["token"]:
                raise Failure("Core did not report a sandbox-owned process; refusing to adopt it")
            self.state["core_pid"] = pid
            self.save()
            compositor_log = self.root / "weston.log"
            compositor_log.unlink(missing_ok=True)
            compositor = self.spawn("display_pid", [self.weston, "--backend=headless", "--renderer=pixman",
                "--xwayland", "--socket=herdr-ui", "--width=1280", "--height=900", "--idle-time=0",
                f"--log={compositor_log}"])

            def display_ready():
                if compositor.poll() is not None:
                    raise Failure(f"Compositor exited; inspect {self.root / 'display_pid.log'}")
                if compositor_log.exists():
                    match = re.search(r"xserver listening on display (:[0-9]+)", compositor_log.read_text())
                    if match:
                        return match[1]
                return None
            self.state["display"] = wait_for(display_ready, "the private Xwayland display")
            self.save()
            self.spawn("app_pid", [self.app], display=True)

            def app_ready():
                if not self.alive("app_pid"):
                    raise Failure(f"App exited; inspect {self.root / 'app_pid.log'}")
                result = self.run([self.app, "status"], check=False, timeout=5)
                return result.returncode == 0
            wait_for(app_ready, "the app command socket")
            return self.info()
        except BaseException as startup_error:
            try:
                self.down()
            except Exception as cleanup_error:
                raise Failure(f"Startup failed: {startup_error}; cleanup incomplete: {cleanup_error}") from startup_error
            raise

    def info(self):
        return {"root": str(self.root), "app_home": str(self.root / "app"),
                "display": self.state.get("display"),
                "running": all(self.alive(role) for role in ("app_pid", "display_pid", "core_pid")),
                "logs": str(self.root)}

    def app_command(self, arguments):
        if not self.alive("app_pid"):
            raise Failure("Sandbox app is not running; run up first")
        result = self.run([self.app, *arguments], display=True, check=False)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    def down(self):
        if self.alive("app_pid"):
            with contextlib.suppress(Failure, OSError, subprocess.SubprocessError):
                self.run([self.app, "do", "window.close"], check=False, timeout=5)
            deadline = time.monotonic() + 3
            while self.alive("app_pid") and time.monotonic() < deadline:
                time.sleep(0.05)
        if self.alive("core_pid"):
            with contextlib.suppress(Failure, OSError, subprocess.SubprocessError):
                self.run([self.core, "server", "stop"], check=False, timeout=10)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in self.owned():
                signal_owned(pid, self.state["token"], sig)
            deadline = time.monotonic() + 3
            while self.owned() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not self.owned():
                break
        if self.owned():
            raise Failure("Some sandbox processes remain; ownership data and logs retained")
        for child in self.children:
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired as error:
                raise Failure("A started sandbox child has not exited; retaining ownership data") from error
        self.children.clear()
        for key in ("app_pid", "core_pid", "display_pid", "display"):
            self.state.pop(key, None)
        self.save()
        return {"stopped": True, "files_retained": str(self.root)}

    def input(self, actions):
        if not all(self.alive(role) for role in ("app_pid", "display_pid")):
            raise Failure("Sandbox app/display is not running")
        if not isinstance(actions, list) or not actions:
            raise Failure("Input sequence must be a nonempty list")
        for action in actions:
            if not isinstance(action, list) or not action:
                raise Failure("Each input action must be a nonempty list")
            command, *values = action
            valid = ((command in ("click", "move") and len(values) == 2
                      and all(isinstance(v, (int, float)) for v in values))
                     or (command == "type" and len(values) == 1 and isinstance(values[0], str))
                     or (command == "key" and bool(values) and all(isinstance(v, str) and v for v in values)))
            if not valid:
                raise Failure(f"Invalid input action: {action}")
        from Xlib import X, XK, display, error
        from Xlib.ext import xtest
        try:
            connection = display.Display(self.state["display"])
        except error.DisplayError as failure:
            raise Failure(f"Cannot connect to the private display: {failure}") from failure
        try:
            root = connection.screen().root
            def find(window):
                if window.get_wm_name() == "Herdr":
                    return window
                for child in window.query_tree().children:
                    result = find(child)
                    if result is not None:
                        return result
                return None
            window = find(root)
            if window is None:
                raise Failure("No Herdr window on the sandbox display")
            position = window.translate_coords(root, 0, 0)
            def move(x, y):
                xtest.fake_input(connection, X.MotionNotify, x=int(x - position.x), y=int(y - position.y))
                connection.sync()
            def key(name, modifiers=()):
                names = [*modifiers, name]
                codes = [connection.keysym_to_keycode(XK.string_to_keysym(n)) for n in names]
                if not all(codes):
                    raise Failure(f"Unsupported keysym: {names}")
                for code in codes:
                    xtest.fake_input(connection, X.KeyPress, code)
                for code in reversed(codes):
                    xtest.fake_input(connection, X.KeyRelease, code)
                connection.sync()
                time.sleep(0.03)
            move(0, 0)
            time.sleep(0.15)
            for action in actions:
                command, *args = action
                if command == "click":
                    move(*args)
                    time.sleep(0.2)
                    xtest.fake_input(connection, X.ButtonPress, 1)
                    connection.sync()
                    time.sleep(0.05)
                    xtest.fake_input(connection, X.ButtonRelease, 1)
                    connection.sync()
                elif command == "move":
                    move(*args)
                elif command == "key":
                    key(args[0], args[1:])
                elif command == "type":
                    punctuation = {" ": "space", ".": "period", "/": "slash", ":": "colon", "-": "minus"}
                    for character in args[0]:
                        name = punctuation.get(character, character)
                        keysym = XK.string_to_keysym(name)
                        shifted = any(index % 2 for _, index in connection.keysym_to_keycodes(keysym))
                        key(name, ["Shift_L"] if shifted or character.isupper() else [])
                else:
                    raise Failure(f"Unsupported input action: {command}")
            time.sleep(0.1)
        except (error.XError, error.ConnectionClosedError) as failure:
            raise Failure(f"Private-display input failed: {failure}") from failure
        finally:
            with contextlib.suppress(error.ConnectionClosedError):
                connection.close()


def main():
    parser = argparse.ArgumentParser(description="Isolated headless Herdr UI development; down retains your files")
    parser.add_argument("--name", default="dev")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--app", required=True, help=argparse.SUPPRESS)
    parser.add_argument("--core", required=True, help=argparse.SUPPRESS)
    parser.add_argument("--weston", required=True, help=argparse.SUPPRESS)
    parser.add_argument("--xwayland", required=True, help=argparse.SUPPRESS)
    parser.add_argument("command", choices=["up", "down", "info", "status", "ls", "commands", "do", "check", "shot", "input"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command in ("up", "down", "info") and args.arguments:
            raise Failure(f"{args.command} takes no arguments; put --name/--root before the command")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
            raise Failure("Name must contain only letters, digits, underscores, or hyphens")
        base = Path(f"/var/tmp/saccade-herdr-ui-{os.getuid()}")
        if args.root is None:
            private_directory(base)
        root = args.root or base / args.name
        sandbox = Sandbox(root, args.app, args.core, args.weston, args.xwayland)
        with sandbox.locked():
            if args.command in ("up", "down", "info"):
                print(json.dumps(getattr(sandbox, args.command)(), indent=2))
            elif args.command == "input":
                if not args.arguments:
                    raise Failure("input requires click X Y, type TEXT, key KEYSYM [MOD...], or sequence JSON")
                if args.arguments[0] == "sequence":
                    sandbox.input(json.loads(" ".join(args.arguments[1:])))
                else:
                    action = args.arguments
                    if action[0] in ("click", "move"):
                        action = [action[0], *map(int, action[1:])]
                    elif action[0] == "type":
                        action = ["type", " ".join(action[1:])]
                    sandbox.input([action])
            else:
                arguments = args.arguments
                if args.command == "shot" and not arguments:
                    arguments = [str(root / "shot.png")]
                return sandbox.app_command([args.command, *arguments])
    except (Failure, OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        print(f"herdr-ui: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
