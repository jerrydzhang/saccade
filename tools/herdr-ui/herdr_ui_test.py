import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from herdr_ui import Failure, Sandbox, private_directory, process_token


FAKE = r'''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

role = Path(sys.argv[0]).name
state = Path(os.environ['XDG_STATE_HOME'])
ready = state / 'app.pid'
core_pid = state / 'core.pid'
args = sys.argv[1:]

def forever():
    while True:
        time.sleep(1)

if role == 'core':
    if args == ['serve']:
        forever()
    elif args == ['server', 'start']:
        process = subprocess.Popen([sys.executable, sys.argv[0], 'serve'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        core_pid.write_text(str(process.pid))
    elif args == ['server', 'status']:
        print(json.dumps({'pid': int(core_pid.read_text())}))
    elif args == ['server', 'stop']:
        if core_pid.exists():
            try:
                os.kill(int(core_pid.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
            core_pid.unlink()
elif role == 'weston':
    if os.environ.get('FAKE_DISPLAY_FAIL'):
        sys.exit(2)
    log = next(x.split('=', 1)[1] for x in args if x.startswith('--log='))
    Path(log).write_text('xserver listening on display :240\n')
    forever()
elif role == 'app':
    if not args:
        if os.environ.get('FAKE_APP_FAIL'):
            sys.exit(2)
        ready.write_text(str(os.getpid()))
        forever()
    elif args == ['status']:
        if not ready.exists():
            sys.exit(1)
        print(json.dumps({'errors': []}))
    elif args == ['do', 'window.close']:
        if os.environ.get('FAKE_CLOSE_HANG'):
            forever()
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
            ready.unlink()
    else:
        print(json.dumps(args))
'''


class SandboxContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.paths = {}
        for role in ('app', 'core', 'weston', 'xwayland'):
            binary = self.base / role
            binary.write_text(f'#!{sys.executable}\n' + FAKE)
            binary.chmod(0o700)
            self.paths[role] = str(binary)
        self.sandbox = self.make('sandbox')

    def make(self, name):
        return Sandbox(self.base / name, **self.paths)

    def tearDown(self):
        for name in ('sandbox', 'failure', 'foreign'):
            sandbox = self.make(name)
            if sandbox.record.exists():
                with sandbox.locked():
                    sandbox.down()
        self.temporary.cleanup()

    def test_lifecycle_and_isolation(self):
        with patch.dict(os.environ, {'HC_TERMINAL_ID': 'production', 'HC_SESSION': 'production',
                                     'SACCADE_SERVER': 'http://production.invalid', 'DISPLAY': ':999',
                                     'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/production/bus',
                                     'WAYLAND_SOCKET': '123', 'SSH_AUTH_SOCK': '/production/agent'}):
            with self.sandbox.locked():
                info = self.sandbox.up()
                self.assertTrue(info['running'])
                environment = self.sandbox.environment(display=True)
                self.assertEqual(environment['HC_SESSION'], 'ui')
                self.assertEqual(environment['DISPLAY'], ':240')
                self.assertNotIn('HC_TERMINAL_ID', environment)
                self.assertNotIn('SACCADE_SERVER', environment)
                for key in ('DBUS_SESSION_BUS_ADDRESS', 'WAYLAND_SOCKET', 'SSH_AUTH_SOCK'):
                    self.assertNotIn(key, environment)
                for key in ('HOME', 'XDG_CONFIG_HOME', 'XDG_STATE_HOME', 'XDG_DATA_HOME',
                            'XDG_CACHE_HOME', 'XDG_RUNTIME_DIR', 'HERDR_APP_HOME'):
                    self.assertTrue(Path(environment[key]).is_relative_to(self.sandbox.root))
                processes = set(self.sandbox.owned())
                self.assertGreaterEqual(len(processes), 3)
                self.sandbox.up()
                self.assertEqual(set(self.sandbox.owned()), processes)
                for actions in ([], [['key']], [['click', 1]], [['type', 12]]):
                    with self.subTest(actions=actions), self.assertRaises(Failure):
                        self.sandbox.input(actions)
                result = self.sandbox.run([self.paths['app'], 'check', 'a path with spaces'])
                self.assertEqual(json.loads(result.stdout), ['check', 'a path with spaces'])
                self.sandbox.run([self.sandbox.root / 'bin/xdg-open', 'https://example.invalid/a b'])
                self.assertEqual((self.sandbox.root / 'opened.log').read_text(), 'https://example.invalid/a b\n')
                kept = self.sandbox.root / 'app/keep.txt'
                kept.write_text('unfinished mod work')
                core_config = self.sandbox.root / 'config/herdr-core/config.toml'
                core_config.write_text('notes = false\nshell = ["sh"]\n')
                self.sandbox.down()
                self.assertFalse(self.sandbox.owned())
                self.assertEqual(kept.read_text(), 'unfinished mod work')
                self.sandbox.down()
                self.assertTrue(self.sandbox.up()['running'])
                self.assertEqual(kept.read_text(), 'unfinished mod work')
                self.assertEqual(core_config.read_text(), 'notes = false\nshell = ["sh"]\n')
                self.sandbox.down()

    def test_failed_start_cleans_partial_processes(self):
        for failure in ('FAKE_APP_FAIL', 'FAKE_DISPLAY_FAIL'):
            with self.subTest(failure=failure):
                sandbox = self.make('failure')
                with patch.dict(os.environ, {failure: '1'}):
                    with sandbox.locked():
                        with self.assertRaises(Failure):
                            sandbox.up()
                        self.assertFalse(sandbox.owned())
                        self.assertFalse(sandbox.info()['running'])

    def test_shutdown_survives_unresponsive_app(self):
        with patch.dict(os.environ, {'FAKE_CLOSE_HANG': '1'}):
            with self.sandbox.locked():
                self.sandbox.up()
                self.sandbox.down()
                self.assertFalse(self.sandbox.owned())

    def test_foreign_processes_and_directories_are_not_adopted(self):
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        try:
            with self.sandbox.locked():
                self.sandbox.state.update(app_pid=sentinel.pid, core_pid=sentinel.pid,
                                          display_pid=sentinel.pid)
                self.sandbox.save()
                self.sandbox.down()
                self.assertIsNone(sentinel.poll())
                self.assertNotEqual(process_token(sentinel.pid), self.sandbox.state['token'])
            target = self.base / 'target'
            private_directory(target)
            symlink = self.base / 'symlink'
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaises(Failure):
                with Sandbox(symlink, **self.paths).locked():
                    pass
            target.chmod(0o755)
            with self.assertRaises(Failure):
                private_directory(target)
            unowned = self.base / 'unowned'
            private_directory(unowned)
            (unowned / 'personal.txt').write_text('do not adopt')
            with self.assertRaises(Failure):
                with Sandbox(unowned, **self.paths).locked():
                    pass
            self.assertEqual((unowned / 'personal.txt').read_text(), 'do not adopt')
            invalid = self.make('invalid')
            private_directory(invalid.root)
            for token in (None, '', 'not-a-token'):
                invalid.record.write_text(json.dumps({'root': str(invalid.root), 'token': token}))
                with self.subTest(token=token), self.assertRaises(Failure):
                    with invalid.locked():
                        invalid.down()
                self.assertIsNone(sentinel.poll())
            with self.sandbox.locked():
                self.sandbox.prepare()
                settings = self.sandbox.root / 'app/settings.json'
                settings.unlink()
                outside = self.base / 'personal-settings.json'
                outside.write_text('personal settings')
                settings.symlink_to(outside)
                with self.assertRaises(Failure):
                    self.sandbox.prepare()
                self.assertEqual(outside.read_text(), 'personal settings')
        finally:
            sentinel.send_signal(signal.SIGTERM)
            sentinel.wait(timeout=5)

    def test_cli_rejects_misplaced_scope_arguments(self):
        script = Path(__file__).with_name('herdr_ui.py')
        root = self.base / 'misplaced'
        binaries = [argument for role, path in self.paths.items() for argument in (f'--{role}', path)]
        for command in ('up', 'down', 'info'):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, str(script), *binaries, '--root', str(root),
                                         command, '--name', 'other'], text=True, capture_output=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn('before the command', result.stderr)
                self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
