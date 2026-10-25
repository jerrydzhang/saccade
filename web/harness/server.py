import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SAC = Path(os.environ.get('SACCADE_WORKBENCH_BINARY', ROOT / 'target/debug/sac')).resolve()
RUNS = ROOT / 'runs'


def fnv(text):
    value = 0x811c9dc5
    for byte in str(text).encode():
        value = ((value ^ byte) * 0x01000193) & 0xffffffff
    return value


class Server:
    """One real sac serve over an owned scratch tracker, seeded through
    the CLI face; the tracker story is the approved prototype seed."""

    def __init__(self, run):
        self.run = run.resolve()
        self.run.mkdir(parents=True, exist_ok=False)
        for name in ('home', 'tmp', 'repo'):
            (self.run / name).mkdir()
        self.db = self.run / 'workbench.db'
        self.repo = self.run / 'repo'
        self.env = {
            'PATH': os.environ['PATH'] + ':/etc/profiles/per-user/jerry/bin:/run/current-system/sw/bin', 'HOME': str(self.run / 'home'),
            'TMPDIR': str(self.run / 'tmp'), 'SACCADE_ACTOR': 'fixture-author',
        }
        git = {'PATH': self.env['PATH'],
               'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
               'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}
        subprocess.run(['git', 'init', '-q', '--initial-branch=main', str(self.repo)], check=True, env=git)
        (self.repo / 'readme').write_text('the project\n')
        for args in (['add', '.'], ['commit', '-qm', 'base']):
            subprocess.run(['git', '-C', str(self.repo)] + args, check=True, env=git)
        self.sac = None
        self.log = (self.run / 'serve.log').open('w')
        self.seed_log = (self.run / 'seed.log').open('w')
        self.url = None

    def cli(self, *args):
        command = [str(SAC), '--db', str(self.db), '--repo', str(self.repo), '--json', *args]
        out = subprocess.run(command, env=self.env, capture_output=True, text=True, timeout=30)
        self.seed_log.write(json.dumps({'argv': command, 'exit': out.returncode,
                                        'stdout': out.stdout, 'stderr': out.stderr}) + '\n')
        self.seed_log.flush()
        if out.returncode:
            raise RuntimeError(f'seed failed: {out.stderr}')
        return json.loads(out.stdout)

    def seed(self):
        for title in ('Preserve the cursor while inserting citations',
                      'Keep buffers independent of panes',
                      'Render rich publication previews'):
            self.cli('create', 'task', title)
        self.cli('comment', 't-0', '--title', 'The insertion bookmark',
                 'Keep a draft cursor bookmark while the citation picker is open. Selecting a source inserts a marker at that bookmark, not at the end of the note.')
        self.cli('comment', 't-0', '--title', 'Refernces corrected',
                 'Refernces already in the draft keep their marker numbers. Selecting the same source again reuses its mapping.')
        self.cli('revise', 'c-1', 'References already in the draft keep their marker numbers. Selecting the same source again reuses its mapping.')
        self.cli('comment', 't-0', '--title', 'The picker contract',
                 'The picker starts from the draft. Search contribution text or enter a comment id. Preview the source before selecting it. These insertion rules [^1] preserve the writing position.', '--ref', 'c-0')
        self.cli('comment', 't-0', '--title', 'Replacing sources',
                 'Replace existing references whenever a different source is selected.')
        self.cli('withdraw', 'c-3', '--note', 'This note described the wrong insertion behavior; earlier mappings stay unchanged.')
        self.cli('comment', 't-1', '--title', 'Buffers outlive panes',
                 'Buffers outlive panes. Closing a pane must not discard an unpublished draft or the last reading position. Two panes showing one buffer retain independent positions.')
        self.cli('comment', 't-1', '--title', 'The source beside the draft',
                 'The source stays visible beside the draft. Pane layout is client convenience, not a tracker workspace. Citation insertion [^1] does not require opening another permanent panel.', '--ref', 'c-2')
        self.cli('comment', 't-2', '--title', 'Rich preview material',
                 'Preview the material, not just its markup. Inline math $x^2+1$ and display math $$\\frac{a}{b}$$ should remain inspectable.\n\n| Action | Record effect |\n| --- | --- |\n| Preview | None |\n| Publish | One note |\n\n```rust\nlet cursor = draft.selection();\n```\n\nUser data remains Unicode-capable: \u03bb. Raw HTML remains text: <img src=x onerror="alert(1)">.')
        import base64
        image = self.run / 'pane-sample.png'
        image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII='))
        artifact = self.cli('artifact', 't-2', str(image), '--name', 'pane-sample.png')
        seq = artifact['records'][0]['seq']
        self.cli('comment', 't-2', '--title', 'The artifact pointer',
                 'A published image pointer can be inspected directly in a preview [^1]. Opening the artifact does not copy its bytes into the record.', '--ref', str(seq))

    def start(self):
        self.seed()
        command = [str(SAC), '--db', str(self.db), '--repo', str(self.repo),
                   'serve', '--bind', '127.0.0.1', '--port', '0']
        self.sac = subprocess.Popen(command, cwd=str(self.repo), env=self.env,
                                    stdin=subprocess.DEVNULL, stdout=self.log,
                                    stderr=subprocess.STDOUT)
        hint = self.db.with_name(self.db.name + '.addr')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.sac.poll() is not None:
                raise RuntimeError('serve exited at startup')
            if hint.exists():
                self.url = hint.read_text().strip()
                try:
                    self.identity()
                    return
                except (OSError, RuntimeError):
                    pass
            time.sleep(0.05)
        raise RuntimeError('serve readiness deadline')

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.url + path, data=body,
                                     headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def identity(self):
        status, raw = self.request('/api/v1/version')
        data = json.loads(raw)
        if status != 200 or data.get('version') != '0.15.0' or data.get('db') != str(self.db):
            raise RuntimeError('serve DB/version gate refused')
        return data

    def close(self):
        if self.sac is not None:
            if self.sac.poll() is None:
                self.sac.terminate()
                try:
                    self.sac.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.sac.kill()
                    self.sac.wait()
            else:
                self.sac.wait()
            print(f'serve reaped: {self.sac.returncode}', flush=True)
        self.log.close()
        self.seed_log.close()
