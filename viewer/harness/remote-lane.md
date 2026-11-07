# The in-sandbox remote integration lane (reviewed design; implemented)

Status: the reviewed design is implemented - `entry.py` verb `remote`,
the payload `remote_lane.py`, the via program `via_b.py`, the fixture
tree `remote_fixture/`. Nothing has been run: the lane executes only
under a named runtime authorization (a run budget the coordinator
issues separately). One implementation note over the text below: the
viewer's render outcomes carry the figure PATH as their message, not a
line, so the assertions address outcomes by path (`figure_status`,
`figure_denial` in remote_lane.py) instead of by authored line; the
pure contract for every shape runs in check-fast (harness_checks).

## What it proves

One bwrap sandbox holds a private two-core herdr mesh and the headless
viewer. Core B serves a fixture board (text + a real binary figure +
a symlink that escapes the board directory). The viewer opens the
location `b:/board/real.yaml` through core A - the same bridge, wire
and containment code the production viewer uses - and the lane asserts:

1. the board renders, with the header naming machine `b` and the
   declaration path (the launch grammar round-trips);
2. the binary figure's bytes arrive through A's mesh (render outcome
   `ok` on the real figure's line);
3. the escape is refused by the REMOTE containment check (render
   outcome `denied` on the escape's line, message naming the directory
   escape), with no bytes rendered.

No network, no SSH, no production core or socket: the namespace
unshares everything, `/run` is a fresh tmpfs, `XDG_RUNTIME_DIR` is the
private 0700 tmpfs, and both cores' state lives in the sandbox-private
HOME, wiped with the run root.

## Fixture tree (checked in: `harness/remote_fixture/`)

```
real.yaml              authored board: one prose block, then two figure
                       blocks - the real figure first (assets/figure.png,
                       line 6), the escape second (escaped.png, line 10)
assets/figure.png      a tiny real PNG (deterministic bytes, committed)
escaped.png            a symlink to /lane/webdriver_session.py - present,
                       readable, and OUTSIDE /board inside the sandbox
```

The tree is bound read-only at the neutral sandbox path `/board` (the
same path the native and session lanes already use for a local board),
so the symlink's absolute target resolves inside the sandbox to a file
that exists but is outside the declaration directory. The fixture is
not under `viewer/fixtures/` - the browser lane's fixture glob and the
vite asset bundler never see it.

## Sandbox setup (host side, entry.py `remote` verb)

Before any spawn, exactly the gated lanes' refusals: pinned absolute
paths for every binary, `static_preflight`, the runtime-authorization
gate (`BOARD_VIEWER_RUNTIME_AUTHORIZED` naming the authorizing review
record - this document's review), the fontconfig and mesa pins of the
native lane, and the tracker-db baseline. Then the run root and the
chain env, and one supervisor over one gate-wrapped bwrap.

Binary resolution (all absolute, all pinned, all refused otherwise):

| role | source |
|---|---|
| bwrap, python, weston, weston-screenshooter, tauri-driver, WebKitWebDriver | exactly the session lane's resolution |
| app | `BOARD_VIEWER_APP_BIN` (the packaged viewer with this worktree's bridge) |
| herdr-core | `BOARD_VIEWER_HERDR_CORE_BIN`, an executable `/nix/store` path whose `--version` reads exactly 0.1.17 (the flake's pin; checked host-side, read-only) |

Read-only binds (beyond `build_command()`'s store bind and fresh tmpfs):

```
(canary, /canary/containment_canary.py)     unchanged
(remote_lane.py, /lane/remote_lane.py)      the payload
(via_b.py, /pin/via_b.py)                   the via program, below
(remote_fixture/, /board)                   the B-side tree, read-only
(app, weston, shooter, drivers: each at its own absolute path)  unchanged
```

Chain env (the fresh allowlist plus; nothing else added):

```
unchanged:  CONTAINMENT_HOST_* baselines, WESTON_BIN, SCREENSHOOTER_BIN,
            TAURI_DRIVER_BIN, WEBKIT_DRIVER_BIN, APP_BIN,
            FONTCONFIG_FILE, SHOTS_DIR, __EGL_VENDOR_LIBRARY_DIRS,
            LIBGL_ALWAYS_SOFTWARE, BOARD_VIEWER_WEBVIEW_CONSOLE,
            GDK_SCALE, GDK_DPI_SCALE
new:        BOARD_VIEWER_HERDR_CORE_BIN   the pinned herdr-core path
            BOARD_LOCATION   b:/board/real.yaml
            BOARD_VIEWER_VIA_PY     the pinned python absolute path (the via
                             string's interpreter)
new value:  HC_SESSION=a     chain-wide: core A serves this session, the
                             lane's CLI calls target it, and the app's
                             bridge resolves a.sock
```

`build_command()` is untouched: `--unshare-all`, `--die-with-parent`,
`--new-session`, private `/proc`, fresh `/dev`, `/dev/shm`, `/tmp`,
`/run`, the 0700 runtime tmpfs mounted after the run-root bind.

## Processes inside the sandbox (payload: remote_lane.py)

Order, enforced; every spawn holds a Popen handle; every wait is
bounded with the failure recorded, never retried:

```
1. containment canary, in-process          unchanged module, main()
2. core A:  [BOARD_VIEWER_HERDR_CORE_BIN, "serve"]      env HC_SESSION=a
           wait for $XDG_RUNTIME_DIR/herdr-core/a.sock (15 s)
3. core B:  [BOARD_VIEWER_HERDR_CORE_BIN, "serve"]      env HC_SESSION=b
           wait for b.sock (15 s)
4. add:     [BOARD_VIEWER_HERDR_CORE_BIN, "machine", "add",
             "--via", "<BOARD_VIEWER_VIA_PY> /pin/via_b.py",
             "--name", "b", "--session", "b", "--json"]
           env HC_SESSION=a, stdin=/dev/null, cwd=run root
5. confirm: poll [BOARD_VIEWER_HERDR_CORE_BIN, "machine", "ls", "--json"]
           until b is state=connected (20 s); a stuck state is read
           with its `why` and fails the lane, nothing is re-dialed
6. weston, tauri-driver -> WebKitWebDriver -> app
           the session lane's facts verbatim, except the app opens
           args [BOARD_LOCATION] - the machine-named location string,
           not a path
7. assertions, via W3C execute (INSPECT_SCRIPT), bounded:
   a. resource.status == "loaded", resource.machine == "b",
      resource.path == "/board/real.yaml"
   b. the real figure's outcome: kind figure, status ok, message
      assets/figure.png
   c. the escape's outcome: kind figure, status denied, message
      starting "denied:" and quoting both the path and the containment
      refusal
   d. errors: empty
8. one weston-screenshooter frame as visual evidence (the native
   lane's capture facts)
```

## The via program (`/pin/via_b.py`)

machine.add `--via` takes a command that "is given a shell line to run
there as its last argument, like ssh; it must pass bytes both ways
untouched and ask nothing" (pinned 0.1.17 CLI help; the schema's
machine.add carries the same fields: via, name, session). A's core
therefore runs, for the dial probe and for the connect:

```
[<BOARD_VIEWER_VIA_PY>, "/pin/via_b.py", "<line>"]
```

The program never spawns a shell of its own and never second-guesses
the line: the contract is A SHELL LINE, and the sandbox is the
containment. The run budget observed the real shapes (run four's
breadcrumb): the dial-time probe `echo ok`, then a connect line that
locates herdr-core on PATH, sets HC_SESSION=b and execs `connect`:

```python
line = " ".join(sys.argv[1:])   # word-by-word or single-argument alike
argv, env = shell_argv(line, os.environ)
os.execve(argv[0], argv, env)

# shell_argv:
#   refuses a NUL byte or an empty handover (exit 2, quoted)
#   requires BOARD_VIEWER_HERDR_CORE_BIN (absolute; pinned bytes)
#   prepends its bin dir to PATH, so the line's own
#     `command -v herdr-core` resolves to exactly the tested bytes
#   returns ["/bin/sh", "-c", line], child env
```

`/bin/sh` is the pinned bash the lane binds (the sandbox has no /bin
otherwise); `HC_SESSION=b` arrives inside the observed line, set by
the core itself.

The stdin/stdout pipe between A's core and this child is the transport
- inherited fds inside the namespace, no listener, no network. If the
composed lines ever change shape again, the child leaves an argv
breadcrumb in the sandbox tmp dir and A surfaces the failure as an
unreachable/`why` or hc_missing error with the full text in the lane
log - the assumption is checked, not silent.

## Env the mesh runs in

- A and B serve distinct sessions in the SAME private
  `XDG_RUNTIME_DIR`: `herdr-core/a.sock`, `herdr-core/b.sock` - the
  viewer's bridge resolves `a.sock` from the chain env `HC_SESSION=a`,
  exactly as production would on the client device.
- A's machine list lives under the sandbox-private
  `HOME/.local/state/herdr-core/a/` - wiped with the run root; nothing
  persists and no production state is touched. Adding B is the lane's
  own host setup for its own private core, never the viewer's: the
  viewer's no-mutation rule stands.
- The via child inherits A's env (the chain env), and overrides
  `HC_SESSION=b` itself - the one place the two sessions meet.
- `machine.add` non-interactive, from the pinned help: `--via` with
  `--name` needs no terminal and asks nothing; without `--install` a
  version mismatch fails as incompatible (B runs the same pinned 0.1.17
  binary, so no install path exists); `stdin=/dev/null` turns any
  accidental prompt read into a loud EOF failure.

## Owned teardown (payload finally block, then the supervisor)

```
1. webdriver session deleted        (the remote end ends the app)
2. tauri-driver TERM/grace/KILL     (its handler ends WebKitWebDriver)
3. core A TERM/grace/KILL           (held handle)
4. core B TERM/grace/KILL           (held handle)
5. weston TERM/grace/KILL           (held handle)
6. survivor /proc scan              now also names "herdr-core";
                                    survivors are recorded, never
                                    signaled by pattern
7. supervisor.stop()                pidfd TERM -> KILL; the PID
                                    namespace teardown is the backstop
                                    that ends anything held no more
```

A and B die before the compositor and after the app chain: in-flight
reads end as ordinary core-unavailable errors in a process that is
already going away. Every handle is ours; nothing is signaled by
pattern; the namespace teardown remains the backstop.

## Bounds and refusal points

One lane, one bound: 180 s wall clock (`REMOTE_TIMEOUT_SECONDS`), the
supervisor enforcing it through pidfd. Refusal points, all loud: any
pinned path missing, bwrap/userns unavailable, gate token mismatch, a
core socket not appearing, add exiting nonzero (its JSON error quoted
in the lane log), b not reaching connected, the app's inspection not
reaching loaded, any assertion failing. Nothing retries; a failed lane
is evidence with its logs under the run root.

## What stays out of scope

No production core, socket, machine or ssh contact; no new wire method
(the viewer speaks the same three reads); no change to the viewer's
no-mutation rule; no mobile path; no persisting state anywhere. The
lane is verification evidence for the boundary, not the human client
path, and grants no acceptance by itself.
