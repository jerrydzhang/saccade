# Board viewer harness

One entry point: `viewer/harness/entry.py`. Agents never spawn app,
browser, compositor, server or SSH processes by hand.

## Lanes and what they claim

- `check-fast`: plain, uncontained tool runs (typecheck, pure contract
  tests, build). They are ordinary build invocations. They make **no
  isolation claim** of any kind.
- `preflight`: the gated minimal containment verification described
  below. It runs the containment canary inside one bwrap sandbox and
  nothing else.
- `browser`: the gated browser lane (source present; runs only under a
  named authorization). Same supervisor, same canary-first order - see
  below for the exact shape.
- `native`: the gated native/WebKitGTK lane (source present; runs only
  under a named authorization). Same supervisor, same canary-first
  order - see below for the exact shape.
- `serve`: not implemented; its own reviewed checkpoint.

Agents never spawn app, browser, compositor, server or SSH processes by
hand; everything sandboxed goes through the harness.

## Fast lane (no services)

```sh
python3 viewer/harness/entry.py check-fast
```

Runs, in order, with measured warm cost printed per step:

1. `npm run check` - svelte-check over every TS/Svelte source (real
   typechecking; transpilation is not typechecking).
2. `npm run test` - pure contract tests via `node --test` (no browser).
3. `npm run build` - the production frontend build.

Measured warm cost on the dev host (2026-11 checkpoint): 9.3 s total
(typecheck 1.7 s, contracts 0.3 s, build 7.2 s). Cold dependency fetches
are one `npm install`, reported separately from the loop.

## Containment contract (the reviewed spec)

The enforced bwrap boundary exists for the service lanes and the
preflight. It is not asserted anywhere else: the plain checks above do
not certify isolation, and no nix-sandbox heuristic stands in for it.

- **Supervisor**: one `bwrap` process per run. Payloads are its children
  in a private PID namespace. Inner processes are never signaled
  individually; the namespace tears down with the supervisor.
- **Namespaces** (`--unshare-all`): user, PID (with fresh `--proc /proc`),
  network (denies the host's abstract Unix sockets too), UTS, IPC,
  cgroup. None retained.
- **Mounts**: fresh tmpfs `/tmp`, `/run` (so the host's
  `/run/user/<uid>`, DBus and display sockets are simply absent),
  `/dev/shm`, and `--perms 0700 --tmpfs <runroot>/runtime` for
  `XDG_RUNTIME_DIR` - mounted AFTER the run-root bind (ordered mounts:
  a later mount covers an earlier one, so the tmpfs is never obscured
  by the descendant bind); fresh `--dev /dev`; read-only `/nix/store`;
  per-lane read-only binds of explicit inputs; read-write bind of the
  run root only. Host HOME is never mounted.
- **Environment**: a fresh dict for the whole chain - the bwrap process
  itself is spawned with it, and the same values are `--setenv` inside:
  PATH, LANG, the private roots. Lane additions (loopback tools, pinned
  browser driver paths, explicit canary baselines) are explicit entries,
  never inherited. `SACCADE_*`, `HERDR_*`, `SSH_*`, `GPG_*`, `DBUS_*`,
  `XAUTHORITY`, `DISPLAY`, `WAYLAND_*` and proxies are absent by
  construction.
- **Capabilities**: the default set inside the private user namespace
  only, so loopback can be brought up (CAP_NET_ADMIN scoped to the
  sandbox's own net namespace). No host capability crosses the boundary.
- **Canary before payload** (`containment_canary.py`): runs first in the
  same sandbox and refuses unless every required baseline is present,
  absolute and non-empty - all six host namespace ids (pid, net, uts,
  ipc, mnt, user) plus host HOME, UID and the tracker DB path - and
  namespaces differ, host paths are invisible, `/tmp` and
  `XDG_RUNTIME_DIR` are private tmpfs, the environment is scrubbed, and
  the network namespace has loopback only. Partial, empty or relative
  baselines are refusals. The canary is read-only enumeration; it never
  opens listeners or connections and never signals.
- **Ownership and cleanup** (`isolation.PidfdSupervisor`): the harness
  holds a pidfd for the supervisor. Cleanup TERM -> bounded grace ->
  KILL goes through `signal.pidfd_send_signal` only; reaping is verified
  by waiting on the real child. If `os.pidfd_open` fails, the run is
  refused - there is no numeric-PID fallback. The child carries
  `PR_SET_PDEATHSIG` with a `getppid` re-check to close the parent-death
  race (signaling only itself). `--die-with-parent` chains supervisor
  death to harness death; PID-namespace teardown covers the inner tree.
  Failure paths run the same cleanup. No `pkill`, no `killall`, no
  pattern sweeps, no signals to unowned handles, no automatic restarts.
- **Fail-closed fallback**: if user namespaces or bwrap are unavailable,
  the lane is refused with an explicit containment-refusal line. There
  is no weaker fallback. `systemd-run --user` is not equivalent
  containment and is not used.
- **Logs and artifacts** survive under the run root
  (`/var/tmp/saccade-board-viewer-<uid>/<run>/logs`).

## The preflight lane, exactly

```
python3 viewer/harness/entry.py preflight
# requires BOARD_VIEWER_RUNTIME_AUTHORIZED=<review record>
```

1. Static reads: bwrap found in the pinned devshell PATH (resolved to an
   absolute path), user-namespace sysctls positive. No unshare happens
   during static reads.
2. One sandbox via `isolation.build_command` with the canary as the sole
   payload: the exact command is printed before it runs. Tool paths are
   absolute pinned paths (bwrap, the canary's python interpreter).
3. Host baselines recorded before the sandbox exists - six namespaces +
   absolute HOME + UID + absolute caller tracker-db path (from
   `--tracker-db` or `SACCADE_DB`; relative or missing is a refusal) -
   and passed as explicit env entries. Empty or malformed baselines are
   refusals.
4. Bounded by 30 s; on timeout the supervisor is stopped through the
   pidfd and the run is a refusal.
5. Canary output is preserved at `<runroot>/logs/canary.log`.

This lane never runs a test payload, browser, core, compositor or SSH
activity; its payload is the read-only canary alone.

## The browser lane, exactly (source; runs only under authorization)

```
BOARD_VIEWER_RUNTIME_AUTHORIZED=<review record> \
nix develop .#board-viewer --command \
  python3 -B viewer/harness/entry.py browser \
  [--tracker-db <absolute path>]
```

Time bound: 360 s, then the supervisor is stopped through the pidfd and
the run is a refusal. No automatic retry.

Order inside the one sandbox: (1) the containment canary runs
in-process FIRST (the reviewed module, invoked as main()) - a refusal
ends the lane before any listener or driver exists; (2) then, and only
then, a stdlib loopback static server thread serves the built frontend
inside the sandbox netns (the only listener this lane ever creates; no
child process, no framework); (3) two engine passes, one reused
headless browser per engine - Chromium first, then Playwright WebKit -
via the Nix-pinned Playwright driver (driver and browsers from the
same nixpkgs revision; the pinned browsers store path carries WebKit;
nothing downloaded or installed at runtime); the page navigates to the
server and waits for the shared viewer API before any action; (4) the
same bounded check list under each engine over the real frontend and
real fixture boards - readiness waits on actual inspection outcomes
(never fixed sleeps), DOM and inspection state first, then screenshots
saved under `<runroot>/shots/` with engine-prefixed names for later
human and agent inspection; (5) the server thread shuts down before
exit.

Binds (all read-only): the canary at `/canary/`, the lane script at
`/lane/`, `viewer/dist` at `/app/dist`. Env additions beyond the fresh
allowlist, all explicit absolute paths or baseline values:
`PLAYWRIGHT_BROWSERS_PATH` (pinned browser package), `FONTCONFIG_FILE`
(the pinned generated fontconfig - a store path over pinned DejaVu
fonts; the existing /nix/store read-only bind covers it, so no host
`/etc/fonts` or font directories are ever mounted), `SHOTS_DIR` (the
run root's own shots directory), `__EGL_VENDOR_LIBRARY_DIRS` (pinned
mesa egl_vendor.d) and `LIBGL_ALWAYS_SOFTWARE` (the WPE browser's
display-less EGL, the native lane's software-GL pin), the six
namespace baselines, host HOME/UID, tracker-db. The payload interpreter is the pinned
withPackages python, its closure verified to contain the playwright
module before any launch (the scrubbed env inherits no host
PYTHONPATH). Chromium runs headless with its own sandbox disabled and
WebKit launches its pinned headless WPE MiniBrowser directly (the
driver's pw_run.sh needs PATH tools the fresh sandbox PATH deliberately
does not carry); the enforced bwrap boundary is the containment.

Clipboard honesty: what the browser writes is the engine's own
clipboard inside the sandbox - it is never the desktop's clipboard.
Native clipboard fidelity is a real-client question for the
actual-client gate and is never implied by this lane.

## The native lane, exactly (source; runs only under authorization)

```
BOARD_VIEWER_RUNTIME_AUTHORIZED=<review record> \
BOARD_VIEWER_APP_BIN=<pinned packaged wrapper> \
nix develop .#board-viewer --command \
  python3 -B viewer/harness/entry.py native \
  --tracker-db <baseline db> --board <absolute board.yaml>
```

Time bound: 120 s, then the supervisor is stopped through the pidfd and
the run is a refusal. No automatic retry.

Order inside the one sandbox: (1) the containment canary in-process
FIRST; (2) then a private headless Weston (pixman renderer, no GPU, no
socket beyond the fresh private XDG_RUNTIME_DIR); weston runs with
--debug (weston(1): only then is the weston-output-capture interface
exposed for weston-screenshooter) and kiosk-shell (the app fills the
output; no desktop-shell helper clients) - both private-sandbox facts;
(3) then the pinned Tauri application binary in the foreground, opened
on the explicit board passed as its launch argument - the same
foreground open a human client uses; the lane env pins mesa's
egl_vendor.d through __EGL_VENDOR_LIBRARY_DIRS with
LIBGL_ALWAYS_SOFTWARE=1 (glvnd doc: vendor JSON dirs override the
compiled-in /usr paths absent in the sandbox; Mesa doc: always software
rendering - llvmpipe/swrast from the pinned mesa drivers output, no
/dev/dri, no host /run, no GPU); (4) three bounded weston-screenshooter
frame captures (a hung capture is a recorded failed check, not a lane
crash) plus the app's own bridge stderr lines as outcome evidence;
(5) owned teardown of the app and compositor children.

Binds (read-only): the canary, the lane script, the pinned app binary
and weston/weston-screenshooter binaries (all /nix/store paths covered
by the store bind - stated explicitly for review), and the explicit
board file. Env additions: WESTON_BIN/APP_BIN/SCREENSHOOTER_BIN,
BOARD_FILE, FONTCONFIG_FILE (the pinned generated config), SHOTS_DIR,
plus the usual baselines. No GPU, no display socket beyond the private
XDG_RUNTIME_DIR, no host service contact.

The board opened is an explicit path - the same foreground open
operation a human client runs (`nix run .#board-viewer -- <board>`).
No browser-fixture fallback disguises native failure: the app starts
empty without a board path. The project/workspace/board selector and
the real remote adapter remain later scope; nothing here invents a
project registry or tracker change.

Cold costs are reported separately: the native package build (WebKit,
Tauri) is a one-time Nix build measured on first run; the lane itself
prints its warm cost.

Ownership: one PidfdSupervisor per run as everywhere else; the lane's
own server child is stopped through its held handle before the
namespace teardown; logs and screenshots survive under the run root.

Cold costs are reported separately from the warm loop: the browser
lane's own warm cost is measured per run and printed; cold devshell
build (playwright pair) is a one-time devshell cost.

## Packaging and the real-client run path

The viewer's packaging lives in `viewer/package.nix`: one
callPackage-able file whose dependencies come from the pkgs it is
called on. The flake builds `packages.board-viewer`,
`packages.board-viewer-web`, the flake checks, and the
`apps.board-viewer` entry through that one derivation, and exposes the
same app as `overlays.default`.

Why the overlay: a binary built against this flake's own pinned
nixpkgs links a foreign libglvnd/mesa stack. On a NixOS client that
stack cannot use the host's `/run/opengl-driver` EGL vendors and the
app aborts at startup (`Could not create default EGL display:
EGL_BAD_PARAMETER`). Building the packaged binary against the
consumer's own nixpkgs keeps the WebKitGTK/GL stack on the revision
the host system was built from.

Consumer (NixOS configuration):

```nix
nixpkgs.overlays = [ inputs.saccade.overlays.default ];
```

then `pkgs.saccade-board-viewer` is the packaged app (main program
`saccade-board-viewer`, opened on an explicit board path). The
derivation takes everything from the pkgs it is called on; it assumes
no input of this flake - the rust-overlay and devenv inputs serve the
development shell only.

One-off review run against a client system revision, without touching
the consumer's configuration:

```sh
nix run --override-input nixpkgs github:NixOS/nixpkgs/<client system rev> \
  <repo flake ref>#board-viewer -- <absolute board.yaml>
```

The flake's own `packages.board-viewer` remains the dev-host build and
the native lane's binary source; neither replaces a real-client
review.

## What this harness is not

No production probes, no host socket/status calls, no tracker database
access, no SSH to any real machine (a future remote-fixture lane runs a
disposable sshd inside the sandbox with its own keys), no automatic
restarts, no service supervision beyond one run's lifetime.
