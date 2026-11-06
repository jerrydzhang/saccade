# Board viewer harness

One entry point: `viewer/harness/entry.py`. Agents never spawn app,
browser, compositor, server or SSH processes by hand.

## Fast lane (runs today, no services)

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

## Gated lanes (source present, execution authorized by review only)

`preflight`, `browser`, `native`, `serve` refuse unless
`BOARD_VIEWER_RUNTIME_AUTHORIZED=<review record>` is set. The gate records
authorization discipline; the actual boundary is the containment below.
Staging: source review first, then at most a minimal containment preflight
(namespace/mount verification before any test payload, browser, core or
network activity), then payload lanes.

## Containment contract (the reviewed spec)

- **Supervisor**: one `bwrap` process per run. Payloads are its children
  in a private PID namespace. Inner processes are never signaled
  individually; the namespace tears down with the supervisor.
- **Namespaces** (`--unshare-all`): user, PID (with fresh `--proc /proc`),
  network (denies abstract Unix sockets of the host too), UTS, IPC, cgroup.
- **Mounts**: fresh tmpfs `/tmp`, `/run` (so the host's
  `/run/user/<uid>`, DBus and display sockets are simply absent),
  `/dev/shm`; fresh `--dev /dev`; read-only `/nix/store`; per-lane
  read-only binds of explicit inputs; read-write bind of the run root
  only. Host HOME is never mounted.
- **Private XDG**: `HOME`, `TMPDIR`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME`, `XDG_RUNTIME_DIR` all point inside the run root;
  `XDG_RUNTIME_DIR` is a fresh tmpfs where a private Weston (native lane)
  creates its own sockets.
- **Environment allowlist**: a fresh dict - PATH, LANG, the six private
  roots above, plus lane additions (loopback tools, pinned browser
  driver paths, explicit canary comparison values). Nothing is inherited:
  `SACCADE_*`, `HERDR_*`, `SSH_*`, `GPG_*`, `DBUS_*`, `XAUTHORITY`,
  `DISPLAY`, `WAYLAND_*` and proxies are absent by construction.
- **Capabilities**: the default set inside the private user namespace
  only, so loopback can be brought up (CAP_NET_ADMIN scoped to the
  sandbox's own net namespace). No host capability crosses the boundary.
- **Canary before payload**: `containment_canary.py` runs first in the
  same sandbox and refuses unless namespaces differ from the host's,
  host paths are invisible, the environment is scrubbed, `/tmp` and
  `XDG_RUNTIME_DIR` are private tmpfs, and the network namespace has
  loopback only. The canary compares state; it never contacts production
  and never opens external connections. Disposable in-sandbox listeners
  are the only network checks, if any.
- **Ownership and cleanup** (`isolation.PidfdSupervisor`): the harness
  holds a pidfd for the supervisor. Cleanup TERM -> bounded grace ->
  KILL goes through the pidfd only (PID reuse cannot redirect it);
  reaping is verified by waiting on the real child. `--die-with-parent`
  plus `prctl(PR_SET_PDEATHSIG)` chains supervisor death to harness
  death, and PID-namespace teardown covers the inner tree. Failure paths
  run the same cleanup. No `pkill`, no `killall`, no pattern sweeps, no
  signals to unowned handles, no automatic restarts.
- **Fail-closed fallback**: if user namespaces or bwrap are unavailable,
  the lane is refused with an explicit containment-refusal line. There
  is no weaker fallback. `systemd-run --user` is not equivalent
  containment and is not used.
- **Logs and artifacts** survive under the run root
  (`/var/tmp/saccade-board-viewer-<uid>/<run>/logs`).

## Nix check lane containment

The flake's `board-viewer-contracts` check runs the same canary inside
the nix build sandbox. It fails closed if the sandbox is off: no
`__noChroot` or impure exceptions exist for these checks. A passing
`checks.board-viewer-contracts` therefore carries its own containment
evidence in the build log (canary output: namespaces, mounts, env,
visibility).

## What this harness is not

No production probes, no host socket/status calls, no tracker database
access, no SSH to any real machine (a future remote-fixture lane runs a
disposable sshd inside the sandbox with its own keys), no automatic
restarts, no service supervision beyond one run's lifetime.
