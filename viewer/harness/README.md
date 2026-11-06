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
- `browser`, `native`, `serve`: service lanes. These require the
  enforced bwrap boundary with a passing canary first. Not implemented
  in this change; each arrives as its own reviewed checkpoint.

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

## What this harness is not

No production probes, no host socket/status calls, no tracker database
access, no SSH to any real machine (a future remote-fixture lane runs a
disposable sshd inside the sandbox with its own keys), no automatic
restarts, no service supervision beyond one run's lifetime.
