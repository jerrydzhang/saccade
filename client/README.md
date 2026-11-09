# Board viewer

A retained, read-only research reader: it renders an explicitly supplied
board declaration and nothing else. Loading a board never touches tracker
storage, authority, or workflow; the `task` field in a declaration is
authored metadata, not a tracker association.

## Run

```sh
# the fast development loop: typecheck, pure contract tests, the
# bridge crate's contract tests, build
python3 harness/entry.py check-fast

# the native client's own guard tests (Tauri shell, no UI start)
nix develop .#board-viewer -c bash -c 'cd src-tauri && cargo test'
```

Measured warm cost of the fast loop: about 9 seconds. Service lanes
(browser, native, remote, session) are separate, gated, and launched only
through `harness/entry.py` under enforced OS isolation - the harness
section below is their contract.

## Reading a board

The standing control is the header location field: give an explicit
location and open it. The viewer is never a local opener (t-24@r11,
t-24@r12): a location is `tracker/t-N[@rK]` (a task board on that
tracker), `machine:/absolute/path` (a machine from the local herdr-core's
list), or a bare `/absolute/path` (a file on the TRACKER's machine,
read through Herdr core: the machine that hosts the tracker is declared
once in the client configuration as `tracker-machine`; on the tracker's
own machine - proven by machine identity, the root page's machine id
equaling this device's own /etc/machine-id (t-24@r14), never the server
address's shape - the local core's own machine is read, and without a
declaration when the proof fails or an id is unavailable the open
refuses saying what to set). The same grammar is every open surface's,
picker
included; there is no launch argument and no path on the device the
human sits at. Fixtures live in `fixtures/` and remain a dev/test path
(the lanes address them through the inspection surface and the
lanes' test-open channel, never the human UI).

Grammar versions: the viewer declares the versions it renders - exactly
1 today, the same range the shared `board-grammar` crate parses. A board
declaring no grammar version, or an unknown major version, refuses with
a message that names the supported versions; there is no legacy
interpretation. A component kind the build does not know (a newer minor
grammar's admission) never refuses the board: it renders as its raw
authored source beside a diagnostic.

The read surface is `/view/v1` only (t-14 "Pages", t-24@r2): the root
page is the version contract (the tracker name, the view majors, the
board grammar range, and the host facts - the machine id and the
repository path - a bare-path read needs to prove the hub, t-24@r14),
the board page resolves a bare task location to its pinned revision and
names the task, and the authored `board.yaml` and its figures ride the
file family at that pinned revision - so one open's text and figures
are one immutable tree, and a revision landing mid-open cannot split
them. `/api/v1` stays `sac`'s own API.

**The suite's only write to the record is publishing task-board
annotations** (t-14, ratified; the law of this package, landed with the
commit that builds the act): `a` drafts in place on the current
revision of a task board - drafts persist on the device keyed by task
until the window-state pool lands (t-24@r11) and are never lost on a
refused publish; `space p` publishes the drafts together as ONE commit
with ONE message through the commit door, under the HUMAN's own
identity: the human tier and, as the actor, the client configuration's
override or the device account name as `sac` derives it without
SACCADE_ACTOR (`$USER`, else "human"), never refusing for lack of it.
The published board.yaml is the shown revision's authored text plus
annotation components and nothing else - byte-level insertions at the
parts' own positions. Before the POST the viewer re-reads the current
revision and refuses, keeping the drafts, when a revision landed after
its read; an offered board refuses, naming the reject-with-grounds CLI
step. Only the current revision of a task board drafts; a pinned older
revision refuses, saying where annotation is possible (t-24@r11).
Explainer-file annotation writes and the deck are t-21, not this
package's.

The validation split: admission is the shared crate's job on the Tauri
side (`validate_board`, a path dependency), and the native lanes refuse
on its verdict; the frontend keeps only the presentation decoder, which
mirrors the crate's contract exactly so the browser lane (no bridge)
behaves identically. Parity over the shared fixture corpus is pinned by
tests on both sides: this repository's node tests and the
`viewer-native-check` cargo suite read the same `fixtures/` files.

Keys:

- `Ctrl+F` opens the search bar; `Escape` closes it; `Enter` and
  `Shift+Enter` walk the matches
- copy is Source by default; `Ctrl+Shift+C` toggles Plain; `Ctrl+C`
  copies the selection in the current mode (a formula is copied whole
  in its current mode whatever fraction of it is selected)

## The look (t-14 "The design language", as clarified by t-24@r3)

The viewer renders in the ledger design language, on the user's own
base16 scheme. The scheme and the text weight are configuration, read
from a file, never chosen in the app: there is no preference UI and no
in-app persistence.

- The configuration file is `<platform config
  directory>/dev.saccade.board-viewer/config.toml`. It names exactly
  one of `scheme-path` (a path to a base16 scheme file, so a
  Stylix-generated scheme can be followed live) or `scheme-inline`
  (the scheme text itself), and optionally `weight` (`regular`,
  `medium`, `semibold`; regular by default). Unknown keys, a bad
  weight, both scheme keys, or an unreadable scheme file are stated
  problems.
- The host reads the file at start and follows changes: a poll thread
  re-reads the config and the named scheme file when their fingerprint
  (mtime, size) moves, republishes, and the frontend re-applies the
  tokens. The frontend waits over the bridge (`look_config`,
  `wait_look_config`); there are no events and no renderer-chosen
  paths.
- Light or dark is the scheme's own: a light base16 scheme (a light
  `base00`) gives a light suite; the ramp always reads in base16
  order. Nothing in the client reverses a scheme or selects a mode.
- A missing or invalid configuration falls back to the built-in
  default (the user's Mountain (Boosted), the round 16 reference
  scheme) and the client says so once, plainly, in a quiet notice
  line; the notice clears when a valid configuration arrives. A
  window with no host at all (the browser lane) has no configuration
  to fix, so it uses the default silently (t-24@r8).
- Meaning-bearing roles keep their hue and are floored in OKLCH
  against the background: about 3:1 for the state shapes, 4.5:1 for
  state text, 3:1 for muted ink, the accent and the selection label.
  The floor is always reachable (white or black clears about 4.58:1
  against any background): the walk goes in the natural direction
  first and falls back to the opposite direction when that cannot
  reach the ratio (t-24@r10). A miss is a defect, caught by the tests
  over dark, light, mid-luminance, hostile and grayscale schemes -
  never a shipped state and never a log line.
- The colour tokens live only in `src/lib/look/` (scheme parsing, the
  floor, the map, the store); the stylesheets read tokens and never
  name a colour. `styles/tokens.css` holds the static tokens (faces,
  the 13px-minimum type scale, rhythm, one motion speed, and the
  radius tokens, which exist and are zero - no rounded corners);
  `styles/board.css` holds the ledger component rules. A node test
  pins both directions of that split: every colour variable the
  styles read, the map produces; every token the map produces is read
  by the styles or named as waiting for a later step's component.
- The inspection surface carries the look's test tools, named as
  such, never human functionality: `setLook({ schemeText, weight })`
  overrides in memory for one run; `look()` answers the resolved
  scheme name, weight, polarity and the active notice. The older
  `setTheme` remains as a DOM marker for the lane contract only; it
  no longer selects colours (t-24@r3 removed the mode feature).

A machine-named board reads through the local core's read-only face:
`file.read` for declaration text, fixed-argv `exec` (`realpath -e`, then
`cat`) for figure bytes over base64 - no shell strings anywhere. The
viewer never adds machines, logs in, updates a core or writes a file;
those are the human's own host setup. Machine states are structured,
human-actionable errors with the machine's own reason (`needs_login`,
`unreachable`, `needs_update`, unknown name) and no retries. Because the
remote host enforces no path containment, the asset's canonical prefix
check runs on the remote side: one `realpath -e -- <base> <candidate>`
answers with both canonical forms, the prefix relation is checked
there, and the read itself uses the canonical path. Bytes over the 8 MB
exec cap, or flagged `truncated`, and text reads that stopped inside
the file, are refused with an error - never partially rendered. The
local-device host reader (`read_declaration`/`read_asset` with no
machine) is test-only machinery since t-24@r11: the lanes' fixture
reader, carrying the same path guards, never presented as human
functionality. The lanes open their boards through the test-open
channel (`BOARD_VIEWER_TEST_OPEN`, one location at startup riding the
same open path the inspection surface uses) - the launch argument left
the human surface with the local-path form.

Contracts this reader holds (pinned by `tests/` and the bridge's cargo
suite):

- malformed input is a structured, positional error, never a
  manufactured board; unknown keys and fields refuse, exactly as the
  crate refuses them, and unknown component kinds render as source
- one source-positioned Markdown pipeline: math spans, mermaid fences and
  ordinary inert code all come from the same parse; raw HTML in authored
  content is reported, never rendered
- math renders at natural size in the surrounding text color; over-wide
  display math scrolls in its own area; an indivisible inline expression
  may overflow naturally rather than shrink
- an authored math expression is one logical copy and search unit:
  Source mode emits authored bytes, Plain mode emits the TeX value, and
  equal formulas at distinct source locations are distinct occurrences
- ordinary code fences never execute; mermaid renders as a figure that
  never captures gestures while embedded and expands on activation;
  rendering failure keeps the authored source visible
- incorporation provenance is displayed exactly as authored and validates
  nothing
- relative assets resolve at the declaration's location on the selected
  machine; traversal and absolute paths are refused on both the frontend
  and the host side

## Layout

- `src/lib/` - read model, decode, Markdown pipeline, projection, copy
  semantics, sources, view model, inspection projection, DOM renderer,
  hydration, diagram overlay, the /view/v1 page client (`pages.ts`)
  and the tracker read source over it (`tracker.ts`)
- `src/App.svelte` - the shell; one view model feeds the human UI, the
  machine-readable inspection (`window.__boardViewer`) and the copy
  handler
- `src/styles/board.css` - the centrally shared stylesheet
- `src-tauri/` - the narrow Tauri 2 shell: read-only declaration and
  asset commands with mirrored path guards; board admission through the
  shared `board-grammar` crate (`validate_board`); the look's local
  configuration reader and follower (`look_config.rs`); no shell, no
  eval, no generic filesystem surface, no plugins. `src-tauri/core-client/` is
  the pure read-only herdr-core client (JSON-RPC over the local core's
  socket, machine-named reads, remote containment), contract-tested in
  the fast loop against fakes
- `fixtures/` - deterministic boards and assets (ASCII bytes; Unicode
  data is exercised at runtime with escapes in tests)
- `tests/` - pure contract tests (node --test)
- `harness/` - the single entry point and the containment contract
  (the section below)

## The test harness

One entry point: `python3 client/harness/entry.py <lane>`. Agents never
spawn the app, a browser, a compositor, a server or SSH processes by
hand; everything sandboxed goes through the harness.

Standing authorization: routine lane and session runs use the standing
authorization recorded on t-71 (`BOARD_VIEWER_RUNTIME_AUTHORIZED=3030`
or its successor record), with no per-run check-in, while its
conditions hold: a committed clean HEAD, the frozen containment
boundary, run roots kept, sessions end with an explicit `down`.
Excluded forever without a fresh boundary decision: GPU access, the
host display, SSH to real machines, any outside network, other
machines, and any capability outside the frozen boundary. Boundary
changes and new capabilities still go to the coordinator first.

Lanes:

- `check-fast` - plain, uncontained tool runs (typecheck, pure contract
  tests, the bridge crate's contract tests, build). No isolation claim.
- `preflight` - the containment canary inside one bwrap sandbox,
  nothing else.
- `browser` - the gated WebKit check list over the real frontend and
  fixture boards (Chromium triage behind an explicit flag, never
  gating), plus the tracker-location contract over a wire stand-in.
  Behind `BOARD_VIEWER_BROWSER_REFERENCE_HTML` (an absolute, readable
  `.html` file) the same sandboxed engine additionally navigates that
  one reference page after the product pass and retains its screenshot
  in the run root: the contained way to render a visual reference
  (the entry point binds the file read-only; set-but-invalid refuses;
  unset changes nothing about the default single navigation).
- `native` - the gated native/WebKitGTK lane: the packaged app in the
  foreground on an explicit board, private headless Weston, bounded
  frame captures, bridge log evidence.
- `session` - the gated interactive WebDriver tool: `up` holds one
  sandbox with the private compositor, the tauri-driver ->
  WebKitWebDriver chain and the packaged app for an explicit max
  lifetime; `inspect`/`input`/`observe`/`shot` act and observe over a
  private run-root Unix socket; `down` tears the chain down in owned
  order; `checks` runs the scripted sequence.
- `remote` - the gated remote mesh lane: two private herdr-core
  instances and the headless viewer opening a machine-named location
  through them, the session lane's containment with the mesh's binds
  on top.

Containment, one reviewed shape for every gated lane: bwrap with
`--unshare-all` (user, pid, net, uts, ipc, mnt, cgroup - loopback is
raised inside the private netns), private `/proc`, fresh `/dev`,
tmpfs `/dev/shm`, `/tmp`, `/run`, `/nix/store` read-only, per-lane
explicit read-only binds, a run-root read-write bind with a 0700
runtime tmpfs, fresh scrubbed environment, default capabilities inside
the private user namespace only. The containment canary runs in the
same sandbox before any payload and refuses unless every host
baseline is present and every private fact holds. One pidfd
supervisor per run: cleanup is TERM -> bounded grace -> KILL through
`pidfd_send_signal` only, reaping verified by waiting on the real
child, `PR_SET_PDEATHSIG` plus a `getppid` re-check, `--die-with-parent`.
A missing capability is a refusal; there is no weaker fallback. Logs
and screenshots survive under the run root
(`/var/tmp/saccade-board-viewer-<uid>/<run>/`).

Invocations (each bounded; on timeout the supervisor is stopped and
the run is a refusal; no automatic retry):

```sh
BOARD_VIEWER_RUNTIME_AUTHORIZED=3030 SACCADE_DB=<absolute db> \
nix develop .#board-viewer -c python3 client/harness/entry.py preflight

BOARD_VIEWER_RUNTIME_AUTHORIZED=3030 SACCADE_DB=<absolute db> \
nix develop .#board-viewer -c python3 client/harness/entry.py browser

BOARD_VIEWER_RUNTIME_AUTHORIZED=3030 BOARD_VIEWER_APP_BIN=<pinned app> \
SACCADE_DB=<absolute db> nix develop .#board-viewer -c \
python3 client/harness/entry.py native --board <absolute board.yaml>

BOARD_VIEWER_RUNTIME_AUTHORIZED=3030 BOARD_VIEWER_APP_BIN=<pinned app> \
SACCADE_DB=<absolute db> nix develop .#board-viewer -c \
python3 client/harness/entry.py remote

BOARD_VIEWER_RUNTIME_AUTHORIZED=3030 BOARD_VIEWER_APP_BIN=<pinned app> \
SACCADE_DB=<absolute db> nix develop .#board-viewer -c \
python3 client/harness/entry.py --board <absolute board.yaml> session up
```

The lanes' `--board` is the harness's own argument, not the app's: it
selects the fixture the lane opens through the test-open channel
(`BOARD_VIEWER_TEST_OPEN`, t-24@r11) - the launch argument left the
human surface with the local-path form.

Session commands ride the run-root socket from the last `up`:
`session status | inspect | input click|type|key|wheel|drag |
coords | observe | shot | checks | down`. Acting is real WebDriver
input only, addressed semantically through the inspection
projection's `targets` (stable `data-control` ids); raw coordinates
are an explicit escape hatch. Open happens only the ways a real user
opens: the UI's own open controls, or the lanes' test-open channel.

The remote lane's fixture board and its escape symlink live in
`harness/remote_fixture/`; the tracker wire stand-in lives in
`harness/tracker_mock.py` and speaks exactly the `/view/v1` page
shapes the real server pins on the server side.

What this harness is not: no production probes, no host socket or
status calls, no tracker database access, no SSH to real machines, no
automatic restarts, no supervision beyond one run's lifetime.

## Packaging and the real-client run path

The viewer's packaging lives in `nix/board-viewer.nix`: one
callPackage-able file whose dependencies come from the pkgs it is
called on. The flake builds `packages.board-viewer`,
`packages.board-viewer-web`, the flake checks, and the
`apps.board-viewer` entry through that one derivation, and exposes
the same app as `overlays.default`. `packages.board-viewer` and
`apps.board-viewer` build against the `nixpkgs-client` input, the
nixpkgs revision the human's system flake is built from, so the build
agents test is the build the client runs; the devshell's WebKitGTK
block rides the same input so its `WebKitWebDriver` matches the
packaged app's WebKitGTK. The other outputs and the devshell's lane
tools stay on the flake's own `nixpkgs` pin. Why the overlay: a
binary built against a nixpkgs revision foreign to the host links a
foreign libglvnd/mesa stack and cannot use the host's
`/run/opengl-driver` EGL vendors - the packaged app aborts at
startup. Building against the consumer's own nixpkgs keeps the
WebKitGTK/GL stack on the revision the host system was built from.

Consumer (NixOS configuration):

```nix
nixpkgs.overlays = [ inputs.saccade.overlays.default ];
```

then `pkgs.saccade-board-viewer` is the packaged app (main program
`saccade-board-viewer`, opened on an explicit board path). The
board-viewer devshell's lane runs use the packaged wrapper from
`nix build .#board-viewer` as `BOARD_VIEWER_APP_BIN`.
