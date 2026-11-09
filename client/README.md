# Saccade app

The Saccade suite's native app (Svelte 5 inside Tauri). Its windows
hold the instruments - the viewer, tasks and agents (t-14). It reads the
record through `/view/v1`, keeps window states in the tracker's window
store (`/api/v1/windows`), and writes only what t-14 ratified: task-board
annotations published as one commit, and annotation components into
explainer files. The viewer is the instrument that reads boards and
files; the app is the whole.

## Run

```sh
# the fast development loop: typecheck, pure contract tests, the
# bridge crate's contract tests, build
python3 harness/entry.py check-fast

# the native client's own guard tests (Tauri shell, no UI start)
nix develop .#saccade-app -c bash -c 'cd src-tauri && cargo test'
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

**The suite's writes.** Two, each ratified, nothing else:

- **The record is written once**: publishing task-board annotations
  (t-14, the law landed with the commit that builds the act). `a`
  drafts in place on the current revision of a task board - drafts
  ride the window's state in the tracker's window store (t-24@r20,
  t-28); before the store landed they persisted on the device, and
  the one-time mover takes each device's interim drafts once, per task,
  into its attached state showing that task or into a new detached
  state, marking the interim store moved only after the window store
  confirms, the rows staying and nothing reading them again - and a
  draft is never lost on a refused publish. `space p` publishes the
  drafts of the window it is pressed in, together as ONE commit with
  ONE message through the commit door, under the HUMAN's own identity:
  the human tier and, as the actor, the client configuration's override
  or the device account name as `sac` derives it without SACCADE_ACTOR
  (`$USER`, else "human"), never refusing for lack of it. The published
  board.yaml is the shown revision's authored text plus annotation
  components and nothing else - byte-level insertions at the parts'
  own positions. Before the POST the viewer re-reads the current
  revision and refuses, keeping the drafts, when a revision landed
  after its read; an offered board refuses, naming the
  reject-with-grounds CLI step. Only the current revision of a task
  board drafts; a pinned older revision refuses, saying where
  annotation is possible (t-24@r11).
- **An explainer board file is written once**: its annotation submit
  (t-14 ratified item 3, t-24@r21). An explainer is a bare-path file
  that decodes as board grammar 1 - a file on the tracker's machine,
  never a named `machine:` path, never a fixture, never a lane's host
  reader. `a` drafts in place; the editor's save IS the submit: one
  annotation component into the file, nothing pooled (`space p` stays
  task-board-only), no record, no identity - a file write lands no
  attribution, and the bridge log line (operation, path, outcome) is
  the act's trace. The edit rides the local herdr core's `file.edit` -
  exact-once, all-or-none - on the machine the bare-path reads already
  resolve: the local core on the hub, the declared tracker machine
  elsewhere. Before anything is sent the client applies the edit to
  the shown text and decodes the result as grammar 1, refusing with
  the annotation editor still open when the result would not decode or
  the annotation would not sit beside the part it addresses; old text
  that does not occur exactly once refuses before the wire. The
  staleness guard is whole-file byte equality: a file that changed
  since this window read it refuses the write, keeps the annotation
  TEXT on the device keyed by the path (the only thing ever kept;
  t-21 was built before the window store landed, and moving this text
  into the window state, as t-24@r21 A6 says, is still open), drops the placement and says so - the next `a` on
  the freshly read file opens the editor with the kept text. The
  wire's own exact-once refusal is the same refusal reached through
  the other half of the guard. The deck is this package's other t-21
  surface, stated in its own section below.

**The deck** (t-14 "The viewer", t-3 "Agent-owned viewer", t-24@r21).
When an agent holds the window's address it may cue slides. The address
is a herdr-core document's name - `deck-` plus 96 random bits (24
lowercase hex, inside the core's 32-character name rule, probed against
the pinned core 0.1.17), on the machine the bare-path rule resolves:
the hub's own core there, the declared tracker machine elsewhere.
Holding the address is the permission: nothing lists documents and the
name is unguessable, so an untold agent can never reach the deck. The
viewer offers the deck when it can reach that core (at start and on the
next open after a failure), shows the address in the header, and its
copy act copies the exact cue command an agent runs verbatim -
`doc.get`, append the location, `doc.set` with that version, retry on a
conflict, the slide number printed back:

```
python3 - '<location>' <<'DECK'
import json, subprocess, sys
name = "deck-<96 bits>"
def call(method, params):
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    out = subprocess.run(["hc", "connect"], input=line + "\n", capture_output=True, text=True)
    return json.loads(out.stdout)
doc = call("doc.get", {"name": name})["result"]
while True:
    slides = (doc.get("value") or [])
    slides.append(sys.argv[1])
    done = call("doc.set", {"name": name, "version": doc["version"], "value": slides})
    if "error" not in done:
        print("slide", len(slides))
        break
    doc = call("doc.get", {"name": name})["result"]
DECK
```

The document is a JSON array of location strings and holds locations
only - never the human's position or history; reading it is the
agent's read-back, a slide's number is its position in the list. The
cue feed is wake-driven: one `events.subscribe` connection read
through bounded holds (`deck_wait`, the hold pattern the look's wait
rides), never a timer poll. The subscription is per connection and
unlimited (pinned against the pinned core: plain and hello-identified
subscribers coexist and fan out), so the companion's own feed and the
deck's coexist as separate processes. Two pinned-core 0.1.17 limits
shape the deck, both probed (`harness/deck_probe.py`,
`harness/deck_relay_probe.py`, retained logs under /var/tmp/saccade-t21):
the promised 15 s heartbeats flow only while the feed is being read,
and a relayed write reaches the owning core's document but never the
viewer's feed when the write travelled through another core's relay
(t-24@r22: unsupported, no polling workaround). The ratified shape is
unaffected: the deck lives on the hub's core and the hub's agents cue
through their own core - `hc connect`, the copied command. The viewer
deletes the document when the window closes (the shell's close hook,
best effort and logged); the address is dead from then on, and a
document deleted from outside ends the deck.

A cue waits unless the human is on the latest slide - "on the latest"
only while the window shows that slide as cued; after following a ref
away, a cue waits and the HUD states it plainly (`slide 4 waiting`, in
the state colour, no badge). The one exception: a deck's first cue
lands on whatever the window shows - handing out the address is asking
for the slideshow. A slide is fixed when cued: a board is pinned by
revision when shown (an agent wanting an exact one cues `t-N@rK`); a
file is read when the cue lands and its snapshot held for the life of
the deck. Slides are ordinary views with full board behaviour -
select, search, history, marginalia, zoom all work on them; a file the
viewer cannot yet interpret shows its own structured refusal. `h` `l`
move between slides (in history they keep stepping time); digits 1-9
jump (single presses only - beyond 9 steps with h l); deck motion
never touches the back/forward stacks, which stay the human's own
trail, and the human adds no slides.

The validation split: admission is the shared crate's job on the Tauri
side (`validate_board`, a path dependency), and the native lanes refuse
on its verdict; the frontend keeps only the presentation decoder, which
mirrors the crate's contract exactly so the browser lane (no bridge)
behaves identically. Parity over the shared fixture corpus is pinned by
tests on both sides: this repository's node tests and the
`viewer-native-check` cargo suite read the same `fixtures/` files.

## The instruments: the pipeline and the glance (t-14 step 3)

The entry pages are tracker locations too: `<tracker>/tasks` and
`<tracker>/agents`. They render as instruments - the pipeline and the
agents glance - reading `/view/v1` only; `/api/v1` stays `sac`'s own.

- **Tasks** (`t-14 "Tasks and agents"`): the page's four recorded
  states section the view - waiting, in flight, awaiting judgment,
  concluded last 24h (the window the page computes; staleness is
  rendering). A row is task, title, agents (each agent's live state
  shape joined through the agents page's terminal addresses and the
  words), latest commit, age. A row selects on click or by label;
  Enter goes through the link to the board; the row menu's `e` shows
  the task's activity - a lane on a 6h/24h axis (the last 6 hours
  taking the right two thirds) and the events beside it, from the
  history page. No dispatch: the suite never launches an agent
  (ratified).
- **Agents**: the page's sessions, sectioned by Herdr's word - error,
  blocked, working, idle, done - within a section ordered by task,
  then workspace. A row is the state shape, the incarnation, Herdr's
  msg and progress exactly as the program sent them (nothing when
  nothing was sent, t-24@r19), the task, and the time. Enter opens the
  owned task's board through the link; the row menu's `e` shows the
  session's details (workspace, harness, last commit); the copies are
  `a` the attach command, `w` the workspace path, `b` the branch - the
  branch read from the workspace's own `.git/HEAD` through the
  tracker's machine, never invented. Proposed and built as proposed:
  available workspaces with no agent hold no row - the view is the
  sessions, nothing else keeps it open.
- **The live words ride one shared subscription, by the client's own
  design** (t-24@r23: subscriptions are per connection and unlimited
  on herdr-core 0.1.17 - the earlier one-subscription rule is
  withdrawn; sharing one across windows remains allowed where
  simpler, and simpler is what this client holds): the shell keeps
  exactly one subscription per core per app process, and every window
  and instrument reads the table it keeps - one reconnect, one
  backoff, one table to trust, however many windows read it. The
  table clears on every (re)connect and
  on the core's restart vocabulary; words refill from the stream.
  Consumers wake through one bounded hold (the wait_look_config
  pattern, never a timer poll). A word Herdr does not report is shown
  as absent: those sessions ride the absent section, shapeless. The
  feed dials nothing when setup resolved no core. The lanes inject
  words through the inspection setter (`setWords`), test-only
  machinery named as such.
- Neither instrument opens terminals; an agent's attach command, a
  workspace's path and a branch are copied (`a`/`w`/`b` in the row's
  menu).

Keys:

- `s` puts labels on what can be selected (lowercase on prose, uppercase
  on the rest); typing a label selects its target and opens its one
  context menu beside it; typing anything else narrows the text targets
  by their text; right-click opens the same menu at the pointer; `Enter`
  opens the selected target's acts; `?` lists keys and commands on
  demand
- motion: `j` `k` lines, `Ctrl-f` `Ctrl-b` pages, `Ctrl-d` `Ctrl-u` and
  `d` `u` half pages, `gg` `G` ends, `Ctrl-o` `Ctrl-i` back and forward
- `space` leads the viewer's own commands: `space space` the picker
  (id, name, word start, substring, scattered letters; recency ties;
  the open item last; kind and name only; a typed location always
  resolves), `space h` history, `space m` marginalia, `space p`
  publish, `space l` link this window by labels, `space u` unlink
- the window jump: `'` labels on every visible suite window, typing
  one jumps (the key is a placeholder, t-14); `"` windows on this
  device, then detached ones - Enter focuses an attached window or
  resumes a detached state here, `^x` twice discards a state on every
  device
- the two acts, never a mode (t-14): `Enter` (or a plain click) is
  the primary act and the viewer follows here; `o` (or shift- or
  middle-click) opens elsewhere through the link - with no link it
  asks once (labels on visible windows, `n` new, `/` a detached one)
  and the answer becomes the link; `i` opens elsewhere once without
  changing the link; `w` always opens a new window. The routing is
  the instrument's declaration (lib/acts.ts): the viewer follows
  here; the tasks and agents instruments (t-18) declare
  Enter-through-the-link and arrive without rework
- in history: `h` `l` step time, `n` `N` jump to changes, `t` tracks the
  reading-height part, `H` `L` walk its changes
- zoom: `+` `-` `=`, carried by the window's state (t-24@r20);
  figures and diagrams enlarge as their primary act; an incorporation
  expands to exactly its cited components
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
  directory>/dev.saccade.app/config.toml`. It names exactly
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
viewer never adds machines, logs in or updates a core; those are the
human's own host setup. The one call beyond the read face is the
explainer annotation submit's `file.edit` (the write section above) -
never machine management, never anything but annotation components into
a shown board file. Machine states are structured,
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
channel (`SAC_APP_TEST_OPEN`, one location at startup riding the
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

## The window and its stored state (t-14 "The suite", t-24@r20)

One window, one window state: what the window shows, the part at the
reading height, its zoom, its drafts - kept on the tracker's server,
outside the record, under /api/v1/windows (the tracker README carries that
surface's law). The client's own half:

- **The device is an opaque client-generated id**, generated once and
  persisted on the device; the window store stores and compares it and never
  interprets it (t-24@r20: not the machine id).
- **The binding is the shell's**: the Tauri shell keeps one
  label-to-state pairing per window, so a window's state survives a
  frontend reload and the shell that closes, jumps and links windows
  knows the pairing. A window that has bound no state creates one
  (the id client-generated and opaque) and binds it. A page with no
  shell (the browser lane) has no binding and runs unbacked.
- **Sync waits, never polls**: the window long-polls its state with
  `?after=` on the window store's own sequence, applies every answer that is
  not its own echo (a write's landing sequence names it - a device
  never jumps its own view), and on a dropped connection retries with
  backoff and re-reads the state whole. Position writes settle first:
  the reading height when it stops moving, zoom when the steps stop,
  drafts when written - never per scroll frame or per key press. A
  settled write that changes nothing is not sent.
- **A window with no tracker server runs unbacked**, like every read
  face without one; its drafts stay in memory, which is still never a
  lost draft, and the move waits for a start with the window store. The server
  setting appears once, typed in the header - a window that started
  before it retries on a quiet timer, and the window store arrives
  without a restart.

## Layout

- `src/lib/` - read model, decode, Markdown pipeline, projection, copy
  semantics, sources, view model, inspection projection, DOM renderer,
  hydration, diagram overlay, the /view/v1 page client (`pages.ts`)
  and the tracker read source over it (`tracker.ts`)
- `src/App.svelte` - the shell; one view model feeds the human UI, the
  machine-readable inspection (`window.__saccadeApp`) and the copy
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
authorization recorded on t-71 (`SAC_APP_RUNTIME_AUTHORIZED=3030`
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
  Behind `SAC_APP_BROWSER_REFERENCE_HTML` (an absolute, readable
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
(`/var/tmp/saccade-app-<uid>/<run>/`).

Invocations (each bounded; on timeout the supervisor is stopped and
the run is a refusal; no automatic retry):

```sh
SAC_APP_RUNTIME_AUTHORIZED=3030 SACCADE_DB=<absolute db> \
nix develop .#saccade-app -c python3 client/harness/entry.py preflight

SAC_APP_RUNTIME_AUTHORIZED=3030 SACCADE_DB=<absolute db> \
nix develop .#saccade-app -c python3 client/harness/entry.py browser

SAC_APP_RUNTIME_AUTHORIZED=3030 SAC_APP_BIN=<pinned app> \
SACCADE_DB=<absolute db> nix develop .#saccade-app -c \
python3 client/harness/entry.py native --board <absolute board.yaml>

SAC_APP_RUNTIME_AUTHORIZED=3030 SAC_APP_BIN=<pinned app> \
SACCADE_DB=<absolute db> nix develop .#saccade-app -c \
python3 client/harness/entry.py remote

SAC_APP_RUNTIME_AUTHORIZED=3030 SAC_APP_BIN=<pinned app> \
SACCADE_DB=<absolute db> nix develop .#saccade-app -c \
python3 client/harness/entry.py --board <absolute board.yaml> session up
```

The lanes' `--board` is the harness's own argument, not the app's: it
selects the fixture the lane opens through the test-open channel
(`SAC_APP_TEST_OPEN`, t-24@r11) - the launch argument left the
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

The viewer's packaging lives in `nix/saccade-app.nix`: one
callPackage-able file whose dependencies come from the pkgs it is
called on. The flake builds `packages.saccade-app`,
`packages.saccade-app-web`, the flake checks, and the
`apps.saccade-app` entry through that one derivation, and exposes
the same app as `overlays.default`. `packages.saccade-app` and
`apps.saccade-app` build against the `nixpkgs-client` input, the
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

then `pkgs.saccade-app` is the packaged app (main program
`saccade-app`, opened on an explicit board path). The
saccade-app devshell's lane runs use the packaged wrapper from
`nix build .#saccade-app` as `SAC_APP_BIN`.
