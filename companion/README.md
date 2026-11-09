# The agents companion

`sac-companion`: client tooling on the hub that keeps one tmux session per
tracker - the agents session - holding one window per live agent session,
each an interactive viewer on that agent's Herdr terminal. It reads the
record through `/view/v1/agents` for enrollment and Herdr's stream
(`hc events`) for motion; the tracker supervises nothing and this program
is not the tracker.

Reader: anyone developing or operating the companion. The record law is
the root `AXIOMS.md`; the suite spec's terminal side is t-14@r3 as
clarified by t-24 (the companion attach behaviour there is ratified law,
restated below as this package carries it).

## The law this package carries

- **Normal interactive attach, never `--observe`.** Each window runs
  `hc attach <address>` as its pane process. The human types directly to
  agents in these windows; that is the main way of interacting with them.
- **Size follows the tmux pane.** The attach rides the pane tty, so the
  pane size is the terminal's size; a pane resize is passed on and the
  agent redraws. The core's no-viewer default (120x200) never arises
  while the companion runs.
- **Re-assert on focus.** A second Herdr viewer at another size takes
  the terminal's size while it is attached, last writer winning. When
  the human returns to a companion window, the window's attach resends
  its own size: a per-session tmux hook on window and pane selection
  restarts the pane's own viewer (`respawn-pane -k`, the one kill this
  program makes - its own `hc attach` child on its own pane, never the
  agent), and the fresh attach sends the pane's size - the viewer the
  human is looking at wins. A bare SIGWINCH does not resend (hc sends
  its tty size only when the tty size changes), and a window clamped to
  the client's size cannot change its tty, so the restart is the one
  lever that always reaches the core. Verified against the pinned
  core.
- **No keys, no kill, no remove.** The companion never sends keys into
  an agent terminal, never `hc kill` or `hc rm` anything, never types
  into a pane it does not own, and never closes a window. Its whole
  tmux vocabulary is: create its own session and windows, rename its
  own windows, respawn a dead pane on a live terminal, set options and
  hooks on its own session, and read state.
- **Ended windows stay.** When an agent's terminal finishes, the attach
  prints the core's end line and exits; the window stays showing that
  end until the human closes it. At startup only agents whose terminals
  are live right now are enrolled, so windows for long-finished agents
  are not recreated.
- **Restart re-derives.** The companion keeps no state file. A restart
  re-derives the window set: live agents get windows again, even ones
  the human closed while it was away; within one run a window the human
  closed is never recreated. (A persistent closed-window set is
  deferred until use asks for it - t-24.)
- **Every EOF is a restart signal.** When the Herdr subscription dies,
  the companion re-subscribes for a fresh picture and re-attaches the
  windows of kept terminals that are live again. The record side
  reconnects by retrying with backoff and re-reading the page whole.
- **Focus never moves.** Sessions and windows are created detached;
  renames, respawns and option writes never switch a client.

## Shape

One daemon, three actors: a page watcher long-parks on
`/view/v1/agents?after=<seq>` (landing wake, no timer poll) for
enrollment; a Herdr watcher reads one `hc events --json` subscription
for motion; a reconciler owns every tmux and `hc` read and mutation.

- **Enrollment** (the page, pure record): each registered session's
  incarnation, terminal address, harness, owned tasks (latest claim
  first) and workspace - the workspace derived server-side by the
  tracker's own saved-session header readers, never from Claude Code's
  project directory name.
- **Motion** (the stream): `status` events drive window renames - the
  window title is `i-<n>[ t-<m>[+k]][ <word>]`, owned tasks latest
  claim first; `finished` events mark the end (with its reason).
  Events are matched locally against enrolled addresses; the stream
  alone cannot see an agent, the record alone cannot see a word.
- **One agents session per tracker**, named `agents-<tracker>` from the
  root page's tracker name (sanitized for tmux; `agents-unknown` when
  the server runs `--db` with no repo). Worktrees of one project share
  it; the human's own sessions are never touched.

## Running it

    sac-companion --repo /home/jerry/Projects/saccade

`--server <url>` overrides server discovery (the address hint beside
that repo's database, the same discovery `sac` uses); `--hc <path>`
overrides the herdr-core CLI (ambient `hc`, or `SACCADE_HERDR`);
`--socket <name>` names a private tmux server socket (`tmux -L`) for
tests - by default the daemon works in the default tmux server, where
the human's own sessions live, and only ever addresses its own session.
The companion runs in the environment whose `hc` reaches the right
core; window panes are composed with the same hc path and the
hc-relevant environment the daemon itself holds.

## Tests

`cargo test -p saccade-companion`: the page decoder pinned against the
agreed `/view/v1` literals, the title and argv builders as
inventories, and the reconciler's whole life cycle over a fake tmux
and fake hc: enrollment, rename on the word, end marking, the focus
re-assert vocabulary, restart re-attach, human-closed suppression, and
an inventory pinning that no command the companion composes can send
keys, kill, or remove anything. The end-to-end loop (real tmux server
on a private socket, the pinned core in the herdr-ui sandbox, typed
interaction, sizing, focus re-assert, reconnect, retained ends) is a
phase-gate activity, exercised per `AGENTS.md` with retained logs.
