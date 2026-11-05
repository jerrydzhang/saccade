# Headless Herdr UI development

Reader: agents and humans developing native Herdr mods on Linux. This is a local,
private test desktop, not the person's client and not a tracker launch door.
The flake package supplies the pinned app/core, software-rendering libraries,
Weston, Xwayland, and Python Xlib. No global install or populated GUI shell is
required.

## Start and inspect

```sh
just herdr-ui up
just herdr-ui info
just herdr-ui status
just herdr-ui shot
```

Equivalent outside just: `nix run .#herdr-ui -- <arguments>`.

`up` reports a private app home and log directory. Repeating it while the whole
sandbox is running is a no-op. After a partial failure, use `down`, inspect logs,
and retry; it does not silently adopt another app or core. `info` does not start
a core or display.

Use `--name demo` before any command for another sandbox:

```sh
just herdr-ui --name demo up
just herdr-ui --name demo shot /tmp/demo.png
just herdr-ui --name demo down
```

The default storage is `/var/tmp/saccade-herdr-ui-<uid>/<name>`. Advanced callers
can use `--root /absolute/private/directory` instead; the helper refuses to adopt
a nonempty directory without its own ownership record. Keep root paths short
because native Unix sockets have path-length limits.

## Develop a mod

Read the reported app home's installed `AGENTS.md`, `herdr.d.ts`, and
`gpui-kit.d.ts`; those describe the pinned app, unlike current upstream docs.
Personal mods live in `APP_HOME/mods/<name>/`, with `main.js` and `mod.json`.
Edit them directly or copy the mod being developed there. Do not edit shipped
mods or generated guides/types.

```sh
just herdr-ui check /absolute/path/to/APP_HOME/mods/my-mod
just herdr-ui commands
```

To display a mod without starting any terminal, create a personal layout at
`APP_HOME/layouts/demo.json`:

```json
{
  "id": "demo-root",
  "type": "block",
  "content": {"mod": "my-mod"}
}
```

```sh
just herdr-ui do layout.switch '{"name":"demo"}'
just herdr-ui status
just herdr-ui shot /tmp/demo.png
```

Mods/layouts hot-reload; no rebuild or push per edit. A broken edit can leave the
last good view visible, so check errors as well as the screenshot. Open and
inspect the saved image; a successful shot command is not visual validation.
The renderer is native GPUI, not a browser. Test actual LaTeX rather than assuming
that Markdown rendering supports it.

## Input

Coordinates are relative to the Herdr window, not the compositor's screen.
Xwayland's synthetic device needs a warm-up. The helper handles that, and a
sequence keeps one connection across related actions:

```sh
just herdr-ui input click 160 317
just herdr-ui input type 'a review question'
just herdr-ui input key Return
just herdr-ui input key a Control_L
just herdr-ui input sequence '[["click",160,317],["type","a review question"],["key","Return"]]'
```

`click` uses the left button; `move`, `type`, and `key` are also available.
Typing is limited to the virtual X keyboard's keymap; unsupported characters
fail rather than being transliterated. This is test input, not a general Unicode
clipboard implementation. Do not assume a failed sequence rolled back earlier
input actions. Inspect the resulting data and capture another frame.

The compositor's display, runtime directory, private core session, and app home
are selected automatically for every command. Do not point xdotool at the
person's display or manually reuse a display number from an older run.

## Stop and recover

```sh
just herdr-ui down
```

Shutdown closes the private app, stops its owned core, then terminates remaining
owned sandbox processes, including the compositor. A malformed ownership record
or foreign process is never permission to signal it. `down` is idempotent and
retains mods, edits, logs, and core state. After inspecting or publishing useful
evidence, manually remove a stopped sandbox if a fresh configuration is needed.
No reset command deletes your working mod files implicitly. Role logs are
replaced on the next start; preserve evidence before restarting if needed.

This isolates processes and configuration for trusted development; it is not a
security boundary for untrusted mods running as your user. Child processes get
private HOME/XDG directories and no inherited session bus or SSH agent. The
helper seeds core notes off, preserves subsequent core-config edits, and manages
the app's core/sound/telemetry settings.

The sandbox disables sound/telemetry, skips first-run onboarding, pins its core,
and intercepts `open`/`xdg-open` through its PATH to `opened.log` rather than
opening links in the real browser. Do not add production remote machines to the
sandbox; their terminals are not test fixtures.

## Checks and limits

```sh
just herdr-ui-test
nix build .#checks.x86_64-linux.herdr-ui
```

The fake-process contract tests require no display and cover lifecycle reuse,
isolation, startup failures, shutdown fallback, and foreign-process/directory
refusal. The package is Linux-only and uses `/proc` for ownership checks.

The local headless loop does not establish remote artifact transport, laptop
focus/clipboard behavior, accessibility, or deployment-shape fidelity. The
person's real-client gate remains separate. This helper does not implement a
blackboard renderer, native math, streaming, remote viewing, or browser embedding.
