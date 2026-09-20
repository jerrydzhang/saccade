# Serving as a service

The dogfood server is infrastructure, not a dev artifact: it runs the
pinned release (`SACCADE_RELEASE` in the justfile), started with
`just serve`. The workspace build serves only smokes (`just serve-dev`).

The unit lives in the machine's own config (declaratively, as a flake
input pin); this is the argument line it runs:

```
ExecStart=sac serve --repo /home/jerry/Projects/saccade --bind 0.0.0.0 --port 8811
Environment=RUST_LOG=info
```

- `--repo` names the tracker: the process may start anywhere, so the
  db's repo is declared, not discovered from the working directory.
- `--bind 0.0.0.0` spans loopback (CLI default) and the tailnet
  (dashboard). Any device that can reach the port writes at human tier
  through the forms — the conceded boundary.
- stderr goes to the journal; `journalctl --user -u saccade -p warning`
  is the audit path.
- Stop is SIGTERM: the server kills its live runs and exits 143. The
  two-stroke Ctrl-C courtesy is for terminals, not services.

Consumers pin a release with a flake input:
`saccade.url = "github:jerrydzhang/saccade/v0.1.0"` — the lockfile is
the version identity, a bump is a config commit, and from the first
tag onward old logs must always load.
