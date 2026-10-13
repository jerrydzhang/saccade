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
  through the forms — the conceded boundary. The wildcard never lands
  in the address hint: `serve` publishes `http://127.0.0.1:<port>`
  beside the served database after a successful bind, and clients
  discover the server from that file — no default port, no localhost
  guess. A scratch `serve --db /tmp/scratch/x.db --port 0` publishes
  beside the scratch file and never disturbs the repo's hint.
- stderr goes to the journal; `journalctl --user -u saccade -p warning`
  is the audit path — and the warns land in the same
  `attempts.jsonl` beside the db as every judged write, so one
  surface greps everything.
- The attempts log is the diagnoser's door: refusals carry the full
  request and the storage-seq cursor they died against. Reproduce a
  refusal with `sac clone --at <cursor> --out <path>`, serve the
  clone, and re-post the logged request.
- Stop is SIGTERM: the server exits 143. It owns no child processes —
  this build runs no sessions. The workspace operations spawn one-shot
  git runs under the writer lock; nothing long-lived is ever kept.

Consumers pin a release with a flake input:
`saccade.url = "github:jerrydzhang/saccade/v0.1.0"` — the lockfile is
the version identity, a bump is a config commit, and from the first
tag onward old logs must always load.
