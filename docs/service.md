# Serving as a service

The dogfood server is infrastructure, not a dev artifact: the flake's
devenv process manager runs it from the pinned release, started with
`devenv up` in the project. The `sac` process is declared in
`flake.nix`; this is the line it runs:

```
sac serve --repo /home/jerry/Projects/saccade --bind 0.0.0.0 --port 8811
```

with `restart.on = "on_failure"` (max 10) and a readiness probe on
`GET /` at port 8811.

- `--repo` names the tracker: the shell may sit anywhere, so the db's
  repo is declared, not discovered from the working directory — and
  the declared root carries the workspace Git effects, artifact store,
  and console chrome, whatever cwd the server was started from.
- `--bind 0.0.0.0` spans loopback (CLI default) and the tailnet
  (dashboard). Any device that can reach the port writes at human tier
  through the forms — the conceded boundary. The wildcard never lands
  in the address hint: `serve` publishes `http://127.0.0.1:<port>`
  beside the served database after a successful bind, and clients
  discover the server from that file — no default port, no localhost
  guess. A scratch `serve --db /tmp/scratch/x.db --port 0` publishes
  beside the scratch file and never disturbs the repo's hint.
- The process manager holds the process output; the warns also land in
  the same `attempts.jsonl` beside the db as every judged write, so
  one surface greps everything.
- The attempts log is the diagnoser's door: refusals carry the full
  request and the storage-seq cursor they died against. Reproduce a
  refusal with `sac clone --at <cursor> --out <path>`, serve the
  clone (`--db` names it), and re-post the logged request.
- Stop is the process manager stopping the process. The server owns no
  child processes — this build runs no sessions. The workspace
  operations spawn one-shot git runs under the writer lock; nothing
  long-lived is ever kept.

Consumers pin a release with a flake input:
`saccade.url = "github:jerrydzhang/saccade/v0.1.0"` — the lockfile is
the version identity, a bump is a config commit, and from the first
tag onward old logs must always load.
