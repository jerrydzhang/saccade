# Global Pi extensions installer

`nix run .#install-pi-extensions` — run explicitly, from a Saccade checkout —
installs the two production extension files, `websearch.ts` and
`bash-background.ts`, into Pi's global auto-discovery directory
(`~/.pi/agent/extensions/`): no `settings.json` edit, no launch flag, loaded in
every project. It is an operator tool; nothing in Saccade runs it, and
projects never reconfigure Pi on their own.

The installed files are a snapshot of the **pinned `saccade` flake input** (the
rev recorded in this checkout's `flake.lock`), never the checkout's live
working tree. The snapshot is an immutable read-only store path, kept alive
against `nix gc` by an out-link, so what is installed stays byte-stable until
you update it.

## What lands in `~/.pi/agent/extensions/`

- `websearch.ts`, `bash-background.ts` — symlinks resolving through the
  snapshot out-link.
- `.saccade-pi-extensions-snapshot` — the out-link; it is the Nix GC root, so
  the snapshot survives collection while it exists. Deleting it hands the
  snapshot back to the collector.
- `.saccade-pi-extensions.json` — the installer's manifest naming exactly the
  files of its previous installation and the snapshot they resolve to.

Nothing else is touched: unrelated extensions, `settings.json`, and everything
else in your Pi configuration are preserved. A colliding file the installer did
not install itself refuses the install rather than overwrite it.

## Update

Bump the `saccade` input pin in the checkout's `flake.lock` (a config commit),
then run the installer again. It replaces only its own previous installation —
the manifest names those files — and flips the out-link to the new snapshot;
the symlinks resolve through it, so installed content follows without any other
change. Each machine updates when you run the installer there.

## Remove

```
rm ~/.pi/agent/extensions/{websearch.ts,bash-background.ts} \
   ~/.pi/agent/extensions/.saccade-pi-extensions.json \
   ~/.pi/agent/extensions/.saccade-pi-extensions-snapshot
```

After the out-link is gone, garbage collection reclaims the snapshot.

## Pin versus development

- **Installed (global):** the pinned input's files — the release line. If the
  pin predates a file the installer ships, the install fails naming the missing
  file; it never silently falls back to the working tree. Update means moving
  the pin, not running from a dirty checkout.
- **Development (per session):** load the working tree's file directly, e.g.
  `pi -e extensions/websearch.ts` from a checkout. This is what the per-task
  sessions do; it installs nothing.
- **Testing the installer against local source** without changing
  `flake.lock`: `nix run .#install-pi-extensions --override-input saccade
  ~/Projects/saccade`. The override is explicit and temporary; the lock file is
  not written.
