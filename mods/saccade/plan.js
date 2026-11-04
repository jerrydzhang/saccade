// The pure placement core: id minting, workspace labels, and the tick
// plan. No herdr imports, so tests run it under plain node.
//
// The mechanism (the app's own, defaults/AGENTS.md): a core terminal
// whose labels carry herdr.space/herdr.tab shows up in that space, even
// one started outside the app - unknown ids become a new space or tab,
// and the space's folder is the terminal's cwd. The mod labels terminals
// with a stable per-machine, per-workspace label value and lets the app
// do the rest; it never writes the layout document itself.
//
// Machines: the app shows every connected machine's terminals, remote
// addresses prefixed <machine>/<id>; label writes go to that machine
// with the prefix stripped. The plan is machine-partitioned throughout,
// so a laptop-side app places box-core terminals into the box's own
// layout document. Note the app's terminals() itself is pane-scoped for
// remote machines - a foreign unlabeled remote terminal never reaches
// it - so the view reads each machine's core table directly.

export function freshId(prefix) {
    let hex = "";
    for (let i = 0; i < 4; i++) {
        hex += Math.floor((1 + Math.random()) * 0x100000000)
            .toString(16)
            .padStart(8, "0")
            .slice(-8);
    }
    return prefix + hex;
}

// A remote address carries <machine>/; a local one is bare.
export function machineOf(address) {
    const at = (address ?? "").indexOf("/");
    return at === -1 ? null : address.slice(0, at);
}

export function localId(address) {
    const at = (address ?? "").indexOf("/");
    return at === -1 ? address : address.slice(at + 1);
}

// The rows from one machine's core table, addressed for the app: a
// machine-scoped `hc ls --json` names local ids, and the app knows them
// prefixed with the machine. Parse failures refuse the whole listing -
// a partial read would label half a workspace.
export function parseRemote(stdout, machine) {
    let value;
    try {
        value = JSON.parse(stdout);
    } catch {
        throw new Error(`${machine}: the terminal table is unreadable`);
    }
    if (!Array.isArray(value.terminals)) {
        throw new Error(`${machine}: the terminal table names no rows`);
    }
    const rows = value.terminals;
    return rows.map((row) => ({
        ...row,
        address: `${machine}/${row.address}`,
    }));
}

// The workspace label a cwd belongs to, or null. The root is the
// saccade state root, which holds every project's tracker side by side
// (<repo>/worktrees/<label>), so the label is the segment right after
// the first worktrees component - not the repo, and not the first
// segment.
export function workspaceOf(cwd, root) {
    if (!cwd || !root) return null;
    const base = root.replace(/\/+$/, "");
    if (cwd === base || !cwd.startsWith(base + "/")) return null;
    const segs = cwd.slice(base.length + 1).split("/");
    const at = segs.indexOf("worktrees");
    if (at === -1) return null;
    return segs[at + 1] || null;
}

// The tick plan: which terminals to label, which spaces to rename. Pure,
// so tests can pin it; the view only executes it.
//   rows:    terminal rows (address, cwd, labels, state); remote
//            addresses carry the <machine>/ prefix
//   spaces_: spaces() rows (id, name, named, folder, machine)
//   map:     the persisted "<machine>::<label>" -> { spaceLabel, name }
//            record
export function plan(rows, spaces_, map, root) {
    const jobs = [];
    // one entry per machine and workspace; the label value labels the
    // first terminal (the app then mints the space), and once the space
    // exists its own id and first tab take over, so every later
    // terminal lands inside the space that already stands
    const entryFor = (key, machine, label, spacesNow) => {
        let entry = map[key];
        const existing = spacesNow.filter(
            (s) =>
                (s.machine ?? null) === machine &&
                workspaceOf(s.folder, root) === label,
        );
        const known = existing.find((s) => s.id === entry?.realId) || existing[0];
        if (!entry) {
            entry = { spaceLabel: freshId("spa"), name: null };
            map[key] = entry;
            jobs.push({ kind: "map", key, entry });
        }
        if (known) {
            entry.realId = known.id;
            entry.tabId = known.tabs?.[0]?.id ?? null;
        }
        return { entry, known };
    };
    for (const t of rows) {
        if (t.state === "finished" || t.state === "killed") continue;
        const machine = machineOf(t.address);
        const label = workspaceOf(t.cwd, root);
        if (!label) continue;
        if (t.labels && t.labels["herdr.space"]) continue; // already placed
        const key = `${machine ?? "local"}::${label}`;
        const { entry, known } = entryFor(key, machine, label, spaces_);
        jobs.push({
            kind: "label",
            terminal: t.address,
            localId: localId(t.address),
            machine,
            space: known ? known.id : entry.spaceLabel,
            tab: known?.tabs?.[0]?.id ?? freshId("tb"),
        });
    }
    for (const s of spaces_) {
        const machine = s.machine ?? null;
        const label = workspaceOf(s.folder, root);
        if (!label) continue;
        const key = `${machine ?? "local"}::${label}`;
        const entry = (map[key] ??= { spaceLabel: freshId("spa"), name: null });
        // an unnamed space under a root takes the label (the app's own
        // derived name would only borrow the repo or folder); a space
        // someone named - including one we named before - keeps its
        // name, because named is forever and ours is no exception
        if (s.name !== label && !s.named) {
            jobs.push({ kind: "rename", id: s.id, name: label, key });
        }
    }
    return jobs;
}
