// The pure placement core: id minting, workspace labels, and the tick
// plan. No herdr imports, so tests run it under plain node.
//
// The mechanism (the app's own, defaults/AGENTS.md): a core terminal
// whose labels carry herdr.space/herdr.tab shows up in that space, even
// one started outside the app - unknown ids become a new space or tab,
// and the space's folder is the terminal's cwd. The mod labels terminals
// with a stable per-workspace label value and lets the app do the rest;
// it never writes the layout document itself.

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

// The workspace label a cwd belongs to, or null when it sits outside the
// worktrees root: the first path segment under the root is the label.
export function workspaceOf(cwd, root) {
    if (!cwd || !root) return null;
    const base = root.replace(/\/+$/, "");
    if (!cwd.startsWith(base + "/")) return null;
    const label = cwd.slice(base.length + 1).split("/")[0];
    return label || null;
}

// The tick plan: which terminals to label, which spaces to rename. Pure,
// so tests can pin it; the view only executes it.
//   rows:    terminals() rows (address, cwd, labels, state)
//   spaces_: spaces() rows (id, name, named, folder)
//   map:     the persisted label -> { spaceLabel, name } record
export function plan(rows, spaces_, map, root) {
    const jobs = [];
    // one entry per workspace; the label value labels the first
    // terminal (the app then mints the space), and once the space
    // exists its own id and first tab take over, so every later
    // terminal lands inside the space that already stands
    const entryFor = (label, spacesNow) => {
        let entry = map[label];
        const existing = spacesNow.filter(
            (s) => workspaceOf(s.folder, root) === label,
        );
        const known = existing.find((s) => s.id === entry?.realId) || existing[0];
        if (!entry) {
            entry = { spaceLabel: freshId("spa"), name: null };
            map[label] = entry;
            jobs.push({ kind: "map", label, entry });
        }
        if (known) {
            entry.realId = known.id;
            entry.tabId = known.tabs?.[0]?.id ?? null;
        }
        return { entry, known };
    };
    for (const t of rows) {
        if (t.state === "finished" || t.state === "killed") continue;
        const label = workspaceOf(t.cwd, root);
        if (!label) continue;
        if (t.labels && t.labels["herdr.space"]) continue; // already placed
        const { entry, known } = entryFor(label, spaces_);
        jobs.push({
            kind: "label",
            terminal: t.address,
            space: known ? known.id : entry.spaceLabel,
            tab: known?.tabs?.[0]?.id ?? freshId("tb"),
        });
    }
    for (const s of spaces_) {
        const label = workspaceOf(s.folder, root);
        if (!label) continue;
        const entry = (map[label] ??= { spaceLabel: freshId("spa"), name: null });
        // an unnamed space under the root takes the label (the app's
        // own derived name would only borrow the repo or folder); a
        // space someone named - including one we named before - keeps
        // its name, because named is forever and ours is no exception
        if (s.name !== label && !s.named) {
            jobs.push({ kind: "rename", id: s.id, name: label, label });
        }
    }
    return jobs;
}
