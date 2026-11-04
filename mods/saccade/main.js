// Saccade spaces: one Herdr space per saccade workspace, keyed by the
// worktree path its terminals' cwd carries. The mod labels door-spawned
// core terminals (any terminal whose cwd sits under the worktrees root)
// with herdr.space/herdr.tab - the app's own adoption path, where
// unknown ids become a new space and the space's folder is the
// terminal's cwd - and names the spaces after their workspace label.
// Moves follow workspace switches: keying is by the label, and the
// space's shown folder is its terminals' live cwd. Terminals outside the
// root are never touched; nothing here breaks non-saccade use. This mod
// never writes the layout document: its writes are core labels (hc
// metadata) and the app's own rename command. Config: config.json beside
// this file, { "root": "..." } relative to home or absolute.

import { View, div } from "gpui-kit";
import { v_flex } from "gpui-base";
import { spaces, terminals, shared, setShared, run, hc, readTextFile, home, appHome, theme } from "herdr";
import { freshId, workspaceOf, plan } from "./plan.js";

const TICK_MS = 4000;

export default class SaccadeSpaces extends View {
    init(_props, cx) {
        this.root = null;
        this.busy = new Set(); // addresses with a labeling in flight
        this.rows = []; // the tracked workspaces for the panel
        cx.spawn(async (cx) => {
            await this.load();
            await this.tick();
            cx.notify();
        });
        cx.timer.every(TICK_MS, (cx) => cx.spawn(async (cx) => {
            await this.tick();
            cx.notify();
        }));
    }

    async load() {
        let config = {};
        try {
            const text = await readTextFile(`${appHome()}/mods/saccade/config.json`);
            if (text) config = JSON.parse(text);
        } catch {
            config = {};
        }
        const raw = config.root || "Projects/saccade-workspaces";
        this.root = raw.startsWith("/") ? raw : `${home().replace(/\/+$/, "")}/${raw}`;
    }

    // One placement pass. Every write is a core label (the app's own
    // adoption input) or the app's rename command - never the layout
    // document directly.
    async tick() {
        if (!this.root) return;
        const rows = terminals();
        const spaces_ = spaces();
        const map = shared("saccade.map") || {};
        const jobs = plan(rows, spaces_, map, this.root);
        for (const job of jobs) {
            if (job.kind === "map") {
                map[job.label] = job.entry;
            } else if (job.kind === "rename") {
                const entry = map[job.label];
                if (entry) entry.name = job.name;
                run("space.rename", { id: job.id, name: job.name });
            } else if (job.kind === "label") {
                if (this.busy.has(job.terminal)) continue;
                this.busy.add(job.terminal);
                try {
                    await hc([
                        "label",
                        job.terminal,
                        `herdr.space=${job.space}`,
                        `herdr.tab=${job.tab}`,
                    ]);
                } catch {
                    // the label refused: the next tick retries
                } finally {
                    this.busy.delete(job.terminal);
                }
            }
        }
        setShared("saccade.map", map);
        const spacesNow = spaces();
        this.rows = Object.keys(map)
            .sort()
            .map((label) => {
                const space = spacesNow.find(
                    (s) => workspaceOf(s.folder, this.root) === label,
                );
                return {
                    label,
                    present: !!space,
                    name: space ? space.name : "waiting for the app",
                };
            });
    }

    render() {
        const t = theme();
        const row = (r) =>
            div()
                .px(10)
                .py(4)
                .text_size(11.8)
                .flex_1()
                .min_w(0)
                .text_color(r.present ? t.foreground : t.muted)
                .child(div().truncate().child(r.label));
        const head = div()
            .px(10)
            .pt(8)
            .pb(4)
            .text_size(11)
            .text_color(t.dim)
            .child("SACCADE");
        const body = this.rows.length
            ? v_flex().children(this.rows.map(row))
            : div()
                  .px(10)
                  .py(6)
                  .text_size(11.8)
                  .text_color(t.muted)
                  .child("no saccade workspaces yet");
        return v_flex()
            .size_full()
            .bg(t.canvas)
            .child(head)
            .child(body)
            .child(div().flex_1());
    }
}
