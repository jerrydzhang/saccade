// Saccade spaces: one Herdr space per saccade workspace, keyed by the
// worktree path its terminals' cwd carries, on every connected machine.
// The mod labels door-spawned core terminals (any terminal whose cwd
// sits under a configured root) with herdr.space/herdr.tab - the app's
// own adoption path, where unknown ids become a new space and the
// space's folder is the terminal's cwd - and names the spaces after
// their workspace label. Remote machines partition the work: a terminal
// the box runs is labeled on the box (the machine prefix stripped), so
// a laptop-side app fills the box's own layout document. Moves follow
// workspace switches: keying is by the label, and the space's shown
// folder is its terminals' live cwd. Terminals outside the roots are
// never touched; nothing here breaks non-saccade use. This mod never
// writes the layout document: its writes are core labels (hc metadata)
// and the app's rename command. Config: config.json beside this file,
// { "roots": ["..."] } absolute, or { "root": "..." } relative to home.

import { View, div } from "gpui-kit";
import { v_flex } from "gpui-base";
import { spaces, terminals, machines, shared, setShared, run, hc, exec, readTextFile, home, appHome, theme } from "herdr";
import { freshId, workspaceOf, parseRemote, plan } from "./plan.js";

const TICK_MS = 4000;

export default class SaccadeSpaces extends View {
    init(_props, cx) {
        this.roots = null;
        this.busy = new Set(); // addresses with a labeling in flight
        this.rows = []; // the tracked workspaces for the panel
        this.remote = false; // a machine besides this one is connected
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
        // the state root saccade itself uses: ~/.local/state/saccade
        // (XDG_STATE_HOME's default shape; config.json overrides for a
        // moved state home)
        let config = {};
        try {
            const text = await readTextFile(`${appHome()}/mods/saccade/config.json`);
            if (text) config = JSON.parse(text);
        } catch {
            config = {};
        }
        this.roots = config.root || `${home().replace(/\/+$/, "")}/.local/state/saccade`;
    }

    // One placement pass. Every write is a core label (the app's own
    // adoption input, on the terminal's own machine) or the app's rename
    // command - never the layout document directly.
    async tick() {
        try {
            await this.place();
        } catch (e) {
            // the diagnostic lane: a throwing tick surfaces its error
            // here instead of dying silently in the runtime
            setShared("saccade.error", String(e && e.stack ? e.stack : e));
        }
    }

    async place() {
        if (!this.roots) return;
        const rows = terminals();
        const spaces_ = spaces();
        const ms = machines();
        this.remote = ms.some((m) => m.name !== null);
        const map = shared("saccade.map") || {};
        setShared("saccade.debug", {
            roots: this.roots,
            count: rows.length,
            first: rows[0] ?? null,
            machines: ms,
        });
        const jobs = plan(rows, spaces_, map, this.roots);
        for (const job of jobs) {
            if (job.kind === "map") {
                map[job.key] = job.entry;
            } else if (job.kind === "rename") {
                const entry = map[job.key];
                if (entry) entry.name = job.name;
                run("space.rename", { id: job.id, name: job.name });
            } else if (job.kind === "label") {
                if (this.busy.has(job.terminal)) continue;
                this.busy.add(job.terminal);
                try {
                    if (job.machine) {
                        // the box labels its own terminal: the machine
                        // prefix strips, and hc runs there
                        await exec(
                            "hc",
                            [
                                "label",
                                job.localId,
                                `herdr.space=${job.space}`,
                                `herdr.tab=${job.tab}`,
                            ],
                            { on: job.machine },
                        );
                    } else {
                        await hc([
                            "label",
                            job.terminal,
                            `herdr.space=${job.space}`,
                            `herdr.tab=${job.tab}`,
                        ]);
                    }
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
            .map((key) => {
                const [machinePart, label] = key.split("::");
                const space = spacesNow.find(
                    (s) =>
                        (s.machine ?? "local") === machinePart &&
                        workspaceOf(s.folder, this.roots) === label,
                );
                const where = machinePart === "local" ? "" : ` - ${machinePart}`;
                return {
                    label: label + where,
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
        const hint =
            !this.remote && this.rows.length === 0
                ? div()
                      .px(10)
                      .py(6)
                      .text_size(11)
                      .text_color(t.muted)
                      .child(
                          "no saccade workspaces yet - if your saccade terminals run on another machine, add it: + Add machine below",
                      )
                : div()
                      .px(10)
                      .py(6)
                      .text_size(11.8)
                      .text_color(t.muted)
                      .child("no saccade workspaces yet");
        const body = this.rows.length
            ? v_flex().children(this.rows.map(row))
            : hint;
        return v_flex()
            .size_full()
            .bg(t.canvas)
            .child(head)
            .child(body)
            .child(div().flex_1());
    }
}
