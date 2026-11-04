// DRAFT: registration API guessed from embedded examples; verify against
// the generated .d.ts beside ~/.config/herdr-app/AGENTS.md, then rename
// to main.js and run: herdr-app check ~/.config/herdr-app/mods/tmuxkeys
// Four commands wrapping pane.go with its direction baked in, so the
// mod's keys.json can bind ctrl-hjkl to bare names.
export default class TmuxKeys extends View {
    init(_props, cx) {
        for (const d of ["left", "right", "up", "down"]) {
            // the registration call to verify: commands and actions are
            // registered in script per the app docs; the exact surface
            // (app.commands.register vs cx.define_action + bind_keys)
            // is named by the .d.ts - do not trust this line's shape
            cx.register_command(`tmux.go.${d}`, (cx2) =>
                cx2.exec("pane.go", { direction: d }));
        }
    }
}
