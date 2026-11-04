// tmux prefix for herdr, built on the win98 menu's proven recipe:
// ctrl-space (the person's keys.json, "tmux.prefix") reaches this mod's
// nextCommand loop in its zero-size layout host; the loop opens a popup
// whose init focuses its own handle; the chord keys arrive as actions -
// bound here in the mod's key context, exactly as the shipped win98
// menu binds escape - and each fires its command and closes.
import { View, div } from "gpui-kit";
import { v_flex } from "gpui-base";
import { openOverlay, closeOverlay, nextCommand, props, run, theme } from "herdr";

const CONTEXT = "TmuxPrefix";

// one-shot chords: keystroke -> [command, args]
// action-safe chord ids: [action id, keystroke, command, args]
const CHORDS = {
  h: ["pane.go", "left"],
  l: ["pane.go", "right"],
  k: ["pane.go", "up"],
  j: ["pane.go", "down"],
  z: ["pane.zoom"],
  c: ["tab.new"],
  n: ["tab.next"],
  p: ["tab.prev"],
  x: ["pane.close"],
  s: ["space.next"],
  a: ["space.prev"],
  pct: ["%", "pane.split", "right"],
  quote: ['"', "pane.split", "down"],
  d1: ["1", "tab.select", "1"],
  d2: ["2", "tab.select", "2"],
  d3: ["3", "tab.select", "3"],
  d4: ["4", "tab.select", "4"],
  d5: ["5", "tab.select", "5"],
  d6: ["6", "tab.select", "6"],
  d7: ["7", "tab.select", "7"],
  d8: ["8", "tab.select", "8"],
  d9: ["9", "tab.select", "9"],
};

export default class TmuxPrefix extends View {
  init(p, cx) {
    this.p = p ?? {};
    this.focus = cx.focus_handle();
    if (this.p.overlay) {
      // this copy is the popup: take the keyboard now, and speak its
      // keys as actions in this mod's context, the win98 menu's recipe;
      // the overlay opens from a background task, so an init-time focus
      // lands before mount and is reconciled away - the prompt mod's
      // timer retry is the documented cure for exactly that race
      this.focus.focus();
      cx.timer.after(600, () => this.focus.focus());
      cx.timer.after(1400, () => this.focus.focus());
      try {
        cx.bind_keys(
          Object.entries(CHORDS).map(([id, chord]) => ({
            keystroke: chord[0],
            action: `tmux_${id}`,
            context: CONTEXT,
          })),
        );
      } catch {}
    } else {
      // this copy is the layout host: listen for the bound key forever
      cx.spawn(async () => {
        for (;;) {
          const c = await nextCommand();
          if (c === "tmux.prefix" || c === "prefix") {
            openOverlay({ mod: "tmux", kind: "popup", frame: "none", at: { x: 24, y: 24 } });
          }
        }
      });
    }
  }

  fire(id) {
    closeOverlay(props().overlay);
    const chord = CHORDS[id];
    if (chord) run(chord[1], chord.slice(2));
  }

  render(_cx) {
    const t = theme();
    let el = v_flex()
      .id("tmux-prefix")
      .key_context(CONTEXT)
      .track_focus(this.focus)
      .on_key_down((e, cx) => {
        cx.stop_propagation();
        closeOverlay(props().overlay);
        const key = e.key_char ?? e.key;
        const hit = Object.values(CHORDS).find((c) => c[0] === key);
        if (hit) run(hit[1], hit.slice(2));
      })
      .p(8)
      .bg(t.popover)
      .rounded(6)
      .text_color(t.foreground)
      .text_size(12)
      .child(div().text_size(10).text_color(t.dim).child("tmux prefix"));
    for (const id of Object.keys(CHORDS)) {
      el = el.on_action(`tmux_${id}`, () => this.fire(id));
    }
    return el;
  }
}
