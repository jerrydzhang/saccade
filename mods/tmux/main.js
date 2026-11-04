// tmux prefix for herdr: ctrl-space opens this mod as a focused overlay,
// the next keystroke fires its mapped command and closes it. The win98
// shipped mod proves the machinery (overlay focus, in-view keys,
// closeOverlay); the exact registration calls below are verified against
// the generated .d.ts beside the laptop AGENTS.md, then herdr-app check.
// The palette stays the escape hatch on its built-in ctrl-shift-p.
import { View, div } from "gpui-kit";
import { closeOverlay, run, theme } from "herdr";

// one-shot chord table: key -> [command, args]
const CHORDS = {
  h: ["pane.go", { direction: "left" }],
  l: ["pane.go", { direction: "right" }],
  k: ["pane.go", { direction: "up" }],
  j: ["pane.go", { direction: "down" }],
  z: ["pane.zoom", {}],
  c: ["tab.new", {}],
  n: ["tab.next", {}],
  p: ["tab.prev", {}],
  x: ["pane.close", {}],
  s: ["space.next", {}],
  a: ["space.prev", {}],
  "%": ["pane.split", { direction: "right" }],
  '"': ["pane.split", { direction: "down" }],
};

for (let i = 1; i <= 9; i++) {
  CHORDS[String(i)] = ["tab.select", { tab: i }];
}

export default class TmuxPrefix extends View {
  // the overlay's whole render: a thin strip saying the prefix is armed,
  // colored from the theme so it belongs to the app's look
  render() {
    const t = theme();
    return div()
      .p(6)
      .bg(t.accent)
      .rounded(4)
      .text_color(t.accent_foreground)
      .text_size(13)
      .child("tmux prefix - h j k l z c n p 1-9 % \" x s a");
  }

  on_key_down(e, cx) {
    const key = e.key; // verify: bare printable name on the event
    const chord = CHORDS[key];
    closeOverlay(props_of(this));
    if (chord) cx.exec(() => run(chord[0], chord[1]));
    return true; // consumed either way: one shot, unknown closes
  }
}

// draft: the app passes the overlay's own props into the view; the win98
// mod reads props().overlay - keep that shape until the .d.ts names it
function props_of(view) {
  return view.props ? view.props() : {};
}
