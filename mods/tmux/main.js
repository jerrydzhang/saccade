// tmux prefix for herdr: ctrl-space (bound in keys.json as "tmux:prefix")
// opens this mod as a small overlay that takes the keyboard; the next
// keystroke fires its mapped command and closes the overlay. All calls
// are the app's own documented surfaces: nextCommand delivers the
// keys.json binding, openOverlay shows the mod, content_focus_handle
// takes the keyboard, run executes an app command.
import { View, div } from "gpui-kit";
import { openOverlay, closeOverlay, nextCommand, props, run, theme } from "herdr";

// one-shot chord table: printable key -> [command, args]
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
  "%": ["pane.split", "right"],
  '"': ["pane.split", "down"],
};

export default class TmuxPrefix extends View {
  init(_props, cx) {
    this.focus = cx.focus_handle();
    // the command loop: every press of the bound key arrives here
    cx.spawn(async () => {
      for (;;) {
        const c = await nextCommand();
        if (c === "tmux:prefix" || c === "prefix") {
          openOverlay({
            mod: "tmux",
            kind: "window",
            frame: "none",
            key: "tmux-prefix",
          });
        }
      }
    });
    // a window-kind overlay does not take the keyboard by itself
    cx.spawn(async () => {
      this.focus.focus();
    });
  }

  render(_cx) {
    const t = theme();
    return div()
      .content_focus_handle(this.focus)
      .on_key_down((e, _cx) => {
        closeOverlay(props().overlay);
        const key = e.key_char ?? e.key;
        const hit = CHORDS[key] ?? (/^[1-9]$/.test(key) ? ["tab.select", key] : null);
        if (hit) run(hit[0], hit.slice(1));
      })
      .p(8)
      .bg(t.accent)
      .rounded(4)
      .text_color(t.accent_foreground)
      .text_size(13)
      .child("prefix - h j k l | z | c n p 1-9 | % \" | x | s a");
  }
}
