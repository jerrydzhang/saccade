// The keys each window answers (round 16's showHelp), one table per
// window kind: the help sheet reads it, and so does anything else that
// names a window's keys. The space commands live beside the acts they
// run (App.svelte's spaceCommands); the which-key menu and the help
// sheet both read that one list. Authored text is ASCII: Enter, not an
// arrow glyph.

export type WindowKind = "viewer" | "tasks" | "agents";

export type HelpContext = {
  kind: WindowKind;
  /** The viewer is in history: h l step time. */
  history: boolean;
  /** The viewer shows a deck: h l step slides. */
  deck: boolean;
};

export function helpKeys(context: HelpContext): Array<[string, string]> {
  if (context.kind !== "viewer") {
    const rows: Array<[string, string]> = [
      ["s", "select (labels, then the menu)"],
      ["Enter", context.kind === "agents" ? "open the session's task board (through the link)" : "open the board (through the link)"],
      ["e", context.kind === "tasks" ? "the task's activity" : "the session's details"],
      ["j k", "scroll"],
      ["gg G", "top, bottom"],
      ["' \"", "jump to a window, window picker"],
      ["?", "this sheet"],
    ];
    if (context.kind === "agents") rows.splice(3, 0, ["a w b", "copy the attach command / workspace path / branch"]);
    return rows;
  }
  return [
    ["s", "select (labels, then the menu)"],
    ["Enter", "the selected part's menu"],
    ["o i w", "open elsewhere: through the link / once / a new window"],
    ["j k", "scroll"],
    ["Ctrl-f Ctrl-b", "page"],
    ["d u", "half page (Ctrl-d Ctrl-u too)"],
    ["gg G", "top, bottom"],
    ["Ctrl-o Ctrl-i", "back, forward"],
    ["+ - =", "zoom, reset"],
    ["h l", context.history ? "step time" : context.deck ? "slides" : "(in history: step time)"],
    ["n N", "changes (history)"],
    ["t H L", "track a part, walk its changes (history)"],
    ["a", "annotate after the part at the reading line"],
    ["Ctrl-Shift-c", "copy source / plain"],
    ["' \"", "jump to a window, window picker"],
    ["?", "this sheet"],
  ];
}
