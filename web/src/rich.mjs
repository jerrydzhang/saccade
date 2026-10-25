import { StateField } from "@codemirror/state";
import { Decoration, EditorView } from "@codemirror/view";

export function richDocuments(buffer) {
  function links(state) {
    if (buffer.kind === "draft") return Decoration.none;
    const ranges = [];
    const text = state.doc.toString();
    for (const regex of [/^home: (t-\d+)/gm, /^\[\^\d+\]: (c-\d+)/gm, /^(c-\d+)  (?:note|offer|accept|reject|use|t-\d+)/gm]) {
      for (const match of text.matchAll(regex)) {
        const from = match.index + match[0].indexOf(match[1]);
        if (from >= buffer.bodyFrom && from < buffer.bodyTo) continue;
        ranges.push(Decoration.mark({ class: "sac-record-link", attributes: { "data-record-target": match[1], title: "gd or click to follow" } }).range(from, from + match[1].length));
      }
    }
    return Decoration.set(ranges, true);
  }
  return StateField.define({
    create: links,
    update: (_, transaction) => links(transaction.state),
    provide: (field) => EditorView.decorations.from(field),
  });
}
