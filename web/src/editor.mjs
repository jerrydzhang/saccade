import { RangeSet } from "@codemirror/state";
import { gutter, GutterMarker } from "@codemirror/view";
import { codeFolding, ensureSyntaxTree, syntaxTree, foldEffect, unfoldEffect, foldedRanges, unfoldAll } from "@codemirror/language";

export function sections(state) {
  const headers = [];
  (ensureSyntaxTree(state, state.doc.length, 50) ?? syntaxTree(state)).iterate({ enter(node) {
    const match = /^(?:ATX|Setext)Heading([1-6])$/.exec(node.name);
    if (match && node.node.parent?.name === "Document") headers.push({ level: Number(match[1]), start: state.doc.lineAt(node.from).from, from: node.to });
  } });
  return headers.map((header, index) => {
    const next = headers.slice(index + 1).find((item) => item.level <= header.level);
    return { ...header, to: next ? next.start - 1 : state.doc.length };
  }).filter((section) => section.to > section.from);
}

function closed(state, section) {
  let found = false;
  foldedRanges(state).between(section.from, section.to, (from, to) => { if (from === section.from && to === section.to) found = true; });
  return found;
}

export function sectionFold(view, action) {
  const inventory = sections(view.state);
  if (action === "R") return unfoldAll(view);
  if (action === "M") {
    view.dispatch({ effects: inventory.filter((section) => !closed(view.state, section)).map((section) => foldEffect.of(section)) });
    return true;
  }
  const at = view.state.selection.main.head;
  if (action === "j" || action === "k") {
    const target = action === "j" ? inventory.find((section) => section.start > at) : inventory.filter((section) => section.start < at).at(-1);
    if (target) view.dispatch({ selection: { anchor: target.start }, scrollIntoView: true });
    return !!target;
  }
  const section = inventory.filter((item) => item.start <= at && at <= item.to).at(-1);
  if (!section) return false;
  const isClosed = closed(view.state, section);
  const open = action === "o" || (action === "a" && isClosed);
  if (open && !isClosed || !open && isClosed) return true;
  view.dispatch({ effects: (open ? unfoldEffect : foldEffect).of(section), selection: { anchor: section.start }, scrollIntoView: true });
  return true;
}

class SectionMarker extends GutterMarker {
  constructor(section, isClosed) { super(); this.section = section; this.isClosed = isClosed; }
  eq(other) { return this.section.from === other.section.from && this.section.to === other.section.to && this.isClosed === other.isClosed; }
  toDOM(view) {
    const button = document.createElement("button");
    button.textContent = this.isClosed ? "+" : "-";
    button.title = `${this.isClosed ? "Open" : "Close"} Markdown section (za)`;
    button.tabIndex = -1;
    button.addEventListener("click", () => {
      view.dispatch({ effects: (this.isClosed ? unfoldEffect : foldEffect).of(this.section) });
      view.focus();
    });
    return button;
  }
}

export const headerFolding = [codeFolding({ placeholderText: "..." }), gutter({
  class: "sac-section-gutter",
  markers: (view) => RangeSet.of(sections(view.state).map((section) => new SectionMarker(section, closed(view.state, section)).range(section.start)), true),
  lineMarkerChange: (update) => update.docChanged || update.transactions.some((transaction) => transaction.effects.length),
})];
