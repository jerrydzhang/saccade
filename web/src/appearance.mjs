import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

export const sourceAppearance = [
  EditorView.theme({}, { dark: true }),
  syntaxHighlighting(HighlightStyle.define([
    { tag: [t.content, t.name, t.literal, t.operator, t.punctuation], color: "var(--primary)" },
    { tag: t.keyword, color: "var(--lavender)" },
    { tag: [t.atom, t.bool], color: "var(--olive)" },
    { tag: [t.number, t.escape, t.regexp], color: "var(--gold)" },
    { tag: [t.string, t.inserted], color: "var(--green)" },
    { tag: [t.comment, t.meta], color: "var(--soft)" },
    { tag: [t.deleted, t.invalid], color: "var(--rose)" },
    { tag: t.link, color: "var(--steel)", textDecoration: "underline" },
    { tag: t.emphasis, fontStyle: "italic" },
    { tag: t.strong, color: "var(--bright)", fontWeight: "bold" },
    { tag: t.strikethrough, textDecoration: "line-through" },
    { tag: t.monospace, color: "var(--primary)" },
    { tag: t.heading, color: "var(--bright)", fontWeight: "600" },
    { tag: t.heading1, color: "var(--lavender)" },
    { tag: t.heading2, color: "var(--rose)" },
    { tag: t.heading3, color: "var(--mauve)" },
    { tag: t.heading4, color: "var(--olive)" },
    { tag: t.heading5, color: "var(--gold)" },
    { tag: t.heading6, color: "var(--bright)" },
  ])),
];
