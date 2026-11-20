// Code highlighting for fenced code (t-40): highlight.js's core with a
// chosen language set, bundled and synchronous - nothing loads at run
// time, so it stays inside the app's CSP. Presentation only: the result
// is a list of runs whose texts join back to the source byte for byte,
// so copy and find keep reading the authored text. The renderer
// (render.ts) turns runs into spans; the colours are theme tokens
// (styles/tokens.css, --port-code-*), never a palette here.
//
// highlight.js answers HTML; this module reads that HTML back into runs
// itself (its own spans and escaped text, nothing else), so no authored
// byte ever reaches the DOM as markup.

import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import ini from "highlight.js/lib/languages/ini";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import nix from "highlight.js/lib/languages/nix";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import shell from "highlight.js/lib/languages/shell";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

/** One stretch of source and what it is: the innermost highlight.js
 * scope around it, its names without their prefix ("keyword", or
 * "title" and "function_"); empty for plain text. */
export type CodeRun = Readonly<{ text: string; kinds: readonly string[] }>;

// a private instance: the set is exactly this, whatever else registers
const engine = hljs.newInstance();
for (const [name, language] of Object.entries({
  bash, css, diff, ini, javascript, json, markdown, nix, python, rust, shell, sql, typescript, xml, yaml,
})) {
  engine.registerLanguage(name, language);
}

/**
 * The fence's source as coloured runs, or null when its language is
 * absent or not in the set (the block stays plain code: no guessing).
 */
export function highlightCode(code: string, lang: string | null | undefined): CodeRun[] | null {
  if (!lang) return null;
  const language = engine.getLanguage(lang);
  if (language === undefined) return null;
  const html = engine.highlight(code, { language: lang, ignoreIllegals: true }).value;
  const runs = readRuns(html);
  // the presentation-only promise, checked: anything else stays plain
  return runs.map((run) => run.text).join("") === code ? runs : null;
}

const TAG = /<span class="([^"]*)">|<\/span>/g;

function readRuns(html: string): CodeRun[] {
  const runs: CodeRun[] = [];
  const stack: (readonly string[])[] = [[]];
  let from = 0;
  const text = (end: number): void => {
    if (end > from) runs.push({ text: unescape(html.slice(from, end)), kinds: stack[stack.length - 1]! });
  };
  for (const match of html.matchAll(TAG)) {
    text(match.index);
    from = match.index + match[0].length;
    if (match[1] === undefined) {
      if (stack.length > 1) stack.pop();
    } else {
      const kinds = match[1].split(/\s+/).filter((name) => name.length > 0).map((name) => name.replace(/^hljs-/, ""));
      stack.push(kinds);
    }
  }
  text(html.length);
  return runs;
}

// highlight.js escapes exactly these five; &amp; goes last
function unescape(escaped: string): string {
  return escaped
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#x27;", "'")
    .replaceAll("&amp;", "&");
}
