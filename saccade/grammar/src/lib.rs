//! The task board grammar, version 1: parse a board from YAML, resolve select
//! paths against a parsed board, and list its declared references. The grammar
//! itself is stated in SPEC.md beside this crate.

use std::ops::RangeInclusive;

mod error;
mod parse;
mod render;
mod select;

pub use error::{ErrorKind, ParseError, SelectError, Where};
pub use parse::parse;
pub use render::render;
pub use select::{SelectPath, select};

/// The grammar versions this build parses; callers negotiating versions read this.
pub const SUPPORTED_GRAMMAR_VERSIONS: RangeInclusive<u64> = 1..=1;

/// The deepest group or incorporation nesting parse accepts.
pub const MAX_NESTING_DEPTH: usize = 100;

/// A parsed board: the task it accounts for and its components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub task: String,
    pub content: Vec<Component>,
}

/// One component of a board; exactly one of the five ratified kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Component {
    /// Markdown with dollar math.
    Text { text: String },
    /// Nested components with an optional title.
    Group {
        title: Option<String>,
        content: Vec<Component>,
    },
    /// A retained file in the board tree plus a caption.
    Figure { input: String, caption: String },
    /// A citation of a pinned source revision plus the receiving text.
    Incorporation {
        source: Source,
        content: Vec<Component>,
    },
    /// A review concern placed near the content it addresses.
    Annotation { text: String },
}

impl Component {
    /// The kind's name as written in the board file.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Component::Text { .. } => "text",
            Component::Group { .. } => "group",
            Component::Figure { .. } => "figure",
            Component::Incorporation { .. } => "incorporation",
            Component::Annotation { .. } => "annotation",
        }
    }
}

/// The incorporation citation: what is taken and from where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub task: String,
    /// The source revision's dense number, 1-based.
    pub revision: usize,
    pub select: Vec<SelectPath>,
}

/// The declared incorporation references, as (source task, source
/// revision number) pairs in document order, nested components included.
pub fn incorporations(board: &Board) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    walk(&board.content, &mut |component| {
        if let Component::Incorporation { source, .. } = component {
            found.push((source.task.clone(), source.revision));
        }
    });
    found
}

/// The figure paths the board references, unique by first occurrence, in
/// document order; callers check existence and size against the board tree.
pub fn figures(board: &Board) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    walk(&board.content, &mut |component| {
        if let Component::Figure { input, .. } = component
            && !found.contains(input)
        {
            found.push(input.clone());
        }
    });
    found
}

/// The images a board's Markdown embeds - `![alt](src)`, `![alt][ref]`
/// or an HTML `<img>` in a text or annotation - as (component path,
/// source) pairs in document order. Parsing accepts them, so a revision
/// that holds one still reads; the write side refuses them (t-23, the
/// human 2026-11-14): a figure component carries an image, with its
/// caption and enlargement. Code (fenced blocks and inline spans) is
/// never an image.
pub fn markdown_images(board: &Board) -> Vec<(String, String)> {
    let mut found = Vec::new();
    walk_paths(&board.content, "/content", &mut |path, component| {
        if let Component::Text { text } | Component::Annotation { text } = component {
            for src in images_in(text) {
                found.push((path.to_string(), src));
            }
        }
    });
    found
}

fn walk_paths(content: &[Component], prefix: &str, visit: &mut impl FnMut(&str, &Component)) {
    for (index, component) in content.iter().enumerate() {
        let path = format!("{prefix}/{index}");
        visit(&path, component);
        if let Component::Group { content, .. } | Component::Incorporation { content, .. } =
            component
        {
            walk_paths(content, &format!("{path}/content"), visit);
        }
    }
}

/// The image sources one Markdown text embeds, outside code.
fn images_in(markdown: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        let marker = trimmed.chars().next().filter(|c| *c == '`' || *c == '~');
        if indent <= 3
            && let Some(c) = marker
        {
            let run = trimmed.chars().take_while(|x| *x == c).count();
            if run >= 3 {
                match fence {
                    None => {
                        fence = Some((c, run));
                        continue;
                    }
                    Some((open, len)) if open == c && run >= len => {
                        fence = None;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        if fence.is_some() {
            continue;
        }
        let prose = strip_code_spans(line);
        found.extend(images_in_line(&prose));
    }
    found
}

/// The line with each inline code span blanked out.
fn strip_code_spans(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '`' {
            let run = chars[i..].iter().take_while(|c| **c == '`').count();
            // the closing run of the same length
            let mut j = i + run;
            let mut closed = None;
            while j < chars.len() {
                if chars[j] == '`' {
                    let close = chars[j..].iter().take_while(|c| **c == '`').count();
                    if close == run {
                        closed = Some(j + close);
                        break;
                    }
                    j += close;
                } else {
                    j += 1;
                }
            }
            match closed {
                Some(end) => {
                    out.push(' ');
                    i = end;
                }
                None => {
                    out.extend(&chars[i..i + run]);
                    i += run;
                }
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn images_in_line(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("![") {
        let after = &rest[start + 2..];
        let Some(close) = after.find(']') else { break };
        let tail = &after[close + 1..];
        if let Some(inner) = tail.strip_prefix('(') {
            if let Some(end) = inner.find(')') {
                let src = inner[..end]
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_string();
                found.push(src);
                rest = &inner[end + 1..];
                continue;
            }
        } else if let Some(inner) = tail.strip_prefix('[')
            && let Some(end) = inner.find(']')
        {
            found.push(format!("[{}]", &inner[..end]));
            rest = &inner[end + 1..];
            continue;
        }
        rest = after;
    }
    let lower = line.to_ascii_lowercase();
    let mut from = 0;
    while let Some(at) = lower[from..].find("<img") {
        let open = from + at;
        let tag_end = lower[open..]
            .find('>')
            .map(|e| open + e)
            .unwrap_or(lower.len());
        let tag = &line[open..tag_end];
        let src = tag
            .split("src=")
            .nth(1)
            .map(|v| {
                v.trim_start_matches(['"', '\''])
                    .split(['"', '\'', ' ', '>'])
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
            .unwrap_or_else(|| "<img>".to_string());
        found.push(src);
        from = tag_end.max(open + 4);
    }
    found
}

fn walk<'a>(content: &'a [Component], visit: &mut impl FnMut(&'a Component)) {
    for component in content {
        visit(component);
        match component {
            Component::Group { content, .. } | Component::Incorporation { content, .. } => {
                walk(content, visit)
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    fn board(yaml: &str) -> Board {
        parse(yaml).expect("the fixture parses")
    }

    #[test]
    fn markdown_images_are_found_with_their_paths_and_code_never_counts() {
        let found = markdown_images(&board(
            "grammar: 1\ntask: t-1\ncontent:\n  - text: |\n      See ![plot](figures/a.png \"t\") here.\n  - group:\n      content:\n        - text: |\n            ![ref][fig] and <IMG src=\"b.svg\">\n  - annotation: |\n      ![x](c.png)\n",
        ));
        assert_eq!(
            found,
            vec![
                ("/content/0".to_string(), "figures/a.png".to_string()),
                ("/content/1/content/0".to_string(), "[fig]".to_string()),
                ("/content/1/content/0".to_string(), "b.svg".to_string()),
                ("/content/2".to_string(), "c.png".to_string()),
            ]
        );
    }

    #[test]
    fn code_spans_fences_and_plain_brackets_are_not_images() {
        let found = markdown_images(&board(
            "grammar: 1\ntask: t-1\ncontent:\n  - text: |\n      Write `![a](b.png)` or ``![c](d)`` - not an image. ! [e](f) neither.\n\n      ```markdown\n      ![g](h.png)\n      ```\n\n      ~~~~\n      <img src=\"i\">\n      ~~~~\n",
        ));
        assert_eq!(found, Vec::<(String, String)>::new());
    }
}
