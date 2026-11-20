//! The written form of one component: the sequence item parse reads
//! back as the same component. The board get door prints this.

use std::fmt::Write as _;

use crate::{Component, SelectPath};

/// Render one component as a sequence item (`- kind: ...`), body
/// indented under it, ending with a newline. Feeding rendered items
/// back through parse as a board's content list reproduces the
/// components exactly.
pub fn render(component: &Component) -> String {
    let mut out = String::new();
    write_item(&mut out, component, "");
    out
}

fn write_item(out: &mut String, component: &Component, indent: &str) {
    // the item marker "- " opens the sequence; the component's own
    // mapping keys sit two columns past that key column, where the
    // parser reads them as the component's fields
    let field = format!("{indent}    ");
    match component {
        Component::Text { text } => keyed_scalar(out, indent, "text", text, &field),
        Component::Annotation { text } => keyed_scalar(out, indent, "annotation", text, &field),
        Component::Figure { input, caption } => {
            let _ = writeln!(out, "{indent}- figure:");
            let _ = writeln!(out, "{field}input: {}", inline(input));
            let _ = writeln!(out, "{field}caption: {}", inline(caption));
        }
        Component::Group { title, content } => {
            let _ = writeln!(out, "{indent}- group:");
            if let Some(title) = title {
                let _ = writeln!(out, "{field}title: {}", inline(title));
            }
            let _ = writeln!(out, "{field}content:");
            for child in content {
                write_item(out, child, &format!("{field}  "));
            }
        }
        Component::Incorporation { source, content } => {
            let _ = writeln!(out, "{indent}- incorporation:");
            let _ = writeln!(out, "{field}source:");
            let inner = format!("{field}  ");
            let _ = writeln!(out, "{inner}task: {}", inline(&source.task));
            let _ = writeln!(out, "{inner}revision: {}", source.revision);
            let paths: Vec<String> = source.select.iter().map(SelectPath::to_string).collect();
            let _ = writeln!(out, "{inner}select: [{}]", paths.join(", "));
            let _ = writeln!(out, "{field}content:");
            for child in content {
                write_item(out, child, &format!("{field}  "));
            }
        }
    }
}

fn keyed_scalar(out: &mut String, indent: &str, key: &str, value: &str, field: &str) {
    match scalar(value) {
        Scalar::Plain(text) | Scalar::Quoted(text) => {
            let _ = writeln!(out, "{indent}- {key}: {text}");
        }
        Scalar::Block { chomp, lines } => {
            let _ = writeln!(out, "{indent}- {key}: |{chomp}");
            for line in lines {
                if line.is_empty() {
                    let _ = writeln!(out);
                } else {
                    let _ = writeln!(out, "{field}  {line}");
                }
            }
        }
    }
}

fn inline(value: &str) -> String {
    match scalar(value) {
        Scalar::Plain(text) | Scalar::Quoted(text) => text,
        // single-line callers only; a newline falls back to quoted
        Scalar::Block { lines, .. } => quote(&lines.join("\n")),
    }
}

enum Scalar {
    Plain(String),
    Quoted(String),
    Block {
        chomp: &'static str,
        lines: Vec<String>,
    },
}

fn scalar(value: &str) -> Scalar {
    if plain_safe(value) {
        return Scalar::Plain(value.to_string());
    }
    // block scalars carry the common prose shapes: interior newlines
    // with at most one trailing one, which clip or strip preserve
    if value.contains('\n') && !value.contains("\n\n") {
        let chomp = if value.ends_with('\n') { "" } else { "-" };
        let body = value.strip_suffix('\n').unwrap_or(value);
        return Scalar::Block {
            chomp,
            lines: body.split('\n').map(String::from).collect(),
        };
    }
    Scalar::Quoted(quote(value))
}

fn plain_safe(value: &str) -> bool {
    if value.is_empty() || value.trim() != value {
        return false;
    }
    if value.contains(": ") || value.contains(" #") {
        return false;
    }
    let first = value.as_bytes()[0];
    if b"-?:,[]{}#&*!|>'\"%@`".contains(&first) {
        return false;
    }
    !(value.ends_with(':') || value.ends_with('#'))
}

fn quote(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
