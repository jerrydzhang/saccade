//! The citation-marker interpretation, shared by every boundary that
//! judges one: the prose's markdown structure decides what a marker
//! is — text runs carry markers, code spans, code blocks, and math
//! are literal, never citations. Numbers parse fallibly: a token too
//! large for an index is an unresolved marker, not a panic.

use pulldown_cmark::{Event as MdEvent, Options, Parser};

/// One marker token found in a text run: its byte range within the
/// run and its number, None when the digits name no indexable number.
pub(crate) struct Token {
    pub start: usize,
    pub end: usize,
    pub number: Option<usize>,
}

/// The markers of one text run, in order. Escaping does not touch the
/// marker vocabulary, so the scan serves the raw and the escaped run
/// alike.
pub(crate) fn tokens_in_run(run: &str) -> Vec<Token> {
    let bytes = run.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' && bytes.get(i + 1) == Some(&b'^') {
            let digits_start = i + 2;
            let mut digits_end = digits_start;
            while digits_end < bytes.len() && bytes[digits_end].is_ascii_digit() {
                digits_end += 1;
            }
            if digits_end > digits_start && bytes.get(digits_end) == Some(&b']') {
                let digits = &run[digits_start..digits_end];
                out.push(Token {
                    start: i,
                    end: digits_end + 1,
                    number: digits.parse().ok(),
                });
                i = digits_end + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// The first marker of the prose that no reference in a mapping of
/// `references` entries resolves: its literal digits, for the refusal
/// to name. None when every marker resolves — and prose whose markers
/// live only in code or math always reads as resolving.
pub fn first_unresolved(raw: &str, references: usize) -> Option<String> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES | Options::ENABLE_MATH);
    let mut run = String::new();
    let mut in_code_block = false;
    let flush = |run: &mut String| -> Option<String> {
        if run.is_empty() {
            return None;
        }
        let found = tokens_in_run(run).into_iter().find_map(|token| {
            let resolved = token.number.is_some_and(|k| k >= 1 && k <= references);
            (!resolved).then(|| run[token.start + 2..token.end - 1].to_string())
        });
        run.clear();
        found
    };
    for event in Parser::new_ext(raw, opts) {
        match event {
            MdEvent::Text(text) if !in_code_block => run.push_str(&text),
            MdEvent::Start(tag) => {
                if let Some(found) = flush(&mut run) {
                    return Some(found);
                }
                if matches!(tag, pulldown_cmark::Tag::CodeBlock(_)) {
                    in_code_block = true;
                }
            }
            MdEvent::End(end) => {
                if let Some(found) = flush(&mut run) {
                    return Some(found);
                }
                if matches!(end, pulldown_cmark::TagEnd::CodeBlock) {
                    in_code_block = false;
                }
            }
            MdEvent::Text(_) => {}
            _ => {
                if let Some(found) = flush(&mut run) {
                    return Some(found);
                }
            }
        }
    }
    flush(&mut run)
}

#[cfg(test)]
mod test {
    use super::*;

    /// Text carries citations; code spans, blocks, and math stay literal.
    #[test]
    fn code_and_math_are_literal_never_citations() {
        assert_eq!(first_unresolved("plain words", 0), None);
        assert_eq!(first_unresolved("resolves [^1]", 1), None);
        assert_eq!(first_unresolved("twice [^1] and [^1]", 1), None);
        assert_eq!(
            first_unresolved("the marker syntax is `[^1]` in code", 0),
            None
        );
        assert_eq!(
            first_unresolved("```\n[^1]\n``` fenced stays literal", 0),
            None
        );
        assert_eq!(first_unresolved("math $x[^1]y$ stays", 0), None);
        assert_eq!(
            first_unresolved("a loose [^1] in text", 0),
            Some("1".into())
        );
        assert_eq!(first_unresolved("[^2] beyond", 1), Some("2".into()));
        assert_eq!(first_unresolved("[^0] is no position", 1), Some("0".into()));
    }

    /// An oversized digit string is an unresolved marker, never a panic.
    #[test]
    fn oversized_numbers_are_unresolved_not_panics() {
        let oversized = "9".repeat(40);
        assert_eq!(
            first_unresolved(&format!("see [^{oversized}]"), 3),
            Some(oversized.clone())
        );
        // the scanner's tokens carry the same discipline
        let tokens = tokens_in_run(&format!("[^{oversized}]"));
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].number, None);
        // the rendered shape keeps its literal bytes
        assert_eq!(tokens_in_run("[^1]")[0].number, Some(1));
    }
}
