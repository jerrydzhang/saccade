use std::fmt;

use crate::error::{ErrorKind, ParseError, SelectError, Where};
use crate::{Board, Component};

/// Which nested-content carrier a path descends through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Via {
    Group,
    Incorporation,
}

impl Via {
    fn as_str(self) -> &'static str {
        match self {
            Via::Group => "group",
            Via::Incorporation => "incorporation",
        }
    }
}

/// A positional structural path into a board, naming whole components:
/// /content/1 or /content/1/group/content/0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectPath {
    first: usize,
    rest: Vec<(Via, usize)>,
}

impl SelectPath {
    /// Parse the written form; every refusal says what a well formed path looks like.
    pub fn parse(input: &str) -> Result<SelectPath, ParseError> {
        let bad = |hint: String| ParseError {
            kind: ErrorKind::BadSelectPath,
            at: Where::Board,
            message: format!("the select path '{input}' is not well formed; {hint}"),
        };
        let Some(rest) = input.strip_prefix('/') else {
            return Err(bad(
                "address the source board from its root, like /content/0".into(),
            ));
        };
        let segments: Vec<&str> = rest.split('/').collect();
        if segments.iter().any(|segment| segment.is_empty()) {
            return Err(bad(
                "write every step: 'content', an index, then 'group' or 'incorporation' and 'content', like /content/1/group/content/0".into(),
            ));
        }
        if segments[0] != "content" {
            return Err(bad(
                "start at the source board's content list, like /content/0".into(),
            ));
        }
        let first = parse_index(segments.get(1).copied()).map_err(|hint| {
            bad(index_error(
                "after 'content' comes a component index, like /content/0",
                &hint,
            ))
        })?;
        let mut tail = &segments[2..];
        let mut steps = Vec::new();
        while !tail.is_empty() {
            let via = match tail[0] {
                "group" => Via::Group,
                "incorporation" => Via::Incorporation,
                _ => {
                    return Err(bad(
                        "enter a component through its kind, 'group' or 'incorporation', like /content/1/group/content/0".into(),
                    ))
                }
            };
            if tail.get(1).copied() != Some("content") {
                return Err(bad(format!(
                    "after '{}' comes its 'content' list, like /content/1/group/content/0",
                    via.as_str()
                )));
            }
            let index = parse_index(tail.get(2).copied()).map_err(|hint| {
                bad(index_error(
                    &format!(
                        "after '{}/content' comes an index, like /content/1/group/content/0",
                        via.as_str()
                    ),
                    &hint,
                ))
            })?;
            steps.push((via, index));
            tail = &tail[3..];
        }
        Ok(SelectPath { first, rest: steps })
    }
}

fn index_error(prefix: &str, hint: &str) -> String {
    if hint.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}; {hint}")
    }
}

fn parse_index(segment: Option<&str>) -> Result<usize, String> {
    let Some(text) = segment else {
        return Err(String::new());
    };
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("indices are plain digits".into());
    }
    text.parse::<usize>()
        .map_err(|_| "the index is too large to be a component position".into())
}

impl fmt::Display for SelectPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "/content/{}", self.first)?;
        for (via, index) in &self.rest {
            write!(f, "/{}/content/{}", via.as_str(), index)?;
        }
        Ok(())
    }
}

/// Resolve a select path against a parsed board, returning the whole component it names.
pub fn select<'a>(board: &'a Board, path: &SelectPath) -> Result<&'a Component, SelectError> {
    let mut at = String::new();
    let mut chosen = pick(&board.content, path.first, &mut at)?;
    for (via, index) in &path.rest {
        let content = match (via, chosen) {
            (Via::Group, Component::Group { content, .. }) => content,
            (Via::Incorporation, Component::Incorporation { content, .. }) => content,
            (_, other) => {
                return Err(SelectError::KindMismatch {
                    at: at.clone(),
                    walked: via.as_str(),
                    found: other.kind_name(),
                });
            }
        };
        at.push_str(&format!("/{}", via.as_str()));
        chosen = pick(content, *index, &mut at)?;
    }
    Ok(chosen)
}

fn pick<'a>(
    list: &'a [Component],
    index: usize,
    at: &mut String,
) -> Result<&'a Component, SelectError> {
    at.push_str("/content");
    let Some(component) = list.get(index) else {
        return Err(SelectError::OutOfBounds {
            at: at.clone(),
            index,
            len: list.len(),
        });
    };
    at.push_str(&format!("/{index}"));
    Ok(component)
}
