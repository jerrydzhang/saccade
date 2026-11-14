//! What changed between two board revisions, by component (t-50): the
//! summary `sac board diff` prints ahead of the line diff, so a reader
//! sees "the r5 group was removed, the Evidence text changed" without
//! reading YAML hunks.
//!
//! Siblings are aligned by a key (a group's title, a text's first
//! heading, a figure's input, an incorporation's source task) with a
//! longest common subsequence; an aligned pair that differs is changed,
//! a group pair is walked into instead. What is left over is removed or
//! added, except that a removed and an added component with the same
//! key and the same content is one that moved.

use board_grammar::{Board, Component};
use serde::Serialize;

/// One reported change. Paths are select paths in the revision each
/// side names; `label` is the component's place by title, for reading.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Change {
    pub change: ChangeKind,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Added,
    Removed,
    Changed,
    Moved,
}

impl ChangeKind {
    fn word(self) -> &'static str {
        match self {
            ChangeKind::Added => "added",
            ChangeKind::Removed => "removed",
            ChangeKind::Changed => "changed",
            ChangeKind::Moved => "moved",
        }
    }
}

/// The changes from `from` to `to`, in the order a reader meets them.
pub fn changes(from: &Board, to: &Board) -> Vec<Change> {
    let mut out = Vec::new();
    walk(
        &from.content,
        &to.content,
        "/content",
        "/content",
        "",
        &mut out,
    );
    pair_moves(&mut out, from, to);
    out
}

/// The summary as `sac board diff` prints it: one line per change.
pub fn render(changes: &[Change], from: usize, to: usize) -> String {
    if changes.is_empty() {
        return format!("components r{from}..r{to}: no change\n");
    }
    let mut text = format!("components r{from}..r{to}\n");
    for change in changes {
        let place = match (&change.from, &change.to) {
            (Some(a), Some(b)) if a == b => a.clone(),
            (Some(a), Some(b)) => format!("{a} -> {b}"),
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (None, None) => String::new(),
        };
        text.push_str(&format!(
            "  {:<8} {}  {}\n",
            change.change.word(),
            change.label,
            place
        ));
    }
    text
}

fn key(component: &Component) -> String {
    match component {
        Component::Group { title, .. } => {
            format!("group:{}", title.as_deref().unwrap_or(""))
        }
        Component::Text { text } => format!("text:{}", heading(text).unwrap_or("")),
        Component::Figure { input, .. } => format!("figure:{input}"),
        Component::Incorporation { source, .. } => format!("incorporation:{}", source.task),
        Component::Annotation { .. } => "annotation".into(),
    }
}

/// A text's first Markdown heading, when it opens with one.
fn heading(text: &str) -> Option<&str> {
    let first = text.lines().find(|line| !line.trim().is_empty())?.trim();
    first.starts_with('#').then_some(first)
}

/// How a component reads in a label: its title, heading or source.
fn name(component: &Component) -> String {
    match component {
        Component::Group { title: Some(t), .. } => format!("group \"{t}\""),
        Component::Group { title: None, .. } => "group".into(),
        Component::Text { text } => match heading(text) {
            Some(h) => format!("text \"{}\"", h.trim_start_matches('#').trim()),
            None => "text".into(),
        },
        Component::Figure { input, .. } => format!("figure {input}"),
        Component::Incorporation { source, .. } => {
            format!("incorporation {}@r{}", source.task, source.revision)
        }
        Component::Annotation { .. } => "annotation".into(),
    }
}

fn children(component: &Component) -> Option<(&'static str, &[Component])> {
    match component {
        Component::Group { content, .. } => Some(("group", content)),
        Component::Incorporation { content, .. } => Some(("incorporation", content)),
        _ => None,
    }
}

fn walk(
    a: &[Component],
    b: &[Component],
    a_at: &str,
    b_at: &str,
    label: &str,
    out: &mut Vec<Change>,
) {
    let keys_a: Vec<String> = a.iter().map(key).collect();
    let keys_b: Vec<String> = b.iter().map(key).collect();
    let pairs = lcs(&keys_a, &keys_b);
    let (mut i, mut j) = (0, 0);
    let place = |at: &str, index: usize| format!("{at}/{index}");
    let named = |component: &Component| {
        if label.is_empty() {
            name(component)
        } else {
            format!("{label} > {}", name(component))
        }
    };
    let emit_until =
        |until_i: usize, until_j: usize, i: &mut usize, j: &mut usize, out: &mut Vec<Change>| {
            while *i < until_i {
                out.push(Change {
                    change: ChangeKind::Removed,
                    label: named(&a[*i]),
                    from: Some(place(a_at, *i)),
                    to: None,
                });
                *i += 1;
            }
            while *j < until_j {
                out.push(Change {
                    change: ChangeKind::Added,
                    label: named(&b[*j]),
                    from: None,
                    to: Some(place(b_at, *j)),
                });
                *j += 1;
            }
        };
    for (pi, pj) in pairs {
        emit_until(pi, pj, &mut i, &mut j, out);
        let (x, y) = (&a[pi], &b[pj]);
        if x != y {
            match (children(x), children(y)) {
                (Some((via, xs)), Some((_, ys))) if same_head(x, y) => walk(
                    xs,
                    ys,
                    &format!("{}/{via}/content", place(a_at, pi)),
                    &format!("{}/{via}/content", place(b_at, pj)),
                    &named(y),
                    out,
                ),
                _ => out.push(Change {
                    change: ChangeKind::Changed,
                    label: named(y),
                    from: Some(place(a_at, pi)),
                    to: Some(place(b_at, pj)),
                }),
            }
        }
        i = pi + 1;
        j = pj + 1;
    }
    emit_until(a.len(), b.len(), &mut i, &mut j, out);
}

/// Two containers whose own fields (title, citation) agree, so only
/// their content can differ and walking into it says what changed.
fn same_head(x: &Component, y: &Component) -> bool {
    match (x, y) {
        (Component::Group { title: s, .. }, Component::Group { title: t, .. }) => s == t,
        (
            Component::Incorporation { source: s, .. },
            Component::Incorporation { source: t, .. },
        ) => s == t,
        _ => false,
    }
}

/// Index pairs of a longest common subsequence of two key lists.
fn lcs(a: &[String], b: &[String]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut table = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[i][j] = if a[i] == b[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    let (mut i, mut j, mut pairs) = (0, 0, Vec::new());
    while i < n && j < m {
        if a[i] == b[j] {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    pairs
}

/// A removed and an added component that are identical is one move.
fn pair_moves(out: &mut Vec<Change>, from: &Board, to: &Board) {
    let at = |board: &Board, path: &str| {
        board_grammar::SelectPath::parse(path)
            .ok()
            .and_then(|p| board_grammar::select(board, &p).ok().cloned())
    };
    let mut index = 0;
    while index < out.len() {
        if out[index].change == ChangeKind::Removed {
            let removed = out[index].from.clone().and_then(|p| at(from, &p));
            let partner = out.iter().position(|other| {
                other.change == ChangeKind::Added
                    && removed.is_some()
                    && other.to.clone().and_then(|p| at(to, &p)) == removed
            });
            if let Some(partner) = partner {
                let added = out.remove(partner);
                let index = if partner < index { index - 1 } else { index };
                out[index].change = ChangeKind::Moved;
                out[index].to = added.to;
                out[index].label = added.label;
            }
        }
        index += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn board(yaml: &str) -> Board {
        board_grammar::parse(yaml).expect("the fixture parses")
    }

    const BASE: &str = "grammar: 1\ntask: t-1\ncontent:\n  - text: |\n      # Title\n\n      intro\n  - group:\n      title: Old idea (r5)\n      content:\n        - text: |\n            long argument\n  - group:\n      title: Evidence\n      content:\n        - text: |\n            ## Table\n\n            one\n        - figure:\n            input: figures/a.png\n            caption: A\n";

    #[test]
    fn no_change_reports_nothing() {
        let a = board(BASE);
        assert!(changes(&a, &a).is_empty());
        assert_eq!(render(&[], 1, 2), "components r1..r2: no change\n");
    }

    #[test]
    fn a_removed_group_and_a_changed_nested_text_are_named_by_title() {
        let a = board(BASE);
        let b = board(
            &BASE
                .replace("  - group:\n      title: Old idea (r5)\n      content:\n        - text: |\n            long argument\n", "")
                .replace("one\n", "two\n"),
        );
        let found = changes(&a, &b);
        assert_eq!(
            found,
            vec![
                Change {
                    change: ChangeKind::Removed,
                    label: "group \"Old idea (r5)\"".into(),
                    from: Some("/content/1".into()),
                    to: None,
                },
                Change {
                    change: ChangeKind::Changed,
                    label: "group \"Evidence\" > text \"Table\"".into(),
                    from: Some("/content/2/group/content/0".into()),
                    to: Some("/content/1/group/content/0".into()),
                },
            ]
        );
        let text = render(&found, 1, 2);
        assert!(
            text.contains("removed  group \"Old idea (r5)\"  /content/1"),
            "{text}"
        );
        assert!(
            text.contains("/content/2/group/content/0 -> /content/1/group/content/0"),
            "{text}"
        );
    }

    #[test]
    fn an_added_group_is_added_and_a_retitled_group_is_removed_and_added() {
        let a = board(BASE);
        let b = board(&format!(
            "{}  - group:\n      title: Set aside\n      content:\n        - text: |\n            - Old idea (r5)\n",
            BASE.replace("title: Evidence", "title: What it rests on")
        ));
        let kinds: Vec<(ChangeKind, String)> = changes(&a, &b)
            .into_iter()
            .map(|c| (c.change, c.label))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (ChangeKind::Removed, "group \"Evidence\"".into()),
                (ChangeKind::Added, "group \"What it rests on\"".into()),
                (ChangeKind::Added, "group \"Set aside\"".into()),
            ]
        );
    }

    #[test]
    fn an_identical_component_in_a_new_place_is_one_move() {
        let a = board(BASE);
        // the figure moves out of Evidence to the board's end
        let b = board(&format!(
            "{}  - figure:\n      input: figures/a.png\n      caption: A\n",
            BASE.replace(
                "        - figure:\n            input: figures/a.png\n            caption: A\n",
                ""
            )
        ));
        let found = changes(&a, &b);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].change, ChangeKind::Moved);
        assert_eq!(found[0].from.as_deref(), Some("/content/2/group/content/1"));
        assert_eq!(found[0].to.as_deref(), Some("/content/3"));
    }

    #[test]
    fn a_changed_text_without_a_heading_is_changed_in_place() {
        let a = board("grammar: 1\ntask: t-1\ncontent:\n  - text: |\n      before\n");
        let b = board("grammar: 1\ntask: t-1\ncontent:\n  - text: |\n      after\n");
        let found = changes(&a, &b);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].change, ChangeKind::Changed);
        assert_eq!(found[0].label, "text");
    }

    #[test]
    fn the_json_form_names_the_change_in_snake_case() {
        let change = Change {
            change: ChangeKind::Moved,
            label: "figure f.png".into(),
            from: Some("/content/0".into()),
            to: Some("/content/1".into()),
        };
        assert_eq!(
            serde_json::to_value(&change).unwrap(),
            serde_json::json!({"change": "moved", "label": "figure f.png", "from": "/content/0", "to": "/content/1"})
        );
    }
}
