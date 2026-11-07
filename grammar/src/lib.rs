//! The task board grammar, version 1: parse a board from YAML, resolve select
//! paths against a parsed board, and list its declared references. The grammar
//! itself is stated in SPEC.md beside this crate.

use std::ops::RangeInclusive;

mod error;
mod parse;
mod select;

pub use error::{ErrorKind, ParseError, SelectError, Where};
pub use parse::parse;
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
    /// The full 40-character lowercase hex commit of the pinned revision.
    pub revision: String,
    pub select: Vec<SelectPath>,
}

/// The declared incorporation references, as (source task, source revision)
/// pairs in document order, nested components included.
pub fn incorporations(board: &Board) -> Vec<(String, String)> {
    let mut found = Vec::new();
    walk(&board.content, &mut |component| {
        if let Component::Incorporation { source, .. } = component {
            found.push((source.task.clone(), source.revision.clone()));
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
