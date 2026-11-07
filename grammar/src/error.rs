use std::fmt;

/// Where in the board a parse error sits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Where {
    /// The board mapping itself, outside any component.
    Board,
    /// A component position, written the way select paths are: /content/1/group/content/0.
    Path(String),
    /// A raw YAML position, for errors before the board has any shape.
    Yaml { line: usize, column: usize },
}

/// The family of a parse refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The YAML text itself does not parse, or holds no document.
    YamlSyntax,
    /// The file holds more than one YAML document.
    MultipleDocuments,
    /// A YAML anchor or alias appeared.
    AnchorOrAlias,
    /// One mapping carries the same key twice.
    DuplicateKey,
    /// A required field is absent.
    MissingField,
    /// A field exists that no shape declares.
    UnknownField,
    /// A component names a kind outside the five kinds.
    UnknownKind,
    /// The grammar version is missing its integer or sits outside the supported range.
    GrammarVersion,
    /// A value has the wrong YAML type for its place.
    WrongType,
    /// A source revision is not the full 40-hex commit hash.
    BadRevision,
    /// A figure input is absolute or escapes the board tree.
    BadFigurePath,
    /// A select path does not match the positional path language.
    BadSelectPath,
    /// An incorporation selects no component.
    EmptySelect,
    /// An incorporation carries no non-blank receiving text.
    NoReceivingText,
    /// Components nest deeper than the supported depth.
    TooDeep,
}

/// A refusal with a programmatic kind, a location, and a message that says what to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ErrorKind,
    pub at: Where,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseError {}

/// Why a select path does not resolve against a parsed board.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError {
    /// The path asks for an index past the end of the list it reaches.
    OutOfBounds {
        at: String,
        index: usize,
        len: usize,
    },
    /// The path steps through a kind the component at that position does not have.
    KindMismatch {
        at: String,
        walked: &'static str,
        found: &'static str,
    },
}

impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectError::OutOfBounds { at, index, len } => write!(
                f,
                "the list at {at} holds {len} components; the path asks for index {index}; re-point the path at a position the pinned revision has"
            ),
            SelectError::KindMismatch { at, walked, found } => write!(
                f,
                "the path steps into '{walked}' at {at}, but that position holds a '{found}' component; continue the path through the kinds the pinned revision has"
            ),
        }
    }
}

impl std::error::Error for SelectError {}
