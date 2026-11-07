//! The admission gate: the native lanes validate boards through the
//! shared board-grammar crate (a path dependency), so the frontend never
//! grows a second validator. The frontend's presentation decoder mirrors
//! this crate's contract; the parity is pinned over the viewer's fixture
//! corpus by tests on both sides. The one divergence is named here: the
//! crate refuses an unknown component kind, while the viewer's verdict
//! downgrades exactly that family to a soft case the frontend renders as
//! raw source.

use board_grammar::{ErrorKind, Where};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "verdict", rename_all = "kebab-case")]
pub enum BoardVerdict {
    /// The board parses.
    Ok,
    /// The board's only found problem is a component kind this build does
    /// not know; the viewer renders those components as raw source.
    UnknownKind { message: String },
    /// The crate's refusal, final in the native lanes; the frontend shows
    /// it verbatim with its family name and, where YAML reported one, its
    /// position.
    Refused {
        kind: String,
        message: String,
        line: Option<usize>,
        column: Option<usize>,
    },
}

fn kind_name(kind: &ErrorKind) -> &'static str {
    match kind {
        ErrorKind::YamlSyntax => "yaml-syntax",
        ErrorKind::MultipleDocuments => "multiple-documents",
        ErrorKind::AnchorOrAlias => "anchor-alias",
        ErrorKind::DuplicateKey => "duplicate-key",
        ErrorKind::MissingField => "missing-field",
        ErrorKind::UnknownField => "unknown-field",
        ErrorKind::UnknownKind => "unknown-kind",
        ErrorKind::GrammarVersion => "grammar-version",
        ErrorKind::WrongType => "wrong-type",
        ErrorKind::BadRevision => "bad-revision",
        ErrorKind::BadFigurePath => "bad-figure-path",
        ErrorKind::BadSelectPath => "bad-select-path",
        ErrorKind::BadTask => "bad-task",
        ErrorKind::EmptySelect => "empty-select",
        ErrorKind::NoReceivingText => "no-receiving-text",
        ErrorKind::TooDeep => "too-deep",
    }
}

fn verdict_from(error: board_grammar::ParseError) -> BoardVerdict {
    let board_grammar::ParseError { kind, at, message } = error;
    if kind == ErrorKind::UnknownKind {
        return BoardVerdict::UnknownKind { message };
    }
    let (line, column) = match at {
        Where::Yaml { line, column } => (Some(line), Some(column)),
        _ => (None, None),
    };
    BoardVerdict::Refused {
        kind: kind_name(&kind).into(),
        message,
        line,
        column,
    }
}

/// The gate every native board read passes before the frontend renders
/// anything. Infallible: a parse failure is a verdict, not an error.
#[tauri::command]
pub fn validate_board(text: String) -> BoardVerdict {
    let verdict = match board_grammar::parse(&text) {
        Ok(_) => BoardVerdict::Ok,
        Err(error) => verdict_from(error),
    };
    let outcome = match &verdict {
        BoardVerdict::Ok => "ok".into(),
        BoardVerdict::UnknownKind { .. } => "unknown-kind".into(),
        BoardVerdict::Refused { kind, .. } => format!("refused ({kind})"),
    };
    eprintln!(
        "[bridge] validate_board {} chars: {}",
        text.chars().count(),
        outcome
    );
    verdict
}

#[cfg(test)]
mod tests {
    use super::{verdict_from, BoardVerdict};
    use board_grammar::{ErrorKind, ParseError, Where, SUPPORTED_GRAMMAR_VERSIONS};

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../fixtures")
                .join(name),
        )
        .unwrap_or_else(|error| panic!("fixture {name} must exist beside the crate: {error}"))
    }

    fn validate(text: &str) -> BoardVerdict {
        match board_grammar::parse(text) {
            Ok(_) => BoardVerdict::Ok,
            Err(error) => verdict_from(error),
        }
    }

    #[test]
    fn the_supported_range_is_exactly_version_one() {
        // the range is pinned by its own cells, not by reading it back
        assert!(SUPPORTED_GRAMMAR_VERSIONS.contains(&1));
        assert!(!SUPPORTED_GRAMMAR_VERSIONS.contains(&0));
        assert!(!SUPPORTED_GRAMMAR_VERSIONS.contains(&2));
        assert!(!SUPPORTED_GRAMMAR_VERSIONS.contains(&3));
        assert_eq!(
            SUPPORTED_GRAMMAR_VERSIONS.start(),
            SUPPORTED_GRAMMAR_VERSIONS.end()
        );
    }

    #[test]
    fn every_valid_fixture_parses() {
        for name in [
            "real-board.yaml",
            "missing-asset.yaml",
            "wrapped-math.yaml",
            "display-scroll.yaml",
            "repeated-math.yaml",
            "long-code.yaml",
            "large-diagram.yaml",
            "saccade-and-migration.yaml",
        ] {
            assert!(
                matches!(validate(&fixture(name)), BoardVerdict::Ok),
                "{name} must parse through the shared crate"
            );
        }
    }

    #[test]
    fn the_unknown_kind_fixture_is_the_soft_case() {
        // the board is otherwise clean, so the crate's first refusal is
        // the unknown kind itself: the viewer renders it as source
        let verdict = validate(&fixture("unknown-kind.yaml"));
        let BoardVerdict::UnknownKind { message } = verdict else {
            panic!("unknown-kind fixture must classify as the soft case, got {verdict:?}");
        };
        assert!(
            message.contains("changelog"),
            "message names the kind: {message}"
        );
    }

    #[test]
    fn an_unknown_major_version_refuses() {
        let verdict = validate(&fixture("grammar-2.yaml"));
        let BoardVerdict::Refused { kind, message, .. } = verdict else {
            panic!("grammar-2 fixture must refuse, got {verdict:?}");
        };
        assert_eq!(kind, "grammar-version");
        assert!(
            message.contains('2'),
            "message names the refused version: {message}"
        );
        assert!(
            message.contains("1"),
            "message names the supported version: {message}"
        );
    }

    #[test]
    fn a_board_without_a_grammar_field_refuses() {
        // the refusal choice: no legacy interpretation, the crate's
        // message teaches the fix
        let verdict = validate(&fixture("missing-grammar.yaml"));
        let BoardVerdict::Refused { kind, message, .. } = verdict else {
            panic!("missing-grammar fixture must refuse, got {verdict:?}");
        };
        assert_eq!(kind, "missing-field");
        assert!(
            message.contains("grammar: 1"),
            "message teaches the fix: {message}"
        );
    }

    #[test]
    fn the_broken_fixture_refuses_with_its_family() {
        let verdict = validate(&fixture("invalid.yaml"));
        let BoardVerdict::Refused { kind, line, .. } = verdict else {
            panic!("invalid fixture must refuse, got {verdict:?}");
        };
        assert_eq!(kind, "yaml-syntax");
        assert!(line.is_some(), "a YAML-position refusal carries its line");
    }

    #[test]
    fn the_too_deep_fixture_refuses_with_its_family() {
        let verdict = validate(&fixture("too-deep.yaml"));
        let BoardVerdict::Refused { kind, .. } = verdict else {
            panic!("too-deep fixture must refuse, got {verdict:?}");
        };
        assert_eq!(kind, "too-deep");
    }

    #[test]
    fn only_an_unknown_kind_downgrades_to_the_soft_verdict() {
        let soft = verdict_from(ParseError {
            kind: ErrorKind::UnknownKind,
            at: Where::Path("/content/1".into()),
            message: "unknown component kind".into(),
        });
        assert!(matches!(soft, BoardVerdict::UnknownKind { .. }));
        // every other family refuses, under the wire name the frontend
        // reads; the list is the kind-name inventory
        let families = [
            ("yaml-syntax", ErrorKind::YamlSyntax),
            ("multiple-documents", ErrorKind::MultipleDocuments),
            ("anchor-alias", ErrorKind::AnchorOrAlias),
            ("duplicate-key", ErrorKind::DuplicateKey),
            ("missing-field", ErrorKind::MissingField),
            ("unknown-field", ErrorKind::UnknownField),
            ("grammar-version", ErrorKind::GrammarVersion),
            ("wrong-type", ErrorKind::WrongType),
            ("bad-revision", ErrorKind::BadRevision),
            ("bad-figure-path", ErrorKind::BadFigurePath),
            ("bad-select-path", ErrorKind::BadSelectPath),
            ("bad-task", ErrorKind::BadTask),
            ("empty-select", ErrorKind::EmptySelect),
            ("no-receiving-text", ErrorKind::NoReceivingText),
            ("too-deep", ErrorKind::TooDeep),
        ];
        for (name, kind) in families {
            let verdict = verdict_from(ParseError {
                kind,
                at: Where::Board,
                message: "refusal".into(),
            });
            let BoardVerdict::Refused { kind: reported, .. } = verdict else {
                panic!("{name} must refuse, never soften");
            };
            assert_eq!(reported, name);
        }
    }

    #[test]
    fn a_yaml_position_refusal_carries_its_position() {
        let verdict = verdict_from(ParseError {
            kind: ErrorKind::GrammarVersion,
            at: Where::Yaml { line: 2, column: 1 },
            message: "grammar 9 is outside".into(),
        });
        let BoardVerdict::Refused { line, column, .. } = verdict else {
            panic!("grammar-version must refuse, got {verdict:?}");
        };
        assert_eq!(line, Some(2));
        assert_eq!(column, Some(1));
    }
}
