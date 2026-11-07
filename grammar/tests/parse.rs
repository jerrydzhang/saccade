mod common;

use board_grammar::{Component, ErrorKind, SUPPORTED_GRAMMAR_VERSIONS, Source};

#[test]
fn valid_fixtures_parse() {
    for name in [
        "valid/minimal.yaml",
        "valid/all-kinds.yaml",
        "valid/board-before.yaml",
        "valid/board-annotated.yaml",
        "valid/board-after.yaml",
    ] {
        let board = common::parse_fixture(name);
        assert!(!board.task.is_empty(), "{name} carries its task name");
    }
}

// the c-885 exercise boards survive the conversion to grammar 1 with their citations intact
#[test]
fn converted_exercise_boards_keep_their_declared_references() {
    let before = common::parse_fixture("valid/board-before.yaml");
    let annotated = common::parse_fixture("valid/board-annotated.yaml");
    let after = common::parse_fixture("valid/board-after.yaml");
    for board in [&before, &annotated, &after] {
        assert_eq!(board.task, "gd-quadratic-a1-b100");
        for (task, revision) in board_grammar::incorporations(board) {
            assert_eq!(task, "source-model");
            assert_eq!(revision, "647de6e6d2b90c507bf2d05dacc32740558e0bfc");
        }
    }
    assert_eq!(before.content.len(), 5);
    assert_eq!(annotated.content.len(), 6);
    assert_eq!(after.content.len(), 3);
}

#[test]
fn every_invalid_fixture_refuses_with_its_declared_kind() {
    let expected = |name: &str| -> ErrorKind {
        match name {
            "anchor-and-alias" => ErrorKind::AnchorOrAlias,
            "duplicate-key" => ErrorKind::DuplicateKey,
            "unknown-kind" => ErrorKind::UnknownKind,
            "unknown-field-root" => ErrorKind::UnknownField,
            "unknown-field-group" => ErrorKind::UnknownField,
            "unknown-field-incorporation" => ErrorKind::UnknownField,
            "unknown-field-figure" => ErrorKind::UnknownField,
            "unknown-field-source" => ErrorKind::UnknownField,
            "missing-grammar" => ErrorKind::MissingField,
            "missing-task" => ErrorKind::MissingField,
            "missing-content" => ErrorKind::MissingField,
            "grammar-version-2" => ErrorKind::GrammarVersion,
            "grammar-not-integer" => ErrorKind::GrammarVersion,
            "incorporation-no-text" => ErrorKind::NoReceivingText,
            "incorporation-empty-content" => ErrorKind::NoReceivingText,
            "revision-short" => ErrorKind::BadRevision,
            "revision-not-hex" => ErrorKind::BadRevision,
            "figure-absolute" => ErrorKind::BadFigurePath,
            "figure-parent" => ErrorKind::BadFigurePath,
            "select-empty" => ErrorKind::EmptySelect,
            "select-no-index" => ErrorKind::BadSelectPath,
            "select-not-positional" => ErrorKind::BadSelectPath,
            "malformed-yaml" => ErrorKind::YamlSyntax,
            "multiple-documents" => ErrorKind::MultipleDocuments,
            "content-not-list" => ErrorKind::WrongType,
            "component-no-kind" => ErrorKind::WrongType,
            "component-two-kinds" => ErrorKind::WrongType,
            "too-deep" => ErrorKind::TooDeep,
            other => panic!("the fixture {other} declares no expected kind"),
        }
    };
    let dir = format!("{}/fixtures/invalid", env!("CARGO_MANIFEST_DIR"));
    for entry in std::fs::read_dir(&dir).expect("the invalid fixture directory lists") {
        let path = entry.expect("the directory entry reads").path();
        let name = path
            .file_stem()
            .expect("the stem")
            .to_string_lossy()
            .into_owned();
        let error = board_grammar::parse(&common::fixture(&format!("invalid/{name}.yaml")))
            .err()
            .unwrap_or_else(|| panic!("the fixture {name} must be refused"));
        assert_eq!(
            error.kind,
            expected(&name),
            "the fixture {name} refuses with the declared kind; the message reads: {}",
            error.message
        );
    }
}

// the refusals say what to do, not only what failed
#[test]
fn refusal_messages_carry_their_next_step() {
    let no_text = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - incorporation:
      source:
        task: source-model
        revision: 647de6e6d2b90c507bf2d05dacc32740558e0bfc
        select: [/content/0]
      content: []
"#,
    )
    .expect_err("the incorporation without receiving text is refused");
    assert_eq!(
        no_text.message,
        "incorporation at /content/0 has no text component; add a text explaining what you take from the source"
    );
    let unknown = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - table:
      columns: []
"#,
    )
    .expect_err("the unknown kind is refused");
    assert!(
        unknown
            .message
            .contains("a new kind is admitted only by ratification"),
        "{}",
        unknown.message
    );
    let revision = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - incorporation:
      source:
        task: source-model
        revision: 647de6e
        select: [/content/0]
      content:
        - text: |
            The receiving text.
"#,
    )
    .expect_err("the short revision is refused");
    assert!(
        revision
            .message
            .contains("full 40-character lowercase hex commit"),
        "{}",
        revision.message
    );
    let figure = board_grammar::parse(
        r#"grammar: 1
task: tone-arm-resonance
content:
  - figure:
      input: /assets/plinth.png
      caption: The plinth section.
"#,
    )
    .expect_err("the absolute figure path is refused");
    assert!(
        figure
            .message
            .contains("relative paths inside the board tree"),
        "{}",
        figure.message
    );
    let version = board_grammar::parse("grammar: 2\ntask: floop-index-backfill\ncontent: []\n")
        .expect_err("grammar 2 is refused");
    assert!(
        version.message.contains("write the board in grammar 1"),
        "{}",
        version.message
    );
}

#[test]
fn nested_duplicate_keys_are_refused() {
    let error = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - group:
      title: A
      title: B
      content:
        - text: |
            The recorded account.
"#,
    )
    .expect_err("the repeated title key is refused");
    assert_eq!(error.kind, ErrorKind::DuplicateKey);
}

#[test]
fn quoted_and_plain_duplicate_keys_are_refused() {
    let error = board_grammar::parse(
        "grammar: 1\n\"task\": floop-index-backfill\ntask: source-model\ncontent: []\n",
    )
    .expect_err("the quoted and plain task keys are one key twice");
    assert_eq!(error.kind, ErrorKind::DuplicateKey);
}

#[test]
fn anchors_are_refused_even_without_an_alias() {
    let error = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - group: &setup
      content:
        - text: |
            The recorded setup.
"#,
    )
    .expect_err("the anchor alone is refused");
    assert_eq!(error.kind, ErrorKind::AnchorOrAlias);
}

#[test]
fn blank_receiving_text_is_not_substantive() {
    let error = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - incorporation:
      source:
        task: source-model
        revision: 647de6e6d2b90c507bf2d05dacc32740558e0bfc
        select: [/content/0]
      content:
        - text: "   "
"#,
    )
    .expect_err("the blank receiving text is refused");
    assert_eq!(error.kind, ErrorKind::NoReceivingText);
}

#[test]
fn receiving_text_nested_in_a_group_counts() {
    let board = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - incorporation:
      source:
        task: source-model
        revision: 647de6e6d2b90c507bf2d05dacc32740558e0bfc
        select: [/content/0]
      content:
        - group:
            content:
              - text: |
                  The receiving text, nested in a group.
"#,
    )
    .expect("the nested receiving text passes");
    assert_eq!(board.content.len(), 1);
}

#[test]
fn uppercase_revisions_are_refused() {
    let error = board_grammar::parse(
        r#"grammar: 1
task: floop-index-backfill
content:
  - incorporation:
      source:
        task: source-model
        revision: 647DE6E6D2B90C507BF2D05DACC32740558E0BFC
        select: [/content/0]
      content:
        - text: |
            The receiving text.
"#,
    )
    .expect_err("the uppercase revision is refused");
    assert_eq!(error.kind, ErrorKind::BadRevision);
}

#[test]
fn the_supported_range_is_exactly_one() {
    assert_eq!(*SUPPORTED_GRAMMAR_VERSIONS.start(), 1);
    assert_eq!(*SUPPORTED_GRAMMAR_VERSIONS.end(), 1);
    for version in [0, 2, -1] {
        let error = board_grammar::parse(&format!(
            "grammar: {version}\ntask: floop-index-backfill\ncontent: []\n"
        ))
        .expect_err("the version outside the range is refused");
        assert_eq!(error.kind, ErrorKind::GrammarVersion);
    }
}

#[test]
fn empty_content_and_empty_text_still_parse() {
    let board = board_grammar::parse("grammar: 1\ntask: floop-index-backfill\ncontent: []\n")
        .expect("the empty content list passes");
    assert!(board.content.is_empty());
    let board =
        board_grammar::parse("grammar: 1\ntask: floop-index-backfill\ncontent:\n  - text: \"\"\n")
            .expect("the empty text component passes");
    assert_eq!(
        board.content[0],
        Component::Text {
            text: String::new()
        }
    );
}

fn nesting(depth: usize) -> String {
    let spaces = |count: usize| " ".repeat(count);
    let mut text = String::from("grammar: 1\ntask: floop-index-backfill\ncontent:\n");
    let mut pad = 2usize;
    for _ in 0..depth {
        text.push_str(&format!(
            "{}- group:\n{}content:\n",
            spaces(pad),
            spaces(pad + 4)
        ));
        pad += 4;
    }
    text.push_str(&format!(
        "{}- text: |\n{}  The innermost statement.\n",
        spaces(pad + 2),
        spaces(pad + 6)
    ));
    text
}

#[test]
fn nesting_at_the_depth_cap_passes_and_one_deeper_refuses() {
    // the chain counts its components: 99 groups plus the innermost text is 100 deep
    let deep = nesting(board_grammar::MAX_NESTING_DEPTH - 1);
    let board = board_grammar::parse(&deep).expect("the board at the depth cap passes");
    assert_eq!(board.task, "floop-index-backfill");
    let deeper = nesting(board_grammar::MAX_NESTING_DEPTH);
    let error =
        board_grammar::parse(&deeper).expect_err("the board one deeper than the cap is refused");
    assert_eq!(error.kind, ErrorKind::TooDeep);
}

#[test]
fn all_kinds_fixture_shapes_match_the_model() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    assert_eq!(board.content.len(), 5);
    assert_eq!(
        board.content[1],
        Component::Group {
            title: Some("Recorded setup".into()),
            content: vec![
                Component::Text {
                    text: "The plinth rebuild replaced the aluminium plate with birch ply,\nand the counterweight moved 12 mm outward. Resonance appears\nonly with the rubber mat removed.\n".into()
                },
                Component::Figure {
                    input: "figures/plinth-section.png".into(),
                    caption: "Section through the rebuilt plinth with the measurement positions marked.".into(),
                },
            ],
        }
    );
    let expected_source = Source {
        task: "cartridge-compliance-notes".into(),
        revision: "4a1b8ef52c0d47e6a90b3f1d2c5a7e8b9c0d1e2f".into(),
        select: vec![
            board_grammar::SelectPath::parse("/content/1").expect("the path parses"),
            board_grammar::SelectPath::parse("/content/2/group/content/0")
                .expect("the path parses"),
        ],
    };
    let Component::Group {
        content: nested, ..
    } = &board.content[2]
    else {
        panic!("the third component is the group carrying the incorporation");
    };
    let Component::Incorporation { source, .. } = &nested[0] else {
        panic!("the group carries the incorporation");
    };
    assert_eq!(source, &expected_source);
}
