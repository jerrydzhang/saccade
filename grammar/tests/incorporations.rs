mod common;

use board_grammar::incorporations;

#[test]
fn the_pairs_come_back_in_document_order_with_nesting() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    assert_eq!(
        incorporations(&board),
        vec![(
            "t-3".to_string(),
            "4a1b8ef52c0d47e6a90b3f1d2c5a7e8b9c0d1e2f".to_string()
        )]
    );
    let board = common::parse_fixture("valid/board-before.yaml");
    assert_eq!(
        incorporations(&board),
        vec![
            (
                "t-1".to_string(),
                "647de6e6d2b90c507bf2d05dacc32740558e0bfc".to_string()
            ),
            (
                "t-1".to_string(),
                "647de6e6d2b90c507bf2d05dacc32740558e0bfc".to_string()
            ),
        ]
    );
}

#[test]
fn a_board_without_incorporations_lists_none() {
    let board = common::parse_fixture("valid/minimal.yaml");
    assert!(incorporations(&board).is_empty());
}

#[test]
fn nested_incorporations_are_listed_after_their_containers() {
    let board = board_grammar::parse(
        r#"grammar: 1
task: t-0
content:
  - incorporation:
      source:
        task: outer-source
        revision: 647de6e6d2b90c507bf2d05dacc32740558e0bfc
        select: [/content/0]
      content:
        - incorporation:
            source:
              task: inner-source
              revision: 4a1b8ef52c0d47e6a90b3f1d2c5a7e8b9c0d1e2f
              select: [/content/1]
            content:
              - text: |
                  The receiving text of the nested incorporation.
        - text: |
            The receiving text of the outer incorporation.
"#,
    )
    .expect("the nested incorporations parse");
    assert_eq!(
        incorporations(&board),
        vec![
            (
                "outer-source".to_string(),
                "647de6e6d2b90c507bf2d05dacc32740558e0bfc".to_string()
            ),
            (
                "inner-source".to_string(),
                "4a1b8ef52c0d47e6a90b3f1d2c5a7e8b9c0d1e2f".to_string()
            ),
        ]
    );
}
