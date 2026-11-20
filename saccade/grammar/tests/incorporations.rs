mod common;

use board_grammar::incorporations;

#[test]
fn the_pairs_come_back_in_document_order_with_nesting() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    assert_eq!(incorporations(&board), vec![("t-3".to_string(), 1)]);
    let board = common::parse_fixture("valid/board-before.yaml");
    assert_eq!(
        incorporations(&board),
        vec![("t-1".to_string(), 1), ("t-1".to_string(), 1)]
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
        revision: 2
        select: [/content/0]
      content:
        - incorporation:
            source:
              task: inner-source
              revision: 1
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
            ("outer-source".to_string(), 2),
            ("inner-source".to_string(), 1)
        ]
    );
}
