mod common;

use board_grammar::figures;

#[test]
fn figure_paths_come_back_in_document_order() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    assert_eq!(figures(&board), vec!["figures/plinth-section.png"]);
}

#[test]
fn repeated_references_list_once() {
    let board = board_grammar::parse(
        r#"grammar: 1
task: t-4
content:
  - group:
      content:
        - figure:
            input: figures/plinth-section.png
            caption: The section.
  - figure:
      input: figures/plinth-section.png
      caption: The same section, again.
  - figure:
      input: figures/sweep-overlay.png
      caption: The sweep with the peak marked.
"#,
    )
    .expect("the repeated figure references parse");
    assert_eq!(
        figures(&board),
        vec![
            "figures/plinth-section.png".to_string(),
            "figures/sweep-overlay.png".to_string(),
        ]
    );
}

#[test]
fn a_board_without_figures_lists_none() {
    let board = common::parse_fixture("valid/minimal.yaml");
    assert!(figures(&board).is_empty());
}
