mod common;

// the rendered item parses back into the same component: the get door
// prints exactly what the grammar reads
#[test]
fn rendered_components_round_trip_through_parse() {
    for name in [
        "minimal.yaml",
        "all-kinds.yaml",
        "board-before.yaml",
        "board-after.yaml",
        "board-annotated.yaml",
    ] {
        let board = common::parse_fixture(&format!("valid/{name}"));
        let mut items = String::new();
        for component in &board.content {
            items.push_str(&board_grammar::render(component));
        }
        let doc = format!(
            "grammar: 1\ntask: {}\ncontent:\n{}",
            board.task,
            items
                .lines()
                .map(|line| format!("  {line}\n"))
                .collect::<String>()
        );
        let reparsed = board_grammar::parse(&doc)
            .unwrap_or_else(|error| panic!("{name} does not round trip: {}", error.message));
        assert_eq!(reparsed.content, board.content, "{name} changed");
    }
}

#[test]
fn awkward_scalars_survive_the_round_trip() {
    let doc = r#"grammar: 1
task: t-1
content:
  - text: |
      a colon: inside
      and a "quote"
  - text: "trailing colon:"
  - annotation: " spaced "
  - text: single line
"#;
    let board = board_grammar::parse(doc).unwrap();
    let mut items = String::new();
    for component in &board.content {
        items.push_str(&board_grammar::render(component));
    }
    let reparsed = board_grammar::parse(&format!(
        "grammar: 1\ntask: {}\ncontent:\n{}",
        board.task,
        items
            .lines()
            .map(|line| format!("  {line}\n"))
            .collect::<String>()
    ))
    .expect("the rendered board parses");
    assert_eq!(reparsed.content, board.content);
}
