pub fn fixture(rel: &str) -> String {
    std::fs::read_to_string(format!("{}/fixtures/{rel}", env!("CARGO_MANIFEST_DIR")))
        .expect("the fixture file reads")
}

pub fn parse_fixture(rel: &str) -> board_grammar::Board {
    board_grammar::parse(&fixture(rel)).expect("the fixture parses")
}
