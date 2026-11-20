mod common;

use board_grammar::{Component, SelectError, SelectPath, select};

fn path(text: &str) -> SelectPath {
    SelectPath::parse(text).expect("the select path parses")
}

#[test]
fn a_path_selects_the_whole_component() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    let chosen = select(&board, &path("/content/1")).expect("the path resolves");
    assert_eq!(chosen.kind_name(), "group");
    let Component::Group { title, .. } = chosen else {
        panic!("the select returns the whole group");
    };
    assert_eq!(title.as_deref(), Some("Recorded setup"));
}

#[test]
fn a_path_walks_through_a_group_into_its_content() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    let chosen = select(&board, &path("/content/1/group/content/0")).expect("the path resolves");
    assert_eq!(
        chosen,
        &Component::Text {
            text: "The plinth rebuild replaced the aluminium plate with birch ply,\nand the counterweight moved 12 mm outward. Resonance appears\nonly with the rubber mat removed.\n".into()
        }
    );
}

#[test]
fn a_path_walks_through_an_incorporation_into_its_content() {
    let board = common::parse_fixture("valid/board-before.yaml");
    let chosen =
        select(&board, &path("/content/1/incorporation/content/0")).expect("the path resolves");
    assert_eq!(chosen.kind_name(), "text");
}

#[test]
fn an_index_past_the_root_list_refuses_with_the_list_in_the_error() {
    let board = common::parse_fixture("valid/minimal.yaml");
    let error = select(&board, &path("/content/3")).expect_err("the out-of-bounds index refuses");
    assert_eq!(
        error,
        SelectError::OutOfBounds {
            at: "/content".into(),
            index: 3,
            len: 1
        }
    );
    assert!(
        error
            .to_string()
            .contains("re-point the path at a position the pinned revision has"),
        "{}",
        error
    );
}

#[test]
fn an_index_past_a_nested_list_refuses_at_the_nested_position() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    let error = select(&board, &path("/content/1/group/content/5"))
        .expect_err("the nested out-of-bounds index refuses");
    assert_eq!(
        error,
        SelectError::OutOfBounds {
            at: "/content/1/group/content".into(),
            index: 5,
            len: 2
        }
    );
}

#[test]
fn stepping_through_a_kind_the_component_lacks_refuses() {
    let board = common::parse_fixture("valid/all-kinds.yaml");
    let error = select(&board, &path("/content/0/group/content/0"))
        .expect_err("walking through a text component refuses");
    assert_eq!(
        error,
        SelectError::KindMismatch {
            at: "/content/0".into(),
            walked: "group",
            found: "text"
        }
    );
    let error = select(&board, &path("/content/1/incorporation/content/0"))
        .expect_err("walking through a group as an incorporation refuses");
    assert_eq!(
        error,
        SelectError::KindMismatch {
            at: "/content/1".into(),
            walked: "incorporation",
            found: "group"
        }
    );
}

#[test]
fn path_forms_outside_the_language_refuse_at_parse() {
    for bad in [
        "content/0",
        "",
        "/",
        "/content",
        "/content/x",
        "/content/0/",
        "/content/0/text",
        "/group/0",
        "/content/-1",
        "/content/0/group",
        "/content/0/group/0",
        "/content/0/text/content/0",
    ] {
        let error = SelectPath::parse(bad)
            .err()
            .unwrap_or_else(|| panic!("the path '{bad}' is refused"));
        assert!(
            error.message.contains("is not well formed"),
            "the path '{bad}' refusal names the language: {}",
            error.message
        );
    }
}

#[test]
fn a_parsed_path_displays_back_in_its_canonical_form() {
    for written in [
        "/content/0",
        "/content/12",
        "/content/1/group/content/0",
        "/content/2/incorporation/content/3/group/content/1",
    ] {
        assert_eq!(path(written).to_string(), written);
    }
}
