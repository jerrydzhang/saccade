// ---- tags: flat names on tasks, from every face ----

use serde_json::{Value, json};

use crate::common::{
    envelope, get_json, json_of, post_command, run_with_terminals, scratch_db, scratch_repo,
    scrubbed, spawn_serve,
};

/// One sac against a local db as an agent (possession of an actor).
fn agent_sac(db: &std::path::Path, args: &[&str]) -> (bool, String, String) {
    let out = scrubbed(std::process::Command::new(env!("CARGO_BIN_EXE_sac")))
        .arg("--db")
        .arg(db)
        .env("SACCADE_ACTOR", "pi")
        .args(args)
        .output()
        .expect("spawn sac");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// One sac against a local db as the human at a terminal.
fn human_sac(db: &std::path::Path, args: &[&str]) -> (bool, String, String) {
    let mut full = vec!["--db".to_string(), db.display().to_string()];
    full.extend(args.iter().map(|a| a.to_string()));
    let out = run_with_terminals(env!("CARGO_BIN_EXE_sac"), &full, true, false);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn ok(result: (bool, String, String)) -> String {
    assert!(result.0, "stderr: {}", result.2);
    result.1
}

fn refusal(result: (bool, String, String)) -> String {
    assert!(!result.0, "expected a refusal, stdout: {}", result.1);
    result.2
}

#[test]
fn tags_ride_the_cli_with_the_human_creating_and_every_face_choosing_among_them() {
    let db = scratch_db("tags-cli");

    // a never-created tag cannot be attached: no tags exist yet
    ok(agent_sac(&db, &["create", "task", "one"]));
    let err = refusal(agent_sac(&db, &["tag", "add", "t-0", "auth"]));
    assert!(err.contains("no tags exist"), "{err}");
    let err = refusal(agent_sac(
        &db,
        &["create", "task", "--tag", "auth", "born tagged"],
    ));
    assert!(err.contains("no tags exist"), "{err}");

    // only the human creates a tag, and says who can
    let err = refusal(agent_sac(
        &db,
        &["tag", "create", "auth", "how people sign in"],
    ));
    assert!(err.contains("only the human creates a tag"), "{err}");
    // the line is required
    refusal(human_sac(&db, &["tag", "create", "auth"]));
    // a multi-line or blank line is refused
    refusal(human_sac(&db, &["tag", "create", "auth", "two\nlines"]));
    ok(human_sac(
        &db,
        &["tag", "create", "auth", "how people sign in"],
    ));
    ok(human_sac(
        &db,
        &["tag", "create", "ui", "what people see and click"],
    ));
    // a name held ignoring case is refused
    let err = refusal(human_sac(&db, &["tag", "create", "AUTH", "again"]));
    assert!(err.contains("exists already"), "{err}");

    // an unknown name from every face names the tags with their lines
    for args in [
        vec!["tag", "add", "t-0", "docs"],
        vec!["create", "task", "--tag", "docs", "x"],
        vec!["list", "--tag", "docs"],
    ] {
        let err = refusal(agent_sac(&db, &args));
        assert!(err.contains("no tag 'docs'; the tags are:"), "{err}");
        assert!(err.contains("auth - how people sign in"), "{err}");
        assert!(err.contains("ui - what people see and click"), "{err}");
    }

    // tag list: birth order, name then line
    assert_eq!(
        ok(agent_sac(&db, &["tag", "list"])).trim_end(),
        "auth - how people sign in\nui - what people see and click"
    );
    let listed: Value = json_of(&ok(agent_sac(&db, &["--json", "tag", "list"])));
    assert_eq!(listed["tags"][0]["name"], "auth");
    assert_eq!(listed["tags"][0]["description"], "how people sign in");
    assert!(
        listed["tags"][0]["by"]
            .as_str()
            .is_some_and(|by| by != "pi")
    );
    assert!(listed["tags"][0]["created_at"].as_u64().is_some());

    // born tagged, add and remove several at once, all or none
    ok(agent_sac(
        &db,
        &["create", "task", "--tag", "ui", "--tag", "auth", "two"],
    ));
    ok(agent_sac(&db, &["tag", "add", "t-0", "ui", "auth"]));
    let err = refusal(agent_sac(&db, &["tag", "add", "t-0", "auth"]));
    assert!(err.contains("already_tagged"), "{err}");
    ok(agent_sac(&db, &["create", "task", "three"]));
    let err = refusal(agent_sac(&db, &["tag", "add", "t-2", "auth", "nope"]));
    assert!(err.contains("no tag 'nope'"), "{err}");
    let shown = ok(agent_sac(&db, &["show", "t-2"]));
    assert!(shown.contains("tags  -"), "all or nothing: {shown}");
    ok(agent_sac(&db, &["tag", "add", "t-2", "ui"]));
    let err = refusal(agent_sac(&db, &["tag", "remove", "t-2", "auth"]));
    assert!(err.contains("not_tagged"), "{err}");
    let err = refusal(agent_sac(&db, &["tag", "remove", "t-2", "ui", "auth"]));
    assert!(err.contains("not_tagged"), "{err}");
    assert!(ok(agent_sac(&db, &["show", "t-2"])).contains("  ui - what people see"));
    ok(agent_sac(&db, &["tag", "remove", "t-2", "ui"]));

    // listings: the last column is the present set in tag birth order
    let rows = ok(agent_sac(&db, &["list"]));
    let row = |id: &str| {
        rows.lines()
            .find(|l| l.starts_with(id))
            .unwrap_or_else(|| panic!("{id} in {rows}"))
            .to_string()
    };
    assert!(row("t-0").ends_with("\tauth,ui"), "{rows}");
    assert!(row("t-1").ends_with("\tauth,ui"), "{rows}");
    assert!(row("t-2").ends_with("\t-"), "{rows}");

    // list --tag: the exact set; a task with two tags answers under both
    let ids = |args: &[&str]| -> Vec<String> {
        ok(agent_sac(&db, args))
            .lines()
            .filter(|l| l.starts_with("t-"))
            .map(|l| l.split('\t').next().unwrap().to_string())
            .collect()
    };
    assert_eq!(ids(&["list", "--tag", "auth"]), ["t-0", "t-1"]);
    assert_eq!(ids(&["list", "--tag", "ui"]), ["t-0", "t-1"]);
    ok(agent_sac(&db, &["tag", "remove", "t-1", "ui"]));
    assert_eq!(ids(&["list", "--tag", "ui"]), ["t-0"]);
    assert_eq!(ids(&["list", "--tag", "auth", "--tag", "ui"]), ["t-0"]);
    assert_eq!(ids(&["list", "--tag", "Auth"]), ["t-0", "t-1"]);
    assert_eq!(ids(&["list", "--untagged"]), ["t-2"]);
    refusal(agent_sac(&db, &["list", "--untagged", "--tag", "auth"]));

    // json: tasks carry tags as names
    let tasks = json_of(&ok(agent_sac(&db, &["--json", "list"])));
    assert_eq!(tasks["tasks"][0]["tags"], json!(["auth", "ui"]));
    assert_eq!(tasks["tasks"][1]["tags"], json!(["auth"]));
    assert_eq!(tasks["tasks"][2]["tags"], json!([]));
    let shown = json_of(&ok(agent_sac(&db, &["--json", "show", "t-1"])));
    assert_eq!(shown["blocks"][0]["tags"], json!(["auth"]));

    // show prints the task's tags with their lines
    let shown = ok(agent_sac(&db, &["show", "t-0"]));
    assert!(
        shown.contains("tags\n  auth - how people sign in\n  ui - what people see and click"),
        "{shown}"
    );

    // describing is the human's; the new line shows, the old stays in the log
    let err = refusal(agent_sac(&db, &["tag", "describe", "auth", "sign-in only"]));
    assert!(err.contains("only the human"), "{err}");
    ok(human_sac(&db, &["tag", "describe", "auth", "sign-in only"]));
    let listed = ok(agent_sac(&db, &["tag", "list"]));
    assert!(listed.contains("auth - sign-in only"), "{listed}");
    assert!(!listed.contains("how people sign in"), "{listed}");
    assert!(ok(agent_sac(&db, &["show", "t-1"])).contains("auth - sign-in only"));
    let log = ok(agent_sac(&db, &["log"]));
    assert!(
        log.contains("how people sign in"),
        "the earlier line stays in the record"
    );
    assert!(log.contains("tag_described"), "{log}");
    let err = refusal(agent_sac(&db, &["tag", "describe", "nope", "x"]));
    assert!(err.contains("no tag 'nope'"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_door_the_board_log_and_the_pages_carry_tags() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("tags-served");
    let (repo, _, _) = scratch_repo(&scratch.dir);
    let (_serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", repo.to_str().unwrap()],
        &[],
    );
    let sac = |args: &[&str]| -> (bool, String, String) {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--server")
            .arg(&url)
            .env("SACCADE_ACTOR", "pi")
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // the door: an agent's tag birth is refused, the human's lands
    let create = |tier: &str, name: &str| {
        post_command(
            &url,
            &envelope(
                "someone",
                tier,
                json!({"create_tag": {"name": name, "description": "what auth covers"}}),
            ),
        )
    };
    let (status, body) = create("agent", "auth");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "human_only");
    assert!(body.contains("only the human creates a tag"), "{body}");
    let (status, body) = create("human", "auth");
    assert_eq!(status, 200, "{body}");
    let auth = json_of(&body)["records"][0]["seq"].as_u64().unwrap();
    let (status, body) = create("human", "Auth");
    assert_ne!(status, 200, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "duplicate_tag");

    // a reference no tag was born at refuses over the door, naming the tags
    let (status, body) = post_command(
        &url,
        &envelope(
            "pi",
            "agent",
            json!({"create_task": {"name": "no such tag", "tags": [auth + 100]}}),
        ),
    );
    assert_ne!(status, 200, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_tag_id");
    assert!(body.contains("auth - what auth covers"), "{body}");

    // a task born without a tags field is born untagged
    let (status, body) = post_command(
        &url,
        &envelope("pi", "agent", json!({"create_task": {"name": "old shape"}})),
    );
    assert_eq!(status, 200, "{body}");

    // the CLI over the server: unknown names refuse from the fold's own
    // tags; an agent tags, and the board log names who
    let (good, _, err) = sac(&["create", "task", "--tag", "nope", "x"]);
    assert!(
        !good && err.contains("no tag 'nope'; the tags are:"),
        "{err}"
    );
    let (good, _, err) = sac(&["tag", "add", "t-0", "auth"]);
    assert!(good, "{err}");
    let (good, out, err) = sac(&["board", "log", "t-0"]);
    assert!(good, "{err}");
    assert!(out.contains("task_tagged auth by pi"), "{out}");
    let (good, _, err) = sac(&["tag", "remove", "t-0", "auth"]);
    assert!(good, "{err}");
    let (_, out, _) = sac(&["board", "log", "t-0"]);
    assert!(out.contains("task_untagged auth by pi"), "{out}");
    let (good, _, err) = sac(&["tag", "add", "t-0", "auth"]);
    assert!(good, "{err}");

    // board edit prints the task's tags after its staged line
    let staged = scratch.dir.join("staged");
    let (good, out, err) = sac(&["board", "edit", "t-0", staged.to_str().unwrap()]);
    assert!(good, "{err}");
    assert!(out.contains("tags\n  auth - what auth covers"), "{out}");
    std::fs::write(
        staged.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: sign-in holds\n",
    )
    .unwrap();
    let (good, _, err) = sac(&[
        "board",
        "commit",
        "t-0",
        staged.to_str().unwrap(),
        "-m",
        "first",
    ]);
    assert!(good, "{err}");

    let (good, _, err) = sac(&["create", "task", "plain"]);
    assert!(good, "{err}");

    // the pages: additive tags; the root carries name and line
    let (_, body) = get_json(&url, "/view/v1");
    let root = json_of(&body);
    assert_eq!(root["view"]["major"], 1);
    assert_eq!(
        root["tags"],
        json!([{"name": "auth", "description": "what auth covers"}])
    );
    let (_, body) = get_json(&url, "/view/v1/t-0");
    assert_eq!(json_of(&body)["tags"], json!(["auth"]));
    let (_, body) = get_json(&url, "/view/v1/tasks");
    let doc = json_of(&body);
    let mut seen = 0;
    for section in doc["content"].as_array().unwrap() {
        for table in section["content"].as_array().unwrap() {
            for row in table["rows"].as_array().unwrap() {
                seen += 1;
                let expected = if row["cells"][0] == "t-0" {
                    json!(["auth"])
                } else {
                    json!([])
                };
                assert_eq!(row["tags"], expected, "{row}");
            }
        }
    }
    assert_eq!(seen, 2, "both tasks ride the pipeline rows");
}
