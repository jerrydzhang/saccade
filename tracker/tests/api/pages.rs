// -- the /view/v1 pages --------------------------------------------------

use serde_json::{Value, json};

use crate::common::{
    ScratchDir, ServerProc, envelope, get_json, json_of, post_command, post_json,
    run_with_terminals, scratch_db, scrubbed, spawn_serve,
};

//
// The instruments' read surface: every answer a view document or raw
// file bytes, every wrapped part carrying its own location, waiting a
// cursor on the same route. The settled shapes are t-24@r2's.

fn view_page_project(tag: &str) -> (ScratchDir, std::path::PathBuf, ServerProc, String) {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db(tag);
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let git = |args: &[&str]| {
        saccade::boards::git_command()
            .args(args)
            .current_dir(&project)
            .output()
            .expect("git init");
    };
    git(&["init", "--quiet", project.to_str().unwrap()]);
    git(&[
        "-C",
        project.to_str().unwrap(),
        "config",
        "user.email",
        "smoke@saccade.invalid",
    ]);
    git(&[
        "-C",
        project.to_str().unwrap(),
        "config",
        "user.name",
        "smoke",
    ]);
    // a HEAD on the project repo: the workspace doors resolve their
    // default base from it
    std::fs::write(project.join("README.md"), "the smoke project\n").unwrap();
    git(&["-C", project.to_str().unwrap(), "add", "README.md"]);
    git(&[
        "-C",
        project.to_str().unwrap(),
        "commit",
        "-q",
        "-m",
        "seed",
    ]);
    let (serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );
    (scratch, project, serve, url)
}

fn view_page_sac(
    url: &str,
    incarnation_seq: Option<u64>,
    task: Option<&str>,
    args: &[&str],
) -> (bool, String, String) {
    let bin = env!("CARGO_BIN_EXE_sac");
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.arg("--server")
        .arg(url)
        .env("SACCADE_ACTOR", "pi")
        .args(args);
    if let Some(seq) = incarnation_seq {
        cmd.env("SACCADE_INCARNATION_SEQ", seq.to_string());
    }
    if let Some(task) = task {
        cmd.env("SACCADE_TASK", task);
    }
    let out = cmd.output().expect("spawn sac");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The fixture the pages read: a session behind the author, a source
/// board at two revisions (text, then group + figure + annotation), a
/// claim with the offer, reject and grounds between them, and a
/// consumer board incorporating the source's second revision.
fn view_page_fixture(
    tag: &str,
) -> (
    ScratchDir,
    ServerProc,
    String,
    usize,
    usize,
    String,
    std::path::PathBuf,
) {
    let (dir, project, serve, url) = view_page_project(tag);
    let _ = dir;
    let saved = project.parent().unwrap().join("viewpage01.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"viewpage01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            project.display()
        ),
    )
    .unwrap();
    let (_, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "viewpage01", "session_file": saved.to_str().unwrap()}),
    );
    let attach = json_of(&body);
    assert_eq!(attach["registered"], json!(true), "{attach}");
    let inc_num = attach["num"].as_u64().unwrap() as usize;
    let inc_seq = attach["incarnation"].as_u64().unwrap() as usize;

    let sac =
        |task: Option<&str>, args: &[&str]| view_page_sac(&url, Some(inc_seq as u64), task, args);
    let sac_json = |task: Option<&str>, args: &[&str]| {
        let mut full = vec!["--json"];
        full.extend_from_slice(args);
        let (ok, out, err) = sac(task, &full);
        assert!(ok, "{}: {err}", args.join(" "));
        json_of(&out)
    };
    // the creates ride no scope: their records name no session
    let (ok, out, err) = view_page_sac(
        &url,
        None,
        None,
        &["--json", "create", "task", "prove the loop"],
    );
    assert!(ok, "{err}");
    let _ = out;
    let (ok, _, err) = view_page_sac(
        &url,
        None,
        None,
        &["--json", "create", "task", "use the proof"],
    );
    assert!(ok, "{err}");
    let _ = sac_json(Some("t-0"), &["claim", "t-0"]);

    let src = project.join("src-dir");
    let (ok, out, _) = sac(
        Some("t-0"),
        &["board", "edit", "t-0", src.to_str().unwrap()],
    );
    assert!(ok, "{out}");
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input\n",
    )
    .unwrap();
    let reply = sac_json(
        Some("t-0"),
        &[
            "board",
            "commit",
            "t-0",
            src.to_str().unwrap(),
            "-m",
            "first account",
        ],
    );
    assert_eq!(reply["revision"], json!(1));
    let (ok, out, _) = sac(Some("t-0"), &["offer", "t-0"]);
    assert!(ok, "{out}");
    let (ok, out, _) = sac(
        None,
        &["reject", "t-0", "--grounds", "the boundary case is missing"],
    );
    assert!(ok, "{out}");

    let figures = src.join("figures");
    std::fs::create_dir_all(&figures).unwrap();
    std::fs::write(figures.join("plot.png"), b"\x89PNG-rusttest-bytes").unwrap();
    std::fs::write(figures.join("history"), b"a tree file named history").unwrap();
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input\n  - group:\n      title: evidence\n      content:\n        - figure:\n            input: figures/plot.png\n            caption: the plotted loop\n        - figure:\n            input: figures/history\n            caption: the lane's history note\n  - annotation: the boundary case is covered in the appendix\n",
    )
    .unwrap();
    let reply = sac_json(
        Some("t-0"),
        &[
            "board",
            "commit",
            "t-0",
            src.to_str().unwrap(),
            "-m",
            "annotate the boundary",
        ],
    );
    assert_eq!(reply["revision"], json!(2));

    let con = project.join("con-dir");
    let (ok, out, _) = sac(None, &["board", "edit", "t-1", con.to_str().unwrap()]);
    assert!(ok, "{out}");
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 2\n        select: [\"/content/1\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let reply = sac_json(
        None,
        &[
            "board",
            "commit",
            "t-1",
            con.to_str().unwrap(),
            "-m",
            "cite the proof",
        ],
    );
    assert_eq!(reply["revision"], json!(1));

    // a decision after the last commit: the current history carries
    // it, the pinned history at r2 cuts before it
    let (ok, out, _) = sac(Some("t-0"), &["offer", "t-0"]);
    assert!(ok, "{out}");
    (
        dir,
        serve,
        url,
        inc_num,
        inc_seq,
        src.join("board.yaml").display().to_string(),
        project,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn view_v1_pages_serve_boards_history_components_and_files() {
    let (_scratch, _serve, url, inc_num, _inc_seq, authored, repo_root) =
        view_page_fixture("view-docs");

    // the root page: what an instrument reads first
    let (status, body) = get_json(&url, "/view/v1");
    assert_eq!(status, 200, "{body}");
    let root = json_of(&body);
    assert_eq!(root["view"]["major"], json!(1));
    assert_eq!(root["kind"], json!("root"));
    assert_eq!(root["location"], json!("/"));
    assert_eq!(root["tracker"], json!("project"));
    assert_eq!(root["view_majors"], json!([1]));
    assert_eq!(root["board_grammar"], json!({"min": 1, "max": 1}));
    // the host facts a consumer needs to prove hub identity (t-24@r14,
    // t-24@r16): the machine id is the trimmed /etc/machine-id or
    // absent, and the repository path names the tracker beside it
    let expected_id = std::fs::read_to_string("/etc/machine-id")
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|id| !id.is_empty());
    assert_eq!(
        root["machine_id"],
        expected_id
            .map(|id| json!(id))
            .unwrap_or(serde_json::Value::Null)
    );
    assert_eq!(root["repo_path"], json!(repo_root.display().to_string()));
    assert!(root["seq"].as_u64().is_some(), "{root}");
    let families: Vec<&str> = root["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|part| part["kind"] == json!("fact"))
        .filter_map(|part| part["value"].as_str())
        .collect();
    assert!(
        families.iter().any(|value| value.starts_with("boards:")),
        "{root}"
    );
    // the trailing-slash root is the same page
    let (status, body) = get_json(&url, "/view/v1/");
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["kind"], json!("root"));

    // the current board page: facts, refs, and every authored part
    // carrying its own location
    let (status, body) = get_json(&url, "/view/v1/t-0");
    assert_eq!(status, 200, "{body}");
    let page = json_of(&body);
    assert_eq!(page["kind"], json!("document"));
    assert_eq!(page["location"], json!("t-0@r2"));
    let content = page["content"].as_array().unwrap();
    let labeled = |label: &str| {
        content
            .iter()
            .find(|part| part["kind"] == json!("fact") && part["label"] == json!(label))
            .unwrap_or_else(|| panic!("no {label} fact: {page}"))
            .clone()
    };
    assert_eq!(labeled("task")["value"], json!("t-0"));
    assert_eq!(labeled("name")["value"], json!("prove the loop"));
    // the fixture ends with t-0 offered after its second commit
    assert_eq!(labeled("state")["value"], json!("offered"));
    assert_eq!(labeled("owner")["value"], json!(format!("i-{inc_num}")));
    assert_eq!(labeled("revision")["value"], json!("r2 of 2"));
    assert!(labeled("commit").is_object());
    let refs: Vec<&Value> = content
        .iter()
        .filter(|part| part["kind"] == json!("ref"))
        .collect();
    assert!(
        refs.iter()
            .any(|r| r["location"] == json!("t-0/history") && r["label"] == json!("history")),
        "{refs:?}"
    );
    assert!(
        refs.iter()
            .any(|r| r["location"] == json!("t-0@r2/file/board.yaml")),
        "{refs:?}"
    );
    let text = content
        .iter()
        .find(|part| part["kind"] == json!("text"))
        .unwrap();
    assert_eq!(text["text"], json!("the loop holds for every input"));
    assert_eq!(text["location"], json!("t-0@r2:/content/0"));
    let group = content
        .iter()
        .find(|part| part["kind"] == json!("group") && part["title"] == json!("evidence"))
        .unwrap();
    assert_eq!(group["location"], json!("t-0@r2:/content/1"));
    let figure = &group["content"][0];
    assert_eq!(figure["kind"], json!("figure"));
    assert_eq!(figure["caption"], json!("the plotted loop"));
    assert_eq!(figure["file"], json!("t-0@r2/file/figures/plot.png"));
    assert_eq!(
        figure["location"],
        json!("t-0@r2:/content/1/group/content/0")
    );
    let annotation = content
        .iter()
        .find(|part| part["title"] == json!("annotation"))
        .unwrap();
    assert_eq!(annotation["location"], json!("t-0@r2:/content/2"));
    assert_eq!(
        annotation["content"][0]["location"],
        json!("t-0@r2:/content/2")
    );

    // the pinned board page: r1, with the switch to the current revision
    let (status, body) = get_json(&url, "/view/v1/t-0@r1");
    assert_eq!(status, 200, "{body}");
    let pinned = json_of(&body);
    assert_eq!(pinned["location"], json!("t-0@r1"));
    let content = pinned["content"].as_array().unwrap();
    let labeled = |label: &str| {
        content
            .iter()
            .find(|part| part["kind"] == json!("fact") && part["label"] == json!(label))
            .unwrap_or_else(|| panic!("no {label} fact: {pinned}"))
            .clone()
    };
    assert_eq!(labeled("revision")["value"], json!("r1 of 2"));
    assert!(
        content.iter().any(|part| part["kind"] == json!("ref")
            && part["location"] == json!("t-0")
            && part["label"] == json!("current revision r2")),
        "{content:?}"
    );
    assert!(
        content.iter().any(|part| part["kind"] == json!("ref")
            && part["location"] == json!("t-0@r1/history")),
        "{content:?}"
    );
    assert_eq!(
        content
            .iter()
            .filter(|p| p["kind"] == json!("text"))
            .count(),
        1
    );
    assert_eq!(
        content.iter().find(|p| p["kind"] == json!("text")).unwrap()["location"],
        json!("t-0@r1:/content/0")
    );

    // the component page: a select path names one part, pinned
    let (status, body) = get_json(&url, "/view/v1/t-0@r2/content/1/group/content/0");
    assert_eq!(status, 200, "{body}");
    let component = json_of(&body);
    assert_eq!(component["kind"], json!("document"));
    assert_eq!(
        component["location"],
        json!("t-0@r2/content/1/group/content/0")
    );
    assert_eq!(component["content"][0]["kind"], json!("figure"));
    assert_eq!(
        component["content"][0]["location"],
        json!("t-0@r2:/content/1/group/content/0")
    );
    let (status, body) = get_json(&url, "/view/v1/t-0@r2/content/9");
    assert_eq!(status, 404, "{body}");
    let refusal = json_of(&body);
    assert_eq!(refusal["kind"], json!("refusal"));
    assert_eq!(refusal["code"], json!("component_missing"));
    assert!(refusal["content"][0]["kind"] == json!("fact"), "{refusal}");
    assert!(
        refusal["content"][1]["text"].as_str().is_some(),
        "{refusal}"
    );

    // the consumer's board page: the incorporation collapsed, its
    // receiving text and citation refs riding inside
    let (status, body) = get_json(&url, "/view/v1/t-1");
    assert_eq!(status, 200, "{body}");
    let consumer = json_of(&body);
    let incorporation = consumer["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|part| part["title"] == json!("incorporates t-0@r2"))
        .unwrap_or_else(|| panic!("no incorporation group: {consumer}"));
    assert_eq!(incorporation["location"], json!("t-1@r1:/content/0"));
    assert_eq!(incorporation["content"][0]["kind"], json!("ref"));
    assert_eq!(incorporation["content"][0]["location"], json!("t-0@r2"));
    assert_eq!(
        incorporation["content"][1]["location"],
        json!("t-1@r1:/content/0/incorporation/content/0")
    );
    assert_eq!(incorporation["content"][2]["kind"], json!("ref"));
    assert_eq!(
        incorporation["content"][2]["location"],
        json!("t-0@r2/content/1")
    );

    // the history page: the full story in record order, commits with
    // their messages, decisions with their grounds, the session behind
    // the record, and the page refs
    let (status, body) = get_json(&url, "/view/v1/t-0/history");
    assert_eq!(status, 200, "{body}");
    let history = json_of(&body);
    assert_eq!(history["kind"], json!("timeline"));
    assert_eq!(history["location"], json!("t-0/history"));
    assert_eq!(history["task"], json!("t-0"));
    assert_eq!(history["revision"], json!(2));
    let entries = history["entries"].as_array().unwrap();
    let events: Vec<&str> = entries
        .iter()
        .map(|entry| entry["event"].as_str().unwrap())
        .collect();
    assert!(
        events.contains(&"task_created")
            && events.contains(&"task_claimed")
            && events.contains(&"task_committed")
            && events.contains(&"task_offered")
            && events.contains(&"task_rejected"),
        "{events:?}"
    );
    let committed = entries
        .iter()
        .find(|entry| entry["event"] == json!("task_committed") && entry["revision"] == json!(2))
        .unwrap();
    assert_eq!(committed["message"], json!("annotate the boundary"));
    assert_eq!(
        committed["session"]["location"],
        json!(format!("i-{inc_num}"))
    );
    let entry_refs = committed["refs"].as_array().unwrap();
    assert!(
        entry_refs
            .iter()
            .any(|r| r["location"] == json!("t-0@r2") && r["label"] == json!("revision r2")),
        "{entry_refs:?}"
    );
    assert!(
        entry_refs
            .iter()
            .any(|r| r["location"] == json!("t-0@r1..r2/diff")),
        "{entry_refs:?}"
    );
    let rejected = entries
        .iter()
        .find(|entry| entry["event"] == json!("task_rejected"))
        .unwrap();
    assert_eq!(rejected["grounds"], json!("the boundary case is missing"));
    // the deciding session rode the reject too
    assert_eq!(
        rejected["session"]["location"],
        json!(format!("i-{inc_num}"))
    );
    // entries whose author carried no scope name none
    let created = entries
        .iter()
        .find(|entry| entry["event"] == json!("task_created"))
        .unwrap();
    assert_eq!(created["session"], json!(Value::Null));
    let claimed = entries
        .iter()
        .find(|entry| entry["event"] == json!("task_claimed"))
        .unwrap();
    assert_eq!(
        claimed["session"]["location"],
        json!(format!("i-{inc_num}"))
    );

    // the pinned history: the record cut at the revision's own commit
    let (status, body) = get_json(&url, "/view/v1/t-0@r1/history");
    assert_eq!(status, 200, "{body}");
    let pinned_history = json_of(&body);
    assert_eq!(pinned_history["location"], json!("t-0@r1/history"));
    let pinned_events: Vec<&str> = pinned_history["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["event"].as_str().unwrap())
        .collect();
    assert!(
        pinned_events.contains(&"task_committed"),
        "{pinned_events:?}"
    );
    assert!(
        !pinned_events.contains(&"task_offered") && !pinned_events.contains(&"task_rejected"),
        "the cut runs at the revision's commit: {pinned_events:?}"
    );

    // the pinned r2 history: r1's offer is inside the cut, the
    // post-commit offer is outside it
    let (status, body) = get_json(&url, "/view/v1/t-0@r2/history");
    assert_eq!(status, 200, "{body}");
    let pinned_r2 = json_of(&body);
    assert_eq!(pinned_r2["location"], json!("t-0@r2/history"));
    let pinned_r2_events: Vec<&str> = pinned_r2["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        pinned_r2_events
            .iter()
            .filter(|event| **event == "task_offered")
            .count(),
        1,
        "only the offer inside the cut: {pinned_r2_events:?}"
    );
    assert!(
        pinned_r2_events.last() == Some(&"task_committed"),
        "the pinned timeline ends at its revision's commit: {pinned_r2_events:?}"
    );

    // the diff page: the stored unified diff, one text part
    let (status, body) = get_json(&url, "/view/v1/t-0@r1..r2/diff");
    assert_eq!(status, 200, "{body}");
    let diff = json_of(&body);
    assert_eq!(diff["kind"], json!("document"));
    assert_eq!(diff["location"], json!("t-0@r1..r2/diff"));
    assert_eq!(diff["content"][0]["kind"], json!("text"));
    let diff_text = diff["content"][0]["text"].as_str().unwrap();
    assert!(diff_text.contains("annotation"), "{diff_text}");
    let (_, api_body) = get_json(&url, "/api/v1/boards/diff?task=0&from=1&to=2");
    assert_eq!(
        diff_text,
        json_of(&api_body)["diff"].as_str().unwrap(),
        "the page carries the stored diff"
    );
    // and the component changes `sac board diff` names (t-57), select
    // paths in the revision each side names
    assert_eq!(
        diff["changes"],
        json!([
            {"change": "added", "label": "group \"evidence\"", "to": "/content/1"},
            {"change": "added", "label": "annotation", "to": "/content/2"},
        ]),
        "{body}"
    );

    // the file family: raw bytes, typed by extension
    let mut response = ureq::get(&format!("{url}/view/v1/t-0@r2/file/board.yaml"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(response.headers().get("content-type").unwrap(), "text/yaml");
    let bytes = response.body_mut().read_to_vec().unwrap();
    assert_eq!(
        bytes,
        std::fs::read(&authored).unwrap(),
        "the authored bytes"
    );
    let mut response = ureq::get(&format!("{url}/view/v1/t-0@r2/file/figures/plot.png"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(response.headers().get("content-type").unwrap(), "image/png");
    let bytes = response.body_mut().read_to_vec().unwrap();
    assert_eq!(bytes, b"\x89PNG-rusttest-bytes");
    let (status, body) = get_json(&url, "/view/v1/t-0@r2/file/figures/absent.png");
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["code"], json!("file_missing"));

    // the refusals are documents with informative statuses
    let (status, body) = get_json(&url, "/view/v1/t-99");
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["code"], json!("invalid_task_id"));
    let (status, body) = get_json(&url, "/view/v1/nonsense");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["code"], json!("unknown_location"));
    let (status, body) = get_json(&url, "/view/v1/t-0@r9");
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["code"], json!("revision_missing"));
    let (status, body) = get_json(&url, "/view/v1/t-0/content/0");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["code"], json!("unknown_location"));
    let (status, body) = get_json(&url, "/view/v1/t-0@r2/file/%2E%2E/secret.yaml");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["code"], json!("bad_file_path"));
    let (status, body) = get_json(&url, "/view/v1/t-0?after=notaseq");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["code"], json!("bad_query"));

    // the page reads ride the same origin policy as every route
    let response = ureq::get(&format!("{url}/view/v1/t-0"))
        .header("Origin", "http://tauri.localhost")
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "http://tauri.localhost"
    );
}

// The landing wake fires wherever a door lands a write outside
// execute: a parked page hold wakes on the workspace, attach,
// registration and address doors, and on the board commit - not just
// on the command face.
#[tokio::test(flavor = "multi_thread")]
async fn a_parked_page_wakes_when_the_other_write_doors_land_records() {
    let (_scratch, _serve, url, inc_num, _inc_seq, _authored, _repo_root) =
        view_page_fixture("view-wakes");

    // the board-commit door: a parked task page wakes on the commit
    let (_, body) = get_json(&url, "/view/v1/");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let commit_wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/t-0?after={cursor}&timeout=8"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let work = std::env::temp_dir().join(format!("sac-view-wakes-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    // the fixture leaves t-0 offered: the wake test's commit needs
    // the offer rejected first
    let (ok, _, err) = view_page_sac(
        &url,
        None,
        None,
        &[
            "reject",
            "t-0",
            "--grounds",
            "the wake test revises the board",
        ],
    );
    assert!(ok, "{err}");
    let (ok, out, err) = view_page_sac(
        &url,
        None,
        None,
        &["board", "edit", "t-0", work.to_str().unwrap()],
    );
    assert!(ok, "{out}{err}");
    std::fs::write(
        work.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the wake test board\n",
    )
    .unwrap();
    let (ok, _, err) = view_page_sac(
        &url,
        None,
        None,
        &[
            "board",
            "commit",
            "t-0",
            work.to_str().unwrap(),
            "-m",
            "wake the parked page",
        ],
    );
    assert!(ok, "{err}");
    let started = std::time::Instant::now();
    let (status, body) = commit_wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["seq"].as_u64().unwrap() > cursor,
        "the board commit woke the parked page"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the commit's landing woke the hold, not the deadline"
    );

    // the workspace create door
    let (_, body) = get_json(&url, "/view/v1/");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let root_wait = {
        let url = url.clone();
        std::thread::spawn(move || get_json(&url, &format!("/view/v1/?after={cursor}&timeout=8")))
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (status, body) = post_json(
        &url,
        "/api/v1/workspace/create",
        &json!({"label": "wake-test"}),
    );
    assert_eq!(status, 200, "{body}");
    let started = std::time::Instant::now();
    let (status, body) = root_wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["seq"].as_u64().unwrap() > cursor,
        "the workspace birth woke the parked root"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(5));

    // the attach door: a new session registers, the parked root wakes
    let (_, body) = get_json(&url, "/view/v1/");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let root_wait = {
        let url = url.clone();
        std::thread::spawn(move || get_json(&url, &format!("/view/v1/?after={cursor}&timeout=8")))
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let saved = work.join("wakesession.jsonl");
    std::fs::write(
        &saved,
        "{\"type\":\"session\",\"version\":3,\"id\":\"wakesession1\",\"timestamp\":\"t\",\"cwd\":\"/tmp\"}\n",
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "wakesession1", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    let started = std::time::Instant::now();
    let (status, body) = root_wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["seq"].as_u64().unwrap() > cursor,
        "the registration woke the parked root"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(5));

    // the address door: the address assignment wakes the parked root
    let (_, body) = get_json(&url, "/view/v1/");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let root_wait = {
        let url = url.clone();
        std::thread::spawn(move || get_json(&url, &format!("/view/v1/?after={cursor}&timeout=8")))
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": inc_num, "address": "wake-tty", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let started = std::time::Instant::now();
    let (status, body) = root_wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["seq"].as_u64().unwrap() > cursor,
        "the address assignment woke the parked root"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

#[tokio::test(flavor = "multi_thread")]
async fn view_v1_wait_parks_until_a_relevant_record_lands() {
    let (_scratch, _serve, url, _inc_num, _inc_seq, _authored, _repo_root) =
        view_page_fixture("view-wait");

    // the cursor the waits hang from
    let (_, body) = get_json(&url, "/view/v1/");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();

    // a wait on the board page parks: an unrelated task's birth does
    // not wake it
    let (tx, rx) = std::sync::mpsc::channel::<(u16, String)>();
    let wait_url = url.clone();
    let waiter = std::thread::spawn(move || {
        let (status, body) = get_json(&wait_url, &format!("/view/v1/t-0?after={cursor}&timeout=8"));
        tx.send((status, body)).unwrap();
    });
    std::thread::sleep(std::time::Duration::from_millis(600));
    let (status, body) = post_command(
        &url,
        &envelope(
            "pi",
            "agent",
            json!({"create_task": {"name": "an unrelated task"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    std::thread::sleep(std::time::Duration::from_millis(1200));
    assert!(
        rx.try_recv().is_err(),
        "an unrelated record must not wake the page's wait"
    );

    // a record the fold attributes to t-0 wakes it, and the answer is
    // the fresh page
    let (status, body) = post_command(
        &url,
        &envelope(
            "pi",
            "agent",
            json!({"claim_task": {"id": 0, "incarnation": null}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let started = std::time::Instant::now();
    waiter.join().unwrap();
    let _ = started;
    let (status, body) = rx.try_recv().expect("the wait woke");
    assert_eq!(status, 200, "{body}");
    let page = json_of(&body);
    assert_eq!(page["kind"], json!("document"));
    let woke_seq = page["seq"].as_u64().unwrap();
    assert!(woke_seq > cursor, "{page}");
    let content = page["content"].as_array().unwrap();
    assert!(
        content.iter().any(|part| part["label"] == json!("owner")),
        "the fresh page carries the claim: {page}"
    );

    // a cursor already passed answers at once
    let started = std::time::Instant::now();
    let (status, body) = get_json(&url, &format!("/view/v1/t-0?after={}&timeout=8", cursor));
    assert_eq!(status, 200, "{body}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "a passed cursor answers at once"
    );

    // the hold expiring is quiet: the page as it stands, a normal answer
    let (_, body) = get_json(&url, "/view/v1/");
    let fresh = json_of(&body)["seq"].as_u64().unwrap();
    let started = std::time::Instant::now();
    let (status, body) = get_json(&url, &format!("/view/v1/t-0?after={fresh}&timeout=1"));
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["kind"], json!("document"));
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(900),
        "the quiet answer held for its bound"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(3),
        "the quiet answer did not over-hold"
    );

    // the root wakes on any record
    let root_url = url.clone();
    let root_wait = std::thread::spawn(move || {
        get_json(&root_url, &format!("/view/v1/?after={fresh}&timeout=8"))
    });
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (status, body) = post_command(
        &url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "wake the root"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let started = std::time::Instant::now();
    let (status, body) = root_wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["seq"].as_u64().unwrap() > fresh,
        "the root woke on any record"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(4));

    // a refusal parks through its watch: a page that answers nothing
    // today answers as it stands when the hold expires
    let started = std::time::Instant::now();
    let (status, body) = get_json(&url, &format!("/view/v1/t-99?after={fresh}&timeout=1"));
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["kind"], json!("refusal"));
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(900),
        "the refusal parked for its bound first"
    );
}

// The agents page (t-24@r4): one group per registered native session,
// its location the incarnation, facts inside naming the terminal
// address, the harness, the owned tasks and the workspace - the
// workspace derived server-side from the saved session's own header.
// The hold wakes on the registration family and the claims, not on
// unrelated records.
#[tokio::test(flavor = "multi_thread")]
async fn the_agents_page_names_sessions_ownership_and_workspaces() {
    let (_scratch, _serve, url, inc_num, inc_seq, _authored, _project) =
        view_page_fixture("agents-page");
    let fact_value = |doc: &Value, incarnation: &str, label: &str| -> Option<String> {
        doc["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["kind"] == "group" && g["location"] == json!(incarnation))
            .expect("the session's group is on the page")["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["kind"] == "fact" && p["label"] == label)
            .map(|f| f["value"].as_str().unwrap().to_string())
    };

    // the root names the hub: the machine id equals the host's file
    // (or is absent when the host carries none) and the repo path is
    // the served repository
    let (_, body) = get_json(&url, "/view/v1/");
    let root = json_of(&body);
    let host_id = std::fs::read_to_string("/etc/machine-id")
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|id| !id.is_empty());
    assert_eq!(
        root["machine_id"].as_str(),
        host_id.as_deref(),
        "the root's machine id is the host's, never defaulted"
    );
    assert!(root["repo_path"].as_str().is_some());

    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    assert_eq!(doc["kind"], "document");
    assert_eq!(doc["location"], "agents");
    let me = format!("i-{inc_num}");
    assert_eq!(
        fact_value(&doc, &me, "incarnation").as_deref(),
        Some(me.as_str())
    );
    assert_eq!(fact_value(&doc, &me, "harness").as_deref(), Some("pi"));
    assert_eq!(fact_value(&doc, &me, "tasks").as_deref(), Some("t-0"));
    // the workspace came from the saved pi header's own cwd
    assert!(fact_value(&doc, &me, "workspace").is_some());

    // a second registration: an address and a harness, nothing saved
    // and nothing claimed - the facts that exist are the only facts
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "agents-page-second", "address": "tA", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let second_num = inc_num + 1;
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    let theirs = format!("i-{second_num}");
    assert_eq!(
        fact_value(&doc, &theirs, "terminal address").as_deref(),
        Some("tA")
    );
    assert_eq!(fact_value(&doc, &theirs, "tasks"), None);
    assert_eq!(fact_value(&doc, &theirs, "open tasks"), None);
    assert_eq!(fact_value(&doc, &theirs, "workspace"), None);

    // ownership: the page carries every owned task, latest claim first
    let (ok, _, err) = view_page_sac(&url, Some(inc_seq as u64), Some("t-1"), &["claim", "t-1"]);
    assert!(ok, "{err}");
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    assert_eq!(
        fact_value(&doc, &me, "tasks").as_deref(),
        Some("t-1 t-0"),
        "latest claim first"
    );
    assert_eq!(
        fact_value(&doc, &me, "open tasks").as_deref(),
        Some("t-1 t-0"),
        "both owned tasks are still open"
    );

    // a task that leaves play stays owned but is no longer open: the
    // human cancels t-0 (human tier: no actor, a terminal attached)
    let out = run_with_terminals(
        env!("CARGO_BIN_EXE_sac"),
        &[
            "--server",
            &url,
            "cancel",
            "t-0",
            "--explanation",
            "set aside",
        ]
        .map(str::to_string),
        true,
        true,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    assert_eq!(
        fact_value(&doc, &me, "tasks").as_deref(),
        Some("t-1 t-0"),
        "ownership outlives the task's end"
    );
    assert_eq!(
        fact_value(&doc, &me, "open tasks").as_deref(),
        Some("t-1"),
        "a cancelled task is no longer open"
    );

    // supersession: a fresh holder of the same address removes the
    // older holder from the page - the fold names one live holder
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "agents-page-third", "address": "tA", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let third_num = inc_num + 2;
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    let locations: Vec<String> = doc["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|g| g["location"].as_str().map(str::to_string))
        .collect();
    assert!(
        locations.contains(&format!("i-{third_num}")),
        "{locations:?}"
    );
    assert!(
        !locations.contains(&format!("i-{second_num}")),
        "the superseded holder is off the page: {locations:?}"
    );

    // the hold wakes when a registration lands
    let (_, body) = get_json(&url, "/view/v1/agents");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/agents?after={cursor}&timeout=8"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "agents-page-fourth", "address": "tB", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["seq"].as_u64().unwrap() > cursor,
        "the hold answered past its cursor"
    );

    // and when a claim lands: the ownership facts are the page's too
    let (_, body) = get_json(&url, "/view/v1/agents");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/agents?after={cursor}&timeout=8"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (ok, _, err) = view_page_sac(&url, Some(inc_seq as u64), Some("t-1"), &["claim", "t-0"]);
    assert!(ok, "{err}");
    let (status, body) = wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(json_of(&body)["seq"].as_u64().unwrap() > cursor);

    // and when an owned task comes back into play: the open tasks are
    // the page's too (the human reopens the cancelled t-0)
    let (_, body) = get_json(&url, "/view/v1/agents");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let hold_started = std::time::Instant::now();
    let wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/agents?after={cursor}&timeout=8"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let out = run_with_terminals(
        env!("CARGO_BIN_EXE_sac"),
        &[
            "--server",
            &url,
            "reopen",
            "t-0",
            "--explanation",
            "back in play",
        ]
        .map(str::to_string),
        true,
        true,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (status, body) = wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        hold_started.elapsed() < std::time::Duration::from_secs(6),
        "the reopen woke the hold, well before its bound"
    );
    let doc = json_of(&body);
    assert!(doc["seq"].as_u64().unwrap() > cursor);
    assert_eq!(
        fact_value(&doc, &me, "open tasks").as_deref(),
        Some("t-0 t-1"),
        "the reopened task is open again, the latest claim first"
    );

    // an unrelated landing keeps the hold quiet for its bound: a task
    // creation is not the agents page's subject
    let (_, body) = get_json(&url, "/view/v1/agents");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let hold_started = std::time::Instant::now();
    let wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/agents?after={cursor}&timeout=3"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (ok, _, err) = view_page_sac(
        &url,
        None,
        None,
        &["create", "task", "unrelated to the agents"],
    );
    assert!(ok, "{err}");
    let (status, body) = wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    // the unrelated landing landed mid-hold: the answer is the bound
    // expiring, not the landing waking it
    assert!(
        hold_started.elapsed() >= std::time::Duration::from_millis(2800),
        "the hold stayed quiet for its bound first"
    );
}

// The pipeline page (t-14 "Tasks and agents"): one view, sectioned by
// recorded state - waiting (open, not claimed), in flight (open and
// claimed), awaiting judgment (offered), concluded in the last 24
// hours. A row is task, title, agents, latest commit, latest revision,
// with the task's birth moment riding the row and its location the board Enter opens
// through the link. The page's watch is the whole fold: any landing
// can move some row.
#[tokio::test(flavor = "multi_thread")]
async fn the_pipeline_page_sections_the_recorded_state() {
    let (_scratch, _serve, url, inc_num, inc_seq, _authored, _project) =
        view_page_fixture("pipeline-page");

    let section_table = |doc: &Value, title: &str| -> Value {
        doc["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["kind"] == "group" && g["title"] == json!(title))
            .unwrap_or_else(|| panic!("the {title} section is on the page"))["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["kind"] == "table")
            .expect("the section holds one table")
            .clone()
    };
    let row_of = |table: &Value, task: &str| -> Option<Value> {
        table["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["cells"].as_array().unwrap().first() == Some(&json!(task)))
            .cloned()
    };

    let (_, body) = get_json(&url, "/view/v1/tasks");
    let doc = json_of(&body);
    assert_eq!(doc["kind"], "document");
    assert_eq!(doc["location"], "tasks");
    let waiting = section_table(&doc, "waiting");
    assert_eq!(
        waiting["columns"],
        json!([
            "task",
            "title",
            "agents",
            "latest commit",
            "latest revision"
        ])
    );
    // the fixture's end state: t-1 open and unclaimed, t-0 offered
    let t1 = row_of(&waiting, "t-1").expect("t-1 waits");
    assert_eq!(t1["cells"][1], json!("use the proof"));
    assert_eq!(t1["cells"][2], json!(""), "unclaimed: no agents");
    assert_eq!(t1["location"], json!("t-1"));
    assert!(
        t1["at"].as_u64().is_some(),
        "the row carries its birth moment"
    );
    let offered = section_table(&doc, "awaiting judgment");
    let t0 = row_of(&offered, "t-0").expect("t-0 is offered");
    assert_eq!(t0["cells"][2], json!(format!("i-{inc_num}")));
    assert_eq!(t0["location"], json!("t-0"));
    // the latest commit is the board page's own commit fact
    let (_, body) = get_json(&url, "/view/v1/t-0");
    let commit = json_of(&body)["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["kind"] == "fact" && p["label"] == "commit")
        .unwrap()["value"]
        .clone();
    assert_eq!(t0["cells"][3], commit);
    // the latest revision names the revision that commit published: the
    // board page's own revision fact reads "rK of N"
    let (_, body) = get_json(&url, "/view/v1/t-0");
    let revision_fact = json_of(&body)["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["kind"] == "fact" && p["label"] == "revision")
        .unwrap()["value"]
        .as_str()
        .unwrap()
        .to_string();
    let shown = revision_fact.split(' ').next().unwrap().to_string();
    assert_eq!(t0["cells"][4], json!(shown));
    // the empty sections are present, plain and empty
    assert!(
        section_table(&doc, "in flight")["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        section_table(&doc, "concluded last 24h")["rows"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    // a claim moves the task into flight, naming its agent
    let (ok, _, err) = view_page_sac(&url, Some(inc_seq as u64), Some("t-1"), &["claim", "t-1"]);
    assert!(ok, "{err}");
    let (_, body) = get_json(&url, "/view/v1/tasks");
    let doc = json_of(&body);
    let t1 = row_of(&section_table(&doc, "in flight"), "t-1").expect("t-1 is in flight");
    assert_eq!(t1["cells"][2], json!(format!("i-{inc_num}")));
    assert!(row_of(&section_table(&doc, "waiting"), "t-1").is_none());

    // the acceptance concludes, inside the 24h window - accepted by
    // another actor, never the claimer (self_acceptance refuses)
    let (status, body) = post_command(
        &url,
        &envelope("hub", "human", json!({"accept_board": {"task": 0}})),
    );
    assert_eq!(status, 200, "{body}");
    let (_, body) = get_json(&url, "/view/v1/tasks");
    let doc = json_of(&body);
    assert!(row_of(&section_table(&doc, "concluded last 24h"), "t-0").is_some());
    assert!(row_of(&section_table(&doc, "awaiting judgment"), "t-0").is_none());

    // the page's watch is the whole fold: a parked hold wakes on the
    // next landing, and the answer carries the landing's row
    let (_, body) = get_json(&url, "/view/v1/tasks");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/tasks?after={cursor}&timeout=8"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (ok, out, _) = view_page_sac(&url, None, None, &["create", "task", "a new waiting row"]);
    assert!(ok, "{out}");
    let started = std::time::Instant::now();
    let (status, body) = wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(7),
        "the landing woke the hold, the deadline did not"
    );
    let doc = json_of(&body);
    let fresh = row_of(&section_table(&doc, "waiting"), "t-2").expect("the new task waits");
    assert_eq!(fresh["cells"][1], json!("a new waiting row"));
}

// The agents page's one t-18 addition (t-24@r4): each session's last
// board commit, from the scope recorded on each commit. Nothing where
// the session landed no commit; a commit moves only its own session's
// fact; the commit landing wakes the page.
#[tokio::test(flavor = "multi_thread")]
async fn the_agents_page_names_each_sessions_last_commit() {
    let (_scratch, _serve, url, inc_num, _inc_seq, _authored, _project) =
        view_page_fixture("agents-last-commit");
    let fact_value = |doc: &Value, incarnation: &str, label: &str| -> Option<String> {
        doc["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["kind"] == "group" && g["location"] == json!(incarnation))
            .expect("the session's group is on the page")["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["kind"] == "fact" && p["label"] == label)
            .map(|f| f["value"].as_str().unwrap().to_string())
    };
    let board_commit = |task: &str| -> String {
        let (_, body) = get_json(&url, &format!("/view/v1/{task}"));
        json_of(&body)["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["kind"] == "fact" && p["label"] == "commit")
            .unwrap()["value"]
            .as_str()
            .unwrap()
            .to_string()
    };

    // the fixture's last session-scoped commit is t-1's r1, after
    // t-0's two revisions
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    let me = format!("i-{inc_num}");
    assert_eq!(
        fact_value(&doc, &me, "last commit").as_deref(),
        Some(board_commit("t-1").as_str()),
        "the last commit the session's scope rode"
    );

    // a second session lands no commit: no fact
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "agents-last-commit-second", "address": "tB", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let second_seq = json_of(&body)["incarnation"].as_u64().unwrap();
    let theirs = format!("i-{}", inc_num + 1);
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    assert_eq!(fact_value(&doc, &theirs, "last commit"), None);

    // the second session's own commit moves only its own fact
    let (ok, _, err) = view_page_sac(&url, None, None, &["create", "task", "a fresh board"]);
    assert!(ok, "{err}");
    let (ok, _, err) = view_page_sac(&url, Some(second_seq), Some("t-2"), &["claim", "t-2"]);
    assert!(ok, "{err}");
    let src = _project.join("lastcommit-dir");
    let (ok, out, _) = view_page_sac(
        &url,
        Some(second_seq),
        Some("t-2"),
        &["board", "edit", "t-2", src.to_str().unwrap()],
    );
    assert!(ok, "{out}");
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-2\ncontent:\n  - text: the second session's own commit\n",
    )
    .unwrap();
    let (ok, _, err) = view_page_sac(
        &url,
        Some(second_seq),
        Some("t-2"),
        &[
            "board",
            "commit",
            "t-2",
            src.to_str().unwrap(),
            "-m",
            "the second session commits",
        ],
    );
    assert!(ok, "{err}");
    let (_, body) = get_json(&url, "/view/v1/agents");
    let doc = json_of(&body);
    assert_eq!(
        fact_value(&doc, &theirs, "last commit").as_deref(),
        Some(board_commit("t-2").as_str())
    );
    assert_eq!(
        fact_value(&doc, &me, "last commit").as_deref(),
        Some(board_commit("t-1").as_str()),
        "the first session's fact is its own last commit"
    );

    // a commit landing wakes the page: the watch grew the commit kind
    let (_, body) = get_json(&url, "/view/v1/agents");
    let cursor = json_of(&body)["seq"].as_u64().unwrap();
    let wait = {
        let url = url.clone();
        std::thread::spawn(move || {
            get_json(&url, &format!("/view/v1/agents?after={cursor}&timeout=8"))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(400));
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-2\ncontent:\n  - text: the second session's waking commit\n",
    )
    .unwrap();
    let (ok, _, err) = view_page_sac(
        &url,
        Some(second_seq),
        Some("t-2"),
        &[
            "board",
            "commit",
            "t-2",
            src.to_str().unwrap(),
            "-m",
            "the waking commit",
        ],
    );
    assert!(ok, "{err}");
    let started = std::time::Instant::now();
    let (status, body) = wait.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(7),
        "the commit landing woke the hold"
    );
}
