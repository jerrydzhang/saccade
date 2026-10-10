//! The /api/v1 surface over a real loopback socket: one envelope, records
//! back or a typed error, the tier law holding through HTTP exactly as it
//! holds in process.

use std::path::PathBuf;

use serde_json::{Value, json};

use saccade::api::AppState;

fn scratch_db(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sac-api-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("saccade.db")
}

async fn spawn_server(db_path: &std::path::Path) -> String {
    let state = AppState::open(db_path).expect("the scratch tracker opens");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = axum::Router::new()
        .merge(saccade::api::routes())
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn post_command(base_url: &str, envelope: &Value) -> (u16, String) {
    let mut r = ureq::post(&format!("{base_url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(envelope.clone())
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

fn post_raw(base_url: &str, body: &str) -> (u16, String) {
    let mut r = ureq::post(&format!("{base_url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .content_type("text/plain")
        .send(body)
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).expect("the body is json")
}

fn envelope(actor: &str, tier: &str, command: Value) -> Value {
    json!({"context": {"actor": actor, "tier": tier}, "command": command})
}

#[tokio::test(flavor = "multi_thread")]
async fn commands_round_trip_over_the_wire() {
    let db = scratch_db("roundtrip");
    let base_url = spawn_server(&db).await;

    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);
    let reply = json_of(&body);
    let records = reply["records"].as_array().expect("records back");
    assert_eq!(records[0]["kind"], "task_created");
    assert_eq!(records[0]["seq"], 0);
    assert_eq!(records[0]["actor"], "human person");
    assert_eq!(records[0]["tier"], "human");
    // the create reply names what was born
    assert_eq!(reply["id"], "t-0");

    // the second write sees the first: the world cache refolded
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"claim_task": {"id": 0, "incarnation": null}}),
        ),
    );
    assert_eq!(status, 200);
    assert_eq!(json_of(&body)["id"], Value::Null);

    // the offer and its accept ride the same shape
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 0}, "kind": {"offer": {"body": "suite green"}}}}),
        ),
    );
    assert_eq!(status, 200);
    let reply = json_of(&body);
    assert_eq!(reply["records"][0]["kind"], "commented");
    assert!(reply["records"][0]["payload"].to_string().contains("offer"));

    let (status, _) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"comment": {"target": {"comment": 2}, "kind": {"accept": {"offer": 2}}}}),
        ),
    );
    assert_eq!(status, 200);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_tier_law_holds_through_http() {
    let db = scratch_db("tierlaw");
    let base_url = spawn_server(&db).await;

    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "implement foo", "parent_id": null}}),
        ),
    );

    // a second offer on the offered task refuses: reject, then offer anew
    post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 0}, "kind": {"offer": {"body": "first account"}}}}),
        ),
    );
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 0}, "kind": {"offer": {"body": "replacement"}}}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_state_transition");

    // human-only acts refuse at agent tier
    for command in [
        json!({"cancel_task": {"id": 0, "explanation": "stop"}}),
        json!({"reopen_task": {"id": 0, "explanation": "resume"}}),
    ] {
        let (status, body) = post_command(&base_url, &envelope("pi", "agent", command));
        assert_eq!(status, 400);
        assert_eq!(json_of(&body)["error"]["code"], "human_only");
    }

    // the self-acceptance gate reads the declared scope through the wire
    let scoped = json!({
        "context": {
            "actor": "pi",
            "tier": "agent",
            "scope": {"task": 0, "incarnation": null, "workspace": null}
        },
        "command": {"comment": {"target": {"comment": 1}, "kind": {"accept": {"offer": 1}}}}
    });
    let (status, body) = post_command(&base_url, &scoped);
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "self_acceptance");

    // system authorship is unrepresentable on the wire
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "saccade",
            "system",
            json!({"claim_task": {"id": 0, "incarnation": null}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    // machinery verbs fail validation under every presentable tier
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"create_workspace": {"task_id": 0, "base_url": "abc123", "branch": "saccade/t-0"}}),
        ),
    );
    assert_eq!(status, 400);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_bodies_get_typed_envelopes() {
    let db = scratch_db("malformed");
    let base_url = spawn_server(&db).await;

    let (status, body) = post_raw(&base_url, "not json at all");
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    // the shape survives exactly once: the line holds the raw bytes
    // with no actor to name, and the code the parse door refused with
    let lines = attempts_lines(&db);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0]["outcome"], "refused");
    assert_eq!(lines[0]["code"], "malformed_request");
    assert_eq!(lines[0]["actor"], Value::Null);
    assert_eq!(lines[0]["request"], "not json at all");

    // empty prose refuses at the boundary, not inside the fold
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "   ", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    // unknown commands are versioned vocabulary, not crashes
    let (status, _) = post_command(
        &base_url,
        &envelope("pi", "agent", json!({"explode_everything": {}})),
    );
    assert_eq!(status, 400);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The client is the CLI's write path: the envelope it builds lands, the
/// records come back, and refusals arrive as their typed failure.
#[tokio::test(flavor = "multi_thread")]
async fn client_send_lands_and_refuses_through_the_wire() {
    use saccade::Command;
    use saccade::client::{self, ClientFail};
    use saccade::store::{Context, Tier};
    use saccade::types::actor::ActorName;

    let db = scratch_db("client");
    let base = spawn_server(&db).await;
    let agent = Context {
        actor: ActorName::new("saccade bot".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };

    let reply = client::send(
        &base,
        &agent,
        Command::CreateTask {
            name: saccade::Prose::new("client smoke".into()).unwrap(),
            parent_id: None,
        },
        Some(7),
    )
    .expect("the server accepts the envelope");
    assert_eq!(reply.records.len(), 1);
    assert_eq!(reply.records[0].kind, "task_created");
    assert_eq!(reply.records[0].seq, 0);
    assert_eq!(reply.records[0].event_time, 7);
    assert_eq!(reply.id.as_deref(), Some("t-0"));

    // a refusal the world can teach carries the teaching as its detail:
    // the birth record used as a comment target names its task
    let fail = client::send(
        &base,
        &agent,
        Command::Comment {
            target: saccade::Target::Comment(saccade::CommentId(saccade::RecordId(0))),
            kind: saccade::CommentKind::Note {
                body: saccade::Prose::new("replying to a birth".into()).unwrap(),
            },
        },
        None,
    )
    .unwrap_err();
    match &fail {
        ClientFail::Refused { code, detail } => {
            assert_eq!(code, "invalid_comment_id");
            assert!(
                detail.contains("c-0 is the birth record of task t-0"),
                "{detail}"
            );
            assert!(detail.contains("address its thread as t-0"), "{detail}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }

    // the extent lesson rides the comment door through the wire too,
    // counted honestly for the one task this tracker holds
    let fail = client::send(
        &base,
        &agent,
        Command::Comment {
            target: saccade::Target::Task(saccade::TaskId(9)),
            kind: saccade::CommentKind::Note {
                body: saccade::Prose::new("talking to nothing".into()).unwrap(),
            },
        },
        None,
    )
    .unwrap_err();
    match &fail {
        ClientFail::Refused { code, detail } => {
            assert_eq!(code, "invalid_task_id");
            assert!(
                detail.contains("no task t-9 exists; this tracker holds 1 task, t-0"),
                "{detail}"
            );
        }
        other => panic!("expected a refusal, got {other:?}"),
    }

    // nothing answering at the url is the typed server_required failure
    let dead = client::send(
        "http://127.0.0.1:1",
        &agent,
        Command::CreateTask {
            name: saccade::Prose::new("never lands".into()).unwrap(),
            parent_id: None,
        },
        None,
    )
    .unwrap_err();
    assert!(matches!(dead, ClientFail::ServerUnreachable { .. }));
    assert_eq!(dead.code(), "server_required");

    // the client names itself: the attempts line carries both versions
    let lines = attempts_lines(&db);
    let landed = &lines[0];
    assert_eq!(landed["client"], env!("CARGO_PKG_VERSION"));
    assert_eq!(landed["server"], env!("CARGO_PKG_VERSION"));
    assert_eq!(landed["actor"], "saccade bot");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

fn attempts_lines(db: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(saccade::paths::attempts_at(db))
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).expect("one json object per line"))
        .collect()
}

/// The reproduction recipe end to end: a refused write's logged line,
/// cloned against its own cursor and replayed, refuses identically —
/// and a landed line is the correlation and nothing else.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_writes_line_replays_against_a_cursor_clone() {
    let db = scratch_db("replay");
    let base_url = spawn_server(&db).await;

    // a landed write names the seq it became; the log has its cursor
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);

    // the refused write: its shape survives only in the attempts line
    let raw = serde_json::to_string(&envelope(
        "pi",
        "agent",
        json!({"claim_task": {"id": 9, "incarnation": null}}),
    ))
    .unwrap();
    let (status, body) = post_raw(&base_url, &raw);
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_task_id");

    let lines = attempts_lines(&db);
    assert_eq!(lines.len(), 2, "{lines:?}");
    let landed = &lines[0];
    let refused = &lines[1];

    // thin on success: the seqs it became, never the payload
    assert_eq!(landed["outcome"], "landed");
    assert_eq!(landed["seqs"], json!([0]));
    assert_eq!(landed.get("request"), None);

    // fat on refusal: the request as received plus the cursor it died against
    assert_eq!(refused["outcome"], "refused");
    assert_eq!(refused["code"], "invalid_task_id");
    assert_eq!(refused["actor"], "pi");
    assert_eq!(refused["request"], raw);
    let cursor = refused["cursor"]
        .as_u64()
        .expect("a cursor against a live log") as usize;
    assert_eq!(cursor, 0);

    // the cursor clone: the log's prefix through the cursor, as a fresh tracker
    let clone_dir = db.parent().unwrap().join("clone");
    std::fs::create_dir_all(&clone_dir).unwrap();
    let clone_db = clone_dir.join("saccade.db");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sac"))
        .arg("--db")
        .arg(&db)
        .arg("clone")
        .arg("--at")
        .arg(cursor.to_string())
        .arg("--out")
        .arg(&clone_db)
        .output()
        .expect("spawn sac clone");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        format!("cloned 1 records through seq 0 to {}", clone_db.display())
    );

    // the replay: the logged request, against the clone, refuses the same
    let clone_base = spawn_server(&clone_db).await;
    let replay = refused["request"]
        .as_str()
        .expect("the request as received");
    let (status, body) = post_raw(&clone_base, replay);
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_task_id");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

// ---- the console: compose through the real HTTP surface ----

use saccade::CommentKind;
use saccade::api::AppState as ConsoleState;
use saccade::objects::comment::{CommentId, Target};
use saccade::store::{Context, RecordId, Tier};
use saccade::types::actor::ActorName;
use saccade::views::task_view;
use saccade::{Command, Prose, TaskId};

async fn spawn_console(db_path: &std::path::Path) -> (String, ConsoleState) {
    let repo = db_path.parent().unwrap().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    sh(&repo, &["init", "--initial-branch=main"]);
    std::fs::write(repo.join("readme"), "the project\n").unwrap();
    sh(&repo, &["add", "."]);
    sh(&repo, &["commit", "-m", "base"]);
    let state = ConsoleState::open(db_path)
        .expect("the scratch tracker opens")
        .with_repo_root(repo);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = saccade::serve::router(state.clone());
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), state)
}

fn sh(dir: &std::path::Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A judged request with no client binary to name and no restatable
/// body: these tests drive the seam directly, not a door.
fn bare_request() -> saccade::attempts::AsReceived {
    saccade::attempts::AsReceived {
        client: None,
        raw: String::new(),
    }
}

fn human_ctx() -> Context {
    Context {
        actor: ActorName::new("jerry".into()).unwrap(),
        tier: Tier::Human,
        scope: None,
    }
}

type FormReply = (u16, String, Option<String>);

/// The form reply plus the Set-Cookie header, for first-claim contracts.
fn post_form_ck(
    url: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (u16, String, Option<String>, Option<String>) {
    let mut r = ureq::post(url)
        .config()
        .max_redirects(0)
        .http_status_as_error(false)
        .build();
    for (k, v) in headers {
        r = r.header(*k, *v);
    }
    let mut r = r.send(body).expect("the loopback server answers");
    let loc = r
        .headers()
        .get("location")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let cookie = r
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    (
        r.status().as_u16(),
        r.body_mut().read_to_string().unwrap(),
        loc,
        cookie,
    )
}

fn world_of(state: &ConsoleState) -> saccade::World {
    match state.snapshot() {
        Ok(s) => s.world,
        Err(_) => panic!("the snapshot refused"),
    }
}

/// The task's thread lines, oldest first.
fn lines_of(world: &saccade::World, n: usize) -> Vec<saccade::views::CommentLine> {
    let mut v: Vec<_> = saccade::views::thread_view(world, TaskId(n))
        .expect("the thread folds")
        .items
        .into_iter()
        .flat_map(|item| match item {
            saccade::views::ThreadItem::Group { root, replies } => {
                let mut all = vec![root];
                all.extend(replies);
                all
            }
            saccade::views::ThreadItem::Note(line) => vec![line],
            saccade::views::ThreadItem::Artifacts(_) => Vec::new(),
        })
        .collect();
    v.sort_by_key(|l| l.seq);
    v
}

fn post_form(url: &str, headers: &[(&str, &str)], body: &str) -> FormReply {
    let mut r = ureq::post(url)
        .config()
        .max_redirects(0)
        .http_status_as_error(false)
        .build();
    for (k, v) in headers {
        r = r.header(*k, *v);
    }
    let mut r = r.send(body).expect("the loopback server answers");
    let loc = r
        .headers()
        .get("location")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    (
        r.status().as_u16(),
        r.body_mut().read_to_string().unwrap(),
        loc,
    )
}

fn get_html(url: &str) -> (u16, String) {
    let mut r = ureq::get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

fn seed_task(state: &ConsoleState, name: &str) {
    state
        .execute(
            &human_ctx(),
            Command::CreateTask {
                name: Prose::new(name.into()).unwrap(),
                parent_id: None,
            },
            None,
            bare_request(),
        )
        .expect("the seed task lands");
}

/// A compose lands a note on the composer's task, anchored by the 303.
#[tokio::test(flavor = "multi_thread")]
async fn compose_lands_a_note_and_routes_addresses() {
    let db = scratch_db("console-note");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    seed_task(&state, "other work");

    // a plain form post takes the 303 to the home thread, anchored
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=kick+off&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-2"));

    let world = world_of(&state);
    let c = &world.comments[&CommentId(RecordId(2))];
    assert!(matches!(c.comment.kind, CommentKind::Note { .. }));
    let lines = lines_of(&world, 0);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].seq, 2);
    assert_eq!(lines[0].depth, 1);
    assert_eq!(lines[0].kind, "note");
    assert_eq!(lines[0].body, "kick off");

    // @c-2 parents: the redirect stays on the reply's home thread
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=saw+it+%40c-2&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-3"));
    // parenting is the thread's shape: depth 2 under the note
    let world = world_of(&state);
    let lines = lines_of(&world, 0);
    assert_eq!(lines[1].seq, 3);
    assert_eq!(lines[1].depth, 2);

    // @t-1 rehomes: the redirect follows the comment to its new task
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=%40t-1+belongs+there&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/1#c-4"));
    let world = world_of(&state);
    assert_eq!(lines_of(&world, 1)[0].seq, 4);
    assert_eq!(lines_of(&world, 0).len(), 2, "it left t-0");

    // bare ids are prose: unaddressed, on the composer's own task
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=see+t-1+and+c-2&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-5"));
    let world = world_of(&state);
    let lines = lines_of(&world, 0);
    let last = lines.last().unwrap();
    assert_eq!(last.seq, 5);
    assert_eq!(last.body, "see t-1 and c-2");
    assert_eq!(last.depth, 1, "bare ids stay on the composer's task");

    // the console renders the focused thread with the comment in it
    let (status, html) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(html.contains("migrate floop"));
    assert!(html.contains("kick off"));
    assert!(html.contains("action=\"/compose\""));
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The fragment contract: a fetch compose gets the thread section back
/// in place of a redirect; a cross-site post is refused outright.
#[tokio::test(flavor = "multi_thread")]
async fn compose_fetch_swaps_the_thread_section() {
    let db = scratch_db("console-fragment");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");

    let (status, body, loc) = post_form(
        &format!("{base}/compose"),
        &[
            ("Sec-Fetch-Site", "same-origin"),
            ("Sec-Fetch-Mode", "cors"),
        ],
        "task=0&body=build+it&who=jerry",
    );
    assert_eq!(status, 200);
    assert!(
        loc.is_none(),
        "a fetch takes the fragment, never a redirect"
    );
    assert!(
        body.starts_with("<section id=\"thread\""),
        "the fragment is the thread section, attributes may follow"
    );
    assert!(body.contains("build it"));
    // the comment landed once
    assert_eq!(world_of(&state).comments.len(), 1);

    // cross-site posts are refused at the door
    let (status, _, _) = post_form(
        &format!("{base}/compose"),
        &[("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "cors")],
        "task=0&body=evil",
    );
    assert_eq!(status, 403);
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The console GET: unfocused at /, focused and composed at /t/N; the
/// strip renders ownership rows for owned tasks.
#[tokio::test(flavor = "multi_thread")]
async fn the_console_renders_forest_and_focused_thread() {
    let db = scratch_db("console-read");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    state
        .execute(
            &human_ctx(),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: None,
            },
            None,
            bare_request(),
        )
        .unwrap();

    let (status, home) = get_html(&format!("{base}/"));
    assert_eq!(status, 200);
    assert!(
        home.contains("migrate floop"),
        "the forest browses live work"
    );
    assert!(home.contains("no task focused"));
    // the ownership row names the owner as a fact
    assert!(home.contains("OWNED · LAST RECORD"), "{home}");
    assert!(home.contains("owned by jerry"), "{home}");

    let (status, focused) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(focused.contains("migrate floop"));
    assert!(
        focused.contains("t-0 · owned by jerry"),
        "the head names the owner"
    );

    // an unknown task is a clean 404
    let (status, _) = get_html(&format!("{base}/t/99"));
    assert_eq!(status, 404);
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A rehoming compose answers a fetch with the same 303 a form post
/// takes: swapping the new task's section into the old page would leave
/// url, ribbon, and composer stale, so the client navigates instead.
#[tokio::test(flavor = "multi_thread")]
async fn compose_fetch_rehome_redirects_instead_of_swapping() {
    let db = scratch_db("console-rehome-fetch");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    seed_task(&state, "other work");

    let (status, body, loc) = post_form(
        &format!("{base}/compose"),
        &[
            ("Sec-Fetch-Site", "same-origin"),
            ("Sec-Fetch-Mode", "cors"),
        ],
        "task=0&body=%40t-1+belongs+there&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/1#c-2"));
    assert!(!body.contains("<section"), "no fragment rides the redirect");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// Every /t/… href the console page emits resolves: the rail is the
/// frame's navigation, so a dead link there is a dead frame. This pins
/// the forest's token-vs-number contract (c-585) across every link
/// form the page carries.
#[tokio::test(flavor = "multi_thread")]
async fn every_task_href_the_console_emits_resolves() {
    let db = scratch_db("console-hrefs");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    seed_task(&state, "other work");
    // a claim feeds the strip's ownership rows, an offer feeds the
    // standing-offer rows and the ribbon
    state
        .execute(
            &human_ctx(),
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: None,
            },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                target: Target::Task(TaskId(1)),
                kind: CommentKind::Offer {
                    body: Prose::new("the standing account".into()).unwrap(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();
    // a comment so the focused thread and its anchors exist
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                kind: CommentKind::Note {
                    body: Prose::new("a note worth keeping".into()).unwrap(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();

    // harvest every /t/… href from the unfocused and focused pages
    let mut hrefs: Vec<String> = Vec::new();
    for path in ["/", "/t/0"] {
        let (_, html) = get_html(&format!("{base}{path}"));
        let mut rest = html.as_str();
        while let Some(i) = rest.find("href=\"/t/") {
            rest = &rest[i + 6..];
            let end = rest.find('"').expect("the href closes");
            let href = &rest[..end];
            hrefs.push(href.to_string());
            rest = &rest[end..];
        }
    }
    hrefs.sort();
    hrefs.dedup();
    // both link forms light up: bare task links (rail, strip) and
    // record anchors (standing-offer rows, ribbon marks)
    assert!(
        hrefs.len() >= 4 && hrefs.iter().any(|h| h.contains("#c-")),
        "the fixture should light up every link form, got {hrefs:?}"
    );
    assert!(
        !hrefs.iter().any(|h| h.starts_with("/t/t-")),
        "no token-in-href may survive: {hrefs:?}"
    );
    for href in &hrefs {
        let path = href.split('#').next().unwrap();
        let (status, _) = get_html(&format!("{base}{path}"));
        assert_eq!(status, 200, "{href} does not resolve");
    }

    // the visible labels stay the token form
    let (_, focused) = get_html(&format!("{base}/t/0"));
    assert!(focused.contains(">t-0<"));
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// Identity is claimed at the act, on every act form: a cookieless
/// browser decides by typing its name into the decision form; the forms
/// carry the field prefilled from the actor cookie when one exists.
#[tokio::test(flavor = "multi_thread")]
async fn decision_forms_carry_the_actor_name() {
    let db = scratch_db("console-identity");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    let pi = Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };
    state
        .execute(
            &pi,
            Command::ClaimTask {
                id: TaskId(0),
                incarnation: None,
            },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &pi,
            Command::Comment {
                target: Target::Task(TaskId(0)),
                kind: CommentKind::Offer {
                    body: Prose::new("suite green".into()).unwrap(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();
    // the focused page renders the decision doors with the who inputs
    let (status, html) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(html.contains("● OFFERED"));
    assert!(html.contains("action=\"/t/0/accept\""));
    assert!(html.contains("action=\"/t/0/reject\""));

    // a cookieless accept without a name refuses, naming the fix
    let (status, body, _) = post_form(&format!("{base}/t/0/accept"), &[], "accept=1");
    assert_eq!(status, 400);
    assert!(body.contains("a name is required to record the act"));

    // a cookieless accept with who records human tier under that name,
    // and the first claim sets the cookie like compose does
    let (status, _, loc, cookie) =
        post_form_ck(&format!("{base}/t/0/accept"), &[], "accept=1&who=jerry");
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0"));
    assert!(
        cookie
            .as_deref()
            .is_some_and(|c| c.starts_with("actor=jerry")),
        "the first claim claims the cookie: {cookie:?}"
    );
    let snap = match state.snapshot() {
        Ok(s) => s,
        Err(_) => panic!("the snapshot refused"),
    };
    let accepted = snap
        .rows
        .iter()
        .rev()
        .find(|r| r.kind == "commented" && r.payload.contains("accept"))
        .expect("the accept landed");
    assert_eq!(accepted.actor, "jerry");
    assert_eq!(accepted.tier, "human");
    assert_eq!(
        task_view(&world_of(&state), TaskId(0)).unwrap().state,
        "concluded"
    );

    // the reject door on a fresh task's standing offer: a producer may
    // reject its own account with grounds
    seed_task(&state, "other work");
    state
        .execute(
            &pi,
            Command::Comment {
                target: Target::Task(TaskId(1)),
                kind: CommentKind::Offer {
                    body: Prose::new("the second account".into()).unwrap(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();
    // grounds are required: the empty form refuses at the console door
    let (status, body, _) = post_form(
        &format!("{base}/t/0/reject"),
        &[],
        "reject=1&who=jerry&grounds=+",
    );
    assert_eq!(status, 400);
    assert!(body.contains("a rejection needs grounds"), "{body}");

    let (status, _, loc, _) = post_form_ck(
        &format!("{base}/t/1/reject"),
        &[],
        "reject=1&who=jerry&grounds=the+edge+case+still+fails",
    );
    assert_eq!(status, 303, "the named rejection lands");
    assert_eq!(loc.as_deref(), Some("/t/1"));
    let snap = match state.snapshot() {
        Ok(s) => s,
        Err(_) => panic!("the snapshot refused"),
    };
    let rejected = snap
        .rows
        .iter()
        .rev()
        .find(|r| r.kind == "commented" && r.payload.contains("reject"))
        .expect("the rejection landed");
    assert_eq!(rejected.actor, "jerry");
    assert!(rejected.payload.contains("the edge case still fails"));
    assert_eq!(
        task_view(&world_of(&state), TaskId(1)).unwrap().state,
        "open"
    );

    // the rendered forms carry the who input, prefilled from the actor
    // cookie when one exists
    state
        .execute(
            &pi,
            Command::Comment {
                target: Target::Task(TaskId(1)),
                kind: CommentKind::Offer {
                    body: Prose::new("the third account".into()).unwrap(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base}/t/1"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("Cookie", "actor=jerry")
        .call()
        .expect("the loopback server answers");
    let with_cookie = r.body_mut().read_to_string().unwrap();
    // every who input the page carries prefills from the cookie
    let who_inputs = with_cookie.matches("name=\"who\"").count();
    assert!(
        who_inputs >= 3,
        "the page carries its act forms: {who_inputs}"
    );
    assert_eq!(
        with_cookie.matches("value=\"jerry\"").count(),
        who_inputs,
        "every act form prefills from the cookie"
    );

    let (status, blank) = get_html(&format!("{base}/t/1"));
    assert_eq!(status, 200);
    assert_eq!(
        blank.matches("name=\"who\"").count(),
        blank.matches("value=\"\"").count(),
        "blank for a fresh browser, never absent"
    );
    // the decision form is the one the CSS themes
    assert!(
        blank.contains("<form class=\"jform\" method=\"post\" action=\"/t/1/accept\">"),
        "the accept form lacks its themed class: {}",
        &blank[..blank.len().min(400)]
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A console refusal through judgment leaves its fat line: the raw form
/// body, the code, and no client binary to name.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_console_decision_leaves_its_line() {
    let db = scratch_db("console-refused");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                kind: CommentKind::Offer {
                    body: Prose::new("the standing account".into()).unwrap(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();
    // the named accept concludes the task
    let (status, _, _, _) = post_form_ck(&format!("{base}/t/0/accept"), &[], "accept=1&who=jerry");
    assert_eq!(status, 303);

    // the second accept on the concluded task refuses
    let (status, body, _) = post_form(&format!("{base}/t/0/accept"), &[], "accept=1&who=jerry");
    assert_eq!(status, 400);
    assert!(body.contains("InvalidStateTransition"), "{body}");
    let lines = attempts_lines(&db);
    let refused = &lines[lines.len() - 1];
    assert_eq!(refused["outcome"], "refused");
    assert_eq!(refused["code"], "invalid_state_transition");
    assert_eq!(refused["actor"], "jerry");
    assert_eq!(refused["client"], Value::Null);
    assert_eq!(refused["request"], "accept=1&who=jerry");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[test]
fn search_reads_the_record_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("search-door");
    let sac = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .arg("--offline")
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

    assert!(sac(&["create", "task", "migrate floop"]).0);
    assert!(sac(&["comment", "t-0", "the floop migration proceeds"]).0);
    assert!(sac(&["create", "task", "floop guard"]).0);
    assert!(sac(&["claim", "t-0"]).0);
    assert!(sac(&["offer", "t-0", "floop landed; suite green"]).0);

    let (ok, out, _) = sac(&["search", "floop"]);
    assert!(ok, "{out}");
    // thread-grouped pointers, record order
    let lines: Vec<&str> = out.lines().collect();
    let t0 = lines
        .iter()
        .position(|l| *l == "t-0  migrate floop")
        .expect("the owning thread groups first");
    assert_eq!(lines[t0 + 1], "  c-0  pi  migrate floop");
    assert_eq!(lines[t0 + 2], "  c-1  pi  the floop migration proceeds");
    assert_eq!(lines[t0 + 3], "  c-4  pi  floop landed; suite green");
    assert!(lines.contains(&"t-1  floop guard"), "{out}");

    // narrowing prints visible counts; the filtered thread stays visible
    let (ok, out, _) = sac(&["search", "floop", "kind:offer"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-0  migrate floop (1 of 3)"), "{out}");
    assert!(out.contains("t-1  floop guard (0 of 1)"), "{out}");

    // a multi-line body rides the json face whole, 'body' its name
    assert!(
        sac(&[
            "comment",
            "t-1",
            "floop guards the door\nthe second line holds the detail"
        ])
        .0
    );
    let (ok, out, _) = sac(&["search", "--json", "guards"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(
        json["groups"][0]["records"][0]["body"],
        "floop guards the door\nthe second line holds the detail"
    );

    // total zero says so in one line and exits clean, terms or facets;
    // the retired '#' term learns the c- form instead of matching words
    let (ok, out, _) = sac(&["search", "c-99999"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (c-99999)");
    let (ok, _, err) = sac(&["search", "#99999"]);
    assert!(!ok, "{err}");
    assert!(
        err.contains("drop the '#': the reference search is c-99999"),
        "{err}"
    );
    let (ok, out, _) = sac(&["search", "by:nobody"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (by:nobody)");

    // the anchor window reads the log's own rows
    let (ok, out, _) = sac(&["search", "c-1", "-C", "1"]);
    assert!(ok, "{out}");
    assert!(out.contains("c-0  task_created  pi/agent"), "{out}");
    assert!(
        out.contains("c-1  commented  pi/agent  the floop migration"),
        "{out}"
    );

    // an empty query refuses at the grammar door, naming the moves
    let (ok, _, err) = sac(&["search"]);
    assert!(!ok);
    assert!(err.contains("give at least one term"), "{err}");
    assert!(err.contains("the moves"), "{err}");
    assert!(err.contains("c-907 -C 3"), "{err}");

    // show opens records in the thread view's body format, either face;
    // the whole body renders, reflowed at the thread's width — while
    // the bare position shows the raw event, header and payload
    let (ok, out, _) = sac(&["show", "c-5"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "c-5  pi [note]\n  floop guards the door the second line holds the detail"
    );
    let (ok, out, _) = sac(&["show", "5"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("c-5  commented  pi\n"),
        "the raw event rides whole: {out}"
    );
    assert!(out.contains("floop guards the door"), "{out}");
    // the retired '#' id refuses, naming the c- form
    let (ok, _, err) = sac(&["show", "#5"]);
    assert!(!ok, "{err}");
    assert!(
        err.contains("drop the '#': the record is addressed as c-5"),
        "{err}"
    );

    // a birth renders as itself: header and relation, never the thread
    // behind it — the relation names t-N, t-N opens the thread
    assert!(sac(&["create", "task", "guard child", "--parent", "t-1"]).0);
    let (ok, out, _) = sac(&["show", "c-6"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "c-6  task  pi\nbirth of t-2 (parent t-1)");
    let (ok, out, _) = sac(&["show", "c-0"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "c-0  task  pi\nbirth of t-0");

    // several ids render each, threads and records in one call
    let (ok, out, _) = sac(&["show", "t-1", "c-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1  open  floop guard"), "{out}");
    assert!(out.contains("c-1  pi"), "{out}");

    // a positional id that parses but names nothing the log holds
    // refuses with the log's own extent, typed or bare
    let (ok, _, err) = sac(&["show", "c-99999"]);
    assert!(!ok);
    assert!(
        err.contains("no record c-99999; the log holds 7 records, 0 through 6"),
        "{err}"
    );
    let (ok, _, err) = sac(&["show", "99999"]);
    assert!(!ok);
    assert!(
        err.contains("no record 99999; the log holds 7 records, 0 through 6"),
        "{err}"
    );
    // the typed door opens any record: an ownership event renders as
    // the log renders it, header and payload
    let (ok, out, _) = sac(&["show", "c-3"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "c-3  task_claimed  pi\n{\"id\":0,\"incarnation\":null}"
    );
    // the typed door opens the offer as its comment block, prose wrapped
    let (ok, out, _) = sac(&["show", "c-4"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "c-4  pi [offer]\n  floop landed; suite green"
    );
    let (ok, _, err) = sac(&["show", "bogus"]);
    assert!(!ok);
    assert!(
        err.contains("'bogus' is not an id (expected c-<seq>, t-<n>, or a bare <seq>)"),
        "{err}"
    );

    // the piped face: one id per line, failures skip with a note
    let pipe = |input: &str, args: &[&str]| {
        use std::io::Write as _;
        use std::process::Stdio;
        let mut child = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .arg("--offline")
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sac");
        child
            .stdin
            .as_mut()
            .expect("stdin pipes")
            .write_all(input.as_bytes())
            .expect("feed stdin");
        let out = child.wait_with_output().expect("wait sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let feed = "c-0\nc-1\nt-0 receipt\nbogus\n";
    let (ok, out, _) = pipe(feed, &["show", "--stdin"]);
    assert!(ok, "{out}");
    assert!(out.contains("skipped 't-0 receipt'"), "{out}");
    assert!(out.contains("skipped 'bogus'"), "{out}");

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn show_and_log_json_read_byte_exact_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("json-read-door");
    let sac = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .arg("--offline")
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

    // the body no honest wrapper survives: quotes, backslashes, a line
    // far past the width, more lines behind it
    let body = format!(
        "first: \"quoted\" and back\\slashed\n{}\nthird line $x^2+1$",
        "word ".repeat(40) + "tail"
    );
    let answer = "the reply rides the thread its root holds";
    assert!(sac(&["create", "task", "byte door"]).0);
    assert!(sac(&["comment", "t-0", &body]).0);
    assert!(sac(&["comment", "c-1", answer]).0);
    assert!(sac(&["comment", "t-0", "a short third note"]).0);
    assert!(sac(&["revise", "c-3", "revised bytes"]).0);

    // the thread's json face: bodies byte-exact, replies folded in
    let (ok, out, _) = sac(&["show", "--json", "t-0"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    let thread = &json[0];
    assert_eq!(thread["id"], json!("t-0"));
    assert_eq!(thread["name"], json!("byte door"));
    let entries = thread["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0]["seq"], json!(1));
    assert_eq!(entries[0]["kind"], json!("note"));
    assert_eq!(entries[0]["body"], json!(body));
    // the entries carry the target edge: roots name the task, replies
    // the comment they address
    assert_eq!(entries[0]["target"], json!({"task": 0}));
    assert_eq!(entries[1]["seq"], json!(2));
    assert_eq!(entries[1]["depth"], json!(2));
    assert_eq!(entries[1]["target"], json!({"comment": 1}));
    assert_eq!(entries[1]["body"], json!(answer));
    // the fold presents the latest revision, byte-exact too
    assert_eq!(entries[2]["seq"], json!(3));
    assert_eq!(entries[2]["body"], json!("revised bytes"));
    assert_eq!(entries[2]["revised"], json!(true));
    assert_eq!(entries[2]["reviser"], json!(null));

    // the record door answers with the same block, alone
    let (ok, out, _) = sac(&["show", "--json", "c-2"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json[0]["body"], json!(answer));
    assert_eq!(json[0]["depth"], json!(2));
    assert_eq!(json[0]["target"], json!({"comment": 1}));

    // the bare position on the json face: the raw log row, no fold
    let (ok, out, _) = sac(&["show", "--json", "2"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json[0]["seq"], json!(2));
    assert_eq!(json[0]["payload"]["target"], json!({"comment": 1}));

    // the log's json face carries folded thread membership: a reply's
    // target is its comment, its thread is still t-0
    let (ok, out, _) = sac(&["log", "--json"]);
    assert!(ok, "{out}");
    let rows: Vec<Value> = serde_json::from_str(&out).expect("the json face parses");
    let kept: Vec<&Value> = rows
        .iter()
        .filter(|r| r["thread"] == json!("t-0"))
        .collect();
    let seqs: Vec<usize> = kept
        .iter()
        .filter_map(|r| r["seq"].as_u64().map(|s| s as usize))
        .collect();
    assert!(seqs.contains(&0), "the birth rides its thread: {seqs:?}");
    assert!(seqs.contains(&2), "the reply rides its thread: {seqs:?}");
    // the revision folds into its comment and claims no thread
    let revised = rows
        .iter()
        .find(|r| r["kind"] == json!("comment_revised"))
        .expect("the revision row");
    assert_eq!(revised["thread"], Value::Null);

    // the pipe's json face: every line's outcome, skips included
    let pipe = |input: &str, args: &[&str]| {
        use std::io::Write as _;
        use std::process::Stdio;
        let mut child = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .arg("--offline")
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sac");
        child
            .stdin
            .as_mut()
            .expect("stdin pipes")
            .write_all(input.as_bytes())
            .expect("feed stdin");
        let out = child.wait_with_output().expect("wait sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let (ok, out, _) = pipe("bogus\nc-1\n", &["show", "--stdin", "--json"]);
    assert!(ok, "{out}");
    let json: Vec<Value> = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json.len(), 2);
    assert_eq!(json[0]["id"], json!("bogus"));
    assert!(json[0]["error"].is_string());
    assert_eq!(json[1]["body"], json!(body));

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn create_reply_names_the_born_task() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("create-reply");

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
        .arg("--offline")
        .env("SACCADE_ACTOR", "assistant")
        .args(["create", "task", "migrate floop"])
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("t-0"));
    assert!(
        lines.next().unwrap().contains("task_created"),
        "the birth record rides under the reference: {stdout}"
    );

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
        .arg("--offline")
        .arg("--json")
        .env("SACCADE_ACTOR", "assistant")
        .args(["claim", "t-0"])
        .output()
        .expect("spawn sac claim");
    let claim: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        claim["records"].is_array(),
        "claim answers the wire's shape: its records under the reply object"
    );

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
        .arg("--offline")
        .arg("--json")
        .env("SACCADE_ACTOR", "assistant")
        .args(["create", "task", "second scratch task"])
        .output()
        .expect("spawn sac create --json");
    assert!(out.status.success());
    let created: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(created["id"], "t-1");
    assert_eq!(created["records"][0]["kind"], "task_created");
}

/// The skill ships with the binary: install deploys the embedded
/// copy — the on-disk source byte for byte, SKILL.md stamped with the
/// binary's version inside its frontmatter — and check verifies the
/// deployed copy through both of its doors.
#[test]
fn the_binary_deploys_and_verifies_its_embedded_skill() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-skill-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let sac = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .current_dir(&dir)
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // absent until deployed, and check says so honestly
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(!ok);
    assert!(out.contains("skill absent"), "{out}");

    // install writes the embedded copy, stamped with the binary version
    let (ok, out, _) = sac(&["skill", "install"]);
    assert!(ok, "{out}");
    assert!(out.contains("skill installed"), "{out}");
    let deployed = std::fs::read_to_string(dir.join(".agents/skills/saccade/SKILL.md"))
        .expect("the skill deployed");
    assert!(deployed.contains(&format!("x-saccade-version: {}", env!("CARGO_PKG_VERSION"))));

    // check now passes: the copy matches the embed beneath the stamp
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(ok, "{out}");
    assert!(out.contains("skill in-sync"), "{out}");

    // drift refuses: a differing copy survives until deleted
    let skill = dir.join(".agents/skills/saccade/SKILL.md");
    let mut drifted = deployed.clone();
    drifted.push_str("\ndrift\n");
    std::fs::write(&skill, drifted).unwrap();
    let (ok, _, err) = sac(&["skill", "install"]);
    assert!(!ok);
    assert_eq!(
        err.trim_end(),
        "error: .agents/skills/saccade differs from this binary's skill — delete the directory and run sac skill install to deploy this version"
    );
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(!ok, "{out}");
    let version = env!("CARGO_PKG_VERSION");
    assert_eq!(
        out.trim_end(),
        format!("skill drifted: .agents/skills/saccade carries {version}, binary is {version}")
    );

    // deletion is the force flag, and a redeploy restores sync
    std::fs::remove_dir_all(dir.join(".agents")).unwrap();
    let (ok, _, _) = sac(&["skill", "install"]);
    assert!(ok);
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(ok);
    assert!(out.contains("skill in-sync"), "{out}");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// The offline reply shape is the wire's: one code path builds both,
/// so a break-glass write and a served write answer identically.
#[tokio::test(flavor = "multi_thread")]
async fn offline_replies_match_the_wire() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let offline_db = scratch_db("offline-shape");
    let served_db = scratch_db("served-shape");

    let offline = |db: &std::path::Path, args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(db)
            .arg("--offline")
            .env("SACCADE_ACTOR", "pi")
            .args(args)
            .output()
            .expect("spawn sac");
        assert!(
            out.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice::<Value>(&out.stdout).expect("json face")
    };

    // the create replies carry the born task's token, both paths
    let created_offline = offline(&offline_db, &["--json", "create", "task", "offline task"]);
    let served = spawn_server(&served_db).await;
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"create_task": {"name": "offline task", "parent_id": null}}}),
    );
    let created_served = json_of(&body);
    assert_eq!(created_offline["id"], created_served["id"]);
    assert_eq!(
        created_offline["records"][0]["kind"],
        created_served["records"][0]["kind"]
    );

    // a comment and a claim answer with their records, both paths
    let noted = offline(&offline_db, &["--json", "comment", "t-0", "the note"]);
    assert_eq!(noted["records"][0]["kind"], "commented");
    let claimed = offline(&offline_db, &["--json", "claim", "t-0"]);
    assert_eq!(claimed["records"][0]["kind"], "task_claimed");

    std::fs::remove_dir_all(offline_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(served_db.parent().unwrap()).unwrap();
}
