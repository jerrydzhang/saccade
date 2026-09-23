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
        &envelope("pi", "agent", json!({"claim_task": {"id": 0}})),
    );
    assert_eq!(status, 200);
    assert_eq!(json_of(&body)["id"], Value::Null);

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

    // an unclaimed task cannot be completed: the transition fires first
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"complete_task": {"id": 0, "receipt": "done"}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_state_transition");

    // the holder completes at agent tier; another agent cannot
    post_command(
        &base_url,
        &envelope("pi", "agent", json!({"claim_task": {"id": 0}})),
    );
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"complete_task": {"id": 0, "receipt": "done"}}),
        ),
    );
    assert_eq!(status, 200);

    // human-only acts refuse at agent tier
    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "second", "parent_id": null}}),
        ),
    );
    post_command(
        &base_url,
        &envelope("pi", "agent", json!({"claim_task": {"id": 1}})),
    );
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"drop_task": {"id": 1, "note": "nope"}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "human_only");

    // another agent's claim is not yours to complete
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "other agent",
            "agent",
            json!({"complete_task": {"id": 1, "receipt": "mine"}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "not_claim_holder");

    // system authorship is unrepresentable on the wire
    let (status, body) = post_command(
        &base_url,
        &envelope("saccade", "system", json!({"claim_task": {"id": 0}})),
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

    // a refusal arrives as the typed failure, not a transport error
    let fail = client::send(
        &base,
        &agent,
        Command::AcceptProposal {
            id: saccade::ProposalId(saccade::RecordId(0)),
        },
        None,
    )
    .unwrap_err();
    match &fail {
        ClientFail::Refused { code, .. } => assert_eq!(code, "human_only"),
        other => panic!("expected a refusal, got {other:?}"),
    }

    // a refusal the world can teach carries the teaching as its detail:
    // the birth record used as a comment target names its task
    let fail = client::send(
        &base,
        &agent,
        Command::Comment {
            target: saccade::Target::Comment(saccade::CommentId(saccade::RecordId(0))),
            body: saccade::Prose::new("replying to a birth".into()).unwrap(),
            kind: CommentKind::Note,
        },
        None,
    )
    .unwrap_err();
    match &fail {
        ClientFail::Refused { code, detail } => {
            assert_eq!(code, "invalid_comment_id");
            assert!(
                detail.contains("#0 is the birth record of task t-0"),
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
            body: saccade::Prose::new("talking to nothing".into()).unwrap(),
            kind: CommentKind::Note,
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
    let raw =
        serde_json::to_string(&envelope("pi", "agent", json!({"claim_task": {"id": 9}}))).unwrap();
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
use saccade::objects::comment::{
    AgentAttemptState, CommentId, CommentState, ResponseState, Target,
};
use saccade::store::{Context, RecordId, Tier};
use saccade::types::actor::ActorName;
use saccade::views::task_view;
use saccade::{Command, Prose, TaskId};

async fn spawn_console(db_path: &std::path::Path) -> (String, ConsoleState) {
    let state = ConsoleState::open(db_path).expect("the scratch tracker opens");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = saccade::serve::router(state.clone());
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), state)
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
            saccade::views::ThreadItem::Exchange {
                root,
                run: _,
                replies,
            } => {
                let mut all = vec![root];
                all.extend(replies.into_iter().filter_map(|r| match r {
                    saccade::views::ThreadEntry::Comment(c) => Some(c),
                    saccade::views::ThreadEntry::Artifact(_) => None,
                }));
                all
            }
            saccade::views::ThreadItem::Group { root, replies } => {
                let mut all = vec![root];
                all.extend(replies.into_iter().filter_map(|r| match r {
                    saccade::views::ThreadEntry::Comment(c) => Some(c),
                    saccade::views::ThreadEntry::Artifact(_) => None,
                }));
                all
            }
            saccade::views::ThreadItem::Note(line) => vec![line],
            saccade::views::ThreadItem::Artifact(_) => Vec::new(),
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

/// A compose addressed @agent lands a demand the state table confirms:
/// AddressedToAgent with live authorization at birth.
#[tokio::test(flavor = "multi_thread")]
async fn compose_agent_demand_lands_authorized() {
    let db = scratch_db("console-demand");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");

    // a plain form post takes the 303 to the home thread, anchored
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=%40agent+build+it&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-1"));

    let world = world_of(&state);
    let c = &world.comments[&CommentId(RecordId(1))];
    assert_eq!(
        c.state,
        CommentState::Demand {
            response: ResponseState::Awaiting,
            attempt: AgentAttemptState::Authorized {
                trigger: RecordId(1)
            },
        }
    );
    // the thread carries the stripped body on the composer's task
    let lines = lines_of(&world, 0);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].seq, 1);
    assert_eq!(lines[0].depth, 1);
    assert_eq!(lines[0].body, "build it");

    // the console renders the focused thread with the comment in it
    let (status, html) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(html.contains("migrate floop"));
    assert!(html.contains("build it"));
    assert!(html.contains("action=\"/compose\""));
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// @c-N parents the comment; @t-N rehomes it; bare ids stay prose.
#[tokio::test(flavor = "multi_thread")]
async fn compose_addresses_route_the_comment() {
    let db = scratch_db("console-addresses");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    seed_task(&state, "other work");

    // a first note to reply to
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=kick+off&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-2"));

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
    // rehoming moves the comment to the named task's thread
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
    assert_eq!(last.state, None, "bare ids address no one");
    assert_eq!(last.body, "see t-1 and c-2");
    assert_eq!(last.depth, 1, "bare ids stay on the composer's task");
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
        "task=0&body=%40agent+build+it&who=jerry",
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
/// next panel renders asked-of-you rows for awaiting human demands.
#[tokio::test(flavor = "multi_thread")]
async fn the_console_renders_forest_and_focused_thread() {
    let db = scratch_db("console-read");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    // an agent-authored human demand awaiting an answer
    state
        .execute(
            &Context {
                actor: ActorName::new("pi".into()).unwrap(),
                tier: Tier::Agent,
            },
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("need a ruling on floop".into()).unwrap(),
                kind: saccade::CommentKind::Ask,
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
    // the asked-of-you row names its asking comment and its author
    assert!(
        home.contains("ASKED OF YOU") && home.contains("need a ruling on floop"),
        "the row renders the asking comment"
    );
    assert!(home.contains(">pi · t-0<"));
    assert!(home.contains("nothing claimed"));

    let (status, focused) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(focused.contains("migrate floop"));
    assert!(focused.contains("need a ruling on floop"));
    assert!(
        home.contains("/t/0#c-1"),
        "asked-of-you rows click through to the record"
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
    // a claim feeds the strip's candidates, an agent demand awaiting a
    // human feeds asked-of-you, a proposal feeds the rail's gate
    state
        .execute(
            &human_ctx(),
            Command::ClaimTask { id: TaskId(0) },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &Context {
                actor: ActorName::new("pi".into()).unwrap(),
                tier: Tier::Agent,
            },
            Command::Comment {
                target: Target::Task(TaskId(1)),
                body: Prose::new("need a ruling on floop".into()).unwrap(),
                kind: saccade::CommentKind::Ask,
            },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::CreateProposal {
                name: Prose::new("void the stray".into()).unwrap(),
                action: saccade::ProposalAction::Drop { task_id: TaskId(1) },
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
                body: Prose::new("a note worth keeping".into()).unwrap(),
                kind: CommentKind::Note,
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
    // both link forms light up: bare task links (rail, strip, gate) and
    // record anchors (asked-of-you rows, ribbon marks)
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
/// browser rules by typing its name into the judgment form; the forms
/// carry the field prefilled from the actor cookie when one exists.
#[tokio::test(flavor = "multi_thread")]
async fn judgment_forms_carry_the_actor_name() {
    let db = scratch_db("console-identity");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    let pi = Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
    };
    state
        .execute(
            &pi,
            Command::ClaimTask { id: TaskId(0) },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::CreateProposal {
                name: Prose::new("hand it back".into()).unwrap(),
                action: saccade::ProposalAction::Release { task_id: TaskId(0) },
            },
            None,
            bare_request(),
        )
        .unwrap();
    let proposal = saccade::views::open_proposals(&world_of(&state))
        .first()
        .expect("the gate holds one proposal")
        .id;

    // a cookieless ruling without a name refuses, naming the fix
    let (status, body, _) = post_form(&format!("{base}/p/{proposal}/ruling"), &[], "ruling=accept");
    assert_eq!(status, 400);
    assert!(body.contains("a name is required to record the act"));

    // a cookieless accept ruling with who records human tier under that
    // name, and the first claim sets the cookie like compose does
    let (status, _, loc, cookie) = post_form_ck(
        &format!("{base}/p/{proposal}/ruling"),
        &[],
        "ruling=accept&who=jerry",
    );
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
        .find(|r| r.kind == "proposal_accepted")
        .expect("the ruling landed");
    assert_eq!(accepted.actor, "jerry");
    assert_eq!(accepted.tier, "human");
    // the embedded act executed: the claim is released
    let task = saccade::views::task_view(&world_of(&state), TaskId(0)).unwrap();
    assert_eq!(task.state, "open");

    // a cookieless reject ruling records the same way — both rulings
    // land at human tier under the typed name
    state
        .execute(
            &human_ctx(),
            Command::CreateProposal {
                name: Prose::new("void it".into()).unwrap(),
                action: saccade::ProposalAction::Drop { task_id: TaskId(0) },
            },
            None,
            bare_request(),
        )
        .unwrap();
    let proposal = saccade::views::open_proposals(&world_of(&state))
        .first()
        .expect("the drop proposal is open")
        .id;
    let (status, _, _, _) = post_form_ck(
        &format!("{base}/p/{proposal}/ruling"),
        &[],
        "ruling=reject&who=jerry&note=not+yet",
    );
    assert_eq!(status, 303);
    let snap = match state.snapshot() {
        Ok(s) => s,
        Err(_) => panic!("the snapshot refused"),
    };
    let rejected = snap
        .rows
        .iter()
        .rev()
        .find(|r| r.kind == "proposal_rejected")
        .expect("the rejection landed");
    assert_eq!(rejected.actor, "jerry");
    assert_eq!(rejected.tier, "human");
    assert_eq!(
        task_view(&world_of(&state), TaskId(0)).unwrap().state,
        "open"
    );

    // the rendered judgment form carries the one who input, prefilled
    // from the actor cookie when one exists, themed by its class
    state
        .execute(
            &pi,
            Command::ClaimTask { id: TaskId(0) },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::CreateProposal {
                name: Prose::new("hand it back again".into()).unwrap(),
                action: saccade::ProposalAction::Release { task_id: TaskId(0) },
            },
            None,
            bare_request(),
        )
        .unwrap();

    let mut r = ureq::get(&format!("{base}/t/0"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("Cookie", "actor=jerry")
        .call()
        .expect("the loopback server answers");
    let with_cookie = r.body_mut().read_to_string().unwrap();
    // the judgment block and the compose dock prefill from the cookie
    assert_eq!(
        with_cookie.matches("value=\"jerry\"").count(),
        2,
        "every act form prefills from the cookie"
    );

    let (status, blank) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(
        blank.matches("value=\"\"").count() >= 2,
        "blank for a fresh browser, never absent"
    );
    // the ruling form is the one the CSS themes
    assert!(
        blank.contains("<form class=\"jform\" method=\"post\" action=\"/p/"),
        "the ruling form lacks its themed class: {}",
        &blank[..blank.len().min(400)]
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The accept door: a delivered task's form carries the actor name,
/// posts at human tier under it, and the deposit becomes done.
#[tokio::test(flavor = "multi_thread")]
async fn the_accept_door_carries_the_actor_name() {
    let db = scratch_db("console-accept");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    // jerry births t-0; the run's derived attribution claims and delivers
    let worker = Context {
        actor: ActorName::new("pi/t-0-1".into()).unwrap(),
        tier: Tier::Agent,
    };
    state
        .execute(
            &worker,
            Command::ClaimTask { id: TaskId(0) },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &worker,
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("suite green".into()).unwrap(),
            },
            None,
            bare_request(),
        )
        .unwrap();

    // the focused page renders the accept form with the who input
    let (status, html) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(html.contains("● DELIVERED"));
    assert!(html.contains("action=\"/t/0/accept\""));
    assert!(html.contains("name=\"who\""));

    // a cookieless accept without a name refuses, naming the fix
    let (status, body, _) = post_form(&format!("{base}/t/0/accept"), &[], "accept=1");
    assert_eq!(status, 400);
    assert!(body.contains("a name is required to record the act"));

    // the named accept lands at human tier and the door opens to done
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

    // a console refusal through judgment leaves its fat line too: the
    // raw form body, the code, and no client binary to name
    let (status, body, _) = post_form(&format!("{base}/t/0/accept"), &[], "accept=1&who=jerry");
    assert_eq!(status, 400);
    assert!(body.contains("InvalidStateTransition"));
    let lines = attempts_lines(&db);
    let refused = &lines[lines.len() - 1];
    assert_eq!(refused["outcome"], "refused");
    assert_eq!(refused["code"], "invalid_state_transition");
    assert_eq!(refused["actor"], "jerry");
    assert_eq!(refused["client"], Value::Null);
    assert_eq!(refused["request"], "accept=1&who=jerry");
    let snap = match state.snapshot() {
        Ok(s) => s,
        Err(_) => panic!("the snapshot refused"),
    };
    let accepted = snap
        .rows
        .iter()
        .rev()
        .find(|r| r.kind == "task_accepted")
        .expect("the accept landed");
    assert_eq!(accepted.actor, "jerry");
    assert_eq!(accepted.tier, "human");
    assert_eq!(
        task_view(&world_of(&state), TaskId(0)).unwrap().state,
        "done"
    );
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
    assert!(sac(&["done", "t-0", "--receipt", "floop landed; suite green"]).0);

    let (ok, out, _) = sac(&["search", "floop"]);
    assert!(ok, "{out}");
    // thread-grouped pointers, record order, the receipt closing its group
    let lines: Vec<&str> = out.lines().collect();
    let t0 = lines
        .iter()
        .position(|l| *l == "t-0  migrate floop")
        .expect("the owning thread groups first");
    assert_eq!(lines[t0 + 1], "  #0  pi  migrate floop");
    assert_eq!(lines[t0 + 2], "  #1  pi  the floop migration proceeds");
    assert_eq!(lines[t0 + 3], "  t-0 receipt  floop landed; suite green");
    assert!(lines.contains(&"t-1  floop guard"), "{out}");

    // narrowing prints visible counts; the filtered thread stays visible
    let (ok, out, _) = sac(&["search", "floop", "kind:receipt"]);
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

    // total zero says so in one line and exits clean, terms or facets
    let (ok, out, _) = sac(&["search", "#99999"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (#99999)");
    let (ok, out, _) = sac(&["search", "by:nobody"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (by:nobody)");

    // the anchor window reads the log's own rows
    let (ok, out, _) = sac(&["search", "#1", "-C", "1"]);
    assert!(ok, "{out}");
    assert!(out.contains("#0  task_created  pi/agent"), "{out}");
    assert!(
        out.contains("#1  commented  pi/agent  the floop migration"),
        "{out}"
    );

    // an empty query refuses at the grammar door, naming the moves
    let (ok, _, err) = sac(&["search"]);
    assert!(!ok);
    assert!(err.contains("give at least one term"), "{err}");
    assert!(err.contains("the moves"), "{err}");
    assert!(err.contains("'#907' -C 3"), "{err}");

    // show opens records in the thread view's body format, either face;
    // the whole body renders, reflowed at the thread's width
    let (ok, out, _) = sac(&["show", "#5"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "#5  pi\n  floop guards the door the second line holds the detail"
    );
    let (ok, out, _) = sac(&["show", "c-5"]);
    assert!(ok, "{out}");
    assert!(out.starts_with("#5  pi"), "{out}");

    // a birth renders as itself: header and relation, never the thread
    // behind it — the relation names t-N, t-N opens the thread
    assert!(sac(&["create", "task", "guard child", "--parent", "t-1"]).0);
    let (ok, out, _) = sac(&["show", "#6"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "#6  task  pi\nbirth of t-2 (parent t-1)");
    let (ok, out, _) = sac(&["show", "#0"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "#0  task  pi\nbirth of t-0");

    // several ids render each, threads and records in one call
    let (ok, out, _) = sac(&["show", "t-1", "#1"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1  open  floop guard"), "{out}");
    assert!(out.contains("#1  pi"), "{out}");

    // a positional id that parses but names nothing the log holds
    // refuses with the log's own extent
    let (ok, _, err) = sac(&["show", "#99999"]);
    assert!(!ok);
    assert!(
        err.contains("no record #99999; the log holds 7 records, #0 through #6"),
        "{err}"
    );
    // the raw-id door opens any record: a machinery event renders as
    // the log renders it, header and payload
    let (ok, out, _) = sac(&["show", "#3"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "#3  task_claimed  pi\n{\"id\":0}");
    let (ok, out, _) = sac(&["show", "#4"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("#4  task_delivered  pi\n{\"id\":0,"),
        "the accept-family payload rides whole: {out}"
    );
    let (ok, _, err) = sac(&["show", "bogus"]);
    assert!(!ok);
    assert!(
        err.contains("'bogus' is not an id (expected t-<n>, #<seq>, or c-<seq>)"),
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

    let (ok, _, err) = pipe("", &["show", "t-0", "--stdin"]);
    assert!(!ok);
    assert!(
        err.contains("ids as arguments or --stdin, not both"),
        "{err}"
    );

    // empty input is silence, exit 0
    let (ok, out, _) = pipe("", &["show", "--stdin"]);
    assert!(ok, "{out}");
    assert_eq!(out, "");

    // blank lines skip silently; bad ids skip with a note; good ids render
    let (ok, out, _) = pipe("\n\nbogus\nt-0 receipt\n#1\n", &["show", "--stdin"]);
    assert!(ok, "{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert!(
        lines.contains(
            &"skipped 'bogus': 'bogus' is not an id (expected t-<n>, #<seq>, or c-<seq>)"
        ),
        "{out}"
    );
    // a receipt's pointer is not an id; the pipe says so and moves on
    assert!(
        out.contains("skipped 't-0 receipt': 't-0 receipt' is not a task id"),
        "{out}"
    );
    assert!(lines.contains(&"#1  pi"), "{out}");
    assert!(lines.contains(&"  the floop migration proceeds"), "{out}");

    // the pipe chain runs end to end: json pointers feed show
    let (ok, out, _) = sac(&["search", "--json", "floop"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    let feed = json["groups"]
        .as_array()
        .expect("groups")
        .iter()
        .flat_map(|g| g["records"].as_array().expect("records"))
        .filter_map(|r| r["pointer"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let (ok, out, _) = pipe(&format!("{feed}\n"), &["show", "--stdin"]);
    assert!(ok, "{out}");
    let lines: Vec<&str> = out.lines().collect();
    // birth pointers render their event; the thread stays behind t-N
    assert!(lines.contains(&"#0  task  pi"), "{out}");
    assert!(lines.contains(&"birth of t-0"), "{out}");
    assert!(lines.contains(&"#1  pi"), "{out}");
    assert!(out.contains("skipped 't-0 receipt'"), "{out}");
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
    let home = dir.join(".agents/skills/saccade");
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".agents/skills/saccade");
    let version = env!("CARGO_PKG_VERSION");
    let in_sync =
        format!("skill in-sync: .agents/skills/saccade carries {version}, binary is {version}");

    // the door carries no db: install lands in a bare directory
    let (ok, out, err) = sac(&["skill", "install"]);
    assert!(ok, "{out}{err}");

    // the deployed files are the on-disk source byte for byte, SKILL.md
    // stamped with the binary's version ahead of its closing fence
    let src_skill = std::fs::read_to_string(source.join("SKILL.md")).unwrap();
    let (front, rest) = src_skill.split_once("\n---\n").unwrap();
    assert_eq!(
        std::fs::read_to_string(home.join("SKILL.md")).unwrap(),
        format!("{front}\nx-saccade-version: {version}\n---\n{rest}")
    );
    assert_eq!(
        std::fs::read(home.join("errors.md")).unwrap(),
        std::fs::read(source.join("errors.md")).unwrap()
    );
    assert_eq!(
        std::fs::read(home.join("diagnosing.md")).unwrap(),
        std::fs::read(source.join("diagnosing.md")).unwrap()
    );

    // check names both versions, and the second install is a no-op
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), in_sync);
    let (ok, out, _) = sac(&["skill", "install"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), in_sync);
    let (ok, out, _) = sac(&["skill", "check", "--json"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        json,
        json!({"verdict": "in-sync", "carries": version, "binary": version})
    );

    // a touched file: install refuses naming the delete path, check drifts
    let mut touched = std::fs::read_to_string(home.join("errors.md")).unwrap();
    touched.push_str("a local edit\n");
    std::fs::write(home.join("errors.md"), touched).unwrap();
    let (ok, _, err) = sac(&["skill", "install"]);
    assert!(!ok);
    assert_eq!(
        err.trim_end(),
        "error: .agents/skills/saccade differs from this binary's skill — delete the directory and run sac skill install to deploy this version"
    );
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(!ok, "{out}");
    assert_eq!(
        out.trim_end(),
        format!("skill drifted: .agents/skills/saccade carries {version}, binary is {version}")
    );

    // no copy at all is absent, named as absent
    std::fs::remove_dir_all(&home).unwrap();
    let (ok, out, _) = sac(&["skill", "check"]);
    assert!(!ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "skill absent: no .agents/skills/saccade in this directory — sac skill install deploys this binary's copy"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

/// One CLI verb run offline against a scratch db, --json on, agent tier.
fn sac_offline(db: &std::path::Path, args: &[&str]) -> String {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sac"))
        .arg("--db")
        .arg(db)
        .arg("--offline")
        .arg("--json")
        .env("SACCADE_ACTOR", "pi")
        .args(args)
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A reply's canonical bytes with the clock zeroed, so the two faces of
/// one command compare equal.
fn reply_shape(text: &str) -> String {
    let mut reply: Value = serde_json::from_str(text).expect("the reply is json");
    for record in reply["records"]
        .as_array_mut()
        .expect("every reply carries its records")
    {
        record["event_time"] = json!(0);
        record["logged_time"] = json!(0);
    }
    serde_json::to_string(&reply).expect("the reply is plain data")
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_create_reply_matches_the_wire() {
    let wire_db = scratch_db("face-create-wire");
    let base_url = spawn_server(&wire_db).await;
    let (status, wire_body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);

    let offline_db = scratch_db("face-create-offline");
    let offline_body = sac_offline(&offline_db, &["create", "task", "migrate floop"]);

    assert_eq!(reply_shape(&offline_body), reply_shape(&wire_body));
    // the born task's token rides both faces
    assert_eq!(json_of(&wire_body)["id"], "t-0");
    assert_eq!(json_of(&offline_body)["id"], "t-0");

    std::fs::remove_dir_all(wire_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(offline_db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_comment_reply_matches_the_wire() {
    let wire_db = scratch_db("face-comment-wire");
    let base_url = spawn_server(&wire_db).await;
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);
    let (status, wire_body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 0}, "body": "implement foo", "kind": "note"}}),
        ),
    );
    assert_eq!(status, 200);

    let offline_db = scratch_db("face-comment-offline");
    sac_offline(&offline_db, &["create", "task", "migrate floop"]);
    let offline_body = sac_offline(&offline_db, &["comment", "t-0", "implement foo"]);

    assert_eq!(reply_shape(&offline_body), reply_shape(&wire_body));
    // only create names a birth
    assert_eq!(json_of(&wire_body)["id"], Value::Null);
    assert_eq!(json_of(&offline_body)["id"], Value::Null);

    std::fs::remove_dir_all(wire_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(offline_db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_claim_reply_matches_the_wire() {
    let wire_db = scratch_db("face-claim-wire");
    let base_url = spawn_server(&wire_db).await;
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);
    let (status, wire_body) = post_command(
        &base_url,
        &envelope("pi", "agent", json!({"claim_task": {"id": 0}})),
    );
    assert_eq!(status, 200);

    let offline_db = scratch_db("face-claim-offline");
    sac_offline(&offline_db, &["create", "task", "migrate floop"]);
    let offline_body = sac_offline(&offline_db, &["claim", "t-0"]);

    assert_eq!(reply_shape(&offline_body), reply_shape(&wire_body));
    assert_eq!(json_of(&wire_body)["id"], Value::Null);
    assert_eq!(json_of(&offline_body)["id"], Value::Null);

    std::fs::remove_dir_all(wire_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(offline_db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_done_reply_matches_the_wire() {
    let wire_db = scratch_db("face-done-wire");
    let base_url = spawn_server(&wire_db).await;
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);
    let (status, _) = post_command(
        &base_url,
        &envelope("pi", "agent", json!({"claim_task": {"id": 0}})),
    );
    assert_eq!(status, 200);
    let (status, wire_body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"complete_task": {"id": 0, "receipt": "the work landed"}}),
        ),
    );
    assert_eq!(status, 200);

    let offline_db = scratch_db("face-done-offline");
    sac_offline(&offline_db, &["create", "task", "migrate floop"]);
    sac_offline(&offline_db, &["claim", "t-0"]);
    let offline_body = sac_offline(
        &offline_db,
        &["done", "t-0", "--receipt", "the work landed"],
    );

    assert_eq!(reply_shape(&offline_body), reply_shape(&wire_body));
    assert_eq!(json_of(&wire_body)["id"], Value::Null);
    assert_eq!(json_of(&offline_body)["id"], Value::Null);

    std::fs::remove_dir_all(wire_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(offline_db.parent().unwrap()).unwrap();
}

// ---- the tagged union: wire shapes and the ask door ----

use saccade::objects::comment::SteerDelivery;
use saccade::views::asked_of_you;

/// The comment door carries the tagged union: each kind lands its fold
/// state, and the stored payload round-trips with its kind tag.
#[tokio::test(flavor = "multi_thread")]
async fn the_comment_kinds_land_their_states_through_the_wire() {
    let db = scratch_db("kinds");
    let base_url = spawn_server(&db).await;
    for name in ["migrate floop", "second work"] {
        post_command(
            &base_url,
            &envelope(
                "human person",
                "human",
                json!({"create_task": {"name": name, "parent_id": null}}),
            ),
        );
    }

    // a demand: authorized on its birth record, reopening done work
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 0}, "body": "run the migration", "kind": "demand"}}),
        ),
    );
    assert_eq!(status, 200);
    let records = json_of(&body)["records"].as_array().unwrap().clone();
    assert_eq!(records[0]["kind"], "commented");
    assert!(records[0]["payload"].to_string().contains("\"demand\""));

    // a steer: standing intent on the thread
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 1}, "body": "also cover the offline path", "kind": "steer"}}),
        ),
    );
    assert_eq!(status, 200);

    let state = AppState::open(&db).unwrap();
    let snapshot = state.snapshot().unwrap();
    let demand = &snapshot.world.comments[&saccade::CommentId(saccade::RecordId(2))];
    assert!(matches!(
        &demand.state,
        CommentState::Demand {
            attempt: saccade::objects::comment::AgentAttemptState::Authorized { .. },
            ..
        }
    ));
    let steer = &snapshot.world.comments[&saccade::CommentId(saccade::RecordId(3))];
    assert_eq!(
        steer.state,
        CommentState::Steer {
            delivery: SteerDelivery::Standing
        }
    );

    // a note carries no machinery: the default kind is note
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"target": {"task": 1}, "body": "for the record"}}),
        ),
    );
    assert_eq!(status, 400, "the kind field is the wire's vocabulary now");
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The ask door: the variant rides the existing comment door at agent
/// tier (the extension's path), waits for its answer, and the first
/// reply answers it — the shape the runner's extension speaks.
#[tokio::test(flavor = "multi_thread")]
async fn an_ask_round_trips_through_the_comment_door() {
    let db = scratch_db("ask-door");
    let base_url = spawn_server(&db).await;
    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "migrate floop", "parent_id": null}}),
        ),
    );

    // the extension's write: agent tier under the run's actor, kind ask
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi/t-0-1",
            "agent",
            json!({"comment": {"target": {"task": 0}, "body": "break-glass or the copy?", "kind": "ask"}}),
        ),
    );
    assert_eq!(status, 200);
    let seq = json_of(&body)["records"][0]["seq"].as_u64().unwrap();

    let state = AppState::open(&db).unwrap();
    let ask = &state.snapshot().unwrap().world.comments
        [&saccade::CommentId(saccade::RecordId(seq as usize))];
    assert_eq!(
        ask.state,
        CommentState::Ask {
            response: saccade::objects::comment::ResponseState::Awaiting,
        }
    );
    // the residual inbox carries the unanswered ask
    let asked = asked_of_you(&state.snapshot().unwrap().world);
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].comment, seq as usize);
    assert_eq!(asked[0].actor, "pi/t-0-1");
    assert_eq!(asked[0].body, "break-glass or the copy?");

    // the human answers: the first reply responds the ask
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"comment": {"target": {"comment": seq}, "body": "break-glass; the copy is dead", "kind": "note"}}),
        ),
    );
    assert_eq!(status, 200);

    // the answer landed through the server's own state, so a fresh
    // read of the db sees the responded ask
    let conn = saccade::db::open_read(&db).unwrap();
    let loadout = saccade::db::load(&conn).unwrap();
    let saccade::db::LoadState::Full(world) = loadout.state else {
        panic!("expected a full load");
    };
    let ask = &world.comments[&saccade::CommentId(saccade::RecordId(seq as usize))];
    assert!(matches!(
        &ask.state,
        CommentState::Ask {
            response: saccade::objects::comment::ResponseState::Responded { .. },
        }
    ));
    // answered: the residual inbox empties
    assert!(asked_of_you(&world).is_empty());

    // the wait release on the ask carries the answer body — the bytes
    // the extension returns as its tool result
    let seen = saccade::runner::wait(
        &db,
        saccade::CommentId(saccade::RecordId(seq as usize)),
        Some(5),
    )
    .unwrap();
    assert!(
        seen.contains(&format!("c-{seq}: answered by c-{}", seq as usize + 1)),
        "{seen}"
    );
    assert!(seen.contains("(human person)"), "{seen}");
    assert!(seen.contains("break-glass; the copy is dead"), "{seen}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

// ---- artifacts: the wire, the bytes door, the console render ----

/// A console whose state also serves an artifact store.
async fn spawn_console_with_artifacts(
    db_path: &std::path::Path,
    dir: &std::path::Path,
) -> (String, ConsoleState) {
    let state = ConsoleState::open(db_path)
        .expect("the scratch tracker opens")
        .with_artifacts(dir.to_path_buf());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = saccade::serve::router(state.clone());
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), state)
}

const PNG_BYTES: &[u8] = b"\x89PNG\r\n\x1a\nfigure bytes";

/// The artifact contract over HTTP: the wire carries the pointer only,
/// the door serves the store's bytes by sniffed content type, and a
/// miss or a malformed token names itself.
#[tokio::test(flavor = "multi_thread")]
async fn artifacts_park_over_the_wire_and_the_door_serves_bytes() {
    let db = scratch_db("artifact-wire");
    let dir = db.parent().unwrap().join("artifacts");
    std::fs::create_dir_all(&dir).unwrap();
    let hash = saccade::ContentHash::of(PNG_BYTES);
    std::fs::write(dir.join(hash.as_str()), PNG_BYTES).unwrap();
    let (base, _state) = spawn_console_with_artifacts(&db, &dir).await;

    let (status, _) = post_command(
        &base,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "hold the figures", "parent_id": null}}),
        ),
    );
    assert_eq!(status, 200);

    // the wire carries {name, hash} only — the bytes stay client-side
    let (status, body) = post_command(
        &base,
        &envelope(
            "pi",
            "agent",
            json!({"artifact": {"root": 0, "artifact": {"name": "sweep figure", "hash": hash.as_str()}}}),
        ),
    );
    assert_eq!(status, 200);
    let record = &json_of(&body)["records"][0];
    assert_eq!(record["kind"], "artifact_added");
    assert_eq!(record["payload"]["artifact"]["name"], "sweep figure");
    assert_eq!(record["payload"]["artifact"]["hash"], hash.as_str());
    assert_eq!(record["payload"]["root"], 0);
    let mut keys: Vec<&str> = record["payload"]["artifact"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["hash", "name"],
        "the pointer is the artifact's whole shape"
    );

    // the db holds no bytes: the figure never touched the columns
    let stored = std::fs::read(&db).unwrap();
    assert!(
        !stored.windows(PNG_BYTES.len()).any(|w| w == PNG_BYTES),
        "the artifact bytes rode the db"
    );

    // the door serves the store's bytes with their sniffed type
    let mut r = ureq::get(&format!("{base}/a/{}", hash.as_str()))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the loopback server answers");
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(r.headers().get("content-type").unwrap(), "image/png");
    assert_eq!(r.body_mut().read_to_vec().unwrap(), PNG_BYTES);

    // other content serves as itself: octet-stream, never an image type
    let plain = b"not a figure, just words";
    let plain_hash = saccade::ContentHash::of(plain);
    std::fs::write(dir.join(plain_hash.as_str()), plain).unwrap();
    let r = ureq::get(&format!("{base}/a/{}", plain_hash.as_str()))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the loopback server answers");
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(
        r.headers().get("content-type").unwrap(),
        "application/octet-stream"
    );

    // a store miss is a 404 that names itself
    let missing = saccade::ContentHash::of(b"bytes never parked");
    let mut r = ureq::get(&format!("{base}/a/{}", missing.as_str()))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the loopback server answers");
    assert_eq!(r.status().as_u16(), 404);
    assert!(
        r.body_mut()
            .read_to_string()
            .unwrap()
            .contains("unavailable")
    );

    // a malformed token never names a path
    let mut r = ureq::get(&format!("{base}/a/../../etc/passwd"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the loopback server answers");
    assert_eq!(r.status().as_u16(), 404);
    assert!(!r.body_mut().read_to_string().unwrap().contains("root"));

    // a non-hex hash refuses at the boundary, writing nothing
    let (status, body) = post_command(
        &base,
        &envelope(
            "pi",
            "agent",
            json!({"artifact": {"root": 0, "artifact": {"name": "bad hash", "hash": "ABC"}}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The console renders the artifact at its home position and resolves
/// a later '#N' mention into the card linking home.
#[tokio::test(flavor = "multi_thread")]
async fn the_console_renders_artifacts_and_their_mentions() {
    let db = scratch_db("artifact-console");
    let dir = db.parent().unwrap().join("artifacts");
    std::fs::create_dir_all(&dir).unwrap();
    let hash = saccade::ContentHash::of(PNG_BYTES);
    std::fs::write(dir.join(hash.as_str()), PNG_BYTES).unwrap();
    let (base, state) = spawn_console_with_artifacts(&db, &dir).await;

    seed_task(&state, "hold the figures");
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("the verdict, figure below".into()).unwrap(),
                kind: CommentKind::Note,
            },
            None,
            bare_request(),
        )
        .unwrap();
    let parked = state
        .execute(
            &human_ctx(),
            Command::Artifact {
                root: TaskId(0),
                artifact: saccade::types::artifact::Artifact {
                    name: Prose::new("sweep figure".into()).unwrap(),
                    hash: hash.clone(),
                },
            },
            None,
            bare_request(),
        )
        .unwrap();
    let seq = parked[0].seq;
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new(format!("citing #{seq} from the verdict")).unwrap(),
                kind: CommentKind::Note,
            },
            None,
            bare_request(),
        )
        .unwrap();

    let (status, page) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    // the figure renders at its position: caption, inline image, anchor
    assert!(page.contains(&format!("id=\"a-{seq}\"")), "{page}");
    assert!(
        page.contains(&format!(
            "<img class=\"aimg\" src=\"/a/{}\" alt=\"sweep figure\">",
            hash.as_str()
        )),
        "{page}"
    );
    assert!(page.contains("ARTIFACT"), "{page}");
    // the verdict sits above the figure it names
    let verdict = page.find("the verdict, figure below").unwrap();
    let figure = page.find("alt=\"sweep figure\"").unwrap();
    assert!(verdict < figure);
    // the later mention renders as the card linking home
    assert!(
        page.contains(&format!("citing <a class=\"acard\" href=\"/t/0#a-{seq}\">")),
        "{page}"
    );

    // a store emptied under the record renders the honest miss
    std::fs::remove_file(dir.join(hash.as_str())).unwrap();
    let (status, page) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(page.contains("sweep figure · unavailable"), "{page}");
    assert!(!page.contains("class=\"aimg\""), "{page}");
    // and the mention card carries the same verdict
    assert!(
        page.contains("sweep figure · unavailable</span></a>"),
        "{page}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}
