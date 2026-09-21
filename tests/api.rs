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
            addressee: None,
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
            addressee: None,
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

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

// ---- the console: compose through the real HTTP surface ----

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
                all.extend(replies);
                all
            }
            saccade::views::ThreadItem::Group { root, replies } => {
                let mut all = vec![root];
                all.extend(replies);
                all
            }
            saccade::views::ThreadItem::Note(line) => vec![line],
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
        CommentState::AddressedToAgent {
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
                addressee: Some(saccade::Addressee::Human),
            },
            None,
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
        .execute(&human_ctx(), Command::ClaimTask { id: TaskId(0) }, None)
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
                addressee: Some(saccade::Addressee::Human),
            },
            None,
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
        )
        .unwrap();
    // a comment so the focused thread and its anchors exist
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                target: Target::Task(TaskId(0)),
                body: Prose::new("a note worth keeping".into()).unwrap(),
                addressee: None,
            },
            None,
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
        .execute(&pi, Command::ClaimTask { id: TaskId(0) }, None)
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::CreateProposal {
                name: Prose::new("hand it back".into()).unwrap(),
                action: saccade::ProposalAction::Release { task_id: TaskId(0) },
            },
            None,
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
        .execute(&pi, Command::ClaimTask { id: TaskId(0) }, None)
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::CreateProposal {
                name: Prose::new("hand it back again".into()).unwrap(),
                action: saccade::ProposalAction::Release { task_id: TaskId(0) },
            },
            None,
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
        .execute(&worker, Command::ClaimTask { id: TaskId(0) }, None)
        .unwrap();
    state
        .execute(
            &worker,
            Command::CompleteTask {
                id: TaskId(0),
                receipt: Prose::new("suite green".into()).unwrap(),
            },
            None,
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
        claim.is_array(),
        "only create replies carry the id; claim stays the bare records"
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
