//! The /api/v1 surface over a real loopback socket: one envelope, records
//! back or a typed error, the tier law holding through HTTP exactly as it
//! holds in process.

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use saccade::api::AppState;

fn scratch_db(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sac-api-{tag}-{}", std::process::id()));
    // a killed run may have left this pid's dir behind; every test
    // here builds its tracker from nothing
    let _ = std::fs::remove_dir_all(&dir);
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
            json!({"create_task": {"name": "migrate floop"}}),
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
            json!({"comment": {"home": 0, "kind": {"offer": {"body": "suite green"}}, "references": []}}),
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
            json!({"comment": {"home": 0, "kind": {"accept": {"offer": 2}}, "references": []}}),
        ),
    );
    assert_eq!(status, 200);

    // the authoring ingress rides the wire: a marker the mapping
    // cannot resolve refuses with its taught reason
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"home": 0, "kind": {"note": {"body": "dangling [^1]"}}, "references": []}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_reference");

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
            json!({"create_task": {"name": "implement foo"}}),
        ),
    );

    // a second offer on the offered task refuses: reject, then offer anew
    post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"home": 0, "kind": {"offer": {"body": "first account"}}, "references": []}}),
        ),
    );
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"home": 0, "kind": {"offer": {"body": "replacement"}}, "references": []}}),
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
        "command": {"comment": {"home": 0, "kind": {"accept": {"offer": 1}}, "references": []}}
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
            json!({"establish_workspace": {"task_id": 0, "base": "abc123", "branch": "saccade/t-0", "worktree": "/repo/wt/t-0"}}),
        ),
    );
    assert_eq!(status, 400);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_native_integration_attaches_sends_and_submits_over_the_wire() {
    let db = scratch_db("native-messages");
    let base_url = spawn_server(&db).await;

    // a fresh native session registers: the machinery's own act
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "aaaa1111",
            "session_file": "/sessions/aaaa1111.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], true);
    assert_eq!(reply["incarnation"], 0);
    assert_eq!(reply["num"], 0);
    let (_, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "bbbb2222",
            "session_file": "/sessions/bbbb2222.jsonl",
        }),
    );
    assert_eq!(json_of(&body)["num"], 1);

    // an existing exact native session resolves, never a second birth
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "aaaa1111",
            "session_file": "/sessions/aaaa1111.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], false);
    assert_eq!(reply["incarnation"], 0);
    assert_eq!(reply["records"].as_array().unwrap().len(), 0);

    // the same native id carrying a different saved locator is not
    // the recorded association: it refuses, it never rebinds history
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "aaaa1111",
            "session_file": "/elsewhere/aaaa1111.jsonl",
        }),
    );
    assert_eq!(status, 400, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["error"]["code"], "saved_session_mismatch");
    let detail = reply["error"]["detail"].as_str().unwrap();
    assert!(detail.contains("i-0"), "{detail}");
    assert!(detail.contains("/sessions/aaaa1111.jsonl"), "{detail}");
    assert!(detail.contains("/elsewhere/aaaa1111.jsonl"), "{detail}");

    // a task for context, and a message without any
    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "hold the failing edge case"}}),
        ),
    );

    // an unscoped caller has no sender identity
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent", "scope": null},
            "to": 1,
            "body": "status?",
        }),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_incarnation_id");

    // the registered session sends: the sender rides the scope
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 1,
            "task": 0,
            "body": "please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["message"]["seq"], 3);
    assert_eq!(reply["message"]["num"], 0);
    assert_eq!(reply["from"]["seq"], 0);
    assert_eq!(reply["from"]["num"], 0);

    // the reply names the message it answers by its public number
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 1, "workspace": null}},
            "to": 0,
            "reply_to": 0,
            "body": "the edge case held under review",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["message"]["seq"], 4);
    assert_eq!(reply["message"]["num"], 1);

    // the mailbox read returns both directions in birth order; the
    // integration filters its own pending inbound
    let (status, body) = get_json(&base_url, "/api/v1/messages/pending?incarnation=1");
    assert_eq!(status, 200, "{body}");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 2);
    assert_eq!(mail[0]["seq"], 3);
    assert_eq!(mail[0]["num"], 0);
    assert_eq!(mail[0]["from"]["seq"], 0);
    assert_eq!(mail[0]["from"]["num"], 0);
    assert_eq!(mail[0]["from"]["actor"], "pi");
    assert_eq!(mail[0]["task"], 0);
    assert_eq!(mail[0]["state"], "pending");
    assert_eq!(
        mail[0]["body"],
        "please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix"
    );
    assert_eq!(mail[1]["seq"], 4);
    assert_eq!(mail[1]["from"]["num"], 1);
    assert_eq!(mail[1]["to"]["num"], 0);
    assert_eq!(mail[1]["reply_to"], 0);

    // unknown recipients refuse the read
    let (status, body) = get_json(&base_url, "/api/v1/messages/pending?incarnation=99");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_incarnation_id");

    // submission is the addressee's machinery act; judgment tiers are
    // refused at the authority table
    let (status, body) = post_command(
        &base_url,
        &envelope("pi", "agent", json!({"submit_message": {"id": 3, "by": 1}})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "human_only");

    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/submit",
        &json!({"incarnation": 1, "message": 3}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["records"][0]["kind"], "message_submitted");

    // submission is a state, not an address change: the mail reads
    // submitted, and only the pending inbound goes
    let (_, body) = get_json(&base_url, "/api/v1/messages/pending?incarnation=1");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    let inbound_pending: Vec<&Value> = mail
        .iter()
        .filter(|m| m["to"]["seq"] == 1 && m["state"] == "pending")
        .collect();
    assert!(inbound_pending.is_empty());
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/submit",
        &json!({"incarnation": 1, "message": 4}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_message_id");

    // the sender's own mail reads in the mailbox too, with its state
    let (_, body) = get_json(&base_url, "/api/v1/messages/pending?incarnation=0");
    let mine = json_of(&body)["messages"].as_array().unwrap().clone();
    let sent_by_me: Vec<&Value> = mine.iter().filter(|m| m["from"]["seq"] == 0).collect();
    assert_eq!(sent_by_me.len(), 1);
    assert_eq!(sent_by_me[0]["seq"], 3);
    // one shared delivery state: the recipient's submission reads here too
    assert_eq!(sent_by_me[0]["state"], "submitted");
    assert_eq!(sent_by_me[0]["to"]["num"], 1);

    // the messages are durable, addressed, and home-free: the raw log
    // holds them off every task thread
    let mut response = ureq::get(&format!("{base_url}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    let rows: Value = serde_json::from_str(&response.body_mut().read_to_string().unwrap()).unwrap();
    let sent = rows["records"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"] == "message_sent")
        .expect("the message rides the raw log");
    assert_eq!(
        sent["payload"]["task"], 0,
        "the task context rides without publication"
    );
    assert_eq!(
        sent["payload"]["body"],
        "please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// Post one JSON body to a path: the typed envelope either way.
fn post_json(base_url: &str, path: &str, body: &Value) -> (u16, String) {
    let mut response = ureq::post(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body.clone())
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

/// Get one JSON path.
fn get_json(base_url: &str, path: &str) -> (u16, String) {
    let mut response = ureq::get(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
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
            json!({"create_task": {"name": "   "}}),
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
    // a binding naming a task's birth record: the world teaches the
    // birth's own thread
    let fail = client::send(
        &base,
        &agent,
        Command::BindUse {
            id: saccade::CommentId(saccade::RecordId(0)),
            offer: saccade::CommentId(saccade::RecordId(0)),
            grounds: saccade::Prose::new("naming a birth".into()).unwrap(),
        },
        None,
    )
    .unwrap_err();
    match &fail {
        ClientFail::Refused { code, detail } => {
            assert_eq!(code, "invalid_comment_id");
            assert!(detail.contains("the birth of task t-0"), "{detail}");
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
            home: saccade::TaskId(9),
            kind: saccade::CommentKind::Note {
                body: saccade::Prose::new("talking to nothing".into()).unwrap(),
            },
            references: Vec::new(),
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
/// cloned against its own cursor and replayed, refuses identically -
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
            json!({"create_task": {"name": "migrate floop"}}),
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

// ---- the workspace operations: real git, the server's own effects ----

async fn spawn_workspace_server(db_path: &std::path::Path, repo: &std::path::Path) -> String {
    let state = AppState::open(db_path)
        .expect("the scratch tracker opens")
        .with_repo_root(repo.to_path_buf());
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

fn post_workspace(base_url: &str, path: &str, body: &Value) -> (u16, String) {
    let mut r = ureq::post(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body.clone())
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

/// A scratch git repo with two commits; returns its root and both tips.
fn scratch_repo(dir: &std::path::Path) -> (std::path::PathBuf, String, String) {
    let root = dir.join("repo");
    std::fs::create_dir_all(&root).unwrap();
    sh(&root, &["init", "-q", "--initial-branch=main"]);
    std::fs::write(root.join("readme"), "the project\n").unwrap();
    std::fs::write(root.join(".envrc"), "export SACCADE_ENVRC_LAYER=envrc\n").unwrap();
    sh(&root, &["add", "."]);
    sh(&root, &["commit", "-qm", "base"]);
    let first = sh(&root, &["rev-parse", "HEAD"]);
    std::fs::write(root.join("readme"), "the project, advanced\n").unwrap();
    sh(&root, &["commit", "-qam", "advanced"]);
    let head = sh(&root, &["rev-parse", "HEAD"]);
    (root, first, head)
}

// ---- the console: compose through the real HTTP surface ----

use saccade::CommentKind;
use saccade::api::AppState as ConsoleState;
use saccade::objects::comment::CommentId;
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
    assert_eq!(loc.as_deref(), Some("/t/0#c-0"));

    let world = world_of(&state);
    let c = &world.comments[&CommentId(RecordId(2))];
    assert!(matches!(c.comment.kind, CommentKind::Note { .. }));
    let lines = lines_of(&world, 0);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].seq, 2);
    assert_eq!(lines[0].num, 0);
    assert_eq!(lines[0].kind, "note");
    assert_eq!(lines[0].body, "kick off");

    // @c-0 cites: the marker rides the body, the mapping carries the
    // edge, the redirect stays on the citing comment's home thread
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=saw+it+%40c-0&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-1"));
    let world = world_of(&state);
    let lines = lines_of(&world, 0);
    assert_eq!(lines[1].seq, 3);
    assert_eq!(lines[1].body, "saw it [^1]");
    assert_eq!(
        lines[1].references,
        vec![saccade::views::RefTarget::Comment { num: 0, task: 0 }]
    );

    // @t-1 rehomes: the redirect follows the comment to its new task
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=%40t-1+belongs+there&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/1#c-2"));
    let world = world_of(&state);
    assert_eq!(lines_of(&world, 1)[0].seq, 4);
    assert_eq!(lines_of(&world, 0).len(), 2, "it left t-0");

    // bare ids are prose: unaddressed, uncited, on the composer's task
    let (status, _, loc) = post_form(
        &format!("{base}/compose"),
        &[],
        "task=0&body=see+t-1+and+c-0&who=jerry",
    );
    assert_eq!(status, 303);
    assert_eq!(loc.as_deref(), Some("/t/0#c-3"));
    let world = world_of(&state);
    let lines = lines_of(&world, 0);
    let last = lines.last().unwrap();
    assert_eq!(last.seq, 5);
    assert_eq!(last.body, "see t-1 and c-0");
    assert!(last.references.is_empty(), "bare ids stay prose");

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

/// A load failure refuses; a degraded log still serves its raw rows.
#[tokio::test(flavor = "multi_thread")]
async fn the_records_face_keeps_degraded_rows_and_refuses_corruption() {
    // a corrupt payload of a known kind: the log will not even load
    let db = scratch_db("records-corrupt");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "before the damage");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (1, 0, 0, 'pi', 'agent', 'commented', '{}', NULL)",
            [],
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 500);
    let body = r.body_mut().read_to_string().unwrap();
    assert_eq!(json_of(&body)["error"]["code"], "database_error");
    let (_, html) = get_html(&format!("{base}/t/0"));
    assert!(html.contains("corrupt"), "{html}");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();

    // an unknown kind: the world will not fold, the rows still travel
    let db = scratch_db("records-degraded");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "before the damage");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (1, 0, 0, 'pi', 'agent', 'time_travelled', '{}', NULL)",
            [],
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let body = r.body_mut().read_to_string().unwrap();
    assert_eq!(json_of(&body)["records"].as_array().unwrap().len(), 2);
    let (_, html) = get_html(&format!("{base}/t/0"));
    assert!(html.contains("unavailable"), "{html}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A supported local write shows on every server face.
#[tokio::test(flavor = "multi_thread")]
async fn a_local_write_shows_on_every_server_face() {
    let db = scratch_db("local-visible");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "server-born task");

    let local = Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };
    let mut conn = saccade::db::open(&db).unwrap();
    saccade::db::record(
        &mut conn,
        &local,
        Command::Comment {
            home: TaskId(0),
            kind: CommentKind::Note {
                body: Prose::new("LOCAL_MARKER".into()).unwrap(),
            },
            references: Vec::new(),
        },
        saccade::db::now_epoch(),
    )
    .unwrap();

    let (_, html) = get_html(&format!("{base}/t/0"));
    assert!(html.contains("LOCAL_MARKER"), "console reads the file");
    let (_, body) = {
        let mut r = ureq::get(&format!("{base}/api/v1/records"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap();
        (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
    };
    assert_eq!(json_of(&body)["records"].as_array().unwrap().len(), 2);

    // the next server write folds the same file, never a stale world
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
        .expect("the server write lands");
    let (_, html) = get_html(&format!("{base}/t/0"));
    assert!(html.contains("LOCAL_MARKER"), "{html}");
    assert!(html.contains("owned by"), "{html}");

    // a refusal names the log's fresh cursor, external writes included
    state
        .execute(
            &human_ctx(),
            Command::ClaimTask {
                id: TaskId(99),
                incarnation: None,
            },
            None,
            bare_request(),
        )
        .expect_err("no task t-99");
    let lines = attempts_lines(&db);
    let refused = lines
        .iter()
        .rev()
        .find(|l| l["outcome"] == "refused")
        .expect("the refusal took a line");
    assert_eq!(refused["cursor"], 2);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The console GET: unfocused at / and focused and composed at /t/N.
#[tokio::test(flavor = "multi_thread")]
async fn the_console_renders_board_and_focused_thread() {
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
        "the board browses live work"
    );
    assert!(home.contains("no task focused"));
    // no strip section renders anywhere
    assert!(!home.contains("OWNED | LAST RECORD"), "{home}");
    assert!(!home.contains("SUPERVISION"), "{home}");

    let (status, focused) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(focused.contains("migrate floop"));
    assert!(
        focused.contains("t-0 | owned by jerry"),
        "the head names the owner"
    );

    // an unknown task is a clean 404
    let (status, _) = get_html(&format!("{base}/t/99"));
    assert_eq!(status, 404);
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A rehoming compose answers a fetch with the same 303 a form post takes.
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
    assert_eq!(loc.as_deref(), Some("/t/1#c-0"));
    assert!(!body.contains("<section"), "no fragment rides the redirect");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// Every /t/... href the console page emits resolves: the rail is the
/// frame's navigation, so a dead link there is a dead frame. This pins
/// the rail's token-vs-number contract (c-585) across every link
/// form the page carries.
#[tokio::test(flavor = "multi_thread")]
async fn every_task_href_the_console_emits_resolves() {
    let db = scratch_db("console-hrefs");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    seed_task(&state, "other work");
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
                home: TaskId(1),
                kind: CommentKind::Offer {
                    body: Prose::new("the standing account".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .unwrap();
    // a comment citing the offer through its marker so the anchor
    // link form lights up
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Note {
                    body: Prose::new("a note worth keeping, citing [^1]".into()).unwrap(),
                },
                references: vec![RecordId(3)],
            },
            None,
            bare_request(),
        )
        .unwrap();

    // harvest every /t/... href from the unfocused and focused pages
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
    // both link forms light up
    assert!(
        hrefs.iter().any(|h| !h.contains('#')) && hrefs.iter().any(|h| h.contains("#c-")),
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

/// Identity is claimed at the act, on every decision form.
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
                home: TaskId(0),
                kind: CommentKind::Offer {
                    body: Prose::new("suite green".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .unwrap();
    // the focused page renders the decision doors on the offer's row
    let (status, html) = get_html(&format!("{base}/t/0"));
    assert_eq!(status, 200);
    assert!(html.contains("* OFFERED"));
    assert!(html.contains("action=\"/t/0/reject\""));
    assert!(html.contains("formaction=\"/t/0/accept\""));

    // a cookieless accept without a name refuses, naming the fix
    let (status, body, _) = post_form(&format!("{base}/t/0/accept"), &[], "accept=1&offer=2");
    assert_eq!(status, 400);
    assert!(body.contains("a name is required to record the act"));

    // a cookieless accept with who records human tier under that name,
    // and the first claim sets the cookie like compose does
    let (status, _, loc, cookie) = post_form_ck(
        &format!("{base}/t/0/accept"),
        &[],
        "accept=1&who=jerry&offer=2",
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
        .find(|r| r.kind == "commented" && r.payload.contains("accept"))
        .expect("the accept landed");
    assert_eq!(accepted.actor, "jerry");
    assert_eq!(accepted.tier, "human");
    assert_eq!(
        task_view(&world_of(&state), TaskId(0)).unwrap().state,
        "concluded"
    );

    // a producer may reject its own account with grounds
    seed_task(&state, "other work");
    state
        .execute(
            &pi,
            Command::Comment {
                home: TaskId(1),
                kind: CommentKind::Offer {
                    body: Prose::new("the second account".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .unwrap();
    // grounds are required: the empty form refuses at the console door
    let (status, body, _) = post_form(
        &format!("{base}/t/1/reject"),
        &[],
        "reject=1&who=jerry&grounds=+&offer=5",
    );
    assert_eq!(status, 400);
    assert!(body.contains("a rejection needs grounds"), "{body}");

    let (status, _, loc, _) = post_form_ck(
        &format!("{base}/t/1/reject"),
        &[],
        "reject=1&who=jerry&grounds=the+edge+case+still+fails&offer=5",
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
                home: TaskId(1),
                kind: CommentKind::Offer {
                    body: Prose::new("the third account".into()).unwrap(),
                },
                references: Vec::new(),
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
    // one form: reject the action, accept on formaction, one who field
    let doors = blank
        .split("class=\"jform\"")
        .nth(1)
        .expect("the decision doors render")
        .split("</form>")
        .next()
        .unwrap();
    assert_eq!(doors.split("name=\"who\"").count() - 1, 1, "{doors}");
    assert!(doors.contains("name=\"accept\""), "{doors}");
    assert!(doors.contains("name=\"reject\""), "{doors}");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A console refusal through judgment leaves its fat line: the raw form
/// body, the code, and no client binary to name. The stale-form
/// scenario rides here: a page rendered for one offer cannot decide its
/// replacement - the form names the offer it showed, and the fold's
/// stale-offer guard refuses it.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_console_decision_leaves_its_line() {
    let db = scratch_db("console-refused");
    let (base, state) = spawn_console(&db).await;
    seed_task(&state, "migrate floop");
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Offer {
                    body: Prose::new("the first account".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .unwrap();

    // the decision forms carry the offer they rendered
    let (_, html) = get_html(&format!("{base}/t/0"));
    assert!(html.contains("name=\"offer\" value=\"1\""), "{html}");

    // the page's reject form rejects offer 1, opening the task again
    let (status, _, _) = post_form(
        &format!("{base}/t/0/reject"),
        &[],
        "reject=1&who=jerry&grounds=the+edge+case+fails&offer=1",
    );
    assert_eq!(status, 303);

    // a replacement offer stands
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Offer {
                    body: Prose::new("the replacement account".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .unwrap();

    // the stale accept form names offer 1 and refuses
    let (status, body, _) = post_form(
        &format!("{base}/t/0/accept"),
        &[],
        "accept=1&who=jerry&offer=1",
    );
    assert_eq!(status, 400);
    assert!(
        body.contains("that offer is not the task's standing one"),
        "{body}"
    );

    // the stale reject form meets the same guard, its grounds kept
    // verbatim as an editable draft
    let (status, body, _) = post_form(
        &format!("{base}/t/0/reject"),
        &[],
        "reject=1&who=jerry&grounds=SPECIFIC_DRAFT_LOSS_CHECK&offer=1",
    );
    assert_eq!(status, 400);
    assert!(
        body.contains("that offer is not the task's standing one"),
        "{body}"
    );
    assert!(
        body.contains(">SPECIFIC_DRAFT_LOSS_CHECK</textarea>"),
        "the refused grounds ride back inside the textarea: {body}"
    );

    let (status, _, _, _) = post_form_ck(
        &format!("{base}/t/0/accept"),
        &[],
        "accept=1&who=jerry&offer=3",
    );
    assert_eq!(status, 303);
    assert_eq!(
        task_view(&world_of(&state), TaskId(0)).unwrap().state,
        "concluded"
    );

    // the decided task's second accept refuses
    let (status, body, _) = post_form(
        &format!("{base}/t/0/accept"),
        &[],
        "accept=1&who=jerry&offer=3",
    );
    assert_eq!(status, 400);
    assert!(body.contains("InvalidStateTransition"), "{body}");
    let lines = attempts_lines(&db);
    let refused = &lines[lines.len() - 1];
    assert_eq!(refused["outcome"], "refused");
    assert_eq!(refused["code"], "invalid_state_transition");
    assert_eq!(refused["actor"], "jerry");
    assert_eq!(refused["client"], Value::Null);
    assert_eq!(refused["request"], "accept=1&who=jerry&offer=3");
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
    assert_eq!(lines[t0 + 1], "  t-0  pi  migrate floop");
    assert_eq!(lines[t0 + 2], "  c-0  pi  the floop migration proceeds");
    assert_eq!(lines[t0 + 3], "  c-1  pi  floop landed; suite green");
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
    // the retired '#' term refuses, naming the c- form instead of matching words
    let (ok, out, _) = sac(&["search", "c-99999"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (c-99999)\ncursor 5");
    let (ok, _, err) = sac(&["search", "#99999"]);
    assert!(!ok, "{err}");
    assert!(
        err.contains("drop the '#': the reference search is c-99999"),
        "{err}"
    );
    // a reference number past the id space refuses at the term door
    // with the ordinary exit, never a panic
    let (ok, _, err) = sac(&["search", "c-184467440737095516160"]);
    assert!(!ok);
    assert!(
        err.contains("'c-184467440737095516160' is not a reference (expected c-<n> or t-<n>)"),
        "{err}"
    );
    let (ok, _, err) = sac(&["search", "t-184467440737095516160"]);
    assert!(!ok);
    assert!(err.contains("is not a reference"), "{err}");
    let (ok, out, _) = sac(&["search", "by:nobody"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (by:nobody)\ncursor 5");

    // the anchor window reads the log's own rows: c-1 is the dense
    // number of the offer at seq 4; rows carry their seq as provenance
    let (ok, out, _) = sac(&["search", "c-1", "-C", "1"]);
    assert!(ok, "{out}");
    assert!(out.contains("seq 3  task_claimed  pi/agent"), "{out}");
    assert!(
        out.contains("seq 4  commented  pi/agent  floop landed"),
        "{out}"
    );

    // an empty query refuses at the grammar door, naming the moves
    let (ok, _, err) = sac(&["search"]);
    assert!(!ok);
    assert!(err.contains("give at least one term"), "{err}");
    assert!(err.contains("the moves"), "{err}");
    assert!(err.contains("c-907 -C 3"), "{err}");

    // show opens comments by their public number in the thread view's
    // body format - the multi-line note is the tracker's third comment
    let (ok, out, _) = sac(&["show", "c-2"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "  c-2  pi [note]\n  floop guards the door the second line holds the detail\ncursor 5"
    );
    let (ok, out, _) = sac(&["show", "5"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("seq 5  commented  pi\n"),
        "the raw event rides whole: {out}"
    );
    assert!(out.contains("floop guards the door"), "{out}");
    // the retired '#' id refuses, naming the bare doors
    let (ok, _, err) = sac(&["show", "#5"]);
    assert!(!ok, "{err}");
    assert!(err.contains("drop the '#'"), "{err}");

    // a birth record is not a comment: the typed door names the fold's
    // comment extent, the bare door shows the raw event
    assert!(sac(&["create", "task", "guard the birth door"]).0);
    let (ok, _, err) = sac(&["show", "c-6"]);
    assert!(!ok, "{err}");
    assert!(
        err.contains("c-6 names no comment; this fold holds 3 comments, c-0 through c-2"),
        "{err}"
    );
    let (ok, out, _) = sac(&["show", "6"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("seq 6  task_created  pi\n"),
        "the raw birth rides whole: {out}"
    );

    // several ids render each, threads and records in one call
    let (ok, out, _) = sac(&["show", "t-1", "c-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1  open  floop guard"), "{out}");
    assert!(out.contains("c-1  pi"), "{out}");

    // a number that parses but names nothing the log holds refuses
    // with the log's own extent, typed or bare
    let (ok, _, err) = sac(&["show", "c-99999"]);
    assert!(!ok);
    assert!(
        err.contains("c-99999 names no comment; this fold holds 3 comments"),
        "{err}"
    );
    let (ok, _, err) = sac(&["show", "99999"]);
    assert!(!ok);
    assert!(
        err.contains("no record 99999; the log holds 7 records, 0 through 6"),
        "{err}"
    );
    // the bare door opens any record: an ownership event renders as
    // the log renders it, header and payload
    let (ok, out, _) = sac(&["show", "3"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("seq 3  task_claimed  pi\n"),
        "the raw event rides whole: {out}"
    );
    // the typed door opens the offer as its comment block, prose wrapped
    let (ok, out, _) = sac(&["show", "c-1"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "  c-1  pi [offer]\n  floop landed; suite green\ncursor 6"
    );
    let (ok, _, err) = sac(&["show", "bogus"]);
    assert!(!ok);
    assert!(
        err.contains("'bogus' is not an id (expected c-<n>, t-<n>, m-<n>, or a bare <seq>)"),
        "{err}"
    );

    // a number past the id space refuses at the grammar door with the
    // ordinary exit, never a panic: digits are grammar, the id space is not
    let (ok, _, err) = sac(&["show", "c-184467440737095516160"]);
    assert!(!ok);
    assert!(
        err.contains(
            "'c-184467440737095516160' is not an id (expected c-<n>, t-<n>, m-<n>, or a bare <seq>)"
        ),
        "{err}"
    );
    let (ok, _, err) = sac(&["show", "m-184467440737095516160"]);
    assert!(!ok);
    assert!(err.contains("is not an id"), "{err}");
    let (ok, _, err) = sac(&["show", "184467440737095516160"]);
    assert!(!ok);
    assert!(err.contains("is not an id"), "{err}");
    let (ok, _, err) = sac(&["show", "#184467440737095516160"]);
    assert!(!ok);
    assert!(
        err.contains("a comment is c-<n>, the raw record at a position is the bare <seq>"),
        "{err}"
    );

    // the piped face: one id per line, failures skip with a note
    let pipe = |input: &str, args: &[&str]| {
        use std::io::Write as _;
        use std::process::Stdio;
        let mut child = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
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
    assert!(sac(&["comment", "c-0", answer]).0);
    assert!(sac(&["comment", "t-0", "a short third note"]).0);
    assert!(sac(&["revise", "c-2", "revised bytes"]).0);

    // the thread's json face: bodies byte-exact, replies folded in,
    // the cursor the read inspected through
    let (ok, out, _) = sac(&["show", "--json", "t-0"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["cursor"], json!(4));
    let thread = &json["blocks"][0];
    assert_eq!(thread["id"], json!("t-0"));
    assert_eq!(thread["name"], json!("byte door"));
    let entries = thread["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0]["num"], json!(0));
    assert_eq!(entries[0]["seq"], json!(1));
    assert_eq!(entries[0]["kind"], json!("note"));
    assert_eq!(entries[0]["body"], json!(body));
    assert_eq!(entries[0]["home"], json!("t-0"));
    // the entries carry the citation mapping: the reply cites the note
    // it answers, its home the same thread
    assert_eq!(entries[0]["references"], json!([]));
    assert_eq!(entries[1]["num"], json!(1));
    assert_eq!(entries[1]["seq"], json!(2));
    assert_eq!(entries[1]["home"], json!("t-0"));
    assert_eq!(entries[1]["references"], json!(["c-0"]));
    assert_eq!(entries[1]["body"], json!(answer));
    // the fold presents the latest revision, byte-exact too
    assert_eq!(entries[2]["seq"], json!(3));
    assert_eq!(entries[2]["body"], json!("revised bytes"));
    assert_eq!(entries[2]["revised"], json!(true));
    assert_eq!(entries[2]["reviser"], json!(null));

    // the record door answers with the same block, alone
    let (ok, out, _) = sac(&["show", "--json", "c-1"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["blocks"][0]["body"], json!(answer));
    assert_eq!(json["blocks"][0]["references"], json!(["c-0"]));

    // the bare position on the json face: the raw log row, no fold
    let (ok, out, _) = sac(&["show", "--json", "2"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["blocks"][0]["seq"], json!(2));
    assert_eq!(json["blocks"][0]["payload"]["home"], json!(0));
    assert_eq!(json["blocks"][0]["payload"]["references"], json!([1]));

    // the log's json face carries folded thread membership: a reply's
    // target is its comment, its thread is still t-0 - and the cursor
    // the read inspected through rides beside the rows
    let (ok, out, _) = sac(&["log", "--json"]);
    assert!(ok, "{out}");
    let log: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(log["cursor"], json!(4));
    let rows: Vec<Value> = log["records"]
        .as_array()
        .expect("the records array")
        .clone();
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
    // the revision rides the provenance index: its home is the
    // comment it repairs
    let revised = rows
        .iter()
        .find(|r| r["kind"] == json!("comment_revised"))
        .expect("the revision row");
    assert_eq!(revised["thread"], json!("t-0"));

    // text reads end with the cursor line too: show, log, list, search, anchor
    for args in [
        vec!["show", "t-0"],
        vec!["show", "c-1"],
        vec!["log"],
        vec!["list"],
        vec!["search", "byte"],
        vec!["search", "c-2", "-C", "1"],
    ] {
        let (ok, out, _) = sac(&args);
        assert!(ok, "{}: {out}", args.join(" "));
        assert_eq!(
            out.lines().last().unwrap(),
            "cursor 4",
            "{}: {out}",
            args.join(" ")
        );
    }

    // the pipe's json face: every line's outcome, skips included
    let pipe = |input: &str, args: &[&str]| {
        use std::io::Write as _;
        use std::process::Stdio;
        let mut child = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
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
    let (ok, out, _) = pipe("bogus\nc-0\n", &["show", "--stdin", "--json"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    let rows = json["blocks"].as_array().expect("the blocks array");
    assert_eq!(rows.len(), 2);
    assert_eq!(json["cursor"], json!(4));
    assert_eq!(rows[0]["id"], json!("bogus"));
    assert!(rows[0]["error"].is_string());
    assert_eq!(rows[1]["body"], json!(body));

    // the stdin text face ends with the cursor line too
    let (ok, out, _) = pipe("t-0\n", &["show", "--stdin"]);
    assert!(ok, "{out}");
    assert_eq!(out.lines().last().unwrap(), "cursor 4", "{out}");

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn create_reply_names_the_born_task() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("create-reply");

    let out = std::process::Command::new(bin)
        .arg("--db")
        .arg(&path)
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
/// copy - the on-disk source byte for byte, SKILL.md stamped with the
/// binary's version inside its frontmatter - and check verifies the
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
        "error: .agents/skills/saccade differs from this binary's skill - delete the directory and run sac skill install to deploy this version"
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

/// A --db write answers with the wire's own reply shape.
#[tokio::test(flavor = "multi_thread")]
async fn local_replies_match_the_wire() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let local_db = scratch_db("local-shape");
    let served_db = scratch_db("served-shape");

    let local = |db: &std::path::Path, args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(db)
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
    let created_local = local(&local_db, &["--json", "create", "task", "offline task"]);
    let served = spawn_server(&served_db).await;
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"create_task": {"name": "offline task"}}}),
    );
    let created_served = json_of(&body);
    assert_eq!(created_local["id"], created_served["id"]);
    assert_eq!(
        created_local["records"][0]["kind"],
        created_served["records"][0]["kind"]
    );

    // a comment and a claim answer with their records, both paths
    let noted = local(&local_db, &["--json", "comment", "t-0", "the note"]);
    assert_eq!(noted["records"][0]["kind"], "commented");
    let claimed = local(&local_db, &["--json", "claim", "t-0"]);
    assert_eq!(claimed["records"][0]["kind"], "task_claimed");

    // an offer answers with its records, both paths, identical shapes
    let offered = local(&local_db, &["--json", "offer", "t-0", "suite green"]);
    assert_eq!(offered["records"][0]["kind"], "commented");
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"comment": {"home": 0, "kind": {"offer": {"body": "suite green"}}, "references": []}}}),
    );
    let served_offered = json_of(&body);
    assert_eq!(
        served_offered["records"][0]["kind"],
        offered["records"][0]["kind"]
    );

    std::fs::remove_dir_all(local_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(served_db.parent().unwrap()).unwrap();
}

// ---- the wait door through the CLI face: committed, delayed, mute ----

#[test]
fn the_wait_door_wakes_on_committed_and_delayed_matches_and_lands_nothing() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("wait-door");
    let sac = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .env("SACCADE_ACTOR", "pi")
            // a working session's declared scope must not leak in
            .env("SACCADE_TASK", "")
            .env("SACCADE_INCARNATION_SEQ", "")
            .env("SACCADE_WORKSPACE", "")
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let sac_quiet = |args: &[&str]| {
        let (ok, out, err) = sac(args);
        assert!(ok, "sac {:?}: {out}{err}", args);
    };
    // the test bounds the blocking observation itself and kills a wedged child
    let spawn_waiter = |args: &[&str]| {
        std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .env("SACCADE_ACTOR", "pi")
            .env("SACCADE_TASK", "")
            .env("SACCADE_INCARNATION_SEQ", "")
            .env("SACCADE_WORKSPACE", "")
            .args(["--json", "wait"])
            .args(args)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn the waiting sac")
    };
    let observed = |mut child: std::process::Child| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            match child.try_wait().expect("poll the waiting sac") {
                Some(_) => break,
                None if std::time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    panic!("the wait never woke");
                }
                None => std::thread::sleep(std::time::Duration::from_millis(100)),
            }
        }
        let out = child.wait_with_output().expect("the wait's output");
        assert!(out.status.success());
        let wait: Value = serde_json::from_slice(&out.stdout).unwrap();
        let seqs: Vec<u64> = wait["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["seq"].as_u64().unwrap())
            .collect();
        (seqs, wait["cursor"].clone())
    };
    let row_count = || {
        let (ok, out, err) = sac(&["--json", "log"]);
        assert!(ok, "{out}{err}");
        let log: Value = serde_json::from_str(&out).unwrap();
        log["records"].as_array().unwrap().len()
    };

    sac_quiet(&["create", "task", "wait for the reply"]); // seq 0
    sac_quiet(&["comment", "t-0", "the opening note"]); // seq 1 -> c-1

    // activity that committed before the wait: caught, never blocked
    let (seqs, cursor) = observed(spawn_waiter(&[
        "t-0", "--after", "0", "--until", "activity",
    ]));
    assert_eq!(seqs, vec![1]);
    assert_eq!(cursor, 1);

    // the cursor a read disclosed: a write landing between the read
    // and the wait is already committed when the wait checks first
    sac_quiet(&["comment", "t-0", "a follow-up note"]); // seq 2
    let (_, out, _) = sac(&["--json", "log"]);
    let log: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(log["cursor"], 2);
    sac_quiet(&["comment", "t-0", "a late addition"]); // seq 3
    let (seqs, cursor) = observed(spawn_waiter(&[
        "t-0", "--after", "2", "--until", "activity",
    ]));
    assert_eq!(seqs, vec![3]);
    assert_eq!(cursor, 3);

    // the blocked wait wakes on another process's committed write
    let mut child = spawn_waiter(&["c-0", "--after", "3", "--until", "activity"]);
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(
        child.try_wait().unwrap().is_none(),
        "the wait woke with nothing matching"
    );
    sac_quiet(&["comment", "c-0", "the reply on the note"]); // seq 4
    let (seqs, cursor) = observed(child);
    assert_eq!(seqs, vec![4]);
    assert_eq!(cursor, 4);

    // a completed wait lands nothing
    let before = row_count();
    let (_, cursor) = observed(spawn_waiter(&[
        "t-0", "--after", "3", "--until", "activity",
    ]));
    assert_eq!(cursor, 4);
    assert_eq!(row_count(), before, "the wait landed nothing");

    // the sibling neither wakes the comment wait nor rides in its records
    let mut child = spawn_waiter(&["c-0", "--after", "4", "--until", "activity"]);
    sac_quiet(&["comment", "t-0", "an unrelated sibling"]); // seq 5
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(
        child.try_wait().unwrap().is_none(),
        "unrelated activity woke the comment wait"
    );
    sac_quiet(&["comment", "c-0", "the real reply"]); // seq 6
    let (seqs, cursor) = observed(child);
    assert_eq!(seqs, vec![6]);
    assert_eq!(cursor, 6);

    // the text face: matched record lines with their seq provenance,
    // then the cursor for the next wait
    let (ok, out, _) = sac(&["wait", "t-0", "--after", "5", "--until", "activity"]);
    assert!(ok, "{out}");
    assert!(out.contains("seq 6"), "{out}");
    assert!(out.trim_end().ends_with("cursor 6"), "{out}");

    // the offer wait stays down through a note and wakes on the offer's birth
    let mut child = spawn_waiter(&["t-0", "--after", "6", "--until", "offer"]);
    sac_quiet(&["comment", "t-0", "still only chatter"]); // seq 7
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(
        child.try_wait().unwrap().is_none(),
        "a note woke the offer wait"
    );
    sac_quiet(&["offer", "t-0", "the account, ready for review"]); // seq 8
    let (seqs, cursor) = observed(child);
    assert_eq!(seqs, vec![8]);
    assert_eq!(cursor, 8);

    // the committed offer is caught before any blocking
    let (seqs, cursor) = observed(spawn_waiter(&["t-0", "--after", "7", "--until", "offer"]));
    assert_eq!(seqs, vec![8]);
    assert_eq!(cursor, 8);

    // an offer revision is not a new birth; the interrupted wait lands nothing
    let mut child = spawn_waiter(&["t-0", "--after", "8", "--until", "offer"]);
    sac_quiet(&["revise", "c-7", "the account, restated"]); // seq 9
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(
        child.try_wait().unwrap().is_none(),
        "a revision woke the offer wait"
    );
    child.kill().unwrap();
    let _ = child.wait();
    assert_eq!(row_count(), 10, "the interrupted wait landed nothing");

    // offer mode addresses a task only
    let (ok, _, stderr) = sac(&["wait", "c-0", "--after", "0", "--until", "offer"]);
    assert!(!ok);
    assert!(stderr.contains("--until offer"), "{stderr}");
    assert!(stderr.contains("activity"), "{stderr}");

    // neither the cursor nor the mode is ever chosen for the caller
    let (ok, _, stderr) = sac(&["wait", "t-0", "--after", "0"]);
    assert!(!ok, "a wait without --until ran: {stderr}");
    let (ok, _, stderr) = sac(&["wait", "t-0", "--until", "activity"]);
    assert!(!ok, "a wait without --after ran: {stderr}");

    // a missing target refuses at the read door
    let (ok, _, stderr) = sac(&["wait", "t-9", "--after", "0", "--until", "activity"]);
    assert!(!ok);
    assert!(stderr.contains("no task t-9"), "{stderr}");

    // a target number past the id space refuses at the token door with
    // the ordinary exit, never a panic
    let (ok, _, stderr) = sac(&[
        "wait",
        "c-184467440737095516160",
        "--after",
        "0",
        "--until",
        "activity",
    ]);
    assert!(!ok);
    assert!(
        stderr.contains("'c-184467440737095516160' is not a wait target (expected t-<n> or c-<n>)"),
        "{stderr}"
    );

    // the same wait over the wire, waking on a local write
    let mut serve = std::process::Command::new(bin)
        .args(["serve", "--db"])
        .arg(&path)
        .args(["--port", "0"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn serve");
    let hint = saccade::paths::address_at(&path);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let url = loop {
        match std::fs::read_to_string(&hint) {
            Ok(text) if !text.trim().is_empty() => break text.trim().to_string(),
            _ if std::time::Instant::now() >= deadline => {
                panic!("serve never published its address")
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    };
    let mut wire_waiter = std::process::Command::new(bin)
        .arg("--server")
        .arg(&url)
        .args([
            "--json", "wait", "t-0", "--after", "9", "--until", "activity",
        ])
        .env("SACCADE_ACTOR", "pi")
        .env("SACCADE_TASK", "")
        .env("SACCADE_INCARNATION_SEQ", "")
        .env("SACCADE_WORKSPACE", "")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn the wire waiter");
    std::thread::sleep(std::time::Duration::from_millis(1200));
    assert!(
        wire_waiter.try_wait().unwrap().is_none(),
        "the wire wait woke with nothing matching"
    );
    sac_quiet(&["comment", "t-0", "the served reply"]); // seq 10
    let (seqs, cursor) = observed(wire_waiter);
    assert_eq!(seqs, vec![10]);
    assert_eq!(cursor, 10);
    let _ = serve.kill();
    let _ = serve.wait();

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

// ---- the workspace doors through the CLI face: a fake host, real exec ----

/// A fake `orca` with pinned wire shapes; shebang /bin/sh - the sandbox has no /usr/bin/env.
fn fake_host(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("host-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let orca = bin_dir.join("orca");
    std::fs::write(
        &orca,
        r#"#!/bin/sh
printf '%s\n' "$@" >> "$FAKE_ORCA_LOG"
if [ "$1" = worktree ]; then exit 0; fi
if [ "$1" = terminal ] && [ "$2" = create ]; then
  n=$(($(wc -l < "$FAKE_ORCA_STATE" 2>/dev/null || echo 0) + 1))
  handle="term_actual_shape_$n"
  selector=""
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--worktree" ]; then selector="$a"; fi
    prev="$a"
  done
  path="${selector#path:}"
  if [ -n "${FAKE_ORCA_CREATE_FAILS:-}" ]; then
    printf 'the host refused the create\n' >&2
    exit 1
  fi
  printf '{"handle":"%s","worktreePath":"%s","orphaned":false}\n' "$handle" "$path" >> "$FAKE_ORCA_STATE"
  if [ -n "${FAKE_ORCA_SPAWN:-}" ]; then
    submitted=""
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--command" ]; then submitted="$a"; fi
      prev="$a"
    done
    child_guard=""
    if [ -f "${FAKE_ORCA_GUARD_FILE:-/dev/null-nothing}" ]; then
      child_guard=$(cat "$FAKE_ORCA_GUARD_FILE")
    fi
    printf 'submitted-command-started\n' >> "$FAKE_ORCA_CHILD_LOG"
    if [ -n "$child_guard" ]; then
      SACCADE_TEST_GUARD="$child_guard" sh -c "$submitted" >> "$FAKE_ORCA_CHILD_LOG" 2>&1 &
    else
      sh -c "$submitted" >> "$FAKE_ORCA_CHILD_LOG" 2>&1 &
    fi
  fi
  printf '{"ok":true,"result":{"terminal":{"handle":"%s"}}}\n' "$handle"
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = list ]; then
  printf '{"ok":true,"result":{"terminals":['
  paste -sd, "$FAKE_ORCA_STATE" 2>/dev/null
  printf ']}}\n'
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = switch ]; then
  if [ "${FAKE_ORCA_NAVIGATE:-1}" = "1" ]; then
    printf '{"ok":true,"result":{"focus":{"navigated":true}}}\n'
  else
    printf '{"ok":true,"result":{"focus":{"navigated":false,"reason":"no renderer"}}}\n'
  fi
  exit 0
fi
exit 1
"#,
    )
    .unwrap();
    std::fs::set_permissions(&orca, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    orca
}

/// A stub harness: one dump of argv, cwd, and the saccade environment.
fn stub_pi(dir: &std::path::Path) -> std::path::PathBuf {
    let pi = dir.join("pi dump");
    std::fs::write(
        &pi,
        r#"#!/bin/sh
{
  printf 'argv: %s\n' "$*"
  printf 'pwd: %s\n' "$PWD"
  env | grep '^SACCADE_' | sort
} > "$PI_DUMP"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&pi, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    pi
}

/// A fixture harness, not actual Pi: it takes the named real kernel guard, records what it saw, and stays alive until killed.
fn guard_pi(dir: &std::path::Path) -> std::path::PathBuf {
    let pi = dir.join("pi guard-holder");
    std::fs::write(
        &pi,
        r#"#!/bin/sh
{
  printf 'argv: %s\n' "$*"
  printf 'pwd: %s\n' "$PWD"
  env | grep '^SACCADE_' | sort
} > "$PI_DUMP"
printf '%s\n' "$$" > "$PI_PIDFILE"
exec 9>>"$SACCADE_TEST_GUARD"
if ! flock -n 9; then
  printf 'guard-busy\n' >> "$FIXTURE_LOG"
  exit 3
fi
printf 'guard-taken\n' >> "$FIXTURE_LOG"
# exec: this pid becomes the sleep and the sole lock holder, so one kill frees the guard
exec sleep 180
"#,
    )
    .unwrap();
    std::fs::set_permissions(&pi, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    pi
}

/// One bounded sac subprocess in its own process group; a bound pass kills the whole group and reports the timeout.
fn bounded_sac<S: AsRef<str>>(
    bin: &str,
    args: &[S],
    envs: &[(&str, String)],
) -> (bool, String, String, bool) {
    let mut cmd = std::process::Command::new(bin);
    cmd.args(args.iter().map(AsRef::as_ref))
        .process_group(0)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // the scrub removes rather than empties: these names carry clap env fallbacks
    for key in [
        "SACCADE_REPO",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_ACTOR",
        "SACCADE_TASK",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "SACCADE_LAUNCH_TOKEN",
        "SACCADE_ACTIVATION_LAUNCHER",
    ] {
        cmd.env_remove(key);
    }
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let mut child = cmd.spawn().expect("the bounded sac spawns");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(75);
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            _ => {
                timed_out = true;
                break None;
            }
        }
    };
    if timed_out {
        // the whole group goes: a plain parent kill would leak descendants holding shared file descriptions
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = std::io::Read::read_to_string(&mut pipe, &mut stdout);
    }
    if let Some(mut pipe) = child.stderr.take() {
        let _ = std::io::Read::read_to_string(&mut pipe, &mut stderr);
    }
    let _ = child.wait();
    (
        !timed_out && status.is_some_and(|s| s.success()),
        stdout,
        stderr,
        timed_out,
    )
}

/// The identity door through the CLI face: the environment's
/// declaration read back, the retired task-bound words refused at the
/// parser, and the discovered server arming a launch it did not name.
#[tokio::test(flavor = "multi_thread")]
async fn identity_reads_the_scope_and_the_retired_words_refuse_at_the_parser() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("ws-cli");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();

    // every sac run rides the fake host's PATH, scrubbed of inherited scope
    let run = |args: &[String], envs: &[(&str, &str)]| {
        let mut cmd = std::process::Command::new(bin);
        cmd.env(
            "PATH",
            format!(
                "{}:{}",
                host.parent().unwrap().display(),
                std::env::var("PATH").unwrap()
            ),
        )
        .env("FAKE_ORCA_LOG", host_log.display().to_string())
        .env("FAKE_ORCA_STATE", host_state.display().to_string())
        .env("SACCADE_PI", &pi)
        .env_remove("SACCADE_ACTOR")
        .env_remove("SACCADE_REPO")
        .env_remove("SACCADE_DB")
        .env_remove("SACCADE_SERVER")
        .env_remove("SACCADE_TASK")
        .env_remove("SACCADE_INCARNATION_SEQ")
        .env_remove("SACCADE_WORKSPACE")
        .current_dir(&root);
        cmd.args(args);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let routed = |sub: &[&str]| -> Vec<String> {
        let mut full = vec!["--server".to_string(), base_url.clone()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };
    let at_repo = |mut args: Vec<String>| -> Vec<String> {
        args.push("--repo".into());
        args.push(root.display().to_string());
        args
    };

    // identity: the environment's declaration, read-only
    let worktree = db.parent().unwrap().join("declared-ws");
    std::fs::create_dir_all(&worktree).unwrap();
    let (ok, out, err) = run(
        &at_repo(routed(&["identity", "--json"])),
        &[
            ("SACCADE_ACTOR", "pi"),
            ("SACCADE_TASK", "t-0"),
            ("SACCADE_INCARNATION_SEQ", "1"),
            ("SACCADE_WORKSPACE", worktree.to_str().unwrap()),
        ],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    let identity: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(identity["actor"], "pi");
    assert_eq!(identity["tier"], "agent");
    assert_eq!(identity["task"], "t-0");
    // the machine channel discloses the birth position as a number;
    // the public i-N is the fold's dense numbering, never this face
    assert_eq!(identity["incarnation_seq"], 1);
    assert!(identity.get("incarnation").is_none());
    assert_eq!(identity["workspace"], worktree.display().to_string());

    // the retired task-bound words are CLI unknown: the parser itself
    // refuses them, no alias or redirect keeps them callable
    for word in ["start", "open", "exec", "checkpoint", "clean", "integrate"] {
        let (ok, out, err) = run(&routed(&[word]), &[("SACCADE_ACTOR", "pi")]);
        assert!(!ok, "{word} must refuse: {out}{err}");
        assert!(err.contains("unrecognized subcommand"), "{word}: {err}");
    }

    // discovered routing: no --server is named; the launch door resolves
    // the repo's published server and arms the native session with it.
    // The tracker lives at the repo's state root, so serve and clients
    // share one XDG_STATE_HOME
    let db2 = scratch_db("ws-cli-discovered");
    let (root2, _first2, _head2) = scratch_repo(db2.parent().unwrap());
    let state_home2 = db2.parent().unwrap().join("state-home");
    let root2c = std::fs::canonicalize(&root2).unwrap();
    let tracker2 = state_home2
        .join("saccade")
        .join(format!(
            "{}-{:06x}",
            saccade::paths::repo_basename(&root2c),
            saccade::paths::fnv1a(root2c.as_os_str().as_encoded_bytes())
        ))
        .join("saccade.db");
    let (mut serve2, discovered_url) = spawn_serve(
        bin,
        &tracker2,
        &["--repo", root2.to_str().unwrap()],
        &[("XDG_STATE_HOME", state_home2.to_str().unwrap())],
    );
    let discovered_envs = [
        ("SACCADE_ACTOR", "pi"),
        ("XDG_STATE_HOME", state_home2.to_str().unwrap()),
    ];
    let discovered_args = |sub: &[&str]| {
        let mut full = vec!["--repo".to_string(), root2.display().to_string()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };
    std::fs::write(&host_log, b"").unwrap();
    let chosen = db2.parent().unwrap().join("chosen-ws");
    std::fs::create_dir_all(&chosen).unwrap();
    let (ok, out, err) = run(
        &discovered_args(&[
            "launch",
            "--workspace",
            chosen.to_str().unwrap(),
            "--",
            "hello",
        ]),
        &discovered_envs,
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    let typed2 = command_of(&host_log, "");
    assert!(
        !typed2.contains("--server"),
        "no explicit flag: the discovery named the destination: {typed2}"
    );
    assert!(
        typed2.contains(&format!("SACCADE_SERVER={}", shell_quoted(&discovered_url))),
        "the discovered server arms the session: {typed2}"
    );
    assert!(
        typed2.contains(&format!(
            "SACCADE_REPO={}",
            shell_quoted(&root2.display().to_string())
        )),
        "{typed2}"
    );
    let logged = std::fs::read_to_string(&host_log).unwrap();
    assert!(
        logged.contains("--title\nsac launch\n"),
        "the terminal title names the door: {logged}"
    );

    let _ = serve2.kill();
    let _ = serve2.wait();
    std::fs::remove_dir_all(db2.parent().unwrap()).unwrap();

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_launch_scrubs_stale_inheritance() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("launch-scrub");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();

    let workspace = db.parent().unwrap().join("chosen-workspace");
    std::fs::create_dir_all(&workspace).unwrap();

    let out = std::process::Command::new(bin)
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--workspace",
            workspace.to_str().unwrap(),
            "--",
            "register and wait",
        ])
        .env(
            "PATH",
            format!(
                "{}:{}",
                host.parent().unwrap().display(),
                std::env::var("PATH").unwrap()
            ),
        )
        .env("FAKE_ORCA_LOG", host_log.display().to_string())
        .env("FAKE_ORCA_STATE", host_state.display().to_string())
        .env("SACCADE_PI", &pi)
        // everything stale the launching shell happens to carry
        .env("SACCADE_INCARNATION_SEQ", "999")
        .env("SACCADE_TASK", "t-0")
        .env("SACCADE_WORKSPACE", "/nowhere")
        .env("SACCADE_DB", "/stale/routing.db")
        .env("SACCADE_CLI", "/stale/sac")
        .output()
        .expect("spawn sac");
    let ok = out.status.success();
    let (out, err) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");

    let typed = command_of(&host_log, &format!("path:{}", workspace.display()));
    let expected_scrub = "env -u SACCADE_INCARNATION -u SACCADE_INCARNATION_SEQ \
                         -u SACCADE_TASK -u SACCADE_WORKSPACE -u SACCADE_DB";
    assert!(
        typed.contains(expected_scrub),
        "the launch scrubs stale inheritance: {typed}"
    );
    assert!(
        typed.contains(&format!(
            "SACCADE_ACTOR={}",
            shell_quoted(&command_label(&pi.display().to_string()))
        )),
        "the launch names the harness as the acting agent: {typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_SERVER={}", shell_quoted(&base_url))),
        "{typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_REPO={}", root.display())),
        "{typed}"
    );
    // the handoff helper's controller door is named, never guessed from
    // a PATH, and a stale inherited CLI path is scrubbed before the
    // routing assignment
    assert!(
        typed.contains("-u SACCADE_CLI"),
        "the launch scrubs a stale inherited CLI path: {typed}"
    );
    assert!(
        typed.contains(&format!(
            "SACCADE_CLI={}",
            shell_quoted(
                &std::fs::canonicalize(env!("CARGO_BIN_EXE_sac"))
                    .unwrap()
                    .display()
                    .to_string()
            )
        )),
        "{typed}"
    );
    assert!(
        !typed.contains("/stale/sac"),
        "the stale CLI path never survives the scrub: {typed}"
    );
    assert!(
        typed.ends_with(&shell_quoted("register and wait")),
        "{typed}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_delivery_bridge_parks_wakes_and_reports_liveness() {
    let db = scratch_db("native-bridge");
    let base_url = spawn_server(&db).await;

    // two registered peers: the addressee parks a bridge read
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "aaaa1111", "session_file": "/sessions/aaaa1111.jsonl"}),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "bbbb2222", "session_file": "/sessions/bbbb2222.jsonl"}),
    );
    assert_eq!(status, 200, "{body}");

    // a parked read holds while the mailbox is empty and announces its
    // incarnation for the call's duration
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let url = base_url.clone();
    std::thread::spawn(move || {
        let (status, body) = get_json(
            &url,
            "/api/v1/messages/pending?incarnation=1&wait_ms=10000&bridge=1",
        );
        done_tx.send((status, body)).unwrap();
    });
    std::thread::sleep(std::time::Duration::from_millis(400));
    if let Ok((status, body)) = done_rx.try_recv() {
        panic!("the parked read returned early: {status} {body}");
    }
    let (_, body) = get_json(&base_url, "/api/v1/native/bridge");
    let live = json_of(&body)["incarnations"].as_array().unwrap().clone();
    assert_eq!(live.len(), 1, "{body}");
    assert_eq!(live[0]["incarnation"], 1);
    assert_eq!(live[0]["connections"], 1);

    // a landed record wakes the parked read - the landing is the
    // trigger, never a timer over the log
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 1,
            "body": "the bridge wakes on the landing",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = done_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the landing woke the parked read");
    assert_eq!(status, 200, "{body}");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert!(
        mail.iter()
            .any(|m| m["to"]["seq"] == 1 && m["state"] == "pending"),
        "{body}"
    );

    // pending mail keeps a re-read immediate; the connection went with
    // the finished read
    let start = std::time::Instant::now();
    let (status, _) = get_json(
        &base_url,
        "/api/v1/messages/pending?incarnation=1&wait_ms=10000&bridge=1",
    );
    assert_eq!(status, 200);
    assert!(start.elapsed() < std::time::Duration::from_secs(2));
    std::thread::sleep(std::time::Duration::from_millis(150));
    let (_, body) = get_json(&base_url, "/api/v1/native/bridge");
    assert!(
        json_of(&body)["incarnations"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{body}"
    );

    // a compose launch token maps to the incarnation that claimed it
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "cccc3333",
            "session_file": "/sessions/cccc3333.jsonl",
            "launch_token": "launch-hex",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = get_json(&base_url, "/api/v1/native/launch?token=launch-hex");
    assert_eq!(status, 200, "{body}");
    // the reply names the birth record's seq and the dense public number
    assert_eq!(json_of(&body)["incarnation"], 3);
    assert_eq!(json_of(&body)["num"], 2);
    let (status, body) = get_json(&base_url, "/api/v1/native/launch?token=launch-unknown");
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "launch_unknown");

    // an existing session reattaching under a fresh token never
    // overwrites the door's mapping: only a fresh birth claims it
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "cccc3333",
            "session_file": "/sessions/cccc3333.jsonl",
            "launch_token": "launch-hijack",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["registered"], false);
    let (status, body) = get_json(&base_url, "/api/v1/native/launch?token=launch-hijack");
    assert_eq!(status, 404, "{body}");
    let (status, body) = get_json(&base_url, "/api/v1/native/launch?token=launch-hex");
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["incarnation"], 3);

    // a fresh fork under a STALE token births its own incarnation but
    // never retargets the door's mapping: the correlation keeps
    // pointing at the birth that claimed it
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "fork000001",
            "session_file": "/sessions/fork000001.jsonl",
            "launch_token": "launch-hex",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["registered"], true);
    let forked = json_of(&body)["incarnation"].as_u64().unwrap();
    assert_ne!(forked, 3, "the fork is a fresh birth");
    let (status, body) = get_json(&base_url, "/api/v1/native/launch?token=launch-hex");
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        json_of(&body)["incarnation"],
        3,
        "the mapping never retargeted"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A serving subprocess: publishes its address file, so activation's
/// routing is the real one, and carries the launcher seam plus the
/// fake host on PATH for its own activation child.
fn spawn_serving_subprocess(
    db: &Path,
    repo: &Path,
    bin: &str,
    host_dir: &Path,
    launcher: Option<&Path>,
    state_home: Option<&Path>,
    extra_envs: &[(&str, &str)],
) -> (String, u32) {
    let mut cmd = std::process::Command::new(bin);
    cmd.args([
        "--db",
        db.to_str().unwrap(),
        "--repo",
        repo.to_str().unwrap(),
        "serve",
        "--bind",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .env(
        "PATH",
        format!("{}:{}", host_dir.display(), std::env::var("PATH").unwrap()),
    );
    if let Some(dir) = state_home {
        // a genuinely cold state root: the guard dirs must come into
        // being from nothing, not from warmed probe residue
        cmd.env("XDG_STATE_HOME", dir);
    }
    if let Some(path) = launcher {
        cmd.env("SACCADE_ACTIVATION_LAUNCHER", path);
    }
    for (key, value) in extra_envs {
        cmd.env(key, value);
    }
    let child = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the serving subprocess spawns");
    let pid = child.id();
    // port 0: the bound address publishes beside the db, so parallel subprocesses never collide
    let hint = saccade::paths::address_at(db);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let url = loop {
        match std::fs::read_to_string(&hint) {
            Ok(text) if !text.trim().is_empty() => break text.trim().to_string(),
            _ if std::time::Instant::now() >= deadline => {
                panic!("the serving subprocess never published its address")
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if ureq::get(&format!("{url}/api/v1/version"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .is_ok()
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the serving subprocess never came up"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    (url, pid)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wire_send_activates_a_stopped_peer_through_the_launcher() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("wire-activate");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());

    // the launcher seam: records the launch argv and exits successfully,
    // so the decision is observable without a real host, and the test
    // fails if activation is disabled
    let bin_dir = db.parent().unwrap().join("probe-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let seam = bin_dir.join("activation-launcher");
    let launch_log = db.parent().unwrap().join("launch.log");
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"{}\"\nexit 0\n",
        launch_log.display()
    );
    std::fs::write(&seam, script).unwrap();
    std::fs::set_permissions(&seam, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();

    // the recipient's saved session: a valid header whose workspace
    // exists and holds no live process - the resumable shape
    let workspace = db.parent().unwrap().join("stopped-peer-ws");
    std::fs::create_dir_all(&workspace).unwrap();
    let saved = db.parent().unwrap().join("peer-session.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"id\":\"stopped001\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();

    let cold_state = db.parent().unwrap().join("xdg-state");
    let (url, pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        &bin_dir,
        Some(&seam),
        Some(&cold_state),
        &[],
    );

    // a live sender and the stopped recipient
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "sender0001",
            "session_file": "/sessions/sender0001.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "stopped001", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");

    // the wire's own send face: the stopped peer must be activated by
    // the landed write, through the launcher seam
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 1,
            "body": "wake through the wire",
        }),
    );
    assert_eq!(status, 200, "{body}");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let logged = std::fs::read_to_string(&launch_log).unwrap_or_default();
        if logged.contains("--resume") && logged.contains("i-1") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "activation never reached the launcher; log: {logged:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // cold-state readiness: the attach reply names the guard file, and
    // the probe that activation ran has created it from nothing
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "guardcheck", "session_file": "/sessions/guardcheck.jsonl"}),
    );
    assert_eq!(status, 200, "{body}");
    let named = json_of(&body)["guard"]
        .as_str()
        .expect("the attach names the guard")
        .to_string();
    assert!(!named.is_empty(), "a server with a repo names every guard");
    // attach readiness: the guard's parent exists on a cold state root
    let guard = std::path::Path::new(&named);
    assert!(
        guard.parent().is_some_and(|p| p.is_dir()),
        "the guard's parent materializes on a cold state root: {named}"
    );
    // and a landed write's probe creates the file itself: activation
    // reached the decision on this cold root
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 2,
            "body": "probe the cold guard",
        }),
    );
    assert_eq!(status, 200, "{body}");
    // the activation is fire-and-forget: poll briefly for its probe
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while !guard.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "activation never probed the cold guard: {named}"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // the command ingress activates through the same seam: a SendMessage
    // over /api/v1/command must append a second observed launch
    let before = std::fs::read_to_string(&launch_log)
        .unwrap_or_default()
        .len();
    let (status, body) = post_json(
        &url,
        "/api/v1/command",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "command": {"send_message": {"from": 0, "to": 1, "body": "wake through the command face"}},
        }),
    );
    assert_eq!(status, 200, "{body}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let logged = std::fs::read_to_string(&launch_log).unwrap_or_default();
        if logged.len() > before {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the command ingress never activated; log: {logged:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let _ = std::process::Command::new("kill")
        .arg(pid.to_string())
        .status();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// One owned real-door fixture: the fake host, the guard-holding harness, their logs, and the cold state home.
struct DoorFixture {
    host: PathBuf,
    pi: PathBuf,
    host_log: PathBuf,
    host_state: PathBuf,
    child_log: PathBuf,
    dump: PathBuf,
    pidfile: PathBuf,
    guard_file: PathBuf,
    cold_state: PathBuf,
}

fn door_fixture(home: &Path) -> DoorFixture {
    let host = fake_host(home);
    let pi = guard_pi(home);
    let host_log = home.join("host.log");
    let host_state = home.join("host-state");
    let child_log = home.join("child.log");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();
    std::fs::write(&child_log, b"").unwrap();
    DoorFixture {
        host,
        pi,
        host_log,
        host_state,
        child_log,
        dump: home.join("stub.dump"),
        pidfile: home.join("stub.pid"),
        guard_file: home.join("guard-path"),
        cold_state: home.join("xdg-state"),
    }
}

/// The environment one real-door subprocess runs under: fake host first on PATH, the fixture harness as pinned Pi, the launcher seam disabled, and the cold state home the fixture server named its guard under.
fn door_envs(fx: &DoorFixture, guard: Option<&Path>) -> Vec<(&'static str, String)> {
    let mut envs = vec![
        (
            "PATH",
            format!(
                "{}:{}",
                fx.host.parent().unwrap().display(),
                std::env::var("PATH").unwrap()
            ),
        ),
        ("FAKE_ORCA_LOG", fx.host_log.display().to_string()),
        ("FAKE_ORCA_STATE", fx.host_state.display().to_string()),
        ("FAKE_ORCA_SPAWN", "1".to_string()),
        ("FAKE_ORCA_CHILD_LOG", fx.child_log.display().to_string()),
        ("XDG_STATE_HOME", fx.cold_state.display().to_string()),
        ("SACCADE_PI", fx.pi.display().to_string()),
        ("PI_DUMP", fx.dump.display().to_string()),
        ("PI_PIDFILE", fx.pidfile.display().to_string()),
        ("FIXTURE_LOG", fx.child_log.display().to_string()),
    ];
    if let Some(guard) = guard {
        envs.push(("SACCADE_TEST_GUARD", guard.display().to_string()));
    }
    envs
}

/// Kill the fixture holder by its pidfile and wait for its guard to actually free.
fn stop_holder(pidfile: &Path, guard: &Path) {
    if let Ok(pid) = std::fs::read_to_string(pidfile) {
        let _ = std::process::Command::new("kill")
            .arg("-9")
            .arg(pid.trim())
            .status();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while saccade::activate::guard_held(guard) != Some(false) {
        assert!(
            std::time::Instant::now() < deadline,
            "the fixture holder's guard never freed"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[test]
fn the_real_resume_door_submits_then_observes_the_guard() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("resume-door");
    let home = db.parent().unwrap();
    let (root, _first, _head) = scratch_repo(home);
    let fx = door_fixture(home);
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        None,
        Some(&fx.cold_state),
        &[],
    );

    // the resumable session: a valid saved header, existing workspace, no live process; the reply names the guard
    let workspace = home.join("resumed-ws");
    std::fs::create_dir_all(&workspace).unwrap();
    let saved = home.join("resumee-session.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"resumee01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 0);
    let guard = PathBuf::from(
        json_of(&body)["guard"]
            .as_str()
            .expect("the attach names the guard"),
    );

    // a tampered association: recorded, then the file's identity rewritten under it
    let tampered = home.join("tampered-session.jsonl");
    std::fs::write(
        &tampered,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"other002\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "other002", "session_file": tampered.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    std::fs::write(
        &tampered,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"rewritten01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    // a missing saved file refuses the same way
    let absent = home.join("absent-session.jsonl");
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "missing003", "session_file": absent.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");

    let envs = door_envs(&fx, Some(&guard));
    let door_args = |resume: &str| {
        vec![
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--resume".to_string(),
            resume.to_string(),
            "--json".to_string(),
        ]
    };
    let (ok, out, err, timed_out) = bounded_sac(bin, &door_args("i-1"), &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{out}");
    assert!(err.contains("carries id rewritten01"), "{err}");
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-2"), &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("cannot resume"), "{err}");
    assert!(
        std::fs::read_to_string(&fx.host_log).unwrap().is_empty(),
        "a refused resume must contact no host"
    );

    // a preexisting live holder refuses before a duplicate is submitted
    use std::os::fd::AsRawFd;
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&guard)
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    assert_eq!(saccade::activate::guard_held(&guard), Some(true));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("already held"), "{err}");
    assert!(
        std::fs::read_to_string(&fx.host_log).unwrap().is_empty(),
        "a busy resume must contact no host"
    );
    drop(held);
    assert_eq!(saccade::activate::guard_held(&guard), Some(false));

    // the corrected door: submit through the host, then observe the launched session's guard
    let (ok, out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &envs);
    assert!(!timed_out, "the door outwaited its guard bound: {err}");
    assert!(ok, "{out}{err}");
    let launched = json_of(&out);
    assert_eq!(
        launched["workspace"],
        json!(workspace.display().to_string()),
        "{out}"
    );
    assert!(!launched["terminal"].as_str().unwrap().is_empty(), "{out}");

    // the submission rode the host before the harness ever ran
    let submitted = command_of(&fx.host_log, "");
    assert!(submitted.contains("--session resumee01"), "{submitted}");
    assert!(submitted.contains("--session-dir"), "{submitted}");
    // the terminal title carries the canonical public token, once
    let logged = std::fs::read_to_string(&fx.host_log).unwrap();
    assert!(logged.contains("--title\nsac resume i-0\n"), "{logged}");
    assert!(!logged.contains("i-i-"), "{logged}");

    // one append-ordered witness: the host started the command, then the harness took the real guard
    let witness = std::fs::read_to_string(&fx.child_log).unwrap();
    let started = witness
        .find("submitted-command-started")
        .unwrap_or_else(|| panic!("{witness}"));
    let taken = witness
        .find("guard-taken")
        .unwrap_or_else(|| panic!("{witness}"));
    assert!(started < taken, "{witness}");

    // kernel evidence: held while the session lives, freed at its death
    assert_eq!(saccade::activate::guard_held(&guard), Some(true));
    let dump = std::fs::read_to_string(&fx.dump).unwrap();
    assert!(dump.contains("--session resumee01"), "{dump}");
    assert!(dump.contains("--session-dir"), "{dump}");
    assert!(
        dump.contains(&format!("pwd: {}", workspace.display())),
        "{dump}"
    );
    assert!(dump.contains(&format!("SACCADE_SERVER={url}")), "{dump}");
    assert!(
        dump.contains(&format!("SACCADE_REPO={}", root.display())),
        "{dump}"
    );
    assert!(dump.contains("SACCADE_ACTOR="), "{dump}");
    assert!(
        !dump.contains("SACCADE_TASK="),
        "stale scope must not ride the launch: {dump}"
    );
    assert!(!dump.contains("SACCADE_INCARNATION_SEQ="), "{dump}");
    stop_holder(&fx.pidfile, &guard);

    let _ = std::process::Command::new("kill")
        .arg(server_pid.to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[test]
fn switch_resume_runs_through_the_real_door() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("switch-resume-door");
    let home = db.parent().unwrap();
    let (root, _first, _head) = scratch_repo(home);
    let fx = door_fixture(home);
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        None,
        Some(&fx.cold_state),
        &[],
    );

    // two explicit existing directories; the stopped rover parked at B
    let selected_a = home.join("selected-a");
    let selected_b = home.join("selected-b");
    std::fs::create_dir_all(&selected_a).unwrap();
    std::fs::create_dir_all(&selected_b).unwrap();
    let saved = home.join("rover-session.jsonl");
    let tail = "{\"type\":\"message\",\"id\":\"aa\",\"parentId\":null,\"timestamp\":\"t\",\"message\":{\"role\":\"user\",\"content\":\"parked at the old workspace\"}}\n";
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"rover0001\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n{tail}",
            selected_b.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "rover0001", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 0);
    let guard = PathBuf::from(
        json_of(&body)["guard"]
            .as_str()
            .expect("the attach names the guard"),
    );

    // the composed door: the change lands, then the same native id resumes through the real submission
    let envs = door_envs(&fx, Some(&guard));
    let (ok, out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server",
            &url,
            "--repo",
            root.to_str().unwrap(),
            "workspace",
            "switch",
            "i-0",
            "--to",
            selected_a.to_str().unwrap(),
            "--resume",
            "--",
            "carry the thread forward",
        ],
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    assert!(out.contains("workspace changed"), "{out}");
    assert!(out.contains("integrated pi launching"), "{out}");
    assert!(out.contains(selected_a.to_str().unwrap()), "{out}");

    // the header moved whole: same native id, same history, new cwd
    let moved = std::fs::read(&saved).unwrap();
    let header: Value =
        serde_json::from_slice(moved.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(header["cwd"], json!(selected_a.display().to_string()));
    assert_eq!(header["id"], "rover0001");
    assert!(moved.ends_with(tail.as_bytes()), "the body rode whole");

    // the change record landed
    let changes_of = |rows: &[Value]| -> Vec<Value> {
        rows.iter()
            .filter(|r| r["kind"] == "workspace_changed")
            .cloned()
            .collect()
    };
    let (_, body) = get_json(&url, "/api/v1/records");
    let changes = changes_of(json_of(&body)["records"].as_array().unwrap());
    assert_eq!(changes.len(), 1);
    assert_eq!(
        changes[0]["payload"]["worktree"],
        json!(selected_a.display().to_string())
    );

    // the resume is the same session, running in the new directory
    let witness = std::fs::read_to_string(&fx.child_log).unwrap();
    let started = witness
        .find("submitted-command-started")
        .unwrap_or_else(|| panic!("{witness}"));
    let taken = witness
        .find("guard-taken")
        .unwrap_or_else(|| panic!("{witness}"));
    assert!(started < taken, "{witness}");
    assert_eq!(saccade::activate::guard_held(&guard), Some(true));
    let dump = std::fs::read_to_string(&fx.dump).unwrap();
    assert!(dump.contains("--session rover0001"), "{dump}");
    assert!(dump.contains("carry the thread forward"), "{dump}");
    assert!(
        dump.contains(&format!("pwd: {}", selected_a.display())),
        "{dump}"
    );
    stop_holder(&fx.pidfile, &guard);

    // a failed launch: the host refuses the create and the standing change fact is disclosed
    let mut fail_envs = envs.clone();
    fail_envs.push(("FAKE_ORCA_CREATE_FAILS", "1".to_string()));
    let switch_args = |to: &Path, json: bool| {
        let mut args = vec![
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "workspace".to_string(),
            "switch".to_string(),
            "i-0".to_string(),
            "--to".to_string(),
            to.to_str().unwrap().to_string(),
            "--resume".to_string(),
        ];
        if json {
            args.push("--json".to_string());
        }
        args.push("--".to_string());
        args.push("carry it back".to_string());
        args
    };
    let (ok, _out, err, timed_out) = bounded_sac(bin, &switch_args(&selected_b, false), &fail_envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "a failed launch must not exit success");
    assert!(err.contains("orca terminal create failed"), "{err}");
    assert!(err.contains("workspace changed"), "{err}");
    assert!(err.contains(selected_b.to_str().unwrap()), "{err}");
    // the change is true: the header really moved and the record landed
    let moved = std::fs::read(&saved).unwrap();
    let header: Value =
        serde_json::from_slice(moved.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(header["cwd"], json!(selected_b.display().to_string()));
    assert_eq!(header["id"], "rover0001");
    let (_, body) = get_json(&url, "/api/v1/records");
    let changes = changes_of(json_of(&body)["records"].as_array().unwrap());
    assert_eq!(changes.len(), 2);
    assert_eq!(
        changes[1]["payload"]["worktree"],
        json!(selected_b.display().to_string())
    );
    // nothing ran: the guard never changed hands
    assert_eq!(saccade::activate::guard_held(&guard), Some(false));
    let witness = std::fs::read_to_string(&fx.child_log).unwrap();
    assert_eq!(witness.matches("guard-taken").count(), 1, "{witness}");

    // the json face: a valid envelope carrying the standing fact
    let (ok, _out, err, timed_out) = bounded_sac(bin, &switch_args(&selected_a, true), &fail_envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    let envelope: Value = serde_json::from_str(err.trim()).expect("a valid json error envelope");
    assert_eq!(envelope["error"], "usage");
    assert!(
        envelope["detail"]
            .as_str()
            .unwrap()
            .contains("workspace changed"),
        "{envelope}"
    );
    let moved = std::fs::read(&saved).unwrap();
    let header: Value =
        serde_json::from_slice(moved.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(header["cwd"], json!(selected_a.display().to_string()));

    let _ = std::process::Command::new("kill")
        .arg(server_pid.to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[test]
fn a_send_activates_through_the_real_door() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("send-real-door");
    let home = db.parent().unwrap();
    let (root, _first, _head) = scratch_repo(home);
    let fx = door_fixture(home);

    // the server's own env carries the launch: no launcher seam, the guard path rides the fixture's pointer file
    std::fs::write(&fx.guard_file, b"").unwrap();
    let server_envs = [
        ("FAKE_ORCA_LOG", fx.host_log.to_str().unwrap()),
        ("FAKE_ORCA_STATE", fx.host_state.to_str().unwrap()),
        ("FAKE_ORCA_SPAWN", "1"),
        ("FAKE_ORCA_CHILD_LOG", fx.child_log.to_str().unwrap()),
        ("FAKE_ORCA_GUARD_FILE", fx.guard_file.to_str().unwrap()),
        ("SACCADE_PI", fx.pi.to_str().unwrap()),
        ("PI_DUMP", fx.dump.to_str().unwrap()),
        ("PI_PIDFILE", fx.pidfile.to_str().unwrap()),
        ("FIXTURE_LOG", fx.child_log.to_str().unwrap()),
    ];
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        None,
        Some(&fx.cold_state),
        &server_envs,
    );

    // a live sender and the stopped peer
    let workspace = home.join("peer-ws");
    std::fs::create_dir_all(&workspace).unwrap();
    let saved = home.join("peer001-session.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"peer001\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "sender001", "session_file": "/sessions/sender001.jsonl"}),
    );
    assert_eq!(status, 200, "{body}");
    let sender_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "peer001", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 1);
    let peer_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let guard = PathBuf::from(
        json_of(&body)["guard"]
            .as_str()
            .expect("the attach names the guard"),
    );
    std::fs::write(&fx.guard_file, guard.to_str().unwrap()).unwrap();

    // the landed write drives the real door: submission first, then the guard
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": sender_birth, "workspace": null}},
            "to": 1,
            "body": "wake through the real door",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let witness = loop {
        let witness = std::fs::read_to_string(&fx.child_log).unwrap_or_default();
        if witness.contains("submitted-command-started") && witness.contains("guard-taken") {
            break witness;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the send never drove the real door; witness: {witness:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    let started = witness.find("submitted-command-started").unwrap();
    let taken = witness.find("guard-taken").unwrap();
    assert!(started < taken, "{witness}");
    let submitted = command_of(&fx.host_log, "");
    assert!(submitted.contains("--session peer001"), "{submitted}");
    assert_eq!(saccade::activate::guard_held(&guard), Some(true));
    let dump = std::fs::read_to_string(&fx.dump).unwrap();
    assert!(dump.contains("--session peer001"), "{dump}");
    assert!(
        dump.contains(&format!("pwd: {}", workspace.display())),
        "{dump}"
    );
    assert!(dump.contains(&format!("SACCADE_SERVER={url}")), "{dump}");

    // delivery facts stay honest: submitted, not read
    let (_, body) = get_json(
        &url,
        &format!("/api/v1/messages/pending?incarnation={peer_birth}"),
    );
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert!(
        mail.iter()
            .any(|m| m["from"]["num"] == 0 && m["state"] == "pending"),
        "{body}"
    );

    stop_holder(&fx.pidfile, &guard);
    let _ = std::process::Command::new("kill")
        .arg(server_pid.to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_send_door_activates_and_the_ask_door_introduces() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("activation-cli");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();
    let host_path = format!(
        "{}:{}",
        host.parent().unwrap().display(),
        std::env::var("PATH").unwrap()
    );

    // the creator registers directly; the ask door launches the peer
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "creator0001",
            "session_file": "/sessions/creator0001.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 0);

    let workspace = db.parent().unwrap().join("peer-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let ask_out = db.parent().unwrap().join("ask.out");
    let ask_err = db.parent().unwrap().join("ask.err");
    let launched = std::thread::spawn({
        let bin = bin.to_string();
        let base_url = base_url.clone();
        let root = root.clone();
        let host_path = host_path.clone();
        let host_log = host_log.display().to_string();
        let host_state = host_state.display().to_string();
        let pi = pi.display().to_string();
        let workspace = workspace.display().to_string();
        let ask_out = ask_out.display().to_string();
        let ask_err = ask_err.display().to_string();
        move || {
            let out = std::fs::File::create(&ask_out).unwrap();
            let err = std::fs::File::create(&ask_err).unwrap();
            std::process::Command::new(&bin)
                .args([
                    "--server",
                    &base_url,
                    "--repo",
                    root.to_str().unwrap(),
                    "launch",
                    "--workspace",
                    &workspace,
                    "--ask",
                    "what holds at the failing edge case?",
                ])
                .env("PATH", &host_path)
                .env("FAKE_ORCA_LOG", &host_log)
                .env("FAKE_ORCA_STATE", &host_state)
                .env("SACCADE_PI", &pi)
                .env("SACCADE_ACTOR", "creator")
                .env("SACCADE_INCARNATION_SEQ", "0")
                .env("SACCADE_TASK", "")
                .stdout(std::process::Stdio::from(out))
                .stderr(std::process::Stdio::from(err))
                .status()
                .expect("spawn sac")
        }
    });

    // the launch line carries the correlation token (the fake host
    // logged it); the session's attach claims it, and the door learns
    // the birth
    let mut token = String::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while token.is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "the launch line never carried a token; log: {:?}; state: {:?}; sac stderr: {:?}; sac stdout: {:?}",
            std::fs::read_to_string(&host_log).unwrap_or_default(),
            std::fs::read_to_string(&host_state).unwrap_or_default(),
            std::fs::read_to_string(&ask_err).unwrap_or_default(),
            std::fs::read_to_string(&ask_out).unwrap_or_default(),
        );
        if let Ok(text) = std::fs::read_to_string(&host_log) {
            for line in text.lines() {
                if let Some(at) = line.find("SACCADE_LAUNCH_TOKEN=") {
                    let raw = &line[at + "SACCADE_LAUNCH_TOKEN=".len()..];
                    let trimmed = raw.trim_matches('\'');
                    let end = trimmed.find(' ').unwrap_or(trimmed.len());
                    token = trimmed[..end].to_string();
                }
            }
        }
        if token.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "peer00000001",
            "session_file": "/sessions/peer00000001.jsonl",
            "launch_token": token,
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 1);

    let status = launched.join().unwrap();
    let (out_text, err) = (
        std::fs::read_to_string(&ask_out).unwrap_or_default(),
        std::fs::read_to_string(&ask_err).unwrap_or_default(),
    );
    assert!(status.success(), "stdout: {out_text}\nstderr: {err}");
    assert!(out_text.contains("i-1 registered"), "{out_text}");
    assert!(
        out_text.contains("your question went as m-0 from i-0"),
        "{out_text}"
    );

    // the question is durable addressed communication from the
    // creator's own scope, never argv-shaped human input
    let (_, body) = get_json(&base_url, "/api/v1/messages/pending?incarnation=1");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 1, "{body}");
    assert_eq!(mail[0]["from"]["num"], 0);
    assert_eq!(mail[0]["to"]["num"], 1);
    assert_eq!(mail[0]["state"], "pending");
    assert_eq!(mail[0]["body"], "what holds at the failing edge case?");

    // a server-routed send delegates activation to the wire handler -
    // one mechanism for every entrance - so the door prints the plain
    // pending line and the wire's own decision runs server-side
    let out = std::process::Command::new(bin)
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "send",
            "--to",
            "i-1",
            "status check",
        ])
        .env("PATH", &host_path)
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", "0")
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.contains("sent m-"), "{text}");
    assert!(!text.contains("activation"), "{text}");

    // without a declared incarnation the ask door refuses, taught
    let out = std::process::Command::new(bin)
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--workspace",
            workspace.to_str().unwrap(),
            "--ask",
            "nobody sent this",
        ])
        .env("PATH", &host_path)
        .env("FAKE_ORCA_LOG", &host_log)
        .env("FAKE_ORCA_STATE", &host_state)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", "")
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("SACCADE_INCARNATION_SEQ"), "{err}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The typed command the host was asked to run, from the per-line log.
fn command_of(log: &std::path::Path, _selector: &str) -> String {
    let logged = std::fs::read_to_string(log).unwrap();
    let args: Vec<&str> = logged.lines().collect();
    let create = args
        .windows(2)
        .position(|w| w == ["terminal", "create"])
        .expect("a terminal create was logged");
    let command = args[create..]
        .iter()
        .position(|a| *a == "--command")
        .expect("the create carries a command");
    args[create + command + 1].to_string()
}

fn command_label(program: &str) -> String {
    std::path::Path::new(program)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap()
}

fn shell_quoted(word: &str) -> String {
    if !word.is_empty()
        && word.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'=' | b'/' | b'.' | b'+')
        })
    {
        return word.to_string();
    }
    let mut out = String::from("'");
    for c in word.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

// ---- the guard's terminal arms: a real pty, piped prose ----

/// Open a pty; the child's stdout and stderr attach to its slave.
fn open_pty() -> (i32, std::path::PathBuf) {
    let master = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
    assert!(master >= 0, "posix_openpt");
    assert_eq!(unsafe { libc::grantpt(master) }, 0);
    assert_eq!(unsafe { libc::unlockpt(master) }, 0);
    let mut buf = [0u8; 64];
    assert_eq!(
        unsafe { libc::ptsname_r(master, buf.as_mut_ptr().cast(), buf.len()) },
        0
    );
    let name = std::ffi::CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    (master, std::path::PathBuf::from(name))
}

use std::os::unix::ffi::OsStrExt;

/// A pty slave as spawnable stdio; the master stays open for the terminal read.
fn pty_stdio(masters: &mut Vec<i32>) -> std::process::Stdio {
    use std::os::unix::io::FromRawFd;
    let (master, slave) = open_pty();
    let fd = unsafe { libc::open(slave.as_os_str().as_bytes().as_ptr().cast(), libc::O_RDWR) };
    assert!(fd >= 0, "open the pty slave");
    masters.push(master);
    std::process::Stdio::from(unsafe { std::fs::File::from_raw_fd(fd) })
}

/// Run sac with piped stdin and stdout/stderr each a pipe or a pty slave.
fn run_with_terminals(
    bin: &str,
    args: &[String],
    stdout_tty: bool,
    stderr_tty: bool,
) -> std::process::Output {
    use std::io::Write;
    let mut masters = Vec::new();
    let mut cmd = std::process::Command::new(bin);
    cmd.args(args)
        .env_remove("SACCADE_ACTOR")
        .env_remove("SACCADE_TASK")
        .env_remove("SACCADE_INCARNATION_SEQ")
        .env_remove("SACCADE_WORKSPACE")
        .stdin(std::process::Stdio::piped())
        .stdout(if stdout_tty {
            pty_stdio(&mut masters)
        } else {
            std::process::Stdio::piped()
        })
        .stderr(if stderr_tty {
            pty_stdio(&mut masters)
        } else {
            std::process::Stdio::piped()
        });
    let mut child = cmd.spawn().expect("spawn sac");
    child.stdin.take().unwrap().write_all(b"").unwrap();
    let out = child.wait_with_output().expect("sac answers");
    for master in masters {
        unsafe { libc::close(master) };
    }
    out
}

/// The guard's terminal arms: a terminal anywhere proceeds, nothing refuses.
#[test]
fn piped_human_prose_passes_and_unattended_refuses() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("ws-guard");

    // the task exists; possession (an agent) seeds it offline
    let seed = std::process::Command::new(bin)
        .arg("--db")
        .arg(&db)
        .env("SACCADE_ACTOR", "pi")
        .args(["create", "task", "guarded writes"])
        .output()
        .expect("spawn sac");
    assert!(
        seed.status.success(),
        "seed: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    let args = |sub: &[&str]| {
        let mut full = vec!["--db".to_string(), db.display().to_string()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };

    // piped prose, terminal stdout and stderr: the human writes
    let out = run_with_terminals(
        bin,
        &args(&["comment", "t-0", "piped prose from a human shell"]),
        true,
        true,
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // the record landed, human-authored
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert_eq!(world.tasks[0].thread.len(), 1);
    assert_eq!(world.tasks[0].thread.len(), 1);

    // no terminal anywhere and no SACCADE_ACTOR: refusal, taught
    let out = run_with_terminals(
        bin,
        &args(&["comment", "t-0", "no terminal in sight"]),
        false,
        false,
    );
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no SACCADE_ACTOR and no terminal"), "{err}");
    assert!(err.contains("export SACCADE_ACTOR"), "{err}");

    // and nothing landed: the thread holds the one piped note
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert_eq!(world.tasks[0].thread.len(), 1);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

// ---- the destination contract through the CLI face ----

/// A real `sac serve` subprocess; the reply is the URL it published.
fn spawn_serve(
    bin: &str,
    db: &std::path::Path,
    extra: &[&str],
    envs: &[(&str, &str)],
) -> (std::process::Child, String) {
    let mut child = std::process::Command::new(bin)
        .args(["serve", "--port", "0", "--db"])
        .arg(db)
        .args(extra)
        .envs(envs.iter().copied())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn serve");
    let hint = saccade::paths::address_at(db);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match std::fs::read_to_string(&hint) {
            Ok(text) if !text.trim().is_empty() => {
                return (child, text.trim().to_string());
            }
            _ if child.try_wait().unwrap().is_some() => {
                panic!("serve exited without publishing {}", hint.display());
            }
            _ if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("serve never published {}", hint.display());
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

/// A runner free of every inherited SACCADE_* value.
fn scrubbed(mut cmd: std::process::Command) -> std::process::Command {
    for key in [
        "SACCADE_ACTOR",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_TASK",
        "SACCADE_INCARNATION",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "SACCADE_REPO",
    ] {
        cmd.env_remove(key);
    }
    cmd
}

fn task_names(list_json: &str) -> Vec<String> {
    let value: Value = serde_json::from_str(list_json).expect("the list face is json");
    value["tasks"]
        .as_array()
        .expect("the list face names its tasks")
        .iter()
        .map(|t| t["name"].as_str().expect("a task names itself").to_string())
        .collect()
}

#[test]
fn explicit_destinations_route_reads_and_writes_to_one_tracker() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-dest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (db_a, db_b, db_c) = (dir.join("a.db"), dir.join("b.db"), dir.join("c.db"));
    let (mut serve_a, url_a) = spawn_serve(bin, &db_a, &[], &[]);
    let (mut serve_b, url_b) = spawn_serve(bin, &db_b, &[], &[]);
    let (mut serve_c, url_c) = spawn_serve(bin, &db_c, &[], &[]);

    let sac = |args: &[&str], envs: &[(&str, &str)]| {
        let mut cmd = scrubbed(std::process::Command::new(bin));
        cmd.args(args);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // the scratch create lands in the selected db, never the exported ones
    let (ok, _, err) = sac(
        &[
            "--db",
            db_a.to_str().unwrap(),
            "--json",
            "create",
            "task",
            "implement foo",
        ],
        &[
            ("SACCADE_DB", db_b.to_str().unwrap()),
            ("SACCADE_SERVER", &url_c),
            ("SACCADE_ACTOR", "pi"),
        ],
    );
    assert!(ok, "{err}");
    let (ok, out, err) = sac(&["--server", &url_b, "--json", "list"], &[]);
    assert!(ok, "{err}");
    assert!(
        task_names(&out).is_empty(),
        "the exported db stayed empty: {out}"
    );
    let (ok, out, err) = sac(&["--server", &url_c, "--json", "list"], &[]);
    assert!(ok, "{err}");
    assert!(
        task_names(&out).is_empty(),
        "the exported server stayed empty: {out}"
    );
    let (ok, out, err) = sac(&["--server", &url_a, "--json", "list"], &[]);
    assert!(ok, "{err}");
    assert_eq!(task_names(&out), vec!["implement foo"]);

    // the explicit server beats the exported db
    let (ok, _, err) = sac(
        &[
            "--server",
            &url_b,
            "--json",
            "create",
            "task",
            "implement foo",
        ],
        &[
            ("SACCADE_DB", db_c.to_str().unwrap()),
            ("SACCADE_ACTOR", "pi"),
        ],
    );
    assert!(ok, "{err}");
    let (ok, out, err) = sac(&["--server", &url_b, "--json", "show", "t-0"], &[]);
    assert!(ok, "{err}");
    assert!(out.contains("implement foo"), "{out}");
    let (ok, out, err) = sac(&["--server", &url_c, "--json", "list"], &[]);
    assert!(ok, "{err}");
    assert!(
        task_names(&out).is_empty(),
        "the exported db never grew: {out}"
    );

    // the server read agrees with the local read of the same file
    let (ok, served, err) = sac(&["--server", &url_a, "--json", "list"], &[]);
    assert!(ok, "{err}");
    let (ok, local, err) = sac(&["--db", db_a.to_str().unwrap(), "--json", "list"], &[]);
    assert!(ok, "{err}");
    assert_eq!(task_names(&served), task_names(&local));

    // a local write rides the selected file; the server read sees it
    let (ok, _, err) = sac(
        &[
            "--db",
            db_a.to_str().unwrap(),
            "comment",
            "t-0",
            "a local note",
        ],
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(ok, "{err}");
    let (ok, out, err) = sac(&["--server", &url_a, "--json", "show", "t-0"], &[]);
    assert!(ok, "{err}");
    assert!(out.contains("a local note"), "{out}");

    // both explicit destinations refuse, on the json face too
    let (ok, _, err) = sac(
        &["--db", db_a.to_str().unwrap(), "--server", &url_a, "list"],
        &[],
    );
    assert!(!ok);
    assert!(err.contains("both explicit destinations"), "{err}");
    let (ok, out, err) = sac(
        &[
            "--db",
            db_a.to_str().unwrap(),
            "--server",
            &url_a,
            "--json",
            "list",
        ],
        &[],
    );
    assert!(!ok);
    assert!(err.contains("\"error\":\"usage\""), "{out}{err}");

    // the server-only launch refuses local mode
    let (ok, _, err) = sac(
        &[
            "--db",
            db_a.to_str().unwrap(),
            "launch",
            "--workspace",
            "/tmp",
        ],
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(!ok);
    assert!(err.contains("runs through the server"), "{err}");

    // serve refuses --server in any combination
    let (ok, _, err) = sac(&["--server", &url_a, "serve", "--port", "0"], &[]);
    assert!(!ok, "serve with --server must refuse");
    assert!(err.contains("serve starts a server"), "{err}");
    let (ok, _, err) = sac(
        &[
            "--db",
            db_a.to_str().unwrap(),
            "--server",
            &url_a,
            "serve",
            "--port",
            "0",
        ],
        &[],
    );
    assert!(!ok, "serve with both flags must refuse");
    assert!(err.contains("serve starts a server"), "{err}");

    for serve in [&mut serve_a, &mut serve_b, &mut serve_c] {
        let _ = serve.kill();
        let _ = serve.wait();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn discovery_resolves_the_repos_server_and_refuses_the_broken_ways() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-disc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (root, _, _) = scratch_repo(&dir);
    let state = dir.join("state");
    let empty_state = dir.join("empty-state");
    std::fs::create_dir_all(&empty_state).unwrap();

    let sac_in = |cwd: &std::path::Path, args: &[&str], state_home: &std::path::Path| {
        let out = scrubbed(std::process::Command::new(bin))
            .args(args)
            .env("XDG_STATE_HOME", state_home)
            .env("SACCADE_ACTOR", "pi")
            .current_dir(cwd)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // no server ever started: discovery refuses, nothing implicit
    let (ok, _, err) = sac_in(&root, &["--json", "list"], &empty_state);
    assert!(!ok);
    assert!(err.contains("server_undiscovered"), "{err}");
    let (ok, out, err) = sac_in(&root, &["--json", "identity"], &empty_state);
    assert!(ok, "identity is a read: {out}{err}");
    assert!(out.contains("unresolved"), "{out}");

    // the repo-derived serve publishes beside its db; the client finds it
    let mut serve_cmd = scrubbed(std::process::Command::new(bin));
    serve_cmd
        .args(["serve", "--port", "0"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .env("XDG_STATE_HOME", &state)
        .env("SACCADE_ACTOR", "pi")
        .current_dir(&root);
    let mut serve = serve_cmd.spawn().expect("spawn serve");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    // the hint is the one saccade.db.addr under the child's state root
    let hint = loop {
        let mut found = Vec::new();
        if let Ok(slugs) = std::fs::read_dir(state.join("saccade")) {
            for slug in slugs.flatten() {
                let candidate = slug.path().join("saccade.db.addr");
                if candidate.is_file() {
                    found.push(candidate);
                }
            }
        }
        if found.len() == 1 {
            break found.remove(0);
        }
        if std::time::Instant::now() >= deadline {
            let _ = serve.kill();
            panic!("serve never published a hint under {}", state.display());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let published = std::fs::read_to_string(&hint).unwrap();
    let published = published.trim().to_string();
    let (ok, _, err) = sac_in(&root, &["--json", "list"], &state);
    assert!(ok, "{err}");
    let (ok, out, err) = sac_in(&root, &["--json", "identity"], &state);
    assert!(ok, "{err}");
    assert!(out.contains("discovered-server"), "{out}");
    assert!(out.contains(&published), "{out}");
    assert!(published.starts_with("http://127.0.0.1:"), "{published}");

    // a worktree resolves the same common dir, so the same server
    let wt = dir.join("wt");
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "side"])
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "worktree add: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (ok, _, err) = sac_in(&wt, &["--json", "list"], &state);
    assert!(ok, "worktree discovery: {err}");

    // a scratch serve never overwrites the repo's hint; wildcard publishes loopback
    let scratch_db = dir.join("scratch").join("saccade.db");
    let (mut scratch_serve, scratch_url) =
        spawn_serve(bin, &scratch_db, &["--bind", "0.0.0.0"], &[]);
    assert!(
        scratch_url.starts_with("http://127.0.0.1:"),
        "{scratch_url}"
    );
    assert_ne!(scratch_url, published, "the scratch port is its own");
    assert_eq!(
        std::fs::read_to_string(&hint).unwrap().trim(),
        published,
        "the repo's hint stands"
    );

    // the hint replaced on restart: it names the server actually serving
    let _ = scratch_serve.kill();
    let _ = scratch_serve.wait();
    std::fs::remove_file(saccade::paths::address_at(&scratch_db)).unwrap();
    let (mut scratch_serve, republished) = spawn_serve(bin, &scratch_db, &[], &[]);
    let (ok, _, err) = sac_in(&root, &["--server", &republished, "--json", "log"], &state);
    assert!(ok, "the republished hint answers: {err}");
    assert_eq!(
        std::fs::read_to_string(&hint).unwrap().trim(),
        published,
        "the scratch restart never touched the repo's hint"
    );

    // a stale hint pointing at a dead port refuses, no local fallback
    std::fs::write(&hint, "http://127.0.0.1:9\n").unwrap();
    let (ok, _, err) = sac_in(&root, &["--json", "list"], &state);
    assert!(!ok);
    assert!(err.contains("server_required"), "{err}");

    // a mismatched hint pointing at another tracker refuses
    std::fs::write(&hint, format!("{republished}\n")).unwrap();
    let (ok, _, err) = sac_in(&root, &["--json", "list"], &state);
    assert!(!ok);
    assert!(err.contains("server_mismatch"), "{err}");

    let _ = scratch_serve.kill();
    let _ = scratch_serve.wait();
    let _ = serve.kill();
    let _ = serve.wait();
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The serving repo is the declared one, never the cwd.
#[tokio::test(flavor = "multi_thread")]
async fn the_declared_repo_carries_the_serving_effects() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-declared-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (root, _, _) = scratch_repo(&dir.join("a"));
    let (other, _, _) = scratch_repo(&dir.join("b"));
    let db = dir.join("served.db");
    let state_home = dir.join("state");
    std::fs::create_dir_all(&state_home).unwrap();
    let host = fake_host(&dir);
    let pi = stub_pi(&dir);
    std::fs::write(dir.join("host.log"), b"").unwrap();
    std::fs::write(dir.join("host-state"), b"").unwrap();

    // repo-derived serve from outside any repo has no tracker to serve
    let mut probe_cmd = scrubbed(std::process::Command::new(bin));
    probe_cmd
        .args(["serve", "--port", "0"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .env("XDG_STATE_HOME", &state_home)
        .current_dir(&dir);
    let mut probe = probe_cmd.spawn().expect("spawn serve");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let refused = loop {
        match probe.try_wait().unwrap() {
            Some(status) => break !status.success(),
            None if std::time::Instant::now() >= deadline => {
                let _ = probe.kill();
                panic!("repo-less serve neither refused nor started cleanly");
            }
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    };
    assert!(refused, "repo-less serve must refuse");

    let mut serve = std::process::Command::new(bin)
        .args(["serve", "--port", "0", "--db"])
        .arg(&db)
        .arg("--repo")
        .arg(&root)
        .env("XDG_STATE_HOME", &state_home)
        .current_dir(&dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn serve");
    let hint = saccade::paths::address_at(&db);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let url = loop {
        match std::fs::read_to_string(&hint) {
            Ok(text) if !text.trim().is_empty() => break text.trim().to_string(),
            _ if std::time::Instant::now() >= deadline => {
                let _ = serve.kill();
                panic!("serve never published {}", hint.display());
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    };

    let run = |cwd: &std::path::Path, args: &[String]| {
        let out = scrubbed(std::process::Command::new(bin))
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    host.parent().unwrap().display(),
                    std::env::var("PATH").unwrap()
                ),
            )
            .env("FAKE_ORCA_LOG", dir.join("host.log").display().to_string())
            .env(
                "FAKE_ORCA_STATE",
                dir.join("host-state").display().to_string(),
            )
            .env("SACCADE_PI", &pi)
            .env("XDG_STATE_HOME", &state_home)
            .env("SACCADE_ACTOR", "pi")
            .current_dir(cwd)
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let at_repo = |sub: &[&str]| -> Vec<String> {
        [
            vec![
                "--server".to_string(),
                url.clone(),
                "--repo".to_string(),
                root.display().to_string(),
            ],
            sub.iter().map(|s| s.to_string()).collect(),
        ]
        .concat()
    };

    // provisioning through the server's own Git effects lands the
    // branch and checkout in the declared repo, never the cwd's
    let (ok, _, err) = run(&other, &at_repo(&["workspace", "new", "declared"]));
    assert!(ok, "{err}");
    assert!(
        sh(&root, &["branch", "--list"]).contains("declared-"),
        "the branch is the declared repo's"
    );
    assert!(
        !sh(&other, &["branch", "--list"]).contains("declared-"),
        "the client's cwd repo grew nothing"
    );
    let checkout = db.parent().unwrap().join("worktrees").join(
        sh(
            &root,
            &["for-each-ref", "--format=%(refname:short)", "refs/heads"],
        )
        .lines()
        .find(|b| b.starts_with("declared-"))
        .expect("the declared branch")
        .to_string(),
    );
    assert!(
        checkout.join(".git").exists(),
        "the worktree sits beside the served db"
    );

    let (ok, _, err) = run(
        &other,
        &at_repo(&["--json", "create", "task", "declared work"]),
    );
    assert!(ok, "{err}");
    let note = dir.join("evidence.txt");
    std::fs::write(&note, b"the bytes\n").unwrap();
    let (ok, _, err) = run(
        &other,
        &at_repo(&[
            "artifact",
            "t-0",
            note.to_str().unwrap(),
            "--name",
            "evidence",
        ]),
    );
    assert!(ok, "{err}");
    let mut store_files = Vec::new();
    for slug in std::fs::read_dir(state_home.join("saccade")).unwrap() {
        let store = slug.unwrap().path().join("artifacts");
        if let Ok(files) = std::fs::read_dir(&store) {
            for f in files.flatten() {
                store_files.push(f.path());
            }
        }
    }
    assert_eq!(store_files.len(), 1, "bytes live in the declared store");
    let hash = store_files[0]
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let mut r = ureq::get(&format!("{url}/a/{hash}"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(r.body_mut().read_to_string().unwrap(), "the bytes\n");

    // a second provisioning, same declared repo
    let (ok, _, err) = run(&other, &at_repo(&["workspace", "new", "declared-two"]));
    assert!(ok, "{err}");
    assert!(
        sh(&root, &["branch", "--list"]).contains("declared-two-"),
        "still the declared repo"
    );
    assert!(
        !sh(&other, &["branch", "--list"]).contains("declared-two-"),
        "still nothing in the cwd repo"
    );

    let _ = serve.kill();
    let _ = serve.wait();
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The use door through the CLI face: a prospective birth, an explicit
/// binding, a source replacement that never touches the binding, an
/// explicit rebind, and the guards that hold throughout - every id
/// read from a printed face resolving to the same account it names.
#[test]
fn the_use_door_binds_rebinds_and_holds_its_guards_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("use-door");
    let sac = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
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

    // the board's earlier work: six noncomment births stand before the
    // first comment, so every storage seq sits far from its public num
    assert!(sac(&["create", "task", "produce the account"]).0); // seq 0, t-0
    assert!(sac(&["create", "task", "file the floop index"]).0); // seq 1, t-1
    assert!(sac(&["create", "task", "chart the floop index"]).0); // seq 2, t-2
    assert!(sac(&["create", "task", "migrate floop"]).0); // seq 3, t-3
    assert!(sac(&["claim", "t-1"]).0); // seq 4
    assert!(sac(&["claim", "t-2"]).0); // seq 5
    assert!(sac(&["offer", "t-0", "the first account"]).0); // seq 6 -> c-0
    assert!(sac(&["create", "task", "consume the account"]).0); // seq 7, t-4
    assert!(
        sac(&[
            "use",
            "t-4",
            "--source",
            "t-0",
            "--reason",
            "builds on the producer"
        ])
        .0
    ); // seq 8 -> c-1

    // the standing offer's printed c-N is its dense comment number,
    // never the seq it was born at - and that same token resolves the
    // comment through every door
    let (ok, out, _) = sac(&["show", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("offer  c-0"), "{out}");
    assert!(!out.contains("offer  c-6"), "{out}");
    let (ok, out, _) = sac(&["show", "c-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("the first account"), "{out}");
    // the bare door still shows the raw event at its position
    let (ok, out, _) = sac(&["show", "6"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("seq 6  commented  pi\n"),
        "the offer's birth record rides whole: {out}"
    );

    // the machine faces keep the birth reference; the dense alias
    // rides beside it, explicitly named
    let (ok, out, _) = sac(&["--json", "list"]);
    assert!(ok, "{out}");
    let list: Value = serde_json::from_str(&out).unwrap();
    let row = list["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"] == "t-0")
        .expect("the offered task lists")
        .clone();
    assert_eq!(row["offer"], json!(6), "the list keeps the birth ref");
    let (ok, out, _) = sac(&["--json", "show", "t-0"]);
    assert!(ok, "{out}");
    let shown: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(shown["blocks"][0]["state"], "offered");
    assert_eq!(shown["blocks"][0]["offer"], json!(6));
    assert_eq!(shown["blocks"][0]["offer_num"], json!(0));

    // born prospective: the source named, no selection yet
    let (ok, out, _) = sac(&["show", "t-4"]);
    assert!(ok, "{out}");
    assert!(out.contains("uses t-0"), "{out}");
    assert!(!out.contains("bound c-"), "{out}");

    // the explicit binding selects the standing account; the record
    // stores the original birth references, never the printed tokens
    assert!(sac(&["bind", "c-1", "c-0", "--grounds", "the standing account"]).0); // seq 9
    let (ok, out, _) = sac(&["log"]);
    assert!(ok, "{out}");
    assert!(
        out.contains(r#"{"grounds":"the standing account","id":8,"offer":6}"#),
        "the binding stores birth refs: {out}"
    );
    // every generic row labels its position as provenance: a seq
    // never borrows the c- face only a comment birth earns
    for line in out.lines().filter(|l| !l.is_empty()) {
        assert!(
            line.starts_with("seq ") || line.starts_with("cursor "),
            "a log row borrowed an alias face: {line}"
        );
    }
    let (ok, out, _) = sac(&["show", "t-4"]);
    assert!(ok, "{out}");
    assert!(out.contains("bound c-0"), "{out}");
    assert!(out.contains("the standing account"), "{out}");

    // the source replaces its conclusion: the binding keeps it, and a
    // rebind to the replaced account refuses
    assert!(sac(&["reject", "c-0", "--grounds", "the fix misses the index"]).0); // seq 10 -> c-2
    let (ok, _, err) = sac(&["bind", "c-1", "c-0", "--grounds", "gone"]);
    assert!(!ok, "{err}");
    assert!(err.contains("not_current_offer"), "{err}");
    let (ok, out, _) = sac(&["show", "t-4"]);
    assert!(ok, "{out}");
    assert!(
        out.contains("bound c-0"),
        "the binding kept the replaced account"
    );

    // the explicit rebind selects the replacement (the replacement
    // offer c-3 lands at seq 11); acceptance of the selected
    // conclusion changes nothing
    assert!(sac(&["offer", "t-0", "the replacement"]).0); // seq 11 -> c-3
    assert!(sac(&["bind", "c-1", "c-3", "--grounds", "the replacement held"]).0); // seq 12
    assert!(sac(&["accept", "c-3"]).0); // seq 13 -> c-4
    let (ok, out, _) = sac(&["show", "t-4"]);
    assert!(ok, "{out}");
    assert!(out.contains("bound c-3"), "{out}");
    assert!(!out.contains("bound c-0"), "{out}");

    // the guards: a task cannot rely on itself; a binding selects from
    // the declared source alone
    let (ok, _, err) = sac(&["use", "t-0", "--source", "t-0", "--reason", "myself"]);
    assert!(!ok, "{err}");
    assert!(err.contains("self_reliance"), "{err}");
    assert!(sac(&["create", "task", "an unrelated producer"]).0); // seq 14, t-5
    assert!(sac(&["offer", "t-5", "an outside account"]).0); // seq 15 -> c-5
    let (ok, _, err) = sac(&["bind", "c-1", "c-5", "--grounds", "not my source"]);
    assert!(!ok, "{err}");
    assert!(err.contains("invalid_state_transition"), "{err}");

    // citations: --ref earns the next marker, the reply target cites
    // its parent, and a marker with no reference refuses at the door
    assert!(
        sac(&[
            "comment",
            "t-4",
            "rests on the conclusion [^1]",
            "--ref",
            "c-3"
        ])
        .0
    );
    let (ok, out, _) = sac(&["show", "t-4"]);
    assert!(ok, "{out}");
    assert!(out.contains("->c-3"), "{out}");
    assert!(sac(&["comment", "c-0", "replying to the account"]).0);
    let (ok, out, _) = sac(&["show", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("->c-0"), "the reply cites its parent: {out}");
    let (ok, _, err) = sac(&["comment", "t-4", "dangling [^2]", "--ref", "c-3"]);
    assert!(!ok, "{err}");
    assert!(err.contains("marker [^2]"), "{err}");

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

/// The migration door through the CLI face: a schema-3 source migrates
/// into a fresh schema-4 tracker, the source stays untouched and
/// unreadable to this binary, and the report names every renumbering.
#[test]
fn the_migrate_door_rewrites_a_schema3_tracker() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-migrate-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let v3 = dir.join("v3.db");
    {
        let conn = rusqlite::Connection::open(&v3).unwrap();
        conn.execute_batch(
            "CREATE TABLE events (
                 seq INTEGER PRIMARY KEY,
                 event_time INTEGER NOT NULL,
                 logged_time INTEGER NOT NULL,
                 actor TEXT NOT NULL,
                 tier TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 payload TEXT NOT NULL,
                 scope TEXT NULL
             );
             PRAGMA application_id = 1935762276;
             PRAGMA user_version = 3;",
        )
        .unwrap();
        let insert = |seq: i64, kind: &str, payload: &str| {
            conn.execute(
                "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
                 VALUES (?1, 1000+?1, 2000, 'jerry', 'human', ?2, ?3, NULL)",
                rusqlite::params![seq, kind, payload],
            )
            .unwrap();
        };
        // the story: a note citing comments, a raw birth, and an
        // out-of-range token; an offer with its accept; another task's
        // use relying on that offer
        insert(0, "task_created", r#"{"name":"implement foo"}"#);
        insert(
            1,
            "commented",
            r#"{"target":{"task":0},"kind":{"note":{"body":"per c-1, c-2, and c-0, also c-9"}}}"#,
        );
        insert(
            2,
            "commented",
            r#"{"target":{"task":0},"kind":{"offer":{"body":"suite green, see c-1"}}}"#,
        );
        insert(
            3,
            "commented",
            r#"{"target":{"comment":2},"kind":{"accept":{"offer":2}}}"#,
        );
        insert(4, "task_created", r#"{"name":"adopt foo's account"}"#);
        insert(
            5,
            "commented",
            r#"{"target":{"task":1},"kind":{"use":{"offer":2,"reason":"builds on foo, per c-1"}}}"#,
        );
    }
    let out_db = dir.join("v4.db");
    let migrated = std::process::Command::new(bin)
        .arg("--db")
        .arg(&v3)
        .args(["migrate", "--out"])
        .arg(&out_db)
        .env("SACCADE_ACTOR", "pi")
        .output()
        .expect("spawn sac migrate");
    assert!(
        migrated.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&migrated.stderr)
    );
    let report = String::from_utf8_lossy(&migrated.stdout).into_owned();
    assert!(report.contains("migrated 6 records"), "{report}");
    assert!(report.contains("1 uses born bound"), "{report}");
    assert!(
        report.contains("1 unresolved references stay legacy"),
        "{report}"
    );
    assert!(report.contains("c-9"), "{report}");
    // the diagnostic row names the record's position as provenance,
    // the stored token it reports stays verbatim
    assert!(report.contains("seq 1 body: c-9"), "{report}");

    // the output reads: prose renumbered onto the dense numbering, the
    // use born bound, the reply rehomed with an empty mapping
    let show = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&out_db)
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
    let (ok, out, _) = show(&["show", "t-0"]);
    assert!(ok, "{out}");
    // dense numbers: the note (seq 1) is c-0, the offer (seq 2) c-1,
    // the task birth rides as seq 0, and c-9 stays legacy
    assert!(
        out.contains("per c-0, c-1, and seq 0, also legacy c-9"),
        "{out}"
    );
    // the consumer's thread: the use born bound to the historical offer
    let (ok, out, _) = show(&["show", "t-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("uses t-0"), "{out}");
    assert!(out.contains("bound c-1"), "{out}");
    // the mapping stayed empty: no graph edges were invented
    assert!(!out.contains("->"), "old comments cite nothing: {out}");

    // seqs and timestamps preserved: the log rides whole
    let (ok, out, _) = show(&["log", "--json"]);
    assert!(ok, "{out}");
    let log: Value = serde_json::from_str(&out).unwrap();
    let rows = log["records"].as_array().unwrap();
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[0]["seq"], json!(0));
    assert_eq!(rows[0]["event_time"], json!(1000));

    // the source stays schema 3 and refuses to open under this binary
    let read = std::process::Command::new(bin)
        .arg("--db")
        .arg(&v3)
        .args(["list"])
        .env("SACCADE_ACTOR", "pi")
        .output()
        .expect("spawn sac");
    assert!(!read.status.success());
    assert!(
        String::from_utf8_lossy(&read.stderr).contains("predates this binary"),
        "{}",
        String::from_utf8_lossy(&read.stderr)
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

/// The task-free controller over real git, wire, and CLI: creation
/// births a workspace identity alone, the stopped-session change
/// proves quiescence before it moves a byte, and the explicit-new
/// choice rides the ordinary launch doors.
#[tokio::test(flavor = "multi_thread")]
async fn the_task_free_controller_creates_and_switches_stopped_sessions() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("ws-controller");
    let home = db.parent().unwrap();
    let (root, _first, head) = scratch_repo(home);
    let base_url = spawn_workspace_server(&db, &root).await;

    // two explicit existing directories, neither created by saccade
    let selected_a = home.join("selected-a");
    let selected_b = home.join("selected-b");
    std::fs::create_dir_all(&selected_a).unwrap();
    std::fs::create_dir_all(&selected_b).unwrap();

    // creation through the wire: actual Git facts, the birth record
    // carrying them, the record's position the only identity
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/create",
        &json!({"label": "expedition"}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["created"], true);
    assert!(
        reply["workspace"]["branch"]
            .as_str()
            .unwrap()
            .starts_with("expedition-"),
        "{reply}"
    );
    assert_eq!(reply["workspace"]["base"], head.as_str());
    assert!(
        reply["workspace"]["path"]
            .as_str()
            .unwrap()
            .ends_with(&format!(
                "expedition-{}",
                reply["workspace"]["branch"]
                    .as_str()
                    .unwrap()
                    .strip_prefix("expedition-")
                    .unwrap()
            ))
    );
    let birth = reply["workspace"]["seq"].as_u64().unwrap();
    let created = reply["records"].as_array().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["kind"], "workspace_created");
    assert_eq!(created[0]["actor"], "saccade");
    assert_eq!(created[0]["tier"], "system");
    // the branch is real: the checkout exists and is a worktree
    assert!(
        std::path::Path::new(reply["workspace"]["path"].as_str().unwrap())
            .join(".git")
            .exists()
    );

    // a second creation with no label: the ratified fallback name, a
    // distinct branch and birth
    let (status, body) = post_workspace(&base_url, "/api/v1/workspace/create", &json!({}));
    assert_eq!(status, 200, "{body}");
    let second = json_of(&body);
    assert!(
        second["workspace"]["branch"]
            .as_str()
            .unwrap()
            .starts_with("workspace-"),
        "{second}"
    );
    assert_ne!(second["workspace"]["branch"], reply["workspace"]["branch"]);
    assert!(second["workspace"]["seq"].as_u64().unwrap() > birth);

    // creation birthed no task, claim, message, or incarnation: the
    // log holds exactly the two creations
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert_eq!(row["kind"], "workspace_created", "{row}");
        let payload = &row["payload"];
        assert!(payload.get("task_id").is_none(), "{row}");
        assert!(payload.get("incarnation").is_none(), "{row}");
    }

    // creation advanced no i-N: the first native session is i-0
    let saved = home.join("rover-session.jsonl");
    let header = format!(
        "{{\"type\":\"session\",\"version\":3,\"id\":\"rover0001\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n",
        selected_b.display()
    );
    let tail = "{\"type\":\"message\",\"id\":\"aa\",\"parentId\":null,\"timestamp\":\"t\",\"message\":{\"role\":\"user\",\"content\":\"parked at the old workspace\"}}\n";
    std::fs::write(&saved, format!("{header}{tail}")).unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "rover0001", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    let rover = json_of(&body);
    assert_eq!(rover["num"], 0);
    let rover_birth = rover["incarnation"].as_u64().unwrap();
    let guard =
        std::path::PathBuf::from(rover["guard"].as_str().expect("the server names the guard"));
    assert!(guard.parent().unwrap().is_dir());

    // the stopped change: guard free, association exact, the record
    // lands only after the header actually moved
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 0, "to": selected_a.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    let changed = json_of(&body);
    assert_eq!(changed["changed"], true);
    assert_eq!(
        changed["from"],
        json!(selected_b.display().to_string()),
        "{changed}"
    );
    assert_eq!(changed["to"], json!(selected_a.display().to_string()));
    assert_eq!(changed["num"], 0);
    let change_records = changed["records"].as_array().unwrap();
    assert_eq!(change_records.len(), 1);
    assert_eq!(change_records[0]["kind"], "workspace_changed");
    assert_eq!(change_records[0]["actor"], "saccade");
    assert_eq!(change_records[0]["tier"], "system");
    // the saved session moved: header cwd is the selected directory,
    // the body is byte-identical, the id and history unchanged
    let moved = std::fs::read(&saved).unwrap();
    let moved_header: Value =
        serde_json::from_slice(moved.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(moved_header["cwd"], selected_a.display().to_string());
    assert_eq!(moved_header["id"], "rover0001");
    assert!(moved.ends_with(tail.as_bytes()), "the body rode whole");
    // the change record landed after the header success: two creations,
    // the registration, the change
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[3]["kind"], "workspace_changed");

    // a held guard is a live process: busy refuses before any mutation
    use std::os::fd::AsRawFd;
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&guard)
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 0, "to": selected_b.to_str().unwrap()}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "session_live");
    let untouched = std::fs::read(&saved).unwrap();
    let untouched_header: Value =
        serde_json::from_slice(untouched.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(untouched_header["cwd"], selected_a.display().to_string());
    drop(held);

    // an unprobeable guard is unknown: refuse, move nothing
    std::fs::remove_file(&guard).unwrap();
    std::fs::create_dir(&guard).unwrap();
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 0, "to": selected_b.to_str().unwrap()}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "session_quiescence_unknown"
    );
    std::fs::remove_dir(&guard).unwrap();

    // with the guard released the same change lands
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 0, "to": selected_b.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        json_of(&body)["from"],
        json!(selected_a.display().to_string())
    );

    // a missing target refuses before anything moves; the recorded
    // association stands
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let before = json_of(&body)["records"].as_array().unwrap().len();
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 0, "to": home.join("nowhere").to_str().unwrap()}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "workspace_path_missing");
    // an unknown incarnation refuses
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 99, "to": selected_a.to_str().unwrap()}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_incarnation_id");
    // a session whose saved locator names no file refuses
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "ghost0001", "session_file": "/sessions/ghost0001.jsonl"}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["num"], 1);
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 1, "to": selected_a.to_str().unwrap()}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "saved_session_missing");
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"].as_array().unwrap().len(),
        before + 1,
        "only the registration wrote past the changes"
    );

    // the retired task-bound effect routes are absent: a 404 that
    // writes nothing - the task-bound establishment is birthable by no
    // public face, its fold contracts pinned in the store and
    // round-trip inventories
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let standing = json_of(&body)["records"].as_array().unwrap().len();
    for path in ["/api/v1/workspace/start", "/api/v1/workspace/open"] {
        let (status, body) = post_workspace(
            &base_url,
            path,
            &json!({"context": {"actor": "jerry", "tier": "human"}, "task": 0}),
        );
        assert_eq!(status, 404, "{path}: {body}");
    }
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"].as_array().unwrap().len(),
        standing,
        "the absent routes wrote nothing"
    );

    // the CLI face: creation and switch through a real subprocess
    let sac = |args: &[&str], envs: &[(&str, &str)]| {
        let out = std::process::Command::new(bin)
            .args(["--server", &base_url, "--repo", root.to_str().unwrap()])
            .args(args)
            .env("SACCADE_DB", "")
            .env("SACCADE_SERVER", "")
            .env("SACCADE_TASK", "")
            .env("SACCADE_WORKSPACE", "")
            .env("SACCADE_LAUNCH_TOKEN", "")
            .env("SACCADE_ACTIVATION_LAUNCHER", "")
            .envs(envs.iter().copied())
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    let (ok, out, err) = sac(&["workspace", "new", "cli-built", "--json"], &[]);
    assert!(ok, "{out}{err}");
    let built = json_of(&out);
    assert_eq!(built["created"], true);
    assert!(
        built["workspace"]["branch"]
            .as_str()
            .unwrap()
            .starts_with("cli-built-"),
        "{out}"
    );

    let (ok, out, err) = sac(
        &[
            "workspace",
            "switch",
            "i-0",
            "--to",
            selected_a.to_str().unwrap(),
            "--json",
        ],
        &[],
    );
    assert!(ok, "{out}{err}");
    let switched = json_of(&out);
    assert_eq!(switched["changed"], true);
    assert_eq!(switched["from"], json!(selected_b.display().to_string()));
    assert_eq!(switched["to"], json!(selected_a.display().to_string()));
    assert_eq!(switched["records"][0]["kind"], "workspace_changed");
    // the text face names the move and the ordinary resume door
    let (ok, out, err) = sac(
        &[
            "workspace",
            "switch",
            "i-0",
            "--to",
            selected_b.to_str().unwrap(),
        ],
        &[],
    );
    assert!(ok, "{out}{err}");
    assert!(out.contains("workspace changed - saved cwd"), "{out}");
    assert!(out.contains("sac launch --resume i-0"), "{out}");

    // the explicit-new choice rides the ordinary fresh-launch door:
    // provision, then launch, one composition
    let host = fake_host(home);
    let host_log = home.join("host.log");
    let host_state = home.join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();
    let host_path = format!(
        "{}:{}",
        host.parent().unwrap().display(),
        std::env::var("PATH").unwrap()
    );
    let pi = stub_pi(home);

    let (ok, out, err) = sac(
        &["launch", "--new", "trailblaze", "--json"],
        &[
            ("PATH", host_path.as_str()),
            ("FAKE_ORCA_LOG", host_log.to_str().unwrap()),
            ("FAKE_ORCA_STATE", host_state.to_str().unwrap()),
            ("SACCADE_PI", pi.to_str().unwrap()),
        ],
    );
    assert!(ok, "{out}{err}");
    let launched = json_of(&out);
    assert!(
        launched["provisioned"]["branch"]
            .as_str()
            .unwrap()
            .starts_with("trailblaze-"),
        "{out}"
    );
    assert_eq!(
        launched["launch"]["workspace"], launched["provisioned"]["path"],
        "{out}"
    );
    assert!(!launched["launch"]["terminal"].as_str().unwrap().is_empty());

    // a failed launch over the standing provision: the birth pointer
    // and path ride the failure face, never a success claim
    let (ok, out, err) = sac(
        &["launch", "--new", "stumble", "--json"],
        &[
            ("PATH", host_path.as_str()),
            ("FAKE_ORCA_LOG", host_log.to_str().unwrap()),
            ("FAKE_ORCA_STATE", host_state.to_str().unwrap()),
            ("FAKE_ORCA_CREATE_FAILS", "1"),
            ("SACCADE_PI", pi.to_str().unwrap()),
        ],
    );
    assert!(!ok, "{out}{err}");
    let envelope: Value = serde_json::from_str(err.trim()).expect("a valid json error envelope");
    assert_eq!(envelope["error"], "usage");
    let detail = envelope["detail"].as_str().unwrap();
    assert!(
        detail.contains("workspace provisioned - record"),
        "{detail}"
    );
    assert!(detail.contains("stumble-"), "{detail}");
    assert!(detail.contains("the launch failed"), "{detail}");
    // the creation fact is true and the only record: provisioned, not
    // launched, and nothing else written
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    let stumbled: Vec<&Value> = rows
        .iter()
        .filter(|r| {
            r["kind"] == "workspace_created"
                && r["payload"]["branch"]
                    .as_str()
                    .unwrap()
                    .starts_with("stumble-")
        })
        .collect();
    assert_eq!(stumbled.len(), 1, "{body}");

    // the ask composition rides the same birth: a registered creator
    // questions the newborn peer. The newborn claims its launch token
    // by attaching, as the integration itself would
    let ask_out = home.join("ask.out");
    let ask_err = home.join("ask.err");
    let ask_cli = std::thread::spawn({
        let base_url = base_url.clone();
        let root = root.clone();
        let host_path = host_path.clone();
        let host_log = host_log.clone();
        let host_state = host_state.clone();
        let pi = pi.clone();
        let ask_out = ask_out.clone();
        let ask_err = ask_err.clone();
        move || {
            std::process::Command::new(bin)
                .args([
                    "--server",
                    &base_url,
                    "--repo",
                    root.to_str().unwrap(),
                    "launch",
                    "--new",
                    "inquiry",
                    "--ask",
                    "what holds at the edge?",
                    "--json",
                ])
                .env("PATH", &host_path)
                .env("FAKE_ORCA_LOG", &host_log)
                .env("FAKE_ORCA_STATE", &host_state)
                .env("SACCADE_PI", &pi)
                .env("SACCADE_ACTOR", "rover")
                .env("SACCADE_INCARNATION_SEQ", rover_birth.to_string())
                .env("SACCADE_TASK", "")
                .env("SACCADE_WORKSPACE", "")
                .env("SACCADE_LAUNCH_TOKEN", "")
                .env("SACCADE_ACTIVATION_LAUNCHER", "")
                .stdout(std::fs::File::create(&ask_out).unwrap())
                .stderr(std::fs::File::create(&ask_err).unwrap())
                .status()
                .expect("spawn sac")
        }
    });
    let mut token = String::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while token.is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "the launch line never carried a token; log: {:?}",
            std::fs::read_to_string(&host_log).unwrap_or_default()
        );
        if let Ok(text) = std::fs::read_to_string(&host_log) {
            for line in text.lines() {
                if let Some(at) = line.find("SACCADE_LAUNCH_TOKEN=") {
                    let raw = &line[at + "SACCADE_LAUNCH_TOKEN=".len()..];
                    let trimmed = raw.trim_matches('\'');
                    let end = trimmed.find(' ').unwrap_or(trimmed.len());
                    token = trimmed[..end].to_string();
                }
            }
        }
        if token.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "inquiry0001",
            "session_file": "/sessions/inquiry0001.jsonl",
            "launch_token": token,
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert!(ask_cli.join().unwrap().success());
    let asked = json_of(&std::fs::read_to_string(&ask_out).unwrap());
    assert!(
        asked["provisioned"]["branch"]
            .as_str()
            .unwrap()
            .starts_with("inquiry-"),
        "{asked}"
    );
    let newborn = asked["launch"]["incarnation"]["num"].as_u64().unwrap();
    // the i-N counted births only: rover i-0, ghost i-1, the newborn
    // i-2 - creation never counted
    assert_eq!(newborn, 2, "{asked}");
    assert!(asked["launch"]["question"]["seq"].is_u64(), "{asked}");
    // the question is durable addressed mail from the creator's scope
    let (_, body) = get_json(
        &base_url,
        &format!("/api/v1/messages/pending?incarnation={rover_birth}"),
    );
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    let sent: Vec<&Value> = mail.iter().filter(|m| m["from"]["num"] == 0).collect();
    assert_eq!(sent.len(), 1, "{body}");
    assert_eq!(sent[0]["to"]["num"], newborn);

    // a resume request cannot outrun the guard: the session is stopped
    // (the change landed), so --resume composes onto a live refusal -
    // busy is decided before any launch, and the resume door's own
    // handoff contract rides the activation suite unchanged
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&guard)
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let (ok, out, err) = sac(
        &[
            "workspace",
            "switch",
            "i-0",
            "--to",
            selected_a.to_str().unwrap(),
            "--resume",
            "--",
            "continue at the new workspace",
        ],
        &[],
    );
    assert!(!ok, "{out}{err}");
    assert!(err.contains("session_live"), "{err}");
    // the refused change moved nothing: the recorded target stands
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    let changed: Vec<&Value> = rows
        .iter()
        .filter(|r| r["kind"] == "workspace_changed")
        .collect();
    let last = changed.last().unwrap()["payload"]["worktree"]
        .as_str()
        .unwrap();
    assert_eq!(last, selected_b.display().to_string());

    std::fs::remove_dir_all(home).unwrap();
}
