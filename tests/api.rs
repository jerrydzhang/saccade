//! The /api/v1 surface over a real loopback socket: one envelope, records
//! back or a typed error, the tier law holding through HTTP exactly as it
//! holds in process.

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

#[tokio::test(flavor = "multi_thread")]
async fn start_establishes_through_the_servers_git_effects() {
    let db = scratch_db("ws-start");
    let (root, _first, head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "migrate the floop guard"}}),
        ),
    );

    // first start: the establishment alone - no request, no ownership
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/start",
        &json!({
            "context": {"actor": "jerry", "tier": "human"},
            "task": 0,
            "base": null
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["established"], true);
    assert_eq!(reply["recreated"], false);
    assert_eq!(reply["workspace"]["branch"], "saccade/t-0");
    assert_eq!(reply["workspace"]["base"], head.as_str());
    let records = reply["records"].as_array().expect("one record");
    let kinds: Vec<&str> = records
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["workspace_established"]);
    let actors: Vec<&str> = records
        .iter()
        .map(|r| r["actor"].as_str().unwrap())
        .collect();
    assert_eq!(actors, ["saccade"]);
    let tiers: Vec<&str> = records
        .iter()
        .map(|r| r["tier"].as_str().unwrap())
        .collect();
    assert_eq!(tiers, ["system"]);
    let birth = records[0]["seq"].as_u64().unwrap() as usize;
    assert_eq!(reply["incarnation"].as_u64().unwrap(), birth as u64);

    // the checkout exists at the tracker's worktree home, on the branch
    let worktree = db.parent().unwrap().join("worktrees").join("t-0");
    assert!(worktree.join("readme").is_file());
    assert_eq!(
        sh(&worktree, &["rev-parse", "--abbrev-ref", "HEAD"]),
        "saccade/t-0"
    );

    // the establishment births the historical attribution identity;
    // no ownership transfer and no thread publication rode it
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert!(world.tasks[0].ownership.is_empty());
    assert!(world.tasks[0].thread.is_empty());

    // repeat start: nothing lands, nothing launches - the association governs
    std::fs::write(worktree.join("dirty.txt"), "uncommitted\n").unwrap();
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/start",
        &json!({
            "context": {"actor": "jerry", "tier": "human"},
            "task": 0
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["established"], false);
    assert_eq!(reply["records"].as_array().unwrap().len(), 0);
    assert_eq!(saccade::db::load(&conn).unwrap().rows.len(), 2);
    assert_eq!(
        std::fs::read_to_string(worktree.join("dirty.txt")).unwrap(),
        "uncommitted\n"
    );

    // a missing checkout is recreated from the branch tip, identity unchanged
    sh(&worktree, &["add", "dirty.txt"]);
    sh(&worktree, &["commit", "-qm", "advancement"]);
    let tip = sh(&worktree, &["rev-parse", "HEAD"]);
    std::fs::remove_dir_all(&worktree).unwrap();
    let (status, body) = post_workspace(&base_url, "/api/v1/workspace/open", &json!({"task": 0}));
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["recreated"], true);
    assert_eq!(reply["incarnation"].as_u64().unwrap(), birth as u64);
    assert_eq!(sh(&worktree, &["rev-parse", "HEAD"]), tip);
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert!(world.tasks[0].ownership.is_empty());

    // a vanished branch is a named refusal, never invented recovery
    std::fs::remove_dir_all(&worktree).unwrap();
    sh(&root, &["worktree", "prune"]);
    sh(&root, &["branch", "-qD", "saccade/t-0"]);
    let (status, body) = post_workspace(&base_url, "/api/v1/workspace/open", &json!({"task": 0}));
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "workspace_branch_missing");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_branch_scheme_and_base_are_effects_not_claims() {
    let db = scratch_db("ws-base");
    let (root, first, head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;
    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "stack work"}}),
        ),
    );

    // a taken branch refuses, records nothing, and the attempt keeps the request
    sh(&root, &["branch", "saccade/t-0"]);
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/start",
        &json!({
            "context": {"actor": "jerry", "tier": "human"},
            "task": 0,
            "client": "test-runner"
        }),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "workspace_branch_exists");
    let conn = saccade::db::open_read(&db).unwrap();
    assert_eq!(saccade::db::load(&conn).unwrap().rows.len(), 1);
    let lines = attempts_lines(&db);
    let refused = lines.last().expect("the refusal left its line");
    assert_eq!(refused["outcome"], "refused");
    assert_eq!(refused["code"], "workspace_branch_exists");
    assert_eq!(refused["client"], "test-runner");
    assert!(
        refused["request"]
            .as_str()
            .is_some_and(|r| r.contains("\"task\": 0")),
        "{refused}"
    );
    sh(&root, &["branch", "-qD", "saccade/t-0"]);

    // an explicit base names the stacked commit
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/start",
        &json!({
            "context": {"actor": "jerry", "tier": "human"},
            "task": 0,
            "base": "HEAD~1"
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["workspace"]["base"], first.as_str());

    // an unknown task refuses before any effect
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/start",
        &json!({
            "context": {"actor": "jerry", "tier": "human"},
            "task": 9
        }),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_task_id");
    let _ = head;

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_tears_the_workspace_down_and_keeps_the_lineage() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("clean-door");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;
    post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "retire the floop"}}),
        ),
    );
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/start",
        &json!({
            "context": {"actor": "jerry", "tier": "human"},
            "task": 0,
            "request": "retire the floop, then the workspace"
        }),
    );
    assert_eq!(status, 200, "{body}");
    let birth = json_of(&body)["incarnation"].as_u64().unwrap() as usize;
    let worktree = db.parent().unwrap().join("worktrees").join("t-0");
    assert!(worktree.join("readme").is_file());

    let sac = |args: &[&str]| {
        let mut cmd = std::process::Command::new(bin);
        cmd.arg("--db")
            .arg(&db)
            .arg("--repo")
            .arg(&root)
            .arg("--json");
        for var in [
            "SACCADE_DB",
            "SACCADE_REPO",
            "SACCADE_ACTOR",
            "SACCADE_TASK",
            "SACCADE_INCARNATION_SEQ",
            "SACCADE_WORKSPACE",
            "SACCADE_SERVER",
            "SACCADE_PI",
        ] {
            cmd.env_remove(var);
        }
        let out = cmd.args(args).output().expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // untracked work refuses the teardown, nothing destructive runs
    std::fs::write(worktree.join("stray.txt"), "untracked\n").unwrap();
    let (ok, out, err) = sac(&["clean", "t-0"]);
    assert!(!ok, "{out}{err}");
    let refused: Value = serde_json::from_str(&err).expect("the refusal is typed json");
    assert_eq!(refused["error"], "workspace_dirty");
    assert!(worktree.join("readme").is_file());

    // staged work refuses the same way
    sh(&worktree, &["add", "stray.txt"]);
    let (ok, _, err) = sac(&["clean", "t-0"]);
    assert!(!ok);
    assert!(err.contains("workspace_dirty"), "{err}");
    assert!(worktree.join("stray.txt").is_file());

    // committed, the teardown runs: checkout gone, branch and unmerged tip stand
    std::fs::remove_file(worktree.join("stray.txt")).unwrap();
    sh(&worktree, &["reset", "-q"]);
    std::fs::write(worktree.join("readme"), "the project, retired\n").unwrap();
    sh(&worktree, &["add", "readme"]);
    sh(
        &worktree,
        &["commit", "-qm", "retirement work, never merged"],
    );
    let tip = sh(&worktree, &["rev-parse", "HEAD"]);
    let (ok, out, err) = sac(&["clean", "t-0"]);
    assert!(ok, "{err}");
    let reply: Value = serde_json::from_str(&out).expect("the report is json");
    assert_eq!(reply["checkout"], "removed");
    assert_eq!(reply["branch"], "saccade/t-0");
    assert_eq!(reply["tip"], tip);
    assert!(!worktree.exists());
    assert_eq!(sh(&root, &["rev-parse", "refs/heads/saccade/t-0"]), tip);

    // a second clean succeeds on absence
    let (ok, out, err) = sac(&["clean", "t-0"]);
    assert!(ok, "{err}");
    let reply: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(reply["checkout"], "absent");
    assert_eq!(reply["tip"], tip);

    // the retained association reopens the checkout on the same identity
    let (status, body) = post_workspace(&base_url, "/api/v1/workspace/open", &json!({"task": 0}));
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["recreated"], true);
    assert_eq!(reply["incarnation"].as_u64().unwrap(), birth as u64);
    assert_eq!(reply["workspace"]["path"], worktree.display().to_string());
    assert_eq!(sh(&worktree, &["rev-parse", "HEAD"]), tip);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
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
    // the retired '#' term learns the c- form instead of matching words
    let (ok, out, _) = sac(&["search", "c-99999"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (c-99999)\ncursor 5");
    let (ok, _, err) = sac(&["search", "#99999"]);
    assert!(!ok, "{err}");
    assert!(
        err.contains("drop the '#': the reference search is c-99999"),
        "{err}"
    );
    let (ok, out, _) = sac(&["search", "by:nobody"]);
    assert!(ok, "{out}");
    assert_eq!(out.trim_end(), "no matches (by:nobody)\ncursor 5");

    // the anchor window reads the log's own rows: c-1 is the dense
    // number of the offer at seq 4
    let (ok, out, _) = sac(&["search", "c-1", "-C", "1"]);
    assert!(ok, "{out}");
    assert!(out.contains("c-3  task_claimed  pi/agent"), "{out}");
    assert!(
        out.contains("c-4  commented  pi/agent  floop landed"),
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
        out.starts_with("c-5  commented  pi\n"),
        "the raw event rides whole: {out}"
    );
    assert!(out.contains("floop guards the door"), "{out}");
    // the retired '#' id refuses, teaching the bare doors
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
        out.starts_with("c-6  task_created  pi\n"),
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
        out.starts_with("c-3  task_claimed  pi\n"),
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
        err.contains("'bogus' is not an id (expected c-<n>, t-<n>, or a bare <seq>)"),
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

    // the text face: matched record lines, then the cursor for the next wait
    let (ok, out, _) = sac(&["wait", "t-0", "--after", "5", "--until", "activity"]);
    assert!(ok, "{out}");
    assert!(out.contains("c-6"), "{out}");
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
  printf '{"handle":"%s","worktreePath":"%s","orphaned":false}\n' "$handle" "$path" >> "$FAKE_ORCA_STATE"
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

#[tokio::test(flavor = "multi_thread")]
async fn start_open_exec_and_identity_through_the_cli_face() {
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

    // NixOS carries no /bin/echo: the binaries resolve from this PATH
    let on_path = |name: &str| {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("command -v {name}"))
            .output()
            .expect("sh runs");
        assert!(out.status.success(), "no {name} on PATH");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let (echo, env_bin, pwd_bin) = (on_path("echo"), on_path("env"), on_path("pwd"));

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

    // the task exists before any start: creation is never implicit
    assert!(
        run(
            &at_repo(routed(&["create", "task", "land the workspace doors"])),
            &[("SACCADE_ACTOR", "pi")]
        )
        .0
    );

    // first start: records land, one terminal carries the harness
    let (ok, out, err) = run(
        &at_repo(routed(&["start", "t-0", "please fix it's 'the' flake now"])),
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("no request or ownership recorded"), "{out}");
    assert!(!out.contains("ownership names"), "{out}");
    assert!(
        out.contains("term_actual_shape_1"),
        "the host's handle: {out}"
    );
    assert!(
        out.contains("harness command submitted (execution unobserved)"),
        "{out}"
    );
    let worktree = db.parent().unwrap().join("worktrees").join("t-0");
    let typed = command_of(&host_log, &format!("path:{}", worktree.display()));
    assert!(
        typed.starts_with(&format!(
            "SACCADE_ACTOR='pi dump' direnv exec {} ",
            worktree.display()
        )),
        "the assignment leads, the dev environment wraps the command: {typed}"
    );
    assert!(
        typed.contains(&format!(
            "exec t-0 -- {}",
            shell_quoted(&pi.display().to_string())
        )),
        "{typed}"
    );
    assert!(
        typed.contains(&format!("--server {}", shell_quoted(&base_url))),
        "the explicit destination rides the typed line: {typed}"
    );
    assert!(
        typed.contains(&format!("--repo {}", root.display())),
        "the repo rides as identity, never db routing: {typed}"
    );
    assert!(!typed.contains("--db"), "no db routing: {typed}");
    assert!(
        typed.ends_with(&shell_quoted("please fix it's 'the' flake now")),
        "the request survives verbatim: {typed}"
    );

    // the typed command runs through the worktree's authorized dev-env layer
    let allowed = std::process::Command::new("direnv")
        .arg("allow")
        .arg(&worktree)
        .output()
        .expect("direnv runs");
    assert!(
        allowed.status.success(),
        "direnv allow: {}",
        String::from_utf8_lossy(&allowed.stderr)
    );
    let dump = db.parent().unwrap().join("pi-dump-run");
    let fresh = std::process::Command::new("bash")
        .arg("-c")
        .arg(&typed)
        .env("PATH", std::env::var("PATH").unwrap())
        .env("PI_DUMP", dump.display().to_string())
        .env_remove("SACCADE_ACTOR")
        .env_remove("SACCADE_TASK")
        .env_remove("SACCADE_INCARNATION_SEQ")
        .env_remove("SACCADE_WORKSPACE")
        .env_remove("SACCADE_REPO")
        .env_remove("SACCADE_DB")
        .env_remove("SACCADE_SERVER")
        .current_dir(&root)
        .output()
        .expect("the typed command runs");
    assert!(
        fresh.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&fresh.stderr)
    );
    let injected = std::fs::read_to_string(&dump).unwrap();
    let line = |key: &str| {
        injected
            .lines()
            .find(|l| l.starts_with(&format!("{key}=")))
            .unwrap_or_else(|| panic!("{key} missing: {injected}"))
            .to_string()
    };
    assert!(
        injected
            .lines()
            .any(|l| l == "argv: please fix it's 'the' flake now"),
        "{injected}"
    );
    assert!(
        injected
            .lines()
            .any(|l| l == format!("pwd: {}", worktree.display())),
        "{injected}"
    );
    assert_eq!(line("SACCADE_ACTOR"), "SACCADE_ACTOR=pi dump");
    assert_eq!(line("SACCADE_ENVRC_LAYER"), "SACCADE_ENVRC_LAYER=envrc");
    assert_eq!(line("SACCADE_TASK"), "SACCADE_TASK=t-0");
    assert_eq!(line("SACCADE_INCARNATION_SEQ"), "SACCADE_INCARNATION_SEQ=1");
    assert_eq!(
        line("SACCADE_WORKSPACE"),
        format!("SACCADE_WORKSPACE={}", worktree.display())
    );
    assert!(
        !injected.lines().any(|l| l.starts_with("SACCADE_DB=")),
        "routing never rides the injection: {injected}"
    );
    assert!(
        !injected.lines().any(|l| l.starts_with("SACCADE_SERVER=")),
        "routing never rides the injection: {injected}"
    );

    // repeat start: the live terminal is revealed, nothing new created
    std::fs::write(&host_log, b"").unwrap();
    let (ok, out, err) = run(
        &at_repo(routed(&["start", "t-0", "again"])),
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("no duplicate"), "{out}");
    assert!(out.contains("term_actual_shape_1"), "{out}");
    assert!(out.contains("revealed"), "{out}");
    let asked = std::fs::read_to_string(&host_log).unwrap();
    assert!(
        asked.contains("terminal\nswitch\n--terminal\nterm_actual_shape_1"),
        "{asked}"
    );
    assert!(!asked.contains("create"), "{asked}");

    // an unconfirmed navigation reports the terminal it found, never a reveal
    std::fs::write(&host_log, b"").unwrap();
    let (ok, out, err) = run(
        &routed(&["open", "t-0"]),
        &[("SACCADE_ACTOR", "pi"), ("FAKE_ORCA_NAVIGATE", "0")],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("term_actual_shape_1"), "{out}");
    assert!(out.contains("would not bring it forward"), "{out}");
    assert!(!out.contains("revealed"), "{out}");
    assert!(!out.contains("no live terminal"), "{out}");
    let asked = std::fs::read_to_string(&host_log).unwrap();
    assert!(
        asked.contains("terminal\nswitch\n--terminal\nterm_actual_shape_1\n--json"),
        "{asked}"
    );
    assert!(!asked.contains("create"), "{asked}");

    // a standing workspace with no live terminal is reported, not grown
    std::fs::write(&host_state, b"").unwrap();
    std::fs::write(&host_log, b"").unwrap();
    let (ok, out, err) = run(&routed(&["open", "t-0"]), &[("SACCADE_ACTOR", "pi")]);
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("no live terminal"), "{out}");
    let asked = std::fs::read_to_string(&host_log).unwrap();
    assert!(!asked.contains("create"), "{asked}");

    // a fresh open births the association and opens one shell terminal
    assert!(
        run(
            &at_repo(routed(&["create", "task", "opened not started"])),
            &[("SACCADE_ACTOR", "pi")]
        )
        .0
    );
    let (ok, out, err) = run(&routed(&["open", "t-1"]), &[("SACCADE_ACTOR", "pi")]);
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("workspace established"), "{out}");
    assert!(out.contains("no request or ownership recorded"), "{out}");
    assert!(out.contains("term_actual_shape"), "{out}");
    assert!(out.contains("| shell"), "{out}");
    let asked = std::fs::read_to_string(&host_log).unwrap();
    assert!(asked.contains("create"), "{asked}");
    // and the records show the birth alone
    let conn = saccade::db::open_read(&db).unwrap();
    let loadout = saccade::db::load(&conn).unwrap();
    assert_eq!(loadout.rows.last().unwrap().kind, "workspace_established");

    // exec: argv boundaries, working directory, stale scope overwritten
    let (ok, out, err) = run(
        &routed(&["exec", "t-0", "--", &echo, "two  spaces", "x y"]),
        &[
            ("SACCADE_ACTOR", "pi"),
            ("SACCADE_TASK", "t-99"),
            ("SACCADE_INCARNATION_SEQ", "999"),
            ("SACCADE_WORKSPACE", "/nowhere"),
        ],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert_eq!(out.trim_end(), "two  spaces x y");

    let env_run = |actor: Option<&str>| {
        let mut envs: Vec<(&str, &str)> = vec![
            ("SACCADE_TASK", "t-99"),
            ("SACCADE_INCARNATION_SEQ", "999"),
            ("SACCADE_WORKSPACE", "/nowhere"),
            ("SACCADE_DB", "/stale/routing.db"),
            ("SACCADE_SERVER", "http://stale:1"),
        ];
        if let Some(actor) = actor {
            envs.push(("SACCADE_ACTOR", actor));
        }
        run(&routed(&["exec", "t-0", "--", &env_bin]), &envs)
    };

    // a declared actor is preserved; an absent one takes the command's label
    let (ok, out, err) = env_run(Some("worker"));
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("SACCADE_ACTOR=worker\n"), "{out}");
    let (ok, out, err) = env_run(None);
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(
        out.contains(&format!("SACCADE_ACTOR={}\n", command_label(&env_bin))),
        "{out}"
    );

    let (ok, out, err) = run(
        &routed(&["exec", "t-0", "--", &env_bin]),
        &[
            ("SACCADE_ACTOR", "pi"),
            ("SACCADE_TASK", "t-99"),
            ("SACCADE_INCARNATION_SEQ", "999"),
            ("SACCADE_WORKSPACE", "/nowhere"),
        ],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    let env_line = |out: &str, key: &str| {
        out.lines()
            .find(|l| l.starts_with(&format!("{key}=")))
            .unwrap_or_else(|| panic!("{key} missing from: {out}"))
            .to_string()
    };
    assert_eq!(env_line(&out, "SACCADE_TASK"), "SACCADE_TASK=t-0");
    assert_eq!(
        env_line(&out, "SACCADE_INCARNATION_SEQ"),
        "SACCADE_INCARNATION_SEQ=1"
    );
    assert_eq!(
        env_line(&out, "SACCADE_WORKSPACE"),
        format!("SACCADE_WORKSPACE={}", worktree.display())
    );
    assert_eq!(
        env_line(&out, "SACCADE_REPO"),
        format!("SACCADE_REPO={}", root.display())
    );
    assert!(
        !out.lines().any(|l| l.starts_with("SACCADE_DB=")),
        "stale routing is cleared, not passed: {out}"
    );
    assert!(
        !out.lines().any(|l| l.starts_with("SACCADE_SERVER=")),
        "stale routing is cleared, not passed: {out}"
    );

    // a relative --db names the same tracker from beside the file
    let out = std::process::Command::new(bin)
        .args([
            "--db",
            "saccade.db",
            "--repo",
            root.to_str().unwrap(),
            "exec",
            "t-0",
            "--",
            &env_bin,
        ])
        .env("PATH", std::env::var("PATH").unwrap())
        .env_remove("SACCADE_ACTOR")
        .current_dir(db.parent().unwrap())
        .output()
        .expect("spawn sac");
    let ok = out.status.success();
    let (out, err) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert_eq!(env_line(&out, "SACCADE_TASK"), "SACCADE_TASK=t-0");

    // the working directory is the checkout
    let (ok, out, err) = run(
        &routed(&["exec", "t-0", "--", &pwd_bin]),
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert_eq!(out.trim_end(), worktree.display().to_string());

    // exec provisions nothing: a task without a workspace refuses
    assert!(
        run(
            &at_repo(routed(&["create", "task", "never started"])),
            &[("SACCADE_ACTOR", "pi")]
        )
        .0
    );
    let (ok, _, err) = run(
        &routed(&["exec", "t-2", "--", &echo]),
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(!ok, "{err}");
    assert!(err.contains("workspace_missing"), "{err}");

    // identity: the environment's declaration, read-only
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

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A fresh integrate launch scrubs stale inherited identity, task,
/// workspace, and db routing the child never selected; the routing it
/// does select rides the launch line.
#[tokio::test(flavor = "multi_thread")]
async fn the_integrate_launch_scrubs_stale_inheritance() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("integrate-scrub");
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
            "integrate",
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
    launcher: &Path,
) -> (String, u32) {
    let port = 8940 + (std::process::id() % 500) as u16;
    let child = std::process::Command::new(bin)
        .args([
            "--db",
            db.to_str().unwrap(),
            "--repo",
            repo.to_str().unwrap(),
            "serve",
            "--bind",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .env(
            "PATH",
            format!("{}:{}", host_dir.display(), std::env::var("PATH").unwrap()),
        )
        // a genuinely cold state root: the guard dirs must come into
        // being from nothing, not from warmed probe residue
        .env("XDG_STATE_HOME", db.parent().unwrap().join("xdg-state"))
        .env("SACCADE_ACTIVATION_LAUNCHER", launcher)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the serving subprocess spawns");
    let pid = child.id();
    let url = format!("http://127.0.0.1:{port}");
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

    let (url, pid) = spawn_serving_subprocess(&db, &root, bin, &bin_dir, &seam);

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
                    "integrate",
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
            "integrate",
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
fn spawn_serve(bin: &str, db: &std::path::Path, extra: &[&str]) -> (std::process::Child, String) {
    let mut child = std::process::Command::new(bin)
        .args(["serve", "--port", "0", "--db"])
        .arg(db)
        .args(extra)
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
    let (mut serve_a, url_a) = spawn_serve(bin, &db_a, &[]);
    let (mut serve_b, url_b) = spawn_serve(bin, &db_b, &[]);
    let (mut serve_c, url_c) = spawn_serve(bin, &db_c, &[]);

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

    // the server-only verbs refuse local mode
    let (ok, _, err) = sac(
        &[
            "--db",
            db_a.to_str().unwrap(),
            "start",
            "t-0",
            "local start",
        ],
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(!ok);
    assert!(err.contains("runs through the server"), "{err}");
    let (ok, _, err) = sac(&["--db", db_a.to_str().unwrap(), "open", "t-0"], &[]);
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
    let (mut scratch_serve, scratch_url) = spawn_serve(bin, &scratch_db, &["--bind", "0.0.0.0"]);
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
    let (mut scratch_serve, republished) = spawn_serve(bin, &scratch_db, &[]);
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

    let (ok, _, err) = run(
        &other,
        &at_repo(&["--json", "create", "task", "declared work"]),
    );
    assert!(ok, "{err}");
    let (ok, out, err) = run(&other, &at_repo(&["start", "t-0", "first request"]));
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(out.contains("workspace established"), "{out}");
    assert!(
        sh(&root, &["branch", "--list", "saccade/t-0"]).contains("saccade/t-0"),
        "the branch is the declared repo's"
    );
    assert!(
        !sh(&other, &["branch", "--list", "saccade/t-0"]).contains("saccade/t-0"),
        "the client's cwd repo grew nothing"
    );
    assert!(
        db.parent().unwrap().join("worktrees").join("t-0").exists(),
        "the worktree sits beside the served db"
    );

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

    let (ok, _, err) = run(
        &other,
        &at_repo(&["--json", "create", "task", "second declared"]),
    );
    assert!(ok, "{err}");
    let (ok, out, err) = run(&other, &at_repo(&["start", "t-1", "second request"]));
    assert!(ok, "stdout: {out}\nstderr: {err}");
    assert!(
        sh(&root, &["branch", "--list", "saccade/t-1"]).contains("saccade/t-1"),
        "still the declared repo"
    );
    assert!(
        !sh(&other, &["branch", "--list", "saccade/t-1"]).contains("saccade/t-1"),
        "still nothing in the cwd repo"
    );

    let _ = serve.kill();
    let _ = serve.wait();
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The use door through the CLI face: a prospective birth, an explicit
/// binding, a source replacement that never touches the binding, an
/// explicit rebind, and the guards that hold throughout.
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

    assert!(sac(&["create", "task", "produce the account"]).0); // seq 0
    assert!(sac(&["offer", "t-0", "the first account"]).0); // seq 1 -> c-0
    assert!(sac(&["create", "task", "consume the account"]).0); // seq 2
    assert!(
        sac(&[
            "use",
            "t-1",
            "--source",
            "t-0",
            "--reason",
            "builds on the producer"
        ])
        .0
    ); // seq 3 -> c-1

    // born prospective: the source named, no selection yet
    let (ok, out, _) = sac(&["show", "t-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("uses t-0"), "{out}");
    assert!(!out.contains("bound c-"), "{out}");

    // the explicit binding selects the standing account
    assert!(sac(&["bind", "c-1", "c-0", "--grounds", "the standing account"]).0);
    let (ok, out, _) = sac(&["show", "t-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("bound c-0"), "{out}");
    assert!(out.contains("the standing account"), "{out}");

    // the source replaces its conclusion: the binding keeps it, and a
    // rebind to the replaced account refuses
    assert!(sac(&["reject", "c-0", "--grounds", "the fix misses the index"]).0);
    let (ok, _, err) = sac(&["bind", "c-1", "c-0", "--grounds", "gone"]);
    assert!(!ok, "{err}");
    assert!(err.contains("not_current_offer"), "{err}");
    let (ok, out, _) = sac(&["show", "t-1"]);
    assert!(ok, "{out}");
    assert!(
        out.contains("bound c-0"),
        "the binding kept the replaced account"
    );

    // the explicit rebind selects the replacement (the reject at seq 5
    // is dense c-2, the replacement offer c-3); acceptance of the
    // selected conclusion changes nothing
    assert!(sac(&["offer", "t-0", "the replacement"]).0); // seq 6 -> c-3
    assert!(sac(&["bind", "c-1", "c-3", "--grounds", "the replacement held"]).0);
    assert!(sac(&["accept", "c-3"]).0);
    let (ok, out, _) = sac(&["show", "t-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("bound c-3"), "{out}");
    assert!(!out.contains("bound c-0"), "{out}");

    // the guards: a task cannot rely on itself; a binding selects from
    // the declared source alone
    let (ok, _, err) = sac(&["use", "t-0", "--source", "t-0", "--reason", "myself"]);
    assert!(!ok, "{err}");
    assert!(err.contains("self_reliance"), "{err}");
    assert!(sac(&["create", "task", "an unrelated producer"]).0); // seq 8
    assert!(sac(&["offer", "t-2", "an outside account"]).0); // seq 9 -> c-5
    let (ok, _, err) = sac(&["bind", "c-1", "c-5", "--grounds", "not my source"]);
    assert!(!ok, "{err}");
    assert!(err.contains("invalid_state_transition"), "{err}");

    // citations: --ref earns the next marker, the reply target cites
    // its parent, and a marker with no reference refuses at the door
    assert!(
        sac(&[
            "comment",
            "t-1",
            "rests on the conclusion [^1]",
            "--ref",
            "c-3"
        ])
        .0
    );
    let (ok, out, _) = sac(&["show", "t-1"]);
    assert!(ok, "{out}");
    assert!(out.contains("->c-3"), "{out}");
    assert!(sac(&["comment", "c-0", "replying to the account"]).0);
    let (ok, out, _) = sac(&["show", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("->c-0"), "the reply cites its parent: {out}");
    let (ok, _, err) = sac(&["comment", "t-1", "dangling [^2]", "--ref", "c-3"]);
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
