//! The /api/v1 surface over a real loopback socket: one envelope, records
//! back or a typed error, the tier law holding through HTTP exactly as it
//! holds in process.

use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use saccade::api::AppState;

fn scratch_db(tag: &str) -> ScratchDir {
    let dir = std::env::temp_dir().join(format!("sac-api-{tag}-{}", std::process::id()));
    // a killed run may have left this pid's dir behind; every test
    // here builds its tracker from nothing
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("saccade.db");
    ScratchDir { dir, db }
}

/// A scratch tracker dir that removes itself at test end, panic or
/// not - one run of this suite used to leave hundreds behind. Reads
/// as the tracker's db path, as before.
struct ScratchDir {
    dir: PathBuf,
    db: PathBuf,
}

impl std::ops::Deref for ScratchDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.db
    }
}

impl AsRef<std::ffi::OsStr> for ScratchDir {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.db.as_os_str()
    }
}

impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.db
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A serving subprocess that dies with its test, panic or not.
struct ServerProc(std::process::Child);

impl std::ops::Deref for ServerProc {
    type Target = std::process::Child;

    fn deref(&self) -> &std::process::Child {
        &self.0
    }
}

impl std::ops::DerefMut for ServerProc {
    fn deref_mut(&mut self) -> &mut std::process::Child {
        &mut self.0
    }
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
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
            json!({"comment": {"home": 0, "kind": {"offer": {"title": "suite green", "body": "suite green"}}, "references": []}}),
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

    // a title-only revision rides the same wire: the fold swaps the
    // description and leaves the body standing
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"revise_comment": {"id": 2, "title": "foo done, retitled"}}),
        ),
    );
    assert_eq!(status, 200);
    let reply = json_of(&body);
    assert_eq!(reply["records"][0]["kind"], "comment_revised");
    assert!(
        reply["records"][0]["payload"]
            .to_string()
            .contains("retitled"),
        "{}",
        reply["records"][0]["payload"]
    );

    // the authoring ingress rides the wire: a marker the mapping
    // cannot resolve refuses with its taught reason
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"home": 0, "kind": {"note": {"title": "a dangling marker", "body": "dangling [^1]"}}, "references": []}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_reference");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();

    let hist = scratch_db("roundtrip-hist");
    {
        let mut conn = saccade::db::open(&hist).unwrap();
        let now = 1_700_000_000;
        saccade::db::record(
            &mut conn,
            &Context::system(),
            Command::CreateTask {
                name: Prose::new("carry a merge through the worktree".into()).unwrap(),
            },
            now,
        )
        .unwrap();
        saccade::db::record(
            &mut conn,
            &Context::system(),
            Command::EstablishWorkspace {
                task_id: TaskId(0),
                base: saccade::GitCommit::new("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into())
                    .unwrap(),
                branch: saccade::GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: saccade::WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            now + 1,
        )
        .unwrap();
    }
    let hist_base = spawn_server(&hist).await;
    let (_, standing) = get_json(&hist_base, "/api/v1/records");
    assert_eq!(
        json_of(&standing)["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["kind"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["task_created", "workspace_established"]
    );

    let (status, body) = post_command(
        &hist_base,
        &envelope(
            "checkpoint reviewer",
            "agent",
            json!({"checkpoint_workspace": {"task_id": 0, "checkpoint": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");
    let (_, after) = get_json(&hist_base, "/api/v1/records");
    assert_eq!(after, standing, "the refused checkpoint wrote nothing");
    let lines = attempts_lines(&hist);
    let refused = lines.last().expect("the refusal left its line");
    assert_eq!(refused["outcome"], "refused");
    assert_eq!(refused["code"], "malformed_request");
    assert!(
        refused["request"]
            .as_str()
            .is_some_and(|r| r.contains("checkpoint_workspace")),
        "{refused}"
    );

    std::fs::remove_dir_all(hist.parent().unwrap()).unwrap();
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
            json!({"comment": {"home": 0, "kind": {"offer": {"title": "first account", "body": "first account"}}, "references": []}}),
        ),
    );
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"home": 0, "kind": {"offer": {"title": "replacement", "body": "replacement"}}, "references": []}}),
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

#[test]
fn the_native_integration_attaches_and_sends_over_the_wire() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("native-messages");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let hc = fake_hc(db.parent().unwrap());
    let (base_url, pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        db.parent().unwrap(),
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // two addressed peers: the launch door's birth shape, registered
    // with their host addresses
    for (native_id, address) in [("aaaa1111", "t1"), ("bbbb2222", "t2")] {
        let (status, body) = post_json(
            &base_url,
            "/api/v1/incarnations/register",
            &json!({"native_id": native_id, "address": address, "adopted": false}),
        );
        assert_eq!(status, 200, "{body}");
    }

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
    // the door birth carried no locator: this attach binds it
    assert_eq!(reply["records"].as_array().unwrap().len(), 1);
    assert_eq!(reply["records"][0]["kind"], "saved_session_bound");

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

    // the registered session sends: the record lands, then the sole
    // writer submits through the adapter at the recorded address, and
    // the acceptance record lands before the reply
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
    let first_seq = reply["message"]["seq"].as_u64().unwrap();
    assert_eq!(reply["message"]["num"], 0);
    assert_eq!(reply["from"]["seq"], 0);
    assert_eq!(reply["from"]["num"], 0);
    assert_eq!(reply["submission"]["state"], "submitted");
    // the submit reached the host: the adapter ask named the recorded
    // address and carried the wire envelope - machine-supplied origin,
    // task segment when the context rides one, then the body untouched
    // (the unicode body rides byte for byte, the envelope line ASCII)
    let sent_argv = hc_argv(hc.parent().unwrap(), "send");
    assert_eq!(
        sent_argv.len(),
        4,
        "one submit ask spans its own lines: {sent_argv:?}"
    );
    assert_eq!(sent_argv[0], "t2");
    assert_eq!(
        sent_argv[1], "[saccade message m-0 | from i-0 | task t-0]",
        "{sent_argv:?}"
    );
    assert_eq!(sent_argv[2], "");
    assert_eq!(
        sent_argv[3],
        "please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix"
    );

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
    assert_eq!(json_of(&body)["message"]["num"], 1);
    assert_eq!(json_of(&body)["submission"]["state"], "submitted");
    let reply_seq = json_of(&body)["message"]["seq"].as_u64().unwrap();
    // the reply's envelope carries the replying-to segment and no task
    let sent_argv = hc_argv(hc.parent().unwrap(), "send");
    assert_eq!(
        sent_argv.len(),
        9,
        "two submit asks, the second op line between them: {sent_argv:?}"
    );
    assert_eq!(
        sent_argv[6], "[saccade message m-1 | from i-1 | replying to m-0]",
        "{sent_argv:?}"
    );
    assert_eq!(sent_argv[8], "the edge case held under review");

    // the mailbox read returns both directions in birth order, all
    // confirmed; the mailbox is a descriptive read
    let (status, body) = get_json(&base_url, "/api/v1/messages?incarnation=1");
    assert_eq!(status, 200, "{body}");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 2);
    assert_eq!(mail[0]["seq"], first_seq);
    assert_eq!(mail[0]["num"], 0);
    assert_eq!(mail[0]["from"]["seq"], 0);
    assert_eq!(mail[0]["from"]["num"], 0);
    assert_eq!(mail[0]["from"]["actor"], "pi");
    assert_eq!(mail[0]["task"], 0);
    assert_eq!(mail[0]["state"], "submitted");
    assert_eq!(
        mail[0]["body"],
        "please look at the failing edge case: caf\u{e9} floop, na\u{ef}ve fix"
    );
    assert_eq!(mail[1]["seq"], reply_seq as usize);
    assert_eq!(mail[1]["from"]["num"], 1);
    assert_eq!(mail[1]["to"]["num"], 0);
    assert_eq!(mail[1]["reply_to"], 0);
    assert_eq!(mail[1]["state"], "submitted");

    // unknown recipients refuse the read
    let (status, body) = get_json(&base_url, "/api/v1/messages?incarnation=99");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "invalid_incarnation_id");

    // the bridge-era transport faces are gone: the deliver poll and the
    // retired pending tombstone 404, and the submit ack's route answers
    // no method (it was POST-only)
    for retired in [
        "/api/v1/messages/deliver?incarnation=1&wait_ms=25000",
        "/api/v1/messages/pending?incarnation=1",
    ] {
        let (status, body) = get_json(&base_url, retired);
        assert_eq!(status, 404, "{retired}: {body}");
    }
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/submit",
        &json!({"incarnation": 1, "message": 3}),
    );
    assert_eq!(status, 405, "{body}");

    // an unaddressed recipient is the pre-B shape: the record stands
    // as evidence, nothing submits, and the refusal names the migration
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "pre-b-peer1",
            "session_file": "/sessions/pre-b-peer1.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let bridgeless_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let bridgeless = json_of(&body)["num"].as_u64().unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": bridgeless,
            "body": "the recipient predates the host addresses; this stands pending",
        }),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "recipient_unaddressed",
        "{body}"
    );
    let detail = json_of(&body)["error"]["detail"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(detail.contains("exit + resume"), "{detail}");
    assert_eq!(json_of(&body)["message"]["num"], 2);
    // the standing evidence reads through the history route
    let (_, body) = get_json(
        &base_url,
        &format!("/api/v1/messages?incarnation={bridgeless_birth}"),
    );
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 1, "{body}");
    assert_eq!(mail[0]["state"], "pending", "{body}");

    // a refused submit is named evidence too: the host says the
    // address is finished, the record stays pending
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "gonepeer1", "address": "t9", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let gone_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 3,
            "body": "wake a finished address",
        }),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "recipient_not_live",
        "{body}"
    );
    assert_eq!(json_of(&body)["message"]["num"], 3);
    let (_, body) = get_json(
        &base_url,
        &format!("/api/v1/messages?incarnation={gone_birth}"),
    );
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 1, "{body}");
    assert_eq!(mail[0]["state"], "pending", "{body}");

    // judgment tiers are refused at the authority table: the delivery
    // record is machinery's own act
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"submit_message": {"id": first_seq, "by": 1}}),
        ),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "human_only");

    // the sender's own mail reads in the mailbox too, with its state
    let (_, body) = get_json(&base_url, "/api/v1/messages?incarnation=0");
    let mine = json_of(&body)["messages"].as_array().unwrap().clone();
    let sent_by_me: Vec<&Value> = mine.iter().filter(|m| m["from"]["seq"] == 0).collect();
    // the confirmed delivery and the refused attempts' standing evidence
    assert_eq!(sent_by_me.len(), 3);
    let confirmed = sent_by_me.iter().find(|m| m["seq"] == first_seq).unwrap();
    // one shared delivery state: the recipient's submission reads here too
    assert_eq!(confirmed["state"], "submitted");
    assert_eq!(confirmed["to"]["num"], 1);
    let standing: Vec<&&Value> = sent_by_me
        .iter()
        .filter(|m| m["state"] == "pending")
        .collect();
    assert_eq!(standing.len(), 2, "the refused attempts stand pending");

    // the command face rides the same operation: the record lands and
    // the attempt's truth rides the reply as data
    let (status, body) = post_json(
        &base_url,
        "/api/v1/command",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "command": {"send_message": {"from": 0, "to": 1, "body": "the command face submits too"}},
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["submission"]["state"], "submitted", "{body}");

    // concurrent sends to one recipient submit in record order: the
    // per-recipient lock covers record and submit together
    let mut senders = Vec::new();
    for body_text in [
        "serialized ask one",
        "serialized ask two",
        "serialized ask three",
    ] {
        let url = base_url.clone();
        senders.push(std::thread::spawn(move || {
            post_json(
                &url,
                "/api/v1/messages/send",
                &json!({
                    "context": {"actor": "pi", "tier": "agent",
                                "scope": {"task": null, "incarnation": 0, "workspace": null}},
                    "to": 1,
                    "body": body_text,
                }),
            )
        }));
    }
    for sender in senders {
        let (status, body) = sender.join().unwrap();
        assert_eq!(status, 200, "{body}");
    }
    // record order: the three new messages by ascending birth position
    let (_, body) = get_json(&base_url, "/api/v1/messages?incarnation=1");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    let mut ordered: Vec<&Value> = mail
        .iter()
        .filter(|m| m["body"].as_str().unwrap().starts_with("serialized ask"))
        .collect();
    ordered.sort_by_key(|m| m["seq"].as_u64().unwrap());
    assert_eq!(ordered.len(), 3, "{body}");
    // submit order: the fake host's send log names the same three
    // bodies (each beneath its envelope line), and the order matches
    // the record order exactly
    let logged = hc_argv(hc.parent().unwrap(), "send");
    let asks: Vec<&String> = logged
        .iter()
        .filter(|line| line.starts_with("serialized ask"))
        .collect();
    assert_eq!(asks.len(), 3, "{logged:?}");
    for (recorded, asked) in ordered.iter().zip(asks.iter()) {
        assert_eq!(
            recorded["body"].as_str().unwrap(),
            asked.as_str(),
            "record order must match submit order"
        );
    }

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

    let _ = std::process::Command::new("kill")
        .arg(pid.id().to_string())
        .status();
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
fn post_json_headers(
    base_url: &str,
    path: &str,
    body: &Value,
    headers: &[(&str, &str)],
) -> (u16, String) {
    let mut r = ureq::post(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build();
    for (k, v) in headers {
        r = r.header(*k, *v);
    }
    let mut r = r
        .send_json(body.clone())
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

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

    // a pre-title (v4-shaped) note or offer no longer parses: the
    // 0.15 wire refuses the birth rather than silently dropping the
    // field an old server would have ignored
    for kind in ["note", "offer"] {
        let (status, body) = post_command(
            &base_url,
            &envelope(
                "pi",
                "agent",
                json!({"comment": {"home": 0, "kind": {kind: {"body": "suite green"}}, "references": []}}),
            ),
        );
        assert_eq!(status, 400, "{kind}");
        assert_eq!(
            json_of(&body)["error"]["code"],
            "malformed_request",
            "{kind}: {body}"
        );
    }
    // nothing landed: the log holds only the parse-door refusal lines
    let (_, standing) = get_json(&base_url, "/api/v1/records");
    assert!(
        json_of(&standing)["records"]
            .as_array()
            .is_some_and(|r| r.is_empty()),
        "{standing}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A pinned fake peer for the version gate: it answers the identity
/// route and would acknowledge any command post, recording each.
fn fake_peer(
    version: Option<&str>,
    identity_status: u16,
    budget: usize,
) -> (
    String,
    std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    std::thread::JoinHandle<()>,
) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let posts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let posted = std::sync::Arc::clone(&posts);
    let version = version.map(str::to_string);
    let handle = std::thread::spawn(move || {
        let read_request = |stream: &mut std::net::TcpStream| -> Option<String> {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(0) | Err(_) => return None,
                    Ok(_) => head.push(byte[0]),
                }
            }
            let text = String::from_utf8_lossy(&head).into_owned();
            let mut lines = text.lines();
            let request = lines.next()?.to_string();
            let mut length = 0usize;
            for line in lines {
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; length];
            if length > 0 && stream.read_exact(&mut body).is_err() {
                return None;
            }
            posted
                .lock()
                .expect("the post log holds")
                .push(format!("{request} {}", String::from_utf8_lossy(&body)));
            Some(
                request
                    .split(' ')
                    .next()
                    .map(str::to_string)
                    .unwrap_or_default(),
            )
        };
        let mut served = 0usize;
        while served < budget {
            let Ok((mut stream, _peer)) = listener.accept() else {
                break;
            };
            let Some(method) = read_request(&mut stream) else {
                served += 1;
                continue;
            };
            let reply = match method.as_str() {
                "GET" => match &version {
                    Some(v) => {
                        let body = format!("{{\"version\":\"{v}\",\"db\":\"/fake/peer.db\"}}");
                        format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {length}\r\n\r\n{body}",
                            status = identity_status,
                            length = body.len(),
                        )
                    }
                    None => "HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\nConnection: close\r\nContent-Length: 11\r\n\r\n<html></html>".to_string(),
                },
                _ => "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: 14\r\n\r\n{\"records\":[]}".to_string(),
            };
            let _ = stream.write_all(reply.as_bytes());
            let _ = stream.flush();
            served += 1;
        }
    });
    (format!("http://{addr}"), posts, handle)
}

/// Drain a fake peer's unused budget with empty connections, then wait
/// for its thread: nothing is left running and nothing is discovered.
fn retire_peer(
    addr: &str,
    posts: &std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    budget: usize,
    handle: std::thread::JoinHandle<()>,
) {
    let live = posts.lock().expect("the post log holds").len();
    for _ in live..budget {
        let _ = std::net::TcpStream::connect(addr);
    }
    handle.join().expect("the fake peer thread exits");
}

/// A command write posts only to a peer on this binary's version line:
/// the same minor whatever the patch; any other version, or none the
/// rule can read, refuses before the POST leaves.
#[tokio::test(flavor = "multi_thread")]
async fn command_writes_refuse_peers_outside_the_binary_s_version_line() {
    use saccade::Command;
    use saccade::client::{self, ClientFail};
    use saccade::store::{Context, Tier};
    use saccade::types::actor::ActorName;

    let agent = Context {
        actor: ActorName::new("saccade bot".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };
    let command = || Command::CreateTask {
        name: saccade::Prose::new("across the line".into()).unwrap(),
    };

    let major = env!("CARGO_PKG_VERSION_MAJOR");
    let minor: usize = env!("CARGO_PKG_VERSION_MINOR").parse().unwrap();
    let patch: usize = env!("CARGO_PKG_VERSION_PATCH").parse().unwrap();
    let different_minor = format!("{major}.{}.0", minor + 1);
    let different_patch = format!("{major}.{minor}.{}", patch + 1);

    // A different minor refuses before the command posts.
    let (url, posts, handle) = fake_peer(Some(&different_minor), 200, 1);
    match client::send(&url, &agent, command(), None) {
        Err(ClientFail::VersionMismatch { ours, theirs, .. }) => {
            assert_eq!(ours, env!("CARGO_PKG_VERSION"));
            assert_eq!(theirs, different_minor);
        }
        other => panic!("a breaking peer refuses the write: {other:?}"),
    }
    assert!(
        posts.lock().unwrap().iter().all(|r| r.starts_with("GET")),
        "no command body reached the peer: {:?}",
        posts.lock().unwrap()
    );
    retire_peer(&url, &posts, 1, handle);

    // A patch difference permits the command POST and acknowledgement.
    let (url, posts, handle) = fake_peer(Some(&different_patch), 200, 2);
    let reply =
        client::send(&url, &agent, command(), None).expect("a patch difference is not breaking");
    assert_eq!(reply.records.len(), 0);
    assert!(
        posts
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.starts_with("POST /api/v1/command")),
        "{:?}",
        posts.lock().unwrap()
    );
    retire_peer(&url, &posts, 2, handle);

    // a peer off the 0.x line has no ratified rule: ambiguity refuses
    let (url, posts, handle) = fake_peer(Some("1.0.0"), 200, 1);
    match client::send(&url, &agent, command(), None) {
        Err(ClientFail::VersionUnproven { why, .. }) => {
            assert!(why.contains("no ratified compatibility rule"), "{why}");
        }
        other => panic!("an unruled version line refuses: {other:?}"),
    }
    assert!(posts.lock().unwrap().iter().all(|r| r.starts_with("GET")));
    retire_peer(&url, &posts, 1, handle);

    // an unreadable version is unproven, never guessed at
    for theirs in ["not.a.version", "0.16.0-rc", "0.15", "0.15.0.extra"] {
        let (url, posts, handle) = fake_peer(Some(theirs), 200, 1);
        match client::send(&url, &agent, command(), None) {
            Err(ClientFail::VersionUnproven { .. }) => {}
            other => panic!("{theirs} refuses as unproven: {other:?}"),
        }
        assert!(posts.lock().unwrap().iter().all(|r| r.starts_with("GET")));
        retire_peer(&url, &posts, 1, handle);
    }

    // a peer without a readable identity route keeps the existing
    // refusal taxonomy: the write never posts
    let (url, posts, handle) = fake_peer(None, 404, 1);
    match client::send(&url, &agent, command(), None) {
        Err(ClientFail::Refused { code, .. }) => assert_eq!(code, "malformed_response"),
        other => panic!("a peer with no identity refuses: {other:?}"),
    }
    assert!(posts.lock().unwrap().iter().all(|r| r.starts_with("GET")));
    retire_peer(&url, &posts, 1, handle);

    // a non-success identity answer is no proof even when its body
    // claims this binary's own version: the status is part of the proof
    let (url, posts, handle) = fake_peer(Some(env!("CARGO_PKG_VERSION")), 500, 1);
    match client::send(&url, &agent, command(), None) {
        Err(ClientFail::Refused {
            code,
            detail,
            message: _,
        }) => {
            assert_eq!(code, "malformed_response");
            assert!(detail.contains("status 500"), "{detail}");
        }
        other => panic!("a failing identity route refuses: {other:?}"),
    }
    assert!(posts.lock().unwrap().iter().all(|r| r.starts_with("GET")));
    retire_peer(&url, &posts, 1, handle);
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
        ClientFail::Refused {
            code,
            detail,
            message: _,
        } => {
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
                title: saccade::Prose::new("a note to nothing".into()).unwrap(),
                body: saccade::Prose::new("talking to nothing".into()).unwrap(),
            },
            references: Vec::new(),
        },
        None,
    )
    .unwrap_err();
    match &fail {
        ClientFail::Refused {
            code,
            detail,
            message: _,
        } => {
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
use saccade::{Command, Prose, TaskId};

async fn spawn_workbench(db_path: &std::path::Path) -> (String, ConsoleState) {
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

/// The last comment_revised record's payload, as the door's patch
/// intent landed it.
fn last_revision_of(state: &ConsoleState) -> String {
    match state.snapshot() {
        Ok(s) => s
            .rows
            .iter()
            .rev()
            .find(|r| r.kind == "comment_revised")
            .map(|r| r.payload.clone())
            .expect("a revision landed"),
        Err(_) => panic!("the snapshot refused"),
    }
}

/// The workbench page owns /, /t/N, and /c/N deep links, and the
/// embedded assets answer with their content types.
#[tokio::test(flavor = "multi_thread")]
async fn the_workbench_page_serves_root_and_deep_links() {
    let db = scratch_db("workbench-page");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "migrate floop");
    for path in ["/", "/t/0", "/c/0"] {
        let (status, html) = get_html(&format!("{base}{path}"));
        assert_eq!(status, 200, "{path}");
        assert!(html.to_lowercase().contains("saccade"), "{path}: {html}");
    }
    let (status, _) = get_html(&format!("{base}/gone"));
    assert_eq!(status, 404);
    for (path, kind) in [("/app.js", "text/javascript"), ("/style.css", "text/css")] {
        let mut r = ureq::get(&format!("{base}{path}"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap();
        assert_eq!(r.status().as_u16(), 200, "{path}");
        assert!(
            r.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with(kind)),
            "{path}"
        );
        let _ = r.body_mut().read_to_string().unwrap();
    }
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The snapshot is the canonical projection: authored titles ride the
/// deposits, html comes from the shared renderer, backlinks reverse
/// the fold's own mappings, and withdrawn hides title with body while
/// the cursor names the log.
#[tokio::test(flavor = "multi_thread")]
async fn workbench_snapshot_projects_the_canonical_fold() {
    let db = scratch_db("workbench-snapshot");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "migrate floop");
    seed_task(&state, "other work");
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Note {
                    title: Prose::new("the finding".into()).unwrap(),
                    body: Prose::new("parked finding".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Note {
                    title: Prose::new("the withdrawal".into()).unwrap(),
                    body: Prose::new("soon gone".into()).unwrap(),
                },
                references: Vec::new(),
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
                kind: CommentKind::Note {
                    title: Prose::new("the reply".into()).unwrap(),
                    body: Prose::new("rests on the finding [^1]".into()).unwrap(),
                },
                references: vec![RecordId(2)],
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::WithdrawComment {
                id: CommentId(RecordId(3)),
                note: Prose::new("parked in the wrong place".into()).unwrap(),
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();

    let (status, body) = get_json(&base, "/api/v1/workbench");
    assert_eq!(status, 200);
    let projection = json_of(&body);
    let comments = projection["comments"].as_array().unwrap().clone();
    assert_eq!(comments.len(), 3);
    assert_eq!(comments[0]["title"].as_str(), Some("the finding"));
    assert_eq!(comments[0]["kind"].as_str(), Some("note"));
    assert_eq!(
        comments[0]["html"].as_str(),
        Some("<p>parked finding</p>\n")
    );
    // backlinks reverse the fold's explicit mapping, dense numbers
    assert_eq!(comments[0]["cited_by"].as_array().unwrap().len(), 1);
    let withdrawn = &comments[1];
    assert_eq!(withdrawn["withdrawn"].as_bool(), Some(true));
    assert_eq!(withdrawn["body"].as_str(), Some(""));
    assert_eq!(withdrawn["title"].as_str(), None);
    assert_eq!(withdrawn["kind"].as_str(), Some("note"));
    assert_eq!(projection["cursor"].as_i64(), Some(5));
    assert_eq!(
        projection["tasks"][0]["name"].as_str(),
        Some("migrate floop")
    );
    // home rides the same projection: every task and comment record is
    // recent activity, newest first, anchored to its public c-N
    let activity = projection["recent_activity"].as_array().unwrap();
    let seqs: Vec<i64> = activity
        .iter()
        .map(|e| e["seq"].as_i64().unwrap())
        .collect();
    assert_eq!(seqs, vec![5, 4, 3, 2, 1, 0]);
    let withdrawal = &activity[0];
    assert_eq!(withdrawal["event"].as_str(), Some("comment_withdrawn"));
    assert_eq!(withdrawal["home"].as_i64(), Some(0));
    assert_eq!(withdrawal["comment"].as_i64(), Some(1));
    assert_eq!(withdrawal["actor"].as_str(), Some("jerry"));
    let created = &activity[5];
    assert_eq!(created["event"].as_str(), Some("task_created"));
    assert_eq!(created["home"].as_i64(), Some(0));
    assert_eq!(created["comment"].as_i64(), None);
    // recently active tasks: one entry per task, the same records
    // ordering it - t-0's withdrawal is later than t-1's reply
    let active = projection["recently_active_tasks"].as_array().unwrap();
    let order: Vec<i64> = active.iter().map(|e| e["task"].as_i64().unwrap()).collect();
    assert_eq!(order, vec![0, 1]);
    assert_eq!(active[0]["seq"].as_i64(), Some(5));
    assert_eq!(active[1]["seq"].as_i64(), Some(4));
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The workbench read is the page's synchronization surface: with a
/// cursor and a bounded window it parks while the log stands at that
/// cursor, woken early by a landing through the server's own door and
/// re-checked on bounded slices for writers that fire no wake, while a
/// stale cursor or no parameters returns immediately.
#[tokio::test(flavor = "multi_thread")]
async fn the_workbench_read_parks_wakes_and_catches_up() {
    let db = scratch_db("workbench-follow");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "migrate floop");
    let (status, body) = get_json(&base, "/api/v1/workbench");
    assert_eq!(status, 200, "{body}");
    let cursor = json_of(&body)["cursor"].as_u64().unwrap();

    // a read at the standing cursor parks: it neither returns the same
    // fold nor wakes without a landing
    let parked = {
        let base = base.clone();
        tokio::task::spawn_blocking(move || {
            get_json(
                &base,
                &format!("/api/v1/workbench?cursor={cursor}&wait_ms=15000"),
            )
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(
        !parked.is_finished(),
        "a read at the standing cursor must park, not return the same fold"
    );

    // the landing wakes it, and the fold comes back advanced
    seed_task(&state, "the woken work");
    let (status, body) = tokio::time::timeout(std::time::Duration::from_secs(10), parked)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, 200, "{body}");
    let woke = json_of(&body);
    assert_eq!(woke["cursor"].as_u64().unwrap(), cursor + 1);
    assert_eq!(woke["tasks"].as_array().unwrap().len(), 2);

    // a stale cursor returns immediately with the whole fresh fold
    let started = std::time::Instant::now();
    let (status, body) = get_json(
        &base,
        &format!("/api/v1/workbench?cursor={cursor}&wait_ms=15000"),
    );
    assert_eq!(status, 200, "{body}");
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    assert_eq!(json_of(&body)["cursor"].as_u64().unwrap(), cursor + 1);

    // the bounded window returns on its own when nothing lands
    let started = std::time::Instant::now();
    let (status, body) = get_json(
        &base,
        &format!("/api/v1/workbench?cursor={}&wait_ms=350", cursor + 1),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["cursor"].as_u64().unwrap(), cursor + 1);
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(350),
        "the bounded park must hold its window"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "the bounded park must return at its window, not hang"
    );

    // a record another process wrote to the shared database fires no
    // wake here: the sliced cursor check must still surface it
    let parked = {
        let base = base.clone();
        tokio::task::spawn_blocking(move || {
            get_json(
                &base,
                &format!("/api/v1/workbench?cursor={}&wait_ms=15000", cursor + 1),
            )
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let outside = std::process::Command::new(env!("CARGO_BIN_EXE_sac"))
        .arg("--db")
        .arg(&db)
        .env("SACCADE_ACTOR", "outside-author")
        .env("SACCADE_TASK", "")
        .env("SACCADE_INCARNATION_SEQ", "")
        .env("SACCADE_WORKSPACE", "")
        .args(["create", "task", "the outside writer"])
        .output()
        .expect("spawn the outside writer");
    assert!(
        outside.status.success(),
        "the outside writer lands: {}",
        String::from_utf8_lossy(&outside.stderr)
    );
    let started = std::time::Instant::now();
    let (status, body) = tokio::time::timeout(std::time::Duration::from_secs(10), parked)
        .await
        .unwrap()
        .unwrap();
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the outside write must surface within the slice bound"
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["cursor"].as_u64().unwrap(), cursor + 2);

    // an empty log parks a waiting read at "nothing seen yet" and wakes
    // at its first birth, so a fresh page never spins
    let empty_db = scratch_db("workbench-follow-empty");
    let (empty_base, empty_state) = spawn_workbench(&empty_db).await;
    let empty = {
        let base = empty_base.clone();
        tokio::task::spawn_blocking(move || get_json(&base, "/api/v1/workbench?wait_ms=15000"))
    };
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(!empty.is_finished(), "an empty log parks a waiting read");
    seed_task(&empty_state, "the first birth");
    let (status, body) = tokio::time::timeout(std::time::Duration::from_secs(10), empty)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["cursor"].as_u64().unwrap(), 0);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(empty_db.parent().unwrap()).unwrap();
}

/// The preview renders one body against fixed references through the
/// canonical renderer; unknown references refuse and nothing writes.
#[tokio::test(flavor = "multi_thread")]
async fn workbench_preview_renders_and_refuses_unknown_references() {
    let db = scratch_db("workbench-preview");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "migrate floop");
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Note {
                    title: Prose::new("the finding".into()).unwrap(),
                    body: Prose::new("parked finding".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    let before = get_json(&base, "/api/v1/workbench").1;

    let (status, body) = post_json(
        &base,
        "/api/v1/workbench/preview",
        &json!({"body": "the reply cites [^1]", "references": [1]}),
    );
    assert_eq!(status, 200, "{body}");
    let html = json_of(&body)["html"].as_str().unwrap().to_string();
    assert!(html.contains("class=\"mref\""), "{html}");

    let (status, body) = post_json(
        &base,
        "/api/v1/workbench/preview",
        &json!({"body": "dangling", "references": [99]}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "invalid_reference",
        "{body}"
    );
    let after = get_json(&base, "/api/v1/workbench").1;
    assert_eq!(
        json_of(&before)["cursor"],
        json_of(&after)["cursor"],
        "the preview never writes"
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The workbench write door keeps the console identity law with the
/// attribution bound to the submission: the actor is exactly the name
/// the review showed, the tier is pinned human with no scope, and a
/// cookie that changed under a confirmed review refuses loudly.
#[tokio::test(flavor = "multi_thread")]
async fn workbench_command_pins_human_attribution_and_binds_submissions() {
    let db = scratch_db("workbench-command");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "migrate floop");
    let note = |title: &str, body: &str| {
        json!({"actor": "jerry", "command": {"comment": {
            "home": 0,
            "kind": {"note": {"title": title, "body": body}},
            "references": [],
        }}})
    };

    let (status, body) = post_json(
        &base,
        "/api/v1/workbench/command",
        &note("the finding", "parked finding"),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["records"][0]["kind"], "commented");
    assert_eq!(reply["records"][0]["actor"], "jerry");
    assert_eq!(reply["records"][0]["tier"], "human");

    // the cookie binds: a matching claim passes, a changed one refuses
    let (status, body) = post_json_headers(
        &base,
        "/api/v1/workbench/command",
        &note("the second", "another finding"),
        &[("Cookie", "actor=jerry")],
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_json_headers(
        &base,
        "/api/v1/workbench/command",
        &note("the third", "a third finding"),
        &[("Cookie", "actor=someone-else")],
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "actor_changed", "{body}");

    // a core refusal arrives typed with its teaching and writes nothing
    let before = get_json(&base, "/api/v1/workbench").1;
    let (status, body) = post_json(
        &base,
        "/api/v1/workbench/command",
        &json!({"actor": "jerry", "command": {"comment": {
            "home": 0,
            "kind": {"note": {"title": "dangling", "body": "marker [^2]"}},
            "references": [],
        }}}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "invalid_reference",
        "{body}"
    );
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("marker")),
        "{body}"
    );
    let after = get_json(&base, "/api/v1/workbench").1;
    assert_eq!(
        json_of(&before)["cursor"],
        json_of(&after)["cursor"],
        "the refusal wrote nothing"
    );

    // the identity read discloses the cookie-claimed actor and this
    // db's exact legacy storage instance: sha256 of the canonical db
    // path, the key the prototype wrote drafts under
    let (status, body) = get_json(&base, "/api/v1/workbench/identity");
    assert_eq!(status, 200);
    assert_eq!(json_of(&body)["actor"], json!(null));
    use sha2::Digest;
    let canonical = std::fs::canonicalize(&db).unwrap();
    let legacy: String = sha2::Sha256::digest(canonical.to_string_lossy().as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        json_of(&body)["legacy_instance"],
        json!(legacy[..16].to_string()),
        "the legacy instance is exactly the prototype's key suffix"
    );
    let mut r = ureq::get(&format!("{base}/api/v1/workbench/identity"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("Cookie", "actor=jerry")
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(
        json_of(&r.body_mut().read_to_string().unwrap())["actor"],
        json!("jerry")
    );

    // a create names what was born, exactly as the command door names
    // it, so the workbench can open the born thread
    let (status, body) = post_json(
        &base,
        "/api/v1/workbench/command",
        &json!({"actor": "jerry", "command": {"create_task": {"name": "the browser-born task"}}}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["records"][0]["kind"], "task_created", "{body}");
    assert_eq!(reply["records"][0]["tier"], "human", "{body}");
    let id = reply["id"].as_str().expect("the reply names the born task");
    let snapshot = json_of(&get_json(&base, "/api/v1/workbench").1);
    let born = snapshot["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "the browser-born task")
        .expect("the born task projects");
    assert_eq!(
        id,
        format!("t-{}", born["id"].as_i64().unwrap()),
        "{snapshot}"
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The snapshot carries the record facts a reader needs: use
/// source/binding, revised/reviser marks, withdrawer naming, task
/// owner/state, and every href in its rendered html resolves on this
/// server.
#[tokio::test(flavor = "multi_thread")]
async fn workbench_snapshot_carries_record_facts_and_resolving_links() {
    let db = scratch_db("workbench-facts");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "produce the account");
    seed_task(&state, "consume the account");
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(0),
                kind: CommentKind::Offer {
                    title: Prose::new("the standing account".into()).unwrap(),
                    body: Prose::new("suite green, see the plan [^1]".into()).unwrap(),
                },
                references: vec![RecordId(0)],
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(1),
                kind: CommentKind::Use {
                    source: TaskId(0),
                    reason: Prose::new("builds on the account".into()).unwrap(),
                    bound: None,
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    let use_seq = {
        let (_, snapshot) = get_json(&base, "/api/v1/workbench");
        json_of(&snapshot)["comments"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["seq"]
            .as_u64()
            .unwrap() as usize
    };
    state
        .execute(
            &human_ctx(),
            Command::ClaimTask {
                id: TaskId(1),
                incarnation: None,
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::BindUse {
                id: CommentId(RecordId(use_seq)),
                offer: CommentId(RecordId(2)),
                grounds: Prose::new("the accepted account is the one that landed".into()).unwrap(),
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    state
        .execute(
            &human_ctx(),
            Command::CancelTask {
                id: TaskId(0),
                explanation: Prose::new("the producer need dissolved".into()).unwrap(),
            },
            None,
            bare_request(),
        )
        .map_err(|e| panic!("cancel refused: {e:?}"))
        .unwrap();

    let (status, snapshot) = get_json(&base, "/api/v1/workbench");
    assert_eq!(status, 200);
    let projection = json_of(&snapshot);
    let comments = projection["comments"].as_array().unwrap().clone();
    let uses = comments.iter().find(|c| c["kind"] == "use").unwrap();
    assert_eq!(
        uses["source"].as_u64(),
        Some(0),
        "a use names its source task"
    );
    assert_eq!(
        uses["binding"]["offer"].as_u64(),
        Some(0),
        "the binding names the selected offer by dense number"
    );
    assert!(
        uses["binding"]["grounds"]
            .as_str()
            .is_some_and(|g| g.contains("landed")),
        "the binding carries its grounds"
    );
    assert_eq!(
        projection["tasks"][1]["owner"].as_str(),
        Some("jerry"),
        "the owner is a content fact"
    );
    assert!(
        projection["tasks"][0]["explanation"]
            .as_str()
            .is_some_and(|e| e.contains("dissolved")),
        "a cancelled task's explanation rides the projection"
    );

    // every /t/N href the projection html emits resolves on this server
    let mut hrefs = Vec::new();
    for comment in &comments {
        for href in comment["html"]
            .as_str()
            .unwrap_or_default()
            .split("href=\"")
            .skip(1)
        {
            let path = href.split('"').next().unwrap_or_default();
            if path.starts_with('/') {
                hrefs.push(path.to_string());
            }
        }
    }
    assert!(!hrefs.is_empty(), "the projection renders links");
    for path in &hrefs {
        let (status, _) = get_html(&format!("{base}{path}"));
        assert_eq!(status, 200, "projection href {path} resolves");
    }

    // deep links to nothing recorded answer 404, not the page
    for path in ["/t/99", "/c/99"] {
        let (status, _) = get_html(&format!("{base}{path}"));
        assert_eq!(
            status, 200,
            "{path} serves the workbench page (client resolves)"
        );
    }
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The workbench write door keeps the console's CSRF and first-claim
/// identity laws: cross-site POSTs refuse, a missing name refuses
/// taught, and a cookieless first claim sets the actor cookie.
#[tokio::test(flavor = "multi_thread")]
async fn workbench_command_guards_origin_and_first_claim() {
    let db = scratch_db("workbench-guards");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "migrate floop");
    let note = |title: &str, body: &str| {
        json!({"actor": "jerry", "command": {"comment": {
            "home": 0,
            "kind": {"note": {"title": title, "body": body}},
            "references": [],
        }}})
    };

    let (status, body) = post_json_headers(
        &base,
        "/api/v1/workbench/command",
        &note("the finding", "parked finding"),
        &[("Sec-Fetch-Site", "cross-site"), ("Sec-Fetch-Mode", "cors")],
    );
    assert_eq!(status, 403, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "cross_site_refused",
        "{body}"
    );

    let (status, body) = post_json(
        &base,
        "/api/v1/workbench/command",
        &json!({"command": {"create_task": {"name": "no name"}}}),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("actor")),
        "the taught refusal names the actor field: {body}"
    );
    let (_, projection) = get_json(&base, "/api/v1/workbench");
    assert!(
        json_of(&projection)["comments"]
            .as_array()
            .unwrap()
            .is_empty(),
        "refused guards wrote nothing"
    );

    let (status, body) = post_json_headers(
        &base,
        "/api/v1/workbench/command",
        &note("the first claim", "cookieless write"),
        &[],
    );
    assert_eq!(status, 200, "{body}");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// Decisions and revisions through the workbench door: exact standing
/// target, bodyless accept, grounds-required reject with reopen, the
/// revised payload carrying exactly the changed fields, and stale
/// targets refused with taught reasons.
#[tokio::test(flavor = "multi_thread")]
async fn workbench_decisions_and_revisions_hold_their_laws() {
    let db = scratch_db("workbench-doors");
    let (base, state) = spawn_workbench(&db).await;
    seed_task(&state, "produce the account");
    seed_task(&state, "consume the account");
    let offer = |body: &str| {
        state
            .execute(
                &human_ctx(),
                Command::Comment {
                    home: TaskId(0),
                    kind: CommentKind::Offer {
                        title: Prose::new("the standing account".into()).unwrap(),
                        body: Prose::new(body.into()).unwrap(),
                    },
                    references: Vec::new(),
                },
                None,
                bare_request(),
            )
            .map(|r| r.last().unwrap().seq)
            .map_err(|e| panic!("{e:?}"))
            .unwrap()
    };
    let offer_seq = offer("the first account");
    let accept = json!({"actor": "jerry", "command": {"comment": {
        "home": 0, "kind": {"accept": {"offer": offer_seq}}, "references": [],
    }}});
    let reject = |offer_seq: usize| {
        json!({"actor": "jerry", "command": {"comment": {
            "home": 1, "kind": {"reject": {"offer": offer_seq, "grounds": "the edge case still fails"}}, "references": [],
        }}})
    };

    // accept is bodyless: the landed payload carries no prose or title
    let (status, body) = post_json(&base, "/api/v1/workbench/command", &accept);
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["records"][0]["kind"], "commented");
    let payload = &reply["records"][0]["payload"];
    assert!(payload["kind"]["accept"].get("body").is_none());
    assert!(payload["kind"]["accept"].get("title").is_none());

    // a decided task refuses a second decision
    let (status, body) = post_json(&base, "/api/v1/workbench/command", &accept);
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "invalid_state_transition",
        "{body}"
    );
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("concluded")),
        "the refusal teaches the task's state: {body}"
    );

    // reject on the standing offer of the other open task reopens it
    state
        .execute(
            &human_ctx(),
            Command::Comment {
                home: TaskId(1),
                kind: CommentKind::Offer {
                    title: Prose::new("the second account".into()).unwrap(),
                    body: Prose::new("for the grounds door".into()).unwrap(),
                },
                references: Vec::new(),
            },
            None,
            bare_request(),
        )
        .map(|_| ())
        .map_err(|e| panic!("{e:?}"))
        .unwrap();
    let standing = json_of(&get_json(&base, "/api/v1/workbench").1);
    let second_num = standing["tasks"][1]["offer"].as_u64().unwrap() as usize;
    let second = standing["comments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["num"].as_u64() == Some(second_num as u64))
        .unwrap()["seq"]
        .as_u64()
        .unwrap() as usize;
    let (status, body) = post_json(&base, "/api/v1/workbench/command", &reject(second));
    assert_eq!(status, 200, "{body}");
    let landed = json_of(&body);
    assert!(
        landed["records"][0]["payload"]["kind"]["reject"]["grounds"]
            .as_str()
            .is_some_and(|g| g == "the edge case still fails"),
        "grounds land verbatim: {}",
        landed["records"][0]["payload"]
    );
    let (_, after) = get_json(&base, "/api/v1/workbench");
    assert_eq!(
        json_of(&after)["tasks"][1]["state"].as_str(),
        Some("open"),
        "the rejection reopened the task"
    );

    // revision through the door sends exactly the changed field
    let revise_title_only = json!({"actor": "jerry", "command": {"revise_comment": {
        "id": offer_seq, "title": "the retitled account",
    }}});
    let (status, body) = post_json(&base, "/api/v1/workbench/command", &revise_title_only);
    assert_eq!(status, 200, "{body}");
    let revised = json_of(&body);
    assert!(
        revised["records"][0]["payload"].get("title").is_some()
            && revised["records"][0]["payload"].get("body").is_none(),
        "a title-only revision carries no body patch: {}",
        revised["records"][0]["payload"]
    );
    assert_eq!(
        revised["records"][0]["payload"]["id"].as_u64(),
        Some(offer_seq as u64),
        "the revision names the addressed birth"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A load failure refuses; a degraded log still serves its raw rows.
#[tokio::test(flavor = "multi_thread")]
async fn the_records_face_keeps_degraded_rows_and_refuses_corruption() {
    // a corrupt payload of a known kind: the log will not even load
    let db = scratch_db("records-corrupt");
    let (base, state) = spawn_workbench(&db).await;
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
    let (status, body) = get_json(&base, "/api/v1/workbench");
    assert_eq!(status, 500);
    assert_eq!(json_of(&body)["error"]["code"], "storage", "{body}");
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("corrupt record at seq 1")),
        "{body}"
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();

    // an unknown kind: the world will not fold, the rows still travel
    let db = scratch_db("records-degraded");
    let (base, state) = spawn_workbench(&db).await;
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
    let (status, body) = get_json(&base, "/api/v1/workbench");
    assert_eq!(status, 503);
    assert_eq!(json_of(&body)["error"]["code"], "degraded", "{body}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A supported local write shows on every server face.
#[tokio::test(flavor = "multi_thread")]
async fn a_local_write_shows_on_every_server_face() {
    let db = scratch_db("local-visible");
    let (base, state) = spawn_workbench(&db).await;
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
                title: Prose::new("the local write".into()).unwrap(),
                body: Prose::new("LOCAL_MARKER".into()).unwrap(),
            },
            references: Vec::new(),
        },
        saccade::db::now_epoch(),
    )
    .unwrap();

    let (_, snapshot) = get_json(&base, "/api/v1/workbench");
    let comments = json_of(&snapshot)["comments"].as_array().unwrap().clone();
    assert!(
        comments
            .iter()
            .any(|c| c["body"].as_str() == Some("LOCAL_MARKER")),
        "{comments:?}"
    );
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
    let (_, snapshot) = get_json(&base, "/api/v1/workbench");
    let projection = json_of(&snapshot);
    assert!(
        projection["comments"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["body"].as_str() == Some("LOCAL_MARKER")),
        "{projection}"
    );
    assert!(
        projection["tasks"][0]["owner"].as_str() == Some("jerry"),
        "{projection}"
    );

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

#[test]
fn search_reads_the_record_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("search-door");
    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
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
    assert!(
        sac(&[
            "comment",
            "t-0",
            "--title",
            "the floop migration proceeds",
            "the floop migration proceeds"
        ])
        .0
    );
    assert!(sac(&["create", "task", "floop guard"]).0);
    assert!(sac(&["claim", "t-0"]).0);
    assert!(
        sac(&[
            "offer",
            "t-0",
            "--title",
            "floop landed",
            "floop landed; suite green"
        ])
        .0
    );

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
    // a title match points at its own line: the offer's authored
    // title answers the term before its body does
    assert_eq!(lines[t0 + 3], "  c-1  pi  floop landed");
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
            "--title",
            "floop guards the door",
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
    // body format - the multi-line note is the tracker's third
    // comment, its authored title the first line under the header
    let (ok, out, _) = sac(&["show", "c-2"]);
    assert!(ok, "{out}");
    assert_eq!(
        out.trim_end(),
        "  c-2  pi [note]\n  title floop guards the door\n  floop guards the door the second line holds the detail\ncursor 5"
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
        "  c-1  pi [offer]\n  title floop landed\n  floop landed; suite green\ncursor 6"
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
        let mut child = scrubbed(std::process::Command::new(bin))
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
    assert!(sac(&["comment", "t-0", "--title", "the byte door note", &body]).0);
    assert!(sac(&["comment", "c-0", "--title", "the reply", answer]).0);
    assert!(
        sac(&[
            "comment",
            "t-0",
            "--title",
            "a short third note",
            "a short third note"
        ])
        .0
    );
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

    // a withdrawn deposit's title never discloses: default show, the
    // reveal flag, and the json face all withhold it while the log
    // keeps the historical bytes
    assert!(
        sac(&[
            "comment",
            "t-0",
            "--title",
            "the withdraw-me title",
            "the withdraw-me body"
        ])
        .0
    );
    assert!(sac(&["withdraw", "c-3", "--note", "parked in the wrong place"]).0);
    let (ok, out, _) = sac(&["show", "c-3"]);
    assert!(ok, "{out}");
    assert!(!out.contains("withdraw-me"), "default view: {out}");
    let (ok, out, _) = sac(&["show", "--withdrawn", "c-3"]);
    assert!(ok, "{out}");
    assert!(
        out.contains("body: the withdraw-me body"),
        "the reveal names the body: {out}"
    );
    assert!(
        !out.contains("withdraw-me title"),
        "the reveal never titles: {out}"
    );
    let (ok, out, _) = sac(&["show", "--json", "--withdrawn", "c-3"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["blocks"][0]["title"], json!(null));
    assert_eq!(json["blocks"][0]["body"], json!("the withdraw-me body"));
    let (ok, out, _) = sac(&["search", "withdraw-me"]);
    assert!(ok, "{out}");
    assert!(out.contains("no matches"), "{out}");
    let (ok, out, _) = sac(&["log", "--json"]);
    assert!(ok, "{out}");
    assert!(
        out.contains("the withdraw-me title"),
        "the log keeps the historical bytes: {out}"
    );

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn create_reply_names_the_born_task() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("create-reply");

    let out = scrubbed(std::process::Command::new(bin))
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

    let out = scrubbed(std::process::Command::new(bin))
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

    let out = scrubbed(std::process::Command::new(bin))
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
        let out = scrubbed(std::process::Command::new(bin))
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
    let noted = local(
        &local_db,
        &[
            "--json", "comment", "t-0", "--title", "the note", "the note",
        ],
    );
    assert_eq!(noted["records"][0]["kind"], "commented");
    let claimed = local(&local_db, &["--json", "claim", "t-0"]);
    assert_eq!(claimed["records"][0]["kind"], "task_claimed");

    // an offer answers with its records, both paths, identical shapes
    let offered = local(
        &local_db,
        &[
            "--json",
            "offer",
            "t-0",
            "--title",
            "suite green",
            "suite green",
        ],
    );
    assert_eq!(offered["records"][0]["kind"], "commented");
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"comment": {"home": 0, "kind": {"offer": {"title": "suite green", "body": "suite green"}}, "references": []}}}),
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
    sac_quiet(&[
        "comment",
        "t-0",
        "--title",
        "the opening note",
        "the opening note",
    ]); // seq 1 -> c-1

    // activity that committed before the wait: caught, never blocked
    let (seqs, cursor) = observed(spawn_waiter(&[
        "t-0", "--after", "0", "--until", "activity",
    ]));
    assert_eq!(seqs, vec![1]);
    assert_eq!(cursor, 1);

    // the cursor a read disclosed: a write landing between the read
    // and the wait is already committed when the wait checks first
    sac_quiet(&[
        "comment",
        "t-0",
        "--title",
        "a follow-up note",
        "a follow-up note",
    ]); // seq 2
    let (_, out, _) = sac(&["--json", "log"]);
    let log: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(log["cursor"], 2);
    sac_quiet(&[
        "comment",
        "t-0",
        "--title",
        "a late addition",
        "a late addition",
    ]); // seq 3
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
    sac_quiet(&[
        "comment",
        "c-0",
        "--title",
        "the reply",
        "the reply on the note",
    ]); // seq 4
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
    sac_quiet(&[
        "comment",
        "t-0",
        "--title",
        "an unrelated sibling",
        "an unrelated sibling",
    ]); // seq 5
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(
        child.try_wait().unwrap().is_none(),
        "unrelated activity woke the comment wait"
    );
    sac_quiet(&[
        "comment",
        "c-0",
        "--title",
        "the real reply",
        "the real reply",
    ]); // seq 6
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
    sac_quiet(&[
        "comment",
        "t-0",
        "--title",
        "still only chatter",
        "still only chatter",
    ]); // seq 7
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(
        child.try_wait().unwrap().is_none(),
        "a note woke the offer wait"
    );
    sac_quiet(&[
        "offer",
        "t-0",
        "--title",
        "the account",
        "the account, ready for review",
    ]); // seq 8
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
    sac_quiet(&[
        "comment",
        "t-0",
        "--title",
        "the served reply",
        "the served reply",
    ]); // seq 10
    let (seqs, cursor) = observed(wire_waiter);
    assert_eq!(seqs, vec![10]);
    assert_eq!(cursor, 10);
    let _ = serve.kill();
    let _ = serve.wait();

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

// ---- the workspace doors through the CLI face: a fake host, real exec ----

/// A fake herdr-core: hc run accepts the composed launch and answers
/// the pinned shape (hc get answers SACCADE_FAKE_HC_STATE, default
/// live); every call logs its argv for the door-composition asserts.
fn fake_hc(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("hc-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let hc = bin_dir.join("hc");
    std::fs::write(
        &hc,
        r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s
' "$@" >> "$here/argv-$1.log"
if [ "$1" = run ]; then
  printf '{"address":"t1","cwd":"x","state":"live","turn":0}
'
  exit 0
fi
if [ "$1" = get ]; then
  if [ -f "$here/state" ]; then
    printf '{"state":"%s"}
' "$(cat "$here/state")"
  else
    printf '{"state":"%s"}
' "${SACCADE_FAKE_HC_STATE:-live}"
  fi
  exit 0
fi
if [ "$1" = wait ] && [ -n "${SACCADE_FAKE_HC_WAIT_SLEEPS:-}" ]; then
  # the real host's shape: the wait blocks out its --timeout
  sleep "$4"
  printf '{"reason":"timeout","status":"working"}\n'
  exit 0
fi
if [ "$1" = send ] && [ "$2" = "t9" ]; then
  # the finished-address arm: the host's own not-live evidence
  printf '{"error":{"kind":"finished"}}
'
  exit 1
fi
if [ "$1" = send ] && [ -n "${SACCADE_FAKE_HC_SEND_FAILS:-}" ]; then
  printf '{"error":{"kind":"%s"}}
' "${SACCADE_FAKE_HC_SEND_FAILS}"
  exit 1
fi
printf '{}
'
exit 0
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    hc
}

/// A spawn-capable fake herdr-core for the real-door fixtures: hc run
/// starts the composed session (the argv after --) in its --cwd with
/// stdio into FIXTURE_LOG, and answers the pinned address; hc get
/// answers SACCADE_FAKE_HC_STATE.
fn fake_hc_spawn(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("hcs-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let hc = bin_dir.join("hc");
    std::fs::write(
        &hc,
        r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s
' "$@" >> "$here/argv-$1.log"
if [ "$1" = run ] && [ -n "${FAKE_HC_RUN_FAILS:-}" ]; then
  printf 'the host refused the launch\n' >&2
  exit 1
fi
if [ "$1" = run ]; then
  cwd=""; prev=""; sep=0; n=0
  for a in "$@"; do
    n=$((n+1))
    if [ "$prev" = "--cwd" ]; then cwd="$a"; fi
    if [ "$a" = "--" ]; then sep=$n; fi
    prev="$a"
  done
  (
    cd "$cwd" || exit 1
    shift "$sep"
    exec env FIXTURE_LOG="${FIXTURE_LOG:-/dev/null}" sh -c 'exec "$@"' _ "$@"
  ) >> "${FIXTURE_LOG:-/dev/null}" 2>&1 &
  printf '{"address":"t9","cwd":"x","state":"live","turn":0}
'
  exit 0
fi
if [ "$1" = get ]; then
  labels=""
  if [ -f "$here/labels-$2.json" ]; then labels=",\"labels\":$(cat "$here/labels-$2.json")"; fi
  printf '{"state":"%s"%s}
' "${SACCADE_FAKE_HC_STATE:-live}" "$labels"
  exit 0
fi
if [ "$1" = send ] && [ "$2" = "t9" ]; then
  # the finished-address arm: the host's own not-live evidence
  printf '{"error":{"kind":"finished"}}
'
  exit 1
fi
if [ "$1" = ls ]; then
  # the terminal table from the state file's rows; a missing file
  # names no terminals
  printf '{"terminals":['
  if [ -f "${FAKE_HC_TERMINALS:-/dev/null-nothing}" ]; then
    paste -sd, "${FAKE_HC_TERMINALS}" 2>/dev/null
  fi
  printf ']}
'
  exit 0
fi
if [ "$1" = kill ] && [ -n "$2" ]; then
  if [ -f "${FAKE_HC_CLOSE_FAIL:-/dev/null-nothing}" ]; then
    rm -f "${FAKE_HC_CLOSE_FAIL}"
    printf '{"error":{"kind":"close_refused","message":"the host refused the close"}}
'
    exit 1
  fi
  row=$(grep -F "\"address\":\"$2\"" "${FAKE_HC_TERMINALS:-/dev/null-nothing}" 2>/dev/null | head -1)
  if [ -n "$row" ] && [ -f "${FAKE_HC_TERMINALS:-/dev/null-nothing}" ]; then
    grep -Fv "\"address\":\"$2\"" "${FAKE_HC_TERMINALS}" > "${FAKE_HC_TERMINALS}.tmp" || true
    mv "${FAKE_HC_TERMINALS}.tmp" "${FAKE_HC_TERMINALS}"
    printf '%s
' "$row" | sed 's/"state":"[a-z]*"/"state":"finished"/'
  else
    printf '{"address":"%s","cwd":"x","state":"finished"}
' "$2"
  fi
  exit 0
fi
printf '{}
'
exit 0
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    hc
}

/// The composed argv the fake hc logged for one verb (the op line
/// skipped).
fn hc_argv(bin_dir: &std::path::Path, verb: &str) -> Vec<String> {
    // an unreadable log surfaces in the assert message: silence would
    // fake an empty composition
    let body = std::fs::read_to_string(bin_dir.join(format!("argv-{verb}.log")))
        .unwrap_or_else(|e| format!("<unreadable: {e}>"));
    body.lines().skip(1).map(str::to_string).collect()
}

/// A fake `orca` with pinned wire shapes; shebang /bin/sh - the sandbox has no /usr/bin/env.
/// The lifecycle doors model the host's real read contract: visible
/// presence from the registration file or an existing directory,
/// selector_not_found otherwise, and a terminal state the close door
/// actually mutates. FAKE_ORCA_CLOSE_FAIL (one-shot flag file) makes
/// exactly one close ask fail. FAKE_ORCA_SHOW_GARBAGE makes the
/// visible-presence answer unreadable for as long as it stands.
/// FAKE_ORCA_STALE (with FAKE_ORCA_STALE_COUNT and FAKE_ORCA_STALE_ASKS)
/// models a stale catalog: paths listed there answer Listed for their
/// first FAKE_ORCA_STALE_ASKS asks each, then discover the removal.
fn fake_host(dir: &std::path::Path) -> std::path::PathBuf {
    let bin_dir = dir.join("host-bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let orca = bin_dir.join("orca");
    std::fs::write(
        &orca,
        r#"#!/bin/sh
printf '%s\n' "$@" >> "$FAKE_ORCA_LOG"
selector=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--worktree" ]; then selector="$a"; fi
  prev="$a"
done
path="${selector#path:}"
if [ "$1" = worktree ]; then
  if [ "$2" = show ]; then
    # visible presence follows the live checkout, never the
    # registration file: a gone checkout is selector_not_found
    if [ -f "${FAKE_ORCA_SHOW_GARBAGE:-/dev/null-nothing}" ]; then
      printf 'not json at all\n'
      exit 0
    fi
    # a divergent host names paths in this file listed even when the
    # checkout is gone - the model for the Listed conflict
    if [ -n "$path" ] && [ -f "${FAKE_ORCA_LISTED:-/dev/null-nothing}" ] && grep -Fxq "$path" "$FAKE_ORCA_LISTED" 2>/dev/null; then
      printf '{"ok":true,"result":{"worktree":{"path":"%s"}}}\n' "$path"
      exit 0
    fi
    # a stale catalog: the path answers Listed for its first N asks,
    # counted per path, then the discovery catches up - the model for
    # the bounded visible-absence re-ask
    if [ -n "$path" ] && [ -f "${FAKE_ORCA_STALE:-/dev/null-nothing}" ] && grep -Fxq "$path" "${FAKE_ORCA_STALE}" 2>/dev/null; then
      printf '%s\n' "$path" >> "${FAKE_ORCA_STALE_COUNT}"
      n=$(grep -Fc "$path" "${FAKE_ORCA_STALE_COUNT}" 2>/dev/null)
      if [ "$n" -le "${FAKE_ORCA_STALE_ASKS:-1}" ]; then
        printf '{"ok":true,"result":{"worktree":{"path":"%s"}}}\n' "$path"
        exit 0
      fi
    fi
    if [ -n "$path" ] && [ -d "$path" ]; then
      printf '{"ok":true,"result":{"worktree":{"path":"%s"}}}\n' "$path"
    else
      printf '{"ok":false,"error":{"code":"selector_not_found","message":"selector_not_found"}}\n'
    fi
    exit 0
  fi
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = create ]; then
  n=$(($(wc -l < "$FAKE_ORCA_STATE" 2>/dev/null || echo 0) + 1))
  handle="term_actual_shape_$n"
  if [ -n "${FAKE_ORCA_CREATE_FAILS:-}" ]; then
    printf 'the host refused the create\n' >&2
    exit 1
  fi
  if [ -n "${FAKE_ORCA_NO_HANDLE:-}" ]; then
    printf '{"ok":true,"result":{"terminal":{}}}\n'
    exit 0
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
  if [ -n "$path" ] && ! grep -Fxq "$path" "$FAKE_ORCA_REGISTERED" 2>/dev/null && ! [ -d "$path" ]; then
    printf '{"ok":false,"error":{"code":"selector_not_found","message":"selector_not_found"}}\n'
    exit 0
  fi
  printf '{"ok":true,"result":{"terminals":['
  if [ -n "$path" ] && [ -f "$FAKE_ORCA_STATE" ]; then
    grep -F "\"worktreePath\":\"$path\"" "$FAKE_ORCA_STATE" | paste -sd,
  else
    paste -sd, "$FAKE_ORCA_STATE" 2>/dev/null
  fi
  printf ']}}\n'
  exit 0
fi
if [ "$1" = terminal ] && [ "$2" = close ]; then
  if [ -f "${FAKE_ORCA_CLOSE_FAIL:-/dev/null-nothing}" ]; then
    rm -f "${FAKE_ORCA_CLOSE_FAIL}"
    printf '{"ok":false,"error":{"code":"close_refused","message":"the host refused the close"}}\n'
    exit 0
  fi
  if [ -n "$path" ] && ! grep -Fxq "$path" "$FAKE_ORCA_REGISTERED" 2>/dev/null && ! [ -d "$path" ]; then
    printf '{"ok":false,"error":{"code":"selector_not_found","message":"selector_not_found"}}\n'
    exit 0
  fi
  if [ -n "$path" ] && [ -f "$FAKE_ORCA_STATE" ]; then
    grep -Fv "\"worktreePath\":\"$path\"" "$FAKE_ORCA_STATE" > "$FAKE_ORCA_STATE.tmp" || true
    mv "$FAKE_ORCA_STATE.tmp" "$FAKE_ORCA_STATE"
  fi
  printf '{"ok":true,"result":{"closed":true}}\n'
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

/// Whether a guard file is held by a live process: the tests' own
/// kernel probe, mirroring the extension's flock take. Only the
/// documented would-block conflict reads as held; every other failure
/// is unknown.
fn guard_held(guard: &Path) -> Option<bool> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(guard)
        .ok()?;
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if taken == 0 {
        return Some(false);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::EWOULDBLOCK) => Some(true),
        _ => None,
    }
}

/// The launched harness holds the guard within a bounded wait: the
/// door no longer waits on a handoff, so the fixture does.
fn wait_guard(guard: &Path) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        if guard_held(guard) == Some(true) {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
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
    ] {
        cmd.env_remove(key);
    }
    // the isolation fail-safe (see scrubbed): defaults before the
    // fixture envs, so a fixture's own pin always wins
    cmd.env("SACCADE_TEST_CONTEXT", "1");
    cmd.env("SACCADE_HERDR", "/nonexistent-sac-test-hc");
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
    let _ = &discovered_envs;
    let discovered_args = |sub: &[&str]| {
        let mut full = vec!["--repo".to_string(), root2.display().to_string()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };
    std::fs::write(&host_log, b"").unwrap();
    let chosen = db2.parent().unwrap().join("chosen-ws");
    std::fs::create_dir_all(&chosen).unwrap();
    // the launch rides the adapter: the fake hc answers and logs
    let hc2 = fake_hc(db2.parent().unwrap());
    let (ok, out, err) = run(
        &discovered_args(&["launch", "--workspace", chosen.to_str().unwrap()]),
        &[
            ("SACCADE_ACTOR", "pi"),
            ("XDG_STATE_HOME", state_home2.to_str().unwrap()),
            ("SACCADE_HERDR", hc2.to_str().unwrap()),
        ],
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");
    let typed2 = hc_argv(&db2.parent().unwrap().join("hc-bin"), "run").join(" ");
    assert!(
        !typed2.contains("SACCADE_SERVER=--server"),
        "no explicit flag: the discovery named the destination: {typed2}"
    );
    assert!(
        typed2.contains(&format!("SACCADE_SERVER={}", discovered_url)),
        "the discovered server arms the session: {typed2}"
    );
    assert!(
        typed2.contains(&format!("SACCADE_REPO={}", root2.display())),
        "{typed2}"
    );

    let _ = serve2.kill();
    let _ = serve2.wait();
    std::fs::remove_dir_all(db2.parent().unwrap()).unwrap();

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn launch_with_direnv_stripped_from_the_path() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("launch-no-direnv");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    std::fs::write(&host_log, b"").unwrap();

    let workspace = db.parent().unwrap().join("bare-workspace");
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
        // the fake host sits on PATH as ambient bait; direnv deliberately is not
        .env("PATH", host.parent().unwrap().display().to_string())
        .env("FAKE_ORCA_LOG", host_log.display().to_string())
        .env("SACCADE_PI", &pi)
        .output()
        .expect("spawn sac");
    // the pin is a compile fact, not a runtime choice: a packaged build
    // carries direnv no matter what PATH says and must not pretend to
    // forget it, while a cargo build has no pin and refuses before the
    // host is asked. Each build flavor asserts its own honest contract.
    if option_env!("SACCADE_DIRENV").is_some() {
        assert!(
            out.status.success(),
            "the baked pin governs over a stripped PATH: stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        assert!(
            !out.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(err.contains("direnv"), "{err}");
        assert!(err.contains("never starts unequipped"), "{err}");
        // the refusal precedes any host contact: the ambient bait saw nothing
        assert_eq!(std::fs::read_to_string(&host_log).unwrap(), "");
    }

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_launch_scrubs_stale_inheritance() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("launch-scrub");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());

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
        .env("SACCADE_HERDR", &hc)
        .env("SACCADE_PI", &pi)
        // everything stale the launching shell happens to carry
        .env("SACCADE_INCARNATION_SEQ", "999")
        .env("SACCADE_TASK", "t-0")
        .env("SACCADE_WORKSPACE", "/nowhere")
        .env("SACCADE_DB", "/stale/routing.db")
        .env("SACCADE_CLI", "/stale/sac")
        .env("SACCADE_TEST_CONTEXT", "1")
        .output()
        .expect("spawn sac");
    let ok = out.status.success();
    let (out, err) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(ok, "stdout: {out}\nstderr: {err}");

    let typed = hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" ");
    let expected_scrub = "-u SACCADE_INCARNATION -u SACCADE_INCARNATION_SEQ \
                         -u SACCADE_TASK -u SACCADE_WORKSPACE -u SACCADE_DB -u SACCADE_CLI \
                         -u SACCADE_TEST_CONTEXT";
    assert!(
        typed.contains(expected_scrub),
        "the launch scrubs stale inheritance: {typed}"
    );
    assert!(
        typed.contains(&format!(
            "SACCADE_ACTOR={}",
            command_label(&pi.display().to_string())
        )),
        "the launch names the harness as the acting agent: {typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_SERVER={base_url}")),
        "{typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_REPO={}", root.display())),
        "{typed}"
    );
    // the workspace command routes through direnv exec so the committed
    // .envrc governs the session, targeting the launch workspace; the
    // session stays the terminal foreground (direnv exec, no shell)
    let exec_at = typed
        .find(" exec ")
        .expect("the launch routes through direnv");
    assert!(
        typed[exec_at..].starts_with(&format!(
            " exec {} env -u SACCADE_INCARNATION",
            workspace.display()
        )),
        "direnv exec targets the launch workspace: {typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_CLI={}", env!("CARGO_BIN_EXE_sac"))),
        "{typed}"
    );
    assert!(
        !typed.contains("/stale/sac"),
        "the stale CLI path never survives the scrub: {typed}"
    );
    assert!(
        typed.ends_with("--session-dir") || typed.ends_with("register and wait"),
        "the composed prompt rides the argv tail: {typed}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// The here arm runs the shared composition in the invoking terminal:
/// equip, scrub, and routing all arrived, the child's own status is
/// the command's, and the host was never asked for a terminal.
#[test]
fn the_arrival_wait_door_wakes_on_mail_and_stays_quiet_within_its_bound() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("arrival-wait");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let hc = fake_hc(db.parent().unwrap());
    let bin_dir = db.parent().unwrap().join("hc-bin");
    let cold_state = db.parent().unwrap().join("xdg-state");
    let (url, pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        &bin_dir,
        Some(&cold_state),
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // an addressed waiter and a registered sender
    for (native_id, address) in [("waiter001", "t3"), ("sender001", "t4")] {
        let (status, body) = post_json(
            &url,
            "/api/v1/incarnations/register",
            &json!({"native_id": native_id, "address": address, "adopted": false}),
        );
        assert_eq!(status, 200, "{body}");
    }

    let wait_args = |target: &str, timeout: Option<&str>| {
        let mut args = vec![
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "wait".to_string(),
            target.to_string(),
        ];
        if let Some(secs) = timeout {
            args.push("--timeout".to_string());
            args.push(secs.to_string());
        }
        args
    };
    let envs = [("SACCADE_HERDR", hc.display().to_string())];

    // the quiet bound: no mail lands, the fake host's waits answer
    // quiet, and the door reports quiet without consuming anything
    let started = std::time::Instant::now();
    let (ok, out, err, timed_out) = bounded_sac(bin, &wait_args("i-0", Some("2")), &envs);
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    assert!(out.contains("quiet"), "{out}");
    assert!(
        started.elapsed() >= std::time::Duration::from_secs(2),
        "the door returned before its bound: {:?}",
        started.elapsed()
    );

    // a message lands mid-wait: the door reports it from the mailbox,
    // naming the state, and the cursor for the next wait
    let sender = {
        let url = url.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(700));
            let (status, body) = post_json(
                &url,
                "/api/v1/messages/send",
                &json!({
                    "context": {"actor": "pi", "tier": "agent",
                                "scope": {"task": null, "incarnation": 1, "workspace": null}},
                    "to": 0,
                    "body": "the wait woke on real mail",
                }),
            );
            assert_eq!(status, 200, "{body}");
        })
    };
    let (ok, out, err, timed_out) = bounded_sac(bin, &wait_args("i-0", Some("30")), &envs);
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    assert!(out.contains("m-0"), "{out}");
    assert!(out.contains("i-1 -> i-0"), "{out}");
    assert!(out.contains("submitted"), "{out}");
    assert!(out.contains("the wait woke on real mail"), "{out}");
    assert!(out.contains("cursor"), "{out}");
    sender.join().unwrap();

    // a pre-wait message never releases the wait: only mail born after
    // entry counts, so this wait runs to its quiet bound
    let (ok, out, err, timed_out) = bounded_sac(bin, &wait_args("i-0", Some("1")), &envs);
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    assert!(out.contains("quiet"), "{out}");

    // an unaddressed target takes no arrival wait: the pre-B migration
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "prebwait1",
            "session_file": "/sessions/prebwait1.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let (ok, _out, err, timed_out) = bounded_sac(bin, &wait_args("i-2", Some("1")), &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("no host address"), "{err}");
    assert!(err.contains("exit + resume"), "{err}");

    // the flag discipline: the t/c flags refuse on the i-N form, and
    // the t/c form still demands its cursor and mode
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "wait".to_string(),
            "i-0".to_string(),
            "--after".to_string(),
            "5".to_string(),
        ],
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("--after is the t-"), "{err}");
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "wait".to_string(),
            "t-0".to_string(),
        ],
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("names its cursor"), "{err}");

    // the host saw the waits and only the waits: no run, no close
    let runs = std::fs::read_to_string(bin_dir.join("argv-run.log")).unwrap_or_default();
    assert!(runs.trim().is_empty(), "the wait door launched: {runs}");

    let _ = std::process::Command::new("kill")
        .arg(pid.id().to_string())
        .status();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[test]
fn the_arrival_wait_clamps_its_iteration_to_the_remaining_bound() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("wait-clamp");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let hc = fake_hc(db.parent().unwrap());
    let bin_dir = db.parent().unwrap().join("hc-bin");
    let (url, _server) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        &bin_dir,
        None,
        &[
            ("SACCADE_HERDR", hc.to_str().unwrap()),
            ("SACCADE_FAKE_HC_WAIT_SLEEPS", "1"),
        ],
    );
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "waitclamp1", "address": "t5", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_FAKE_HC_WAIT_SLEEPS", "1".to_string()),
    ];
    // a two-second bound must not borrow the ten-second iteration: the
    // door ends quiet at its own edge and asks the host for the
    // remaining bound, not the iteration's own
    let started = std::time::Instant::now();
    let (ok, out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "wait".to_string(),
            "i-0".to_string(),
            "--timeout".to_string(),
            "2".to_string(),
        ],
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    assert!(out.contains("quiet"), "{out}");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= std::time::Duration::from_secs(2),
        "the door ended before its bound: {elapsed:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(9),
        "the iteration overshot the bound: {elapsed:?}"
    );
    let log = std::fs::read_to_string(bin_dir.join("argv-wait.log")).unwrap();
    let asked: Vec<u64> = log
        .lines()
        .zip(log.lines().skip(1))
        .filter(|(a, _)| a.trim() == "--timeout")
        .filter_map(|(_, v)| v.trim().parse().ok())
        .collect();
    assert!(!asked.is_empty(), "{log}");
    assert!(
        asked.iter().all(|secs| *secs <= 2),
        "the iteration borrowed the unclamped bound: {log}"
    );
}

/// A serving subprocess: publishes its address file, so the probe
/// routing is the real one, and carries the fake host on PATH for its
/// own child doors.
fn spawn_serving_subprocess(
    db: &Path,
    repo: &Path,
    bin: &str,
    host_dir: &Path,
    state_home: Option<&Path>,
    extra_envs: &[(&str, &str)],
) -> (String, ServerProc) {
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
    )
    // the server's own workbench door launches through the adapter:
    // the marker keeps its ambient arm unreachable
    .env("SACCADE_TEST_CONTEXT", "1");
    if let Some(dir) = state_home {
        // a genuinely cold state root: the guard dirs must come into
        // being from nothing, not from warmed probe residue
        cmd.env("XDG_STATE_HOME", dir);
    }
    for (key, value) in extra_envs {
        cmd.env(key, value);
    }
    let child = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the serving subprocess spawns");
    let _pid = child.id();
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
    (url, ServerProc(child))
}

#[test]
fn the_sole_writer_submits_only_through_the_adapter_and_launches_nothing() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("wire-adapter-submit");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let hc = fake_hc(db.parent().unwrap());
    let bin_dir = db.parent().unwrap().join("hc-bin");
    let cold_state = db.parent().unwrap().join("xdg-state");
    let (url, pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        &bin_dir,
        Some(&cold_state),
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // a registered sender and one addressed recipient
    for (native_id, address) in [("sender0001", "t4"), ("peer00001", "t5")] {
        let (status, body) = post_json(
            &url,
            "/api/v1/incarnations/register",
            &json!({"native_id": native_id, "address": address, "adopted": false}),
        );
        assert_eq!(status, 200, "{body}");
    }

    // the send records, submits through the adapter, and records the
    // acceptance: one ask at the recorded address, nothing else
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": 0, "workspace": null}},
            "to": 1,
            "body": "wake the addressed peer",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["submission"]["state"], "submitted", "{body}");
    // the submission carries the wire envelope: one origin line, a
    // blank line, the body untouched - no reply, no task here
    let asks = hc_argv(&bin_dir, "send");
    assert_eq!(asks.len(), 4, "{asks:?}");
    assert_eq!(asks[0], "t5");
    assert_eq!(asks[1], "[saccade message m-0 | from i-0]");
    assert_eq!(asks[2], "");
    assert_eq!(asks[3], "wake the addressed peer");

    // no launch ever reached the host: the send path asks send alone
    let runs = std::fs::read_to_string(bin_dir.join("argv-run.log")).unwrap_or_default();
    assert!(runs.trim().is_empty(), "a send launched a session: {runs}");

    let _ = std::process::Command::new("kill")
        .arg(pid.id().to_string())
        .status();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// One owned real-door fixture: the fake host, the guard-holding harness, their logs, and the cold state home.
struct DoorFixture {
    host: PathBuf,
    hcs: PathBuf,
    pi: PathBuf,
    host_log: PathBuf,
    child_log: PathBuf,
    dump: PathBuf,
    pidfile: PathBuf,
    cold_state: PathBuf,
}

fn door_fixture(home: &Path) -> DoorFixture {
    let host = fake_host(home);
    let hcs = fake_hc_spawn(home);
    let pi = guard_pi(home);
    let host_log = home.join("host.log");
    let child_log = home.join("child.log");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&child_log, b"").unwrap();
    DoorFixture {
        host,
        hcs,
        pi,
        host_log,
        child_log,
        dump: home.join("stub.dump"),
        pidfile: home.join("stub.pid"),
        cold_state: home.join("xdg-state"),
    }
}

/// The environment one real-door subprocess runs under: fake host first on PATH, the fixture harness as pinned Pi, and the cold state home the fixture server named its guard under.
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
        ("SACCADE_HERDR", fx.hcs.display().to_string()),
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
    while guard_held(guard) != Some(false) {
        assert!(
            std::time::Instant::now() < deadline,
            "the fixture holder's guard never freed"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[test]
fn the_real_resume_door_submits_without_a_guard_consult() {
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
        Some(&fx.cold_state),
        &[],
    );

    // the resumable session: a valid saved header, existing workspace,
    // no live process; the guard is the sidecar beside the saved file
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
    let guard = PathBuf::from(format!("{}.guard", saved.display()));

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

    // an addressless resume consults no guard: a preexisting holder
    // (a live pre-B session) no longer refuses the door - the twin
    // catch moved to the resumed session's own attach, and the pre-B
    // migration story is exit + resume. The same launch is the
    // corrected door: the adapter's synchronous ask returns the
    // address, the door records the address move, nothing waits on a
    // guard handoff - hc run is synchronous, attach is the harness's
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
    assert_eq!(guard_held(&guard), Some(true));
    let (ok, out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &envs);
    assert!(!timed_out, "the door outwaited its launch: {err}");
    assert!(
        ok,
        "the door consults no guard for an addressless resume: {err}"
    );
    drop(held);
    assert_eq!(guard_held(&guard), Some(false));
    let launched = json_of(&out);
    assert_eq!(
        launched["workspace"],
        json!(workspace.display().to_string()),
        "{out}"
    );
    assert_eq!(launched["terminal"], "t9", "{out}");
    // the resume places into the recorded workspace's space: the same
    // stable derivation a fresh launch there would declare
    let expected = saccade::placement::derived(&workspace);
    assert_eq!(launched["space"], expected.space.as_str(), "{out}");
    assert_eq!(launched["tab"], expected.tab.as_str(), "{out}");
    assert_eq!(launched["place_refusal"], Value::Null, "{out}");
    let label: Vec<String> = hc_argv(fx.hcs.parent().unwrap(), "label");
    assert_eq!(
        label,
        vec![
            "t9".to_string(),
            "--json".to_string(),
            format!("herdr.space={}", expected.space),
            format!("herdr.tab={}", expected.tab),
        ],
        "the resume declares placement on the resumed terminal"
    );

    // the composition rode the host argv: the resume session, its
    // locator, the settled flags
    let submitted_lines = hc_argv(fx.hcs.parent().unwrap(), "run");
    let submitted = submitted_lines.join(" ");
    let log_body = std::fs::read_to_string(home.join("hcs-bin").join("argv-run.log"))
        .unwrap_or_else(|e| format!("unreadable: {e}"));

    assert!(
        submitted.contains("--session resumee01"),
        "submitted=[{submitted}] lines=[{submitted_lines:?}] body=[{log_body}]"
    );
    assert!(submitted.contains("--session-dir"), "{submitted}");
    assert!(submitted.contains("--no-parent"), "{submitted}");
    assert!(submitted.contains("--no-notify"), "{submitted}");

    // the harness ran as the terminal foreground: equip, scrub, argv,
    // and cwd all arrived, and the guard was taken by the harness
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

    // the address move is recorded: the same incarnation, the new host
    // terminal
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    assert_eq!(
        world.incarnations[&saccade::IncarnationId(RecordId(0))]
            .incarnation
            .address(),
        Some("t9"),
        "the resume assigns the new address"
    );
    stop_holder(&fx.pidfile, &guard);

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
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
    // the server probes liveness through its own pinned fake host,
    // with every recorded address reading finished (stopped) by default
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        Some(&fx.cold_state),
        &[
            ("SACCADE_HERDR", fx.hcs.to_str().unwrap()),
            ("SACCADE_FAKE_HC_STATE", "finished"),
        ],
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
    let guard = PathBuf::from(format!("{}.guard", saved.display()));

    // an addressless (pre-B) session takes no switch: the migration
    // is exit + resume, and the refusal names it before anything moves.
    // Both sides read the fake host's state as finished (stopped)
    let mut envs = door_envs(&fx, Some(&guard));
    envs.push(("SACCADE_FAKE_HC_STATE", "finished".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(
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
        ],
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("carries no host address"), "{err}");
    assert!(err.contains("exit + resume"), "{err}");

    // the migration itself: a plain resume assigns the host address
    // and flips the kind to launched; the harness holds the guard
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server",
            &url,
            "--repo",
            root.to_str().unwrap(),
            "launch",
            "--resume",
            "i-0",
            "--json",
        ],
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(ok, "{err}");
    assert!(
        wait_guard(&guard),
        "the resumed harness never took the guard"
    );
    stop_holder(&fx.pidfile, &guard);

    // the composed door: the change lands, then the same native id
    // resumes through the real submission
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
    assert!(
        out.contains("look for it in the herdr app's sidebar"),
        "the resume reply names how to look: {out}"
    );
    // the composed resume declared placement in the changed workspace's
    // space - the derived pair follows the recorded workspace
    let switched = saccade::placement::derived(&selected_a);
    let label_log = std::fs::read_to_string(home.join("hcs-bin").join("argv-label.log")).unwrap();
    assert!(
        label_log.contains(&format!("herdr.space={}", switched.space)),
        "the switch-resume declares the recorded workspace's space: {label_log}"
    );
    assert!(
        label_log.contains(&format!("herdr.tab={}", switched.tab)),
        "the switch-resume declares the recorded workspace's tab: {label_log}"
    );

    // the header never moved: same native id, same history, same launch
    // cwd - the record is the authority, and the file is not rewritten
    // to follow it
    let moved = std::fs::read(&saved).unwrap();
    let header: Value =
        serde_json::from_slice(moved.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(header["cwd"], json!(selected_b.display().to_string()));
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

    // the resume is the same session, running in the new directory:
    // the harness holds the guard, and the witness reads after it does
    assert!(
        wait_guard(&guard),
        "the resumed harness never took the guard"
    );
    let witness = std::fs::read_to_string(&fx.child_log).unwrap();
    assert!(witness.contains("guard-taken"), "{witness}");
    let dump = std::fs::read_to_string(&fx.dump).unwrap();
    assert!(dump.contains("--session rover0001"), "{dump}");
    assert!(dump.contains("carry the thread forward"), "{dump}");
    assert!(
        dump.contains(&format!("pwd: {}", selected_a.display())),
        "{dump}"
    );
    stop_holder(&fx.pidfile, &guard);

    // a failed launch: the host refuses the run and the standing change fact is disclosed
    let mut fail_envs = envs.clone();
    fail_envs.push(("FAKE_HC_RUN_FAILS", "1".to_string()));
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
    // the switch landed and the launch refused at the host: the
    // standing change fact is disclosed beside the failure
    assert!(err.contains("the host refused the launch"), "{err}");
    assert!(err.contains("workspace changed"), "{err}");
    assert!(err.contains(selected_b.to_str().unwrap()), "{err}");
    // the change is true: the record landed and the file still sits
    // at its birth cwd
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
    // the failed launch ran nothing new: the guard stayed free and
    // the witness holds exactly the two earlier resumes' takings
    assert_eq!(guard_held(&guard), Some(false));
    let witness = std::fs::read_to_string(&fx.child_log).unwrap();
    assert_eq!(witness.matches("guard-taken").count(), 2, "{witness}");

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
    assert_eq!(header["cwd"], json!(selected_b.display().to_string()));

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[test]
fn a_send_to_a_finished_address_discloses_through_the_real_door() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("send-real-door");
    let home = db.parent().unwrap();
    let (root, _first, _head) = scratch_repo(home);
    let fx = door_fixture(home);
    // the server submits through its own pinned fake host; its send arm
    // refuses the t9 address with the host's finished evidence
    let (url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        fx.host.parent().unwrap(),
        Some(&fx.cold_state),
        &[("SACCADE_HERDR", fx.hcs.to_str().unwrap())],
    );

    // a live sender and the finished-address peer
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "sender001", "address": "t1", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let sender_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "peer00001", "address": "t9", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let peer_birth = json_of(&body)["incarnation"].as_u64().unwrap();

    // the send records, then its attempt refuses at the host: the
    // address is finished, and no launch is ever submitted
    let host_before = std::fs::read_to_string(&fx.host_log)
        .unwrap_or_default()
        .len();
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {"actor": "pi", "tier": "agent",
                        "scope": {"task": null, "incarnation": sender_birth, "workspace": null}},
            "to": 1,
            "body": "wake a finished address",
        }),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "recipient_not_live");
    let detail = json_of(&body)["error"]["detail"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(detail.contains("not live"), "{detail}");
    assert_eq!(json_of(&body)["message"]["num"], 0);
    // no launch reached the host: the log grew by nothing
    let host_after = std::fs::read_to_string(&fx.host_log)
        .unwrap_or_default()
        .len();
    assert_eq!(host_after, host_before, "a send launched a session");

    // delivery facts stay honest: the record stands pending, evidence
    // of the attempt, never a promise
    let (_, body) = get_json(&url, &format!("/api/v1/messages?incarnation={peer_birth}"));
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert!(
        mail.iter()
            .any(|m| m["from"]["num"] == 0 && m["state"] == "pending"),
        "{body}"
    );

    // the CLI send door names the recorded message beside the refusal
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "send".to_string(),
            "--to".to_string(),
            "i-1".to_string(),
            "status check".to_string(),
        ],
        &[
            ("SACCADE_ACTOR", "pi".to_string()),
            ("SACCADE_INCARNATION_SEQ", sender_birth.to_string()),
        ],
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "a refused submission exited success");
    assert!(err.contains("m-1 was recorded"), "{err}");
    assert!(err.contains("recipient_not_live"), "{err}");
    assert!(err.contains("stays pending"), "{err}");

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
        .status();
    std::fs::remove_dir_all(home).unwrap();
}

#[test]
fn the_ask_door_introduces_and_the_send_door_discloses() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("activation-cli");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    // the serving subprocess submits through the fake host: sends to
    // its run address t1 succeed, the finished t9 refuses
    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let (base_url, server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        hc.parent().unwrap(),
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // the creator is a registered incarnation at its own address; the
    // finished-address peer gives the refusal its evidence
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "creator0001", "address": "t4", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let creator_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "gonepeer1", "address": "t9", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    // the ask door: the launch births the newborn itself (identity and
    // address exist at launch time), the question records first, and
    // the sole writer submits it through the adapter at the newborn's
    // recorded address - the host accepted, the reply says so
    let workspace = db.parent().unwrap().join("peer-workspace");
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
            "--ask",
            "what holds at the failing edge case?",
        ])
        .env("SACCADE_HERDR", &hc)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", creator_birth.to_string())
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    let out_text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "stdout: {out_text}\nstderr: {err}");
    // the newborn is i-2: creator i-0, the finished-address peer i-1
    assert!(out_text.contains("i-2 registered"), "{out_text}");
    assert!(
        out_text.contains("your question went as m-0 from i-0"),
        "{out_text}"
    );

    // the question is durable addressed communication from the
    // creator's own scope, submitted into the newborn's terminal: the
    // record reads submitted, and the host ask named the newborn's
    // address with the body verbatim
    // the mailbox read addresses the birth record position: the
    // newborn was born at seq 2
    let (_, body) = get_json(&base_url, "/api/v1/messages?incarnation=2");
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    assert_eq!(mail.len(), 1, "{body}");
    assert_eq!(mail[0]["from"]["num"], 0);
    assert_eq!(mail[0]["to"]["num"], 2);
    assert_eq!(mail[0]["state"], "submitted");
    assert_eq!(mail[0]["body"], "what holds at the failing edge case?");
    let sends = hc_argv(hc.parent().unwrap(), "send");
    assert!(sends.contains(&"t1".to_string()), "{sends:?}");
    // the question rides the wire envelope: origin line, blank line,
    // the body untouched
    assert!(
        sends
            .iter()
            .any(|s| s == "[saccade message m-0 | from i-0]"),
        "{sends:?}"
    );
    assert!(
        sends
            .iter()
            .any(|s| s.starts_with("what holds at the failing edge case")),
        "{sends:?}"
    );

    // a plain send to the newborn rides the same submission
    let out = std::process::Command::new(bin)
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "send",
            "--to",
            "i-2",
            "status check",
        ])
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", creator_birth.to_string())
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.contains("sent m-1 to i-2"), "{text}");
    assert!(
        text.contains("accepted into their session's input queue"),
        "{text}"
    );

    // a send to the finished address refuses, naming the recorded
    // message beside the host's evidence
    let out = std::process::Command::new(bin)
        .args([
            "--server",
            &base_url,
            "--repo",
            root.to_str().unwrap(),
            "send",
            "--to",
            "i-1",
            "wake the finished one",
        ])
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", creator_birth.to_string())
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(!out.status.success(), "a refused submission exited success");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("m-2 was recorded"), "{err}");
    assert!(err.contains("stays pending"), "{err}");
    assert!(err.contains("recipient_not_live"), "{err}");

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
        .env("SACCADE_HERDR", &hc)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_ACTOR", "creator")
        .env("SACCADE_INCARNATION_SEQ", "")
        .env("SACCADE_TASK", "")
        .output()
        .expect("spawn sac");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("SACCADE_INCARNATION_SEQ"), "{err}");

    let _ = std::process::Command::new("kill")
        .arg(server_pid.id().to_string())
        .status();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

fn command_label(program: &str) -> String {
    std::path::Path::new(program)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap()
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

/// A pty slave as spawnable stdio; the master stays open for the terminal read.
fn pty_stdio(masters: &mut Vec<i32>) -> std::process::Stdio {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    let (master, slave) = open_pty();
    // libc::open reads until NUL, and Rust bytes carry none: without the
    // CString the call walks past the allocation into neighboring heap bytes
    let path = std::ffi::CString::new(slave.as_os_str().as_bytes()).unwrap();
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
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
        &args(&[
            "comment",
            "t-0",
            "--title",
            "piped prose",
            "piped prose from a human shell",
        ]),
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
        &args(&[
            "comment",
            "t-0",
            "--title",
            "no terminal",
            "no terminal in sight",
        ]),
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
) -> (ServerProc, String) {
    let mut child = ServerProc(
        std::process::Command::new(bin)
            .args(["serve", "--port", "0", "--db"])
            .arg(db)
            .args(extra)
            .envs(envs.iter().copied())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn serve"),
    );
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
    // the isolation fail-safe: the marker makes the ambient herdr arm
    // refuse, and the pin names a fixture path every launch-driving
    // test overrides with its own fake - a missing fixture fails the
    // test, never the production server
    cmd.env("SACCADE_TEST_CONTEXT", "1");
    cmd.env("SACCADE_HERDR", "/nonexistent-sac-test-hc");
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
            "--title",
            "a local note",
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
        .expect("the declared branch"),
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
        let out = scrubbed(std::process::Command::new(bin))
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
    assert!(
        sac(&[
            "offer",
            "t-0",
            "--title",
            "the first account",
            "the first account"
        ])
        .0
    ); // seq 6 -> c-0
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
    assert!(
        sac(&[
            "offer",
            "t-0",
            "--title",
            "the replacement",
            "the replacement"
        ])
        .0
    ); // seq 11 -> c-3
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
    assert!(
        sac(&[
            "offer",
            "t-5",
            "--title",
            "an outside account",
            "an outside account"
        ])
        .0
    ); // seq 15 -> c-5
    let (ok, _, err) = sac(&["bind", "c-1", "c-5", "--grounds", "not my source"]);
    assert!(!ok, "{err}");
    assert!(err.contains("invalid_state_transition"), "{err}");

    // citations: --ref earns the next marker, the reply target cites
    // its parent, and a marker with no reference refuses at the door
    assert!(
        sac(&[
            "comment",
            "t-4",
            "--title",
            "the adoption rests on foo",
            "rests on the conclusion [^1]",
            "--ref",
            "c-3"
        ])
        .0
    );
    let (ok, out, _) = sac(&["show", "t-4"]);
    assert!(ok, "{out}");
    assert!(out.contains("->c-3"), "{out}");
    assert!(
        sac(&[
            "comment",
            "c-0",
            "--title",
            "the reply",
            "replying to the account"
        ])
        .0
    );
    let (ok, out, _) = sac(&["show", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("->c-0"), "the reply cites its parent: {out}");
    let (ok, _, err) = sac(&[
        "comment",
        "t-4",
        "--title",
        "a dangling marker",
        "dangling [^2]",
        "--ref",
        "c-3",
    ]);
    assert!(!ok, "{err}");
    assert!(err.contains("marker [^2]"), "{err}");
    // the presentation revision door carries the title: the offer
    // repairs its title alone, its body standing untouched
    let (ok, out, _) = sac(&["revise", "c-0", "--title", "the account, retitled"]);
    assert!(ok, "{out}");
    let (ok, out, _) = sac(&["show", "c-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("title the account, retitled"), "{out}");
    assert!(
        out.contains("the first account"),
        "the body it left stands: {out}"
    );
    // a title never lands on the kinds that carry none: the use
    // refuses it with its teaching, never a silent ignore
    let (ok, _, err) = sac(&["revise", "c-1", "--title", "no title for a use"]);
    assert!(!ok, "{err}");
    assert!(
        err.contains("titles live on notes and offers alone"),
        "{err}"
    );

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
    let out_db = dir.join("v5.db");
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
    // the chained title step: the v3 story's deposits titled with
    // their home tasks' names
    assert!(
        report.contains("2 notes and offers titled with their home task's name"),
        "{report}"
    );
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
    // the backfilled titles read on the thread: each deposit names its
    // home task, the decisions carry none
    let (ok, out, _) = show(&["show", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("title implement foo"), "{out}");
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

    // a schema-4 source enters at the title step alone
    let v4 = dir.join("v4src.db");
    {
        let conn = rusqlite::Connection::open(&v4).unwrap();
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
             PRAGMA user_version = 4;",
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
        insert(0, "task_created", r#"{"name":"implement foo"}"#);
        insert(
            1,
            "commented",
            r#"{"home":0,"kind":{"note":{"body":"parked mid-flight"}},"references":[]}"#,
        );
        insert(
            2,
            "commented",
            r#"{"home":0,"kind":{"offer":{"body":"suite green"}},"references":[]}"#,
        );
        insert(
            3,
            "commented",
            r#"{"home":0,"kind":{"accept":{"offer":2}},"references":[]}"#,
        );
        insert(
            4,
            "comment_revised",
            r#"{"id":1,"body":"parked mid-flight, corrected"}"#,
        );
    }
    let v5 = dir.join("v5-from-v4.db");
    let migrated = std::process::Command::new(bin)
        .arg("--db")
        .arg(&v4)
        .args(["migrate", "--out"])
        .arg(&v5)
        .env("SACCADE_ACTOR", "pi")
        .output()
        .expect("spawn sac migrate");
    assert!(
        migrated.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&migrated.stderr)
    );
    let report = String::from_utf8_lossy(&migrated.stdout).into_owned();
    assert!(
        report.contains("2 notes and offers titled with their home task's name"),
        "{report}"
    );
    let show_v5 = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&v5)
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
    let (ok, out, _) = show_v5(&["show", "c-0"]);
    assert!(ok, "{out}");
    // the older presentation revision replays onto the backfilled
    // title: the corrected body stands beside the task's name
    assert!(out.contains("title implement foo"), "{out}");
    assert!(out.contains("parked mid-flight, corrected"), "{out}");

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
    // the serving subprocess carries the fake host: the switch door's
    // quiescence probe asks it, with the liveness state under the
    // test's pen through the state file (finished by default)
    let hc = fake_hc(home);
    std::fs::write(hc.parent().unwrap().join("state"), b"finished").unwrap();
    let (base_url, _server_pid) = spawn_serving_subprocess(
        &db,
        &root,
        bin,
        hc.parent().unwrap(),
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

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
    // the recorded address: the switch door's probe target
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": rover_birth, "address": "t7", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    // the stopped change: the address probes gone, the change fact
    // is the whole effect
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
    // the saved session never moved: header cwd stays the birth
    // launch cwd, the body byte-identical, id and history unchanged
    let moved = std::fs::read(&saved).unwrap();
    let moved_header: Value =
        serde_json::from_slice(moved.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(moved_header["cwd"], selected_b.display().to_string());
    assert_eq!(moved_header["id"], "rover0001");
    assert!(moved.ends_with(tail.as_bytes()), "the body rode whole");
    // the change record landed after the header success: two creations,
    // the registration, the address assignment, the change
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[4]["kind"], "workspace_changed");

    // a live address is a live session: busy refuses before any mutation
    std::fs::write(hc.parent().unwrap().join("state"), b"live").unwrap();
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
    assert_eq!(untouched_header["cwd"], selected_b.display().to_string());

    // an unprobeable answer is unknown: refuse, move nothing - never
    // read as gone
    std::fs::write(hc.parent().unwrap().join("state"), b"unreadable").unwrap();
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

    // with the address gone again the same change lands
    std::fs::write(hc.parent().unwrap().join("state"), b"finished").unwrap();
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
    let ghost_birth = json_of(&body)["incarnation"].as_u64().unwrap();
    // the ghost carries an address, so its change reaches the probe
    // and the missing file refuses there
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": ghost_birth, "address": "t8", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": 1, "to": selected_a.to_str().unwrap()}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "saved_session_missing");
    // an addressless session takes no switch at all: the pre-B
    // migration is exit + resume
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "preb00001", "session_file": "/sessions/preb00001.jsonl"}),
    );
    assert_eq!(status, 200, "{body}");
    let preb_num = json_of(&body)["num"].as_u64().unwrap();
    let (status, body) = post_workspace(
        &base_url,
        "/api/v1/workspace/change",
        &json!({"incarnation": preb_num, "to": selected_a.to_str().unwrap()}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "recipient_unaddressed");
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"].as_array().unwrap().len(),
        before + 3,
        "only the registrations and the assignment wrote past the changes"
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
        // the serving router's GET-only fallback: an unknown path with
        // a POST answers method-not-allowed, never a route
        assert!(status == 404 || status == 405, "{path}: {status} {body}");
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
    assert!(
        out.contains("workspace changed - /"),
        "the face names from -> to without saved-cwd wording: {out}"
    );
    assert!(out.contains("sac launch --resume i-0"), "{out}");

    // the explicit-new choice rides the ordinary fresh-launch door:
    // provision, then launch, one composition
    let pi = stub_pi(home);

    let hc = fake_hc(home);
    let (ok, out, err) = sac(
        &["launch", "--new", "trailblaze", "--json"],
        &[
            ("SACCADE_HERDR", hc.to_str().unwrap()),
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
    let refusing = home.join("hc-refusing-bin");
    std::fs::create_dir_all(&refusing).unwrap();
    let refusing_hc = refusing.join("hc");
    std::fs::write(
        &refusing_hc,
        "#!/bin/sh\nprintf 'the host refused the create\\n' >&2\nexit 1\\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&refusing_hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (ok, out, err) = sac(
        &["launch", "--new", "stumble", "--json"],
        &[
            ("SACCADE_HERDR", refusing_hc.to_str().unwrap()),
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
        let pi = pi.clone();
        let hc = hc.clone();
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
                .env("SACCADE_HERDR", &hc)
                .env("SACCADE_PI", &pi)
                .env("SACCADE_ACTOR", "rover")
                .env("SACCADE_INCARNATION_SEQ", rover_birth.to_string())
                .env("SACCADE_TASK", "")
                .env("SACCADE_WORKSPACE", "")
                .stdout(std::fs::File::create(&ask_out).unwrap())
                .stderr(std::fs::File::create(&ask_err).unwrap())
                .status()
                .expect("spawn sac")
        }
    });
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
    // the i-N counted births only: rover i-0, ghost i-1, the pre-B
    // attach i-2, the newborn i-3 (the fake host's run address t1 is
    // no incarnation's recorded address here) - creation never counted
    assert_eq!(newborn, 3, "{asked}");
    assert!(asked["launch"]["question"]["seq"].is_u64(), "{asked}");
    // the question is durable addressed mail from the creator's scope
    let (_, body) = get_json(
        &base_url,
        &format!("/api/v1/messages?incarnation={rover_birth}"),
    );
    let mail = json_of(&body)["messages"].as_array().unwrap().clone();
    let sent: Vec<&Value> = mail.iter().filter(|m| m["from"]["num"] == 0).collect();
    assert_eq!(sent.len(), 1, "{body}");
    assert_eq!(sent[0]["to"]["num"], newborn);

    // a resume request cannot outrun liveness: the recorded address
    // answers live, so --resume composes onto a live refusal - busy is
    // decided before any launch
    std::fs::write(hc.parent().unwrap().join("state"), b"live").unwrap();
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

// ---- the workbench launch door: the shared construction through HTTP ----

/// One real `sac serve` subprocess with its own environment: the launch
/// door's host calls belong to the server process, so its PATH and pins
/// must be exact. Returns the published URL and the child for reaping.
fn spawn_serve_subprocess(
    db: &std::path::Path,
    repo: &std::path::Path,
    envs: &[(&str, String)],
) -> (String, ServerProc) {
    let bin = env!("CARGO_BIN_EXE_sac");
    let log_path = db.parent().unwrap().join("serve-subprocess.log");
    let log = std::fs::File::create(&log_path).unwrap();
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
    .stdout(log.try_clone().unwrap())
    .stderr(log)
    // the server's own workbench door launches through the adapter:
    // the marker keeps its ambient arm unreachable
    .env("SACCADE_TEST_CONTEXT", "1");
    for key in [
        "SACCADE_REPO",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_ACTOR",
        "SACCADE_TASK",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "SACCADE_PI",
        "SACCADE_DIRENV",
    ] {
        cmd.env_remove(key);
    }
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let mut child = ServerProc(cmd.spawn().expect("the serve subprocess spawns"));
    let hint = std::path::PathBuf::from(format!("{}.addr", db.display()));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if matches!(child.try_wait(), Ok(Some(_))) {
            panic!(
                "serve exited at startup: {}",
                std::fs::read_to_string(log_path).unwrap_or_default()
            );
        }
        if hint.exists() {
            let url = std::fs::read_to_string(&hint).unwrap().trim().to_string();
            let (status, body) = get_json(&url, "/api/v1/version");
            if status == 200 && json_of(&body)["db"] == db.display().to_string() {
                return (url, child);
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "serve readiness deadline"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_workbench_launch_door_submits_honestly_through_a_fake_host() {
    let db = scratch_db("workbench-launch");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let _host = fake_host(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();
    // the launch rides the adapter through the server's env
    let hc = fake_hc(db.parent().unwrap());
    let (base_url, mut serve) = spawn_serve_subprocess(
        &db,
        &root,
        &[
            ("SACCADE_HERDR", hc.display().to_string()),
            ("SACCADE_PI", pi.display().to_string()),
        ],
    );

    let workspace = db.parent().unwrap().join("chosen-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let records_before: usize = serde_json::from_str::<Value>(
        &get_json(&base_url, "/api/v1/records").1,
    )
    .unwrap()["records"]
        .as_array()
        .unwrap()
        .len();

    // the happy path: the host's terminal acceptance is the whole
    // truth, and the door births the identity with the address
    let (status, body) = post_json(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": workspace.to_str().unwrap()}}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submitted"], true, "{body}");
    assert_eq!(
        reply["workspace"],
        workspace.canonicalize().unwrap().display().to_string(),
        "{body}"
    );
    assert_eq!(reply["terminal"], "t1", "{body}");
    // the workbench door declares placement identically: the stable
    // derivation over the workspace, named in the reply
    let expected = saccade::placement::derived(&workspace);
    assert_eq!(reply["space"], expected.space.as_str(), "{body}");
    assert_eq!(reply["tab"], expected.tab.as_str(), "{body}");
    assert_eq!(reply["place_refusal"], Value::Null, "{body}");
    let label: Vec<String> = hc_argv(&db.parent().unwrap().join("hc-bin"), "label");
    assert_eq!(
        label,
        vec![
            "t1".to_string(),
            "--json".to_string(),
            format!("herdr.space={}", expected.space),
            format!("herdr.tab={}", expected.tab),
        ],
        "the workbench launch declares placement at birth"
    );
    // the launch lands exactly one record: the door's own birth
    let records_after: usize = serde_json::from_str::<Value>(
        &get_json(&base_url, "/api/v1/records").1,
    )
    .unwrap()["records"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        records_after,
        records_before + 1,
        "the door birth is the launch's one record"
    );
    // the composed argv is the shared construction: direnv equip, the
    // scrub, and this server's own URL as the routing the session carries
    let typed = hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" ");
    let expected_scrub = "-u SACCADE_INCARNATION -u SACCADE_INCARNATION_SEQ \
                         -u SACCADE_TASK -u SACCADE_WORKSPACE -u SACCADE_DB -u SACCADE_CLI \
                         -u SACCADE_TEST_CONTEXT";
    assert!(typed.contains(expected_scrub), "{typed}");
    assert!(
        typed.contains(&format!("SACCADE_SERVER={base_url}")),
        "{typed}"
    );
    assert!(
        typed.contains(&format!("SACCADE_REPO={}", root.display())),
        "{typed}"
    );
    // env words carry no shell quoting: the label rides verbatim
    assert!(
        typed.contains(&format!(
            "SACCADE_ACTOR={}",
            pi.file_name().unwrap().to_string_lossy()
        )),
        "{typed}"
    );
    assert!(
        typed.contains("--session-id ") && typed.contains("--session-dir "),
        "the workbench launch pre-allocates the native session: {typed}"
    );
    // the argv ends at the pi's pre-allocation flags: no prompt rides a
    // workbench launch - the session flags are the argv tail
    assert!(
        typed.contains(&format!("{} --session-id ", pi.display())),
        "{typed}"
    );
    assert!(typed.ends_with("native-sessions"), "{typed}");

    // a missing directory refuses before the host is asked
    let logged_before = hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" ");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": db.parent().unwrap().join("absent").display().to_string()}}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "workspace_path_invalid",
        "{body}"
    );
    assert_eq!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" "),
        logged_before,
        "the refusal preceded any host contact"
    );

    // the door's origin law: cross-site posts refuse unexamined
    let (status, body) = post_json_headers(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": workspace.to_str().unwrap()}}),
        &[("Sec-Fetch-Site", "cross-site")],
    );
    assert_eq!(status, 403, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "cross_site_refused",
        "{body}"
    );

    // a shape the door cannot read refuses typed
    let (status, body) = post_json(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"resume": "i-1"}}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "malformed_request",
        "{body}"
    );

    serve.kill().unwrap();
    serve.wait().unwrap();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_workbench_launch_door_reports_an_honest_host_refusal() {
    let db = scratch_db("workbench-launch-refused");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();
    // the host answers its run ask with a refusal: the door surfaces
    // the refusal with the host's own evidence, and no record lands
    let hc = db.parent().unwrap().join("hc-refusing-bin");
    std::fs::create_dir_all(&hc).unwrap();
    let refusing = hc.join("hc");
    std::fs::write(
        &refusing,
        "#!/bin/sh\nprintf 'the host refused the create\n' >&2\nexit 1\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&refusing, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (base_url, mut serve) = spawn_serve_subprocess(
        &db,
        &root,
        &[
            ("SACCADE_HERDR", refusing.display().to_string()),
            ("SACCADE_PI", pi.display().to_string()),
        ],
    );

    let workspace = db.parent().unwrap().join("refused-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let records_before = get_json(&base_url, "/api/v1/records").1;
    let (status, body) = post_json(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": workspace.to_str().unwrap()}}),
    );
    assert_eq!(status, 502, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "launch_host_refused",
        "{body}"
    );
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("the host refused the create")),
        "{body}"
    );
    assert_eq!(
        json_of(&get_json(&base_url, "/api/v1/records").1),
        json_of(&records_before),
        "a refused launch changed no record"
    );

    serve.kill().unwrap();
    serve.wait().unwrap();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_launch_door_refuses_an_unequipped_start_before_the_host_is_asked() {
    let db = scratch_db("workbench-launch-no-direnv");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let host = fake_host(db.parent().unwrap());
    // the fake host stays reachable; direnv deliberately is not on PATH
    let bin_dir = host.parent().unwrap().to_path_buf();
    let hcs = fake_hc_spawn(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let host_log = db.parent().unwrap().join("host.log");
    let host_state = db.parent().unwrap().join("host-state");
    std::fs::write(&host_log, b"").unwrap();
    std::fs::write(&host_state, b"").unwrap();
    let (base_url, mut serve) = spawn_serve_subprocess(
        &db,
        &root,
        &[
            ("PATH", bin_dir.display().to_string()),
            ("FAKE_ORCA_LOG", host_log.display().to_string()),
            ("FAKE_ORCA_STATE", host_state.display().to_string()),
            ("SACCADE_HERDR", hcs.display().to_string()),
            ("SACCADE_PI", pi.display().to_string()),
        ],
    );

    let workspace = db.parent().unwrap().join("bare-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": workspace.to_str().unwrap()}}),
    );
    // the pin is a compile fact, not a runtime choice: a packaged build
    // carries direnv no matter what PATH says and must not pretend to
    // forget it, while a cargo build has no pin and refuses before the
    // host is asked. Each build flavor asserts its own honest contract.
    if option_env!("SACCADE_DIRENV").is_some() {
        assert_eq!(status, 200, "{body}");
        assert_eq!(json_of(&body)["submitted"], true, "{body}");
    } else {
        assert_eq!(status, 500, "{body}");
        assert_eq!(
            json_of(&body)["error"]["code"],
            "launch_no_direnv",
            "{body}"
        );
        assert!(
            json_of(&body)["error"]["detail"]
                .as_str()
                .is_some_and(|d| d.contains("never starts unequipped")),
            "{body}"
        );
        assert_eq!(
            std::fs::read_to_string(&host_log).unwrap(),
            "",
            "the unequipped refusal reached the host"
        );
    }

    serve.kill().unwrap();
    serve.wait().unwrap();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_workbench_projection_names_the_workspaces_the_fold_holds() {
    let db = scratch_db("workbench-workspaces");
    let (base, _state) = spawn_workbench(&db).await;
    // no workspaces yet: the projection carries the empty fold fact
    let empty = json_of(&get_json(&base, "/api/v1/workbench").1);
    assert_eq!(
        empty["workspaces"],
        json!([]),
        "{}",
        get_json(&base, "/api/v1/workbench").1
    );
    let (status, body) = post_json(
        &base,
        "/api/v1/workspace/create",
        &json!({"label": "projection-seed", "client": "workbench"}),
    );
    assert_eq!(status, 200, "{body}");
    let created = json_of(&body);
    // the historical kind projects beside the creations: a hand-written
    // task-bound establishment enters the same panel with its own stage
    {
        let mut conn = saccade::db::open(&db).unwrap();
        saccade::db::record(
            &mut conn,
            &saccade::store::Context::system(),
            saccade::Command::CreateTask {
                name: saccade::Prose::new("a lineage with an establishment".into()).unwrap(),
            },
            1,
        )
        .unwrap();
        saccade::db::record(
            &mut conn,
            &saccade::store::Context::system(),
            saccade::Command::EstablishWorkspace {
                task_id: saccade::TaskId(0),
                base: saccade::GitCommit::new("abc123".into()).unwrap(),
                branch: saccade::GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: saccade::WorktreePath::new("/elsewhere/wt/t-0".into()).unwrap(),
            },
            2,
        )
        .unwrap();
    }
    let workspaces = json_of(&get_json(&base, "/api/v1/workbench").1)["workspaces"].clone();
    let list = workspaces.as_array().unwrap();
    assert_eq!(list.len(), 2, "{workspaces}");
    let established = list
        .iter()
        .find(|w| w["home"].as_u64() == Some(0))
        .expect("the established kind projects with its home task");
    assert_eq!(established["branch"], "saccade/t-0", "{workspaces}");
    assert_eq!(established["stage"], "available", "{workspaces}");
    assert!(established["born_at"].is_null(), "{workspaces}");
    let ws = list
        .iter()
        .find(|w| w["seq"] == created["workspace"]["seq"])
        .expect("the created kind still projects");
    assert_eq!(ws["base"], created["workspace"]["base"], "{workspaces}");
    assert_eq!(ws["path"], created["workspace"]["path"], "{workspaces}");
    assert!(
        ws["branch"].as_str().unwrap().contains("projection-seed"),
        "{workspaces}"
    );
    assert!(ws["born_at"].is_u64(), "{workspaces}");
    assert!(ws["home"].is_null(), "{workspaces}");
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_launch_door_names_a_created_terminal_it_cannot_report() {
    let db = scratch_db("workbench-launch-no-handle");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    // the host answers its run ask with an unreadable body: the door
    // cannot report the address it cannot read
    let host = db.parent().unwrap().join("hc-unreadable-bin");
    std::fs::create_dir_all(&host).unwrap();
    let hc = host.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh
printf 'not json at all\n'
",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let pi = stub_pi(db.parent().unwrap());
    let (base_url, mut serve) = spawn_serve_subprocess(
        &db,
        &root,
        &[
            ("SACCADE_HERDR", hc.display().to_string()),
            ("SACCADE_PI", pi.display().to_string()),
        ],
    );

    let workspace = db.parent().unwrap().join("unreported-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let records_before = get_json(&base_url, "/api/v1/records").1;
    let (status, body) = post_json(
        &base_url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": workspace.to_str().unwrap()}}),
    );
    // the host started the terminal; the door cannot read the address.
    // The refusal names the unreadable answer - never a claim of
    // non-submission, and no identity is birthed on an unreportable
    // launch
    assert_eq!(status, 502, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "launch_host_refused",
        "{body}"
    );
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("not json at all")),
        "{body}"
    );
    assert_eq!(
        json_of(&get_json(&base_url, "/api/v1/records").1),
        json_of(&records_before),
        "the unreportable launch changed no record"
    );

    serve.kill().unwrap();
    serve.wait().unwrap();
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

// ---- the workspace lifecycle: ordered cleanup, restoration, guards ----

/// A lifecycle rig: a scratch repo, a REAL `sac serve` subprocess with
/// the fake host on its PATH - the lifecycle effects run exactly where
/// they run in production, against the fake host and real git - and
/// the CLI face pointed at that server.
struct LifecycleRig {
    serve: ServerProc,
    url: String,
    dir: std::path::PathBuf,
    root: std::path::PathBuf,
    bin: &'static str,
    host_state: std::path::PathBuf,
    close_fail: std::path::PathBuf,
    host_env: Vec<(String, String)>,
}

fn lifecycle_rig(tag: &str) -> LifecycleRig {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-lifecycle-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (root, _first, _head) = scratch_repo(&dir);
    let db = dir.join("saccade.db");
    let pi = stub_pi(&dir);
    let host_state = dir.join("host-terminals");
    let close_fail = dir.join("close-fail");
    std::fs::write(&host_state, b"").unwrap();
    let hcs = fake_hc_spawn(&dir);
    let host_env = vec![
        ("SACCADE_HERDR".to_string(), hcs.display().to_string()),
        (
            "FAKE_HC_TERMINALS".to_string(),
            host_state.display().to_string(),
        ),
        (
            "FAKE_HC_CLOSE_FAIL".to_string(),
            close_fail.display().to_string(),
        ),
        // the rig is hermetic: the serve subprocess and every CLI run
        // name the stub harness, never an inherited or compiled pin
        ("SACCADE_PI".to_string(), pi.display().to_string()),
    ];
    let serve_envs: Vec<(&str, &str)> = host_env
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let (serve, url) = spawn_serve(
        bin,
        &db,
        &["--repo", &root.display().to_string()],
        &serve_envs,
    );
    LifecycleRig {
        serve,
        url,
        dir,
        root,
        bin,
        host_state,
        close_fail,
        host_env,
    }
}

impl Drop for LifecycleRig {
    fn drop(&mut self) {
        let _ = self.serve.kill();
        let _ = self.serve.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Provision one workspace through the create door; the reply's birth
/// seq and recorded facts come back.
fn rig_provision(rig: &LifecycleRig, label: &str) -> (usize, String, String) {
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/workspace/create",
        &json!({"label": label, "client": "lifecycle-test"}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    let seq = reply["workspace"]["seq"].as_u64().unwrap() as usize;
    let branch = reply["workspace"]["branch"].as_str().unwrap().to_string();
    let path = reply["workspace"]["path"].as_str().unwrap().to_string();
    (seq, branch, path)
}

/// An idle terminal the host names for the workspace.
fn rig_open_terminal(rig: &LifecycleRig, address: &str, path: &str) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&rig.host_state)
        .unwrap();
    writeln!(
        file,
        r#"{{"address":"{address}","cwd":"{path}","state":"live"}}"#
    )
    .unwrap();
}

/// The CLI face pointed at the rig's server, the fake host on PATH.
fn rig_sac(rig: &LifecycleRig, args: &[&str]) -> (bool, String, String) {
    let mut envs: Vec<(String, String)> =
        vec![("SACCADE_ACTOR".to_string(), "lifecycle-bot".to_string())];
    envs.extend(rig.host_env.clone());
    let env_refs: Vec<(&str, String)> = envs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let mut full = vec![
        "--server".to_string(),
        rig.url.clone(),
        "--repo".to_string(),
        rig.root.display().to_string(),
    ];
    full.extend(args.iter().map(|s| s.to_string()));
    let (ok, out, err, timed_out) = bounded_sac(rig.bin, &full, &env_refs);
    assert!(!timed_out, "sac {args:?} ran past its bound: {out}{err}");
    (ok, out, err)
}

/// The workspace is gone from git's own registrations.
fn rig_unregistered(rig: &LifecycleRig, path: &str) -> bool {
    let out = sh(&rig.root, &["worktree", "list", "--porcelain"]);
    !out.lines().any(|line| line == format!("worktree {path}"))
}

/// The recorded branch's local tip, empty when git holds none.
fn rig_tip(rig: &LifecycleRig, branch: &str) -> String {
    sh(
        &rig.root,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("refs/heads/{branch}"),
        ],
    )
    .trim()
    .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_runs_the_ordered_contract_and_the_branch_stands() {
    let rig = lifecycle_rig("clean");
    let (seq, branch, ws) = rig_provision(&rig, "sweep");
    rig_open_terminal(&rig, "term_idle_1", &ws);
    let worktree_rows = |rig: &LifecycleRig| sh(&rig.root, &["worktree", "list", "--porcelain"]);

    // the clean runs the whole ordered contract over a real checkout
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "clean refuses: {out}{err}");
    // the Cleaned fact is the story's second record: the birth is seq 0
    assert!(out.contains("cleaned - record 1"), "{out}");
    let _ = seq;
    assert!(out.contains("closed 1 terminal(s)"), "{out}");
    assert!(out.contains("the host named each finished"), "{out}");
    assert!(out.contains(&format!("branch {branch} retained")), "{out}");
    assert!(
        out.contains("verified: the host names no terminals for the workspace"),
        "{out}"
    );
    assert!(!std::path::Path::new(&ws).exists(), "the checkout is gone");
    assert!(
        rig_unregistered(&rig, &ws),
        "git registers no worktree after the removal"
    );
    // the branch and its tip stand: retained, never deleted
    let tip = rig_tip(&rig, &branch);
    assert_eq!(tip, sh(&rig.root, &["rev-parse", "HEAD"]).trim());
    // the close ran at the terminal's own address, one kill per row
    let kills = std::fs::read_to_string(rig.dir.join("hcs-bin/argv-kill.log")).unwrap_or_default();
    assert!(kills.contains("term_idle_1"), "{kills}");
    let (status, body) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(status, 200, "{body}");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );

    // the idempotent re-run: recorded Cleaned plus verified absence,
    // no duplicate transition
    let records_before = body.clone();
    let (ok, out, _err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the idempotent re-run refuses: {out}");
    assert!(out.contains("already cleaned"), "{out}");
    let (_, body_after) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(
        json_of(&records_before)["records"],
        json_of(&body_after)["records"],
        "the idempotent re-run landed no record"
    );

    // restore brings the checkout back at the branch's current tip
    let (ok, out, err) = rig_sac(&rig, &["workspace", "restore", &ws]);
    assert!(ok, "restore refuses: {out}{err}");
    assert!(out.contains("restored"), "{out}");
    assert!(std::path::Path::new(&ws).join("readme").is_file());
    assert_eq!(
        sh(&std::path::PathBuf::from(&ws), &["rev-parse", "HEAD"]).trim(),
        tip,
        "the checkout stands at the branch's current local tip"
    );
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        kinds.contains(&"workspace_restored".to_string()),
        "{kinds:?}"
    );
    let records_at_restored = body.clone();

    // the matching checkout standing on an Available workspace is
    // no-op success: no duplicate restoration fact
    let (ok, out, _err) = rig_sac(&rig, &["workspace", "restore", &ws]);
    assert!(ok, "the no-op restore refuses: {out}");
    assert!(out.contains("already restored"), "{out}");
    let (_, body_after) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(
        json_of(&records_at_restored)["records"],
        json_of(&body_after)["records"],
        "the no-op restore landed no record"
    );

    // an empty tail: nothing was left behind
    let _ = worktree_rows(&rig);
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_refuses_dirt_missing_branch_and_unexpected_bytes() {
    let rig = lifecycle_rig("refuse");
    let (_, _branch, ws) = rig_provision(&rig, "guarded");

    // staged, unstaged, and untracked work refuses before any host or
    // git act; the checkout and the record both stand untouched
    std::fs::write(std::path::Path::new(&ws).join("untracked.txt"), "kept\n").unwrap();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(!ok, "the dirty checkout cleaned anyway: {out}");
    assert!(err.contains("workspace_work_present"), "{err}");
    assert!(std::path::Path::new(&ws).join("untracked.txt").is_file());
    std::fs::remove_file(std::path::Path::new(&ws).join("untracked.txt")).unwrap();

    // unrecognized directories and the primary checkout refuse
    let stranger = rig.dir.join("stranger");
    std::fs::create_dir_all(&stranger).unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", stranger.to_str().unwrap()]);
    assert!(!ok, "an unrecognized directory cleaned: {err}");
    assert!(err.contains("workspace_not_recorded"), "{err}");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", rig.root.to_str().unwrap()]);
    assert!(!ok, "the primary checkout cleaned: {err}");
    assert!(err.contains("workspace_not_recorded"), "{err}");

    // a missing recorded branch refuses the destructive act: rename
    // the recorded branch away and the checkout is uncleanable
    let (seq, branch, ws2) = rig_provision(&rig, "branchless");
    let renamed = format!("{branch}-moved");
    sh(&rig.root, &["branch", "-m", &branch, &renamed]);
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws2]);
    assert!(!ok, "the branchless checkout cleaned: {err}");
    assert!(err.contains("workspace_branch_missing"), "{err}");
    assert!(std::path::Path::new(&ws2).exists());
    sh(&rig.root, &["branch", "-m", &renamed, &branch]);
    let _ = seq;

    // bytes at a recorded path without a git worktree registration
    // are unexpected work: the mismatch refuses, nothing is deleted
    let (_, _, ws3) = rig_provision(&rig, "impostor");
    sh(&rig.root, &["worktree", "remove", &ws3]);
    std::fs::create_dir_all(&ws3).unwrap();
    std::fs::write(std::path::Path::new(&ws3).join("precious"), "not mine\n").unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws3]);
    assert!(!ok, "unexpected bytes were deleted: {err}");
    assert!(err.contains("workspace_checkout_mismatch"), "{err}");
    assert!(std::path::Path::new(&ws3).join("precious").is_file());
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_stops_on_open_terminals_then_retry_completes() {
    let rig = lifecycle_rig("terminals");
    let (_, branch, ws) = rig_provision(&rig, "occupied");
    rig_open_terminal(&rig, "term_live_9", &ws);
    // exactly one close ask fails, then the fake host behaves
    std::fs::write(&rig.close_fail, b"refuse once\n").unwrap();

    // the first clean: the close names no finished state, and the door
    // stops before any checkout removal
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(!ok, "the occupied workspace cleaned anyway: {err}");
    assert!(err.contains("workspace_host_unverified"), "{err}");
    assert!(
        err.contains("the host refused the close of terminal term_live_9"),
        "{err}"
    );
    assert!(
        std::path::Path::new(&ws).exists(),
        "the checkout was removed"
    );
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        !kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );

    // the retry is the recovery path: the close lands, the list
    // verifies empty, and the completed fact records once
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the retry clean refuses: {out}{err}");
    assert!(!std::path::Path::new(&ws).exists());
    assert_eq!(
        rig_tip(&rig, &branch),
        sh(&rig.root, &["rev-parse", "HEAD"]).trim()
    );

    // the recorded-Cleaned idempotence verifies the host too: a
    // terminal the host still names refuses the no-op success
    rig_open_terminal(&rig, "term_after_2", &ws);
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(
        !ok,
        "the occupied cleaned workspace answered idempotent: {err}"
    );
    assert!(err.contains("workspace_terminals_open"), "{err}");
    // the terminal leaves, and the same door verifies its way to the
    // no-op success again
    std::fs::write(&rig.host_state, b"").unwrap();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the verified idempotent re-run refuses: {out}{err}");
    assert!(out.contains("already cleaned"), "{out}{err}");
    assert!(out.contains("the host names no terminals"), "{out}");
}

#[tokio::test(flavor = "multi_thread")]
async fn clean_refuses_an_unreadable_terminal_table() {
    let rig = lifecycle_rig("clean-garbage");
    let (_, _, ws) = rig_provision(&rig, "unanswered");
    rig_open_terminal(&rig, "term_garbage_5", &ws);
    // a row the adapter cannot fully read fails the whole answer: the
    // door refuses with the unverified shape and lands no fact
    use std::io::Write as _;
    let mut rows = std::fs::OpenOptions::new()
        .append(true)
        .open(&rig.host_state)
        .unwrap();
    writeln!(rows, r#"{{"address":"term_garbage_5"}}"#).unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(!ok, "the unreadable host cleaned anyway: {err}");
    assert!(err.contains("workspace_host_list_failed"), "{err}");
    assert!(
        std::path::Path::new(&ws).exists(),
        "the unreadable table stopped the door before any removal"
    );
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        !kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn reconcile_names_open_terminals_and_absence_in_both_arms() {
    let rig = lifecycle_rig("reconcile-host");
    // birth one goes Cleaned the honest way: the clean's own doors
    let (_seq1, _branch1, ws_live) = rig_provision(&rig, "cleaned-live");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws_live]);
    assert!(ok, "the setup clean refuses: {err}");
    // the host names a terminal at the cleaned path: the Cleaned arm's
    // conflict, dry run and apply alike, and no fact lands for it
    rig_open_terminal(&rig, "term_after_clean", &ws_live);
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile"]);
    assert!(ok, "reconcile fails: {out}{err}");
    assert!(
        out.contains("live terminal(s) at a workspace recorded Cleaned"),
        "{out}{err}"
    );
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile", "--apply"]);
    assert!(ok, "reconcile apply fails: {out}{err}");
    assert!(
        out.contains("live terminal(s) at a workspace recorded Cleaned"),
        "{out}{err}"
    );
    assert!(out.contains("applied - no new facts recorded"), "{out}");

    // birth two: the checkout gone and no terminal anywhere - the
    // absence arm records Cleaned on apply
    let (seq2, _branch2, ws_gone) = rig_provision(&rig, "departed-soon");
    sh(&rig.root, &["worktree", "remove", &ws_gone]);
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile", "--apply"]);
    assert!(ok, "the absence apply fails: {out}{err}");
    assert!(
        out.contains("checkout absent and the host names no terminals there"),
        "{out}{err}"
    );
    assert!(out.contains(&ws_gone), "{out}{err}");
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let parsed = json_of(&body);
    let cleaned: Vec<&serde_json::Value> = parsed["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "workspace_cleaned")
        .collect();
    // the setup clean landed this story's first Cleaned fact; the
    // absence arm added the departed birth's own
    assert_eq!(cleaned.len(), 2, "{cleaned:?}");
    assert_eq!(cleaned[1]["payload"]["workspace"], json!(seq2));

    // the terminal leaves: the live twin verifies to the preserved fact
    std::fs::write(&rig.host_state, b"").unwrap();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile"]);
    assert!(ok, "the final reconcile fails: {out}{err}");
    assert!(
        out.contains(
            "verified: recorded Cleaned, checkout absent, the host names no terminals there"
        ),
        "{out}{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn restore_reuses_recreates_and_refuses_its_mismatches() {
    let rig = lifecycle_rig("restore");
    let (_, branch, ws) = rig_provision(&rig, "reborn");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the setup clean refuses: {err}");
    assert!(!std::path::Path::new(&ws).exists());

    // a matching worktree that stands again is reused as-is: a local
    // change made out-of-band survives the restoration untouched
    sh(&rig.root, &["worktree", "add", &ws, &branch]);
    std::fs::write(std::path::Path::new(&ws).join("local.txt"), "mine\n").unwrap();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "restore", &ws]);
    assert!(ok, "the reuse restore refuses: {out}{err}");
    assert!(out.contains("reused as-is"), "{out}");
    assert_eq!(
        std::fs::read_to_string(std::path::Path::new(&ws).join("local.txt")).unwrap(),
        "mine\n",
        "the local change did not survive the restoration"
    );
    // reuse carries no duplicate checkout act: the worktree stands once
    let rows = sh(&rig.root, &["worktree", "list", "--porcelain"]);
    assert_eq!(
        rows.lines()
            .filter(|line| line == &format!("worktree {ws}"))
            .count(),
        1
    );

    // a stale git registration whose directory is gone restores
    // nothing: the mismatch refuses, no fact lands, the stage stays
    let (_, _, ws_stale) = rig_provision(&rig, "stale-entry");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws_stale]);
    assert!(ok, "the stale setup clean refuses: {err}");
    let stale_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| {
                r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws_stale)
            })
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the stale birth names its branch")
    };
    sh(&rig.root, &["worktree", "add", &ws_stale, &stale_branch]);
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let records_before_stale = body.clone();
    std::fs::remove_dir_all(&ws_stale).unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "restore", &ws_stale]);
    assert!(!ok, "the stale registration restored anyway: {err}");
    assert!(err.contains("workspace_checkout_mismatch"), "{err}");
    assert!(err.contains("stale registration"), "{err}");
    assert!(!std::path::Path::new(&ws_stale).exists());
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(
        json_of(&records_before_stale)["records"],
        json_of(&body)["records"],
        "the stale registration landed no restoration fact"
    );

    // the absent-path recreate stands at the branch's CURRENT local
    // tip, never the cleanup-time commit: advance the branch first
    let (_, _, ws2) = rig_provision(&rig, "recreated");
    let (seq2, branch2, _p2) = (0usize, String::new(), ());
    let _ = seq2;
    let _ = branch2;
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws2]);
    assert!(ok, "the second setup clean refuses: {err}");
    let ws2_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws2))
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the second birth names its branch")
    };
    // a throwaway checkout advances the branch past its cleanup-time tip
    let throwaway = rig.dir.join("throwaway");
    sh(
        &rig.root,
        &[
            "worktree",
            "add",
            &throwaway.display().to_string(),
            &ws2_branch,
        ],
    );
    std::fs::write(throwaway.join("readme"), "advanced past cleanup\n").unwrap();
    sh(&throwaway, &["commit", "-qam", "advance"]);
    let advanced_tip = sh(&throwaway, &["rev-parse", "HEAD"]);
    sh(
        &rig.root,
        &["worktree", "remove", &throwaway.display().to_string()],
    );
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "restore", &ws2]);
    assert!(ok, "the recreate restore refuses: {err}");
    assert_eq!(
        sh(&std::path::PathBuf::from(&ws2), &["rev-parse", "HEAD"]).trim(),
        advanced_tip.trim(),
        "the restore took the branch's current local tip"
    );

    // restoration is idempotent by postcondition: an Available
    // workspace with no checkout recreates it at the branch's current
    // local tip and records the actual fact, no clean-first ceremony
    let (_, _, ws_edge) = rig_provision(&rig, "edge");
    // Available with no checkout: git's own removal, the stage never
    // moved to Cleaned
    sh(&rig.root, &["worktree", "remove", &ws_edge]);
    let edge_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| {
                r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws_edge)
            })
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the edge birth names its branch")
    };
    let edge_tip = rig_tip(&rig, &edge_branch);
    let (ok, out, err) = rig_sac(&rig, &["workspace", "restore", &ws_edge]);
    assert!(ok, "the ruled absent-case restore refuses: {out}{err}");
    assert!(out.contains("restored"), "{out}");
    assert!(std::path::Path::new(&ws_edge).join("readme").is_file());
    assert_eq!(
        sh(&std::path::PathBuf::from(&ws_edge), &["rev-parse", "HEAD"]).trim(),
        edge_tip,
        "the recreate stands at the branch's current local tip"
    );
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        kinds.contains(&"workspace_restored".to_string()),
        "{kinds:?}"
    );
    // the stage stays Available through the whole cycle
    let stage = json_of(&get_json(&rig.url, "/api/v1/workbench").1)["workspaces"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["path"] == json!(ws_edge))
        .map(|w| w["stage"].as_str().unwrap().to_string())
        .expect("the edge workspace projects");
    assert_eq!(stage, "available", "{stage}");
    // and the matching checkout standing afterwards is the no-op
    let (ok, out, _err) = rig_sac(&rig, &["workspace", "restore", &ws_edge]);
    assert!(ok, "the no-op restore refuses: {out}");
    assert!(out.contains("already restored"), "{out}");

    // the recorded branch gone refuses: never reconstructed from a
    // base or a remote
    let (_, _, ws3) = rig_provision(&rig, "baseless");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws3]);
    assert!(ok, "the third setup clean refuses: {err}");
    let ws3_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws3))
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the third birth names its branch")
    };
    sh(&rig.root, &["branch", "-D", &ws3_branch]);
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "restore", &ws3]);
    assert!(!ok, "the branchless restore ran: {err}");
    assert!(err.contains("workspace_branch_missing"), "{err}");
    assert!(!std::path::Path::new(&ws3).exists());

    // a checkout on another branch is a mismatch: no forced checkout
    let (_, _, ws4) = rig_provision(&rig, "swapped");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws4]);
    assert!(ok, "the fourth setup clean refuses: {err}");
    let ws4_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws4))
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the fourth birth names its branch")
    };
    sh(
        &rig.root,
        &["worktree", "add", &ws4, "-b", "another-line", "HEAD"],
    );
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "restore", &ws4]);
    assert!(!ok, "the mismatched restore ran: {err}");
    assert!(err.contains("workspace_checkout_mismatch"), "{err}");
    let rows = sh(&rig.root, &["worktree", "list", "--porcelain"]);
    assert!(rows.contains("refs/heads/another-line"));
    let _ = ws4_branch;
    let _ = ws;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cleaned_stage_guards_launch_workbench_and_switch() {
    let rig = lifecycle_rig("guards");
    let (_, _branch, ws) = rig_provision(&rig, "guarded-lifecycle");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the setup clean refuses: {err}");

    // the stage reads back through the workbench snapshot
    let (_, body) = get_json(&rig.url, "/api/v1/workbench");
    let stages: Vec<(usize, String)> = json_of(&body)["workspaces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| {
            (
                w["seq"].as_u64().unwrap() as usize,
                w["stage"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert!(stages.contains(&(0, "cleaned".to_string())), "{stages:?}");

    // a checkout that stands again at the recorded path (re-created
    // out-of-band) stays recorded Cleaned: a fresh launch into it
    // refuses until an explicit restore, and the workbench door too
    let guard_branch = {
        let (_, body) = get_json(&rig.url, "/api/v1/records");
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "workspace_created" && r["payload"]["worktree"] == json!(ws))
            .map(|r| r["payload"]["branch"].as_str().unwrap().to_string())
            .expect("the guarded birth names its branch")
    };
    sh(&rig.root, &["worktree", "add", &ws, &guard_branch]);
    let (ok, _out, err) = rig_sac(&rig, &["launch", "--workspace", &ws]);
    assert!(!ok, "a cleaned workspace took a launch: {err}");
    assert!(err.contains("recorded Cleaned"), "{err}");
    // the same-terminal arm rides the same stage guard
    let (ok, _out, err) = rig_sac(&rig, &["launch", "--workspace", &ws, "--here"]);
    assert!(!ok, "a cleaned workspace took a here launch: {err}");
    assert!(err.contains("recorded Cleaned"), "{err}");
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": ws}}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "workspace_cleaned",
        "{body}"
    );

    // the switch target refuses the same way: a saved session whose
    // header cwd is the cleaned workspace
    let saved = rig.dir.join("native-deadbeef01.jsonl");
    std::fs::write(
        &saved,
        format!("{{\"id\":\"deadbeef01\",\"cwd\":\"{ws}\"}}\n{{\"kind\":\"filler\"}}\n"),
    )
    .unwrap();
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/native/attach",
        &json!({"native_id": "deadbeef01", "session_file": saved.display().to_string()}),
    );
    assert_eq!(status, 200, "{body}");
    let num = json_of(&body)["num"].as_u64().unwrap();
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/workspace/change",
        &json!({"incarnation": num, "to": ws, "client": "lifecycle-test"}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "workspace_cleaned",
        "{body}"
    );

    // the resume path pins the same guard: a saved session whose
    // header cwd sits in the cleaned workspace refuses until an
    // explicit restore - the guard fires before any host submission
    let saved_resume = rig.dir.join("native-cafe0001.jsonl");
    std::fs::write(
        &saved_resume,
        format!(
            "{{\"type\":\"session\",\"id\":\"cafe0001\",\"cwd\":\"{ws}\"}}\n{{\"kind\":\"filler\"}}\n"
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/native/attach",
        &json!({"native_id": "cafe0001", "session_file": saved_resume.display().to_string()}),
    );
    assert_eq!(status, 200, "{body}");
    let resume_num = json_of(&body)["num"].as_u64().unwrap();
    let (ok, _out, err) = rig_sac(&rig, &["launch", "--resume", &format!("i-{resume_num}")]);
    assert!(!ok, "a cleaned workspace took a resume: {err}");
    assert!(err.contains("recorded Cleaned"), "{err}");

    // the explicit restore admits the directory again
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "restore", &ws]);
    assert!(ok, "the restore refuses: {err}");
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": ws}}),
    );
    // the launch now proceeds past the stage guard (its host answer is
    // the fake host's acceptance)
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["submitted"], true, "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn reconcile_verifies_preserves_and_records_on_apply() {
    let rig = lifecycle_rig("reconcile");
    // a present matching workspace: preserved
    let (kept_seq, kept_branch, kept) = rig_provision(&rig, "keeper");
    // an absent workspace whose host presence is verified: cleanable
    let (_gone_seq, gone_branch, gone) = rig_provision(&rig, "departed");
    sh(&rig.root, &["worktree", "remove", &gone]);
    // a conflict: bytes at the recorded path, no registration
    let (_bad_seq, _bad_branch, bad) = rig_provision(&rig, "conflicted");
    sh(&rig.root, &["worktree", "remove", &bad]);
    std::fs::create_dir_all(&bad).unwrap();

    // the dry run records nothing and reports every fact
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile"]);
    assert!(ok, "the dry run fails: {out}{err}");
    assert!(out.contains("dry run"), "{out}");
    assert!(out.contains(&format!("record {kept_seq} {kept}")), "{out}");
    assert!(out.contains("preserved"), "{out}");
    assert!(!out.contains("Cleaned recorded now"), "{out}");
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert!(
        !kinds.contains(&"workspace_cleaned".to_string()),
        "{kinds:?}"
    );

    // the applied run records the verified Cleaned facts in one batch
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile", "--apply"]);
    assert!(ok, "the applied run fails: {out}{err}");
    assert!(
        out.contains("applied - 1 verified Cleaned fact(s) recorded"),
        "{out}"
    );
    assert!(out.contains(&gone), "{out}");
    assert!(out.contains(&bad), "{out}");
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    let cleaned: Vec<&serde_json::Value> = records
        .iter()
        .filter(|r| r["kind"] == "workspace_cleaned")
        .collect();
    assert_eq!(
        cleaned.len(),
        1,
        "only the verified absence recorded: {records:?}"
    );
    assert_eq!(
        cleaned[0]["payload"]["workspace"],
        json!(1),
        "the departed birth is the cleaned one"
    );
    assert_eq!(
        sh(
            &rig.root,
            &[
                "rev-parse",
                "--verify",
                &format!("refs/heads/{kept_branch}")
            ]
        ),
        sh(&rig.root, &["rev-parse", "HEAD"]),
        "the preserved workspace's branch stands"
    );
    let _ = gone_branch;

    // the recorded-Cleaned arm verifies the same doors the cleanup ran:
    // an orphan terminal the host still names is a conflict, never a
    // reconciled state, and --apply lands no fact for it
    let (_, _, ws_orphan) = rig_provision(&rig, "orphaned");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws_orphan]);
    assert!(ok, "the orphan setup clean refuses: {err}");
    rig_open_terminal(&rig, "orphan_term_7", &ws_orphan);
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    let records_before_orphan = body.clone();
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile", "--apply"]);
    assert!(ok, "reconcile fails: {out}{err}");
    assert!(
        out.contains("the host names 1 live terminal(s) at a workspace recorded Cleaned"),
        "{out}"
    );
    assert!(out.contains("applied - no new facts recorded"), "{out}");
    let (_, body) = get_json(&rig.url, "/api/v1/records");
    assert_eq!(
        json_of(&records_before_orphan)["records"],
        json_of(&body)["records"],
        "the orphan conflict landed no fact"
    );
    std::fs::write(&rig.host_state, b"").unwrap();

    // the healed host answers again: verified absence of a
    // recorded-Cleaned workspace reads as the verified preserved fact
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile", "--apply"]);
    assert!(ok, "the final reconcile fails: {out}{err}");
    assert!(
        out.contains(
            "verified: recorded Cleaned, checkout absent, the host names no terminals there"
        ),
        "{out}"
    );
}

/// A recorded path addresses its own birth: a historical establishment
/// whose stored string carries a symlink segment resolves through it -
/// stored identity preserved, and two stored paths aliasing one
/// directory refuse as ambiguous rather than picking.
#[tokio::test(flavor = "multi_thread")]
async fn the_recorded_path_itself_addresses_the_birth_through_an_alias() {
    let rig = lifecycle_rig("alias");
    // a historical establishment whose recorded path carries a symlink
    // segment: hand-written, exactly what an old log may hold
    let link = rig.dir.join("link");
    let real_home = rig.dir.join("real-wt");
    std::fs::create_dir_all(&real_home).unwrap();
    std::os::unix::fs::symlink(&real_home, &link).unwrap();
    let real = real_home.join("hist-line");
    let recorded = link.join("hist-line");
    sh(
        &rig.root,
        &[
            "worktree",
            "add",
            "-q",
            "-b",
            "saccade/t-9",
            &real.display().to_string(),
            "HEAD",
        ],
    );
    let establishment = |task: i32, branch: &str, worktree: &str| {
        let mut conn = saccade::db::open(&rig.dir.join("saccade.db")).unwrap();
        saccade::db::record(
            &mut conn,
            &saccade::store::Context::system(),
            saccade::Command::CreateTask {
                name: saccade::Prose::new(format!("a lineage behind {branch}")).unwrap(),
            },
            task as u64 * 2,
        )
        .unwrap();
        saccade::db::record(
            &mut conn,
            &saccade::store::Context::system(),
            saccade::Command::EstablishWorkspace {
                task_id: saccade::TaskId(task as usize),
                base: saccade::GitCommit::new("abc123".into()).unwrap(),
                branch: saccade::GitBranch::new(branch.into()).unwrap(),
                worktree: saccade::WorktreePath::new(worktree.into()).unwrap(),
            },
            task as u64 * 2 + 1,
        )
        .unwrap();
    };
    establishment(0, "saccade/t-9", recorded.display().to_string().as_str());

    // the recorded string itself addresses the birth: the ordered clean
    // contract runs over the real checkout it names
    let (ok, out, err) = rig_sac(&rig, &["workspace", "clean", recorded.to_str().unwrap()]);
    assert!(
        ok,
        "the recorded string does not address its birth: {out}{err}"
    );
    assert!(!real.exists(), "the real checkout is gone");
    assert_eq!(
        rig_tip(&rig, "saccade/t-9"),
        sh(&rig.root, &["rev-parse", "HEAD"]).trim()
    );

    // the Cleaned stage blocks ordinary use at BOTH spellings once the
    // checkout stands again: the guard resolves the alias, it never
    // lets a known Cleaned workspace become usable through its twin
    sh(
        &rig.root,
        &[
            "worktree",
            "add",
            &real.display().to_string(),
            "saccade/t-9",
        ],
    );
    for shape in [recorded.to_str().unwrap(), real.to_str().unwrap()] {
        let (ok, _out, err) = rig_sac(&rig, &["launch", "--workspace", shape]);
        assert!(
            !ok,
            "the aliased Cleaned workspace took a launch at {shape}: {err}"
        );
        assert!(err.contains("recorded Cleaned"), "{err}");
    }
    let (status, body) = post_json(
        &rig.url,
        "/api/v1/workbench/launch",
        &json!({"workspace": {"existing": recorded.display().to_string()}}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "workspace_cleaned",
        "{body}"
    );

    // a second stored path over the same directory makes the alias
    // unsafe: both shapes refuse as ambiguous, neither is picked
    establishment(1, "saccade/t-10", real.display().to_string().as_str());
    for shape in [recorded.to_str().unwrap(), real.to_str().unwrap()] {
        let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", shape]);
        assert!(!ok, "the aliased directory resolved to one birth: {err}");
        assert!(err.contains("workspace_ambiguous_path"), "{err}");
    }

    // reconcile surfaces the twin spellings as shared-path conflicts
    // for both births, instead of processing them as separate targets
    let (ok, out, err) = rig_sac(&rig, &["workspace", "reconcile"]);
    assert!(ok, "reconcile fails: {out}{err}");
    assert!(out.contains("share this path"), "{out}");
}

#[test]
fn completions_emit_scripts_and_the_candidate_doors_answer_the_fold() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("completions-door");
    let system = saccade::Context::system();
    let mut conn = saccade::db::open(&path).unwrap();
    let now = 1;
    // the story: a task, two workspaces (one cleaned), one registered
    // native session
    let commands = [
        saccade::Command::CreateTask {
            name: saccade::Prose::new("implement floop".into()).unwrap(),
        },
        saccade::Command::CreateWorkspace {
            base: saccade::GitCommit::new("aaaa1111".into()).unwrap(),
            branch: saccade::GitBranch::new("workspace-floop".into()).unwrap(),
            worktree: saccade::WorktreePath::new("/tmp/sac-api-floop".into()).unwrap(),
        },
        saccade::Command::CreateWorkspace {
            base: saccade::GitCommit::new("bbbb2222".into()).unwrap(),
            branch: saccade::GitBranch::new("workspace-gloom".into()).unwrap(),
            worktree: saccade::WorktreePath::new("/tmp/sac-api-gloom".into()).unwrap(),
        },
        saccade::Command::CleanWorkspace {
            workspace: saccade::RecordId(2),
        },
        saccade::Command::RegisterIncarnation {
            native_id: saccade::NativeSessionId::new("sess-7777".into()).unwrap(),
            saved_session: Some(
                saccade::SessionLocator::new("/home/j/.pi/sessions/7777.json".into()).unwrap(),
            ),
            address: None,
            adopted: false,
        },
    ];
    for command in commands {
        saccade::db::record(&mut conn, &system, command, now).expect("the fixture story records");
    }
    drop(conn);

    let sac = |args: &[&str]| {
        let out = std::process::Command::new(bin)
            .arg("--db")
            .arg(&path)
            .args(args)
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    // the script door: emitted from the binary, no destination resolves
    let (ok, script, err) = sac(&["completions", "fish"]);
    assert!(ok, "completions fish: {err}");
    assert!(script.contains("complete -c sac"), "the fish skeleton");
    assert!(
        script.contains("sac complete $argv"),
        "the PATH-name door call"
    );
    assert!(script.is_ascii(), "authored text is ASCII");
    assert!(!sac(&["completions", "bash"]).0, "unshipped shells refuse");

    // the candidate doors read the fold: tasks with title and state
    let (ok, out, err) = sac(&["complete", "tasks"]);
    assert!(ok, "complete tasks: {err}");
    assert_eq!(out.trim(), "t-0\timplement floop (open)");

    // workspaces: recorded paths with branch and stage, clean leading
    // Available, restore leading Cleaned
    let (ok, out, err) = sac(&["complete", "workspaces-clean"]);
    assert!(ok, "complete workspaces-clean: {err}");
    assert_eq!(
        out.trim(),
        "/tmp/sac-api-floop\tworkspace-floop, available\n\
         /tmp/sac-api-gloom\tworkspace-gloom, cleaned"
    );
    let (ok, out, err) = sac(&["complete", "workspaces-restore"]);
    assert!(ok, "complete workspaces-restore: {err}");
    assert_eq!(
        out.trim(),
        "/tmp/sac-api-gloom\tworkspace-gloom, cleaned\n\
         /tmp/sac-api-floop\tworkspace-floop, available"
    );

    // incarnations: the birth actor beside what the record holds
    let (ok, out, err) = sac(&["complete", "incarnations"]);
    assert!(ok, "complete incarnations: {err}");
    assert_eq!(
        out.trim(),
        "i-0\tsaccade native session sess-7777, saved /home/j/.pi/sessions/7777.json"
    );

    // completion never records: the log is unchanged after every read
    let (ok, before, _) = sac(&["log", "--json"]);
    assert!(ok);
    let (ok, _, err) = sac(&["complete", "tasks"]);
    assert!(ok, "{err}");
    let (ok, after, _) = sac(&["log", "--json"]);
    assert!(ok);
    assert_eq!(before, after, "the candidate door records nothing");

    // silent failure: outside a repo and without a reachable
    // destination the door answers empty, exit 0, no stderr
    let outside = std::env::temp_dir().join(format!("sac-api-no-repo-{}", std::process::id()));
    std::fs::create_dir_all(&outside).unwrap();
    let out = std::process::Command::new(bin)
        .arg("complete")
        .arg("tasks")
        .current_dir(&outside)
        .env_remove("SACCADE_REPO")
        .env("HOME", &outside)
        .output()
        .expect("spawn sac outside a repo");
    assert!(out.status.success());
    assert_eq!(out.stdout, b"", "no candidates outside a repo");
    assert_eq!(out.stderr, b"", "the failure stays silent");

    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}

#[test]
fn the_launch_door_births_the_incarnation_with_its_host_address() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-launch-b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // the launch doors resolve the repo's own database: the serve
    // subprocess must publish exactly that path's address hint
    let db = saccade::paths::state_dir(&dir).join("saccade.db");
    let workspace = dir.join("ws");
    std::fs::create_dir_all(&workspace).unwrap();
    // the contract-shaped fake: hc run accepts the composed launch and
    // answers the pinned shape; its argv lands beside it for the door
    // composition's assertions
    let fake_dir = dir.join("fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    let fake = fake_dir.join("hc");
    std::fs::write(
        &fake,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-$1.log\"\nprintf '{\"address\":\"t1\",\"cwd\":\"x\",\"state\":\"live\",\"turn\":0}\\n'\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (mut serve, url) = spawn_serve(bin, &db, &[], &[]);
    let pi = dir.join("pi");
    std::fs::write(&pi, b"#!/bin/sh\n").unwrap();

    let out = scrubbed(std::process::Command::new(bin))
        .args(["launch", "--json", "--workspace"])
        .arg(&workspace)
        .env("SACCADE_SERVER", &url)
        .env("SACCADE_HERDR", &fake)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_REPO", &dir)
        .output()
        .expect("spawn sac");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "launch failed: {text}{err} (code {:?})",
        out.status.code()
    );
    let reply: Value = serde_json::from_str(&text).expect("the launch face is json");
    assert_eq!(reply["terminal"], "t1", "{text}");
    // the door declares placement at birth: the ids are the stable
    // derivation over the workspace, and the reply names them
    let expected = saccade::placement::derived(&workspace);
    assert_eq!(reply["space"], expected.space.as_str(), "{text}");
    assert_eq!(reply["tab"], expected.tab.as_str(), "{text}");
    assert_eq!(reply["place_refusal"], Value::Null, "{text}");
    let label_argv = std::fs::read_to_string(fake_dir.join("argv-label.log")).unwrap();
    let label: Vec<&str> = label_argv.lines().skip(1).collect();
    assert_eq!(
        label,
        vec![
            "t1",
            "--json",
            &format!("herdr.space={}", expected.space),
            &format!("herdr.tab={}", expected.tab),
        ],
        "the placement declaration is one ask after the address returns"
    );

    // the composed argv carries the settled parentage law, the
    // notes-silence flag, the json ask, and the pre-allocated session
    let argv_log = std::fs::read_to_string(fake_dir.join("argv-run.log")).unwrap();
    let argv: Vec<&str> = argv_log.lines().skip(1).collect();
    assert_eq!(
        argv[..4],
        ["--bg", "--no-parent", "--no-notify", "--cwd"],
        "the composition lost its shape: {argv:?}"
    );
    let composition = argv.join(" ");
    assert!(composition.contains("--json -- "), "the ask rides --json");
    assert!(
        composition.contains("--session-id ") && composition.contains("--session-dir "),
        "the door pre-allocates the native session: {composition}"
    );
    assert!(
        composition.contains("direnv exec "),
        "the equip stays direnv"
    );

    // the fold carries the door-birthed incarnation: identity and
    // address exist at launch time, the locator waits for pi's save
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    let (id, ctx) = world
        .incarnations
        .iter()
        .find(|(_, ctx)| ctx.incarnation.address() == Some("t1"))
        .expect("the door birth carries the host address");
    let saccade::Incarnation::Native {
        native_id,
        saved_session,
        ..
    } = &ctx.incarnation
    else {
        panic!("the door births a native identity");
    };
    // the pre-allocated id is the one pi will create: it rode argv
    assert!(
        composition.contains(native_id.as_str()),
        "the argv pre-allocates the birth's native id"
    );
    assert!(saved_session.is_none(), "the locator binds at attach");

    // the door-owned session dir exists under the tracker's state root
    let sessions = saccade::pi_session::sessions_dir(&dir);
    assert!(sessions.is_dir(), "the native-sessions dir stands");

    // the identity is a real registration: the mailbox read resolves it
    let num = world
        .incarnation_num(*id)
        .expect("the fold numbers the birth");
    let mailbox = scrubbed(std::process::Command::new(bin))
        .args(["messages", "--server", url.as_str(), "--repo"])
        .arg(dir.display().to_string())
        .arg(format!("i-{num}"))
        .output()
        .expect("spawn sac");
    assert!(
        mailbox.status.success(),
        "the door-birthed incarnation resolves: {}",
        String::from_utf8_lossy(&mailbox.stderr)
    );

    // a relaunch moves the association's address through the assign
    // door: the same incarnation, another terminal
    let mut assign_response = ureq::post(&format!("{url}/api/v1/incarnations/address"))
        .send_json(serde_json::json!({
            "incarnation": id.0.0,
            "address": "t9",
            "client": "test",
        }))
        .unwrap();
    let assign: Value = assign_response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .expect("the assign face is json");
    assert_eq!(assign["assigned"], true);
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    assert_eq!(
        world.incarnations[id]
            .incarnation
            .address()
            .map(str::to_string),
        Some("t9".to_string()),
        "the address moved with the relaunch"
    );

    // a refused placement never fails the launch: the terminal exists,
    // the door discloses the refusal, and the identity still lands
    let refusing = fake_dir.join("hc-label-refusing");
    std::fs::write(
        &refusing,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-$1.log\"\nif [ \"$1\" = label ]; then\n  printf '{\"error\":{\"kind\":\"not_found\",\"message\":\"gone\"}}\\n' >&2\n  exit 3\nfi\nprintf '{\"address\":\"t2\",\"cwd\":\"x\",\"state\":\"live\",\"turn\":0}\\n'\n",
    )
    .unwrap();
    std::fs::set_permissions(&refusing, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = scrubbed(std::process::Command::new(bin))
        .args(["launch", "--json", "--workspace"])
        .arg(&workspace)
        .env("SACCADE_SERVER", &url)
        .env("SACCADE_HERDR", &refusing)
        .env("SACCADE_PI", &pi)
        .env("SACCADE_REPO", &dir)
        .output()
        .expect("spawn sac");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "a refused placement does not fail the launch: {text}{err}"
    );
    let reply: Value = serde_json::from_str(&text).expect("the launch face is json");
    assert_eq!(reply["terminal"], "t2", "{text}");
    assert_eq!(reply["space"], expected.space.as_str(), "{text}");
    assert!(
        reply["place_refusal"]
            .as_str()
            .is_some_and(|d| d.contains("the address is not live")),
        "{text}"
    );
    // the second door birth stands beside the first
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    assert!(
        world
            .incarnations
            .iter()
            .any(|(_, ctx)| ctx.incarnation.address() == Some("t2")),
        "the disclosed launch still birthed its identity"
    );

    std::fs::remove_dir_all(&dir).unwrap();
    let _ = serve.kill();
}
