//! The /api/v1 surface over a real loopback socket: one envelope, records
//! back or a typed error, the tier law holding through HTTP exactly as it
//! holds in process.

use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use saccade::api::AppState;

/// The suite's own state parent and working directory, under the temp
/// tree: every spawned sac resolves state and repo discovery here, so
/// a test run writes nothing into the host's real state root - the
/// live tracker's state included.
fn suite_state() -> PathBuf {
    std::env::temp_dir().join(format!("sac-suite-state-{}", std::process::id()))
}

fn suite_cwd() -> PathBuf {
    let dir = suite_state().join("cwd");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The host's real state parent, as the unhelped environment resolves
/// it: read-only evidence for the helper's pin.
fn real_state_parent() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from) {
        return dir;
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".local")
        .join("state")
}

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
    let router = saccade::serve::router(state);
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
    // raw bytes, but named JSON: the content-type policy is pinned in
    // the wire-contracts test, not here
    let mut r = ureq::post(&format!("{base_url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .content_type("application/json")
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
        ["task_created"]
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

    // human-only acts refuse at agent tier
    for command in [
        json!({"cancel_task": {"id": 0, "explanation": "stop"}}),
        json!({"reopen_task": {"id": 0, "explanation": "resume"}}),
    ] {
        let (status, body) = post_command(&base_url, &envelope("pi", "agent", command));
        assert_eq!(status, 400);
        assert_eq!(json_of(&body)["error"]["code"], "human_only");
    }

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
            json!({"create_workspace": {"base": "abc123", "branch": "feature-abc123ab", "worktree": "/repo/wt/feature-abc123ab"}}),
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
    assert_eq!(status, 404, "{body}");

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

    // attach first, then register resolves the attach-born birth
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "cccc4444",
            "session_file": "/sessions/cccc4444.jsonl",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], true);
    let attach_birth = reply["incarnation"].as_u64().unwrap();
    assert_eq!(reply["records"][0]["kind"], "incarnation_registered");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "cccc4444", "address": "t3", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], false);
    assert_eq!(reply["incarnation"], attach_birth);
    let records = reply["records"].as_array().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["kind"], "host_address_assigned");
    assert_eq!(records[0]["payload"]["incarnation"], attach_birth);
    assert_eq!(records[0]["payload"]["address"], "t3");

    // a re-register of the addressed+bound session keeps the prior refusal
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "cccc4444", "address": "t6", "adopted": false}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "saved_session_mismatch",
        "{body}"
    );
    // the refusal moved no address: the recovery owns only the unaddressed case
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    let resolved = world
        .incarnations
        .values()
        .find(|ctx| ctx.incarnation.native_id().map(|n| n.as_str()) == Some("cccc4444"))
        .unwrap();
    assert_eq!(resolved.incarnation.address(), Some("t3"));

    // a second door birth for a known unbound id is the fold's duplicate
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "dddd5555", "address": "t4", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "dddd5555", "address": "t5", "adopted": false}),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        "duplicate_native_session",
        "{body}"
    );

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

    // a pre-title note kind (the 0.15 wire shape) no longer parses: the
    // wire refuses the birth rather than silently dropping the field an
    // old server would have ignored
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"comment": {"home": 0, "kind": {"note": {"body": "suite green"}}, "references": []}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(
        json_of(&body)["error"]["code"],
        "malformed_request",
        "{body}"
    );
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
    // a claim naming nothing names what the tracker holds
    let fail = client::send(
        &base,
        &agent,
        Command::ClaimTask {
            id: saccade::TaskId(9),
            incarnation: None,
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
    let router = saccade::serve::router(state);
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

fn sh(dir: &std::path::Path, args: &[&str]) -> String {
    let out = saccade::boards::git_command()
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

use saccade::store::{Context, RecordId, Tier};
use saccade::types::actor::ActorName;
use saccade::{Command, Prose, TaskId};

/// The records face keeps degraded rows and refuses corruption.
#[tokio::test(flavor = "multi_thread")]
async fn the_records_face_keeps_degraded_rows_and_refuses_corruption() {
    // a corrupt payload of a known kind: the log will not even load
    let db = scratch_db("records-corrupt");
    let base_url = spawn_server(&db).await;
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "before the damage"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (1, 0, 0, 'pi', 'agent', 'task_created', '{}', NULL)",
            [],
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base_url}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 500);
    let body = r.body_mut().read_to_string().unwrap();
    assert_eq!(json_of(&body)["error"]["code"], "database_error");
    assert!(
        json_of(&body)["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("corrupt record at seq 1")),
        "{body}"
    );
    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();

    // an unknown kind: the world will not fold, the rows still travel
    let db = scratch_db("records-degraded");
    let base_url = spawn_server(&db).await;
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "before the damage"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "INSERT INTO events (seq, event_time, logged_time, actor, tier, kind, payload, scope)
             VALUES (1, 0, 0, 'pi', 'agent', 'time_travelled', '{}', NULL)",
            [],
        )
        .unwrap();
    let mut r = ureq::get(&format!("{base_url}/api/v1/records"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let body = r.body_mut().read_to_string().unwrap();
    assert_eq!(json_of(&body)["records"].as_array().unwrap().len(), 2);
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "blocked while degraded"}}),
        ),
    );
    assert_eq!(status, 503);
    assert_eq!(json_of(&body)["error"]["code"], "degraded", "{body}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

/// A supported local write shows on the server's records face: the
/// next server write folds the same file, never a stale world.
#[tokio::test(flavor = "multi_thread")]
async fn a_local_write_shows_on_the_server_face() {
    let db = scratch_db("local-visible");
    let base_url = spawn_server(&db).await;
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "server-born task"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");

    let local = Context {
        actor: ActorName::new("pi".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };
    let mut conn = saccade::db::open(&db).unwrap();
    saccade::db::record(
        &mut conn,
        &local,
        Command::ClaimTask {
            id: TaskId(0),
            incarnation: None,
        },
        saccade::db::now_epoch(),
    )
    .unwrap();

    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(json_of(&body)["records"].as_array().unwrap().len(), 2);

    // the next server write folds the same file, never a stale world
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "jerry",
            "human",
            json!({"create_task": {"name": "after the local write"}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let records = json_of(&body);
    let kinds: Vec<String> = records["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(kinds, vec!["task_created", "task_claimed", "task_created"]);

    // a refusal names the log's fresh cursor, external writes included
    let (status, _) = post_command(
        &base_url,
        &envelope(
            "jerry",
            "human",
            json!({"claim_task": {"id": 99, "incarnation": null}}),
        ),
    );
    assert_eq!(status, 400);
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
fn board_doors_run_the_full_loop_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("board-loop");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    // a real repo: the doors' telemetry resolves its state dir the way
    // production resolves the tracker it talks to
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
    let (mut serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );
    let sac = |args: &[&str]| {
        let mut cmd = scrubbed(std::process::Command::new(bin));
        cmd.arg("--repo")
            .arg(&project)
            .arg("--server")
            .arg(&url)
            .env("SACCADE_ACTOR", "pi")
            .args(args);
        let out = cmd.output().expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let sac_json = |args: &[&str]| {
        let mut full = vec!["--json"];
        full.extend_from_slice(args);
        let (ok, out, err) = sac(&full);
        assert!(ok, "{}: {err}", args.join(" "));
        let value: Value = serde_json::from_str(&out).expect("the json face parses");
        value
    };

    // two tasks: the source account and its consumer
    let source = sac_json(&["create", "task", "prove the loop"])["id"]
        .as_str()
        .unwrap()
        .to_string();
    let consumer = sac_json(&["create", "task", "use the proof"])["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(source, "t-0");
    assert_eq!(consumer, "t-1");

    // the source's first board: edit writes the fresh template, the
    // author replaces it, check passes, commit publishes r1
    let src = project.join("src-dir");
    let (ok, out, _) = sac(&["board", "edit", &source, src.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(src.join("board.yaml").exists(), "the template landed");
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input\n",
    )
    .unwrap();
    let (ok, out, _) = sac(&["board", "check", src.to_str().unwrap()]);
    assert!(ok, "{out}");
    let reply = sac_json(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "first account",
    ]);
    assert_eq!(reply["revision"], json!(1));
    // a board matching its parent refuses, naming the parent revision, and records nothing
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "the edit never reached the board",
    ]);
    assert!(!ok, "the unchanged board refuses");
    assert!(
        err.contains("boards_unchanged") && err.contains("t-0@r1"),
        "{err}"
    );

    // the consumer incorporates the source's r1 by number; an unknown
    // number refuses with the teaching line
    let con = project.join("con-dir");
    let (ok, out, _) = sac(&["board", "edit", &consumer, con.to_str().unwrap()]);
    assert!(ok, "{out}");
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 9\n        select: [\"/content/0\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        &consumer,
        con.to_str().unwrap(),
        "-m",
        "cites a revision the source never published",
    ]);
    assert!(!ok, "the unknown revision refuses");
    assert!(
        err.contains("boards_unknown_source_revision") && err.contains("r9"),
        "{err}"
    );
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 1\n        select: [\"/content/0\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let reply = sac_json(&[
        "board",
        "commit",
        &consumer,
        con.to_str().unwrap(),
        "-m",
        "consumer cites the proof",
    ]);
    assert_eq!(reply["revision"], json!(1));

    // the acceptance refuses while the source stands unaccepted
    let (ok, _, _) = sac(&["offer", &consumer]);
    assert!(ok);
    let (ok, _, err) = sac(&["accept", &consumer]);
    assert!(!ok, "the unaccepted source must refuse the acceptance");
    assert!(err.contains("source_not_accepted"), "{err}");
    let (ok, out, _) = sac(&[
        "reject",
        &consumer,
        "--grounds",
        "wait for the source to conclude",
    ]);
    assert!(ok, "{out}");

    // the source's review loop: offer, reject, annotate, recommit.
    let (ok, out, _) = sac(&["offer", &source]);
    assert!(ok, "{out}");
    // committing a changed board refuses while offered
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input, early\n",
    )
    .unwrap();
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "too early",
    ]);
    assert!(!ok, "the offered stage refuses a commit");
    assert!(err.contains("invalid_state_transition"), "{err}");
    let (ok, out, _) = sac(&[
        "reject",
        &source,
        "--grounds",
        "the boundary case is missing",
    ]);
    assert!(ok, "{out}");
    assert!(
        out.contains("task open; annotate the board at t-0@r1"),
        "the reject teaches the annotation step: {out}"
    );
    std::fs::write(
        src.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent:\n  - text: the loop holds for every input\n  - annotation: the boundary case is covered in the appendix\n",
    )
    .unwrap();
    let reply = sac_json(&[
        "board",
        "commit",
        &source,
        src.to_str().unwrap(),
        "-m",
        "annotate the boundary",
    ]);
    assert_eq!(reply["revision"], json!(2));
    let (ok, _, _) = sac(&["offer", &source]);
    assert!(ok);
    let (ok, _, _) = sac(&["accept", &source]);
    assert!(ok, "the source accepts once its incorporations hold");

    // the consumer reconciles onto the accepted revision and concludes
    std::fs::write(
        con.join("board.yaml"),
        "grammar: 1\ntask: t-1\ncontent:\n  - incorporation:\n      source:\n        task: t-0\n        revision: 2\n        select: [\"/content/0\"]\n      content:\n        - text: relying on the proof\n",
    )
    .unwrap();
    let reply = sac_json(&[
        "board",
        "commit",
        &consumer,
        con.to_str().unwrap(),
        "-m",
        "rebind to the accepted revision",
    ]);
    assert_eq!(reply["revision"], json!(2));
    let (ok, _, _) = sac(&["offer", &consumer]);
    assert!(ok);

    let (ok, out, _) = sac(&["accept", &consumer]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1 concluded"), "{out}");

    // the task history interleaves commits with the decisions
    let history = sac_json(&["board", "log", &source]);
    let entries = history["history"].as_array().unwrap();
    let names: Vec<&str> = entries
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"task_created")
            && names.contains(&"task_committed")
            && names.contains(&"task_offered")
            && names.contains(&"task_rejected")
            && names.contains(&"task_accepted"),
        "{names:?}"
    );
    let committed = entries
        .iter()
        .find(|e| e["event"] == "task_committed")
        .unwrap();
    assert_eq!(committed["message"], json!("first account"));
    assert_eq!(committed["revision"], json!(1));
    let rejected = entries
        .iter()
        .find(|e| e["event"] == "task_rejected")
        .unwrap();
    assert_eq!(rejected["grounds"], json!("the boundary case is missing"));

    // the diff names the annotation; get reads any revision; the
    // component form prints one component; the file read rides the
    // flag; the materialized path exists on the server's disk
    let (ok, diff, _) = sac(&["board", "diff", "t-0@r1..r2"]);
    assert!(ok, "{diff}");
    assert!(diff.contains("annotation"), "{diff}");
    // the component summary leads (t-50): the added annotation, by
    // place, before the line diff
    assert!(
        diff.starts_with("components r1..r2\n  added    annotation  /content/1\n"),
        "{diff}"
    );
    let (ok, json_diff, _) = sac(&["--json", "board", "diff", "t-0@r1..r2"]);
    assert!(ok, "{json_diff}");
    let json_diff: serde_json::Value = serde_json::from_str(&json_diff).unwrap();
    assert_eq!(
        json_diff["components"],
        json!([{"change": "added", "label": "annotation", "to": "/content/1"}])
    );
    assert!(json_diff["diff"].as_str().unwrap().contains("annotation"));
    let (ok, component, _) = sac(&["board", "get", "t-0@r2:/content/0"]);
    assert!(ok, "{component}");
    assert!(
        component.contains("the loop holds") && !component.contains("annotation"),
        "{component}"
    );
    let (ok, note, _) = sac(&["board", "get", "t-0@r2:/content/1"]);
    assert!(ok, "{note}");
    assert!(note.contains("annotation"), "{note}");
    let (ok, _, err) = sac(&["board", "get", "t-0@r2:/content/0", "--file", "board.yaml"]);
    assert!(!ok, "the component and file forms refuse each other: {err}");
    assert!(err.contains("not both"), "{err}");
    let (ok, old, _) = sac(&["board", "get", "t-0@r1", "--file", "board.yaml"]);
    assert!(ok, "{old}");
    assert!(
        old.contains("the loop holds") && !old.contains("annotation"),
        "{old}"
    );
    let tree_out = project.join("exported");
    let (ok, out, _) = sac(&["board", "get", "t-0", "--out", tree_out.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(tree_out.join("board.yaml").exists(), "{out}");
    // the boards/get wire the viewer reads: the file list nests under
    // files.files, each entry a path and base64 content
    let tree: Value = ureq::get(&format!("{url}/api/v1/boards/get?task=0"))
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    let listed = tree["files"]["files"]
        .as_array()
        .expect("the viewer reads files.files as a list");
    assert!(
        listed
            .iter()
            .any(|f| f["path"] == json!("board.yaml") && f["content"].is_string()),
        "{tree}"
    );
    assert_eq!(tree["revision"], json!(2));
    let (ok, path, _) = sac(&["board", "path", &source]);
    assert!(ok, "{path}");
    assert!(Path::new(path.trim()).join("board.yaml").exists(), "{path}");

    // search: current boards one line per task, history changes,
    // references are incorporations and messages
    let (ok, out, _) = sac(&["search", "loop"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-0@r2"), "{out}");
    assert!(
        !out.contains("t-1"),
        "the consumer board names no loop: {out}"
    );
    let (ok, out, _) = sac(&["search", "relying"]);
    assert!(ok, "{out}");
    assert!(out.contains("t-1@r2"), "{out}");
    let (ok, out, _) = sac(&["search", "--history", "boundary"]);
    assert!(ok, "{out}");
    assert!(out.contains("appeared"), "{out}");
    assert!(out.contains("annotate the boundary"), "{out}");
    let (ok, out, _) = sac(&["search", "t-0"]);
    assert!(ok, "{out}");
    assert!(out.contains("incorporates t-0@r"), "{out}");

    // telemetry rides in the repo's state dir: verb, flags, outcome, tier
    let telemetry = std::fs::read_to_string(saccade::telemetry::at(&saccade::paths::db_at_in(
        &suite_state(),
        &project,
    )))
    .unwrap();
    assert!(
        telemetry.contains("\"verb\":\"board.commit\""),
        "{telemetry}"
    );
    assert!(telemetry.contains("\"tier\":\"agent\""), "{telemetry}");
    assert!(
        telemetry.contains("\"outcome\":\"source_not_accepted\""),
        "{telemetry}"
    );

    serve.kill().unwrap();
    let _ = serve.wait();
}

#[test]
fn board_size_refusals_name_the_total_and_the_request_limit() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("board-size");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    saccade::boards::git_command()
        .args(["init", "--quiet", project.to_str().unwrap()])
        .output()
        .expect("git init");
    let (mut serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );
    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
            .arg("--repo")
            .arg(&project)
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
    let figure = |seed: u8, len: usize| -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    };
    // a working directory naming each figure; the answer is the files' total
    let stage = |dir: &Path, figures: &[(&str, Vec<u8>)]| -> usize {
        let mut board = String::from("grammar: 1\ntask: t-0\ncontent:\n  - text: the survey\n");
        let mut total = 0;
        for (path, bytes) in figures {
            std::fs::create_dir_all(dir.join(path).parent().unwrap()).unwrap();
            std::fs::write(dir.join(path), bytes).unwrap();
            total += bytes.len();
            board.push_str(&format!(
                "  - figure:\n      input: {path}\n      caption: {path}\n"
            ));
        }
        std::fs::write(dir.join("board.yaml"), &board).unwrap();
        total + board.len()
    };
    let (ok, _, err) = sac(&["create", "task", "show the survey"]);
    assert!(ok, "{err}");
    let dir = project.join("survey");
    let (ok, _, err) = sac(&["board", "edit", "t-0", dir.to_str().unwrap()]);
    assert!(ok, "{err}");

    // the reported board (a 1,987,615-byte file) beside one at the per-file bound publishes
    let figures = [
        ("figures/ux-study.png", figure(1, 1_987_615)),
        ("figures/survey-map.png", figure(2, 8 * 1024 * 1024)),
    ];
    stage(&dir, &figures);
    let (ok, out, err) = sac(&[
        "board",
        "commit",
        "t-0",
        dir.to_str().unwrap(),
        "-m",
        "the survey",
    ]);
    assert!(ok, "the board within one request publishes: {err}");
    assert!(out.contains("published t-0@r1"), "{out}");
    // and reads back whole
    let back = project.join("survey-back");
    let (ok, _, err) = sac(&["board", "edit", "t-0", back.to_str().unwrap()]);
    assert!(ok, "{err}");
    for (path, bytes) in &figures {
        assert!(
            std::fs::read(back.join(path)).unwrap() == *bytes,
            "{path} round-trips"
        );
    }

    // a board past one request: check warns with the total, commit refuses naming it and the limit
    let big = [
        ("figures/a.png", figure(3, 7_000_000)),
        ("figures/b.png", figure(4, 7_000_000)),
        ("figures/c.png", figure(5, 7_000_000)),
        ("figures/d.png", figure(6, 7_000_000)),
    ];
    let total = stage(&dir, &big);
    let (ok, out, err) = sac(&["board", "check", dir.to_str().unwrap()]);
    assert!(ok, "the warning does not fail the check: {err}");
    assert!(
        out.contains("warning") && out.contains(&total.to_string()),
        "{out}"
    );
    let (ok, _, err) = sac(&[
        "board",
        "commit",
        "t-0",
        dir.to_str().unwrap(),
        "-m",
        "too much",
    ]);
    assert!(!ok, "the oversized request refuses");
    assert!(
        err.contains(&total.to_string()) && err.contains("per request"),
        "{err}"
    );
    assert!(!err.contains("malformed_response"), "{err}");

    // a body just past the limit refuses in the error envelope
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .content_type("application/json")
        .send(" ".repeat(32 * 1024 * 1024 + 1).as_str())
        .expect("the server answers");
    let status = response.status().as_u16();
    let body = response.body_mut().read_to_string().unwrap();
    assert_eq!(status, 413, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        json!("request_too_large"),
        "{body}"
    );

    // a non-JSON refusal surfaces with its status and text
    let error =
        saccade::serve::client::board_get_component(&url, 0, None, "/content/0&revision=zz")
            .err()
            .expect("the bad query refuses");
    let shown = format!("{error:?}");
    assert!(!shown.contains("malformed_response"), "{shown}");
    assert!(
        shown.contains("400") && shown.contains("revision"),
        "{shown}"
    );

    serve.kill().unwrap();
    let _ = serve.wait();
}

#[test]
fn omitted_prose_refuses_without_a_terminal_and_names_the_move() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("prose-door");
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
    let (ok, _, _) = sac(&["create", "task", "named prose"]);
    assert!(ok);

    // no terminal, no prose: the door refuses and teaches both moves
    let (ok, _, err) = sac(&["create", "task"]);
    assert!(!ok);
    assert!(err.contains("task name is required"), "{err}");
    let (ok, _, err) = sac(&["cancel", "t-0"]);
    assert!(!ok);
    assert!(
        err.contains("cancellation explanation is required"),
        "{err}"
    );
    let (ok, _, err) = sac(&["reject", "t-0"]);
    assert!(!ok);
    assert!(err.contains("rejection grounds is required"), "{err}");

    // explicit prose passes at the doors agents use; cancel and reopen
    // stay the human's acts
}

#[test]
fn board_wire_routes_hold_their_contracts() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("board-wire");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let (mut serve, url) = spawn_serve(
        bin,
        scratch.db.as_path(),
        &["--repo", project.to_str().unwrap()],
        &[],
    );

    // the version route names the supported grammar range
    let body: Value = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap()
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["board_grammar"]["min"], json!(1));
    assert_eq!(body["board_grammar"]["max"], json!(1));
    // the tracker's name for tracker-named locations: the repo's
    // directory basename; the fixture repo directory is "project"
    assert_eq!(body["tracker"], json!("project"));

    // a doubled slash lands on no route: 404, never a collapsed-path
    // answer, so a client that joins one slash too many sees a refusal
    for path in ["//api/v1/version", "//api/v1/boards/get?task=0"] {
        let mut response = ureq::get(&format!("{url}{path}"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap();
        assert_eq!(response.status().as_u16(), 404, "{path}");
        let body = response.body_mut().read_to_string().unwrap();
        assert!(
            !body.contains("board_grammar") && !body.contains("files"),
            "{path}: {body}"
        );
    }

    // the path route refuses a task with no board, and a bad task id
    let created = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({
            "context": {"actor": "pi", "tier": "agent", "scope": null},
            "command": {"create_task": {"name": "wire route task"}},
            "at": null,
            "client": "test"
        }))
        .unwrap();
    assert_eq!(created.status().as_u16(), 200);
    let status = ureq::get(&format!("{url}/api/v1/boards/path/0"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap()
        .status();
    assert_eq!(status.as_u16(), 404, "no board yet: {status}");
    let status = ureq::get(&format!("{url}/api/v1/boards/path/99"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap()
        .status();
    assert_eq!(status.as_u16(), 400);

    // a commit with an unparseable board refuses without recording
    let request = json!({
        "context": {"actor": "pi", "tier": "agent", "scope": null},
        "task": 0,
        "files": {"files": [{"path": "board.yaml", "content": "bm90IGEgYm9hcmQK"}]},
        "message": "broken",
        "client": "test"
    });
    // task 0 exists and the board does not parse: the validation refuses
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(request.clone())
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("boards_board_invalid"));
    // an unknown task id still refuses at the id
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({
            "context": {"actor": "pi", "tier": "agent", "scope": null},
            "task": 99,
            "files": {"files": []},
            "message": "nowhere",
            "client": "test"
        }))
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("invalid_task_id"));

    // malformed json refuses as malformed, never panics
    // malformed JSON still refuses as malformed: the bytes name
    // themselves JSON even when they do not parse
    let mut response = ureq::post(&format!("{url}/api/v1/boards/commit"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send(&b"{not json"[..])
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("malformed_request"));

    // the origin policy: a browser page always names its origin, so a
    // request from a non-allowlisted origin refuses on reads and writes
    // alike; a shell origin reads with its origin echoed, never a
    // wildcard; a POST names its body JSON, which a cross-site form
    // cannot
    let header = |name: (&str, &str)| {
        ureq::get(&format!("{url}/api/v1/version"))
            .config()
            .http_status_as_error(false)
            .build()
            .header(name.0, name.1)
            .call()
            .unwrap()
    };
    let mut page = header(("origin", "https://evil.example"));
    assert_eq!(page.status().as_u16(), 403, "a stranger origin refuses");
    let body: Value = page.body_mut().read_to_string().unwrap().parse().unwrap();
    assert_eq!(body["error"]["code"], json!("origin_forbidden"));

    let shell = header(("origin", "tauri://localhost"));
    assert_eq!(shell.status().as_u16(), 200, "the shell origin reads");
    assert_eq!(
        shell
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost"),
        "the origin is echoed, never a wildcard",
    );
    assert!(
        shell.headers().get("vary").is_some(),
        "the answer varies by origin"
    );

    // a browser preflights every cross-origin write (Content-Type:
    // application/json is not a simple request): an allowed origin's
    // preflight answers 204 with its origin echoed and the verbs the
    // doors carry; a stranger's refuses; a preflight naming no origin
    // is no browser's
    let preflight = |origin: Option<&str>| {
        let request = ureq::options(&format!("{url}/api/v1/windows/states"))
            .config()
            .http_status_as_error(false)
            .build();
        let request = match origin {
            Some(value) => request
                .header("origin", value)
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type"),
            None => request,
        };
        request.call().unwrap()
    };
    let shell_preflight = preflight(Some("tauri://localhost"));
    assert_eq!(
        shell_preflight.status().as_u16(),
        204,
        "the shell's preflight answers"
    );
    assert_eq!(
        shell_preflight
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost")
    );
    assert_eq!(
        shell_preflight
            .headers()
            .get("access-control-allow-methods")
            .and_then(|v| v.to_str().ok()),
        Some("GET, POST, PATCH, PUT, DELETE, OPTIONS")
    );
    let stranger_preflight = preflight(Some("https://evil.example"));
    assert_eq!(stranger_preflight.status().as_u16(), 403);
    let bare_preflight = preflight(None);
    assert_eq!(bare_preflight.status().as_u16(), 400);

    // the echo rides writes: a write from an allowed origin carries its
    // origin on the answer, so the client reads the sequence its echo
    // filter and its next settled write need
    let write = ureq::post(&format!("{url}/api/v1/windows/states"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("origin", "tauri://localhost")
        .send_json(json!({"id": "w-cors", "device": "dev-1"}))
        .unwrap();
    assert_eq!(write.status().as_u16(), 200);
    assert_eq!(
        write
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost")
    );

    // the rebinding guard: a domain name that is not this server
    // refuses before routing, and an allowlisted page can read why; an
    // IP literal never comes from a rebinding and answers
    let rebound = |host: &str| {
        ureq::get(&format!("{url}/api/v1/version"))
            .config()
            .http_status_as_error(false)
            .build()
            .header("host", host)
            .header("origin", "tauri://localhost")
            .call()
            .unwrap()
    };
    let mut stranger = rebound("evil.example:8811");
    assert_eq!(stranger.status().as_u16(), 403, "a stranger name refuses");
    assert_eq!(
        stranger
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("tauri://localhost"),
        "the shell reads the refusal"
    );
    let body: Value = stranger
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["error"]["code"], json!("host_forbidden"));
    assert_eq!(
        rebound("192.168.0.51:8811").status().as_u16(),
        200,
        "a LAN IP literal answers"
    );
    assert_eq!(
        rebound("[fd00::5]:8811").status().as_u16(),
        200,
        "an IPv6 literal answers"
    );

    let anonymous = ureq::get(&format!("{url}/api/v1/version"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .unwrap();
    assert_eq!(
        anonymous.status().as_u16(),
        200,
        "no origin header, no page"
    );
    assert!(
        anonymous
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );

    // the forged-write hole: a cross-site form or no-cors POST arrives
    // with text/plain or a form content type and refuses however it came
    let mut form = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("origin", "tauri://localhost")
        .header("content-type", "application/x-www-form-urlencoded")
        .send("context=tier%3Dhuman")
        .unwrap();
    assert_eq!(form.status().as_u16(), 415, "a form body refuses");
    let body: Value = form.body_mut().read_to_string().unwrap().parse().unwrap();
    assert_eq!(body["error"]["code"], json!("content_type_required"));

    let stranger_post = ureq::post(&format!("{url}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("origin", "https://evil.example")
        .header("content-type", "application/json")
        .send("{}")
        .unwrap();
    assert_eq!(
        stranger_post.status().as_u16(),
        403,
        "the stranger origin refuses writes too"
    );

    // the search route answers empty on an empty tracker
    let mut response = ureq::post(&format!("{url}/api/v1/boards/search"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({"terms": ["anything"], "history": false, "within": null}))
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let body: Value = response
        .body_mut()
        .read_to_string()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(body["lines"], json!([]));

    serve.kill().unwrap();
    let _ = serve.wait();
}

#[test]
fn show_and_log_json_read_byte_exact_through_the_cli_face() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let path = scratch_db("json-read-door");
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

    // the name no honest wrapper survives: quotes and backslashes in
    // one line
    let name = "first: \"quoted\" and back\\slashed";
    assert!(sac(&["create", "task", name]).0);
    assert!(sac(&["claim", "t-0"]).0);
    assert!(sac(&["create", "task", "second task"]).0);

    // the task's json face: the authored name byte-exact
    let (ok, out, _) = sac(&["show", "--json", "t-0"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["cursor"], json!(2));
    let block = &json["blocks"][0];
    assert_eq!(block["id"], json!("t-0"));
    assert_eq!(block["name"], json!(name));

    // the bare position on the json face: the raw log row, no fold
    let (ok, out, _) = sac(&["show", "--json", "0"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(json["blocks"][0]["seq"], json!(0));
    assert_eq!(json["blocks"][0]["payload"]["name"], json!(name));

    // the log's json face rides the rows with the cursor beside them
    let (ok, out, _) = sac(&["log", "--json"]);
    assert!(ok, "{out}");
    let log: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(log["cursor"], json!(2));
    let rows: Vec<Value> = log["records"]
        .as_array()
        .expect("the records array")
        .clone();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1]["kind"], json!("task_claimed"));

    // text reads end with the cursor line too: show, log, list. The
    // search face left for the boards: its answers are board lines,
    // not record rows, and carry no cursor
    for args in [
        vec!["show", "t-0"],
        vec!["show", "1"],
        vec!["log"],
        vec!["list"],
    ] {
        let (ok, out, _) = sac(&args);
        assert!(ok, "{}: {out}", args.join(" "));
        assert_eq!(
            out.lines().last().unwrap(),
            "cursor 2",
            "{}: {out}",
            args.join(" ")
        );
    }

    // the pipe's json face: every line's outcome, skips included
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
    let (ok, out, _) = pipe("bogus\nt-0\n", &["show", "--stdin", "--json"]);
    assert!(ok, "{out}");
    let json: Value = serde_json::from_str(&out).expect("the json face parses");
    let rows = json["blocks"].as_array().expect("the blocks array");
    assert_eq!(rows.len(), 2);
    assert_eq!(json["cursor"], json!(2));
    assert_eq!(rows[0]["id"], json!("bogus"));
    assert!(rows[0]["error"].is_string());
    assert_eq!(rows[1]["name"], json!(name));

    // the stdin text face ends with the cursor line too
    let (ok, out, _) = pipe("t-0\n", &["show", "--stdin"]);
    assert!(ok, "{out}");
    assert_eq!(out.lines().last().unwrap(), "cursor 2", "{out}");

    // a number past the id space refuses at the grammar door with the
    // ordinary exit, never a panic: digits are grammar, the id space is not
    let (ok, _, err) = sac(&["show", "184467440737095516160"]);
    assert!(!ok);
    assert!(err.contains("is not an id"), "{err}");
    let (ok, _, err) = sac(&["show", "#184467440737095516160"]);
    assert!(!ok);
    assert!(
        err.contains("the raw record at a position is the bare <seq>"),
        "{err}"
    );
    let (ok, _, err) = sac(&["show", "bogus"]);
    assert!(!ok);
    assert!(
        err.contains("'bogus' is not an id (expected t-<n>, m-<n>, or a bare <seq>)"),
        "{err}"
    );
    // the raw door opens any record: an ownership event renders as
    // the log renders it, header and payload
    let (ok, out, _) = sac(&["show", "1"]);
    assert!(ok, "{out}");
    assert!(
        out.starts_with("seq 1  task_claimed  pi\n"),
        "the raw event rides whole: {out}"
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
        let out = scrubbed(std::process::Command::new(bin))
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

    // a claim answers with its records, both paths, identical shapes
    let claimed = local(&local_db, &["--json", "claim", "t-0"]);
    assert_eq!(claimed["records"][0]["kind"], "task_claimed");
    let (_, body) = post_command(
        &served,
        &json!({"context": {"actor": "pi", "tier": "agent"},
                 "command": {"claim_task": {"id": 0, "incarnation": null}}}),
    );
    let served_claimed = json_of(&body);
    assert_eq!(
        served_claimed["records"][0]["kind"],
        claimed["records"][0]["kind"]
    );

    std::fs::remove_dir_all(local_db.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(served_db.parent().unwrap()).unwrap();
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
if [ "$1" = run ] && [ -n "${FAKE_HC_RUN_WAITS:-}" ]; then
  # the attach-before-register window: the host ask holds until the
  # fixture's trigger file lands, so the newborn can attach while the
  # door's host effect is still outside the writer lock
  n=0
  while [ ! -f "$here/attach-done" ] && [ "$n" -lt 200 ]; do
    sleep 0.1
    n=$((n+1))
  done
fi
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
if [ "$1" = attach ]; then
  if [ -n "${SACCADE_FAKE_HC_ATTACH_FAILS:-}" ]; then
    printf 'no attachable terminal here\n' >&2
    exit 4
  fi
  printf 'attached view\n'
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
    let mut cmd = scrubbed(std::process::Command::new(bin));
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
        let mut cmd = scrubbed(std::process::Command::new(bin));
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

    // this one spawns raw on purpose: the contract under test is the
    // ambient resolution itself (PATH bait, no override), so the
    // helper's fail-safes would change the answer. Only the state
    // root and the cwd still point at scratch.
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
        // the launch composes the caller's location, so this child
        // runs from the scratch workspace
        .current_dir(&workspace)
        .env("XDG_STATE_HOME", suite_state())
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

    let out = scrubbed(std::process::Command::new(bin))
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

/// A sac run with stdin on a pty slave: the here door's attach
/// inherits a real terminal, and the door's own faces come back
/// through collected pipes.
fn run_here_in_pty(bin: &str, args: &[String], envs: &[(&str, String)]) -> std::process::Output {
    let mut masters = Vec::new();
    let mut cmd = scrubbed(std::process::Command::new(bin));
    cmd.args(args.iter().map(String::as_str))
        .stdin(pty_stdio(&mut masters));
    for key in [
        "SACCADE_REPO",
        "SACCADE_DB",
        "SACCADE_SERVER",
        "SACCADE_ACTOR",
        "SACCADE_TASK",
        "SACCADE_INCARNATION_SEQ",
        "SACCADE_WORKSPACE",
        "HC_TERMINAL_ID",
    ] {
        cmd.env_remove(key);
    }
    // no pty door ever resolves the ambient adapter: the default host
    // path denies loudly (no such executable, every ask fails) unless
    // the case names its own fake
    cmd.env(
        "SACCADE_HERDR",
        std::env::temp_dir().join("saccade-pty-deny-hc"),
    );
    cmd.env("SACCADE_TEST_CONTEXT", "1");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("spawn sac");
    for master in masters {
        unsafe { libc::close(master) };
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn the_here_door_launches_through_the_host_then_attaches_this_terminal() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("here-attach");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let workspace = db.parent().unwrap().join("here-workspace");
    std::fs::create_dir_all(&workspace).unwrap();

    let args = |extra: &[&str]| {
        let mut args = vec![
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--workspace".to_string(),
            workspace.to_str().unwrap().to_string(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        args
    };
    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_PI", pi.display().to_string()),
    ];

    // the attached view: a real stdin terminal launches through the
    // host, births the incarnation, then attaches; the attach's own
    // exit is the command's and the json face names the harness
    let out = run_here_in_pty(bin, &args(&["--here", "--json"]), &envs);
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // the attached view rides the inherited stdio beside the reply:
    // the fake's own screen bytes land in the same stream
    let text = String::from_utf8_lossy(&out.stdout);
    let json_at = text.find('{').expect("the json face answers");
    let launched: Value = serde_json::from_str(&text[json_at..]).unwrap();
    assert_eq!(launched["terminal"], json!("t1"), "{launched}");
    assert_eq!(launched["harness"], json!("pi"), "{launched}");
    assert_eq!(launched["resumed"], Value::Null, "{launched}");
    assert_eq!(launched["exit"], json!(0), "{launched}");
    assert!(launched["native_id"].is_string(), "{launched}");
    // the attach went to the launched address, bare
    assert_eq!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "attach"),
        vec!["t1".to_string()]
    );
    // exactly one host run carried the launch, and no label was asked
    // the fake logs one word per line; every ask starts with its verb
    let asks = |verb: &str| {
        std::fs::read_to_string(
            db.parent()
                .unwrap()
                .join("hc-bin")
                .join(format!("argv-{verb}.log")),
        )
        .unwrap_or_default()
        .lines()
        .filter(|line| *line == verb)
        .count()
    };
    assert_eq!(asks("run"), 1, "exactly one host run");
    assert_eq!(asks("label"), 0);
    assert_eq!(asks("ls"), 0);
    assert_eq!(asks("get"), 0);
    // the json face carries no space, no tab, no refusal
    assert_eq!(launched.get("space"), None);
    assert_eq!(launched.get("tab"), None);
    assert_eq!(launched.get("place_refusal"), None);
    // the birth is a launched kind carrying its harness - never this
    // pane, never an adoption
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let fold = json_of(&body);
    let births: Vec<&Value> = fold["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "incarnation_registered")
        .collect();
    assert_eq!(births.len(), 1, "{fold}");
    assert_eq!(births[0]["payload"]["address"], json!("t1"), "{fold}");
    assert_eq!(births[0]["payload"]["adopted"], json!(false), "{fold}");
    assert_eq!(births[0]["payload"]["harness"], json!("pi"), "{fold}");

    // the text face names the detach law: the attach ended, the
    // session keeps running, and the address stands for a reattach
    let out = run_here_in_pty(bin, &args(&["--here"]), &envs);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("the session keeps running"), "{text}");
    assert!(text.contains("hc attach t1"), "{text}");

    // the resumed here face: one get is the liveness ask, no label asks
    let here_saved = db.parent().unwrap().join("hereres1.jsonl");
    std::fs::write(
        &here_saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"hereres001\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n{{\"type\":\"message\",\"id\":\"aa\",\"parentId\":null}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "hereres001", "session_file": here_saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");
    let here_num = json_of(&body)["num"].as_u64().unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": here_num, "address": "t1", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    std::fs::write(
        db.parent().unwrap().join("hc-bin").join("state"),
        b"finished",
    )
    .unwrap();
    let gets_before = asks("get");
    let runs_before = asks("run");
    let resumed_args = vec![
        "--server".to_string(),
        base_url.clone(),
        "--repo".to_string(),
        root.to_str().unwrap().to_string(),
        "launch".to_string(),
        "--here".to_string(),
        "--resume".to_string(),
        format!("i-{here_num}"),
        "--json".to_string(),
    ];
    let out = run_here_in_pty(bin, &resumed_args, &envs);
    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let json_at = text.find('{').expect("the json face answers");
    let resumed_here: Value = serde_json::from_str(&text[json_at..]).unwrap();
    assert_eq!(resumed_here["resumed"], json!(here_num), "{resumed_here}");
    assert_eq!(resumed_here["terminal"], json!("t1"), "{resumed_here}");
    assert_eq!(resumed_here.get("space"), None);
    assert_eq!(resumed_here.get("tab"), None);
    assert_eq!(asks("run"), runs_before + 1, "exactly one more run");
    let gets = asks("get");
    assert_eq!(gets, gets_before + 1, "one get: the liveness ask");
    assert_eq!(asks("label"), 0);
    assert_eq!(asks("ls"), 0);
    assert_eq!(asks("attach"), 3, "one attach per here launch");

    // a refused attach discloses the standing launch: the address and
    // the by-hand command; the birth stands either way
    let mut failed_envs = envs.to_vec();
    failed_envs.push(("SACCADE_FAKE_HC_ATTACH_FAILS", "1".to_string()));
    let out = run_here_in_pty(bin, &args(&["--here"]), &failed_envs);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("the launch stands at address t1"), "{err}");
    assert!(err.contains("hc attach t1"), "{err}");
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == "incarnation_registered")
            .count(),
        4,
        "{body}"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_here_door_refuses_a_pipe_and_a_nested_pane_before_any_launch() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("here-refusals");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let workspace = db.parent().unwrap().join("here-refused");
    std::fs::create_dir_all(&workspace).unwrap();

    // an incarnation already carries a pane: the nested-guard target
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "nestor001", "address": "tp", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    let mut args = vec![
        "--server".to_string(),
        base_url.clone(),
        "--repo".to_string(),
        root.to_str().unwrap().to_string(),
        "launch".to_string(),
        "--workspace".to_string(),
        workspace.to_str().unwrap().to_string(),
        "--here".to_string(),
    ];
    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_PI", pi.display().to_string()),
    ];

    // no real stdin terminal: the attach would inherit a pipe, and the
    // door refuses before any host contact
    let (ok, _out, err, timed_out) = bounded_sac(bin, &args, &envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("stdin must be a real terminal"), "{err}");
    assert!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").is_empty(),
        "the refusal preceded any host contact"
    );

    // the invoking pane is a recorded incarnation's address: a
    // foreground attach would route that incarnation's input into the
    // child, so the door refuses before anything runs
    args.push("--json".to_string());
    let mut nested = envs.to_vec();
    nested.push(("HC_TERMINAL_ID", "tp".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &args, &nested);
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("already the address of i-0"), "{err}");
    assert!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").is_empty(),
        "the nested refusal preceded any host contact"
    );

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resume_runs_its_recorded_harness_and_an_explicit_mismatch_refuses() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("resume-harness");
    let (root, _first, _head) = scratch_repo(db.parent().unwrap());
    let base_url = spawn_workspace_server(&db, &root).await;

    let hc = fake_hc(db.parent().unwrap());
    let pi = stub_pi(db.parent().unwrap());
    let workspace = db.parent().unwrap().join("pi-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let saved = db.parent().unwrap().join("resumee01.jsonl");
    std::fs::write(
        &saved,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"resumee01\",\"timestamp\":\"t\",\"cwd\":\"{}\"}}\n{{\"type\":\"message\",\"id\":\"aa\",\"parentId\":null}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap()}),
    );
    assert_eq!(status, 200, "{body}");

    let resume_args = |extra: &[&str]| {
        let mut args = vec![
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--resume".to_string(),
            "i-0".to_string(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        args
    };
    let envs = [
        ("SACCADE_HERDR", hc.display().to_string()),
        ("SACCADE_PI", pi.display().to_string()),
    ];

    // the recorded kind is the authority: claiming another kind is a
    // wrong claim about the session, refused before any host contact
    let (ok, _out, err, timed_out) = bounded_sac(
        bin,
        &resume_args(&["--harness", "claude-code", "--json"]),
        &envs,
    );
    assert!(!timed_out, "{err}");
    assert!(!ok, "{err}");
    assert!(err.contains("runs pi as recorded"), "{err}");
    assert!(
        hc_argv(&db.parent().unwrap().join("hc-bin"), "run").is_empty(),
        "the mismatch refused before any host contact"
    );

    // the explicit matching claim rides the recorded harness: the
    // launch proceeds and the reply names pi
    let (ok, out, err, timed_out) =
        bounded_sac(bin, &resume_args(&["--harness", "pi", "--json"]), &envs);
    assert!(!timed_out, "{err}");
    assert!(ok, "{out}{err}");
    let launched = json_of(&out);
    assert_eq!(launched["harness"], json!("pi"), "{out}");
    assert_eq!(launched["native_id"], Value::Null, "{out}");
    let run_log = hc_argv(&db.parent().unwrap().join("hc-bin"), "run").join(" ");
    assert!(run_log.contains("--session resumee01"), "{run_log}");

    // the reassignment moved to the attach face: the submitted resume
    // records no address move at all
    let (_, body) = get_json(&base_url, "/api/v1/records");
    let fold = json_of(&body);
    let rows = fold["records"].as_array().unwrap();
    assert!(
        rows.iter().all(|r| r["kind"] != "host_address_assigned"),
        "a submitted resume moves no address: {fold}"
    );

    // the presented pair moves the association at attach evidence
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t1", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records.len(), 1, "{body}");
    assert_eq!(records[0]["kind"], "host_address_assigned", "{body}");
    assert_eq!(records[0]["payload"]["address"], "t1", "{body}");
    assert_eq!(records[0]["payload"]["adopted"], false, "{body}");

    // the equal pair is silent: reload storms re-attach, not re-move
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t1", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        json_of(&body)["records"].as_array().unwrap().is_empty(),
        "{body}"
    );

    // the mode rides the pair: an adopted pane is the recorded kind
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t5", "adopted": true}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records[0]["payload"]["address"], "t5", "{body}");
    assert_eq!(records[0]["payload"]["adopted"], true, "{body}");
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    let resolved = world
        .incarnations
        .values()
        .find(|ctx| ctx.incarnation.native_id().map(|n| n.as_str()) == Some("resumee01"))
        .unwrap();
    assert_eq!(resolved.incarnation.address(), Some("t5"), "{resolved:?}");
    assert!(resolved.incarnation.adopted());

    // a fresh birth keeps its launch-time address: the same-session
    // attach presenting the equal pair binds the locator, moves nothing
    let (ok, out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.to_str().unwrap().to_string(),
            "launch".to_string(),
            "--workspace".to_string(),
            workspace.display().to_string(),
            "--json".to_string(),
        ],
        &envs,
    );
    assert!(ok, "{out}{err}");
    assert!(!timed_out, "{err}");
    let fresh_id = json_of(&out)["native_id"].as_str().unwrap().to_string();
    let fresh_saved = format!("/sessions/{fresh_id}.jsonl");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": fresh_id, "session_file": fresh_saved,
                "terminal": {"address": "t1", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records.len(), 1, "{body}");
    assert_eq!(records[0]["kind"], "saved_session_bound", "{body}");

    // the claude-code face presents the same pair through its own
    // harness: the wire carries it, the birth lands addressed
    let (status, body) = post_json(
        &base_url,
        "/api/v1/native/attach",
        &json!({"native_id": "hookcc001", "session_file": "/sessions/hookcc001.jsonl",
                "harness": "claude-code",
                "terminal": {"address": "t8", "adopted": true}}),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], true, "{body}");
    let records = reply["records"].as_array().unwrap();
    assert_eq!(records[0]["kind"], "incarnation_registered", "{body}");
    assert_eq!(records[0]["payload"]["address"], "t8", "{body}");
    assert_eq!(records[0]["payload"]["adopted"], true, "{body}");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
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
    let mut cmd = scrubbed(std::process::Command::new(bin));
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
    // the marker keeps the ambient arm unreachable
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
    // terminal, the association waits for the attach face, nothing
    // waits on a guard handoff - hc run is synchronous, attach is
    // the harness's
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
    // the door consults no guard for an addressless resume: it succeeded
    // while the guard was held above, and it took nothing itself. After
    // the drop, the resumed session's own attach may already hold the
    // guard - that holder is the session's, never the door's, so the
    // free-state read here would race the child and pin nothing
    drop(held);
    let launched = json_of(&out);
    assert_eq!(
        launched["workspace"],
        json!(workspace.display().to_string()),
        "{out}"
    );
    assert_eq!(launched["terminal"], "t9", "{out}");
    // the launch consults no label ask
    assert_eq!(launched["space"], Value::Null, "{out}");
    assert_eq!(launched["tab"], Value::Null, "{out}");
    assert_eq!(launched["place_refusal"], Value::Null, "{out}");
    let label: Vec<String> = hc_argv(fx.hcs.parent().unwrap(), "label");
    assert!(label.is_empty(), "no label ask rides a resume: {label:?}");

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

    // the address move waits for the attach face: a submitted resume
    // records no move, and the presented pair lands it at evidence
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log will not fold"),
    };
    assert_eq!(
        world.incarnations[&saccade::IncarnationId(RecordId(0))]
            .incarnation
            .address(),
        None,
        "a submitted resume moves no address"
    );
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({"native_id": "resumee01", "session_file": saved.to_str().unwrap(),
                "terminal": {"address": "t9", "adopted": false}}),
    );
    assert_eq!(status, 200, "{body}");
    let records = json_of(&body)["records"].as_array().unwrap().to_vec();
    assert_eq!(records.len(), 1, "{body}");
    assert_eq!(records[0]["kind"], "host_address_assigned", "{body}");
    assert_eq!(records[0]["payload"]["address"], "t9", "{body}");
    stop_holder(&fx.pidfile, &guard);

    // a live recorded address refuses the resume
    let run_log = fx.hcs.parent().unwrap().join("argv-run.log");
    let runs_before = std::fs::read_to_string(&run_log).unwrap();
    let mut live_envs = envs.clone();
    live_envs.push(("SACCADE_FAKE_HC_STATE", "live".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &live_envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "a live address must refuse the resume");
    assert!(
        err.contains("is live at its recorded host address"),
        "{err}"
    );
    let mut unknown_envs = envs.clone();
    unknown_envs.push(("SACCADE_FAKE_HC_STATE", "mangled".to_string()));
    let (ok, _out, err, timed_out) = bounded_sac(bin, &door_args("i-0"), &unknown_envs);
    assert!(!timed_out, "{err}");
    assert!(!ok, "an unprobeable address must refuse the resume");
    assert!(err.contains("liveness cannot be read"), "{err}");
    assert!(err.contains("mangled"), "{err}");
    // the refusals probed the address and submitted no launch
    let runs_after = std::fs::read_to_string(&run_log).unwrap();
    assert_eq!(runs_after, runs_before, "the refusals submitted no launch");

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

fn bind_next_ask(url: &str) -> std::thread::JoinHandle<()> {
    let (_, body) = get_json(url, "/api/v1/records");
    let rows = json_of(&body)["records"].as_array().unwrap().clone();
    let after = rows.last().map(|r| r["seq"].as_u64().unwrap());
    let url = url.to_string();
    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let (native_id, birth) = loop {
            let (status, body) = get_json(&url, "/api/v1/records");
            assert_eq!(status, 200, "{body}");
            let rows = json_of(&body)["records"].as_array().unwrap().clone();
            if let Some(row) = rows.iter().find(|row| {
                row["kind"] == "incarnation_registered"
                    && after.is_none_or(|cursor| row["seq"].as_u64().unwrap() > cursor)
                    && row["payload"]["saved_session"].is_null()
            }) {
                break (
                    row["payload"]["native_id"].as_str().unwrap().to_string(),
                    row["seq"].as_u64().unwrap(),
                );
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no launch birth: {body}"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        std::thread::sleep(std::time::Duration::from_millis(300));
        let (status, body) = get_json(&url, "/api/v1/records");
        assert_eq!(status, 200, "{body}");
        assert!(
            !json_of(&body)["records"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| { row["kind"] == "message_sent" && row["payload"]["to"] == birth }),
            "the initial question preceded native startup: {body}"
        );
        let (status, body) = post_json(
            &url,
            "/api/v1/native/attach",
            &json!({"native_id": native_id, "session_file": format!("/sessions/{native_id}.jsonl")}),
        );
        assert_eq!(status, 200, "{body}");
        assert_eq!(json_of(&body)["incarnation"], birth);
    })
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

    // The initial question waits for the new session's startup binding.
    let workspace = db.parent().unwrap().join("peer-workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let binding = bind_next_ask(&base_url);
    let out = scrubbed(std::process::Command::new(bin))
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
    binding.join().unwrap();
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
    let out = scrubbed(std::process::Command::new(bin))
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
    let out = scrubbed(std::process::Command::new(bin))
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
    let out = scrubbed(std::process::Command::new(bin))
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

    // An unbound launch remains recorded, but no initial question is recorded or sent.
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/address",
        &json!({"incarnation": 2, "address": "t2", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");
    let send_log = hc.parent().unwrap().join("argv-send.log");
    let sends_before = std::fs::read_to_string(&send_log).unwrap();
    let started = std::time::Instant::now();
    let (ok, out, err, timed_out) = bounded_sac(
        bin,
        &[
            "--server".to_string(),
            base_url.clone(),
            "--repo".to_string(),
            root.display().to_string(),
            "launch".to_string(),
            "--workspace".to_string(),
            workspace.display().to_string(),
            "--ask".to_string(),
            "this question waits for startup".to_string(),
        ],
        &[
            ("SACCADE_HERDR", hc.display().to_string()),
            ("SACCADE_PI", pi.display().to_string()),
            ("SACCADE_ACTOR", "creator".to_string()),
            ("SACCADE_INCARNATION_SEQ", creator_birth.to_string()),
            // the test harness alone shortens the binding bound: the
            // timeout path costs seconds, never the production 30
            ("SACCADE_ASK_BIND_BOUND_SECS", "2".to_string()),
        ],
    );
    assert!(!timed_out, "{out}{err}");
    assert!(!ok, "an unbound startup was reported successful: {out}");
    assert!(err.contains("i-3 was launched at address t1"), "{err}");
    assert!(
        err.contains("initial question was not recorded or sent"),
        "{err}"
    );
    assert!(err.contains("within 2 seconds"), "{err}");
    assert!(started.elapsed() >= std::time::Duration::from_secs(2));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(30),
        "the test knob bounded the wait; the production 30 s never ran"
    );
    assert_eq!(std::fs::read_to_string(send_log).unwrap(), sends_before);
    let (_, body) = get_json(&base_url, "/api/v1/records");
    assert_eq!(
        json_of(&body)["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["kind"] == "message_sent")
            .count(),
        3,
        "{body}"
    );

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
    let mut cmd = scrubbed(std::process::Command::new(bin));
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
    let seed = scrubbed(std::process::Command::new(bin))
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

    // piped stdin, terminal stdout and stderr: the human writes
    let out = run_with_terminals(
        bin,
        &args(&[
            "cancel",
            "t-0",
            "--explanation",
            "stopping from a human shell",
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
    assert!(matches!(
        world.tasks[0].task.state,
        saccade::TaskState::Cancelled(_)
    ));

    // no terminal anywhere and no SACCADE_ACTOR: refusal, taught
    let out = run_with_terminals(bin, &args(&["claim", "t-0"]), false, false);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no SACCADE_ACTOR and no terminal"), "{err}");
    assert!(err.contains("export SACCADE_ACTOR"), "{err}");

    // and nothing landed: the task stays cancelled
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert!(matches!(
        world.tasks[0].task.state,
        saccade::TaskState::Cancelled(_)
    ));

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
        scrubbed(std::process::Command::new(bin))
            .args(["serve", "--port", "0", "--db"])
            .arg(db)
            .args(extra)
            .envs(envs.iter().copied())
            .env("XDG_STATE_HOME", suite_state())
            .current_dir(suite_cwd())
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
    saccade::boards::scrub_inherited_repo_env(&mut cmd);
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
    // the state root and the working directory are the helper's own:
    // discovery from the real tree would find the live tracker
    cmd.env("XDG_STATE_HOME", suite_state());
    cmd.current_dir(suite_cwd());
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
        &["--db", db_a.to_str().unwrap(), "claim", "t-0"],
        &[("SACCADE_ACTOR", "pi")],
    );
    assert!(ok, "{err}");
    let (ok, out, err) = sac(&["--server", &url_a, "--json", "list"], &[]);
    assert!(ok, "{err}");
    let list: Value = serde_json::from_str(&out).expect("the json face parses");
    assert_eq!(
        list["tasks"][0]["owner"],
        json!("pi"),
        "the local write shows through the server read: {out}"
    );

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
    let out = saccade::boards::git_command()
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

    let mut serve = scrubbed(std::process::Command::new(bin))
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
    let (_, body) = get_json(&url, "/api/v1/records");
    let kinds: Vec<String> = json_of(&body)["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        kinds.last(),
        Some(&"task_created".to_string()),
        "the write landed in the declared tracker: {kinds:?}"
    );

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
#[tokio::test(flavor = "multi_thread")]
async fn the_task_free_controller_creates_workspaces() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("ws-controller");
    let home = db.parent().unwrap();
    let (root, _first, head) = scratch_repo(home);
    // the serving subprocess carries the fake host, with the liveness
    // state under the test's pen through the state file (finished by
    // default)
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

    // an explicit existing directory, not created by saccade
    let selected_b = home.join("selected-b");
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
    // the retired task-bound effect routes are absent: a 404 that
    // writes nothing
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

    // the CLI face: creation through a real subprocess
    let sac = |args: &[&str], envs: &[(&str, &str)]| {
        let out = scrubbed(std::process::Command::new(bin))
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
    // the provisioned workspace is a recorded birth: the launch
    // carries no space, and no label ask rides it
    assert_eq!(launched["launch"]["space"], Value::Null, "{out}");
    assert_eq!(launched["launch"]["tab"], Value::Null, "{out}");
    assert_eq!(launched["launch"]["place_refusal"], Value::Null, "{out}");
    let label =
        std::fs::read_to_string(home.join("hc-bin").join("argv-label.log")).unwrap_or_default();
    assert!(label.is_empty(), "{label}");

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

    // A provisioned peer binds its native session before receiving the initial question.
    let binding = bind_next_ask(&base_url);
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
            scrubbed(std::process::Command::new(bin))
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
    binding.join().unwrap();
    let asked = json_of(&std::fs::read_to_string(&ask_out).unwrap());
    assert!(
        asked["provisioned"]["branch"]
            .as_str()
            .unwrap()
            .starts_with("inquiry-"),
        "{asked}"
    );
    let newborn = asked["launch"]["incarnation"]["num"].as_u64().unwrap();
    // the i-N counted births only: rover i-0, the trailblaze door
    // birth i-1, the newborn i-2 - creation never counted. The ask
    // resolves its own exact birth, the (harness, native id) pair it
    // composed: a first-address match would name the older trailblaze
    // session, which carries the same fake host address t1
    assert_eq!(newborn, 2, "{asked}");
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

    std::fs::remove_dir_all(home).unwrap();
}

// ---- the workspace lifecycle: ordered cleanup, guards ----

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
async fn the_cleaned_stage_guards_launch() {
    let rig = lifecycle_rig("guards");
    let (_, _branch, ws) = rig_provision(&rig, "guarded-lifecycle");
    let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", &ws]);
    assert!(ok, "the setup clean refuses: {err}");

    // a checkout that stands again at the recorded path (re-created
    // out-of-band) stays recorded Cleaned: a fresh launch into it
    // refuses
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

    // the resume path pins the same guard: a saved session whose
    // header cwd sits in the cleaned workspace refuses - the guard
    // fires before any host submission
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
}

/// A recorded path addresses its own birth: a birth
/// whose stored string carries a symlink segment resolves through it -
/// stored identity preserved, and two stored paths aliasing one
/// directory refuse as ambiguous rather than picking.
#[tokio::test(flavor = "multi_thread")]
async fn the_recorded_path_itself_addresses_the_birth_through_an_alias() {
    let rig = lifecycle_rig("alias");
    // a birth whose recorded path carries a symlink segment:
    // hand-written, exactly what a log may hold
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
    let birth = |at: u64, branch: &str, worktree: &str| {
        let mut conn = saccade::db::open(&rig.dir.join("saccade.db")).unwrap();
        saccade::db::record(
            &mut conn,
            &saccade::store::Context::system(),
            saccade::Command::CreateWorkspace {
                base: saccade::GitCommit::new("abc123".into()).unwrap(),
                branch: saccade::GitBranch::new(branch.into()).unwrap(),
                worktree: saccade::WorktreePath::new(worktree.into()).unwrap(),
            },
            at,
        )
        .unwrap();
    };
    birth(1, "saccade/t-9", recorded.display().to_string().as_str());

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
    // a second stored path over the same directory makes the alias
    // unsafe: both shapes refuse as ambiguous, neither is picked
    birth(3, "saccade/t-10", real.display().to_string().as_str());
    for shape in [recorded.to_str().unwrap(), real.to_str().unwrap()] {
        let (ok, _out, err) = rig_sac(&rig, &["workspace", "clean", shape]);
        assert!(!ok, "the aliased directory resolved to one birth: {err}");
        assert!(err.contains("workspace_ambiguous_path"), "{err}");
    }
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
            harness: saccade::Harness::Pi,
        },
    ];
    for command in commands {
        saccade::db::record(&mut conn, &system, command, now).expect("the fixture story records");
    }
    drop(conn);

    let sac = |args: &[&str]| {
        let out = scrubbed(std::process::Command::new(bin))
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
    // the harness flag names both kinds in the value list
    assert!(
        script.contains("-l harness"),
        "the launch harness flag completes: {script}"
    );
    assert!(script.is_ascii(), "authored text is ASCII");
    assert!(!sac(&["completions", "bash"]).0, "unshipped shells refuse");

    // the candidate doors read the fold: tasks with title and state
    let (ok, out, err) = sac(&["complete", "tasks"]);
    assert!(ok, "complete tasks: {err}");
    assert_eq!(out.trim(), "t-0\timplement floop (open)");

    // workspaces: recorded paths with branch and stage, clean leading
    // Available
    let (ok, out, err) = sac(&["complete", "workspaces-clean"]);
    assert!(ok, "complete workspaces-clean: {err}");
    assert_eq!(
        out.trim(),
        "/tmp/sac-api-floop\tworkspace-floop, available\n\
         /tmp/sac-api-gloom\tworkspace-gloom, cleaned"
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
    let out = scrubbed(std::process::Command::new(bin))
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
    let db = saccade::paths::state_dir_in(&suite_state(), &dir).join("saccade.db");
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
    // no label ask rides the launch
    assert_eq!(reply["space"], Value::Null, "{text}");
    assert_eq!(reply["tab"], Value::Null, "{text}");
    assert_eq!(reply["place_refusal"], Value::Null, "{text}");
    assert!(
        !fake_dir.join("argv-label.log").exists(),
        "no label ask rides the launch"
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
    } = &ctx.incarnation;
    // the pre-allocated id is the one pi will create: it rode argv
    assert!(
        composition.contains(native_id.as_str()),
        "the argv pre-allocates the birth's native id"
    );
    assert!(saved_session.is_none(), "the locator binds at attach");

    // the door-owned session dir exists under the suite's state root,
    // where the helper pointed the child
    let sessions = saccade::paths::state_dir_in(&suite_state(), &dir).join("native-sessions");
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

    std::fs::remove_dir_all(&dir).unwrap();
    let _ = serve.kill();
}

#[test]
fn the_naming_doors_record_every_name_and_label_the_terminal() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-naming-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = saccade::paths::state_dir_in(&suite_state(), &dir).join("saccade.db");
    let workspace = dir.join("ws");
    std::fs::create_dir_all(&workspace).unwrap();
    // the fake host: every verb's argv lands beside it; the first two
    // launches answer t1, a later one t2; a label refuses on demand
    let fake_dir = dir.join("fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    let fake = fake_dir.join("hc");
    std::fs::write(
        &fake,
        r#"#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
printf '%s\n' "$@" >> "$here/argv-$1.log"
if [ "$1" = label ] && [ -n "${FAKE_LABEL_REFUSES:-}" ]; then
  printf '{"error":{"kind":"refused","message":"labels are off"}}\n' >&2
  exit 5
fi
if [ "$1" = run ]; then
  n=$(($(cat "$here/.runs" 2>/dev/null || echo 0) + 1))
  echo "$n" > "$here/.runs"
  addr=t1; [ "$n" -gt 2 ] && addr=t2
  printf '{"address":"%s","cwd":"x","state":"live","turn":0}\n' "$addr"
  exit 0
fi
printf '{}\n'
"#,
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
    let sac = |args: &[&str], envs: &[(&str, &str)]| {
        let out = scrubbed(std::process::Command::new(bin))
            .args(args)
            .env("SACCADE_SERVER", &url)
            .env("SACCADE_HERDR", &fake)
            .env("SACCADE_PI", &pi)
            .env("SACCADE_REPO", &dir)
            .envs(envs.iter().copied())
            .output()
            .expect("spawn sac");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let world = || {
        let conn = saccade::db::open_read(&db).unwrap();
        match saccade::db::load(&conn).unwrap().state {
            saccade::db::LoadState::Full(world) => world,
            _ => panic!("the log will not fold"),
        }
    };
    let name_at = |world: &saccade::World, address: &str| {
        world
            .incarnations
            .iter()
            .filter(|(_, ctx)| ctx.incarnation.address() == Some(address))
            .map(|(id, ctx)| (*id, ctx.name.as_ref().map(|n| n.as_str().to_string())))
            .next_back()
            .expect("a birth carries the address")
    };
    let labels = || hc_argv_all(&fake_dir, "label");
    let ws = workspace.to_str().unwrap();

    // a bad name refuses before the host is ever asked
    let (ok, _, err) = sac(&["launch", "--workspace", ws, "--name", " padded"], &[]);
    assert!(!ok, "a padded name refuses");
    assert!(err.contains("invalid_incarnation_name"), "{err}");
    assert!(
        !fake_dir.join("argv-run.log").exists(),
        "no launch was asked"
    );

    // a named fresh launch records the name with its birth and labels
    // the launched terminal
    let (ok, out, err) = sac(
        &["launch", "--json", "--workspace", ws, "--name", "theory"],
        &[],
    );
    assert!(ok, "{out}{err}");
    let reply: Value = serde_json::from_str(&out).expect("the launch face is json");
    assert_eq!(reply["name"], "theory", "{out}");
    assert_eq!(reply["name_refusal"], Value::Null, "{out}");
    assert_eq!(labels(), ["label", "t1", "--json", "name=theory"]);
    let (first, name) = name_at(&world(), "t1");
    assert_eq!(name.as_deref(), Some("theory"));
    let first_seq = first.0.0.to_string();
    let as_first = [
        ("SACCADE_ACTOR", "pi"),
        ("SACCADE_INCARNATION_SEQ", first_seq.as_str()),
    ];

    // the session reads its own name, then renames itself
    let (ok, out, err) = sac(&["identity", "--json"], &as_first);
    assert!(ok, "{out}{err}");
    assert_eq!(json_of(&out)["name"], "theory", "{out}");
    let (ok, out, err) = sac(&["name", "empirical"], &as_first);
    assert!(ok, "{out}{err}");
    assert!(out.contains("is named empirical"), "{out}");
    assert!(out.contains("label is set on t1"), "{out}");
    assert_eq!(&labels()[4..], ["label", "t1", "--json", "name=empirical"]);
    assert_eq!(name_at(&world(), "t1").1.as_deref(), Some("empirical"));

    // the agents page carries the name fact beside the number
    let (_, body) = get_json(&url, "/view/v1/agents");
    let page = json_of(&body);
    let group = page["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["kind"] == "group")
        .expect("the named session's group");
    assert!(
        group["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["label"] == "name" && f["value"] == "empirical"),
        "{body}"
    );

    // a refused label never fails the naming: the evidence rides the
    // face and the name is recorded
    let (ok, out, err) = sac(
        &["name", "--json", "theory"],
        &[as_first[0], as_first[1], ("FAKE_LABEL_REFUSES", "1")],
    );
    assert!(ok, "{out}{err}");
    let reply = json_of(&out);
    assert_eq!(reply["name"], "theory");
    assert_eq!(reply["label"]["address"], "t1");
    assert!(
        reply["label"]["refusal"]
            .as_str()
            .is_some_and(|r| r.contains("labels are off")),
        "{out}"
    );
    assert_eq!(name_at(&world(), "t1").1.as_deref(), Some("theory"));

    // a second session at the same terminal supersedes the first: names
    // need not be unique, and the superseded holder's old address is
    // never labeled again
    let (ok, out, err) = sac(&["launch", "--workspace", ws, "--name", "theory"], &[]);
    assert!(ok, "{out}{err}");
    assert!(out.contains("named theory"), "{out}");
    let (second, name) = name_at(&world(), "t1");
    assert_ne!(second, first);
    assert_eq!(name.as_deref(), Some("theory"));
    let before = labels().len();
    let (ok, out, err) = sac(&["name", "--clear"], &as_first);
    assert!(ok, "{out}{err}");
    assert!(out.contains("no live terminal address"), "{out}");
    assert_eq!(
        labels().len(),
        before,
        "the superseded address is not labeled"
    );
    assert_eq!(world().incarnations[&first].name, None);

    // an agent names only its own incarnation
    let second_num = format!("i-{}", world().incarnation_num(second).unwrap());
    let (ok, _, err) = sac(
        &["name", "--incarnation", &second_num, "taken over"],
        &as_first,
    );
    assert!(!ok, "an agent cannot name another session");
    assert!(err.contains("not_own_incarnation"), "{err}");
    // the human names any session, the clear included, through the wire
    let (status, body) = post_command(
        &url,
        &envelope(
            "jerry",
            "human",
            json!({"name_incarnation": {"incarnation": second.0.0, "name": null}}),
        ),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(world().incarnations[&second].name, None);
    let namings = world()
        .incarnations
        .values()
        .filter(|ctx| ctx.name.is_some())
        .count();
    assert_eq!(namings, 0);

    std::fs::remove_dir_all(&dir).unwrap();
    let _ = serve.kill();
}

/// Every argv line the fake hc logged for one verb, the op line kept.
fn hc_argv_all(bin_dir: &std::path::Path, verb: &str) -> Vec<String> {
    std::fs::read_to_string(bin_dir.join(format!("argv-{verb}.log")))
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn serve_startup_repairs_boards_and_refuses_a_missing_logged_commit() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let dir = std::env::temp_dir().join(format!("sac-api-startup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    let db = saccade::paths::state_dir_in(&suite_state(), &repo).join("saccade.db");

    // a log whose fold carries a board commit the repository must hold
    let boards =
        saccade::boards::Boards::open(&saccade::paths::state_dir_in(&suite_state(), &repo))
            .expect("the boards repository opens");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(
        work.join("board.yaml"),
        "grammar: 1\ntask: t-0\ncontent: []\n",
    )
    .unwrap();
    let landed = boards
        .commit("0", &work, None, "the recorded account")
        .expect("the objects land");
    let commit = saccade::GitCommit::new(landed.commit.clone()).unwrap();

    let mut conn = saccade::db::open(&db).unwrap();
    saccade::db::record(
        &mut conn,
        &saccade::store::Context {
            actor: saccade::types::actor::ActorName::new("pi".into()).unwrap(),
            tier: saccade::store::Tier::Agent,
            scope: None,
        },
        saccade::Command::CreateTask {
            name: saccade::Prose::new("implement foo".into()).unwrap(),
        },
        1,
    )
    .unwrap();
    saccade::db::record(
        &mut conn,
        &saccade::store::Context {
            actor: saccade::types::actor::ActorName::new("pi".into()).unwrap(),
            tier: saccade::store::Tier::Agent,
            scope: None,
        },
        saccade::Command::CommitBoard {
            task: saccade::TaskId(0),
            commit: commit.clone(),
            parent: None,
            incorporations: Vec::new(),
        },
        2,
    )
    .unwrap();
    drop(conn);
    let _ = boards.repair(&[("0".to_string(), commit.as_str().to_string())]);
    assert_eq!(
        landed.commit,
        commit.as_str(),
        "the test seeds the exact commit the log carries"
    );
    let _ = boards.repair(&[("0".to_string(), commit.as_str().to_string())]);

    let log = std::fs::File::create(dir.join("serve.log")).unwrap();
    let mut cmd = scrubbed(std::process::Command::new(bin));
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
    .stderr(log);
    let mut serve = cmd.spawn().expect("serve spawns");
    std::thread::sleep(std::time::Duration::from_millis(1500));
    assert!(
        serve.try_wait().unwrap().is_none(),
        "serve stays up when the repository holds every logged commit"
    );
    let _ = serve.kill();
    let _ = serve.wait();

    // a logged commit missing from the repository refuses to serve
    std::fs::remove_dir_all(saccade::paths::state_dir_in(&suite_state(), &repo).join("boards.git"))
        .unwrap();
    let log2 = std::fs::File::create(dir.join("serve2.log")).unwrap();
    let mut cmd2 = scrubbed(std::process::Command::new(bin));
    cmd2.args([
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
    .stdout(log2.try_clone().unwrap())
    .stderr(log2);
    let mut serve2 = cmd2.spawn().expect("serve spawns again");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match serve2.try_wait().expect("poll serve") {
            Some(status) => {
                assert!(!status.success(), "the refusal is a failure exit");
                break;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = serve2.kill();
                panic!("serve stayed up past a missing logged commit");
            }
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
    let stderr = std::fs::read_to_string(dir.join("serve2.log")).unwrap();
    assert!(
        stderr.contains("boards repair refused to serve"),
        "{stderr}"
    );
    let _ = serve2.wait();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_helper_run_writes_nothing_outside_its_scratch() {
    let bin = env!("CARGO_BIN_EXE_sac");
    // the host's real state root, read as evidence only: the listing
    // before and after the runs must not move
    let root = real_state_parent().join("saccade");
    let listing = || -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    };
    let before = listing();

    // a bare run from the helper's cwd passes the identity guard and
    // finds no tracker; the refusal lands nowhere
    let out = scrubbed(std::process::Command::new(bin))
        .env("SACCADE_ACTOR", "pin bot")
        .args(["create", "task", "nowhere to land"])
        .output()
        .expect("spawn sac");
    assert!(!out.status.success(), "a bare run finds no tracker");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("pass --db"), "the refusal teaches: {err}");

    // an explicit run through the helper writes into the suite state
    // root and nowhere else
    let scratch = scratch_db("helper-pin");
    let out = scrubbed(std::process::Command::new(bin))
        .env("SACCADE_ACTOR", "pin bot")
        .arg("--db")
        .arg(&scratch)
        .args(["create", "task", "scratch only"])
        .output()
        .expect("spawn sac");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // the run's own evidence sits inside the scratch: the db carries
    // the record and its telemetry rides beside it
    let out = scrubbed(std::process::Command::new(bin))
        .env("SACCADE_ACTOR", "pin bot")
        .arg("--db")
        .arg(&scratch)
        .args(["--json", "list"])
        .output()
        .expect("spawn sac");
    let out_text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out_text.contains("scratch only"), "{out_text}");
    assert!(
        scratch.with_file_name("telemetry.jsonl").exists(),
        "the run's telemetry rides beside the scratch db"
    );

    let after = listing();
    assert_eq!(before, after, "the real state root gained nothing");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_git_dir_inherited_by_the_server_never_reaches_the_board_store() {
    // the hook-leak shape: the server runs with a caller's GIT_DIR and
    // friends set. The decoy is a real bare repo - if the store's git
    // calls resolved the environment, the commit would be visible
    // there
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("decoy-gitdir");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let decoy = scratch.dir.join("decoy.git");
    saccade::boards::git_command()
        .args(["init", "--bare", "--quiet", decoy.to_str().unwrap()])
        .output()
        .expect("the decoy repo initializes");
    let config_before = std::fs::read_to_string(decoy.join("config")).unwrap();
    let objects_before = std::fs::read_dir(decoy.join("objects")).unwrap().count();
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[
            ("GIT_DIR", decoy.to_str().unwrap()),
            ("GIT_COMMON_DIR", decoy.to_str().unwrap()),
            (
                "GIT_OBJECT_DIRECTORY",
                decoy.join("objects").to_str().unwrap(),
            ),
            ("GIT_INDEX_FILE", decoy.join("index").to_str().unwrap()),
            ("GIT_WORK_TREE", project.to_str().unwrap()),
        ],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/command",
        &json!({
            "context": {"actor": "lane", "tier": "agent", "scope": null},
            "command": {"create_task": {"name": "the decoy probe"}},
            "at": null,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = post_json(
        &url,
        "/api/v1/boards/commit",
        &json!({
            "context": {"actor": "lane", "tier": "agent", "scope": null},
            "task": 0,
            "files": {"files": [{"path": "board.yaml",
                "content": "Z3JhbW1hcjogMQp0YXNrOiB0LTAKY29udGVudDoKICAtIHRleHQ6IHRoZSBwcm9iZSBib2FyZCB0aGUgc3RvcmUgbXVzdCBob2xkIGl0c2VsZgo="}]},
            "message": "the decoy probe commit",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["revision"], json!(1));

    // the decoy never heard about any of it
    let config_after = std::fs::read_to_string(decoy.join("config")).unwrap();
    assert_eq!(config_before, config_after, "the decoy's config changed");
    assert_eq!(
        objects_before,
        std::fs::read_dir(decoy.join("objects")).unwrap().count(),
        "objects landed in the decoy"
    );
    let refs = saccade::boards::git_command()
        .arg("--git-dir")
        .arg(&decoy)
        .args(["for-each-ref"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&refs.stdout).trim().is_empty(),
        "refs landed in the decoy"
    );

    // and the store itself holds the board on its own repo
    let (_, got) = get_json(&url, "/api/v1/boards/get?task=0");
    let got = json_of(&got);
    let files = got["files"]["files"].as_array().unwrap();
    assert!(!files.is_empty(), "{got}");

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_send_to_a_superseded_incarnation_refuses_and_discloses_the_supersession() {
    // t-10, the ratified law: an address belongs to one live
    // incarnation. A send to an incarnation whose slot a newer session
    // took refuses at the fold - the host would answer live and the
    // delivery would land in the successor's terminal - and discloses
    // the supersession's evidence: who took the address, at which
    // record
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("recycled-address");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    // the fake host: get answers live for every address; send accepts
    // and logs - a wrong delivery would show in this log
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    // the sender: a registered incarnation declaring its seq in the
    // scope, the way a session's own send door does
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the gone incarnation: registered at the slot; its session is
    // gone and nothing records that
    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "gone00000001",
            "address": "t65",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let gone_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the successor: a fresh session attaches at the same slot, the
    // way a new launch reuses the host's address numbering. The fold
    // derives the supersession from this assignment's record
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "newborn000001",
            "session_file": "/sessions/newborn000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let successor_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // a send addressed to the gone incarnation, by its own identity
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": gone_seq,
            "body": "never lands in the successor's slot",
            "client": "test",
        }),
    );
    // THE LAW: the reply refuses - the send faces' established 409
    // refusal shape - and discloses the supersession: who took the
    // address, at which record, and the message's own reference. The
    // host's log proves nothing was delivered into the slot
    assert_eq!(status, 409, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["error"]["code"], json!("recipient_gone"), "{body}");
    let detail = reply["error"]["detail"].as_str().unwrap();
    assert!(
        detail.contains(&format!("i-{successor_seq}")),
        "the refusal names the superseding incarnation: {body}"
    );
    assert!(
        detail.contains("t65"),
        "the refusal names the contested address: {body}"
    );
    assert!(
        detail.contains("record seq"),
        "the refusal discloses the supersession's record: {body}"
    );
    assert_eq!(
        reply["message"]["seq"],
        json!(3),
        "the message record stands as evidence though nothing delivered"
    );
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log"));
    assert!(
        log.is_err() || !log.unwrap().contains("never lands in the successor's slot"),
        "the host must never receive the message"
    );

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_assignment_of_its_own_un_supersedes_the_incarnation() {
    // the way back: an incarnation superseded at its old slot, then
    // reassigned to a fresh terminal of its own, is live again -
    // delivery reaches it at the new address and never at the old
    // slot's holder
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("un-supersede");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // A: resident at t65
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "a0000000001",
            "session_file": "/sessions/a0000000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let a_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // B takes t65: A is superseded
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "b0000000001",
            "session_file": "/sessions/b0000000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");

    // the hole as it stood: A's delivery refuses
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": a_seq,
            "body": "refused while superseded",
            "client": "test",
        }),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        json_of(&body)["error"]["code"],
        json!("recipient_gone"),
        "{body}"
    );

    // A relaunches onto a fresh terminal of its own: t70. The
    // reassignment is A's way back
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "a0000000001",
            "session_file": "/sessions/a0000000001.jsonl",
            "terminal": {"address": "t70", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(
        reply["records"].as_array().map(Vec::len),
        Some(1),
        "the reassignment lands: {body}"
    );

    // A's delivery submits, at t70 - and never at B's t65
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": a_seq,
            "body": "live again at the new address",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submission"]["state"], json!("submitted"), "{body}");
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log")).unwrap();
    let args: Vec<&str> = log.lines().collect();
    assert!(
        args.contains(&"t70") && !args.contains(&"t65"),
        "the delivery reached t70 and never the old slot: {log}"
    );
    assert!(
        log.contains("live again at the new address"),
        "A's body rode the envelope: {log}"
    );

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_relaunch_that_takes_another_incarnations_address_supersedes_that_holder() {
    // the law on the reassignment path: an existing incarnation whose
    // relaunch lands on an address another incarnation holds
    // supersedes that holder - the older holder's delivery refuses,
    // the mover's own delivery keeps working
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("relaunch-takes-slot");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the mover: resident at t80
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "mover00000001",
            "session_file": "/sessions/mover00000001.jsonl",
            "terminal": {"address": "t80", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let mover_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the neighbor: resident at t65
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "neighbor000001",
            "session_file": "/sessions/neighbor000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let neighbor_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the mover relaunches and the host hands it t65 - the neighbor's
    // slot. The reassignment carries the law with it
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "mover00000001",
            "session_file": "/sessions/mover00000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");

    // the neighbor's delivery refuses, naming the mover
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": neighbor_seq,
            "body": "the neighbor is gone",
            "client": "test",
        }),
    );
    assert_eq!(status, 409, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["error"]["code"], json!("recipient_gone"), "{body}");
    let detail = reply["error"]["detail"].as_str().unwrap();
    assert!(
        detail.contains(&format!("i-{mover_seq}")) && detail.contains("t65"),
        "the refusal names the mover and the address: {body}"
    );

    // the mover's own delivery keeps working at the slot
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": mover_seq,
            "body": "the mover holds the slot now",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submission"]["state"], json!("submitted"), "{body}");
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log")).unwrap();
    assert!(
        log.lines().any(|l| l == "t65"),
        "the mover's delivery submitted at t65: {log}"
    );

    let _ = server.kill();
    let _ = server.wait();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resume_at_its_own_address_never_supersedes_the_incarnation() {
    // the ratified law's exemption: the holder of an address who
    // re-presents the same pair is the same live incarnation - no
    // supersession, and delivery keeps working
    let bin = env!("CARGO_BIN_EXE_sac");
    let scratch = scratch_db("resume-own-address");
    let project = scratch.dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let hc = scratch.dir.join("hc");
    std::fs::write(
        &hc,
        "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-send.log\"\nif [ \"$1\" = get ]; then printf '{\"state\":\"live\"}\\n'; exit 0; fi\nprintf '{}\\n'\nexit 0\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hc, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (url, mut server) = spawn_serving_subprocess(
        &scratch.db,
        &project,
        bin,
        &scratch.dir,
        None,
        &[("SACCADE_HERDR", hc.to_str().unwrap())],
    );

    let (status, body) = post_json(
        &url,
        "/api/v1/incarnations/register",
        &json!({
            "native_id": "sender000001",
            "address": "t90",
            "adopted": false,
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let sender_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the resident: registered and attached at its slot
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "resident000001",
            "session_file": "/sessions/resident000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let resident_seq = json_of(&body)["incarnation"].as_u64().unwrap();

    // the resume: the same native session re-attaches at its own
    // address, the ordinary compaction-and-resume path
    let (status, body) = post_json(
        &url,
        "/api/v1/native/attach",
        &json!({
            "native_id": "resident000001",
            "session_file": "/sessions/resident000001.jsonl",
            "terminal": {"address": "t65", "adopted": false},
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["registered"], json!(false), "{body}");
    assert_eq!(
        reply["incarnation"],
        json!(resident_seq),
        "the resume resolves to the same incarnation"
    );
    assert_eq!(
        reply["records"].as_array().map(Vec::len),
        Some(0),
        "an equal pair records no assignment at all: {body}"
    );

    // delivery still works, and the fold never learned a supersession
    let (status, body) = post_json(
        &url,
        "/api/v1/messages/send",
        &json!({
            "context": {
                "actor": "sender",
                "tier": "agent",
                "scope": {"incarnation": sender_seq},
            },
            "to": resident_seq,
            "body": "still the same live holder",
            "client": "test",
        }),
    );
    assert_eq!(status, 200, "{body}");
    let reply = json_of(&body);
    assert_eq!(reply["submission"]["state"], json!("submitted"), "{body}");
    let log = std::fs::read_to_string(scratch.dir.join("argv-send.log")).unwrap();
    let args: Vec<&str> = log.lines().collect();
    assert!(
        args.contains(&"send") && args.contains(&"t65"),
        "the delivery submitted at the resident's own address: {log}"
    );
    assert!(
        log.contains("still the same live holder"),
        "the body rode the envelope: {log}"
    );

    let _ = server.kill();
    let _ = server.wait();
}

// -- the /view/v1 pages --------------------------------------------------
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

// -- the window store (/api/v1/windows): the suite's window-state door ------------------
//
// t-14 ratified the window store outside the record; t-24@r20 settled the wire
// (moved under /api/v1 by t-28), the storage (its own
// SQLite file beside the event log, own schema version), the sync (the
// pages' hold contract on the window store's own sequence), the drafts (riding
// the window state) and the device identity (an opaque client id).
// These are the wire contracts over a real server on a throwaway db.

fn patch_json(base_url: &str, path: &str, body: &Value) -> (u16, String) {
    let mut response = ureq::patch(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body.clone())
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

fn put_json(base_url: &str, path: &str, body: &Value) -> (u16, String) {
    let mut response = ureq::put(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(body.clone())
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

fn delete_json(base_url: &str, path: &str) -> (u16, String) {
    let mut response = ureq::delete(&format!("{base_url}{path}"))
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .expect("the server answers");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().unwrap();
    (status, text)
}

fn window_states_of(base_url: &str, device: &str) -> Vec<Value> {
    let (_status, body) = get_json(base_url, &format!("/api/v1/windows/states?device={device}"));
    json_of(&body)["states"]
        .as_array()
        .cloned()
        .expect("the states list answers")
}

fn window_state(base_url: &str, device: &str, id: &str) -> Value {
    let (_status, body) = get_json(
        base_url,
        &format!("/api/v1/windows/states/{id}?device={device}"),
    );
    json_of(&body)["state"].clone()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_windows_serve_window_states_outside_the_record() {
    let db = scratch_db("windows");
    let base_url = spawn_server(&db).await;

    // the root names the surface, its schema version and its verbs
    let (status, body) = get_json(&base_url, "/api/v1/windows");
    assert_eq!(status, 200, "{body}");
    let doc = json_of(&body);
    assert_eq!(doc["windows"], 1);
    assert_eq!(doc["schema"], 2);
    assert_eq!(doc["seq"], 0);

    // a read names its device: the window store answers one device's view alone
    let (status, body) = get_json(&base_url, "/api/v1/windows/states");
    assert_eq!(status, 400, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "bad_query");

    // birth: a device creates the state its window shows
    let (status, body) = post_json(
        &base_url,
        "/api/v1/windows/states",
        &json!({"id": "w-alpha", "device": "dev-1", "location": "tracker/t-19", "zoom": 1.2, "width": 1.5}),
    );
    assert_eq!(status, 200, "{body}");
    let doc = json_of(&body);
    assert_eq!(doc["state"]["location"], "tracker/t-19");
    assert_eq!(doc["state"]["zoom"], 1.2);
    assert_eq!(doc["state"]["width"], 1.5);
    assert_eq!(doc["state"]["attached_here"], false);
    assert_eq!(doc["state"]["link"], serde_json::Value::Null);

    // a second birth of the same id refuses: the caller re-reads
    let (status, body) = post_json(
        &base_url,
        "/api/v1/windows/states",
        &json!({"id": "w-alpha", "device": "dev-2"}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "state_exists");

    // attach: the device starts showing the state
    let (status, body) = post_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/attach",
        &json!({"state": "w-alpha"}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["state"]["attached_here"], true);

    // another device's view of the same state: not attached there, and
    // no other device's attached set is ever answered
    let state = window_state(&base_url, "dev-2", "w-alpha");
    assert_eq!(state["attached_here"], false);

    // an unknown state refuses on every door
    let (status, body) = get_json(&base_url, "/api/v1/windows/states/missing?device=dev-1");
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "state_missing");
    let (status, body) = post_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/attach",
        &json!({"state": "missing"}),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "state_missing");

    // a settled write names the device and sets what it names; absent
    // fields stay, null clears
    let (status, body) = patch_json(
        &base_url,
        "/api/v1/windows/states/w-alpha",
        &json!({"device": "dev-1", "part": "/content/2", "zoom": serde_json::Value::Null, "width": serde_json::Value::Null}),
    );
    assert_eq!(status, 200, "{body}");
    let doc = json_of(&body);
    assert_eq!(doc["state"]["part"], "/content/2");
    assert_eq!(doc["state"]["zoom"], serde_json::Value::Null);
    assert_eq!(doc["state"]["width"], serde_json::Value::Null);
    assert_eq!(doc["state"]["location"], "tracker/t-19");
    assert_eq!(doc["state"]["last_device"], "dev-1");

    // drafts ride the state (t-24@r20)
    let (status, body) = patch_json(
        &base_url,
        "/api/v1/windows/states/w-alpha",
        &json!({"device": "dev-1", "drafts": [{"path": "/content/2", "text": "here"}]}),
    );
    assert_eq!(status, 200, "{body}");
    let doc = json_of(&body);
    assert_eq!(doc["state"]["drafts"][0]["path"], "/content/2");
    assert_eq!(doc["state"]["drafts"][0]["text"], "here");

    // the window store file sits beside the event db, never inside it
    let windows_file = db.with_file_name("saccade.db.windows");
    assert!(
        windows_file.exists(),
        "the window store file lives at <db>.windows"
    );
    let conn =
        rusqlite::Connection::open_with_flags(&*db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    let names: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    assert!(
        names
            .iter()
            .all(|name| !name.starts_with("window_") && !name.starts_with("windows_")),
        "the event db carries no window store tables: {names:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_window_links_lapse_and_discard_removes_everywhere() {
    let db = scratch_db("windowslinks");
    let base_url = spawn_server(&db).await;

    for id in ["w-alpha", "w-beta", "w-gamma"] {
        let (status, body) = post_json(
            &base_url,
            "/api/v1/windows/states",
            &json!({"id": id, "device": "dev-1", "location": format!("tracker/t-{id}")}),
        );
        assert_eq!(status, 200, "{body}");
    }
    for id in ["w-alpha", "w-beta"] {
        let (status, _) = post_json(
            &base_url,
            "/api/v1/windows/devices/dev-1/attach",
            &json!({"state": id}),
        );
        assert_eq!(status, 200);
    }

    // links run only between windows attached on the device
    let (status, body) = put_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/links/w-alpha",
        &json!({"to": "w-gamma"}),
    );
    assert_eq!(status, 409, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "not_attached");
    let (status, body) = put_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/links/w-alpha",
        &json!({"to": "w-alpha"}),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "self_link");

    // one link out of a window; many windows may link to one target
    let (status, _) = put_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/links/w-alpha",
        &json!({"to": "w-beta"}),
    );
    assert_eq!(status, 200);
    let (status, _) = put_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/links/w-beta",
        &json!({"to": "w-alpha"}),
    );
    assert_eq!(status, 200);
    let states = window_states_of(&base_url, "dev-1");
    let alpha = states.iter().find(|s| s["id"] == "w-alpha").unwrap();
    assert_eq!(alpha["link"], "w-beta");

    // the target detaches: the link pointing at it lapses; the detached
    // state stays listed
    let (status, _) = post_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/detach",
        &json!({"state": "w-beta"}),
    );
    assert_eq!(status, 200);
    let states = window_states_of(&base_url, "dev-1");
    let alpha = states.iter().find(|s| s["id"] == "w-alpha").unwrap();
    assert_eq!(alpha["link"], serde_json::Value::Null);
    assert!(states.iter().any(|s| s["id"] == "w-beta"));
    let beta = states.iter().find(|s| s["id"] == "w-beta").unwrap();
    assert_eq!(beta["attached_here"], false);

    // a second device never sees the first's links
    let states = window_states_of(&base_url, "dev-2");
    assert!(states.iter().all(|s| s["link"] == serde_json::Value::Null));

    // the rare act: discard removes the state, its attachments and
    // every link touching it, on every device
    let (status, body) = post_json(
        &base_url,
        "/api/v1/windows/states",
        &json!({"id": "w-delta", "device": "dev-2"}),
    );
    assert_eq!(status, 200, "{body}");
    let (status, _) = post_json(
        &base_url,
        "/api/v1/windows/devices/dev-2/attach",
        &json!({"state": "w-delta"}),
    );
    assert_eq!(status, 200);
    let (status, _) = post_json(
        &base_url,
        "/api/v1/windows/devices/dev-1/attach",
        &json!({"state": "w-delta"}),
    );
    assert_eq!(status, 200);
    let (status, _) = put_json(
        &base_url,
        "/api/v1/windows/devices/dev-2/links/w-delta",
        &json!({"to": "w-alpha"}),
    );
    assert_eq!(
        status, 409,
        "dev-2 shows w-delta alone: w-alpha is not attached there"
    );
    let (status, _) = delete_json(&base_url, "/api/v1/windows/states/w-alpha");
    assert_eq!(status, 200);
    for device in ["dev-1", "dev-2"] {
        let states = window_states_of(&base_url, device);
        assert!(!states.iter().any(|s| s["id"] == "w-alpha"));
    }
    let (status, body) = get_json(&base_url, "/api/v1/windows/states/w-alpha?device=dev-1");
    assert_eq!(status, 404, "{body}");
    let (status, body) = delete_json(&base_url, "/api/v1/windows/states/w-alpha");
    assert_eq!(status, 404, "{body}");
    assert_eq!(json_of(&body)["error"]["code"], "state_missing");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_windows_wait_parks_until_a_write_lands_and_answers_quiet() {
    let db = scratch_db("windowswait");
    let base_url = spawn_server(&db).await;

    let (_status, body) = post_json(
        &base_url,
        "/api/v1/windows/states",
        &json!({"id": "w-alpha", "device": "dev-1", "location": "tracker/t-19"}),
    );
    assert_eq!(json_of(&body)["state"]["location"], "tracker/t-19");
    let (status, body) = get_json(&base_url, "/api/v1/windows");
    assert_eq!(status, 200);
    let cursor = json_of(&body)["seq"].as_u64().unwrap();

    // a write landing past the cursor wakes the parked hold: the answer
    // carries a sequence past it, not the quiet one
    let writer = {
        let base_url = base_url.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(400));
            patch_json(
                &base_url,
                "/api/v1/windows/states/w-alpha",
                &json!({"device": "dev-2", "zoom": 1.5}),
            )
        })
    };
    let started = std::time::Instant::now();
    let (status, body) = get_json(
        &base_url,
        &format!("/api/v1/windows/states/w-alpha?device=dev-1&after={cursor}&timeout=10"),
    );
    let waited = started.elapsed();
    assert_eq!(status, 200, "{body}");
    let doc = json_of(&body);
    assert!(doc["seq"].as_u64().unwrap() > cursor, "{doc}");
    assert_eq!(doc["state"]["zoom"], 1.5);
    assert!(
        waited >= std::time::Duration::from_millis(350),
        "the hold parked first: {waited:?}"
    );
    writer.join().unwrap();

    // quiet is a normal answer: with a cursor past every write, the
    // hold expires with the list as it stands, at its standing sequence
    let (status, body) = get_json(&base_url, "/api/v1/windows");
    assert_eq!(status, 200);
    let standing = json_of(&body)["seq"].as_u64().unwrap();
    assert!(standing > cursor);
    let started = std::time::Instant::now();
    let (status, body) = get_json(
        &base_url,
        &format!("/api/v1/windows/states?device=dev-1&after={standing}&timeout=1"),
    );
    let waited = started.elapsed();
    assert_eq!(status, 200, "{body}");
    assert_eq!(json_of(&body)["seq"].as_u64().unwrap(), standing);
    assert!(
        waited >= std::time::Duration::from_millis(900),
        "{waited:?}"
    );
}
