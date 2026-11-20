// ---- the command door: the wire, the tiers, discovery ----

use serde_json::{Value, json};

use saccade::record::events::Command;
use saccade::record::types::prose::Prose;
use saccade::store::Context;

use crate::common::{
    attempts_lines, envelope, fake_hc, get_json, hc_argv, json_of, post_command, post_json,
    scratch_db, scratch_repo, spawn_server, spawn_serving_subprocess,
};

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
                tags: Vec::new(),
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
    use saccade::client::{self, ClientFail};
    use saccade::record::events::Command;
    use saccade::store::{Context, Tier};
    use saccade::types::actor::ActorName;

    let agent = Context {
        actor: ActorName::new("saccade bot".into()).unwrap(),
        tier: Tier::Agent,
        scope: None,
    };
    let command = || Command::CreateTask {
        tags: Vec::new(),
        name: saccade::record::types::prose::Prose::new("across the line".into()).unwrap(),
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
    use saccade::client::{self, ClientFail};
    use saccade::record::events::Command;
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
            tags: Vec::new(),
            name: saccade::record::types::prose::Prose::new("client smoke".into()).unwrap(),
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
            id: saccade::record::objects::task::TaskId(9),
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
            tags: Vec::new(),
            name: saccade::record::types::prose::Prose::new("never lands".into()).unwrap(),
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

/// The attempts log names its door: a landed and a refused attempt
/// each carry the door that judged them.
#[tokio::test(flavor = "multi_thread")]
async fn the_attempts_line_names_its_door() {
    let db = scratch_db("doorline");
    let base_url = spawn_server(&db).await;

    // a landed write through a helper door: the line names the route
    let (status, body) = post_json(
        &base_url,
        "/api/v1/incarnations/register",
        &json!({"native_id": "doorline001", "address": "tp", "adopted": false}),
    );
    assert_eq!(status, 200, "{body}");

    // a refused write through the command door names that door
    let (status, body) = post_command(
        &base_url,
        &envelope(
            "pi",
            "agent",
            json!({"claim_task": {"id": 9, "incarnation": null}}),
        ),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "invalid_task_id");

    let lines = attempts_lines(&db);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(lines[0]["door"], "incarnations/register");
    assert_eq!(lines[1]["door"], "command");

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}
