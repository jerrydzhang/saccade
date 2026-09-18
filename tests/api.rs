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

/// Post one envelope, get status and body back.
fn post_command(base: &str, envelope: &Value) -> (u16, String) {
    let mut r = ureq::post(&format!("{base}/api/v1/command"))
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(envelope.clone())
        .expect("the loopback server answers");
    (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
}

fn post_raw(base: &str, body: &str) -> (u16, String) {
    let mut r = ureq::post(&format!("{base}/api/v1/command"))
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
    let base = spawn_server(&db).await;

    let (status, body) = post_command(
        &base,
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

    // the second write sees the first: the world cache refolded
    let (status, _) = post_command(
        &base,
        &envelope("pi", "agent", json!({"claim_task": {"id": 0}})),
    );
    assert_eq!(status, 200);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_tier_law_holds_through_http() {
    let db = scratch_db("tierlaw");
    let base = spawn_server(&db).await;

    post_command(
        &base,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "implement foo", "parent_id": null}}),
        ),
    );

    // an unclaimed task cannot be completed: the transition fires first
    let (status, body) = post_command(
        &base,
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
        &base,
        &envelope("pi", "agent", json!({"claim_task": {"id": 0}})),
    );
    let (status, _) = post_command(
        &base,
        &envelope(
            "pi",
            "agent",
            json!({"complete_task": {"id": 0, "receipt": "done"}}),
        ),
    );
    assert_eq!(status, 200);

    // human-only acts refuse at agent tier
    post_command(
        &base,
        &envelope(
            "human person",
            "human",
            json!({"create_task": {"name": "second", "parent_id": null}}),
        ),
    );
    post_command(
        &base,
        &envelope("pi", "agent", json!({"claim_task": {"id": 1}})),
    );
    let (status, body) = post_command(
        &base,
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
        &base,
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
        &base,
        &envelope("saccade", "system", json!({"claim_task": {"id": 0}})),
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    // machinery verbs fail validation under every presentable tier
    let (status, _) = post_command(
        &base,
        &envelope(
            "pi",
            "agent",
            json!({"create_workspace": {"task_id": 0, "base": "abc123", "branch": "saccade/t-0"}}),
        ),
    );
    assert_eq!(status, 400);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_bodies_get_typed_envelopes() {
    let db = scratch_db("malformed");
    let base = spawn_server(&db).await;

    let (status, body) = post_raw(&base, "not json at all");
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["code"], "malformed_request");

    // empty prose refuses at the boundary, not inside the fold
    let (status, body) = post_command(
        &base,
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
        &base,
        &envelope("pi", "agent", json!({"explode_everything": {}})),
    );
    assert_eq!(status, 400);

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}
