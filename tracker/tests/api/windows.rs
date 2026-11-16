// -- the window store (/api/v1/windows): the suite's window-state door ------------------

use serde_json::{Value, json};

use crate::common::{get_json, json_of, post_json, scratch_db, spawn_server};

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
