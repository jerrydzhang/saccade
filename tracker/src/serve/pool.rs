//! The /pool/v1 doors: the suite's window-state pool, the third
//! surface beside /api/v1 (sac's own) and /view/v1 (the pure record
//! pages). The pool is client workspace state OUTSIDE the record
//! (t-14, ratified item 1; the wire and storage settled on t-24@r20):
//! what each suite window shows, its reading height, its zoom, its
//! annotation drafts, each device's attached set and per-device links,
//! kept in the pool's own SQLite file beside the event log and
//! versioned on its own. Sync is the pages' hold contract on the
//! pool's own change sequence: `?after=<seq>` parks wake-driven until
//! a pool write lands or the hold expires, quiet is a normal answer,
//! and every write names its device so a device recognises its own
//! echo. Nothing here reads or writes the record.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::{AppState, LandingWake};
use crate::serve::pool_store::{
    self, LinkFail, OpenFail, Pool, StatePatch, StateRow, StateValue, StoreFail,
};

/// The hold a `?after=` request gets without `?timeout=`, and the
/// ceiling any client may ask for: the pages' own contract.
const DEFAULT_HOLD: Duration = Duration::from_secs(25);
const MAX_HOLD: Duration = Duration::from_secs(55);

/// The pool major this surface serves; the path's `v1` names it.
pub const POOL_MAJOR: u64 = 1;

/// One shared handle per server: the pool file opened on first use
/// beside the db. Other doors never touch the pool, so a tracker that
/// serves no pool traffic creates no pool file.
pub struct PoolShared {
    pool: Mutex<Pool>,
}

impl PoolShared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.pool.lock().expect("the pool lock holds")
    }
}

/// The pool doors' shared state, held by AppState: the lazily opened
/// pool file and the pool's own landing wake. The record's landing
/// wake is a different instance - record writes never wake a pool
/// hold, pool writes never wake a page hold.
#[derive(Clone)]
pub struct PoolDoor {
    shared: Arc<Mutex<Option<Arc<PoolShared>>>>,
    pub wake: LandingWake,
}

impl Default for PoolDoor {
    fn default() -> PoolDoor {
        PoolDoor {
            shared: Arc::new(Mutex::new(None)),
            wake: LandingWake::default(),
        }
    }
}

impl PoolDoor {
    /// The open pool, opening the file on first use at `<db>.pool`.
    /// A file that refuses (a newer schema, unreadable) refuses the
    /// request with its reason and is retried on the next one.
    fn open(&self, db_path: &std::path::Path) -> Result<Arc<PoolShared>, PoolFail> {
        let mut guard = self.shared.lock().expect("the pool cell holds");
        if let Some(shared) = guard.as_ref() {
            return Ok(shared.clone());
        }
        let mut path = db_path.to_path_buf().into_os_string();
        path.push(".pool");
        let pool = Pool::open(std::path::Path::new(&path)).map_err(|fail| match fail {
            OpenFail::TooNew { found } => PoolFail::refused(
                StatusCode::INTERNAL_SERVER_ERROR,
                "pool_too_new",
                format!(
                    "the pool file {} carries schema version {found}, newer than this \
                     tracker writes ({}) - serve it with a newer tracker",
                    path.display(),
                    pool_store::SCHEMA_VERSION
                ),
            ),
            OpenFail::Storage(reason) => {
                PoolFail::refused(StatusCode::INTERNAL_SERVER_ERROR, "storage", reason)
            }
        })?;
        let shared = Arc::new(PoolShared {
            pool: Mutex::new(pool),
        });
        *guard = Some(shared.clone());
        Ok(shared)
    }

    /// Fire the pool wake: every landed pool write calls this, so a
    /// parked `?after=` hold re-reads.
    pub fn wake_writes(&self) {
        self.wake.fire();
    }
}

/// A pool refusal: the same shape the /api/v1 door answers.
struct PoolFail {
    status: StatusCode,
    code: &'static str,
    detail: String,
}

impl PoolFail {
    fn refused(status: StatusCode, code: &'static str, detail: String) -> PoolFail {
        PoolFail {
            status,
            code,
            detail,
        }
    }

    fn bad_query(detail: String) -> PoolFail {
        PoolFail::refused(StatusCode::BAD_REQUEST, "bad_query", detail)
    }

    fn malformed(detail: String) -> PoolFail {
        PoolFail::refused(StatusCode::BAD_REQUEST, "malformed_request", detail)
    }

    fn state_missing(id: &str) -> PoolFail {
        PoolFail::refused(
            StatusCode::NOT_FOUND,
            "state_missing",
            format!("no pool state {id}"),
        )
    }

    fn storage(reason: String) -> PoolFail {
        PoolFail::refused(StatusCode::INTERNAL_SERVER_ERROR, "storage", reason)
    }

    fn response(&self) -> Response {
        (
            self.status,
            Json(json!({"error": {"code": self.code, "detail": self.detail}})),
        )
            .into_response()
    }
}

impl From<String> for PoolFail {
    fn from(reason: String) -> PoolFail {
        PoolFail::storage(reason)
    }
}

impl From<StoreFail> for PoolFail {
    fn from(fail: StoreFail) -> PoolFail {
        match fail {
            StoreFail::Missing(id) => PoolFail::state_missing(&id),
            StoreFail::Exists(id) => PoolFail::refused(
                StatusCode::CONFLICT,
                "state_exists",
                format!("pool state {id} already exists"),
            ),
            StoreFail::Storage(reason) => PoolFail::storage(reason),
        }
    }
}

impl From<LinkFail> for PoolFail {
    fn from(fail: LinkFail) -> PoolFail {
        match fail {
            LinkFail::SelfLink(id) => PoolFail::refused(
                StatusCode::UNPROCESSABLE_ENTITY,
                "self_link",
                format!("a window never links to itself: {id}"),
            ),
            LinkFail::NotAttached(id) => PoolFail::refused(
                StatusCode::CONFLICT,
                "not_attached",
                format!(
                    "links run only between windows attached on the device; {id} is not attached"
                ),
            ),
            LinkFail::Storage(reason) => PoolFail::storage(reason),
        }
    }
}

// -- the documents -------------------------------------------------------

/// One annotation draft riding a state (t-24@r20): the part's path and
/// the draft text.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PoolDraft {
    pub path: String,
    pub text: String,
}

/// The state document: the state's own fields plus the reading
/// device's view of them - whether this device shows the state, and
/// the link this device carries out of it. No other device's attached
/// set is ever answered (t-24@r20).
#[derive(Serialize)]
struct StateDoc {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    part: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zoom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drafts: Option<Vec<PoolDraft>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_device: Option<String>,
    changed: u64,
    attached_here: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    link: Option<String>,
}

fn parse_drafts(raw: Option<&str>) -> Result<Option<Vec<PoolDraft>>, PoolFail> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    serde_json::from_str(raw)
        .map(Some)
        .map_err(|e| PoolFail::storage(format!("stored drafts do not decode: {e}")))
}

fn state_doc(row: StateRow, device: &str, pool: &Pool) -> Result<StateDoc, PoolFail> {
    let attached_here = pool
        .attached(device)
        .map_err(PoolFail::storage)?
        .iter()
        .any(|id| id == &row.id);
    let link = pool.link(device, &row.id).map_err(PoolFail::storage)?;
    Ok(StateDoc {
        id: row.id,
        location: row.location,
        part: row.part,
        zoom: row.zoom,
        drafts: parse_drafts(row.drafts.as_deref())?,
        last_device: row.last_device,
        changed: row.changed,
        attached_here,
        link,
    })
}

/// One annotation-draft list, validated and serialised for storage.
fn drafts_json(drafts: &[PoolDraft]) -> Result<String, PoolFail> {
    serde_json::to_string(drafts).map_err(|e| PoolFail::malformed(e.to_string()))
}

// -- the query -----------------------------------------------------------

/// The query a read accepts: `device=` names the reading device (its
/// attached flag and links are answered to it alone), `after=<seq>`
/// waits, `?timeout=` bounds the hold. The pages' hold contract, on
/// the pool's own sequence.
fn parse_query(
    params: &HashMap<String, String>,
) -> Result<(String, Option<u64>, Duration), PoolFail> {
    let device = match params.get("device") {
        Some(device) if !device.trim().is_empty() => device.clone(),
        _ => {
            return Err(PoolFail::bad_query(
                "device= names the reading device; the pool answers its attached set and \
                 links to it alone"
                    .to_string(),
            ));
        }
    };
    let after = match params.get("after") {
        None => None,
        Some(value) => match value.parse::<u64>() {
            Ok(parsed) => Some(parsed),
            Err(_) => {
                return Err(PoolFail::bad_query(format!(
                    "'{value}' is not a pool sequence; after= names a cursor"
                )));
            }
        },
    };
    let timeout = match params.get("timeout") {
        None => None,
        Some(value) => match value.parse::<u64>() {
            Ok(parsed) => Some(parsed),
            Err(_) => {
                return Err(PoolFail::bad_query(format!(
                    "'{value}' is not a seconds bound; timeout= names seconds"
                )));
            }
        },
    };
    let hold = match after {
        None => Duration::ZERO,
        Some(_) => timeout
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_HOLD)
            .min(MAX_HOLD),
    };
    Ok((device, after, hold))
}

// -- the doors -----------------------------------------------------------

/// The pool root: what the surface is and the verbs it serves.
pub async fn pool_root(State(app): State<AppState>) -> Response {
    let shared = match app.pool_door.open(app.db_path()) {
        Ok(shared) => shared,
        Err(fail) => return fail.response(),
    };
    let guard = shared.lock();
    let seq = match guard.seq() {
        Ok(seq) => seq,
        Err(reason) => return PoolFail::storage(reason).response(),
    };
    Json(json!({
        "pool": POOL_MAJOR,
        "schema": pool_store::SCHEMA_VERSION,
        "seq": seq,
        "verbs": [
            "GET /pool/v1/states?device=<id>[&after=<seq>&timeout=<s>]",
            "POST /pool/v1/states {id, device[, location, part, zoom, drafts]}",
            "GET /pool/v1/states/<id>?device=<id>[&after=<seq>&timeout=<s>]",
            "PATCH /pool/v1/states/<id> {device[, location, part, zoom, drafts]}",
            "DELETE /pool/v1/states/<id>",
            "POST /pool/v1/devices/<device>/attach {state}",
            "POST /pool/v1/devices/<device>/detach {state}",
            "PUT /pool/v1/devices/<device>/links/<from> {to}",
            "DELETE /pool/v1/devices/<device>/links/<from>"
        ],
        "note": "the suite's window-state pool, outside the record: waiting is \
                 ?after=<seq> on the reads, every write names its device"
    }))
    .into_response()
}

/// What a read built now: an answer, or a park (the subject is not
/// there yet - a wait parks through a missing state, like a page
/// parks through a missing board).
enum Built {
    Serve(Response),
    Park,
}

/// The pool reads' wait loop: parse the waiting query, open, park on
/// the pool wake until a pool write lands past the cursor or the hold
/// expires, answer as it stands. Quiet is a normal answer. The hold is
/// wake-driven: the registration enters the wake before the deciding
/// read runs, so a write landing between the read and the park can
/// never be missed. `expired` tells the build the hold is over: a
/// still-parked subject answers its refusal then.
async fn serve_read<F>(
    door: &PoolDoor,
    db_path: &std::path::Path,
    params: &HashMap<String, String>,
    build: F,
) -> Response
where
    F: Fn(&Pool, &str, u64, bool) -> Result<Built, PoolFail>,
{
    let (device, after, hold) = match parse_query(params) {
        Ok(parsed) => parsed,
        Err(fail) => return fail.response(),
    };
    let shared = match door.open(db_path) {
        Ok(shared) => shared,
        Err(fail) => return fail.response(),
    };
    let deadline = Instant::now() + hold;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let served = door
            .wake
            .hold(
                || {
                    let guard = shared.lock();
                    let seq = match guard.seq() {
                        Ok(seq) => seq,
                        Err(reason) => return Some(PoolFail::storage(reason).response()),
                    };
                    if after.is_some_and(|cursor| seq <= cursor) {
                        return None;
                    }
                    match build(&guard, &device, seq, false) {
                        Ok(Built::Serve(response)) => Some(response),
                        Ok(Built::Park) => None,
                        Err(fail) => Some(fail.response()),
                    }
                },
                remaining,
            )
            .await;
        if let Some(response) = served {
            return response;
        }
        if Instant::now() >= deadline {
            // the hold expired quiet: answer as it stands
            let guard = shared.lock();
            let seq = match guard.seq() {
                Ok(seq) => seq,
                Err(reason) => return PoolFail::storage(reason).response(),
            };
            return match build(&guard, &device, seq, true) {
                Ok(Built::Serve(response)) => response,
                Ok(Built::Park) => unreachable!("an expired build answers, never parks"),
                Err(fail) => fail.response(),
            };
        }
    }
}

/// GET /pool/v1/states: every state, with the reading device's
/// attached flag and its links beside them.
pub async fn pool_states(
    State(app): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    serve_read(
        &app.pool_door,
        app.db_path(),
        &params,
        |pool, device, seq, _expired| {
            let rows = pool.states().map_err(PoolFail::storage)?;
            let mut docs = Vec::with_capacity(rows.len());
            for row in rows {
                docs.push(state_doc(row, device, pool)?);
            }
            Ok(Built::Serve(
                Json(json!({"pool": POOL_MAJOR, "seq": seq, "states": docs})).into_response(),
            ))
        },
    )
    .await
}

/// GET /pool/v1/states/<id>: one state, waitable.
pub async fn pool_state_get(
    State(app): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    serve_read(
        &app.pool_door,
        app.db_path(),
        &params,
        move |pool, device, seq, expired| match pool.state(&id).map_err(PoolFail::storage)? {
            Some(row) => Ok(Built::Serve(
                Json(json!({
                    "pool": POOL_MAJOR,
                    "seq": seq,
                    "state": state_doc(row, device, pool)?
                }))
                .into_response(),
            )),
            None if expired => Err(PoolFail::state_missing(&id)),
            None => Ok(Built::Park),
        },
    )
    .await
}

/// POST /pool/v1/states: birth of a state. An id that already exists
/// refuses - the caller re-reads instead of clobbering.
#[derive(Deserialize)]
pub struct CreateBody {
    id: String,
    device: String,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    part: Option<String>,
    #[serde(default)]
    zoom: Option<f64>,
    #[serde(default)]
    drafts: Option<Vec<PoolDraft>>,
}

pub async fn pool_state_create(State(app): State<AppState>, body: Json<CreateBody>) -> Response {
    let drafts = match body.drafts.as_deref().map(drafts_json) {
        None => None,
        Some(Ok(json)) => Some(json),
        Some(Err(fail)) => return fail.response(),
    };
    write_door(app, |guard| {
        let seq = guard.create(
            &body.id,
            &body.device,
            body.location.as_deref(),
            body.part.as_deref(),
            body.zoom,
            drafts.as_deref(),
        )?;
        let row = guard
            .state(&body.id)
            .map_err(PoolFail::storage)?
            .ok_or_else(|| PoolFail::state_missing(&body.id))?;
        let doc = state_doc(row, &body.device, guard)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq, "state": doc}))
    })
    .await
}

/// A PATCH field's three states on the wire: absent from the body
/// (untouched - the field's `default` None), present as null (cleared
/// - `Some(None)`), present as a value (written - `Some(Some(v))`).
///
/// Plain `Option<Option<T>>` cannot tell null from absent, so the
/// present cases deserialize through this visitor.
fn double_option<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(Some)
}

/// PATCH /pool/v1/states/<id>: a settled change to the state's own
/// fields. Absent fields stay; null clears.
#[derive(Deserialize)]
pub struct PatchBody {
    device: String,
    #[serde(default, deserialize_with = "double_option")]
    location: Option<Option<String>>,
    #[serde(default, deserialize_with = "double_option")]
    part: Option<Option<String>>,
    #[serde(default, deserialize_with = "double_option")]
    zoom: Option<Option<f64>>,
    #[serde(default, deserialize_with = "double_option")]
    drafts: Option<Option<Vec<PoolDraft>>>,
}

pub async fn pool_state_patch(
    State(app): State<AppState>,
    Path(id): Path<String>,
    body: Json<PatchBody>,
) -> Response {
    let drafts_text: Option<Option<String>> = match body.drafts.as_ref() {
        None => None,
        Some(None) => Some(None),
        Some(Some(list)) => match drafts_json(list) {
            Ok(json) => Some(Some(json)),
            Err(fail) => return fail.response(),
        },
    };
    let patch = StatePatch {
        location: body.location.as_ref().map(|v| match v {
            Some(text) => StateValue::Text(text.as_str()),
            None => StateValue::Clear,
        }),
        part: body.part.as_ref().map(|v| match v {
            Some(text) => StateValue::Text(text.as_str()),
            None => StateValue::Clear,
        }),
        zoom: body.zoom.map(|v| match v {
            Some(number) => StateValue::Number(number),
            None => StateValue::Clear,
        }),
        drafts: drafts_text.as_ref().map(|v| match v {
            Some(text) => StateValue::Text(text.as_str()),
            None => StateValue::Clear,
        }),
    };
    write_door(app, |guard| {
        let seq = guard.update(&id, &body.device, &patch)?;
        let row = guard
            .state(&id)
            .map_err(PoolFail::storage)?
            .ok_or_else(|| PoolFail::state_missing(&id))?;
        let doc = state_doc(row, &body.device, guard)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq, "state": doc}))
    })
    .await
}

/// DELETE /pool/v1/states/<id>: the rare act - the state leaves the
/// pool on every device, its attachments and every link touching it
/// with it.
pub async fn pool_state_discard(State(app): State<AppState>, Path(id): Path<String>) -> Response {
    write_door(app, |guard| {
        let seq = guard.discard(&id)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq}))
    })
    .await
}

/// POST /pool/v1/devices/<device>/attach: this device starts showing
/// the state.
#[derive(Deserialize)]
pub struct StateRef {
    state: String,
}

pub async fn pool_attach(
    State(app): State<AppState>,
    Path(device): Path<String>,
    body: Json<StateRef>,
) -> Response {
    write_door(app, |guard| {
        let seq = guard.attach(&device, &body.state)?;
        let row = guard
            .state(&body.state)
            .map_err(PoolFail::storage)?
            .ok_or_else(|| PoolFail::state_missing(&body.state))?;
        let doc = state_doc(row, &device, guard)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq, "state": doc}))
    })
    .await
}

/// POST /pool/v1/devices/<device>/detach: this device stops showing
/// the state; the state stays. Links on this device pointing at it
/// lapse in the same write (t-14).
pub async fn pool_detach(
    State(app): State<AppState>,
    Path(device): Path<String>,
    body: Json<StateRef>,
) -> Response {
    write_door(app, |guard| {
        let seq = guard.detach(&device, &body.state)?;
        let row = guard
            .state(&body.state)
            .map_err(PoolFail::storage)?
            .ok_or_else(|| PoolFail::state_missing(&body.state))?;
        let doc = state_doc(row, &device, guard)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq, "state": doc}))
    })
    .await
}

/// PUT /pool/v1/devices/<device>/links/<from>: set or replace this
/// device's one outgoing link.
#[derive(Deserialize)]
pub struct LinkBody {
    to: String,
}

pub async fn pool_link_set(
    State(app): State<AppState>,
    Path((device, from)): Path<(String, String)>,
    body: Json<LinkBody>,
) -> Response {
    write_door(app, |guard| {
        let seq = guard.set_link(&device, &from, &body.to)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq, "from": from, "to": body.to}))
    })
    .await
}

/// DELETE /pool/v1/devices/<device>/links/<from>: unlink. Unlinking a
/// window that carries no link answers the standing sequence.
pub async fn pool_link_clear(
    State(app): State<AppState>,
    Path((device, from)): Path<(String, String)>,
) -> Response {
    write_door(app, |guard| {
        let seq = guard.clear_link(&device, &from)?;
        Ok(json!({"pool": POOL_MAJOR, "seq": seq}))
    })
    .await
}

/// The one write-door shape: open, lock, act, wake the parked holds,
/// answer.
async fn write_door<F>(app: AppState, act: F) -> Response
where
    F: FnOnce(&Pool) -> Result<serde_json::Value, PoolFail>,
{
    let door = &app.pool_door;
    let shared = match door.open(app.db_path()) {
        Ok(shared) => shared,
        Err(fail) => return fail.response(),
    };
    let outcome = {
        let guard = shared.lock();
        act(&guard)
    };
    match outcome {
        Ok(value) => {
            door.wake_writes();
            Json(value).into_response()
        }
        Err(fail) => fail.response(),
    }
}
