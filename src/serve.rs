//! The serve edge: one sole-writer process behind the console and the
//! /api/v1 surface alike. Form posts stay the human surface: the actor
//! is claimed at the act, the tier is pinned human, and that wire never
//! carries a tier. Routing, form parsing, execution, and redirects live
//! here; the console renderer lives in web.rs.

use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::info;

use crate::api::AppState;
use crate::db::ExecuteFail;
use crate::objects::comment::{CommentKind, Target};
use crate::objects::task::TaskId;
use crate::views::{closed_tasks, comment_line, forest, show_view, thread_view};
use crate::web;
use crate::{ActorName, Command, Context, Prose, RecordId, Reject, Tier};

pub async fn run(
    db_path: &std::path::Path,
    bind: &str,
    port: u16,
) -> Result<std::convert::Infallible, String> {
    let addr = format!("{bind}:{port}");
    let repo_root = crate::paths::repo_root(std::path::Path::new("."))
        .map_err(|e| format!("the serving repo: {e}"))?;
    let repo_name = web::RepoName::of(&repo_root);
    let state = AppState::open(db_path)?
        .with_artifacts(crate::paths::artifacts_at(&repo_root))
        .with_repo_name(repo_name)
        .with_repo_root(repo_root);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("cannot bind {addr}: {e}"))?;
    if !matches!(bind, "127.0.0.1" | "localhost" | "::1" | "[::1]") {
        eprintln!(
            "bound {bind}: any device that can reach port {port} can write events (human tier)"
        );
    }
    info!(%addr, "serving http");
    let router = router(state);
    axum::serve(listener, router)
        .await
        .expect("axum serves until killed");
    unreachable!("axum::serve returns only on shutdown")
}

/// The whole HTTP surface: the versioned API plus the console fallback.
pub fn router(state: AppState) -> axum::Router {
    crate::api::routes()
        .fallback(get(web_get).post(web_post))
        .with_state(state)
}

struct Req {
    url: String,
    sec_fetch_site: Option<String>,
    sec_fetch_mode: Option<String>,
    actor: Option<String>,
    body: String,
}

/// A fetch compose swaps the returned section in place; a plain form
/// post takes the redirect.
fn is_fetch(req: &Req) -> bool {
    req.sec_fetch_mode.as_deref() == Some("cors")
}

/// A console post as the attempts log receives it: the raw form body,
/// with no client binary to name.
fn console_request(req: &Req) -> crate::attempts::AsReceived {
    crate::attempts::AsReceived {
        client: None,
        raw: req.body.clone(),
    }
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .map(|v| v.to_str().unwrap_or_default().to_string())
}

async fn web_get(
    State(app): State<AppState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
) -> Response {
    let req = Req {
        url: uri.to_string(),
        sec_fetch_site: header_value(&headers, "sec-fetch-site"),
        sec_fetch_mode: header_value(&headers, "sec-fetch-mode"),
        actor: header_value(&headers, "cookie").and_then(|c| cookie_actor(&c)),
        body: String::new(),
    };
    respond_get(&req, &app)
}

async fn web_post(
    State(app): State<AppState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req = Req {
        url: uri.to_string(),
        sec_fetch_site: header_value(&headers, "sec-fetch-site"),
        sec_fetch_mode: header_value(&headers, "sec-fetch-mode"),
        actor: header_value(&headers, "cookie").and_then(|c| cookie_actor(&c)),
        body: String::from_utf8_lossy(&body).into_owned(),
    };
    respond_post(&req, &app)
}

fn respond_get(req: &Req, app: &AppState) -> Response {
    match parse_route(&req.url) {
        Route::Home => console(req, app, None),
        Route::Task(n) => console(req, app, Some(n)),
        Route::Artifact(hash) => artifact_bytes(app, &hash),
        Route::NotFound => page(app, 404, "nothing here — try /"),
    }
}

/// The artifact bytes door: store bytes by sniffed content type, an
/// honest miss for anything the store does not hold. The hash is a
/// validated token before it ever names a path.
fn artifact_bytes(app: &AppState, hash: &str) -> Response {
    let valid = hash.len() == 64 && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if !valid {
        return page(app, 404, "nothing here — try /");
    }
    let Some(dir) = app.artifacts_dir() else {
        return page(app, 404, "artifact unavailable");
    };
    let path = dir.join(hash);
    let Ok(bytes) = std::fs::read(&path) else {
        return page(app, 404, "artifact unavailable");
    };
    let content_type = web::sniff_image(&bytes).unwrap_or("application/octet-stream");
    let mut response = (StatusCode::OK, bytes).into_response();
    if let Ok(value) = header::HeaderValue::from_str(content_type) {
        response.headers_mut().insert(header::CONTENT_TYPE, value);
    }
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}

/// The console page, optionally focused on one task.
fn console(req: &Req, app: &AppState, focus_id: Option<usize>) -> Response {
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(degraded) => {
            return page(
                app,
                503,
                &format!("world projection unavailable: {}", degraded.reason),
            );
        }
    };
    let focused = match focus_id.map(|n| focus(&snapshot.world, n)) {
        Some(Some(f)) => Some(f),
        Some(None) => {
            return page(
                app,
                404,
                &format!("no task t-{n}", n = focus_id.unwrap_or(0)),
            );
        }
        None => None,
    };
    let store = match app.artifacts_dir() {
        Some(dir) => web::ArtifactStore::at(dir.to_path_buf()),
        None => web::ArtifactStore::default(),
    };
    let c = web::Console {
        forest: forest(&snapshot.world),
        closed: closed_tasks(&snapshot.world),
        focus: focused,
        form: web::FormState {
            who: req.actor.clone().unwrap_or_default(),
            ..Default::default()
        },
        store,
        repo_name: app.repo_name().cloned(),
    };
    html(200, &web::page(&c))
}

/// The focused task's panel facts: identity, open judgment, clustered
/// thread. The strip's movement marks are world-wide, not per focus.
fn focus(world: &crate::store::World, n: usize) -> Option<web::Focus> {
    let show = show_view(world, TaskId(n))?;
    Some(web::Focus {
        show,
        thread: thread_view(world, TaskId(n))?,
        refs: crate::views::ref_index(world),
    })
}

fn respond_post(req: &Req, app: &AppState) -> Response {
    if cross_site(req.sec_fetch_site.as_deref()) {
        return page(app, 403, "cross-site POST refused");
    }
    let fields = parse_form(&req.body);
    match parse_post(&req.url) {
        PostRoute::Compose => match form_field(&fields, "task").parse::<usize>() {
            Ok(n) => compose(req, app, n, &fields),
            Err(_) => page(app, 404, "no task named"),
        },
        PostRoute::Accept(n) => accept(req, app, n, &fields),
        PostRoute::Reject(n) => reject(req, app, n, &fields),
        PostRoute::Revise(seq) => revise(req, app, seq, &fields),
        PostRoute::NotFound => page(app, 404, "nothing here — try /"),
    }
}

/// The compose door: the @-compiler turns the body into a command; a
/// fetch gets the thread-section fragment back, anything else a 303 to
/// the comment's home thread.
fn compose(req: &Req, app: &AppState, n: usize, fields: &[(String, String)]) -> Response {
    let raw = form_field(fields, "body");
    let addr = web::compile(raw, Target::Task(TaskId(n)));
    if Prose::new(addr.body.clone()).is_err() {
        return compose_reject(req, app, n, fields, "a comment needs words");
    }
    let (context, first_claim) = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return compose_reject(req, app, n, fields, &msg),
    };
    let command = Command::Comment {
        target: addr.target,
        kind: CommentKind::Note {
            body: Prose::new(addr.body.clone()).unwrap(),
        },
    };
    match app.execute(&context, command, None, console_request(req)) {
        Ok(stored) => {
            let seq = stored.last().map(|s| s.seq).unwrap_or(0);
            let root = match addr.target {
                Target::Task(t) => t.0,
                Target::Comment(c) => app
                    .snapshot()
                    .ok()
                    .and_then(|s| s.world.comments.get(&c).map(|cc| cc.comment.root))
                    .map(|root| root.0)
                    .unwrap_or(n),
            };
            // a rehome (or a reply landing on another task's thread) must
            // not swap the new home's section into the old page: the fetch
            // follows the same 303 a plain form post would take
            if is_fetch(req) && root == n {
                match thread_fragment(app, root) {
                    Some(body) => {
                        let mut response = html(200, &body);
                        if let Some(name) = first_claim
                            && let Some(cookie) = actor_cookie(&name)
                        {
                            response.headers_mut().insert(header::SET_COOKIE, cookie);
                        }
                        response
                    }
                    None => page(app, 404, &format!("no task t-{root}")),
                }
            } else {
                redirect(&format!("/t/{root}#c-{seq}"), first_claim.as_deref())
            }
        }
        Err(ExecuteFail::Reject(r)) => compose_reject(req, app, n, fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(app, 503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(app, 500, &format!("database: {e}")),
    }
}

/// The reject door: the form names the offer it was rendered against,
/// the grounds are required, identity is claimed at the act. A stale
/// form's offer meets the fold's stale-offer guard. A refusal re-renders
/// the page with the reason beside the doors and the grounds kept.
fn reject(req: &Req, app: &AppState, n: usize, fields: &[(String, String)]) -> Response {
    let Some(offer) = submitted_offer(n, fields) else {
        return judge_reject(req, app, n, fields, "the decision names no offer");
    };
    let grounds = form_field(fields, "grounds");
    if Prose::new(grounds.to_string()).is_err() {
        return judge_reject(req, app, n, fields, "a rejection needs grounds");
    }
    let (context, first_claim) = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return judge_reject(req, app, n, fields, &msg),
    };
    let command = Command::Comment {
        target: Target::Comment(offer),
        kind: CommentKind::Reject {
            offer,
            grounds: Prose::new(grounds.to_string()).unwrap(),
        },
    };
    match app.execute(&context, command, None, console_request(req)) {
        Ok(_) => redirect(&format!("/t/{n}"), first_claim.as_deref()),
        Err(ExecuteFail::Reject(r)) => judge_reject(req, app, n, fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(app, 503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(app, 500, &format!("database: {e}")),
    }
}

/// The offer a decision form names — the one its page rendered, never
/// silently the currently standing one.
fn submitted_offer(n: usize, fields: &[(String, String)]) -> Option<crate::CommentId> {
    let seq = form_field(fields, "offer").parse::<usize>().ok()?;
    (seq != usize::MAX && n != usize::MAX).then_some(crate::CommentId(RecordId(seq)))
}

/// The accept door: the form names the offer it was rendered against;
/// identity is claimed at the act; execute and 303 back to the focused
/// task. A stale form's offer meets the fold's stale-offer guard.
fn accept(req: &Req, app: &AppState, n: usize, fields: &[(String, String)]) -> Response {
    let Some(offer) = submitted_offer(n, fields) else {
        return judge_reject(req, app, n, fields, "the decision names no offer");
    };
    let (context, first_claim) = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return judge_reject(req, app, n, fields, &msg),
    };
    let command = Command::Comment {
        target: Target::Comment(offer),
        kind: CommentKind::Accept { offer },
    };
    match app.execute(&context, command, None, console_request(req)) {
        Ok(_) => redirect(&format!("/t/{n}"), first_claim.as_deref()),
        Err(ExecuteFail::Reject(r)) => judge_reject(req, app, n, fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(app, 503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(app, 500, &format!("database: {e}")),
    }
}

/// The revise door: one comment, one body, one name. Identity is
/// claimed at the act, the tier pinned human — the console revises
/// any comment — and the 303 returns to the comment's home thread.
/// A refusal re-renders with the reason and the kept draft at the
/// comment's own reveal, so the wording stays editable in place.
fn revise(req: &Req, app: &AppState, seq: usize, fields: &[(String, String)]) -> Response {
    let Some(root) = comment_root(app, seq) else {
        return page(app, 404, &format!("no comment c-{}", seq));
    };
    let body = form_field(fields, "body");
    if Prose::new(body.to_string()).is_err() {
        return revise_reject(req, app, root, seq, body, fields, "a comment needs words");
    }
    let (context, first_claim) = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return revise_reject(req, app, root, seq, body, fields, &msg),
    };
    let command = Command::ReviseComment {
        id: crate::CommentId(RecordId(seq)),
        body: Prose::new(body.to_string()).unwrap(),
    };
    match app.execute(&context, command, None, console_request(req)) {
        Ok(_) => redirect(&format!("/t/{root}#c-{seq}"), first_claim.as_deref()),
        Err(ExecuteFail::Reject(r)) => {
            revise_reject(req, app, root, seq, body, fields, &reject_text(&r))
        }
        Err(ExecuteFail::Degraded(reason)) => {
            page(app, 503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(app, 500, &format!("database: {e}")),
    }
}

/// Whether the comment carries prose — a refused revise can only ride
/// a row that has a reveal to carry it.
fn comment_has_prose(app: &AppState, seq: usize) -> bool {
    app.snapshot()
        .ok()
        .and_then(|s| comment_line(&s.world, crate::CommentId(RecordId(seq))))
        .is_some_and(|l| !l.body.is_empty())
}

/// The thread a comment lives on, for routing the revise door's
/// redirect back to the comment's home.
fn comment_root(app: &AppState, seq: usize) -> Option<usize> {
    let snapshot = app.snapshot().ok()?;
    snapshot
        .world
        .comments
        .get(&crate::CommentId(RecordId(seq)))
        .map(|c| c.comment.root)
        .map(|root| root.0)
}

/// The thread section alone, for the fetch swap; the reader's place
/// rides with it, never a landing scroll.
fn thread_fragment(app: &AppState, n: usize) -> Option<String> {
    let snapshot = app.snapshot().ok()?;
    let f = focus(&snapshot.world, n)?;
    let store = match app.artifacts_dir() {
        Some(dir) => web::ArtifactStore::at(dir.to_path_buf()),
        None => web::ArtifactStore::default(),
    };
    Some(web::thread_section(&f, &Default::default(), &store))
}

/// The actor's identity is claimed at the act: cookie first, then the
/// form's name field. Returns the context and the first-claim cookie
/// name, or the refusal message.
fn identity(req: &Req, fields: &[(String, String)]) -> Result<(Context, Option<String>), String> {
    let who = req.actor.clone().filter(|a| !a.is_empty()).or_else(|| {
        let w = clean_actor(form_field(fields, "who"));
        (!w.is_empty()).then_some(w)
    });
    let Some(w) = who else {
        return Err("a name is required to record the act".into());
    };
    let first_claim = req.actor.is_none();
    let actor =
        ActorName::new(w.clone()).map_err(|_| "that name is not a valid actor name".to_string())?;
    Ok((
        Context {
            actor,
            tier: Tier::Human,
            scope: None,
        },
        first_claim.then_some(w),
    ))
}

/// The actor name a re-render prefills: the cookie's value when one
/// rides the request, else the name the act itself submitted — a
/// refused attempt keeps its typed name, never re-asks for it.
fn form_who(req: &Req, fields: &[(String, String)]) -> String {
    match req.actor.clone().filter(|a| !a.is_empty()) {
        Some(who) => who,
        None => form_field(fields, "who").to_string(),
    }
}

/// A refused compose re-renders the console with the reason and the
/// draft preserved in the dock; a fetch gets the compose section alone.
fn compose_reject(
    req: &Req,
    app: &AppState,
    n: usize,
    fields: &[(String, String)],
    msg: &str,
) -> Response {
    let form = web::FormState {
        error: Some(msg.to_string()),
        draft: form_field(fields, "body").to_string(),
        who: form_who(req, fields),
        ..Default::default()
    };
    if is_fetch(req) {
        return html(400, &web::compose_section(n, &form));
    }
    reject_page(app, n, form)
}

/// A refused decision re-renders with the reason beside the doors and
/// the refused grounds kept editable — never feedback far from the act.
/// When the world moved past the doors between render and act, the
/// refusal falls to the dock: nowhere-hidden is the one wrong answer.
fn judge_reject(
    req: &Req,
    app: &AppState,
    n: usize,
    fields: &[(String, String)],
    msg: &str,
) -> Response {
    let standing = app
        .snapshot()
        .ok()
        .and_then(|s| show_view(&s.world, TaskId(n)))
        .is_some_and(|s| s.state == "offered");
    let form = web::FormState {
        judge_error: standing.then(|| msg.to_string()),
        error: (!standing).then(|| msg.to_string()),
        // the grounds ride the note draft verbatim: refused wording
        // stays editable, never re-authored
        note: form_field(fields, "grounds").to_string(),
        who: form_who(req, fields),
        ..Default::default()
    };
    reject_page(app, n, form)
}

/// A refused revise re-renders with the reason and the kept draft at
/// the comment's own reveal; a bodyless row carries no reveal, so its
/// refusal falls to the dock.
fn revise_reject(
    req: &Req,
    app: &AppState,
    n: usize,
    seq: usize,
    body: &str,
    fields: &[(String, String)],
    msg: &str,
) -> Response {
    let carried = comment_has_prose(app, seq);
    let form = web::FormState {
        revise: carried.then(|| web::ReviseForm {
            seq,
            draft: body.to_string(),
            error: msg.to_string(),
        }),
        error: (!carried).then(|| msg.to_string()),
        who: form_who(req, fields),
        ..Default::default()
    };
    reject_page(app, n, form)
}

fn reject_page(app: &AppState, n: usize, form: web::FormState) -> Response {
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(_) => return page(app, 503, "world projection unavailable"),
    };
    let c = web::Console {
        forest: forest(&snapshot.world),
        closed: closed_tasks(&snapshot.world),
        focus: focus(&snapshot.world, n),
        form,
        store: match app.artifacts_dir() {
            Some(dir) => web::ArtifactStore::at(dir.to_path_buf()),
            None => web::ArtifactStore::default(),
        },
        repo_name: app.repo_name().cloned(),
    };
    html(400, &web::page(&c))
}

fn redirect(location: &str, set_actor: Option<&str>) -> Response {
    let mut response = (StatusCode::SEE_OTHER, String::new()).into_response();
    if let Ok(loc) = header::HeaderValue::from_str(location) {
        response.headers_mut().insert(header::LOCATION, loc);
    }
    if let Some(name) = set_actor
        && let Some(cookie) = actor_cookie(name)
    {
        response.headers_mut().insert(header::SET_COOKIE, cookie);
    }
    response
}

fn actor_cookie(name: &str) -> Option<header::HeaderValue> {
    header::HeaderValue::from_str(&format!(
        "actor={name}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax"
    ))
    .ok()
}

fn cross_site(sec_fetch: Option<&str>) -> bool {
    matches!(sec_fetch, Some(v) if v != "same-origin" && v != "none")
}

fn cookie_actor(header_value: &str) -> Option<String> {
    header_value.split(';').find_map(|pair| {
        pair.trim()
            .strip_prefix("actor=")
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    })
}

fn clean_actor(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, ' ' | '_' | '-') {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .trim()
        .to_string()
}

fn parse_form(body: &str) -> Vec<(String, String)> {
    body.split('&')
        .filter(|p| !p.is_empty())
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((urldecode(k), urldecode(v)))
        })
        .collect()
}

fn form_field<'a>(fields: &'a [(String, String)], key: &str) -> &'a str {
    fields
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or("")
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3])
                    .ok()
                    .and_then(|h| u8::from_str_radix(h, 16).ok());
                match hex {
                    Some(b) => {
                        out.push(b);
                        i += 3;
                    }
                    None => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn reject_text(r: &Reject) -> String {
    use Reject::*;
    match r {
        ReasonRequired => "words are required".into(),
        HumanOnly => "human-only act".into(),
        InvalidTaskId => "no such task".into(),
        InvalidCommentId => "no comment with that id".into(),
        NotAnOffer => "that comment is not an offer".into(),
        NotCurrentOffer => "that offer is not the task's standing one".into(),
        SelfReliance => "a task cannot rely on its own conclusion".into(),

        SelfAcceptance => {
            "the declared task scope covers this task: a task-bound producer cannot accept its \
             own conclusion"
                .into()
        }
        InvalidParentTaskId => "no such parent task".into(),
        other => format!("{other:?}"),
    }
}

#[derive(Debug)]
enum Route {
    Home,
    Task(usize),
    Artifact(String),
    NotFound,
}
enum PostRoute {
    Compose,
    Accept(usize),
    Reject(usize),
    Revise(usize),
    NotFound,
}

fn parse_route(url: &str) -> Route {
    let path = url.split('?').next().unwrap_or(url);
    match path {
        "/" => Route::Home,
        _ => {
            if let Some(n) = path.strip_prefix("/t/").and_then(|rest| rest.parse().ok()) {
                return Route::Task(n);
            }
            if let Some(hash) = path.strip_prefix("/a/") {
                return Route::Artifact(hash.to_string());
            }
            Route::NotFound
        }
    }
}

fn parse_post(url: &str) -> PostRoute {
    let path = url.split('?').next().unwrap_or("");
    let num = |s: &str| s.parse::<usize>().ok();
    if path == "/compose" {
        return PostRoute::Compose;
    }
    if let Some(n) = path
        .strip_prefix("/t/")
        .and_then(|rest| rest.strip_suffix("/accept"))
        .and_then(num)
    {
        return PostRoute::Accept(n);
    }
    if let Some(n) = path
        .strip_prefix("/t/")
        .and_then(|rest| rest.strip_suffix("/reject"))
        .and_then(num)
    {
        return PostRoute::Reject(n);
    }
    if let Some(seq) = path
        .strip_prefix("/c/")
        .and_then(|rest| rest.strip_suffix("/revise"))
        .and_then(num)
    {
        return PostRoute::Revise(seq);
    }
    PostRoute::NotFound
}

fn html(status: u16, body: &str) -> Response {
    response(status, body)
}

fn page(app: &AppState, status: u16, msg: &str) -> Response {
    let title = app
        .repo_name()
        .map(|n| n.tab_title())
        .unwrap_or_else(|| "saccade".to_string());
    let favicon = app.repo_name().map(|n| web::favicon_link(n.hue));
    response(
        status,
        &format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{title}</title>{favicon}<style>{}</style></head>\n<body>\n<div class=\"err\">{} · <a href=\"/\">console</a></div>\n</body></html>\n",
            web::STYLE,
            esc(msg),
            title = title,
            favicon = favicon.unwrap_or_default(),
        ),
    )
}

fn response(status: u16, body: &str) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body.to_string(),
    )
        .into_response()
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_parse() {
        assert!(matches!(parse_route("/"), Route::Home));
        assert!(matches!(parse_route("/t/17"), Route::Task(17)));
        assert!(matches!(parse_route("/t/17?x=1"), Route::Task(17)));
        assert!(matches!(parse_route("/nope"), Route::NotFound));
        assert!(matches!(parse_route("/t/x"), Route::NotFound));
        assert!(matches!(parse_route("/stream"), Route::NotFound));
        match parse_route("/a/3941d4453740985f0c363433c74070c2c4aa649d8a73fe52a573831c0f87aa7e") {
            Route::Artifact(hash) => assert_eq!(hash.len(), 64),
            other => panic!("the bytes door parses: {other:?}"),
        }
    }

    #[test]
    fn post_routes_and_forms_parse() {
        assert!(matches!(parse_post("/compose"), PostRoute::Compose));
        assert!(matches!(parse_post("/t/3/accept"), PostRoute::Accept(3)));
        assert!(matches!(parse_post("/t/3/reject"), PostRoute::Reject(3)));
        assert!(matches!(parse_post("/c/7/revise"), PostRoute::Revise(7)));
        assert!(matches!(parse_post("/p/9/ruling"), PostRoute::NotFound));
        assert!(matches!(parse_post("/t/3/comment"), PostRoute::NotFound));
        assert!(matches!(parse_post("/t/3"), PostRoute::NotFound));
        assert!(matches!(parse_post("/c/7/edit"), PostRoute::NotFound));
        let fields = parse_form("body=hello+world%3C1%3E&task=12&who=jerry");
        assert_eq!(form_field(&fields, "body"), "hello world<1>");
        assert_eq!(form_field(&fields, "task"), "12");
        assert_eq!(form_field(&fields, "who"), "jerry");
        assert_eq!(form_field(&fields, "missing"), "");
        assert_eq!(urldecode("a%2Bb"), "a+b");
    }

    #[test]
    fn cross_site_posts_refused() {
        assert!(cross_site(Some("cross-site")));
        assert!(cross_site(Some("https://evil.example")));
        assert!(!cross_site(Some("same-origin")));
        assert!(!cross_site(Some("none")));
        assert!(!cross_site(None));
        assert_eq!(cookie_actor("a=b; actor=jerry").as_deref(), Some("jerry"));
        assert_eq!(cookie_actor("actor="), None);
        assert_eq!(clean_actor("je;rry\n"), "je rry");
    }

    #[test]
    fn fetch_is_the_cors_mode_header() {
        let req = |mode: Option<&str>| Req {
            url: "/compose".into(),
            sec_fetch_site: Some("same-origin".into()),
            sec_fetch_mode: mode.map(str::to_string),
            actor: None,
            body: String::new(),
        };
        assert!(is_fetch(&req(Some("cors"))));
        assert!(!is_fetch(&req(Some("navigate"))));
        assert!(!is_fetch(&req(None)));
    }
}
