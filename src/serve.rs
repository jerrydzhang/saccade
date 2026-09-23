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
use std::process::exit;
use tracing::{info, warn};

use crate::api::AppState;
use crate::db::{self, ExecuteFail};
use crate::objects::comment::Target;
use crate::objects::proposal::ProposalId;
use crate::objects::task::TaskId;
use crate::views::{
    closed_tasks, forest, next_panel, open_proposals, proposal_view, ribbon_marks, show_view,
    thread_view,
};
use crate::web;
use crate::{ActorName, Command, Context, Prose, RecordId, Reject, Tier};

pub async fn run(
    db_path: &std::path::Path,
    bind: &str,
    port: u16,
    executor: &str,
) -> Result<std::convert::Infallible, String> {
    let addr = format!("{bind}:{port}");
    let repo_root = crate::paths::repo_root(std::path::Path::new("."))
        .map_err(|e| format!("the serving repo: {e}"))?;
    let actor = ActorName::new(executor.to_string())
        .map_err(|e| format!("'{executor}' is not a valid actor name: {e:?}"))?;
    // sessions reach this server at its loopback face, whatever it
    // bound to for the console
    let host = match bind {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        other => other,
    };
    let server_url = format!("http://{host}:{port}");
    let state = match crate::supervisor::RunnerConfig::serving(
        repo_root.clone(),
        actor,
        &server_url,
    ) {
        Some(runner) => {
            let state = AppState::with_runner(db_path, runner)?;
            info!(%server_url, "demands will fire runs; boot scan next");
            state
        }
        None => {
            let state = AppState::open(db_path)?;
            warn!(
                "no pinned executor: SACCADE_PI_PATH was not baked at build and SACCADE_PI is unset; \
                 demands queue but never fire"
            );
            state
        }
    };
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("cannot bind {addr}: {e}"))?;
    if !matches!(bind, "127.0.0.1" | "localhost" | "::1" | "[::1]") {
        eprintln!(
            "bound {bind}: any device that can reach port {port} can write events (human tier)"
        );
    }
    info!(%addr, "serving http");
    let router = router(state.clone());
    // boot reconciliation before the scan: orphans close, then the
    // demands that queued behind them fire
    tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            crate::supervisor::recover(&state, &repo_root);
            crate::supervisor::sweep(&state);
        }
    });
    watch_signals(state);
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

/// Two-strike interrupt: the first Ctrl-C with live runs warns and
/// keeps serving — the operator should know what they are about to
/// orphan; the second kills the children and exits. SIGTERM skips the
/// courtesy.
fn watch_signals(state: AppState) {
    tokio::spawn(async move {
        let runs = state.runs();
        let mut interrupt =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                .expect("SIGINT is watchable");
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("SIGTERM is watchable");
        let mut armed = false;
        loop {
            tokio::select! {
                _ = interrupt.recv() => {
                    if armed || runs.is_empty() {
                        runs.kill_all();
                        exit(130);
                    }
                    let live = runs
                        .ids()
                        .iter()
                        .map(|id| format!("i-{}", id.0 .0))
                        .collect::<Vec<_>>()
                        .join(", ");
                    warn!("active runs: {live}; interrupt again to abandon them");
                    armed = true;
                }
                _ = terminate.recv() => {
                    runs.kill_all();
                    exit(143);
                }
            }
        }
    });
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
        Route::NotFound => page(404, "nothing here — try /"),
    }
}

/// The console page, optionally focused on one task.
fn console(req: &Req, app: &AppState, focus_id: Option<usize>) -> Response {
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(degraded) => {
            return page(
                503,
                &format!("world projection unavailable: {}", degraded.reason),
            );
        }
    };
    let now = db::now_epoch();
    let focused = match focus_id.map(|n| focus(&snapshot.world, n)) {
        Some(Some(f)) => Some(f),
        Some(None) => return page(404, &format!("no task t-{n}", n = focus_id.unwrap_or(0))),
        None => None,
    };
    let c = web::Console {
        forest: forest(&snapshot.world),
        closed: closed_tasks(&snapshot.world),
        gate: open_proposals(&snapshot.world),
        next: next_panel(&snapshot.world, now),
        marks: ribbon_marks(&snapshot.world, now),
        focus: focused,
        form: web::FormState {
            who: req.actor.clone().unwrap_or_default(),
            ..Default::default()
        },
        now,
    };
    html(200, &web::page(&c))
}

/// The focused task's panel facts: identity, open judgment, clustered
/// thread. The strip's movement marks are world-wide, not per focus.
fn focus(world: &crate::store::World, n: usize) -> Option<web::Focus> {
    let show = show_view(world, TaskId(n))?;
    let proposals = open_proposals(world)
        .into_iter()
        .filter(|v| task_num(&v.task) == Some(n))
        .collect();
    Some(web::Focus {
        show,
        proposals,
        thread: thread_view(world, TaskId(n))?,
    })
}

fn respond_post(req: &Req, app: &AppState) -> Response {
    if cross_site(req.sec_fetch_site.as_deref()) {
        return page(403, "cross-site POST refused");
    }
    let fields = parse_form(&req.body);
    match parse_post(&req.url) {
        PostRoute::Compose => match form_field(&fields, "task").parse::<usize>() {
            Ok(n) => compose(req, app, n, &fields),
            Err(_) => page(404, "no task named"),
        },
        PostRoute::Ruling(seq) => match proposal_task(app, seq) {
            Some(n) => rule(req, app, n, &fields, seq),
            None => page(404, &format!("no open proposal #{seq}")),
        },
        PostRoute::Accept(n) => accept(req, app, n, &fields),
        PostRoute::NotFound => page(404, "nothing here — try /"),
    }
}

/// The compose door: the @-compiler turns the body into a command; a
/// fetch gets the thread-section fragment back, anything else a 303 to
/// the comment's home thread.
fn compose(req: &Req, app: &AppState, n: usize, fields: &[(String, String)]) -> Response {
    let raw = form_field(fields, "body");
    let addr = web::compile(raw, Target::Task(TaskId(n)));
    if Prose::new(addr.body.clone()).is_err() {
        return console_reject(req, app, n, fields, "a comment needs words");
    }
    let (context, first_claim) = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return console_reject(req, app, n, fields, &msg),
    };
    let command = Command::Comment {
        target: addr.target,
        body: Prose::new(addr.body.clone()).unwrap(),
        kind: addr.kind,
    };
    match app.execute(&context, command, None, console_request(req)) {
        Ok(stored) => {
            let fired = app.clone();
            tokio::task::spawn_blocking(move || crate::supervisor::sweep(&fired));
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
                match thread_fragment(app, root, seq) {
                    Some(body) => {
                        let mut response = html(200, &body);
                        if let Some(name) = first_claim
                            && let Some(cookie) = actor_cookie(&name)
                        {
                            response.headers_mut().insert(header::SET_COOKIE, cookie);
                        }
                        response
                    }
                    None => page(404, &format!("no task t-{root}")),
                }
            } else {
                redirect(&format!("/t/{root}#c-{seq}"), first_claim.as_deref())
            }
        }
        Err(ExecuteFail::Reject(r)) => console_reject(req, app, n, fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(500, &format!("database: {e}")),
    }
}

/// The judgment door: one form, one name, two buttons. The clicked
/// button's name/value names the ruling; identity is claimed at the
/// act; execute, sweep, and 303 back to the focused task.
fn rule(req: &Req, app: &AppState, n: usize, fields: &[(String, String)], seq: usize) -> Response {
    let command = match form_field(fields, "ruling") {
        "accept" => Command::AcceptProposal {
            id: ProposalId(RecordId(seq)),
        },
        "reject" => {
            let note = form_field(fields, "note");
            if Prose::new(note.to_string()).is_err() {
                return console_reject(req, app, n, fields, "a ruling needs a note");
            }
            Command::RejectProposal {
                id: ProposalId(RecordId(seq)),
                note: Prose::new(note.to_string()).unwrap(),
            }
        }
        _ => return console_reject(req, app, n, fields, "the ruling is accept or reject"),
    };
    let who = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return console_reject(req, app, n, fields, &msg),
    };
    match app.execute(&who.0, command, None, console_request(req)) {
        Ok(_) => {
            let fired = app.clone();
            tokio::task::spawn_blocking(move || crate::supervisor::sweep(&fired));
            redirect(&format!("/t/{n}"), who.1.as_deref())
        }
        Err(ExecuteFail::Reject(r)) => console_reject(req, app, n, fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(500, &format!("database: {e}")),
    }
}

/// The accept door: one form, one name, one button. The receipt stands
/// as deposited above; identity is claimed at the act; execute, sweep,
/// and 303 back to the focused task.
fn accept(req: &Req, app: &AppState, n: usize, fields: &[(String, String)]) -> Response {
    let who = match identity(req, fields) {
        Ok(ok) => ok,
        Err(msg) => return console_reject(req, app, n, fields, &msg),
    };
    let command = Command::AcceptTask { id: TaskId(n) };
    match app.execute(&who.0, command, None, console_request(req)) {
        Ok(_) => {
            let fired = app.clone();
            tokio::task::spawn_blocking(move || crate::supervisor::sweep(&fired));
            redirect(&format!("/t/{n}"), who.1.as_deref())
        }
        Err(ExecuteFail::Reject(r)) => console_reject(req, app, n, fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(500, &format!("database: {e}")),
    }
}

/// The thread section alone, for the fetch swap, naming the comment
/// the swap should land on.
fn thread_fragment(app: &AppState, n: usize, landed: usize) -> Option<String> {
    let snapshot = app.snapshot().ok()?;
    let f = focus(&snapshot.world, n)?;
    Some(web::thread_section(&f, &Default::default(), Some(landed)))
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
        },
        first_claim.then_some(w),
    ))
}

/// Re-render the console with the reason inline and the drafts preserved;
/// a fetch gets the compose section alone.
fn console_reject(
    req: &Req,
    app: &AppState,
    n: usize,
    fields: &[(String, String)],
    msg: &str,
) -> Response {
    if is_fetch(req) {
        let form = web::FormState {
            error: Some(msg.to_string()),
            draft: form_field(fields, "body").to_string(),
            note: form_field(fields, "note").to_string(),
            who: req.actor.clone().unwrap_or_default(),
        };
        return html(400, &web::compose_section(n, &form));
    }
    let snapshot = match app.snapshot() {
        Ok(s) => s,
        Err(_) => return page(503, "world projection unavailable"),
    };
    let now = db::now_epoch();
    let c = web::Console {
        forest: forest(&snapshot.world),
        closed: closed_tasks(&snapshot.world),
        gate: open_proposals(&snapshot.world),
        next: next_panel(&snapshot.world, now),
        marks: ribbon_marks(&snapshot.world, now),
        focus: focus(&snapshot.world, n),
        form: web::FormState {
            error: Some(msg.to_string()),
            draft: form_field(fields, "body").to_string(),
            note: form_field(fields, "note").to_string(),
            who: req.actor.clone().unwrap_or_default(),
        },
        now,
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
        InvalidProposalId => "no open proposal with that id".into(),
        ProposalAlreadyOpen => "that task already has an open proposal".into(),
        InvalidCommentId => "no comment with that id".into(),
        NotBirthAttribution => "only the task's birth attribution or a human may accept".into(),
        InvalidParentTaskId => "no such parent task".into(),
        other => format!("{other:?}"),
    }
}

/// The task an open proposal targets, for routing judgment acts back to
/// the focused thread.
fn proposal_task(app: &AppState, seq: usize) -> Option<usize> {
    let snapshot = app.snapshot().ok()?;
    let view = proposal_view(&snapshot.world, ProposalId(RecordId(seq)))?;
    if view.state != "open" {
        return None;
    }
    task_num(&view.task)
}

enum Route {
    Home,
    Task(usize),
    NotFound,
}

enum PostRoute {
    Compose,
    Ruling(usize),
    Accept(usize),
    NotFound,
}

fn parse_route(url: &str) -> Route {
    let path = url.split('?').next().unwrap_or(url);
    match path {
        "/" => Route::Home,
        _ => match path.strip_prefix("/t/").and_then(|rest| rest.parse().ok()) {
            Some(n) => Route::Task(n),
            None => Route::NotFound,
        },
    }
}

fn parse_post(url: &str) -> PostRoute {
    let path = url.split('?').next().unwrap_or("");
    let num = |s: &str| s.parse::<usize>().ok();
    if path == "/compose" {
        return PostRoute::Compose;
    }
    if let Some(seq) = path
        .strip_prefix("/p/")
        .and_then(|rest| rest.strip_suffix("/ruling"))
        .and_then(num)
    {
        return PostRoute::Ruling(seq);
    }
    if let Some(n) = path
        .strip_prefix("/t/")
        .and_then(|rest| rest.strip_suffix("/accept"))
        .and_then(num)
    {
        return PostRoute::Accept(n);
    }
    PostRoute::NotFound
}

fn html(status: u16, body: &str) -> Response {
    response(status, body)
}

fn page(status: u16, msg: &str) -> Response {
    response(
        status,
        &format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade</title><style>{}</style></head>\n<body>\n<div class=\"err\">{} · <a href=\"/\">console</a></div>\n</body></html>\n",
            web::STYLE,
            esc(msg)
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

fn task_num(id: &str) -> Option<usize> {
    id.strip_prefix("t-").and_then(|n| n.parse().ok())
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
    }

    #[test]
    fn post_routes_and_forms_parse() {
        assert!(matches!(parse_post("/compose"), PostRoute::Compose));
        assert!(matches!(parse_post("/p/9/ruling"), PostRoute::Ruling(9)));
        assert!(matches!(parse_post("/t/3/accept"), PostRoute::Accept(3)));
        assert!(matches!(parse_post("/t/3/comment"), PostRoute::NotFound));
        assert!(matches!(parse_post("/t/3"), PostRoute::NotFound));
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
