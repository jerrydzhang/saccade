//! The serve edge: canvas, dock, and stream over HTTP — the served
//! rendering of the view model; the CLI is the first.
//! Reads fold the world per request; WAL keeps that safe beside CLI writes.
//! Writes are the human surface: the actor is claimed at the act, the tier is
//! pinned human, and the wire never carries a tier.

use saccade::db::{self, ExecuteFail, LoadState};
use saccade::objects::comment::{CommentId, Target};
use saccade::objects::proposal::ProposalId;
use saccade::objects::task::TaskId;
use saccade::views::{
    CommentLine, ProposalView, TaskView, proposal_view, show_view, task_view, thread_view,
};
use saccade::{Command, Context, Prose, RecordId, Reject, Tier, World};
use std::io::Cursor;

pub fn run(
    db_path: &std::path::Path,
    bind: &str,
    port: u16,
) -> Result<std::convert::Infallible, String> {
    let addr = format!("{bind}:{port}");
    let server = tiny_http::Server::http(&addr).map_err(|e| format!("cannot bind {addr}: {e}"))?;
    if !matches!(bind, "127.0.0.1" | "localhost" | "::1" | "[::1]") {
        eprintln!(
            "bound {bind}: any device that can reach port {port} can write events (human tier)"
        );
    }
    eprintln!("serving http://{addr}");
    for request in server.incoming_requests() {
        let mut req = Req {
            method: request.method().to_string(),
            url: request.url().to_string(),
            sec_fetch: None,
            actor: None,
            body: String::new(),
        };
        for h in request.headers() {
            if h.field.equiv("Sec-Fetch-Site") {
                req.sec_fetch = Some(h.value.as_str().to_string());
            } else if h.field.equiv("Cookie") {
                req.actor = cookie_actor(h.value.as_str());
            }
        }
        let mut request = request;
        if req.method == "POST" {
            let _ = request.as_reader().read_to_string(&mut req.body);
        }
        let _ = request.respond(respond(&req, db_path));
    }
    unreachable!("the incoming-requests iterator never ends")
}

struct Req {
    method: String,
    url: String,
    sec_fetch: Option<String>,
    actor: Option<String>,
    body: String,
}

fn respond(req: &Req, db_path: &std::path::Path) -> tiny_http::Response<Cursor<Vec<u8>>> {
    match req.method.as_str() {
        "GET" => respond_get(req, db_path),
        "POST" => respond_post(req, db_path),
        _ => page(405, "GET and POST only."),
    }
}

fn respond_get(req: &Req, db_path: &std::path::Path) -> tiny_http::Response<Cursor<Vec<u8>>> {
    let conn = match db::open_read(db_path) {
        Ok(c) => c,
        Err(e) => return page(500, &format!("database: {e}")),
    };
    let loadout = match db::load(&conn) {
        Ok(l) => l,
        Err(e) => return page(500, &format!("database: {e}")),
    };
    let mut form = FormState::default();
    form.need_who = req.actor.is_none();
    match parse_route(&req.url) {
        Route::Stream => html(200, &render_stream(&loadout.rows)),
        Route::Canvas(dock) => match loadout.state {
            LoadState::Full(world) => {
                html(200, &render_canvas(&world, &canvas(&world), dock, &form))
            }
            LoadState::Degraded(reason) => {
                page(503, &format!("world projection unavailable: {reason}"))
            }
        },
        Route::Task(n, reply) => match loadout.state {
            LoadState::Full(world) => match task_view(&world, TaskId(n)) {
                Some(view) => {
                    form.reply = reply;
                    html(200, &render_object(&view, &world, &form))
                }
                None => page(404, &format!("no task t-{n}")),
            },
            LoadState::Degraded(reason) => {
                page(503, &format!("world projection unavailable: {reason}"))
            }
        },
        Route::NotFound => page(404, "nothing here — try / or /stream"),
    }
}

fn respond_post(req: &Req, db_path: &std::path::Path) -> tiny_http::Response<Cursor<Vec<u8>>> {
    if cross_site(req.sec_fetch.as_deref()) {
        return page(403, "cross-site POST refused");
    }
    let fields = parse_form(&req.body);
    let (n, command, anchor): (usize, Command, bool) = match parse_post(&req.url) {
        PostRoute::Comment(n) => {
            let body = form_field(&fields, "body");
            if Prose::new(body.to_string()).is_err() {
                return post_reject(req, db_path, n, &fields, "a comment needs words");
            }
            let target = match form_field(&fields, "reply").parse::<usize>() {
                Ok(seq) => Target::Comment(CommentId(RecordId(seq))),
                Err(_) => Target::Task(TaskId(n)),
            };
            (
                n,
                Command::Comment {
                    target,
                    body: Prose::new(body.to_string()).unwrap(),
                },
                true,
            )
        }
        PostRoute::Accept(seq) => match proposal_task(db_path, seq) {
            Some(n) => (
                n,
                Command::AcceptProposal {
                    id: ProposalId(RecordId(seq)),
                },
                false,
            ),
            None => return page(404, &format!("no open proposal #{seq}")),
        },
        PostRoute::Reject(seq) => match proposal_task(db_path, seq) {
            None => return page(404, &format!("no open proposal #{seq}")),
            Some(n) => {
                let note = form_field(&fields, "note");
                if Prose::new(note.to_string()).is_err() {
                    return post_reject(req, db_path, n, &fields, "a ruling needs a note");
                }
                (
                    n,
                    Command::RejectProposal {
                        id: ProposalId(RecordId(seq)),
                        note: Prose::new(note.to_string()).unwrap(),
                    },
                    false,
                )
            }
        },
        PostRoute::NotFound => return page(404, "nothing here — try / or /stream"),
    };
    let who = req.actor.clone().filter(|a| !a.is_empty()).or_else(|| {
        let w = clean_actor(form_field(&fields, "who"));
        (!w.is_empty()).then_some(w)
    });
    let (actor, first_claim) = match who {
        Some(w) => (w, req.actor.is_none()),
        None => {
            return post_reject(
                req,
                db_path,
                n,
                &fields,
                "a name is required to record the act",
            );
        }
    };
    let set_actor = first_claim.then(|| actor.clone());
    let mut conn = match db::open(db_path) {
        Ok(c) => c,
        Err(e) => return page(500, &format!("database: {e}")),
    };
    let context = Context {
        actor,
        tier: Tier::Human,
    };
    match db::execute(&mut conn, &context, command, db::now_epoch()) {
        Ok(stored) => {
            let fragment = if anchor {
                stored
                    .first()
                    .map(|s| format!("#c-{}", s.seq))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            redirect(&format!("/t/{n}{fragment}"), set_actor.as_deref())
        }
        Err(ExecuteFail::Reject(r)) => post_reject(req, db_path, n, &fields, &reject_text(&r)),
        Err(ExecuteFail::Degraded(reason)) => {
            page(503, &format!("world projection unavailable: {reason}"))
        }
        Err(ExecuteFail::Db(e)) => page(500, &format!("database: {e}")),
    }
}

/// Re-render the object page with the reason inline and the drafts preserved.
fn post_reject(
    req: &Req,
    db_path: &std::path::Path,
    n: usize,
    fields: &[(String, String)],
    msg: &str,
) -> tiny_http::Response<Cursor<Vec<u8>>> {
    let Ok(conn) = db::open_read(db_path) else {
        return page(500, "database unavailable");
    };
    let Ok(loadout) = db::load(&conn) else {
        return page(500, "database unavailable");
    };
    let LoadState::Full(world) = loadout.state else {
        return page(503, "world projection unavailable");
    };
    let Some(view) = task_view(&world, TaskId(n)) else {
        return page(404, &format!("no task t-{n}"));
    };
    let form = FormState {
        error: Some(msg.to_string()),
        draft: form_field(fields, "body").to_string(),
        note: form_field(fields, "note").to_string(),
        reply: form_field(fields, "reply").parse().ok(),
        need_who: req.actor.is_none(),
    };
    html(400, &render_object(&view, &world, &form))
}

fn redirect(location: &str, set_actor: Option<&str>) -> tiny_http::Response<Cursor<Vec<u8>>> {
    let mut r = tiny_http::Response::from_string(String::new()).with_status_code(303);
    r = r.with_header(header("Location", location));
    if let Some(name) = set_actor {
        r = r.with_header(header(
            "Set-Cookie",
            &format!("actor={name}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax"),
        ));
    }
    r
}

fn header(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
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
        InvalidParentTaskId => "no such parent task".into(),
        other => format!("{other:?}"),
    }
}

/// The task an open proposal targets, for routing judgment acts back to the dock.
fn proposal_task(db_path: &std::path::Path, seq: usize) -> Option<usize> {
    let conn = db::open_read(db_path).ok()?;
    let loadout = db::load(&conn).ok()?;
    let world = match loadout.state {
        LoadState::Full(w) => w,
        LoadState::Degraded(_) => return None,
    };
    let view = proposal_view(&world, ProposalId(RecordId(seq)))?;
    if view.state != "open" {
        return None;
    }
    task_num(&view.task)
}

#[derive(Default)]
struct FormState {
    error: Option<String>,
    draft: String,
    note: String,
    reply: Option<usize>,
    need_who: bool,
}

enum Route {
    Canvas(Option<usize>),
    Stream,
    Task(usize, Option<usize>),
    NotFound,
}

enum PostRoute {
    Comment(usize),
    Accept(usize),
    Reject(usize),
    NotFound,
}

fn parse_route(url: &str) -> Route {
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (url, None),
    };
    match path {
        "/" => Route::Canvas(
            query
                .and_then(|q| q.strip_prefix("t="))
                .and_then(|v| v.parse().ok()),
        ),
        "/stream" => Route::Stream,
        _ => match path
            .strip_prefix("/t/")
            .or_else(|| path.strip_prefix("/t-"))
            .and_then(|rest| rest.parse().ok())
        {
            Some(n) => Route::Task(
                n,
                query
                    .and_then(|q| q.strip_prefix("reply="))
                    .and_then(|v| v.parse().ok()),
            ),
            None => Route::NotFound,
        },
    }
}

fn parse_post(url: &str) -> PostRoute {
    let path = url.split('?').next().unwrap_or("");
    let num = |s: &str| s.parse::<usize>().ok();
    if let Some(n) = path
        .strip_prefix("/t/")
        .and_then(|rest| rest.strip_suffix("/comment"))
        .and_then(num)
    {
        return PostRoute::Comment(n);
    }
    if let Some(seq) = path
        .strip_prefix("/p/")
        .and_then(|rest| rest.strip_suffix("/accept"))
        .and_then(num)
    {
        return PostRoute::Accept(seq);
    }
    if let Some(seq) = path
        .strip_prefix("/p/")
        .and_then(|rest| rest.strip_suffix("/reject"))
        .and_then(num)
    {
        return PostRoute::Reject(seq);
    }
    PostRoute::NotFound
}

fn html(status: u16, body: &str) -> tiny_http::Response<Cursor<Vec<u8>>> {
    response(status, body)
}

fn page(status: u16, msg: &str) -> tiny_http::Response<Cursor<Vec<u8>>> {
    response(
        status,
        &format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade</title><style>{STYLE}</style></head>\n<body>\n<div class=\"err\">{} · <a href=\"/\">canvas</a></div>\n</body></html>\n",
            esc(msg)
        ),
    )
}

fn response(status: u16, body: &str) -> tiny_http::Response<Cursor<Vec<u8>>> {
    let content_type =
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
            .unwrap();
    let no_store = tiny_http::Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]).unwrap();
    tiny_http::Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(content_type)
        .with_header(no_store)
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn trunc(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

fn task_num(id: &str) -> Option<usize> {
    id.strip_prefix("t-").and_then(|n| n.parse().ok())
}

/// The canvas panels in render order — render emits, never derives.
struct Canvas {
    ready: Vec<TaskView>,
    inflight: Vec<TaskView>,
    history: Vec<TaskView>,
    gate: Vec<ProposalView>,
}

/// View construction for the canvas, all of it: panels partitioned, history
/// ordered most-recently-closed first: for a closed task the last state
/// change is the closing act, so `last_updated` is the panel's key.
fn canvas(world: &World) -> Canvas {
    let mut ready: Vec<TaskView> = Vec::new();
    let mut inflight: Vec<TaskView> = Vec::new();
    let mut history: Vec<(RecordId, TaskView)> = Vec::new();
    for (i, ctx) in world.tasks.iter().enumerate() {
        let view = TaskView::of(
            TaskId(i),
            ctx,
            ctx.proposal.and_then(|p| world.proposals.get(&p)),
        );
        match view.state {
            "open" => ready.push(view),
            "claimed" => inflight.push(view),
            _ => history.push((ctx.last_updated, view)),
        }
    }
    history.sort_by_key(|(seq, _)| std::cmp::Reverse(*seq));
    let gate: Vec<ProposalView> = world
        .proposals
        .keys()
        .filter_map(|id| proposal_view(world, *id))
        .filter(|v| v.state == "open")
        .collect();
    Canvas {
        ready,
        inflight,
        history: history.into_iter().map(|(_, v)| v).collect(),
        gate,
    }
}

fn task_rows(rows: &[TaskView]) -> String {
    rows.iter()
        .map(|v| {
            let mark = if v.n_comments > 0 { "#" } else { "" };
            let href = task_num(&v.id).map(|n| format!("/?t={n}")).unwrap_or_default();
            format!(
                "<a class=\"row\" href=\"{href}\"><span class=\"id\">{}</span><span class=\"state\">{}</span><span class=\"name\">{}</span><span class=\"mark\">{mark}</span></a>\n",
                esc(&v.id),
                esc(v.state),
                esc(&v.name),
            )
        })
        .collect()
}

fn render_canvas(world: &World, c: &Canvas, dock_id: Option<usize>, form: &FormState) -> String {
    let gate_rows: String = c
        .gate
        .iter()
        .map(|p| {
            let href = task_num(&p.task).map(|n| format!("/t/{n}")).unwrap_or_default();
            format!(
                "<a class=\"row gate\" href=\"{href}\"><span class=\"id\">#{}</span><span class=\"verb\">{} {}</span><span class=\"name\">{}</span></a>\n",
                p.id,
                esc(p.action),
                esc(&p.task),
                esc(&p.name),
            )
        })
        .collect();

    let docked = dock_id.and_then(|n| task_view(world, TaskId(n)));
    let main_class = if docked.is_some() {
        "main"
    } else {
        "main nodock"
    };
    // details state dies at every navigation; a docked history entry re-opens the panel
    let hist_open = match &docked {
        Some(v) if v.state == "done" || v.state == "dropped" => " open",
        _ => "",
    };
    let dock_html = docked.map(|view| {
        format!(
            "<div id=\"dock\">\n{}\n</div>\n",
            dock_content(&view, world, form)
        )
    });

    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · canvas</title><style>{STYLE}</style></head>\n<body>\n<header><span>SACCADE · CANVAS</span><a href=\"/stream\">stream →</a></header>\n<div class=\"{main_class}\">\n<div id=\"panels\">\n<h2>Tasks</h2>\n<h2 class=\"sub\">Ready</h2>\n{}\n<h2 class=\"sub\">In-flight</h2>\n{}\n<details class=\"hist\"{hist_open}><summary><h2>History</h2></summary>\n{}\n</details>\n<h2>Gate queue · judgment</h2>\n{gate_rows}\n</div>\n{}\n</div>\n<div id=\"hint\">canvas · stream holds the history · dock summoned per object</div>\n</body></html>\n",
        task_rows(&c.ready),
        task_rows(&c.inflight),
        task_rows(&c.history),
        dock_html.unwrap_or_default(),
    )
}

fn thread_html(thread: &[CommentLine], n: usize) -> String {
    thread
        .iter()
        .map(|c| {
            format!(
                "<div class=\"cmt\" id=\"c-{}\" style=\"margin-left:{}px\"><span class=\"meta\">#{}</span> <span class=\"meta\">{}</span>\n<div class=\"body\">{}</div>\n<a class=\"reply\" href=\"/t/{n}?reply={}\">reply</a></div>\n",
                c.seq,
                c.depth * 18,
                c.seq,
                esc(&c.actor),
                esc(&c.body),
                c.seq,
            )
        })
        .collect()
}

/// The dock's body: identity, judgment blocks, receipt, thread, and the comment form.
fn dock_content(view: &TaskView, world: &World, form: &FormState) -> String {
    let n = task_num(&view.id).unwrap_or(0);
    let mut s = format!(
        "<div class=\"dockhead\">DOCK · {}</div>\n<div class=\"dockstate\">{}</div>\n<div class=\"docktitle\">{}</div>\n",
        esc(&view.id),
        esc(view.state),
        esc(&view.name)
    );
    for pid in world.proposals.keys() {
        let pv = proposal_view(world, *pid).expect("ids come from the map itself");
        if pv.state != "open" || task_num(&pv.task) != Some(n) {
            continue;
        }
        s.push_str(&format!(
            "<div class=\"pblock\">\n<div class=\"pmeta\">#{}</div>\n<div class=\"pname\">{} {} · {}</div>\n<form method=\"post\" action=\"/p/{}/accept\"><button class=\"judge\" type=\"submit\">accept</button></form>\n<form class=\"cform\" method=\"post\" action=\"/p/{}/reject\">\n<textarea name=\"note\" rows=\"2\" placeholder=\"ruling note\">{}</textarea>\n<button class=\"judge\" type=\"submit\">reject</button>\n</form>\n</div>\n",
            pv.id,
            esc(pv.action),
            esc(&pv.task),
            esc(&pv.name),
            pv.id,
            pv.id,
            esc(&form.note),
        ));
    }
    let show = show_view(world, TaskId(n)).expect("dock id validated upstream");
    if let Some(receipt) = &show.receipt {
        s.push_str(&format!(
            "<div class=\"cmt\"><span class=\"meta\">receipt</span>\n<div class=\"body\">{}</div>\n</div>\n",
            esc(receipt)
        ));
    }
    let thread = thread_view(world, TaskId(n)).expect("dock id validated upstream");
    s.push_str(&format!(
        "<div class=\"thread\">{}</div>\n",
        thread_html(&thread, n)
    ));
    if let Some(e) = &form.error {
        s.push_str(&format!("<div class=\"formerr\">{}</div>\n", esc(e)));
    }
    let reply_banner = form
        .reply
        .map(|seq| {
            format!(
                "<div class=\"rbanner\">→ replying to #{seq} · <a href=\"/t/{n}\">clear</a></div>\n"
            )
        })
        .unwrap_or_default();
    let reply_hidden = form
        .reply
        .map(|seq| format!("<input type=\"hidden\" name=\"reply\" value=\"{seq}\">\n"))
        .unwrap_or_default();
    let who = if form.need_who {
        "<input name=\"who\" placeholder=\"your name\" autocomplete=\"name\">\n".to_string()
    } else {
        String::new()
    };
    s.push_str(&format!(
        "{reply_banner}<form class=\"cform\" method=\"post\" action=\"/t/{n}/comment\">\n{reply_hidden}<textarea name=\"body\" rows=\"3\" placeholder=\"…\">{}</textarea>\n{who}<button type=\"submit\">comment</button>\n</form>\n",
        esc(&form.draft),
    ));
    s
}

fn render_object(view: &TaskView, world: &World, form: &FormState) -> String {
    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · {}</title><style>{STYLE}</style></head>\n<body>\n<header><span>SACCADE · {}</span><a href=\"/\">← canvas</a></header>\n<div class=\"obj\">\n{}\n</div>\n</body></html>\n",
        esc(&view.id),
        esc(&view.id),
        dock_content(view, world, form),
    )
}

fn render_stream(rows: &[db::StoredRecord]) -> String {
    let mut body = String::new();
    for r in rows.iter().rev() {
        let rule = r.kind.contains("proposal") || r.kind.contains("drop");
        let kind_class = if rule { "kind rule" } else { "kind" };
        body.push_str(&format!(
            "<div class=\"srow\"><span class=\"seq\">#{}</span><span class=\"{kind_class}\">{}</span><span class=\"actor\">{}</span><span class=\"payload\">{}</span></div>\n",
            r.seq,
            esc(&r.kind),
            esc(&r.actor),
            esc(trunc(&r.payload, 90)),
        ));
    }
    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · stream</title><style>{STYLE}</style></head>\n<body>\n<header><span>SACCADE · STREAM</span><a href=\"/\">← canvas</a></header>\n<div class=\"stream\">\n{body}\n</div>\n</body></html>\n"
    )
}

const STYLE: &str = r#"
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; background: #16161a; color: #c8c8cE;
  font: 14px/1.45 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
header {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 10px 18px; border-bottom: 1px solid #2a2a30;
  letter-spacing: 0.2em; font-size: 12px; color: #7f7f88;
}
header a { color: #7f7f88; text-decoration: none; letter-spacing: normal; }
header a:hover { color: #d19a66; }
main, .main {
  display: grid; grid-template-columns: 1fr 360px; gap: 0;
  max-width: 1100px; margin: 0 auto; min-height: calc(100vh - 42px);
}
.main.nodock { grid-template-columns: 1fr; }
#panels { border-right: 1px solid #2a2a30; padding: 14px 18px 40px; }
#dock { padding: 14px 18px 40px; background: #1a1a1f; }
h2 {
  font-size: 11px; letter-spacing: 0.25em; color: #6a6a72;
  text-transform: uppercase; margin: 22px 0 8px; font-weight: 500;
}
h2:first-child { margin-top: 4px; }
a.row { color: inherit; text-decoration: none; }
.row { display: flex; gap: 10px; padding: 3px 6px; border-radius: 3px; }
.row:hover { background: #202027; }
.id { color: #6a6a72; min-width: 42px; flex: none; }
.state { color: #55555e; min-width: 58px; flex: none; font-size: 12px; padding-top: 1px; }
.name { color: #d5d5dc; min-width: 0; overflow-wrap: anywhere; }
.mark { color: #6a6a72; flex: none; }
.gate { border-left: 2px solid #d19a66; margin: 2px 0; }
.gate .verb { color: #d19a66; }
.dockhead { font-size: 12px; color: #6a6a72; letter-spacing: 0.15em; }
.docktitle { color: #e6e6ec; margin: 4px 0 14px; overflow-wrap: anywhere; }
.dockstate { color: #d19a66; font-size: 12px; letter-spacing: 0.1em; }
.thread { margin-top: 14px; border-top: 1px solid #2a2a30; padding-top: 10px; }
.cmt { margin: 8px 0; }
.cmt .meta { color: #6a6a72; font-size: 12px; }
.cmt .body { color: #b8b8c0; white-space: pre-wrap; overflow-wrap: anywhere; }
.obj { max-width: 760px; margin: 0 auto; padding: 14px 18px 40px; }
.err { max-width: 760px; margin: 40px auto; padding: 0 18px; color: #8f8f98; }
.err a { color: #d19a66; text-decoration: none; }
#hint {
  padding: 6px 18px; border-top: 1px solid #2a2a30; color: #55555e;
  font-size: 12px; letter-spacing: 0.1em;
}
details.hist summary { cursor: pointer; }
details.hist summary h2 { display: inline; }
details.hist summary::marker { color: #55555e; }
details.hist[open] summary h2::after { content: " –"; color: #55555e; }
h2.sub { margin-top: 14px; font-size: 10px; color: #55555e; }
.stream { max-width: 1100px; margin: 0 auto; padding: 12px 18px 40px; }
.srow { display: flex; gap: 12px; padding: 1px 0; font-size: 13px; }
.srow .seq { color: #55555e; min-width: 36px; flex: none; text-align: right; }
.srow .kind { min-width: 130px; flex: none; color: #8f8f98; }
.srow .kind.rule { color: #d19a66; }
.srow .actor { min-width: 90px; flex: none; color: #6a6a72; }
.srow .payload { color: #55555e; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cform { margin: 10px 0 0; }
.cform textarea, .pblock textarea {
  width: 100%; background: #16161a; color: #c8c8ce; border: 1px solid #2a2a30;
  border-radius: 3px; padding: 6px 8px; font: inherit; resize: vertical;
}
.cform input {
  background: #16161a; color: #c8c8ce; border: 1px solid #2a2a30; border-radius: 3px;
  padding: 4px 6px; font: inherit; margin-top: 4px;
}
.cform button, .pblock button {
  background: #202027; color: #c8c8ce; border: 1px solid #2a2a30; border-radius: 3px;
  padding: 4px 12px; font: inherit; margin-top: 6px; cursor: pointer;
}
.pblock { border-left: 2px solid #d19a66; padding: 8px 10px; margin: 10px 0; }
.pblock button.judge { border-color: #d19a66; color: #d19a66; }
.pmeta { color: #6a6a72; font-size: 12px; }
.pname { color: #d19a66; margin: 2px 0 6px; overflow-wrap: anywhere; }
a.reply { color: #55555e; font-size: 12px; text-decoration: none; }
a.reply:hover { color: #d19a66; }
.rbanner { color: #8f8f98; font-size: 12px; margin-top: 10px; }
.rbanner a { color: #6a6a72; text-decoration: none; }
.formerr { color: #8f8f98; border-left: 2px solid #8f8f98; padding: 4px 8px; margin-top: 10px; }
@media (max-width: 720px) {
  main, .main { grid-template-columns: 1fr; }
  #panels { border-right: none; border-bottom: 1px solid #2a2a30; }
  #dock { background: transparent; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use saccade::objects::proposal::ProposalAction;
    use saccade::{Context, Event, Prose, Record, RecordId, Tier, World};

    fn world_with(name: &str) -> World {
        World::replay(vec![Record {
            id: RecordId(0),
            timestamp: 0,
            context: Context {
                actor: "jerry".into(),
                tier: Tier::Human,
            },
            event: Event::TaskCreated {
                id: TaskId(0),
                name: Prose::new(name.into()).unwrap(),
                parent_id: None,
            },
        }])
    }

    #[test]
    fn routes_parse() {
        assert!(matches!(parse_route("/"), Route::Canvas(None)));
        assert!(matches!(parse_route("/?t=17"), Route::Canvas(Some(17))));
        assert!(matches!(parse_route("/?t=x"), Route::Canvas(None)));
        assert!(matches!(parse_route("/stream"), Route::Stream));
        assert!(matches!(parse_route("/t/17"), Route::Task(17, None)));
        assert!(matches!(
            parse_route("/t/17?reply=4"),
            Route::Task(17, Some(4))
        ));
        assert!(matches!(parse_route("/t-17"), Route::Task(17, None)));
        assert!(matches!(parse_route("/nope"), Route::NotFound));
        assert!(matches!(parse_route("/t/x"), Route::NotFound));
    }

    #[test]
    fn post_routes_and_forms_parse() {
        assert!(matches!(parse_post("/t/3/comment"), PostRoute::Comment(3)));
        assert!(matches!(parse_post("/p/9/accept"), PostRoute::Accept(9)));
        assert!(matches!(parse_post("/p/9/reject"), PostRoute::Reject(9)));
        assert!(matches!(parse_post("/t/3"), PostRoute::NotFound));
        let fields = parse_form("body=hello+world%3C1%3E&reply=12&who=jerry");
        assert_eq!(form_field(&fields, "body"), "hello world<1>");
        assert_eq!(form_field(&fields, "reply"), "12");
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
    fn canvas_escapes_names_and_links_rows() {
        let world = world_with("implement <foo>");
        let html = render_canvas(&world, &canvas(&world), None, &FormState::default());
        assert!(html.contains("implement &lt;foo&gt;"));
        assert!(!html.contains("implement <foo>"));
        assert!(html.contains("href=\"/?t=0\""));
        assert!(!html.contains("id=\"dock\""));
    }

    #[test]
    fn unknown_dock_leaves_canvas_whole() {
        let world = world_with("implement foo");
        let html = render_canvas(&world, &canvas(&world), Some(9), &FormState::default());
        assert!(!html.contains("id=\"dock\""));
    }

    #[test]
    fn judgment_block_and_comment_form_render() {
        let world = World::replay(vec![
            record(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("implement foo".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                Event::ProposalCreated {
                    name: Prose::new("stale by supersession".into()).unwrap(),
                    action: ProposalAction::Drop { task_id: TaskId(0) },
                },
            ),
        ]);
        let view = task_view(&world, TaskId(0)).unwrap();
        let html = render_object(&view, &world, &FormState::default());
        assert!(html.contains("action=\"/p/1/accept\""));
        assert!(html.contains("action=\"/p/1/reject\""));
        assert!(html.contains("action=\"/t/0/comment\""));
    }

    /// Only a done task's dock carries the receipt
    #[test]
    fn dock_gates_the_receipt_on_done() {
        let done = World::replay(vec![
            record(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("implement foo".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(1, Event::TaskClaimed { id: TaskId(0) }),
            record(
                2,
                Event::TaskDone {
                    id: TaskId(0),
                    receipt: Prose::new("suite green <34 unit>".into()).unwrap(),
                },
            ),
        ]);
        let html = render_canvas(&done, &canvas(&done), Some(0), &FormState::default());
        assert!(html.contains("suite green &lt;34 unit&gt;"));
        assert!(!html.contains("<34 unit>"));

        let open = world_with("implement foo");
        let html = render_canvas(&open, &canvas(&open), Some(0), &FormState::default());
        assert!(!html.contains("receipt"));
    }

    #[test]
    fn reply_retargets_the_form() {
        let world = World::replay(vec![
            record(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("implement foo".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                Event::Commented {
                    target: Target::Task(TaskId(0)),
                    body: Prose::new("receipt lands here".into()).unwrap(),
                },
            ),
        ]);
        let view = task_view(&world, TaskId(0)).unwrap();
        let html = render_object(
            &view,
            &world,
            &FormState {
                reply: Some(1),
                ..Default::default()
            },
        );
        assert!(html.contains("id=\"c-1\""));
        assert!(html.contains("replying to #1"));
        assert!(html.contains("name=\"reply\" value=\"1\""));
        assert!(html.contains("?reply=1\""));
    }

    /// t-0 closed last, so it leads history despite its lower id.
    #[test]
    fn history_reads_most_recently_closed_first() {
        let events = vec![
            record(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("migrate floop".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                Event::TaskCreated {
                    id: TaskId(1),
                    name: Prose::new("implement foo".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(2, Event::TaskClaimed { id: TaskId(1) }),
            record(
                3,
                Event::TaskDone {
                    id: TaskId(1),
                    receipt: Prose::new("suite green".into()).unwrap(),
                },
            ),
            record(4, Event::TaskClaimed { id: TaskId(0) }),
            record(
                5,
                Event::TaskDone {
                    id: TaskId(0),
                    receipt: Prose::new("smoke clean".into()).unwrap(),
                },
            ),
        ];
        let world = World::replay(events.clone());
        let html = render_canvas(&world, &canvas(&world), None, &FormState::default());
        let hist = html.split("<details class=\"hist\">").nth(1).unwrap();
        assert!(hist.find("t-0").unwrap() < hist.find("t-1").unwrap());
    }

    /// The history panel renders open when the dock holds a closed task, so clicking
    /// between history entries never collapses it.
    #[test]
    fn history_panel_opens_when_the_dock_is_history() {
        let events = vec![
            record(
                0,
                Event::TaskCreated {
                    id: TaskId(0),
                    name: Prose::new("implement foo".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                Event::TaskCreated {
                    id: TaskId(1),
                    name: Prose::new("migrate floop".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(2, Event::TaskClaimed { id: TaskId(1) }),
            record(
                3,
                Event::TaskDone {
                    id: TaskId(1),
                    receipt: Prose::new("suite green".into()).unwrap(),
                },
            ),
        ];
        let world = World::replay(events.clone());
        let c = canvas(&world);
        let docked_history = render_canvas(&world, &c, Some(1), &FormState::default());
        assert!(docked_history.contains("<details class=\"hist\" open>"));
        let docked_open = render_canvas(&world, &c, Some(0), &FormState::default());
        assert!(docked_open.contains("<details class=\"hist\">"));
    }

    #[test]
    fn object_page_carries_the_back_link() {
        let world = world_with("implement foo");
        let view = task_view(&world, TaskId(0)).unwrap();
        let html = render_object(&view, &world, &FormState::default());
        assert!(html.contains("DOCK · t-0"));
        assert!(html.contains("href=\"/\""));
    }

    fn record(seq: usize, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: 0,
            context: Context {
                actor: "jerry".into(),
                tier: Tier::Human,
            },
            event,
        }
    }
}
