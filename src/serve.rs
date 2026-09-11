//! The read-only serve edge: canvas, dock, and stream over HTTP — §9's second
//! rendering. One request, one read: the world folds per request, and WAL
//! keeps that safe beside CLI writes. No mutation crosses this edge.

use saccade::db::{self, LoadState};
use saccade::objects::task::TaskId;
use saccade::wire::{self, TaskView};
use std::io::Cursor;

pub fn run(
    db_path: &std::path::Path,
    bind: &str,
    port: u16,
) -> Result<std::convert::Infallible, String> {
    let addr = format!("{bind}:{port}");
    let server =
        tiny_http::Server::http(&addr).map_err(|e| format!("cannot bind {addr}: {e}"))?;
    eprintln!("serving http://{addr} · read-only");
    for request in server.incoming_requests() {
        let method = request.method().to_string();
        let url = request.url().to_string();
        let _ = request.respond(respond(&method, &url, db_path));
    }
    unreachable!("the incoming-requests iterator never ends")
}

fn respond(
    method: &str,
    url: &str,
    db_path: &std::path::Path,
) -> tiny_http::Response<Cursor<Vec<u8>>> {
    if method != "GET" {
        return page(405, "GET only — this edge is read-only.");
    }
    let conn = match db::open_read(db_path) {
        Ok(c) => c,
        Err(e) => return page(500, &format!("database: {e}")),
    };
    let loadout = match db::load(&conn) {
        Ok(l) => l,
        Err(e) => return page(500, &format!("database: {e}")),
    };
    match parse_route(url) {
        Route::Stream => html(200, &render_stream(&loadout.rows)),
        Route::Canvas(dock) => match loadout.state {
            LoadState::Full(world) => html(200, &render_canvas(&world, dock)),
            LoadState::Degraded(reason) => {
                page(503, &format!("world projection unavailable: {reason}"))
            }
        },
        Route::Task(n) => match loadout.state {
            LoadState::Full(world) => match wire::view_of(&TaskId(n), &world) {
                Some(view) => html(
                    200,
                    &render_object(&view, &wire::comment_thread(&world, &TaskId(n))),
                ),
                None => page(404, &format!("no task t-{n}")),
            },
            LoadState::Degraded(reason) => {
                page(503, &format!("world projection unavailable: {reason}"))
            }
        },
        Route::NotFound => page(404, "nothing here — try / or /stream"),
    }
}

enum Route {
    Canvas(Option<usize>),
    Stream,
    Task(usize),
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
            Some(n) => Route::Task(n),
            None => Route::NotFound,
        },
    }
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
    let no_store =
        tiny_http::Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]).unwrap();
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

fn task_rows(rows: &[&TaskView]) -> String {
    rows.iter()
        .map(|v| {
            let mark = if v.comments > 0 { "#" } else { "" };
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

fn render_canvas(world: &saccade::World, dock_id: Option<usize>) -> String {
    let mut all: Vec<TaskView> = (0..world.tasks.len())
        .filter_map(|i| wire::view_of(&TaskId(i), world))
        .collect();
    all.sort_by_key(|v| task_num(&v.id).unwrap_or(0));
    let (ready, inflight, history): (Vec<&TaskView>, Vec<&TaskView>, Vec<&TaskView>) = (
        all.iter().filter(|v| v.state == "open").collect(),
        all.iter().filter(|v| v.state == "claimed").collect(),
        all.iter()
            .filter(|v| v.state == "done" || v.state == "dropped")
            .collect(),
    );
    let gate: Vec<wire::ProposalView> = world
        .proposals
        .iter()
        .map(|(id, p)| wire::view_of_proposal(id, p, world))
        .filter(|v| v.state == "open")
        .collect();
    let gate_rows: String = gate
        .iter()
        .map(|p| {
            let href = task_num(&p.task).map(|n| format!("/t/{n}")).unwrap_or_default();
            format!(
                "<a class=\"row gate\" href=\"{href}\"><span class=\"id\">#{}</span><span class=\"verb\">{} {}</span><span class=\"name\">{}</span></a>\n",
                p.id,
                esc(&p.action),
                esc(&p.task),
                esc(&p.name),
            )
        })
        .collect();

    let dock_html = dock_id.and_then(|n| {
        let view = wire::view_of(&TaskId(n), world)?;
        let thread = wire::comment_thread(world, &TaskId(n));
        Some(format!(
            "<div id=\"dock\">\n<div class=\"dockhead\">DOCK · {}</div>\n<div class=\"dockstate\">{}</div>\n<div class=\"docktitle\">{}</div>\n<div class=\"thread\">{}</div>\n</div>\n",
            esc(&view.id),
            esc(view.state),
            esc(&view.name),
            thread_html(&thread),
        ))
    });
    let main_class = if dock_html.is_some() { "main" } else { "main nodock" };

    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · canvas</title><style>{STYLE}</style></head>\n<body>\n<header><span>SACCADE · CANVAS</span><a href=\"/stream\">stream →</a></header>\n<div class=\"{main_class}\">\n<div id=\"panels\">\n<h2>Tasks</h2>\n<h2 class=\"sub\">Ready</h2>\n{}\n<h2 class=\"sub\">In-flight</h2>\n{}\n<details class=\"hist\"><summary><h2>History</h2></summary>\n{}\n</details>\n<h2>Gate queue · judgment</h2>\n{gate_rows}\n</div>\n{}\n</div>\n<div id=\"hint\">canvas · stream holds the history · dock summoned per object</div>\n</body></html>\n",
        task_rows(&ready),
        task_rows(&inflight),
        task_rows(&history),
        dock_html.unwrap_or_default(),
    )
}

fn thread_html(thread: &[wire::CommentLine]) -> String {
    thread
        .iter()
        .map(|c| {
            format!(
                "<div class=\"cmt\" style=\"margin-left:{}px\"><span class=\"meta\">#{}</span> <span class=\"meta\">{}</span>\n<div class=\"body\">{}</div></div>\n",
                c.depth * 18,
                c.seq,
                esc(&c.actor),
                esc(&c.body),
            )
        })
        .collect()
}

fn render_object(view: &TaskView, thread: &[wire::CommentLine]) -> String {
    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · {}</title><style>{STYLE}</style></head>\n<body>\n<header><span>SACCADE · {}</span><a href=\"/\">← canvas</a></header>\n<div class=\"obj\">\n<div class=\"dockhead\">DOCK · {}</div>\n<div class=\"dockstate\">{}</div>\n<div class=\"docktitle\">{}</div>\n<div class=\"thread\">{}</div>\n</div>\n</body></html>\n",
        esc(&view.id),
        esc(&view.id),
        esc(&view.id),
        esc(view.state),
        esc(&view.name),
        thread_html(thread),
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
@media (max-width: 720px) {
  main, .main { grid-template-columns: 1fr; }
  #panels { border-right: none; border-bottom: 1px solid #2a2a30; }
  #dock { background: transparent; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(matches!(parse_route("/t/17"), Route::Task(17)));
        assert!(matches!(parse_route("/t-17"), Route::Task(17)));
        assert!(matches!(parse_route("/nope"), Route::NotFound));
        assert!(matches!(parse_route("/t/x"), Route::NotFound));
    }

    #[test]
    fn canvas_escapes_names_and_links_rows() {
        let html = render_canvas(&world_with("implement <foo>"), None);
        assert!(html.contains("implement &lt;foo&gt;"));
        assert!(!html.contains("implement <foo>"));
        assert!(html.contains("href=\"/?t=0\""));
        assert!(!html.contains("id=\"dock\""));
    }

    #[test]
    fn unknown_dock_leaves_canvas_whole() {
        let html = render_canvas(&world_with("implement foo"), Some(9));
        assert!(!html.contains("id=\"dock\""));
    }

    #[test]
    fn object_page_carries_the_back_link() {
        let world = world_with("implement foo");
        let view = wire::view_of(&TaskId(0), &world).unwrap();
        let html = render_object(&view, &[]);
        assert!(html.contains("DOCK · t-0"));
        assert!(html.contains("href=\"/\""));
    }
}
