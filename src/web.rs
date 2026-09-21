//! The console: one full-height instrument — supervision strip, forest
//! rail, focused thread, docked compose. Every section is a renderer
//! over view types; fragments are sections, so the compose fetch swaps
//! what the page itself renders. The @-compiler is the composer's input
//! syntax over Commented events — the parse lives here, never in the
//! record.

use crate::views::{
    CommentLine, ForestRow, MarkKind, NextPanel, ProposalView, RIBBON_WINDOW_SECS, RibbonMark,
    ShowView, ThreadItem, ThreadView,
};
use crate::{Addressee, CommentId, RecordId, Target, TaskId};
use time::OffsetDateTime;
use time::macros::format_description;

/// The task state's chip color, one per state out of the #522 palette:
/// open whispers, claimed is gold like the work it holds, done is green
/// like a settled run, dropped is rose like the human act it was.
fn state_color(state: &str) -> &'static str {
    match state {
        "claimed" => "#dac09a",
        "done" => "#a2c4a3",
        "dropped" => "#c4a6a8",
        _ => "#6d6562",
    }
}

/// Ribbon layout: percent kept clear at each edge; a chip's footprint
/// in percent — a chip landing on a covered row drops a row down, the
/// only deviation from time-truth the bar allows.
const PAD_PCT: f64 = 4.0;
const CHIP_GAP_PCT: f64 = 3.0;
const RIBBON_ROWS: usize = 3;
const ROW_HEIGHT_PX: f64 = 13.0;

pub struct Console {
    pub forest: Vec<ForestRow>,
    pub closed: Vec<ForestRow>,
    pub gate: Vec<ProposalView>,
    pub next: NextPanel,
    /// The supervision strip's movement marks: every task's window.
    pub marks: Vec<RibbonMark>,
    /// The focused task's panel facts, when one is focused.
    pub focus: Option<Focus>,
    pub form: FormState,
    /// The read's now, in epoch seconds; the ages and the ribbon window
    /// hang off it.
    pub now: u64,
}

pub struct Focus {
    pub show: ShowView,
    /// Open proposals targeting the focused task, in birth order.
    pub proposals: Vec<ProposalView>,
    pub thread: ThreadView,
}

#[derive(Default)]
pub struct FormState {
    pub error: Option<String>,
    pub draft: String,
    pub note: String,
    /// The actor name prefilled into every act form — the cookie's
    /// value when one exists, blank for a fresh browser.
    pub who: String,
}

pub struct Address {
    pub body: String,
    pub addressee: Option<Addressee>,
    pub target: Target,
}

enum Tok {
    Agent,
    Target(Target),
}

/// The composer's grammar: `@agent` makes the comment a demand, `@c-N`
/// parents it to that comment, `@t-N` rehomes it to that task; bare
/// ids and `@human` stay prose. A token must stand alone (the char
/// before `@` is not alphanumeric, the char after is neither
/// alphanumeric nor `-`). The first target token wins; later target
/// tokens stay prose. Consumed tokens leave the body.
pub(crate) fn compile(body: &str, fallback: Target) -> Address {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut addressee = None;
    let mut target = fallback;
    let mut target_taken = false;
    let mut i = 0;
    while i < chars.len() {
        let boundary = chars[i] == '@' && (i == 0 || !chars[i - 1].is_alphanumeric());
        let token = if boundary {
            token_at(&chars, i + 1)
        } else {
            None
        };
        match token {
            Some((Tok::Agent, end)) => {
                addressee = Some(Addressee::Agent);
                i = skip_space(&chars, end, &out);
            }
            Some((Tok::Target(t), end)) if !target_taken => {
                target = t;
                target_taken = true;
                i = skip_space(&chars, end, &out);
            }
            // a second target token is prose, @ and all
            Some((Tok::Target(_), end)) => {
                out.extend(&chars[i..end]);
                i = end;
            }
            None => {
                out.push(chars[i]);
                i += 1;
            }
        }
    }
    Address {
        body: out.trim().to_string(),
        addressee,
        target,
    }
}

/// One following space goes with a consumed token when it would double up.
fn skip_space(chars: &[char], end: usize, out: &str) -> usize {
    end + usize::from(
        end < chars.len()
            && (chars[end] == ' ' || chars[end] == '\t')
            && (out.is_empty() || out.ends_with([' ', '\t', '\n'])),
    )
}

fn token_at(chars: &[char], p: usize) -> Option<(Tok, usize)> {
    let ends = |len: usize| {
        let q = p + len;
        q >= chars.len() || !(chars[q].is_alphanumeric() || chars[q] == '-')
    };
    let word: String = chars[p..]
        .iter()
        .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || **c == '-')
        .collect();
    if word == "agent" && ends(word.len()) {
        return Some((Tok::Agent, p + word.len()));
    }
    for (prefix, target) in [
        (
            "c-",
            (|n: usize| Target::Comment(CommentId(RecordId(n)))) as fn(usize) -> Target,
        ),
        (
            "t-",
            (|n: usize| Target::Task(TaskId(n))) as fn(usize) -> Target,
        ),
    ] {
        let Some(digits) = word.strip_prefix(prefix) else {
            continue;
        };
        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if let Ok(n) = digits.parse()
            && ends(word.len())
        {
            return Some((Tok::Target(target(n)), p + word.len()));
        }
    }
    None
}

// ---- local time labels: ids, timestamps, and labels are mono facts ----

fn local(ts: u64) -> OffsetDateTime {
    let dt = OffsetDateTime::from_unix_timestamp(ts as i64).unwrap_or(OffsetDateTime::UNIX_EPOCH);
    dt.to_offset(time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC))
}

/// 13:07
fn fmt_t(ts: u64) -> String {
    let desc = format_description!("[hour repr:24]:[minute]");
    local(ts).format(&desc).unwrap_or_default()
}

/// Sun Sep 20
fn fmt_d(ts: u64) -> String {
    let desc = format_description!("[weekday repr:short] [month repr:short] [day]");
    local(ts).format(&desc).unwrap_or_default()
}

/// Sun Sep 20 18:02 — the strip's axis
fn fmt_dt(ts: u64) -> String {
    let desc = format_description!(
        "[weekday repr:short] [month repr:short] [day] [hour repr:24]:[minute]"
    );
    local(ts).format(&desc).unwrap_or_default()
}

fn ago(secs: u64) -> String {
    match secs {
        0..=59 => "now".into(),
        60..=3599 => format!("{}m", secs / 60),
        3600..=86399 => format!("{}h", secs / 3600),
        _ => format!("{}d", secs / 86400),
    }
}

// ---- the page ----

pub fn page(c: &Console) -> String {
    let strip = strip_section(c);
    let forest = forest_section(&c.forest, &c.closed, &c.gate, focus_num(c).as_deref());
    let panel = match &c.focus {
        Some(f) => format!(
            "<div class=\"thead\"><span class=\"tstate\" style=\"color:{state_color}\">● {state}</span>\n<div class=\"ttitle\">{title}</div>\n<div class=\"tmeta mono\">{meta}</div>\n</div>\n{thread}{compose}",
            state_color = state_color(f.show.state),
            state = f.show.state.to_uppercase(),
            title = esc(&f.show.name),
            meta = esc(&head_meta(&f.show)),
            thread = thread_section(f, &c.form, None),
            compose = compose_section(task_num(&f.show.id).unwrap_or(0), &c.form),
        ),
        None => "<div class=\"thead\"></div>\n<section id=\"thread\"><div class=\"nempty\">no task focused</div></section>\n".to_string(),
    };
    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · console</title><style>{STYLE}</style></head>\n<body>\n<header><span class=\"brand\">SACCADE · CONSOLE</span><span><label class=\"fbtn\" for=\"nav\">tasks ▸</label></span></header>\n<input type=\"checkbox\" id=\"nav\">\n<div class=\"if\">\n{strip}\n<div class=\"cols withforest\">\n{forest}\n<div class=\"tpanel\">\n{panel}\n</div>\n</div>\n</div>\n<script>{JS}</script>\n</body></html>\n"
    )
}

fn focus_num(c: &Console) -> Option<String> {
    c.focus.as_ref().map(|f| f.show.id.clone())
}

fn head_meta(show: &ShowView) -> String {
    match &show.parent {
        Some(p) => format!("{} · under {}", show.id, p),
        None => show.id.clone(),
    }
}

fn task_num(id: &str) -> Option<usize> {
    id.strip_prefix("t-").and_then(|n| n.parse().ok())
}

// ---- the supervision strip ----

fn strip_section(c: &Console) -> String {
    let mut s = String::from("<div class=\"strip\">\n");
    s.push_str(&format!(
        "<div class=\"striphead\"><span class=\"strip-title\">SUPERVISION</span><span class=\"strip-axis mono\">{}</span></div>\n",
        esc(&fmt_dt(c.now)),
    ));
    s.push_str("<div class=\"nsec\">RUNS IN FLIGHT</div>\n");
    if c.next.runs.is_empty() {
        s.push_str("<div class=\"nempty\">no runs in flight</div>\n");
    }
    for r in &c.next.runs {
        s.push_str(&format!(
            "<a class=\"nxrow\" href=\"/t/{}#c-{}\"><span class=\"nid mono\">i-{}</span><span class=\"nname\">t-{} · {}</span><span class=\"nfact mono\">demand c-{} · born {} ago</span></a>\n",
            r.task, r.demand, r.incarnation, r.task, esc(&r.actor), r.demand, ago(c.now.saturating_sub(r.born_at)),
        ));
    }
    // asked-of-you is silent when the world is silent: no section, no
    // empty fact — its rows appear only while an answer is awaited
    if !c.next.asked_of_you.is_empty() {
        s.push_str("<div class=\"nsec\">ASKED OF YOU</div>\n");
        for a in &c.next.asked_of_you {
            s.push_str(&format!(
                "<a class=\"nxrow ask\" href=\"/t/{}#c-{}\"><span class=\"nid mono\">c-{}</span><span class=\"nname\">{}</span><span class=\"nfact mono\">{} · t-{}</span></a>\n",
                a.task, a.comment, a.comment, esc(&a.body), esc(&a.actor), a.task,
            ));
        }
    }
    s.push_str("<div class=\"nsec\">CLAIMED · LAST RECORD</div>\n");
    if c.next.candidates.is_empty() {
        s.push_str("<div class=\"nempty\">nothing claimed</div>\n");
    }
    for cand in &c.next.candidates {
        let tint = if cand.adrift { " stale" } else { "" };
        s.push_str(&format!(
            "<a class=\"nxrow\" href=\"/t/{}\"><span class=\"nid mono\">{}</span><span class=\"nname\">{}</span><span class=\"nfact mono{tint}\">claim {} · last record {}</span></a>\n",
            task_num(&cand.task).unwrap_or(0),
            esc(&cand.task),
            esc(&cand.name),
            ago(cand.claim_age),
            ago(cand.last_record_age),
        ));
    }
    s.push_str(&ribbon_section(c));
    s.push_str("</div>\n");
    s
}

fn ribbon_section(c: &Console) -> String {
    let mut s = String::from(
        "<div class=\"nsec ribhead\">MOVEMENT · 72H · <span class=\"k-human\">your notes</span> · <span class=\"k-agent\">agent notes</span> · <span class=\"k-run\">runs</span> · <span class=\"k-demand\">demands</span></div>\n",
    );
    if c.marks.is_empty() {
        s.push_str("<div class=\"rib\"></div>\n");
        return s;
    }
    let start = c.now.saturating_sub(RIBBON_WINDOW_SECS);
    let span = RIBBON_WINDOW_SECS.max(1);
    // positions derive from event_time across the inset window. The
    // only deviation from time-truth is vertical: a chip that would
    // cover a newer one on its row drops a row down, so crowded marks
    // stay individually visible without ever leaving the bar.
    let mut last = [-f64::INFINITY; RIBBON_ROWS];
    let mut placed: Vec<(&RibbonMark, f64, usize)> = Vec::with_capacity(c.marks.len());
    for m in &c.marks {
        let raw =
            PAD_PCT + (m.at.saturating_sub(start)) as f64 / span as f64 * (100.0 - 2.0 * PAD_PCT);
        // the first row with room; when every row is covered at this
        // position, cycle rows so identical positions stack evenly
        let row = (0..RIBBON_ROWS)
            .find(|r| raw - last[*r] >= CHIP_GAP_PCT)
            .unwrap_or(placed.len() % RIBBON_ROWS);
        last[row] = raw;
        placed.push((m, raw, row));
    }
    s.push_str("<div class=\"rib\">");
    // 24h gridlines and the now marker
    for hours in [24, 48] {
        let pct = PAD_PCT
            + (RIBBON_WINDOW_SECS - hours * 3600) as f64 / span as f64 * (100.0 - 2.0 * PAD_PCT);
        s.push_str(&format!(
            "<span class=\"rv\" style=\"left:{pct:.1}%\"></span>\n"
        ));
    }
    s.push_str(&format!(
        "<span class=\"rv now\" style=\"left:{:.1}%\"></span>\n",
        100.0 - PAD_PCT
    ));
    for (m, left, row) in placed {
        let (class, label) = match m.kind {
            MarkKind::Note { human: true } => ("human", "note"),
            MarkKind::Note { human: false } => ("agent", "note"),
            MarkKind::Demand => ("demand", "demand"),
            MarkKind::Run => ("run", "run"),
        };
        let top = row as f64 * ROW_HEIGHT_PX + 1.0;
        s.push_str(&format!(
            "<a class=\"mrk {class}\" style=\"left:{left:.1}%;top:{top:.0}px\" href=\"/t/{}#c-{}\" title=\"t-{} · {label} #{} · {}\"></a>\n",
            m.task, m.seq, m.task, m.seq, esc(&fmt_t(m.at)),
        ));
    }
    s.push_str("</div>\n");
    s
}

// ---- the forest rail ----

fn forest_section(
    rows: &[ForestRow],
    closed: &[ForestRow],
    gate: &[ProposalView],
    focus: Option<&str>,
) -> String {
    let mut s = String::from("<aside class=\"forest\">\n");
    for row in rows {
        let sel = focus == Some(row.task.id.as_str());
        let pad = row.depth * 14;
        s.push_str(&format!(
            "<a class=\"frow{}\" style=\"margin-left:{pad}px\" href=\"/t/{}\"><span class=\"fid mono\">{}</span><span class=\"schip\" style=\"color:{}\">{}</span><span class=\"fname\">{}</span></a>\n",
            if sel { " sel" } else { "" },
            task_num(&row.task.id).unwrap_or(0),
            esc(&row.task.id),
            state_color(row.task.state),
            esc(row.task.state.to_uppercase().as_str()),
            esc(&row.task.name),
        ));
    }
    if rows.is_empty() {
        s.push_str("<div class=\"fgate-empty\">no live tasks</div>\n");
    }
    s.push_str("<div class=\"fsect\">GATE · JUDGMENT</div>\n");
    for p in gate {
        s.push_str(&format!(
            "<a class=\"frow\" href=\"/t/{}\"><span class=\"fid mono\">#{}</span><span class=\"fname\">{} {} · {}</span></a>\n",
            task_num(&p.task).unwrap_or(0),
            p.id,
            esc(p.action),
            esc(&p.task),
            esc(&p.name),
        ));
    }
    if gate.is_empty() {
        s.push_str("<div class=\"fgate-empty\">nothing awaiting judgment</div>\n");
    }
    if !closed.is_empty() {
        s.push_str("<div class=\"fsect\">CLOSED</div>\n");
        for row in closed {
            let pad = row.depth * 14;
            s.push_str(&format!(
                "<a class=\"frow closed\" style=\"margin-left:{pad}px\" href=\"/t/{}\"><span class=\"fid mono\">{}</span><span class=\"schip\" style=\"color:{}\">{}</span><span class=\"fname\">{}</span></a>\n",
                task_num(&row.task.id).unwrap_or(0),
                esc(&row.task.id),
                state_color(row.task.state),
                esc(row.task.state.to_uppercase().as_str()),
                esc(&row.task.name),
            ));
        }
    }
    s.push_str("</aside>\n");
    s
}

// ---- the thread panel ----

/// `landed` names the comment a just-accepted compose created, so the
/// swap can meet the reader's eyes with it.
pub fn thread_section(f: &Focus, form: &FormState, landed: Option<usize>) -> String {
    let mut s = match landed {
        Some(seq) => format!("<section id=\"thread\" data-focus=\"c-{seq}\">\n"),
        None => String::from("<section id=\"thread\">\n"),
    };
    for p in &f.proposals {
        s.push_str(&format!(
            "<div class=\"judge\"><span class=\"jhead mono\">#{}</span> <span class=\"jname\">{} {} · {}</span>\n<form class=\"jform\" method=\"post\" action=\"/p/{}/ruling\">\n<textarea name=\"note\" rows=\"2\" placeholder=\"ruling note\">{}</textarea>\n<div class=\"jbtns\">{}<button class=\"sendbtn\" name=\"ruling\" value=\"accept\" type=\"submit\">accept</button>\n<button class=\"sendbtn\" name=\"ruling\" value=\"reject\" type=\"submit\">reject</button></div>\n</form>\n</div>\n",
            p.id,
            esc(p.action),
            esc(&p.task),
            esc(&p.name),
            p.id,
            esc(&form.note),
            who_input(&form.who),
        ));
    }
    if let Some(receipt) = &f.show.receipt {
        s.push_str(&format!(
            "<div class=\"receipt\"><span class=\"xk\" style=\"color:#dac09a\">RECEIPT</span>\n<div class=\"nbody\">{}</div>\n</div>\n",
            esc(receipt),
        ));
    }
    let mut last_time: Option<u64> = None;
    for item in &f.thread.items {
        let (first_at, last_at) = item_span(item);
        if let Some(prev) = last_time
            && first_at.saturating_sub(prev) > 4 * 3600
        {
            s.push_str(&format!(
                "<div class=\"gapdiv mono\">{}</div>\n",
                esc(&fmt_d(first_at)),
            ));
        }
        last_time = Some(last_at);
        s.push_str(&item_html(item));
    }
    if f.thread.items.is_empty() && f.show.receipt.is_none() && f.proposals.is_empty() {
        s.push_str("<div class=\"nempty\">no comments yet</div>\n");
    }
    s.push_str("</section>\n");
    s
}

fn item_span(item: &ThreadItem) -> (u64, u64) {
    match item {
        ThreadItem::Note(line) => (line.born_at, line.born_at),
        ThreadItem::Group { root, replies } | ThreadItem::Exchange { root, replies, .. } => {
            let last = replies.last().map(|r| r.born_at).unwrap_or(root.born_at);
            (root.born_at, last)
        }
    }
}

fn item_html(item: &ThreadItem) -> String {
    match item {
        ThreadItem::Exchange { root, run, replies } => match run {
            Some(run) => {
                let open = run.in_flight();
                let window = match run.done_at {
                    Some(done) => format!("{}–{}", fmt_t(run.born_at), fmt_t(done)),
                    None => format!("{}– in flight", fmt_t(run.born_at)),
                };
                let mut s = format!(
                    "<div class=\"xg{}\">\n<div class=\"xhead\"><span class=\"xk\">EXCHANGE</span><span class=\"xd mono\">#{}</span><span class=\"xmeta mono\">{} · {}</span></div>\n<div class=\"xbody\">\n",
                    if open { " open" } else { "" },
                    root.seq,
                    esc(&window),
                    if open { "in flight" } else { "settled" },
                );
                s.push_str(&node_html(root, 0, Some(("DEMAND", "#dac09a")), None));
                s.push_str(&format!(
                    "<div class=\"nrow2 runrow\"><div class=\"nmeta\"><span class=\"xk\" style=\"color:#a2c4a3\">RUN</span><span class=\"nwho\">{}</span><span class=\"nseq mono\">{}</span></div>\n</div>\n",
                    esc(&run.actor),
                    esc(&window),
                ));
                for r in replies {
                    s.push_str(&node_html(
                        r,
                        r.depth.saturating_sub(2),
                        Some(("REPLY", "#a2c4a3")),
                        None,
                    ));
                }
                s.push_str("</div>\n</div>\n");
                s
            }
            None => {
                // the demand awaits its run: a plain row, not a card
                let tag = if root.refusal.is_some() {
                    "refused"
                } else {
                    "awaiting incarnation"
                };
                let mut s = node_html(root, 0, Some(("DEMAND", "#dac09a")), Some(tag));
                // the refusal fact, where the run row would sit: reason and time
                if let Some(r) = &root.refusal {
                    s.push_str(&format!(
                        "<div class=\"nrow2 runrow\"><div class=\"nmeta\"><span class=\"xk\" style=\"color:#c4a6a8\">REFUSED</span><span class=\"nseq mono\">{}</span></div>\n<div class=\"nbody\">{}</div>\n</div>\n",
                        esc(&fmt_t(r.at)),
                        esc(&r.reason),
                    ));
                }
                for r in replies {
                    s.push_str(&node_html(
                        r,
                        r.depth.saturating_sub(1),
                        Some(("REPLY", "#a2c4a3")),
                        None,
                    ));
                }
                s
            }
        },
        ThreadItem::Group { root, replies } => {
            let mut s = String::from("<div class=\"ntg\">\n");
            s.push_str(&node_html(root, 0, None, None));
            for r in replies {
                s.push_str(&node_html(r, r.depth.saturating_sub(2), None, None));
            }
            s.push_str("</div>\n");
            s
        }
        ThreadItem::Note(line) => node_html(line, 0, None, None),
    }
}

/// One comment row: kind chip, whisper mono meta (actor, seq, time),
/// full body.
fn node_html(
    line: &CommentLine,
    indent: usize,
    kind: Option<(&str, &str)>,
    tag: Option<&str>,
) -> String {
    let chip = kind
        .map(|(k, color)| format!("<span class=\"xk\" style=\"color:{color}\">{k}</span>"))
        .unwrap_or_default();
    let state = line
        .state
        .as_deref()
        .filter(|_| kind.is_none())
        .map(|s| format!("<span class=\"nseq\">{}</span>", esc(s)))
        .unwrap_or_default();
    let extra = tag
        .map(|t| format!("<span class=\"nseq\">{}</span>", esc(t)))
        .unwrap_or_default();
    format!(
        "<div class=\"nrow2\" id=\"c-{seq}\" style=\"padding-left:{pad}px\">\n<div class=\"nmeta\">{chip}<span class=\"nwho {tier}\">{actor}</span><span class=\"nseq mono\">#{seq}</span>{state}{extra}<span class=\"nseq mono\">{time}</span></div>\n<div class=\"nbody\">{body}</div>\n</div>\n",
        seq = line.seq,
        pad = 26 + indent * 22,
        tier = esc(&line.tier),
        actor = esc(&line.actor),
        time = esc(&fmt_t(line.born_at)),
        body = esc(&line.body),
    )
}

pub fn compose_section(task: usize, form: &FormState) -> String {
    let error = form
        .error
        .as_deref()
        .map(|e| format!("<div class=\"formerr\">{}</div>\n", esc(e)))
        .unwrap_or_default();
    format!(
        "<section id=\"compose\">\n<form id=\"cform\" data-task=\"{task}\" method=\"post\" action=\"/compose\">\n<input type=\"hidden\" name=\"task\" value=\"{task}\">\n<div id=\"address\" class=\"addrline mono\"></div>\n<textarea id=\"body\" name=\"body\" rows=\"2\" placeholder=\"write\">{}</textarea>\n{error}<div class=\"sendrow\">{who}<button class=\"sendbtn\" type=\"submit\">send</button></div>\n</form>\n</section>\n",
        esc(&form.draft),
        who = who_input(&form.who),
    )
}

/// Every act form carries the actor name: prefilled from the cookie
/// when one exists, blank for a fresh browser — the no-JS path types
/// its name and posts.
fn who_input(prefill: &str) -> String {
    format!(
        "<input class=\"who\" name=\"who\" placeholder=\"your name\" autocomplete=\"name\" value=\"{}\">\n",
        esc(prefill)
    )
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const JS: &str = r#"
const re = /(^|[^A-Za-z0-9])@(agent|c-[0-9]+|t-[0-9]+)(?![A-Za-z0-9-])/g;
function resolve(text, home) {
  if (!text.trim()) return '';
  let agent = false, target = null;
  for (const m of text.matchAll(re)) {
    const tok = m[2];
    if (tok === 'agent') agent = true;
    else if (target === null) target = tok;
  }
  const parts = [];
  if (agent) parts.push('demand \u2192 agent');
  if (target) parts.push((target[0] === 'c' ? 'reply \u2192 ' : 'rehome \u2192 ') + target);
  else parts.push('home \u2192 t-' + home);
  return parts.join(' \u00b7 ');
}
document.addEventListener('input', (e) => {
  if (e.target.id !== 'body') return;
  const f = e.target.closest('form');
  document.getElementById('address').textContent =
    resolve(e.target.value, f ? f.dataset.task : '');
});
document.addEventListener('submit', (e) => {
  const f = e.target;
  if (f.id !== 'cform') return;
  e.preventDefault();
  fetch('/compose', { method: 'POST', body: new URLSearchParams(new FormData(f)) })
    .then((r) => {
      // a rehome answers with the 303 to the comment's new home:
      // the section would land on a stale page, so navigate instead
      if (r.redirected) { location.assign(r.url); return null; }
      return r.text().then((html) => ({ r, html }));
    })
    .then((swap) => {
      if (!swap) return;
      const doc = new DOMParser().parseFromString(swap.html, 'text/html');
      const sec = doc.body.firstElementChild;
      const cur = sec && document.getElementById(sec.id);
      if (cur) cur.replaceWith(sec);
      if (swap.r.ok) {
        f.elements.body.value = '';
        document.getElementById('address').textContent = '';
        const landed = sec && sec.dataset.focus;
        if (landed) document.getElementById(landed)
          ?.scrollIntoView({ block: 'center' });
      } else {
        document.querySelector('.formerr')
          ?.scrollIntoView({ block: 'center' });
      }
    });
});
"#;

pub(crate) const STYLE: &str = r#"
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
#nav { position: absolute; opacity: 0; }
body {
  background: #100f0e; color: #d4ceca;
  font: 500 13.5px/1.55 "Noto Sans", system-ui, sans-serif;
}
.mono, .nseq, .nfact, .nid, .schip, .xd, .xmeta, .tmeta, .addrline, .gapdiv, .strip-axis, .strip-title, .nsec, .fsect, .jhead, .xk, .tstate {
  font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
a { color: #a2c3c4; text-decoration: none; }

/* mode line */
header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 16px; background: #1b1918;
}
header .brand {
  font: 600 11px "JetBrains Mono", ui-monospace, monospace;
  letter-spacing: .2em; color: #e8e2dd;
}
.fbtn {
  display: none; cursor: pointer; color: #6d6562;
  font: 500 11px "JetBrains Mono", ui-monospace, monospace;
  border: 1px solid #2e2a28; padding: 2px 9px; border-radius: 3px;
}
.fbtn:hover { color: #e8e2dd; }

/* full-height frame: strip over (rail | thread) */
.if { display: grid; grid-template-rows: auto 1fr; height: calc(100vh - 41px); }

/* supervision strip: one surface band, dense rows */
.strip { background: #1b1918; padding: 10px 16px 12px; }
.striphead { display: flex; gap: 14px; align-items: baseline; }
.strip-title { font: 600 10.5px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .2em; color: #6d6562; }
.strip-axis { margin-left: auto; font-size: 10px; color: #4a4543; }
.nsec { font: 600 9.5px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .16em; color: #6d6562; margin: 8px 0 2px; }
.nxrow {
  display: grid; grid-template-columns: 52px minmax(180px, 300px) 1fr; gap: 12px;
  padding: 3px 6px; border-radius: 3px; align-items: baseline; color: inherit;
}
.nxrow:hover { background: #292624; }
.nid { color: #dac09a; font-size: 11.5px; white-space: nowrap; }
.nname { font-size: 12.5px; color: #d4ceca; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nfact { font-size: 11px; color: #6d6562; text-align: right; }
.nfact.stale { color: #dac09a; }
.nempty { font-size: 11px; color: #4a4543; padding: 1px 6px 2px; }

/* movement ribbon */
.ribhead { margin-top: 12px; }
.k-human { color: #c4a6a8; }
.k-agent { color: #6d6562; }
.k-run { color: #a2c4a3; }
.k-demand { color: #dac09a; }
.rib {
  position: relative; height: 40px; background: #100f0e;
  border-radius: 2px; margin-top: 4px;
}
.rv { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1px solid #292624; }
.rv.now { border-left: 2px solid #d4ceca; }
.mrk { position: absolute; width: 2px; height: 11px; }
.mrk.human { background: #c4a6a8; }
.mrk.agent { background: #6d6562; }
.mrk.demand { background: #dac09a; }
.mrk.run { background: #a2c4a3; }
.mrk:hover { width: 3px; outline: 1px solid #e8e2dd; z-index: 2; }

/* columns: forest surface | thread deep — tone separates, no dividers */
.cols { display: grid; min-height: 0; }
.cols.withforest { grid-template-columns: 252px 1fr; }
.forest { background: #1b1918; overflow-y: auto; padding: 10px 8px; }
.frow {
  display: flex; flex-wrap: nowrap; gap: 7px; align-items: center;
  padding: 4px 8px; border-radius: 3px; color: inherit;
}
.frow:hover { background: #292624; }
.frow.sel { background: #292624; }
.frow.sel .fname { color: #e8e2dd; }
.fid { color: #dac09a; font-size: 11.5px; white-space: nowrap; flex-shrink: 0; }
.schip { white-space: nowrap; flex-shrink: 0; font: 500 9.5px/1.6 "JetBrains Mono", ui-monospace, monospace; letter-spacing: .06em; }
.fname { flex: 1; min-width: 0; font-size: 12.5px; color: #d4ceca; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fsect { font: 600 9.5px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .16em; color: #6d6562; margin: 16px 8px 4px; }
.fgate-empty { padding: 4px 8px; font-size: 11.5px; color: #4a4543; }
.frow.closed .fname { color: #6d6562; }
.frow.closed:hover .fname { color: #b3aca6; }

/* thread panel: head / scroll / compose */
.tpanel { display: grid; grid-template-rows: auto 1fr auto; min-height: 0; min-width: 0; }
.thead { padding: 14px 18px 10px; }
.ttitle { font: 800 16px/1.3 "Noto Sans", system-ui, sans-serif; color: #e8e2dd; margin: 4px 0 3px; overflow-wrap: anywhere; }
.tmeta { font-size: 11.5px; color: #6d6562; }
.tstate { font: 600 10.5px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .12em; }

#thread { overflow-y: auto; min-height: 0; padding: 4px 0 12px; }

/* day dividers at gaps */
.gapdiv {
  display: flex; align-items: center; gap: 10px; margin: 12px 18px 6px;
  color: #4a4543; font-size: 10px; letter-spacing: .14em;
}
.gapdiv::before, .gapdiv::after { content: ""; flex: 1; border-top: 1px solid #1e1c1a; }

/* nodes: whisper mono meta, full body */
.nrow2 { padding: 5px 18px 5px 26px; position: relative; }
.nrow2:hover { background: #1b1918; }
.nmeta { display: flex; gap: 8px; align-items: baseline; font-size: 11px; color: #4a4543; margin-bottom: 2px; flex-wrap: wrap; }
.nwho { color: #6d6562; }
.nwho.human { color: #c4a6a8; }
.nwho.agent { color: #6d6562; }
.nseq { font-size: 11px; color: #4a4543; }
.nbody { font-size: 13.5px; line-height: 1.55; color: #d4ceca; white-space: pre-wrap; overflow-wrap: anywhere; }

/* conversations: raised tone cards, borderless */
.xg { margin: 4px 18px 6px 16px; background: #1e1c1a; border-radius: 6px; }
.xg.open { background: rgba(218,192,154,.05); }
.xhead { display: flex; gap: 10px; align-items: center; padding: 7px 12px; font-size: 11px; }
.xk { color: #a2c4a3; font-size: 10px; letter-spacing: .14em; }
.xg.open .xhead .xk { color: #dac09a; }
.xd { color: #6d6562; }
.xmeta { color: #4a4543; font-size: 10.5px; margin-left: auto; }
.xbody { border-top: 1px solid #2e2a28; padding: 4px 0; }
.xg .nrow2 { padding-left: 14px; padding-right: 14px; }
.xg .nrow2.runrow { padding-top: 0; padding-bottom: 2px; }
.ntg { margin: 4px 18px 6px 16px; background: #1e1c1a; border-radius: 6px; padding: 3px 0; }
.ntg .nrow2 { padding-left: 14px; padding-right: 14px; }

/* receipt and judgment: quiet gold facts */
.receipt { margin: 4px 18px 6px 16px; background: #1e1c1a; border-radius: 6px; padding: 7px 12px; }
.receipt .nbody { color: #b3aca6; margin-top: 2px; }
.judge { margin: 4px 18px 8px 16px; background: #1e1c1a; border-radius: 6px; padding: 8px 12px; }
.jhead { color: #6d6562; font-size: 11px; }
.jname { color: #dac09a; overflow-wrap: anywhere; }
.jrow {
  display: flex; gap: 14px; align-items: flex-end; margin-top: 6px; flex-wrap: wrap;
}
.jrow form { margin: 0; display: flex; gap: 8px; align-items: flex-end; }
.jform { flex: 1; min-width: 220px; }
.jform textarea { width: 100%; }
.jreject { display: flex; gap: 8px; align-items: flex-end; }
input.who {
  background: #100f0e; color: #d4ceca; border: 1px solid #100f0e; border-radius: 4px;
  font: 500 11px "JetBrains Mono", ui-monospace, monospace;
  padding: 5px 8px; width: 110px;
}
input.who:focus { outline: none; border-color: #2e2a28; caret-color: #c4a6a8; }

/* compose: docked surface band */
#compose { background: #1b1918; padding: 10px 18px 12px; }
.addrline { font-size: 11px; color: #6d6562; margin-bottom: 6px; min-height: 14px; }
#compose textarea {
  width: 100%; background: #100f0e; color: #d4ceca;
  border: 1px solid #100f0e; border-radius: 4px;
  font: 500 13.5px "Noto Sans", system-ui, sans-serif; padding: 9px 11px; resize: vertical;
}
#compose textarea:focus { outline: none; border-color: #2e2a28; caret-color: #c4a6a8; }
.sendrow { display: flex; justify-content: flex-end; gap: 10px; align-items: center; margin-top: 6px; }
.sendrow .who {
  width: auto; font: 500 12.5px "Noto Sans", system-ui, sans-serif; padding: 6px 10px;
}
.sendrow .who:focus { outline: none; border-color: #2e2a28; }
.sendbtn {
  background: #292624; color: #d4ceca; border: none; border-radius: 3px;
  padding: 6px 18px; cursor: pointer; font: 600 12px "Noto Sans", sans-serif;
}
.sendbtn:hover { color: #dac09a; }
.jform textarea {
  width: 100%; background: #100f0e; color: #d4ceca;
  border: 1px solid #100f0e; border-radius: 4px;
  font: 500 12.5px "Noto Sans", system-ui, sans-serif; padding: 6px 9px; resize: vertical;
}
.formerr {
  color: #c4a6a8; background: rgba(196,166,168,.08);
  border-left: 2px solid #c4a6a8; border-radius: 0 3px 3px 0;
  padding: 7px 10px; margin-top: 8px;
  font: 500 12px "JetBrains Mono", ui-monospace, monospace;
  overflow-wrap: anywhere;
}

/* narrow: the rail collapses behind the mode-line toggle */
@media (max-width: 1100px) {
  .fbtn { display: inline-block; }
  .cols.withforest { grid-template-columns: 1fr; }
  .forest {
    position: fixed; left: 0; top: 41px; bottom: 0; width: 280px; z-index: 40;
    transform: translateX(-102%); transition: transform .15s ease;
    box-shadow: 8px 0 24px rgba(0,0,0,.6);
  }
  #nav:checked ~ .if .forest { transform: none; }
}
"#;

#[cfg(test)]
mod tests {

    use super::*;
    use crate::Prose;
    use crate::events::Event;
    use crate::objects::comment::Target;
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::proposal::ProposalAction;
    use crate::store::{Context, Record, Tier, World};
    use crate::types::actor::ActorName;
    use crate::types::pointers::SessionPointer;
    use crate::views::{
        closed_tasks, forest, next_panel, open_proposals, ribbon_marks, show_view, thread_view,
    };
    use std::sync::OnceLock;

    fn human() -> &'static Context {
        static H: OnceLock<Context> = OnceLock::new();
        H.get_or_init(|| Context {
            actor: ActorName::new("jerry".into()).unwrap(),
            tier: Tier::Human,
        })
    }

    fn agent() -> &'static Context {
        static A: OnceLock<Context> = OnceLock::new();
        A.get_or_init(|| Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
        })
    }

    fn record(seq: usize, at: u64, ctx: &Context, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: at,
            context: ctx.clone(),
            event,
        }
    }

    fn task(n: usize) -> Target {
        Target::Task(TaskId(n))
    }

    // ---- the @-compiler's equivalence classes ----

    #[test]
    fn bare_body_passes_through() {
        let a = compile("look at t-1 and c-2", task(7));
        assert_eq!(a.body, "look at t-1 and c-2");
        assert_eq!(a.addressee, None);
        assert_eq!(a.target, task(7));
    }

    #[test]
    fn agent_makes_the_demand() {
        let a = compile("@agent build the thing", task(7));
        assert_eq!(a.body, "build the thing");
        assert_eq!(a.addressee, Some(Addressee::Agent));
        assert_eq!(a.target, task(7));
    }

    #[test]
    fn trailing_agent_leaves_the_words() {
        let a = compile("do it @agent", task(7));
        assert_eq!(a.body, "do it");
        assert_eq!(a.addressee, Some(Addressee::Agent));
    }

    #[test]
    fn comment_token_parents() {
        let a = compile("saw it @c-19", task(7));
        assert_eq!(a.body, "saw it");
        assert_eq!(a.addressee, None);
        assert_eq!(a.target, Target::Comment(CommentId(RecordId(19))));
    }

    #[test]
    fn task_token_rehomes() {
        let a = compile("@t-3 belongs there", task(7));
        assert_eq!(a.body, "belongs there");
        assert_eq!(a.target, task(3));
    }

    #[test]
    fn demand_and_rehome_combine() {
        let a = compile("@agent @t-3 fix it there", task(7));
        assert_eq!(a.body, "fix it there");
        assert_eq!(a.addressee, Some(Addressee::Agent));
        assert_eq!(a.target, task(3));
    }

    #[test]
    fn first_target_token_wins_and_the_rest_is_prose() {
        let a = compile("@t-1 then @t-2", task(7));
        assert_eq!(a.body, "then @t-2");
        assert_eq!(a.target, task(1));
    }

    #[test]
    fn at_human_is_not_console_grammar() {
        let a = compile("ask @human to rule", task(7));
        assert_eq!(a.body, "ask @human to rule");
        assert_eq!(a.addressee, None);
    }

    #[test]
    fn tokens_must_stand_alone() {
        let a = compile("mail me@agent now", task(7));
        assert_eq!(a.body, "mail me@agent now");
        assert_eq!(a.addressee, None);
        let a = compile("see x@t-1", task(7));
        assert_eq!(a.body, "see x@t-1");
        assert_eq!(a.target, task(7));
    }

    #[test]
    fn partial_tokens_stay_prose() {
        for body in ["@agentx", "@c-1x", "@t-", "@c- 5", "@agent-3", "@c-x"] {
            let a = compile(body, task(7));
            assert_eq!(a.body, body, "{body}");
            assert_eq!(a.addressee, None, "{body}");
            assert_eq!(a.target, task(7), "{body}");
        }
    }

    #[test]
    fn whitespace_is_tidy_after_a_consumed_token() {
        assert_eq!(compile("  @agent   spaced  ", task(7)).body, "spaced");
        assert_eq!(compile("@agent\ndo it", task(7)).body, "do it");
    }

    // ---- the frame ----

    const NOW: u64 = 3_210 * 3600;

    /// t-0 done with receipt; t-1 open under t-0 with a settled exchange
    /// (demand c-5, run i-6, reply c-6) and an orphan note c-7;
    /// t-2 claimed and long adrift.
    fn fixture() -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("ship it".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(1, 10, human(), Event::TaskClaimed { id: TaskId(0) }),
            record(
                2,
                20,
                human(),
                Event::TaskDone {
                    id: TaskId(0),
                    receipt: Prose::new("suite green <34>".into()).unwrap(),
                },
            ),
            record(
                3,
                30,
                human(),
                Event::TaskCreated {
                    name: Prose::new("build <foo>".into()).unwrap(),
                    parent_id: Some(TaskId(0)),
                },
            ),
            record(
                4,
                40,
                human(),
                Event::ProposalCreated {
                    name: Prose::new("stale by supersession".into()).unwrap(),
                    action: ProposalAction::Drop { task_id: TaskId(1) },
                },
            ),
            record(
                5,
                3_140 * 3600,
                human(),
                Event::Commented {
                    target: task(1),
                    body: Prose::new("demand body".into()).unwrap(),
                    addressee: Some(Addressee::Agent),
                },
            ),
            record(
                6,
                3_150 * 3600,
                &Context {
                    actor: ActorName::new("system".into()).unwrap(),
                    tier: Tier::System,
                },
                Event::IncarnationBound {
                    task_id: TaskId(1),
                    response_target: CommentId(RecordId(5)),
                    trigger: RecordId(5),
                    actor: ActorName::new("pi".into()).unwrap(),
                    session: SessionPointer::new("/tmp/s".into()).unwrap(),
                },
            ),
            record(
                7,
                3_155 * 3600,
                &Context {
                    actor: ActorName::new("system".into()).unwrap(),
                    tier: Tier::System,
                },
                Event::IncarnationPromptAccepted {
                    id: IncarnationId(RecordId(6)),
                },
            ),
            record(
                8,
                3_160 * 3600,
                agent(),
                Event::Commented {
                    target: Target::Comment(CommentId(RecordId(5))),
                    body: Prose::new("reply body".into()).unwrap(),
                    addressee: None,
                },
            ),
            record(
                9,
                3_170 * 3600,
                &Context {
                    actor: ActorName::new("system".into()).unwrap(),
                    tier: Tier::System,
                },
                Event::IncarnationSettled {
                    id: IncarnationId(RecordId(6)),
                },
            ),
            record(
                10,
                3_200 * 3600,
                human(),
                Event::Commented {
                    target: task(1),
                    body: Prose::new("lonely note".into()).unwrap(),
                    addressee: None,
                },
            ),
            record(
                11,
                40,
                human(),
                Event::TaskCreated {
                    name: Prose::new("adrift work".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                12,
                100 * 3600,
                human(),
                Event::TaskClaimed { id: TaskId(2) },
            ),
        ])
        .unwrap()
    }

    fn focus_of(world: &World, n: usize) -> Focus {
        Focus {
            show: show_view(world, TaskId(n)).unwrap(),
            proposals: open_proposals(world)
                .into_iter()
                .filter(|v| v.task == format!("t-{n}"))
                .collect(),
            thread: thread_view(world, TaskId(n)).unwrap(),
        }
    }

    fn console_of(world: &World, focus: Option<Focus>) -> Console {
        Console {
            forest: forest(world),
            closed: closed_tasks(world),
            gate: open_proposals(world),
            next: next_panel(world, NOW),
            marks: ribbon_marks(world, NOW),
            focus,
            form: FormState::default(),
            now: NOW,
        }
    }

    #[test]
    fn the_receipt_renders_only_when_done() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 0), &Default::default(), None);
        assert!(html.contains("suite green &lt;34&gt;"));
        let open = thread_section(&focus_of(&world, 1), &Default::default(), None);
        assert!(!open.contains("receipt"));
    }

    #[test]
    fn judgment_forms_render_on_the_focused_task() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1), &Default::default(), None);
        // one form, one route, two rulings — and the themed class on it
        assert!(html.contains("<form class=\"jform\" method=\"post\" action=\"/p/4/ruling\">"));
        assert!(html.contains("name=\"note\""));
        assert!(html.contains("name=\"ruling\" value=\"accept\""));
        assert!(html.contains("name=\"ruling\" value=\"reject\""));
    }

    #[test]
    fn bodies_and_names_escape() {
        let world = fixture();
        let page = page(&console_of(&world, Some(focus_of(&world, 1))));
        assert!(page.contains("build &lt;foo&gt;"));
        assert!(!page.contains("build <foo>"));
    }

    #[test]
    fn the_exchange_is_a_settled_card() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1), &Default::default(), None);
        assert!(html.contains("EXCHANGE"));
        assert!(html.contains("#5"));
        assert!(html.contains("settled"), "the run's window closed");
        assert!(html.contains("DEMAND"));
        assert!(html.contains("REPLY"));
        assert!(html.contains(">RUN<"));
        assert!(html.contains("demand body"));
        assert!(html.contains("reply body"));
        // the reply lives inside the settled card, the orphan after it
        assert!(html.contains("<div class=\"xg\">"));
        let open_card = html.find("EXCHANGE").unwrap();
        let reply = html.find("id=\"c-8\"").unwrap();
        let orphan = html.find("id=\"c-10\"").unwrap();
        assert!(open_card < reply, "the reply nests in the card");
        assert!(reply < orphan, "the orphan note follows the card");
    }

    #[test]
    fn the_head_names_state_title_and_lineage() {
        let world = fixture();
        let page = page(&console_of(&world, Some(focus_of(&world, 1))));
        assert!(page.contains("● OPEN"));
        assert!(page.contains("build &lt;foo&gt;"));
        assert!(page.contains("t-1 · under t-0"));
        assert!(
            page.contains("class=\"frow sel\""),
            "the forest marks focus"
        );
    }

    #[test]
    fn the_strip_carries_supervision_rows() {
        let world = fixture();
        let page = page(&console_of(&world, None));
        assert!(page.contains("SUPERVISION"));
        assert!(page.contains("RUNS IN FLIGHT"));
        assert!(page.contains("CLAIMED · LAST RECORD"));
        // the adrift claim tints its ages gold
        assert!(page.contains("adrift work"));
        assert!(page.contains("nfact mono stale"));
        // nothing awaits the human, so the section stays silent
        assert!(!page.contains("ASKED OF YOU"));
    }

    #[test]
    fn a_refused_demand_renders_its_refusal_on_the_thread() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                3_140 * 3600,
                human(),
                Event::Commented {
                    target: task(0),
                    body: Prose::new("run it again".into()).unwrap(),
                    addressee: Some(Addressee::Agent),
                },
            ),
            record(
                2,
                3_150 * 3600,
                &Context {
                    actor: ActorName::new("system".into()).unwrap(),
                    tier: Tier::System,
                },
                Event::DemandRefused {
                    demand: CommentId(RecordId(1)),
                    reason: Prose::new(
                        "t-0 branch saccade/t-0 diverged from the recorded checkpoint".into(),
                    )
                    .unwrap(),
                },
            ),
        ])
        .unwrap();
        let html = thread_section(&focus_of(&world, 0), &Default::default(), None);
        // the demand names its refusal, and the fact carries reason and time
        assert!(html.contains(">refused<"), "{html}");
        assert!(html.contains("REFUSED"));
        assert!(html.contains("diverged from the recorded checkpoint"));
        assert!(!html.contains("awaiting incarnation"));
        assert!(!html.contains("RUN"), "no run ever bound");
    }

    #[test]
    fn asked_of_you_renders_its_items_and_hides_when_empty() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("ship it".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                10,
                agent(),
                Event::Commented {
                    target: task(0),
                    body: Prose::new("need a ruling on the checkpoint rule".into()).unwrap(),
                    addressee: Some(Addressee::Human),
                },
            ),
        ])
        .unwrap();
        let pending = page(&console_of(&world, None));
        assert!(pending.contains("ASKED OF YOU"));
        assert!(pending.contains("need a ruling on the checkpoint rule"));
        assert!(pending.contains("href=\"/t/0#c-1\""));
        assert!(pending.contains("pi"));

        // answered: the section vanishes entirely
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("ship it".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                10,
                agent(),
                Event::Commented {
                    target: task(0),
                    body: Prose::new("need a ruling".into()).unwrap(),
                    addressee: Some(Addressee::Human),
                },
            ),
            record(
                2,
                20,
                human(),
                Event::Commented {
                    target: Target::Comment(CommentId(RecordId(1))),
                    body: Prose::new("ruled".into()).unwrap(),
                    addressee: None,
                },
            ),
        ])
        .unwrap();
        let answered = page(&console_of(&world, None));
        assert!(!answered.contains("ASKED OF YOU"));
    }

    #[test]
    fn closed_tasks_render_in_the_rail_with_state_colors() {
        let world = fixture();
        let html = page(&console_of(&world, Some(focus_of(&world, 1))));
        // the done task browses from the rail, dimmed, chip green
        assert!(html.contains("CLOSED"));
        assert!(html.contains("class=\"frow closed\""));
        assert!(html.contains("style=\"color:#a2c4a3\">DONE<"));
        // the live tree and the head keep the deterministic mapping
        assert!(html.contains("style=\"color:#6d6562\">OPEN<"));
        assert!(html.contains("● OPEN"));
        // a dropped chip is rose
        let dropped = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("void work".into()).unwrap(),
                    parent_id: None,
                },
            ),
            record(
                1,
                10,
                human(),
                Event::TaskDropped {
                    id: TaskId(0),
                    note: Prose::new("void".into()).unwrap(),
                },
            ),
        ])
        .unwrap();
        let dropped_html = page(&console_of(&dropped, None));
        assert!(dropped_html.contains("style=\"color:#c4a6a8\">DROPPED<"));
    }

    /// The chip positions the ribbon rendered, in render order.
    fn mark_chips(ribbon_html: &str) -> Vec<(f64, f64)> {
        ribbon_html
            .split("class=\"mrk ")
            .skip(1)
            .filter_map(|seg| {
                let left: f64 = seg.split("left:").nth(1)?.split('%').next()?.parse().ok()?;
                let top: f64 = seg.split("top:").nth(1)?.split('p').next()?.parse().ok()?;
                Some((left, top))
            })
            .collect()
    }

    #[test]
    fn sparse_marks_land_time_true() {
        let world = fixture();
        let html = ribbon_section(&console_of(&world, None));
        let chips = mark_chips(&html);
        assert_eq!(chips.len(), 4, "demand, run, reply note, orphan note");
        // window [3138h, 3210h]: each chip sits at its event_time's
        // inset position, on the first row — no packing was needed
        let raw = |hours: f64| 4.0 + (hours - 3_138.0) / 72.0 * 92.0;
        let expect = [6.6, 19.3, 32.1, 83.2];
        for (chip, want) in chips.iter().zip(expect) {
            assert!(
                (chip.0 - want).abs() < 0.15,
                "not time-true: {} vs {want}",
                chip.0
            );
            assert_eq!(chip.1, 1.0, "sparse marks share the first row");
        }
        assert!((chips[3].0 - raw(3_200.0)).abs() < 0.15);
    }

    #[test]
    fn a_burst_packs_without_leaving_the_bar() {
        let now = 3_210 * 3600u64;
        let mut records = vec![record(
            0,
            0,
            human(),
            Event::TaskCreated {
                name: Prose::new("ship it".into()).unwrap(),
                parent_id: None,
            },
        )];
        for i in 0..30u64 {
            records.push(record(
                1 + i as usize,
                now - 60 + i,
                human(),
                Event::Commented {
                    target: task(0),
                    body: Prose::new("burst".into()).unwrap(),
                    addressee: None,
                },
            ));
        }
        let world = World::replay(records).unwrap();
        let html = ribbon_section(&console_of(&world, None));
        let chips = mark_chips(&html);
        assert_eq!(chips.len(), 30, "every mark renders");
        for (left, top) in &chips {
            assert!(*left >= PAD_PCT - 0.1, "off the left edge: {left}");
            assert!(*left <= 100.0 - PAD_PCT + 0.1, "off the right edge: {left}");
            assert!(
                *top <= RIBBON_ROWS as f64 * ROW_HEIGHT_PX,
                "off the rows: {top}"
            );
        }
        // the newest chip sits at the now marker's inset position
        let newest = chips
            .iter()
            .map(|(l, _)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (newest - (100.0 - PAD_PCT)).abs() < 0.5,
            "newest off now: {newest}"
        );
        // visibility: the burst engages every row — the bar spreads the
        // instant's cluster RIBBON_ROWS wide, its honest limit
        let tops: std::collections::BTreeSet<u32> =
            chips.iter().map(|(_, t)| (*t * 10.0) as u32).collect();
        assert!(tops.len() >= 2, "a burst must use the rows: {tops:?}");
    }

    #[test]
    fn the_swap_knows_where_the_thread_landed() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1), &Default::default(), Some(8));
        assert!(
            html.contains("data-focus=\"c-8\""),
            "no landed anchor: {}",
            &html[..html.len().min(200)]
        );
        // a page load names nothing
        let plain = thread_section(&focus_of(&world, 1), &Default::default(), None);
        assert!(!plain.contains("data-focus"));
    }

    #[test]
    fn the_compose_error_renders_legible_in_its_dock() {
        let html = compose_section(
            9,
            &FormState {
                error: Some("a comment needs words".into()),
                ..Default::default()
            },
        );
        assert!(html.contains("<div class=\"formerr\">a comment needs words</div>"));
        // the error stands as its own block above the send row, not inside it
        let err = html.find("formerr").unwrap();
        let send = html.find("sendrow").unwrap();
        assert!(err < send, "the error leads the send row");
        assert!(!html[err..send].contains("sendbtn"));
    }

    #[test]
    fn the_compose_dock_holds_the_grammar() {
        let html = compose_section(9, &FormState::default());
        assert!(html.contains("<section id=\"compose\">"));
        assert!(html.contains("name=\"task\" value=\"9\""));
        assert!(html.contains("action=\"/compose\""));
        assert!(html.contains("placeholder=\"write\""));
        assert!(
            html.contains("name=\"who\""),
            "every act form carries the name"
        );
        assert!(html.contains("value=\"\""));
        assert!(html.contains("type=\"submit\">send<"));

        let prefilled = compose_section(
            9,
            &FormState {
                who: "jerry".into(),
                ..Default::default()
            },
        );
        assert!(prefilled.contains("value=\"jerry\""), "the cookie prefills");
    }

    #[test]
    fn the_judgment_form_carries_one_name() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1), &Default::default(), None);
        assert_eq!(
            html.matches("name=\"who\"").count(),
            1,
            "one ruling, one name"
        );
        let prefilled = thread_section(
            &focus_of(&world, 1),
            &FormState {
                who: "jerry".into(),
                ..Default::default()
            },
            None,
        );
        assert_eq!(prefilled.matches("value=\"jerry\"").count(), 1);
    }

    #[test]
    fn an_unfocused_column_states_it() {
        let world = fixture();
        let page = page(&console_of(&world, None));
        assert!(page.contains("no task focused"));
    }
}
