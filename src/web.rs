//! The console: the webui's one page. Every section is a renderer over
//! view types; fragments are sections, so the compose fetch swaps what
//! the page itself renders. The @-compiler is the composer's input
//! syntax over Commented events — the parse lives here, never in the
//! record.

use crate::views::{
    CommentLine, ForestRow, MarkKind, NextPanel, ProposalView, RIBBON_WINDOW_SECS, RibbonMark,
    ShowView, ThreadView,
};
use crate::{Addressee, CommentId, RecordId, Target, TaskId};

pub struct Console {
    pub forest: Vec<ForestRow>,
    pub gate: Vec<ProposalView>,
    pub next: NextPanel,
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
    pub marks: Vec<RibbonMark>,
}

#[derive(Default)]
pub struct FormState {
    pub error: Option<String>,
    pub draft: String,
    pub note: String,
    pub need_who: bool,
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

// ---- sections ----

pub fn page(c: &Console) -> String {
    let forest = forest_section(&c.forest, &c.gate);
    let next = next_section(&c.next);
    let (thread, ribbon, compose) = match &c.focus {
        Some(f) => (
            thread_section(f, &c.form),
            ribbon_section(&f.show.id, &f.marks, c.now),
            compose_section(task_num(&f.show.id).unwrap_or(0), &c.form),
        ),
        None => (
            "<section id=\"thread\"><div class=\"fact\">no task focused</div></section>\n"
                .to_string(),
            String::new(),
            String::new(),
        ),
    };
    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>saccade · console</title><style>{STYLE}</style></head>\n<body>\n<input type=\"checkbox\" id=\"nav\">\n<header><span>SACCADE · CONSOLE</span><label class=\"navlabel\" for=\"nav\">forest</label></header>\n<div class=\"console\">\n{forest}\n<div class=\"col\">\n{next}\n{thread}\n{ribbon}\n{compose}\n</div>\n</div>\n<script>{JS}</script>\n</body></html>\n"
    )
}

fn task_num(id: &str) -> Option<usize> {
    id.strip_prefix("t-").and_then(|n| n.parse().ok())
}

fn forest_section(rows: &[ForestRow], gate: &[ProposalView]) -> String {
    let mut s = String::from("<aside id=\"forest\">\n<h2>forest</h2>\n");
    for row in rows {
        let n = task_num(&row.task.id).unwrap_or(0);
        let pad = "· ".repeat(row.depth);
        let st = if row.task.state == "open" {
            ""
        } else {
            " · claimed"
        };
        let mark = if row.task.proposal.is_some() {
            " <span class=\"pmark\">judgment</span>"
        } else {
            ""
        };
        s.push_str(&format!(
            "<a class=\"frow\" href=\"/t/{n}\"><span class=\"tid\">{}</span><span class=\"fname\">{pad}{}{st}{mark}</span></a>\n",
            esc(&row.task.id),
            esc(&row.task.name),
        ));
    }
    if rows.is_empty() {
        s.push_str("<div class=\"fact\">no live tasks</div>\n");
    }
    s.push_str("<h2>gate · judgment</h2>\n");
    for p in gate {
        let n = task_num(&p.task).unwrap_or(0);
        s.push_str(&format!(
            "<a class=\"frow gate\" href=\"/t/{n}\"><span class=\"tid\">#{}</span><span class=\"fname\">{} {} · {}</span></a>\n",
            p.id,
            esc(p.action),
            esc(&p.task),
            esc(&p.name),
        ));
    }
    if gate.is_empty() {
        s.push_str("<div class=\"fact\">gate clear</div>\n");
    }
    s.push_str("</aside>\n");
    s
}

fn next_section(next: &NextPanel) -> String {
    let mut s = String::from("<section id=\"next\">\n<h2>next</h2>\n");
    for r in &next.runs {
        s.push_str(&format!(
            "<a class=\"nrow run\" href=\"/t/{}\"><span class=\"tid\">i-{}</span><span class=\"fname\">t-{} · {}</span></a>\n",
            r.task, r.incarnation, r.task, esc(&r.actor),
        ));
    }
    if next.runs.is_empty() {
        s.push_str("<div class=\"fact\">no runs in flight</div>\n");
    }
    for a in &next.asked_of_you {
        s.push_str(&format!(
            "<a class=\"nrow ask\" href=\"/t/{}#c-{}\"><span class=\"tid\">c-{}</span><span class=\"fname\">t-{} · {}</span></a>\n",
            a.task, a.comment, a.comment, a.task, esc(&a.actor),
        ));
    }
    if next.asked_of_you.is_empty() {
        s.push_str("<div class=\"fact\">nothing asked of you</div>\n");
    }
    for c in &next.candidates {
        let tint = if c.adrift { " adrift" } else { "" };
        let n = task_num(&c.task).unwrap_or(0);
        s.push_str(&format!(
            "<a class=\"nrow cand{tint}\" href=\"/t/{n}\"><span class=\"tid\">{}</span><span class=\"fname\">{}<span class=\"age\"> · claim {} · quiet {}</span></span></a>\n",
            esc(&c.task),
            esc(&c.name),
            ago(c.claim_age),
            ago(c.last_record_age),
        ));
    }
    if next.candidates.is_empty() {
        s.push_str("<div class=\"fact\">nothing claimed</div>\n");
    }
    s.push_str("</section>\n");
    s
}

pub fn thread_section(f: &Focus, form: &FormState) -> String {
    let mut s = format!(
        "<section id=\"thread\">\n<div class=\"thead\"><span class=\"tid\">{}</span> <span class=\"tstate\">{}</span> <span class=\"tname\">{}</span>{}</div>\n",
        esc(&f.show.id),
        esc(f.show.state),
        esc(&f.show.name),
        match &f.show.parent {
            Some(p) => format!(
                " <a class=\"parent\" href=\"/t/{}\">↑ {}</a>",
                task_num(p).unwrap_or(0),
                esc(p)
            ),
            None => String::new(),
        },
    );
    if let Some(receipt) = &f.show.receipt {
        s.push_str(&format!(
            "<div class=\"cmt receipt\"><span class=\"meta\">receipt</span>\n<div class=\"body\">{}</div>\n</div>\n",
            esc(receipt),
        ));
    }
    for p in &f.proposals {
        s.push_str(&format!(
            "<div class=\"pblock\"><span class=\"meta\">#{}</span> <span class=\"pname\">{} {} · {}</span>\n<form method=\"post\" action=\"/p/{}/accept\"><button class=\"judge\" type=\"submit\">accept</button></form>\n<form class=\"jform\" method=\"post\" action=\"/p/{}/reject\">\n<textarea name=\"note\" rows=\"2\" placeholder=\"ruling note\">{}</textarea>\n<button class=\"judge\" type=\"submit\">reject</button>\n</form>\n</div>\n",
            p.id,
            esc(p.action),
            esc(&p.task),
            esc(&p.name),
            p.id,
            p.id,
            esc(&form.note),
        ));
    }
    if f.thread.conversations.is_empty() && f.thread.stream.is_empty() {
        s.push_str("<div class=\"fact\">no comments yet</div>\n");
    }
    for conv in &f.thread.conversations {
        s.push_str("<div class=\"conv\">\n");
        s.push_str(&cmt_html(&conv.root));
        for reply in &conv.replies {
            s.push_str(&cmt_html(reply));
        }
        s.push_str("</div>\n");
    }
    for line in &f.thread.stream {
        s.push_str(&cmt_html(line));
    }
    s.push_str("</section>\n");
    s
}

fn cmt_html(c: &CommentLine) -> String {
    format!(
        "<div class=\"cmt\" id=\"c-{}\" style=\"margin-left:{}px\"><span class=\"meta\">#{}</span> <span class=\"who {}\">{}</span><span class=\"tag\">{}</span>\n<div class=\"body\">{}</div>\n</div>\n",
        c.seq,
        (c.depth - 1) * 18,
        c.seq,
        esc(&c.tier),
        esc(&c.actor),
        c.state.as_deref().map(esc).unwrap_or_default(),
        esc(&c.body),
    )
}

fn ago(secs: u64) -> String {
    match secs {
        0..=59 => "now".into(),
        60..=3599 => format!("{}m", secs / 60),
        3600..=86399 => format!("{}h", secs / 3600),
        _ => format!("{}d", secs / 86400),
    }
}

fn ribbon_section(task: &str, marks: &[RibbonMark], now: u64) -> String {
    let mut s = format!(
        "<section id=\"ribbon\">\n<h2>movement · {}</h2>\n",
        esc(task)
    );
    if marks.is_empty() {
        s.push_str("<div class=\"fact\">no movement in 72h</div>\n</section>\n");
        return s;
    }
    let start = now.saturating_sub(RIBBON_WINDOW_SECS);
    let span = RIBBON_WINDOW_SECS.max(1);
    s.push_str("<div class=\"rline\">");
    for m in marks {
        let pct = (m.at.saturating_sub(start)) as f64 / span as f64 * 100.0;
        let kind = match m.kind {
            MarkKind::Comment => "comment",
            MarkKind::Demand => "demand",
            MarkKind::Run => "run",
        };
        s.push_str(&format!(
            "<a class=\"mark k-{kind}\" style=\"left:{pct:.1}%\" href=\"#c-{}\" title=\"c-{} · {kind}\">c-{}</a>\n",
            m.seq, m.seq, m.seq,
        ));
    }
    s.push_str("</div>\n</section>\n");
    s
}

pub fn compose_section(task: usize, form: &FormState) -> String {
    let who = if form.need_who {
        "<input name=\"who\" placeholder=\"your name\" autocomplete=\"name\">\n"
    } else {
        ""
    };
    let error = form
        .error
        .as_deref()
        .map(|e| format!("<div class=\"formerr\">{}</div>\n", esc(e)))
        .unwrap_or_default();
    format!(
        "<section id=\"compose\">\n<form id=\"cform\" data-task=\"{task}\" method=\"post\" action=\"/compose\">\n<input type=\"hidden\" name=\"task\" value=\"{task}\">\n<textarea id=\"body\" name=\"body\" rows=\"3\" placeholder=\"one idea per comment\">{}</textarea>\n<div id=\"address\" class=\"address\"></div>\n{who}{error}<button type=\"submit\">post</button>\n</form>\n</section>\n",
        esc(&form.draft),
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
    .then(async (r) => {
      const doc = new DOMParser().parseFromString(await r.text(), 'text/html');
      const sec = doc.body.firstElementChild;
      const cur = sec && document.getElementById(sec.id);
      if (cur) cur.replaceWith(sec);
      if (r.ok) {
        f.elements.body.value = '';
        document.getElementById('address').textContent = '';
      }
    });
});
"#;

pub(crate) const STYLE: &str = r#"
:root { color-scheme: dark; }
* { box-sizing: border-box; }
#nav { position: absolute; opacity: 0; }
body {
  margin: 0; background: #100f0e; color: #b6b0a7;
  font: 15px/1.55 "Noto Sans", system-ui, sans-serif;
}
header {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 10px 18px; border-bottom: 1px solid #211e1b;
  font: 12px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  letter-spacing: 0.25em; color: #6f675c;
}
.navlabel { display: none; cursor: pointer; letter-spacing: 0.1em; }
.console {
  display: grid; grid-template-columns: 270px 1fr;
  max-width: 1180px; margin: 0 auto; min-height: calc(100vh - 40px);
}
#forest {
  border-right: 1px solid #211e1b; padding: 10px 14px 40px;
  background: #141210;
}
.col { padding: 6px 18px 60px; max-width: 780px; }
h2 {
  font: 500 11px ui-monospace, "SF Mono", Menlo, monospace;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: #6f675c; margin: 18px 0 6px;
}
h2:first-child { margin-top: 6px; }
a { color: #8b98a8; text-decoration: none; }
a:hover { color: #c9b99b; }
.frow, .nrow {
  display: flex; gap: 8px; padding: 2px 6px; border-radius: 3px;
  align-items: baseline; color: inherit;
}
.frow:hover, .nrow:hover { background: #1b1918; }
.tid {
  font: 12px ui-monospace, "SF Mono", Menlo, monospace;
  color: #6f675c; flex: none; min-width: 40px;
}
.fname { overflow-wrap: anywhere; color: #cfc8bd; }
.fname .age { color: #57504a; font: 12px ui-monospace, "SF Mono", Menlo, monospace; }
.frow.gate { border-left: 2px solid #b3905f; margin: 2px 0; }
.pmark { color: #b3905f; font: 11px ui-monospace, "SF Mono", Menlo, monospace; }
.nrow.ask { border-left: 2px solid #b3905f; }
.nrow.run { border-left: 2px solid #87996f; }
.nrow.adrift { background: #1c1614; }
.nrow.adrift .fname { color: #c39a90; }
.fact { color: #57504a; padding: 1px 6px; font-size: 13px; }
.thead { margin: 14px 0 4px; }
.tstate {
  font: 11px ui-monospace, "SF Mono", Menlo, monospace;
  letter-spacing: 0.12em; color: #b3905f;
}
.tname { color: #e2dcd2; overflow-wrap: anywhere; }
.parent { font-size: 13px; }
.conv { margin: 10px 0; border-left: 1px solid #211e1b; padding-left: 10px; }
.cmt { margin: 8px 0; }
.cmt .meta {
  font: 12px ui-monospace, "SF Mono", Menlo, monospace; color: #6f675c;
}
.cmt .who { font: 12px ui-monospace, "SF Mono", Menlo, monospace; }
.cmt .who.human { color: #c39a90; }
.cmt .who.agent { color: #8a8172; }
.cmt .tag {
  font: 11px ui-monospace, "SF Mono", Menlo, monospace;
  color: #b3905f; margin-left: 6px;
}
.cmt .body {
  color: #b6b0a7; white-space: pre-wrap; overflow-wrap: anywhere;
  margin-top: 1px;
}
.cmt.receipt .body { color: #8a8172; }
.pblock {
  border-left: 2px solid #b3905f; padding: 6px 10px; margin: 10px 0;
  background: #161311;
}
.pblock .pname { color: #b3905f; overflow-wrap: anywhere; }
.jform textarea {
  width: 100%; background: #1b1918; color: #b6b0a7;
  border: 1px solid #26221f; border-radius: 3px;
  padding: 6px 8px; font: inherit; resize: vertical;
}
button {
  background: #292624; color: #b6b0a7; border: 1px solid #211e1b;
  border-radius: 3px; padding: 4px 14px; font: inherit;
  margin-top: 6px; cursor: pointer;
}
button:hover { border-color: #b3905f; }
button.judge { border-color: #5a4a33; color: #b3905f; }
#cform { margin: 14px 0 0; }
#cform textarea {
  width: 100%; background: #1b1918; color: #b6b0a7;
  border: 1px solid #26221f; border-radius: 3px;
  padding: 8px 10px; font: inherit; resize: vertical;
}
#cform textarea:focus { outline: none; border-color: #5a4a33; }
#cform input {
  background: #1b1918; color: #b6b0a7; border: 1px solid #26221f;
  border-radius: 3px; padding: 4px 8px; font: inherit; margin-top: 4px;
}
.address {
  font: 12px ui-monospace, "SF Mono", Menlo, monospace;
  color: #8b98a8; min-height: 18px; margin-top: 4px;
}
.formerr {
  color: #c39a90; border-left: 2px solid #c39a90;
  padding: 4px 8px; margin-top: 8px;
}
.rline {
  position: relative; height: 22px; margin: 10px 0 2px;
  border-bottom: 1px solid #211e1b;
}
.mark {
  position: absolute; bottom: 0; transform: translateX(-50%);
  font: 10px ui-monospace, "SF Mono", Menlo, monospace;
  padding: 0 2px; border-radius: 2px 2px 0 0;
}
.k-comment { color: #57504a; }
.k-demand { color: #b3905f; }
.k-run { color: #87996f; }
.mark:hover { background: #292624; }
@media (max-width: 760px) {
  .console { grid-template-columns: 1fr; }
  #forest { display: none; border-right: none; border-bottom: 1px solid #211e1b; }
  .navlabel { display: inline; }
  #nav:checked ~ .console #forest { display: block; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Prose;
    use crate::events::Event;
    use crate::objects::proposal::ProposalAction;
    use crate::store::{Context, Record, Tier, World};
    use crate::types::actor::ActorName;
    use crate::views::{forest, next_panel, open_proposals, ribbon_marks, show_view, thread_view};
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

    // ---- section rendering ----

    /// t-0 ship it (done, receipt); t-1 <foo> task with a proposal and a
    /// demand exchange; now = 4000h puts everything in the ribbon window.
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
                agent(),
                Event::Commented {
                    target: task(1),
                    body: Prose::new("demand body".into()).unwrap(),
                    addressee: Some(Addressee::Agent),
                },
            ),
            record(
                6,
                3_160 * 3600,
                agent(),
                Event::Commented {
                    target: Target::Comment(CommentId(RecordId(5))),
                    body: Prose::new("reply body".into()).unwrap(),
                    addressee: None,
                },
            ),
            record(
                7,
                3_200 * 3600,
                human(),
                Event::Commented {
                    target: task(1),
                    body: Prose::new("lonely note".into()).unwrap(),
                    addressee: None,
                },
            ),
        ])
        .unwrap()
    }

    fn focus_of(world: &World, n: usize, now: u64) -> Focus {
        Focus {
            show: show_view(world, TaskId(n)).unwrap(),
            proposals: open_proposals(world)
                .into_iter()
                .filter(|v| v.task == format!("t-{n}"))
                .collect(),
            thread: thread_view(world, TaskId(n)).unwrap(),
            marks: ribbon_marks(world, TaskId(n), now),
        }
    }

    #[test]
    fn the_receipt_renders_only_when_done() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 0, 0), &Default::default());
        assert!(html.contains("suite green &lt;34&gt;"));
        let open = thread_section(&focus_of(&world, 1, 0), &Default::default());
        assert!(!open.contains("receipt"));
    }

    #[test]
    fn judgment_forms_render_on_the_focused_task() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1, 0), &Default::default());
        assert!(html.contains("action=\"/p/4/accept\""));
        assert!(html.contains("action=\"/p/4/reject\""));
        assert!(html.contains("name=\"note\""));
    }

    #[test]
    fn bodies_and_names_escape() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1, 0), &Default::default());
        assert!(html.contains("build &lt;foo&gt;"));
        assert!(!html.contains("build <foo>"));
        let console = Console {
            forest: forest(&world),
            gate: open_proposals(&world),
            next: next_panel(&world, 0),
            focus: None,
            form: FormState::default(),
            now: 0,
        };
        let page = page(&console);
        assert!(page.contains("build &lt;foo&gt;"));
    }

    #[test]
    fn the_exchange_nests_under_its_demand_and_orphans_stay_flat() {
        let world = fixture();
        let html = thread_section(&focus_of(&world, 1, 0), &Default::default());
        let reply = html.split("id=\"c-6\"").nth(1).expect("reply rendered");
        let root_left = html
            .split("id=\"c-5\"")
            .nth(1)
            .expect("demand rendered")
            .split("style=\"margin-left:")
            .nth(1)
            .and_then(|s| s.split('p').next())
            .and_then(|s| s.parse::<usize>().ok());
        assert_eq!(root_left, Some(0));
        let reply_left = reply
            .split("style=\"margin-left:")
            .nth(1)
            .and_then(|s| s.split('p').next())
            .and_then(|s| s.parse::<usize>().ok());
        assert_eq!(reply_left, Some(18));
        assert!(html.contains("lonely note"));
    }

    #[test]
    fn ribbon_marks_position_and_click_through() {
        let world = fixture();
        let now = 3_210 * 3600u64;
        let f = focus_of(&world, 1, now);
        // c-5 demand at 3140h, c-6 reply at 3160h, c-7 note at 3200h;
        // window [3138h, 3210h] holds all three, positioned by time
        let html = ribbon_section("t-1", &f.marks, now);
        assert!(html.contains("href=\"#c-5\""));
        assert!(html.contains("href=\"#c-7\""));
        assert!(html.contains("k-demand"));
        assert!(html.contains("k-comment"));
        assert!(html.contains("left:2.8%"));
        assert!(html.contains("left:86.1%"));
        assert!(!html.contains("k-run"), "no run bound in this fixture");
        // outside the window: nothing renders at all
        let late = ribbon_section(
            "t-1",
            &focus_of(&world, 1, 4_000 * 3600).marks,
            4_000 * 3600,
        );
        assert!(late.contains("no movement in 72h"));
    }

    #[test]
    fn the_compose_form_names_its_task() {
        let html = compose_section(
            9,
            &FormState {
                need_who: true,
                ..Default::default()
            },
        );
        assert!(html.contains("name=\"task\" value=\"9\""));
        assert!(html.contains("action=\"/compose\""));
        assert!(html.contains("name=\"who\""));
        assert!(html.contains("placeholder=\"one idea per comment\""));
    }
}
