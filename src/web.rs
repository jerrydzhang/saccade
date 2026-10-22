//! The console: task rail, focused thread, docked compose.
//! Every section is a renderer
//! over view types; fragments are sections, so the compose fetch swaps
//! what the page itself renders. The @-compiler is the composer's input
//! syntax over Commented events - the parse lives here, never in the
//! record. So does the prose renderer: bodies and explanations parse
//! as markdown with math on the way out, never in.

use crate::store::World;
use crate::views::{
    ArtifactLine, CommentLine, RefTarget, ShowView, TaskView, ThreadItem, ThreadView,
};
use crate::{RecordId, TaskId};
use math_core::{LatexToMathML, MathDisplay};
use pulldown_cmark::{Alignment, Event as MdEvent, Options, Parser, Tag, TagEnd};
use std::io::Read;
use std::path::PathBuf;
use time::OffsetDateTime;
use time::macros::format_description;

/// The artifact store's render-side face: where the bytes live, whether
/// they are present, and whether they sniff as an image. The one sniff
/// definition - the read door serves the same content types it inlines.
#[derive(Clone, Default)]
pub struct ArtifactStore {
    dir: Option<PathBuf>,
}

impl ArtifactStore {
    pub fn at(dir: PathBuf) -> Self {
        ArtifactStore { dir: Some(dir) }
    }

    /// One artifact's bytes in the store.
    pub fn path(&self, hash: &str) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(hash))
    }

    /// Some(true): present and an image. Some(false): present, other
    /// content. None: absent - the honest miss.
    pub fn probe(&self, hash: &str) -> Option<bool> {
        let mut prefix = [0u8; 16];
        let read = std::fs::File::open(self.path(hash)?)
            .ok()?
            .read(&mut prefix)
            .ok()?;
        Some(sniff_image(&prefix[..read]).is_some())
    }
}

/// Content type by magic bytes - the formats the console inlines.
pub fn sniff_image(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

/// The task state's chip color, one per state out of the c-522 palette:
/// open whispers, claimed is gold like the work it holds, delivered is
/// the link tone like the hand-off it awaits, done is green like a
/// settled run, dropped is rose like the human act it was.
fn state_color(state: &str) -> &'static str {
    match state {
        "offered" => "#dac09a",
        "concluded" => "#a2c4a3",
        "cancelled" => "#c4a6a8",
        _ => "#6d6562",
    }
}

pub struct Console {
    pub board: Vec<TaskView>,
    pub closed: Vec<TaskView>,
    /// The focused task's panel facts, when one is focused.
    pub focus: Option<Focus>,
    pub form: FormState,
    /// The artifact bytes door's store, for rendering at position.
    pub store: ArtifactStore,
    /// The serving repo's name, when serve resolved a root; the
    /// chrome names its instance by it, and pages built without one
    /// render the unnamed chrome of before.
    pub repo_name: Option<RepoName>,
}

/// The serving repo's name: the basename names the tab title and the
/// header mark, and the hue - the same root hash the state-dir slug
/// tails - dots the favicon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoName {
    pub basename: String,
    pub hue: u32,
}

impl RepoName {
    pub fn of(repo_root: &std::path::Path) -> Self {
        RepoName {
            basename: crate::paths::repo_basename(repo_root),
            hue: crate::paths::repo_hue(repo_root),
        }
    }

    /// The tab title a full document carries under this name.
    pub fn tab_title(&self) -> String {
        format!("{} | saccade", esc(&self.basename))
    }
}

/// The favicon: a dot in the repo's hue, inline as an SVG data URI.
pub fn favicon_link(hue: u32) -> String {
    let svg = format!(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><circle cx='8' cy='8' r='7' fill='hsl({hue},60%,55%)'/></svg>"
    );
    let uri = svg.replace('<', "%3C").replace('>', "%3E");
    format!("<link rel=\"icon\" href=\"data:image/svg+xml,{uri}\">")
}

pub struct Focus {
    pub show: ShowView,
    pub thread: ThreadView,
}

/// A refused revise, rendered at its own comment's reveal: the draft
/// stays editable and the reason sits beside it.
pub struct ReviseForm {
    pub seq: usize,
    pub draft: String,
    pub error: String,
}

#[derive(Default)]
pub struct FormState {
    pub error: Option<String>,
    pub draft: String,
    pub note: String,
    /// The actor name prefilled into every act form - the cookie's
    /// value when one exists, blank for a fresh browser.
    pub who: String,
    /// A refused decision: the reason rendered beside the doors.
    pub judge_error: Option<String>,
    /// A refused revise: the reason and the kept draft, at the row.
    pub revise: Option<ReviseForm>,
}

pub struct Address {
    pub body: String,
    pub home: TaskId,
    /// The citation mapping in marker order: marker k names
    /// references[k-1]
    pub references: Vec<RecordId>,
}

enum Tok {
    Cite(usize),
    Home(TaskId),
}

/// The composer's grammar over the fold's public numbers: `@c-N`
/// becomes an inline marker citing that comment, `@t-N` rehomes the
/// comment to that task; bare ids stay prose. A token must stand
/// alone (the char before `@` is not alphanumeric, the char after is
/// neither alphanumeric nor `-`). The first home token wins; later
/// home tokens stay prose. Cite tokens leave a `[^k]` marker at their
/// position and a reference in the mapping. The parse lives here,
/// never in the record.
pub(crate) fn compile(body: &str, home: TaskId, world: &World) -> Address {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(body.len());
    let mut home = home;
    let mut home_taken = false;
    let mut references: Vec<RecordId> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let boundary = chars[i] == '@' && (i == 0 || !chars[i - 1].is_alphanumeric());
        let token = if boundary {
            token_at(&chars, i + 1)
        } else {
            None
        };
        match token {
            Some((Tok::Cite(n), end)) => {
                // a cite the fold cannot resolve stays the words it was
                match world.comment_at_num(n) {
                    Some(id) => {
                        references.push(id.0);
                        out.push_str(&format!("[^{}]", references.len()));
                        i = end;
                    }
                    None => {
                        out.extend(&chars[i..end]);
                        i = end;
                    }
                }
            }
            Some((Tok::Home(t), end)) if !home_taken => {
                home = t;
                home_taken = true;
                i = skip_space(&chars, end, &out);
            }
            // a second home token is prose, @ and all
            Some((Tok::Home(_), end)) => {
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
        home,
        references,
    }
}

/// One following space goes with a consumed home token when it would double up.
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
    for (prefix, token) in [
        ("c-", Tok::Cite as fn(usize) -> Tok),
        ("t-", |n: usize| Tok::Home(TaskId(n))),
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
            return Some((token(n), p + word.len()));
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

/// Sun Sep 20 - the thread's day dividers
fn fmt_d(ts: u64) -> String {
    let desc = format_description!("[weekday repr:short] [month repr:short] [day]");
    local(ts).format(&desc).unwrap_or_default()
}

// ---- the page ----

pub fn page(c: &Console) -> String {
    let board = board_section(&c.board, &c.closed, focus_num(c).as_deref());
    let panel = match &c.focus {
        Some(f) => format!(
            "<div class=\"thead\"><span class=\"tstate\" style=\"color:{state_color}\">* {state}</span>\n<div class=\"ttitle\">{title}</div>\n<div class=\"tmeta mono\">{meta}</div>\n</div>\n{thread}{compose}",
            state_color = state_color(f.show.state),
            state = f.show.state.to_uppercase(),
            title = esc(&f.show.name),
            meta = esc(&head_meta(&f.show)),
            thread = thread_section(f, &c.form, &c.store),
            compose = compose_section(task_num(&f.show.id).unwrap_or(0), &c.form),
        ),
        None => "<div class=\"thead\"></div>\n<section id=\"thread\"><div class=\"nempty\">no task focused</div></section>\n".to_string(),
    };
    let title = c
        .repo_name
        .as_ref()
        .map(|n| n.tab_title())
        .unwrap_or_else(|| "saccade | console".into());
    let favicon = c
        .repo_name
        .as_ref()
        .map(|n| favicon_link(n.hue))
        .unwrap_or_default();
    let header = match &c.repo_name {
        Some(n) => format!(
            "<header><span class=\"hgroup\"><span class=\"brand\">SACCADE | CONSOLE</span><span class=\"rname\">{name}</span></span><span><label class=\"fbtn\" for=\"nav\">tasks &gt;</label></span></header>",
            name = esc(&n.basename)
        ),
        None => "<header><span class=\"brand\">SACCADE | CONSOLE</span><span><label class=\"fbtn\" for=\"nav\">tasks &gt;</label></span></header>".into(),
    };
    format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{title}</title>{favicon}<style>{STYLE}</style></head>\n<body>\n{header}\n<input type=\"checkbox\" id=\"nav\">\n<div class=\"if\">\n<div class=\"cols withboard\">\n{board}\n<div class=\"tpanel\">\n{panel}\n</div>\n</div>\n</div>\n<script>{JS}</script>\n</body></html>\n"
    )
}

fn focus_num(c: &Console) -> Option<String> {
    c.focus.as_ref().map(|f| f.show.id.clone())
}

fn head_meta(show: &ShowView) -> String {
    let lineage = show.id.clone();
    let owned = match &show.owner {
        Some(owner) => format!("{lineage} | owned by {owner}"),
        None => lineage,
    };
    match &show.workspace {
        Some(workspace) => format!("{owned} | {workspace}"),
        None => owned,
    }
}

fn task_num(id: &str) -> Option<usize> {
    id.strip_prefix("t-").and_then(|n| n.parse().ok())
}

// ---- the task rail ----

fn board_section(rows: &[TaskView], closed: &[TaskView], focus: Option<&str>) -> String {
    let mut s = String::from("<aside class=\"board\">\n");
    for row in rows {
        let sel = focus == Some(row.id.as_str());
        s.push_str(&format!(
            "<a class=\"frow{}\" href=\"/t/{}\"><span class=\"fid mono\">{}</span><span class=\"schip\" style=\"color:{}\">{}</span><span class=\"fname\">{}</span></a>\n",
            if sel { " sel" } else { "" },
            task_num(&row.id).unwrap_or(0),
            esc(&row.id),
            state_color(row.state),
            esc(row.state.to_uppercase().as_str()),
            esc(&row.name),
        ));
    }
    if rows.is_empty() {
        s.push_str("<div class=\"fgate-empty\">no live tasks</div>\n");
    }
    if !closed.is_empty() {
        s.push_str("<div class=\"fsect\">CLOSED</div>\n");
        for row in closed {
            s.push_str(&format!(
                "<a class=\"frow closed\" href=\"/t/{}\"><span class=\"fid mono\">{}</span><span class=\"schip\" style=\"color:{}\">{}</span><span class=\"fname\">{}</span></a>\n",
                task_num(&row.id).unwrap_or(0),
                esc(&row.id),
                state_color(row.state),
                esc(row.state.to_uppercase().as_str()),
                esc(&row.name),
            ));
        }
    }
    s.push_str("</aside>\n");
    s
}
// ---- the thread panel ----

struct Judge {
    task: usize,
    offer: usize,
    grounds: String,
    error: Option<String>,
    who: String,
}

pub fn thread_section(f: &Focus, form: &FormState, store: &ArtifactStore) -> String {
    let mut s = String::from("<section id=\"thread\" data-task=\"");
    s.push_str(&f.show.id);
    s.push_str("\">\n");
    if let Some(why) = &f.show.explanation {
        s.push_str(&format!(
            "<div class=\"receipt\"><span class=\"xk\" style=\"color:#dac09a\">EXPLANATION</span>\n<div class=\"nbody\">{}</div>\n</div>\n",
            prose_html(why, &[], store),
        ));
    }
    // the form names the offer it was rendered against, so a stale
    // page cannot decide an account it never showed
    let judge = (f.show.state == "offered")
        .then(|| Some((task_num(&f.show.id)?, f.show.offer?)))
        .flatten()
        .map(|(task, offer)| Judge {
            task,
            offer,
            grounds: form.note.clone(),
            error: form.judge_error.clone(),
            who: form.who.clone(),
        });
    let mut last_time: Option<u64> = None;
    for item in &f.thread.items {
        // artifacts carry no time in the fold; they ride beside the
        // utterances, never opening a gap
        if matches!(item, ThreadItem::Artifacts(_)) {
            s.push_str(&item_html(item, form, judge.as_ref(), store));
            continue;
        }
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
        s.push_str(&item_html(item, form, judge.as_ref(), store));
    }
    if f.thread.items.is_empty() && f.show.explanation.is_none() {
        s.push_str("<div class=\"nempty\">no comments yet</div>\n");
    }
    s.push_str("</section>\n");
    s
}

fn item_span(item: &ThreadItem) -> (u64, u64) {
    match item {
        // artifacts carry no time in the fold; the gap logic skips them
        ThreadItem::Artifacts(_) => (0, 0),
        ThreadItem::Note(line) => (line.born_at, line.born_at),
    }
}

fn item_html(
    item: &ThreadItem,
    form: &FormState,
    judge: Option<&Judge>,
    store: &ArtifactStore,
) -> String {
    match item {
        // the cluster is its own card: each artifact renders inside,
        // association with a comment only by citation
        ThreadItem::Artifacts(lines) => {
            let mut s = String::from("<div class=\"agrp\">\n");
            for line in lines {
                s.push_str(&artifact_html(line, 0, store));
            }
            s.push_str("</div>\n");
            s
        }
        ThreadItem::Note(line) => node_html(line, kind_chip(line.kind), None, form, judge, store),
    }
}

/// The chip each variant carries; a note carries none - the plain row
/// it always was.
fn kind_chip(kind: &str) -> Option<(&'static str, &'static str)> {
    match kind {
        "offer" => Some(("OFFER", "#dac09a")),
        "accept" => Some(("ACCEPT", "#a2c4a3")),
        "reject" => Some(("REJECT", "#c4a6a8")),
        "use" => Some(("USE", "#a2a3c4")),
        _ => None,
    }
}

/// One comment row: kind chip, whisper mono meta (actor, number,
/// time), full body. The body renders its `[^k]` markers through the
/// comment's own reference mapping; bare ids stay the words they
/// were. Every live row carries the revise reveal - the human
/// surface's inline door - and a withdrawn note renders its tombstone
/// instead, with no door: repair of a withdrawn deposit is a new
/// deposit.
fn node_html(
    line: &CommentLine,
    kind: Option<(&str, &str)>,
    tag: Option<&str>,
    form: &FormState,
    judge: Option<&Judge>,
    store: &ArtifactStore,
) -> String {
    let chip = kind
        .map(|(k, color)| format!("<span class=\"xk\" style=\"color:{color}\">{k}</span>"))
        .unwrap_or_default();
    // the citation edges, carried explicitly: a reply names the
    // comment or record it cites through its mapping
    let target = line
        .references
        .iter()
        .map(ref_chip)
        .collect::<Vec<_>>()
        .join("");
    // the reliance's declared source, named beside the home: a use
    // points at another task, its binding at that task's conclusion
    let uses = line
        .source
        .map(|t| format!("<span class=\"nseq mono\">uses t-{}</span>", t.0))
        .unwrap_or_default();
    let bound = line
        .binding
        .as_ref()
        .map(|b| {
            format!(
                "<span class=\"nseq mono\" title=\"{}\">bound c-{}</span>",
                esc(&b.grounds),
                b.offer
            )
        })
        .unwrap_or_default();
    if line.withdrawn {
        // the tombstone: one line, the note - and a withdrawer who
        // differs from the birth author is named
        let withdrawn = match line.withdrawer.as_deref() {
            Some(who) => format!("<span class=\"nseq\">withdrawn by {}</span>", esc(who)),
            None => "<span class=\"nseq\">withdrawn</span>".to_string(),
        };
        let note = esc(line.withdrawal_note.as_deref().unwrap_or_default());
        return format!(
            "<div class=\"nrow2\" id=\"c-{num}\" style=\"padding-left:26px\">\n<div class=\"nmeta\">{chip}<span class=\"nwho {tier}\">{actor}</span><span class=\"nseq mono\">c-{num}</span>{target}{withdrawn}<span class=\"nseq mono\">{time}</span></div>\n<div class=\"nbody tomb\">withdrawn: {note}</div>\n</div>\n",
            num = line.num,
            tier = esc(&line.tier),
            actor = esc(&line.actor),
            time = esc(&fmt_t(line.born_at)),
        );
    }
    // the descriptive mark: the fold presents a revised body, and a
    // reviser who differs from the birth author is named
    let revised = match (line.revised, line.reviser.as_deref()) {
        (true, Some(who)) => format!("<span class=\"nseq\">revised by {}</span>", esc(who)),
        (true, None) => "<span class=\"nseq\">revised</span>".to_string(),
        _ => String::new(),
    };
    let extra = tag
        .map(|t| format!("<span class=\"nseq\">{}</span>", esc(t)))
        .unwrap_or_default();
    // reject is the default button: a stray Enter meets the grounds
    // guard, never a silent acceptance
    let doors = judge
        .filter(|j| j.offer == line.seq)
        .map(|j| {
            format!(
                "<form class=\"jform\" method=\"post\" action=\"/t/{task}/reject\">\n<input type=\"hidden\" name=\"offer\" value=\"{offer}\">\n<textarea name=\"grounds\" rows=\"2\" placeholder=\"grounds - what needs reconsideration\">{grounds}</textarea>\n{err}<div class=\"jbtns\">{who}<button class=\"sendbtn\" name=\"reject\" value=\"1\" type=\"submit\">reject</button><button class=\"sendbtn\" name=\"accept\" value=\"1\" type=\"submit\" formaction=\"/t/{task}/accept\">accept</button></div>\n</form>\n",
                task = j.task,
                offer = j.offer,
                grounds = esc(&j.grounds),
                err = j
                    .error
                    .as_deref()
                    .map(|e| format!("<div class=\"formerr\">{}</div>\n", esc(e)))
                    .unwrap_or_default(),
                who = who_input(&j.who),
            )
        })
        .unwrap_or_default();
    // the bodyless accept carries no prose to repair, so no reveal
    let body = if line.body.is_empty() {
        String::new()
    } else {
        let refused = form
            .revise
            .as_ref()
            .filter(|r| r.seq == line.num)
            .map(|r| format!("<div class=\"formerr\">{}</div>\n", esc(&r.error)))
            .unwrap_or_default();
        let draft = form
            .revise
            .as_ref()
            .filter(|r| r.seq == line.num)
            .map(|r| esc(&r.draft))
            .unwrap_or_else(|| esc(&line.body));
        let open = if form.revise.as_ref().is_some_and(|r| r.seq == line.num) {
            " open"
        } else {
            ""
        };
        format!(
            "<div class=\"nbody\">{}</div>\n<details class=\"revise\"{open}><summary>revise</summary>\n<form class=\"rform\" method=\"post\" action=\"/c/{seq}/revise\">\n<textarea name=\"body\" rows=\"2\">{draft}</textarea>\n{refused}<div class=\"jbtns\">{who_input}<button class=\"sendbtn\" name=\"revise\" value=\"1\" type=\"submit\">revise</button></div>\n</form>\n</details>\n",
            prose_html(&line.body, &line.references, store),
            seq = line.num,
            who_input = who_input(&form.who),
        )
    };
    format!(
        "<div class=\"nrow2\" id=\"c-{num}\" style=\"padding-left:26px\">\n<div class=\"nmeta\">{chip}<span class=\"nwho {tier}\">{actor}</span><span class=\"nseq mono\">c-{num}</span>{target}{uses}{bound}{extra}{revised}<span class=\"nseq mono\">{time}</span></div>\n{body}{doors}</div>\n",
        num = line.num,
        tier = esc(&line.tier),
        actor = esc(&line.actor),
        time = esc(&fmt_t(line.born_at)),
    )
}

/// One citation's chip: the target's public name, linking its home.
fn ref_chip(target: &RefTarget) -> String {
    match target {
        RefTarget::Comment { num, task } => {
            format!("<a class=\"nseq mono\" href=\"/t/{task}#c-{num}\">->c-{num}</a>")
        }
        RefTarget::Task { task } => {
            format!("<a class=\"nseq mono\" href=\"/t/{task}\">->t-{task}</a>")
        }
        RefTarget::Incarnation { num, task } => match task {
            Some(task) => format!("<a class=\"nseq mono\" href=\"/t/{task}\">->i-{num}</a>"),
            None => format!("<span class=\"nseq mono\">->i-{num}</span>"),
        },
        RefTarget::Message { num } => format!("<span class=\"nseq mono\">->m-{num}</span>"),
        RefTarget::Artifact { seq, task, .. } => {
            format!("<a class=\"nseq mono\" href=\"/t/{task}#a-{seq}\">->seq {seq}</a>")
        }
        RefTarget::Record { seq, task } => match task {
            Some(task) => {
                format!("<a class=\"nseq mono\" href=\"/t/{task}\">->seq {seq}</a>")
            }
            None => format!("<span class=\"nseq mono\">->seq {seq}</span>"),
        },
    }
}

/// One artifact at its home position: caption plus inline image by
/// content type, a pointer for other bytes, honestly unavailable on a
/// store miss. The image links the bytes door.
fn artifact_html(line: &ArtifactLine, indent: usize, store: &ArtifactStore) -> String {
    let pad = 26 + indent * 22;
    let short: String = line.hash.chars().take(12).collect();
    let name = esc(&line.name);
    let figure = match store.probe(&line.hash) {
        Some(true) => format!(
            "<a class=\"afull\" href=\"/a/{hash}\"><img class=\"aimg\" src=\"/a/{hash}\" alt=\"{name}\"></a>\n",
            hash = esc(&line.hash),
        ),
        Some(false) => format!(
            "<a class=\"afile\" href=\"/a/{hash}\">{name} | {short}</a>\n",
            hash = esc(&line.hash),
        ),
        None => format!("<span class=\"acap\">{name} | unavailable</span>\n"),
    };
    format!(
        "<div class=\"nrow2 arow\" id=\"a-{seq}\" style=\"padding-left:{pad}px\">\n{figure}<div class=\"nmeta\"><span class=\"xk\" style=\"color:#a2c3c4\">ARTIFACT</span><span class=\"nseq mono\">seq {seq}</span></div>\n</div>\n",
        seq = line.seq,
    )
}

/// The mention's render: a thumbnail-and-caption card linking the
/// artifact's home position. A non-image is a caption card; a miss is
/// the card with its honest verdict.
fn mention_card(seq: usize, task: usize, name: &str, hash: &str, store: &ArtifactStore) -> String {
    let home = format!("/t/{task}#a-{seq}");
    let caption = esc(name);
    let short: String = hash.chars().take(12).collect();
    match store.probe(hash) {
        Some(true) => format!(
            "<a class=\"acard\" href=\"{home}\"><img src=\"/a/{hash}\" alt=\"{caption}\"><span class=\"acap\">{caption}</span></a>"
        ),
        Some(false) => format!(
            "<a class=\"acard file\" href=\"{home}\"><span class=\"acap\">{caption} | {short}</span></a>"
        ),
        None => format!(
            "<a class=\"acard miss\" href=\"{home}\"><span class=\"acap\">{caption} | unavailable</span></a>"
        ),
    }
}

/// The reading-side resolver over escaped text: a `[^k]` marker whose
/// comment carries a reference at that position resolves - a comment
/// to its home anchor, an artifact to its card, a task or incarnation
/// birth to its thread; every other token stays the words it was.
/// Bare c-N tokens are prose: navigation follows the mapping, never
/// the mention. The parse lives here, never in the record.
fn markerize(escaped: &str, refs: &[RefTarget], store: &ArtifactStore) -> String {
    let mut out = String::with_capacity(escaped.len());
    let mut at = 0usize;
    for token in crate::markers::tokens_in_run(escaped) {
        // a number the mapping cannot take stays the words it was
        let resolved = token
            .number
            .and_then(|k| refs.get(k - 1))
            .filter(|_| token.number.is_some_and(|k| k >= 1));
        match resolved {
            Some(target) => {
                out.push_str(&escaped[at..token.start]);
                out.push_str(&marker_link(token.number.unwrap_or(0), target, store));
                at = token.end;
            }
            None => continue,
        }
    }
    out.push_str(&escaped[at..]);
    out
}

/// One resolved marker's render: a superscript link for records, the
/// card for an artifact.
fn marker_link(k: usize, target: &RefTarget, store: &ArtifactStore) -> String {
    match target {
        RefTarget::Comment { num, task } => {
            format!("<a class=\"mref\" href=\"/t/{task}#c-{num}\">[^{k}]</a>")
        }
        RefTarget::Task { task } => {
            format!("<a class=\"mref\" href=\"/t/{task}\">[^{k}]</a>")
        }
        RefTarget::Incarnation { num, task } => match task {
            Some(task) => {
                format!("<a class=\"mref\" href=\"/t/{task}\" title=\"i-{num}\">[^{k}]</a>")
            }
            None => format!("<span class=\"mref dead\" title=\"i-{num}\">[^{k}]</span>"),
        },
        RefTarget::Message { num } => {
            format!("<span class=\"mref dead\" title=\"m-{num}\">[^{k}]</span>")
        }
        RefTarget::Artifact {
            seq,
            task,
            name,
            hash,
        } => format!(
            "<span class=\"mrefa\">{}<sup><a class=\"mref\" href=\"/t/{task}#a-{seq}\">[^{k}]</a></sup></span>",
            mention_card(*seq, *task, name, hash, store)
        ),
        RefTarget::Record { seq, task } => match task {
            Some(task) => format!("<a class=\"mref\" href=\"/t/{task}\">[^{k}]</a>"),
            None => format!("<span class=\"mref dead\">[^{k}] | seq {seq}</span>"),
        },
    }
}

/// The one scheme door for a markdown link: http or https renders as a
/// link; every other scheme - and no scheme - renders as words.
fn http_s(dest: &str) -> bool {
    let d = dest.as_bytes();
    let head = |p: &[u8]| d.len() >= p.len() && d[..p.len()].eq_ignore_ascii_case(p);
    head(b"http://") || head(b"https://")
}

/// A conversion that fails - or markup beyond the crate's own
/// vocabulary - renders no math at all; the fence renderers fall back
/// to the literal source. The conversion is unwound by catch: a parser
/// panic on legal latex is a render miss, never a dead request.
fn math_mml(src: &str, display: MathDisplay) -> Option<String> {
    let converted = LatexToMathML::new(Default::default()).ok().and_then(|c| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            c.convert_with_local_state(src, display)
                .ok()
                .map(|r| r.mathml)
        }))
        .ok()
        .flatten()
    });
    let m = converted.filter(|m| crate_authored(m))?;
    // the return path: a click on the formula copies these bytes
    let raw = esc(src);
    Some(m.replacen(
        "<math",
        &format!("<math data-raw=\"{raw}\" title=\"{raw}\""),
        1,
    ))
}

fn math_html(src: &str, fence: &str, display: MathDisplay) -> String {
    match math_mml(src, display) {
        Some(m) => m,
        None => format!("{fence}{}{fence}", esc(src)),
    }
}

/// The converter renders author bytes into markup; only its own tag
/// vocabulary may pass, so any other tag shape is a miss, never markup.
fn crate_authored(mathml: &str) -> bool {
    const TAGS: &[&str] = &[
        "math",
        "mi",
        "mn",
        "mo",
        "mtext",
        "mspace",
        "mpadded",
        "mrow",
        "mfrac",
        "msqrt",
        "mroot",
        "msub",
        "msup",
        "msubsup",
        "mmultiscripts",
        "mprescripts",
        "mover",
        "munder",
        "munderover",
        "menclose",
        "merror",
        "mtable",
        "mtr",
        "mtd",
    ];
    let b = mathml.as_bytes();
    let mut i = 0;
    while let Some(at) = b[i..].iter().position(|&c| c == b'<') {
        let open = i + at + 1;
        let open = if open < b.len() && b[open] == b'/' {
            open + 1
        } else {
            open
        };
        let end = b[open..]
            .iter()
            .position(|&c| !c.is_ascii_alphanumeric())
            .map_or(b.len(), |p| open + p);
        let name = std::str::from_utf8(&b[open..end]).unwrap_or("");
        if !TAGS.contains(&name) {
            return false;
        }
        i = end;
    }
    true
}

/// The prose renderer over raw bytes: CommonMark with tables and
/// math, the same law as the @-compiler and the marker resolver - the
/// parse lives here, never in the record. Text events escape and
/// resolve `[^k]` markers through the given mapping; code spans and
/// code blocks are verbatim; raw HTML never passes, rendering as the
/// words it was; a link renders only on the http(s) schemes with
/// rel=noopener; soft breaks are whitespace and hard breaks the
/// CommonMark kinds.
fn prose_html(raw: &str, refs: &[RefTarget], store: &ArtifactStore) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES | Options::ENABLE_MATH);
    let mut out = String::with_capacity(raw.len());
    let mut in_code = false;
    let mut code_buf = String::new();
    let mut in_head = false;
    let mut link_open = false;
    let mut aligns: Vec<Alignment> = Vec::new();
    let mut col = 0;
    // text accumulates until the next event: a marker's brackets can
    // span the parser's own text splits, so the resolver reads whole runs
    let mut text_buf = String::new();
    let flush_text = |buf: &mut String, out: &mut String| {
        if !buf.is_empty() {
            out.push_str(&markerize(buf, refs, store));
            buf.clear();
        }
    };
    for ev in Parser::new_ext(raw, opts) {
        if !matches!(ev, MdEvent::Text(_)) {
            flush_text(&mut text_buf, &mut out);
        }
        match ev {
            MdEvent::Start(tag) => match tag {
                Tag::Paragraph => out.push_str("<p>"),
                Tag::Heading { level, .. } => out.push_str(&format!("<h{}>", level as u8)),
                Tag::BlockQuote(_) => out.push_str("<blockquote>\n"),
                Tag::CodeBlock(_) => {
                    in_code = true;
                    code_buf.clear();
                }
                Tag::List(None) => out.push_str("<ul>\n"),
                Tag::List(Some(1)) => out.push_str("<ol>\n"),
                Tag::List(Some(start)) => out.push_str(&format!("<ol start=\"{start}\">\n")),
                Tag::Item => out.push_str("<li>"),
                Tag::Table(align) => {
                    aligns = align;
                    out.push_str("<table>\n");
                }
                Tag::TableHead => {
                    in_head = true;
                    col = 0;
                    out.push_str("<thead><tr>");
                }
                Tag::TableRow => {
                    col = 0;
                    out.push_str("<tr>");
                }
                Tag::TableCell => {
                    let style = match aligns.get(col).copied().unwrap_or(Alignment::None) {
                        Alignment::None => "",
                        Alignment::Left => " style=\"text-align:left\"",
                        Alignment::Center => " style=\"text-align:center\"",
                        Alignment::Right => " style=\"text-align:right\"",
                    };
                    let cell = if in_head {
                        format!("<th{style}>")
                    } else {
                        format!("<td{style}>")
                    };
                    out.push_str(&cell);
                    col += 1;
                }
                Tag::Emphasis => out.push_str("<em>"),
                Tag::Strong => out.push_str("<strong>"),
                Tag::Link {
                    dest_url, title, ..
                } if http_s(&dest_url) => {
                    link_open = true;
                    let t = if title.is_empty() {
                        String::new()
                    } else {
                        format!(" title=\"{}\"", esc(&title))
                    };
                    out.push_str(&format!(
                        "<a href=\"{}\" rel=\"noopener\"{t}>",
                        esc(&dest_url)
                    ));
                }
                _ => {}
            },
            MdEvent::End(end) => match end {
                TagEnd::Paragraph => out.push_str("</p>\n"),
                TagEnd::Heading(level) => out.push_str(&format!("</h{}>\n", level as u8)),
                TagEnd::BlockQuote(_) => out.push_str("</blockquote>\n"),
                TagEnd::CodeBlock => {
                    in_code = false;
                    // the whole block's bytes ride the pre for the copy button
                    out.push_str(&format!(
                        "<pre data-raw=\"{}\"><code>{}</code></pre>\n",
                        esc(&code_buf),
                        esc(&code_buf)
                    ));
                }
                TagEnd::List(ordered) => out.push_str(if ordered { "</ol>\n" } else { "</ul>\n" }),
                TagEnd::Item => out.push_str("</li>\n"),
                TagEnd::Table => {
                    aligns.clear();
                    out.push_str("</table>\n");
                }
                TagEnd::TableHead => {
                    in_head = false;
                    out.push_str("</tr></thead>\n");
                }
                TagEnd::TableRow => out.push_str("</tr>\n"),
                TagEnd::TableCell => {
                    out.push_str(if in_head { "</th>" } else { "</td>" });
                }
                TagEnd::Emphasis => out.push_str("</em>"),
                TagEnd::Strong => out.push_str("</strong>"),
                TagEnd::Link if link_open => {
                    link_open = false;
                    out.push_str("</a>");
                }
                _ => {}
            },
            MdEvent::Text(t) => {
                if in_code {
                    code_buf.push_str(&t);
                } else {
                    text_buf.push_str(&esc(&t));
                }
            }
            MdEvent::Code(c) => out.push_str(&format!("<code>{}</code>", esc(&c))),
            MdEvent::InlineMath(src) => out.push_str(&math_html(&src, "$", MathDisplay::Inline)),
            MdEvent::DisplayMath(src) => match math_mml(&src, MathDisplay::Block) {
                Some(m) => out.push_str(&format!("<span class=\"dmath\">{m}</span>")),
                // a miss is literal bytes, not a broken block
                None => out.push_str(&format!("$${}$$", esc(&src))),
            },
            MdEvent::Html(h) | MdEvent::InlineHtml(h) => out.push_str(&esc(&h)),
            MdEvent::SoftBreak => out.push('\n'),
            MdEvent::HardBreak => out.push_str("<br />\n"),
            MdEvent::Rule => out.push_str("<hr>\n"),
            _ => {}
        }
    }
    flush_text(&mut text_buf, &mut out);
    out
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
const re = /(^|[^A-Za-z0-9])@(c-[0-9]+|t-[0-9]+)(?![A-Za-z0-9-])/g;
function resolve(text, home) {
  if (!text.trim()) return '';
  let cite = [];
  let rehomed = null;
  for (const m of text.matchAll(re)) {
    if (m[2][0] === 'c') cite.push(m[2]);
    else if (rehomed === null) rehomed = m[2];
  }
  const parts = [];
  if (rehomed) parts.push('home -> ' + rehomed);
  else parts.push('home -> t-' + home);
  if (cite.length) parts.push('cites ' + cite.join(' '));
  return parts.join(' | ');
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
  const anchor = readingAnchor();
  fetch('/compose', { method: 'POST', body: new URLSearchParams(new FormData(f)) })
    .then((r) => {
      // a rehome answers with the 303 to the comment's new home:
      // the section would land on a stale page, so navigate instead
      if (r.redirected) {
        sessionStorage.setItem('action-return', '1');
        location.assign(r.url);
        return null;
      }
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
      }
      restoreReading(anchor);
    });
});

// the topmost visible record and its offset name the place; an
// action's redirect is a return, not navigation - a chosen #anchor
// still wins
function readingAnchor() {
  const t = document.getElementById('thread');
  if (!t) return null;
  const top = t.getBoundingClientRect().top;
  for (const el of t.querySelectorAll('[id]')) {
    if (el.getBoundingClientRect().bottom > top)
      return { id: el.id, off: el.getBoundingClientRect().top - top };
  }
  return null;
}
function restoreReading(saved) {
  let p;
  try { p = typeof saved === 'string' ? JSON.parse(saved) : saved; } catch { return; }
  const t = document.getElementById('thread');
  const el = p && document.getElementById(p.id);
  if (!t || !el) return;
  t.scrollTop = el.getBoundingClientRect().top - t.getBoundingClientRect().top + t.scrollTop - p.off;
}
document.addEventListener('submit', (e) => {
  if (e.target.id !== 'cform') sessionStorage.setItem('action-return', '1');
});
const rkey = 'reading:' + (document.getElementById('thread')?.dataset.task || '');
addEventListener('pagehide', () => {
  const p = readingAnchor();
  if (p) sessionStorage.setItem(rkey, JSON.stringify(p));
});
const cameBack = sessionStorage.getItem('action-return');
if (cameBack) sessionStorage.removeItem('action-return');
if (cameBack || !location.hash) {
  // before load the grid is half-laid-out and the clamp eats the
  // restored position
  addEventListener('load', () => restoreReading(sessionStorage.getItem(rkey)));
}

// sessionStorage: the position survives full navigations and dies with
// the tab
const fkey = 'board-scroll';
addEventListener('pagehide', () => {
  const f = document.querySelector('.board');
  if (f) sessionStorage.setItem(fkey, String(f.scrollTop));
});
const saved = sessionStorage.getItem(fkey);
if (saved !== null) {
  // before load the grid is half-laid-out and the clamp eats the
  // saved position
  addEventListener('load', () => {
    const f = document.querySelector('.board');
    if (f) f.scrollTop = Number(saved);
  });
}

// the return path: rendered math and code carry their author bytes in
// data-raw; a click on a formula copies them whole, and code blocks
// get a corner button - clicks inside code stay free to anchor partial
// selections. Nothing is sent anywhere; the clipboard is the only exit.
function sacCopy(text, done) {
  // the legacy door: plain http off localhost has no async clipboard,
  // and some engines refuse writeText even in a gesture - the
  // synchronous path still works there
  const legacy = () => {
    const t = document.createElement('textarea');
    t.value = text;
    t.setAttribute('readonly', '');
    t.style.position = 'fixed';
    t.style.opacity = '0';
    document.body.appendChild(t);
    t.select();
    const ok = document.execCommand('copy');
    t.remove();
    done(ok);
  };
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(text).then(() => done(true), legacy);
  else legacy();
}
document.addEventListener('click', (e) => {
  const m = e.target.closest && e.target.closest('math[data-raw]');
  if (!m) return;
  sacCopy(m.getAttribute('data-raw'), (ok) => {
    m.classList.add(ok ? 'copied' : 'copyfail');
    setTimeout(() => { m.classList.remove('copied'); m.classList.remove('copyfail'); }, 900);
  });
});
document.querySelectorAll('pre[data-raw]').forEach((pre) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'copybtn';
  b.title = 'copy code';
  b.setAttribute('aria-label', 'copy code');
  b.textContent = 'copy';
  b.addEventListener('click', () => {
    sacCopy(pre.getAttribute('data-raw'), (ok) => {
      b.textContent = ok ? 'ok' : 'err';
      setTimeout(() => { b.textContent = 'copy'; }, 900);
    });
  });
  pre.appendChild(b);
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
.mono, .nseq, .schip, .xd, .xmeta, .tmeta, .addrline, .gapdiv, .fsect, .xk, .tstate {
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
.hgroup { display: flex; gap: 12px; align-items: baseline; }
header .rname {
  font: 500 11px "JetBrains Mono", ui-monospace, monospace;
  color: #6d6562;
}

/* full-height frame: (rail | thread) under the mode line */
.if { height: calc(100vh - 41px); }

/* the thread track is minmax(0,1fr) so wide math scrolls in its block,
   never widens the frame */
.cols { display: grid; min-height: 0; height: 100%; }
.cols.withboard { grid-template-columns: 252px minmax(0, 1fr); }
.board { background: #1b1918; overflow-y: auto; padding: 10px 8px; }
.frow {
  display: flex; flex-wrap: nowrap; gap: 7px; align-items: center;
  padding: 4px 8px; border-radius: 3px; color: inherit;
}
.frow:hover { background: #292624; }
.frow.sel { background: #292624; }
.frow.sel .fname { color: #e8e2dd; }
.fid { color: #dac09a; font-size: 11.5px; white-space: nowrap; flex-shrink: 0; }
.schip { white-space: nowrap; flex-shrink: 0; font: 500 9.5px/1.6 "JetBrains Mono", ui-monospace, monospace; letter-spacing: .06em; }
.fname { flex: 1; min-width: 0; font-size: 12.5px; color: #d4ceca; overflow-wrap: anywhere; }
.fsect { font: 600 9.5px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .16em; color: #6d6562; margin: 16px 8px 4px; }
.fgate-empty { padding: 4px 8px; font-size: 11.5px; color: #4a4543; }
.nempty { font-size: 11px; color: #4a4543; padding: 1px 6px 2px; }
.frow.closed .fname { color: #6d6562; }
.frow.closed:hover .fname { color: #b3aca6; }

/* thread panel: head / scroll / compose */
.tpanel { display: grid; grid-template-rows: auto 1fr auto; min-height: 0; min-width: 0; }
.thead { padding: 14px 18px 10px; }
.ttitle { font: 800 16px/1.3 "Noto Sans", system-ui, sans-serif; color: #e8e2dd; margin: 4px 0 3px; overflow-wrap: anywhere; }
.tmeta { font-size: 11.5px; color: #6d6562; }
.tstate { font: 600 10.5px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .12em; }

#thread { overflow-y: auto; min-height: 0; padding: 4px 0 12px; }
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
.nbody { font-size: 13.5px; line-height: 1.55; color: #d4ceca; overflow-wrap: anywhere; }
/* markdown children: restrained to the thread's scale */
.nbody p { margin: 0 0 .45em; }
.nbody > :last-child { margin-bottom: 0; }
.nbody h1, .nbody h2, .nbody h3, .nbody h4, .nbody h5, .nbody h6 {
  font: 650 14px/1.35 "Noto Sans", system-ui, sans-serif; color: #e8e2dd; margin: .8em 0 .35em;
}
.nbody h1 { font-size: 16px; }
.nbody h2 { font-size: 15px; }
.nbody h1:first-child, .nbody h2:first-child, .nbody h3:first-child,
.nbody h4:first-child, .nbody h5:first-child, .nbody h6:first-child { margin-top: 0; }
.nbody ul, .nbody ol { margin: .35em 0; padding-left: 1.5em; }
.nbody li { margin: .12em 0; }
.nbody li > p { margin: .12em 0; }
.nbody blockquote {
  margin: .45em 0; padding: .05em .9em; border-left: 2px solid #2e2a28; color: #b3aca6;
}
.nbody code {
  font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px; background: #1e1c1a; border-radius: 3px; padding: .1em .35em;
}
.nbody pre { margin: .45em 0; padding: 7px 10px; background: #1e1c1a; border-radius: 4px; overflow-x: auto; position: relative; }
.nbody pre code { background: none; padding: 0; font-size: 12px; }
math[data-raw] { cursor: copy; }
math.copied { background: #2c2825; border-radius: 3px; }
math.copyfail { outline: 1px dashed #b08a8a; }

/* display math: its own centered block, long equations scroll inside */
.dmath { display: block; overflow-x: auto; margin: .6em 0; }
.dmath math[display="block"] {
  width: max-content; margin-inline: auto;
  max-width: none; text-align: left;
}
.copybtn {
  position: absolute; top: 4px; right: 4px;
  padding: 1px 6px; border: 1px solid #3a3531; border-radius: 4px;
  background: #262220; color: #a39c95;
  font: 500 11px/1.4 "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
  cursor: copy; opacity: 0;
}
pre:hover .copybtn, .copybtn:focus-visible { opacity: 1; }
.nbody table { border-collapse: collapse; margin: .45em 0; font-size: 12.5px; }
.nbody th, .nbody td { border: 1px solid #2e2a28; padding: 3px 9px; }
.nbody th { color: #b3aca6; background: #1e1c1a; }
.nbody hr { border: none; border-top: 1px solid #2e2a28; margin: .9em 0; }
.nbody math { font-size: 106%; }
.mref { font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 11px; vertical-align: super; line-height: 1; }
.mrefa { display: inline-flex; align-items: flex-end; gap: 1px; vertical-align: top; }
.mrefa .mref { vertical-align: bottom; }
.mref.dead { color: #6d6562; }

/* artifacts: figure at position, card for mentions */
.arow { padding-top: 2px; }
.aimg { display: block; max-width: min(480px, 100%); max-height: 360px; border-radius: 4px; margin: 4px 0 2px; }
.afull { display: block; }
.afile { color: #a2c3c4; font-size: 12.5px; display: inline-block; margin: 4px 0 2px; }
.acap { font-size: 11.5px; color: #6d6562; overflow-wrap: anywhere; }
.arow .acap { display: inline-block; margin-top: 2px; }
.acard {
  display: inline-flex; flex-direction: column; align-items: flex-start; gap: 2px;
  vertical-align: top; margin: 2px 6px 2px 0; padding: 4px;
  background: #1e1c1a; border-radius: 6px; color: inherit;
}
.acard:hover { background: #292624; }
.acard img { max-width: 180px; max-height: 120px; border-radius: 3px; display: block; }
.acard.miss img { display: none; }
.xg .acard, .ntg .acard { background: #100f0e; }

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
.agrp { margin: 4px 18px 6px 16px; background: #1e1c1a; border-radius: 6px; padding: 3px 0; }
.agrp .nrow2 { padding-left: 14px; padding-right: 14px; }

/* receipt and judgment: quiet gold facts */
.receipt { margin: 4px 18px 6px 16px; background: #1e1c1a; border-radius: 6px; padding: 7px 12px; }
.receipt .nbody { color: #b3aca6; margin-top: 2px; }
.nrow2 form.jform { margin: 6px 0 2px; }
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

/* the revise reveal: a whisper door on every live comment */
.revise { margin-top: 3px; }
/* the tombstone: a withdrawn note's one line, quieter than a body */
.nbody.tomb { color: #6d6562; font-style: italic; }
.revise summary {
  cursor: pointer; color: #4a4543; width: fit-content;
  font: 500 10px "JetBrains Mono", ui-monospace, monospace; letter-spacing: .08em;
}
.revise summary:hover { color: #6d6562; }
.rform { margin-top: 4px; }
.rform textarea {
  width: 100%; background: #100f0e; color: #d4ceca;
  border: 1px solid #100f0e; border-radius: 4px;
  font: 500 12.5px "Noto Sans", system-ui, sans-serif; padding: 6px 9px; resize: vertical;
}
.rform textarea:focus { outline: none; border-color: #2e2a28; caret-color: #c4a6a8; }
.rform .jbtns { display: flex; gap: 8px; align-items: flex-end; margin-top: 6px; }

/* narrow: the rail collapses behind the mode-line toggle */
@media (max-width: 1100px) {
  .fbtn { display: inline-block; }
  .cols.withboard { grid-template-columns: minmax(0, 1fr); }
  .board {
    position: fixed; left: 0; top: 41px; bottom: 0; width: 280px; z-index: 40;
    transform: translateX(-102%); transition: transform .15s ease;
    box-shadow: 8px 0 24px rgba(0,0,0,.6);
  }
  #nav:checked ~ .if .board { transform: none; }
}
"#;

#[cfg(test)]
mod tests {

    use super::*;
    use crate::CommentId;
    use crate::CommentKind;
    use crate::Prose;
    use crate::events::Event;
    use crate::store::{Context, Record, Tier, World};
    use crate::types::actor::ActorName;
    use crate::views::{board, closed_tasks, show_view, thread_view};
    use std::sync::OnceLock;

    fn human() -> &'static Context {
        static H: OnceLock<Context> = OnceLock::new();
        H.get_or_init(|| Context {
            actor: ActorName::new("jerry".into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        })
    }

    fn agent() -> &'static Context {
        static A: OnceLock<Context> = OnceLock::new();
        A.get_or_init(|| Context {
            actor: ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
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

    fn note(body: &str) -> CommentKind {
        CommentKind::Note {
            body: Prose::new(body.into()).unwrap(),
        }
    }

    fn note_on(
        seq: usize,
        at: u64,
        ctx: &Context,
        home: TaskId,
        body: &str,
        refs: Vec<RecordId>,
    ) -> Record {
        record(
            seq,
            at,
            ctx,
            Event::Commented {
                home,
                kind: note(body),
                references: refs,
            },
        )
    }

    fn offer_kind(body: &str) -> CommentKind {
        CommentKind::Offer {
            body: Prose::new(body.into()).unwrap(),
        }
    }

    // ---- the @-compiler's equivalence classes ----

    /// Three comments born at seq 2, 4, 6: dense c-0, c-1, c-2.
    fn cite_world() -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(2, 2, human(), TaskId(0), "the finding", Vec::new()),
            note_on(4, 4, agent(), TaskId(0), "the answer", Vec::new()),
            note_on(6, 6, human(), TaskId(0), "the verdict", Vec::new()),
        ])
        .unwrap()
    }

    #[test]
    fn bare_body_passes_through() {
        let world = cite_world();
        let a = compile("look at t-1 and c-2", TaskId(7), &world);
        assert_eq!(a.body, "look at t-1 and c-2");
        assert_eq!(a.home, TaskId(7));
        assert!(a.references.is_empty());
    }

    #[test]
    fn comment_token_cites_inline() {
        let world = cite_world();
        let a = compile("saw @c-2, which settles it", TaskId(7), &world);
        assert_eq!(a.body, "saw [^1], which settles it");
        assert_eq!(a.home, TaskId(7));
        assert_eq!(a.references, vec![RecordId(6)]);
        // repeated tokens are separate markers, repeated markers allowed
        let a = compile("@c-0 then @c-1 then @c-0", TaskId(7), &world);
        assert_eq!(a.body, "[^1] then [^2] then [^3]");
        assert_eq!(a.references, vec![RecordId(2), RecordId(4), RecordId(2)]);
    }

    #[test]
    fn task_token_rehomes() {
        let world = cite_world();
        let a = compile("@t-3 belongs there", TaskId(7), &world);
        assert_eq!(a.body, "belongs there");
        assert_eq!(a.home, TaskId(3));
        assert!(a.references.is_empty());
    }

    #[test]
    fn first_home_token_wins_and_the_rest_is_prose() {
        let world = cite_world();
        let a = compile("@t-1 then @t-2", TaskId(7), &world);
        assert_eq!(a.body, "then @t-2");
        assert_eq!(a.home, TaskId(1));
    }

    #[test]
    fn tokens_must_stand_alone() {
        let world = cite_world();
        let a = compile("mail me@agent now", TaskId(7), &world);
        assert_eq!(a.body, "mail me@agent now");
        assert_eq!(a.home, TaskId(7));
        let a = compile("see x@t-1", TaskId(7), &world);
        assert_eq!(a.body, "see x@t-1");
        assert_eq!(a.home, TaskId(7));
    }

    #[test]
    fn partial_tokens_stay_prose() {
        let world = cite_world();
        for body in ["@agentx", "@c-1x", "@t-", "@c- 5", "@agent-3", "@c-x"] {
            let a = compile(body, TaskId(7), &world);
            assert_eq!(a.body, body, "{body}");
            assert_eq!(a.home, TaskId(7), "{body}");
            assert!(a.references.is_empty(), "{body}");
        }
    }

    /// A cite the fold cannot resolve stays the words it was: no
    /// invented edges, no dangling markers.
    #[test]
    fn an_unresolvable_cite_stays_prose() {
        let world = cite_world();
        let a = compile("as they said @c-9", TaskId(7), &world);
        assert_eq!(a.body, "as they said @c-9");
        assert!(a.references.is_empty());
    }

    #[test]
    fn whitespace_is_tidy_after_a_consumed_home_token() {
        let world = cite_world();
        assert_eq!(
            compile("  @t-3   spaced  ", TaskId(7), &world).body,
            "spaced"
        );
        assert_eq!(compile("@t-2\ndo it", TaskId(7), &world).body, "do it");
    }

    // ---- the frame ----

    /// t-0 concluded; t-1 conversing; t-2 offered. Comments born at
    /// seq 2,3,5,6,7,10 carry the public numbers c-0..c-5.
    fn fixture() -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("ship it".into()).unwrap(),
                },
            ),
            record(
                1,
                10,
                human(),
                Event::TaskClaimed {
                    id: TaskId(0),
                    incarnation: None,
                },
            ),
            record(
                2,
                20,
                agent(),
                Event::Commented {
                    home: TaskId(0),
                    kind: offer_kind("suite green <34>"),
                    references: Vec::new(),
                },
            ),
            record(
                3,
                30,
                human(),
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Accept {
                        offer: CommentId(RecordId(2)),
                    },
                    references: Vec::new(),
                },
            ),
            record(
                4,
                40,
                human(),
                Event::TaskCreated {
                    name: Prose::new("build <foo>".into()).unwrap(),
                },
            ),
            note_on(
                5,
                3_140 * 3600,
                human(),
                TaskId(1),
                "please build it, per c-0",
                Vec::new(),
            ),
            note_on(
                6,
                3_160 * 3600,
                agent(),
                TaskId(1),
                "reply body",
                Vec::new(),
            ),
            note_on(
                7,
                3_200 * 3600,
                human(),
                TaskId(1),
                "lonely note",
                Vec::new(),
            ),
            record(
                8,
                40,
                human(),
                Event::TaskCreated {
                    name: Prose::new("adrift work".into()).unwrap(),
                },
            ),
            record(
                9,
                100 * 3600,
                human(),
                Event::TaskClaimed {
                    id: TaskId(2),
                    incarnation: None,
                },
            ),
            record(
                10,
                110 * 3600,
                agent(),
                Event::Commented {
                    home: TaskId(2),
                    kind: offer_kind("awaiting its consumer"),
                    references: Vec::new(),
                },
            ),
        ])
        .unwrap()
    }

    fn focus_of(world: &World, n: usize) -> Focus {
        Focus {
            show: show_view(world, TaskId(n)).unwrap(),
            thread: thread_view(world, TaskId(n)).unwrap(),
        }
    }

    fn console_of(world: &World, focus: Option<Focus>) -> Console {
        Console {
            board: board(world),
            closed: closed_tasks(world),
            focus,
            form: FormState::default(),
            store: ArtifactStore::default(),
            repo_name: None,
        }
    }

    #[test]
    fn an_offered_task_renders_its_decision_doors() {
        let world = fixture();
        let html = thread_section(
            &focus_of(&world, 2),
            &Default::default(),
            &ArtifactStore::default(),
        );
        // one form, reject the action, accept on formaction
        assert!(
            html.contains("<form class=\"jform\" method=\"post\" action=\"/t/2/reject\">"),
            "{html}"
        );
        assert!(html.contains("formaction=\"/t/2/accept\""), "{html}");
        assert!(html.contains("name=\"offer\" value=\"10\""), "{html}");
        assert!(html.contains("placeholder=\"grounds"), "{html}");
        // the doors ride the offer's own row
        let offer_row = html
            .split("id=\"c-5\"")
            .nth(1)
            .expect("the offer row renders");
        assert!(offer_row.contains("/t/2/reject"), "{offer_row}");
        // exactly one decision form, one shared actor field in it
        let form_html = html
            .split("class=\"jform\"")
            .nth(1)
            .expect("the doors render");
        assert_eq!(
            form_html.split("name=\"who\"").count() - 1,
            1,
            "{form_html}"
        );
    }

    #[test]
    fn a_refused_decision_renders_its_reason_beside_the_doors() {
        let world = fixture();
        let form = FormState {
            judge_error: Some("a rejection needs grounds".into()),
            note: "kept wording".into(),
            who: "jerry".into(),
            ..Default::default()
        };
        let html = thread_section(&focus_of(&world, 2), &form, &ArtifactStore::default());
        assert!(html.contains("a rejection needs grounds"), "{html}");
        // the refused grounds stay editable in the textarea
        assert!(html.contains(">kept wording</textarea>"), "{html}");
    }

    #[test]
    fn a_concluded_task_renders_no_decision_doors() {
        let world = fixture();
        let html = thread_section(
            &focus_of(&world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert!(!html.contains("/accept\""), "{html}");
        assert!(!html.contains("/reject\""), "{html}");
        // the offer's prose renders as the account that concluded
        assert!(html.contains("suite green &lt;34&gt;"), "{html}");
    }

    #[test]
    fn a_cancelled_task_renders_its_explanation() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("stop it".into()).unwrap(),
                },
            ),
            record(
                1,
                1,
                human(),
                Event::TaskCancelled {
                    id: TaskId(0),
                    explanation: Prose::new("the funding dissolved".into()).unwrap(),
                },
            ),
        ])
        .unwrap();
        let html = thread_section(
            &focus_of(&world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert!(html.contains("EXPLANATION"), "{html}");
        assert!(html.contains("the funding dissolved"), "{html}");
    }

    #[test]
    fn the_head_names_state_title_and_id() {
        let world = fixture();
        let page = page(&console_of(&world, Some(focus_of(&world, 1))));
        assert!(page.contains(">* OPEN<"), "{page}");
        assert!(page.contains("build &lt;foo&gt;"), "{page}");
        assert!(
            page.contains("<div class=\"tmeta mono\">t-1</div>"),
            "{page}"
        );
    }

    #[test]
    fn the_named_page_names_title_header_and_favicon() {
        let world = fixture();
        let mut c = console_of(&world, None);
        c.repo_name = Some(RepoName {
            basename: "saccade".into(),
            hue: 210,
        });
        let page = page(&c);
        assert!(page.contains("<title>saccade | saccade</title>"), "{page}");
        assert!(
            page.contains("<span class=\"rname\">saccade</span>"),
            "{page}"
        );
        assert!(page.contains("data:image/svg+xml"), "{page}");
    }

    #[test]
    fn the_unnamed_page_keeps_the_chrome_of_before() {
        let world = fixture();
        let page = page(&console_of(&world, None));
        assert!(page.contains("SACCADE | CONSOLE"), "{page}");
        assert!(!page.contains("class=\"rname\""), "{page}");
    }

    #[test]
    fn the_favicon_hue_follows_the_root_hash() {
        let root = std::path::Path::new("/tmp/some-repo-name");
        let name = RepoName::of(root);
        assert_eq!(name.basename, "some-repo-name");
        assert_ne!(name.hue, 0);
        assert!(favicon_link(name.hue).contains("data:image/svg+xml"));
    }

    #[test]
    fn closed_tasks_render_in_the_rail_with_state_colors() {
        let world = fixture();
        let page = page(&console_of(&world, None));
        assert!(page.contains("CLOSED"), "{page}");
        assert!(page.contains(">CONCLUDED<"), "{page}");
        assert!(page.contains("ship it"), "{page}");
    }

    #[test]
    fn the_compose_dock_holds_the_grammar() {
        let page = compose_section(3, &Default::default());
        assert!(page.contains("action=\"/compose\""), "{page}");
        assert!(page.contains("name=\"task\" value=\"3\""), "{page}");
        assert!(!page.contains("@agent"), "{page}");
    }

    #[test]
    fn the_compose_error_renders_legible_in_its_dock() {
        let form = FormState {
            error: Some("a comment needs words".into()),
            draft: "the draft".into(),
            ..Default::default()
        };
        let html = compose_section(3, &form);
        assert!(html.contains("a comment needs words"), "{html}");
        assert!(html.contains("the draft"), "{html}");
    }

    #[test]
    fn the_revise_reveal_renders_on_every_prose_carrying_comment() {
        let world = fixture();
        let html = thread_section(
            &focus_of(&world, 1),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert_eq!(
            html.matches("<details class=\"revise\">").count(),
            3,
            "{html}"
        );
        // the bodyless accept carries no reveal: t-0's accept has none
        let html = thread_section(
            &focus_of(&world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        );
        // the offer reveals; the accept row is bodyless
        assert_eq!(
            html.matches("<details class=\"revise\">").count(),
            1,
            "{html}"
        );
    }

    #[test]
    fn a_revised_comment_renders_its_mark() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(1, 10, agent(), TaskId(0), "first take", Vec::new()),
            record(
                2,
                20,
                human(),
                Event::CommentRevised {
                    id: CommentId(RecordId(1)),
                    body: Prose::new("second take".into()).unwrap(),
                },
            ),
        ])
        .unwrap();
        let html = thread_section(
            &focus_of(&world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert!(html.contains("second take"), "{html}");
        assert!(html.contains("revised by jerry"), "{html}");
        assert!(!html.contains("first take"), "{html}");
    }

    #[test]
    fn a_withdrawn_comment_renders_its_tombstone() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(1, 10, agent(), TaskId(0), "hasty words", Vec::new()),
            record(
                2,
                20,
                human(),
                Event::CommentWithdrawn {
                    id: CommentId(RecordId(1)),
                    note: Prose::new("parked in the wrong place".into()).unwrap(),
                },
            ),
        ])
        .unwrap();
        let html = thread_section(
            &focus_of(&world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert!(html.contains("withdrawn by jerry"), "{html}");
        assert!(
            html.contains("withdrawn: parked in the wrong place"),
            "{html}"
        );
        assert!(!html.contains("hasty words"), "{html}");
    }

    #[test]
    fn an_unfocused_column_states_it() {
        let world = fixture();
        let page = page(&console_of(&world, None));
        assert!(page.contains("no task focused"), "{page}");
    }

    // ---- citations, markers, and the retired auto-link ----

    const PNG_HEAD: &[u8] = b"\x89PNG\r\n\x1a\nrest-of-the-bytes";

    /// A throwaway store holding the named contents, keyed by hash.
    fn store_of(contents: &[&[u8]], tag: &str) -> (ArtifactStore, std::path::PathBuf) {
        let dir = std::env::temp_dir()
            .join(format!("sac-web-store-{}", std::process::id()))
            .join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for bytes in contents {
            std::fs::write(dir.join(crate::ContentHash::of(bytes).as_str()), bytes).unwrap();
        }
        (ArtifactStore::at(dir.clone()), dir)
    }

    /// t-0 holds the figure, then an offer citing it through a marker,
    /// then its accept; t-1's note mentions ids in prose only.
    fn resolver_world() -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            record(
                1,
                1,
                human(),
                Event::ArtifactAdded {
                    root: TaskId(0),
                    artifact: crate::types::artifact::Artifact {
                        name: Prose::new("sweep figure".into()).unwrap(),
                        hash: crate::ContentHash::of(PNG_HEAD),
                    },
                },
            ),
            record(
                2,
                2,
                human(),
                Event::Commented {
                    home: TaskId(0),
                    kind: offer_kind("suite green, per the figure [^1]"),
                    references: vec![RecordId(1)],
                },
            ),
            record(
                3,
                3,
                human(),
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Accept {
                        offer: CommentId(RecordId(2)),
                    },
                    references: Vec::new(),
                },
            ),
            record(
                5,
                5,
                human(),
                Event::TaskCreated {
                    name: Prose::new("other work".into()).unwrap(),
                },
            ),
            note_on(
                6,
                6,
                human(),
                TaskId(1),
                "see c-0 and the figure - no links, just words; c-99 unresolved",
                Vec::new(),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn markers_resolve_to_links_and_cards_or_stay_text() {
        let world = resolver_world();
        let (store, dir) = store_of(&[PNG_HEAD], "png");
        let hash = crate::ContentHash::of(PNG_HEAD).as_str().to_string();

        // the citing offer: a marker names the artifact, the card rides
        let html = thread_section(&focus_of(&world, 0), &Default::default(), &store);
        assert!(
            html.contains("<a class=\"mref\" href=\"/t/0#a-1\">[^1]</a>"),
            "{html}"
        );
        assert!(
            html.contains(&format!(
                "<a class=\"acard\" href=\"/t/0#a-1\"><img src=\"/a/{hash}\" alt=\"sweep figure\"><span class=\"acap\">sweep figure</span></a>"
            )),
            "{html}"
        );

        // the foreign body: bare ids are prose, no edges, no links
        let html = thread_section(&focus_of(&world, 1), &Default::default(), &store);
        assert!(html.contains("see c-0 and the figure"), "{html}");
        assert!(!html.contains("c-0</a>"), "{html}");
        assert!(!html.contains("c-99</a>"), "{html}");

        // the artifact row labels its position as provenance, the
        // anchor stays the seq the hrefs cite
        let html = thread_section(&focus_of(&world, 0), &Default::default(), &store);
        let row = html
            .split("id=\"a-1\"")
            .nth(1)
            .expect("the artifact row renders");
        let row = row.split("</div>").next().unwrap();
        assert!(
            row.contains("<span class=\"nseq mono\">seq 1</span>"),
            "{row}"
        );
        assert!(!row.contains("c-1"), "{row}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A marker naming a comment links its home anchor; one naming a
    /// task birth links the thread; a marker with no mapping stays the
    /// words it was.
    #[test]
    fn markers_name_comments_tasks_and_stay_words_without_a_mapping() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(1, 1, human(), TaskId(0), "the finding", Vec::new()),
            note_on(
                2,
                2,
                human(),
                TaskId(0),
                "the comment [^1], the task [^2], the void [^3]",
                vec![RecordId(1), RecordId(0)],
            ),
        ])
        .unwrap();
        let (store, dir) = store_of(&[], "markers");
        let html = thread_section(&focus_of(&world, 0), &Default::default(), &store);
        assert!(
            html.contains("<a class=\"mref\" href=\"/t/0#c-0\">[^1]</a>"),
            "{html}"
        );
        assert!(
            html.contains("<a class=\"mref\" href=\"/t/0\">[^2]</a>"),
            "{html}"
        );
        // a marker the mapping cannot resolve stays literal words
        assert!(html.contains("the void [^3]"), "{html}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_resolver_escapes_before_it_links() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(
                1,
                1,
                human(),
                TaskId(0),
                "the <b>bold</b> verdict & [^1]",
                vec![RecordId(0)],
            ),
        ])
        .unwrap();
        let (store, dir) = store_of(&[], "escape");
        let html = thread_section(&focus_of(&world, 0), &Default::default(), &store);
        // the words escaped, and the marker still resolved
        assert!(
            html.contains(
                "the &lt;b&gt;bold&lt;/b&gt; verdict &amp; <a class=\"mref\" href=\"/t/0\">[^1]</a>"
            ),
            "{html}"
        );
        assert!(!html.contains("<b>"), "{html}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The use row names its source task and its binding's selection.
    #[test]
    fn a_use_row_names_its_source_and_bound_selection() {
        let world = World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("consume it".into()).unwrap(),
                },
            ),
            record(
                1,
                1,
                human(),
                Event::TaskCreated {
                    name: Prose::new("produce it".into()).unwrap(),
                },
            ),
            record(
                2,
                2,
                agent(),
                Event::Commented {
                    home: TaskId(1),
                    kind: offer_kind("suite green"),
                    references: Vec::new(),
                },
            ),
            record(
                3,
                3,
                agent(),
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Use {
                        source: TaskId(1),
                        reason: Prose::new("builds on the account".into()).unwrap(),
                        bound: None,
                    },
                    references: Vec::new(),
                },
            ),
            record(
                4,
                4,
                human(),
                Event::UseBound {
                    id: CommentId(RecordId(3)),
                    offer: CommentId(RecordId(2)),
                    grounds: Prose::new("the account that landed".into()).unwrap(),
                },
            ),
        ])
        .unwrap();
        let html = thread_section(
            &focus_of(&world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert!(html.contains("uses t-1"), "{html}");
        assert!(html.contains("bound c-0"), "{html}");
        // the grounds ride the binding's title, the record's words
        assert!(html.contains("the account that landed"), "{html}");
    }

    // ---- the prose renderer: markdown with math ----

    /// One comment whose body exercises the prose renderer's doors.
    fn prose_world(body: &str) -> World {
        World::replay(vec![
            record(
                0,
                0,
                human(),
                Event::TaskCreated {
                    name: Prose::new("real work".into()).unwrap(),
                },
            ),
            note_on(1, 1, human(), TaskId(0), body, Vec::new()),
        ])
        .unwrap()
    }

    fn prose_html_of(world: &World) -> String {
        thread_section(
            &focus_of(world, 0),
            &Default::default(),
            &ArtifactStore::default(),
        )
    }

    #[test]
    fn math_renders_to_inline_mathml_and_a_miss_keeps_the_source() {
        let html = prose_html_of(&prose_world(
            "the loss $x^2+1$ and a miss $\\notacommand{x}$ stay honest",
        ));
        assert!(html.contains("<math"), "{html}");
        assert!(html.contains("msup"), "{html}");
        // a parse that fails keeps its source bytes
        assert!(html.contains("$\\notacommand{x}$"), "{html}");
    }

    #[test]
    fn display_math_renders_a_block_and_a_miss_keeps_its_fences() {
        let html = prose_html_of(&prose_world("sets it out: $$a^2+b^2=c^2$$ done"));
        // display math is its own block, distinct from inline
        assert!(
            html.contains(
                "<span class=\"dmath\"><math data-raw=\"a^2+b^2=c^2\" title=\"a^2+b^2=c^2\" display=\"block\">"
            ),
            "{html}"
        );
        // a display miss keeps both fences and renders no block
        let html = prose_html_of(&prose_world("$$\\notacommand{x}$$"));
        assert!(html.contains("$$\\notacommand{x}$$"), "{html}");
        assert!(!html.contains("dmath"), "{html}");
    }

    #[test]
    fn aligned_derivations_render_inside_display_math() {
        let body = "$$\\begin{aligned}\\sigma \\odot w_S &= 0\\\\G &= X^\\top X\\end{aligned}$$";
        let html = prose_html_of(&prose_world(body));
        assert!(html.contains("<mtable displaystyle=\"true\""), "{html}");
        assert_eq!(html.matches("<mtr>").count(), 2, "{html}");
        assert!(html.contains("<mtd"), "{html}");
    }

    #[test]
    fn negative_space_after_a_scripted_group_renders() {
        let html = prose_html_of(&prose_world("$G_{SS}^{-1}\\!x$ holds"));
        assert!(
            html.contains("<mspace style=\"margin-left:-0.16666667em;\"/>"),
            "{html}"
        );
        // the author bytes ride the data-raw copy hook, not the text
        assert!(html.contains("title=\"G_{SS}^{-1}\\!x\""), "{html}");
        assert!(!html.contains("<p>$G"), "{html}");
    }

    #[test]
    fn markup_beyond_the_crate_vocabulary_stays_out() {
        // the allow-list, not the converter, decides what passes
        assert!(crate_authored("<math><mi>x</mi></math>"));
        assert!(!crate_authored(
            "<math><mi>x</mi><script>alert(1)</script></math>"
        ));
        assert!(!crate_authored("<math><foreignobject/></math>"));
    }

    #[test]
    fn a_refused_revise_renders_open_at_its_row_with_the_draft_kept() {
        let world = fixture();
        // the refused row carries its public number, c-3
        let form = FormState {
            revise: Some(ReviseForm {
                seq: 3,
                draft: "the refused wording".into(),
                error: "a comment needs words".into(),
            }),
            ..Default::default()
        };
        let html = thread_section(&focus_of(&world, 1), &form, &ArtifactStore::default());
        assert!(html.contains("<details class=\"revise\" open>"), "{html}");
        assert!(html.contains("a comment needs words"), "{html}");
        // the draft rides the textarea, never re-authored
        assert!(html.contains(">the refused wording</textarea>"), "{html}");
        // other rows keep their closed reveals
        let html = thread_section(
            &focus_of(&world, 1),
            &Default::default(),
            &ArtifactStore::default(),
        );
        assert!(html.contains("<details class=\"revise\">"), "{html}");
        assert!(!html.contains("revise\" open"), "{html}");
    }

    #[test]
    fn code_spans_stay_verbatim() {
        let html = prose_html_of(&prose_world("run `sac show c-1 [^1]` now"));
        assert!(html.contains("<code>sac show c-1 [^1]</code>"), "{html}");
        assert!(!html.contains("<code>sac show <a"), "{html}");
    }

    #[test]
    fn code_blocks_carry_their_bytes_for_copy() {
        let html = prose_html_of(&prose_world("```rust\nlet x = 1;\n```"));
        assert!(
            html.contains("<pre data-raw=\"let x = 1;\n\"><code>let x = 1;\n</code></pre>"),
            "{html}"
        );
    }

    #[test]
    fn raw_html_never_passes() {
        let html = prose_html_of(&prose_world("<script>alert(1)</script> after"));
        assert!(html.contains("&lt;script&gt;"), "{html}");
        assert!(!html.contains("<script>"), "{html}");
    }

    #[test]
    fn links_render_only_on_http_s() {
        let html = prose_html_of(&prose_world(
            "[safe](https://example.com) and [evil](javascript:alert(1))",
        ));
        assert!(
            html.contains("<a href=\"https://example.com\" rel=\"noopener\">safe</a>"),
            "{html}"
        );
        assert!(!html.contains("href=\"javascript"), "{html}");
    }

    #[test]
    fn tables_render_aligned() {
        let html = prose_html_of(&prose_world(
            "| left | right |\n|-----|------:|\n| a   | 1    |",
        ));
        assert!(html.contains("<table>"), "{html}");
        assert!(html.contains("text-align:right"), "{html}");
    }
}
