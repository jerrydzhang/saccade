//! The workbench read surface: the canonical folded projection and the
//! stateless draft preview, both derived straight from the store's own
//! fold and the console prose renderer. Nothing here writes, caches, or
//! infers; every html byte comes from the same renderer the record
//! already claims. Asset bytes are the committed single-binary bundle.

use serde::Deserialize;
use serde_json::{Value, json};

use crate::db::{self, StoredRecord};
use crate::store::{RecordId, World};
use crate::views::{self, RefTarget, comment_line, show_view};
use crate::web::{self, ArtifactStore};

/// The workbench assets, embedded at compile time from OUT_DIR - the
/// build script stages them there with parity proven per input, never
/// from committed bytes. `just web-build` regenerates from web/src.
pub mod assets {
    pub const INDEX: &str = include_str!(concat!(env!("OUT_DIR"), "/workbench/index.html"));
    pub const APP: &str = include_str!(concat!(env!("OUT_DIR"), "/workbench/app.js"));
    pub const STYLE: &str = include_str!(concat!(env!("OUT_DIR"), "/workbench/style.css"));
}

/// One read the workbench cannot serve honestly: a world that will not
/// fold, or a projection the fold cannot give.
#[derive(Debug)]
pub enum WorkbenchFail {
    Degraded(String),
    Db(String),
}

impl WorkbenchFail {
    pub fn reason(&self) -> &str {
        match self {
            WorkbenchFail::Degraded(reason) => reason,
            WorkbenchFail::Db(reason) => reason,
        }
    }
}

#[derive(Deserialize)]
pub struct PreviewRequest {
    pub body: String,
    pub references: Vec<usize>,
}

fn rich_range(
    raw: &str,
    from: usize,
    to: usize,
    kind: &str,
    refs: &[RefTarget],
    store: &ArtifactStore,
) -> Value {
    let rendered = web::prose_html(&raw[from..to], refs, store);
    let html = rendered
        .strip_prefix("<p>")
        .and_then(|s| s.strip_suffix("</p>\n"))
        .unwrap_or(&rendered);
    json!({
        "from": raw[..from].encode_utf16().count(),
        "to": raw[..to].encode_utf16().count(),
        "kind": kind,
        "html": html,
    })
}

fn rich_ranges(raw: &str, refs: &[RefTarget], store: &ArtifactStore) -> Vec<Value> {
    use pulldown_cmark::{Event as MdEvent, Options, Parser, Tag, TagEnd};
    let mut out = Vec::new();
    let mut table = None;
    for (event, range) in
        Parser::new_ext(raw, Options::ENABLE_TABLES | Options::ENABLE_MATH).into_offset_iter()
    {
        match event {
            MdEvent::Start(Tag::Table(_)) => table = Some(range.start),
            MdEvent::End(TagEnd::Table) => {
                if let Some(from) = table.take() {
                    out.push(rich_range(raw, from, range.end, "table", refs, store));
                }
            }
            MdEvent::InlineMath(_) | MdEvent::DisplayMath(_) if table.is_none() => {
                let value = rich_range(raw, range.start, range.end, "math", refs, store);
                if value["html"]
                    .as_str()
                    .is_some_and(|html| html.contains("<math"))
                {
                    out.push(value);
                }
            }
            _ => {}
        }
    }
    out
}

/// The one folded snapshot the workbench reads: tasks and comments with
/// the canonical rendered bodies, the citation edges the fold gives,
/// and the log cursor. A degraded fold refuses rather than projecting.
pub fn snapshot(
    world: &World,
    rows: &[StoredRecord],
    store: &ArtifactStore,
) -> Result<Value, WorkbenchFail> {
    let mut comments = Vec::new();
    for (id, context) in &world.comments {
        let line = comment_line(world, *id).ok_or_else(|| {
            WorkbenchFail::Degraded(format!("comment at record {:?} does not project", id))
        })?;
        let body = if line.withdrawn {
            String::new()
        } else {
            line.body.clone()
        };
        let html = if line.withdrawn {
            web::prose_html(
                line.withdrawal_note.as_deref().unwrap_or("withdrawn"),
                &[],
                store,
            )
        } else {
            web::prose_html(&body, &line.references, store)
        };
        let cited_by: Vec<usize> = world
            .comments
            .iter()
            .filter(|(_, other)| other.references.contains(&id.0))
            .filter_map(|(other, _)| world.comment_num(*other))
            .collect();
        let artifact_previews: Vec<Value> = line
            .references
            .iter()
            .enumerate()
            .filter_map(|(index, target)| {
                matches!(target, RefTarget::Artifact { .. }).then(|| {
                    json!({"index": index, "html": web::prose_html(&format!("[^{}]", index + 1), &line.references, store)})
                })
            })
            .collect();
        comments.push(json!({
            "id": format!("c-{}", line.num), "num": line.num, "seq": line.seq,
            "home": line.home.0, "actor": line.actor, "tier": line.tier,
            "kind": line.kind, "title": line.title, "body": body, "html": html,
            "rich_ranges": rich_ranges(&body, &line.references, store),
            "artifact_previews": artifact_previews,
            "born_at": line.born_at, "revised": line.revised, "reviser": line.reviser,
            "withdrawn": line.withdrawn, "withdrawal_note": line.withdrawal_note,
            "offer": line.offer, "source": line.source.map(|t| t.0),
            "binding": line.binding.map(|b| json!({"offer": b.offer, "grounds": b.grounds})),
            "references": context.references.iter().map(|r| r.0).collect::<Vec<_>>(),
            "cited_by": cited_by,
        }));
    }
    let mut tasks = Vec::new();
    for n in 0..world.tasks.len() {
        let view = show_view(world, crate::TaskId(n))
            .ok_or_else(|| WorkbenchFail::Degraded(format!("task t-{n} does not project")))?;
        tasks.push(json!({
            "id": n, "name": view.name, "state": view.state, "owner": view.owner,
            "explanation": view.explanation, "offer": view.offer_num,
        }));
    }
    Ok(json!({"tasks": tasks, "comments": comments, "cursor": rows.last().map(|r| r.seq)}))
}

/// The stateless draft preview: a pure render of the proposed body
/// against the draft's fixed references. Unknown references refuse;
/// nothing writes and nothing is remembered.
pub fn preview(
    loadout: &db::Loadout,
    world: &World,
    request: &PreviewRequest,
    store: &ArtifactStore,
) -> Result<Value, String> {
    let mut targets = Vec::new();
    for seq in &request.references {
        if !loadout.rows.iter().any(|row| row.seq == *seq) {
            return Err(format!("reference {seq} names no record"));
        }
        targets.push(views::ref_target(world, RecordId(*seq)).ok_or("unresolved reference")?);
    }
    Ok(json!({"html": web::prose_html(&request.body, &targets, store)}))
}
