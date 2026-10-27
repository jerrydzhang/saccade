//! The workbench read surface: the canonical folded projection and the
//! stateless draft preview, both derived straight from the store's own
//! fold and the console prose renderer. Nothing here writes, caches, or
//! infers; every html byte comes from the same renderer the record
//! already claims. Asset bytes are the committed single-binary bundle.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::db::{self, StoredRecord};
use crate::events::Event;
use crate::objects::comment::CommentId;
use crate::store::{RecordId, World};
use crate::views::{self, RefTarget, comment_line, show_view};
use crate::web::{self, ArtifactStore};
use crate::wire;

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
    let (recent_activity, recently_active_tasks) = home_facts(world, rows)?;
    Ok(json!({
        "tasks": tasks, "comments": comments, "cursor": rows.last().map(|r| r.seq),
        "recent_activity": recent_activity, "recently_active_tasks": recently_active_tasks,
    }))
}

/// One home activity entry's record facts: what the fold gives an
/// activity line to name. `comment` is the entry's public c-N when the
/// record anchors a contribution card; `artifact` the parked name.
struct ActivityEntry {
    seq: usize,
    at: u64,
    actor: String,
    tier: String,
    event: String,
    home: usize,
    comment: Option<usize>,
    artifact: Option<String>,
}

impl ActivityEntry {
    fn value(self) -> Value {
        json!({
            "seq": self.seq, "at": self.at, "actor": self.actor, "tier": self.tier,
            "event": self.event, "home": self.home,
            "comment": self.comment, "artifact": self.artifact,
        })
    }
}

/// The home read facts over the ratified boundary: every task and
/// comment record is recent activity - newest first - and the same
/// records order recently active tasks, one entry per task. Session,
/// workspace, and message-delivery records are bookkeeping, not home
/// activity; they name no entry here.
fn home_facts(
    world: &World,
    rows: &[StoredRecord],
) -> Result<(Vec<Value>, Vec<Value>), WorkbenchFail> {
    let mut entries: Vec<ActivityEntry> = Vec::new();
    // per task, the latest included record: seq order is the log's own
    // chronology, so the last write wins
    let mut latest: BTreeMap<usize, (usize, u64)> = BTreeMap::new();
    for row in rows {
        if let Some(entry) = activity_entry(world, row)? {
            latest.insert(entry.home, (row.seq, row.event_time));
            entries.push(entry);
        }
    }
    entries.reverse();
    let mut order: Vec<(usize, (usize, u64))> = latest.into_iter().collect();
    order.sort_by_key(|(_, (seq, _))| std::cmp::Reverse(*seq));
    let activity = entries.into_iter().map(ActivityEntry::value).collect();
    let active = order
        .into_iter()
        .map(|(task, (seq, at))| json!({"task": task, "seq": seq, "at": at}))
        .collect();
    Ok((activity, active))
}

/// The ratified inventory, one arm per included record kind; the kinds
/// outside it are Ok(None) bookkeeping. A record the fold cannot place
/// degrades loudly, like every projection in this file.
fn activity_entry(
    world: &World,
    row: &StoredRecord,
) -> Result<Option<ActivityEntry>, WorkbenchFail> {
    let event = wire::assemble(&row.kind, &row.payload)
        .map_err(|_| WorkbenchFail::Degraded(format!("record {} does not assemble", row.seq)))?;
    let resolved = |home: Option<usize>| {
        home.ok_or_else(|| {
            WorkbenchFail::Degraded(format!("record {} names no home task", row.seq))
        })
    };
    let (home, comment, artifact) = match event {
        Event::TaskCreated { .. } => (
            resolved(world.task_born_at(RecordId(row.seq)).map(|t| t.0))?,
            None,
            None,
        ),
        Event::TaskClaimed { id, .. }
        | Event::TaskCancelled { id, .. }
        | Event::TaskReopened { id, .. } => (resolved(Some(id.0))?, None, None),
        Event::Commented { home, .. } => (
            resolved(Some(home.0))?,
            world.comment_num(CommentId(RecordId(row.seq))),
            None,
        ),
        Event::CommentRevised { id, .. }
        | Event::CommentWithdrawn { id, .. }
        | Event::UseBound { id, .. } => {
            let context = world.comments.get(&id).ok_or_else(|| {
                WorkbenchFail::Degraded(format!("record {} names no resident comment", row.seq))
            })?;
            (
                resolved(Some(context.comment.home.0))?,
                world.comment_num(id),
                None,
            )
        }
        Event::ArtifactAdded { root, artifact } => (
            resolved(Some(root.0))?,
            None,
            Some(artifact.name.as_str().to_string()),
        ),
        // workspace, incarnation, and message records: excluded
        Event::WorkspaceEstablished { .. }
        | Event::TaskWorkspaceCheckpointed { .. }
        | Event::WorkspaceCreated { .. }
        | Event::WorkspaceChanged { .. }
        | Event::IncarnationRegistered { .. }
        | Event::MessageSent { .. }
        | Event::MessageSubmitted { .. } => return Ok(None),
    };
    Ok(Some(ActivityEntry {
        seq: row.seq,
        at: row.event_time,
        actor: row.actor.clone(),
        tier: row.tier.clone(),
        event: row.kind.clone(),
        home,
        comment,
        artifact,
    }))
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::Event;
    use crate::objects::comment::{CommentId, CommentKind};
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::message::MessageId;
    use crate::objects::task::TaskId;
    use crate::store::{Context, Record, Scope, Tier};
    use crate::types::artifact::{Artifact, ContentHash};
    use crate::types::pointers::{
        GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath,
    };
    use crate::types::prose::Prose;

    fn human(name: &str) -> Context {
        Context {
            actor: crate::ActorName::new(name.into()).unwrap(),
            tier: Tier::Human,
            scope: None,
        }
    }

    fn agent(scope: Option<Scope>) -> Context {
        Context {
            actor: crate::ActorName::new("pi".into()).unwrap(),
            tier: Tier::Agent,
            scope,
        }
    }

    fn scoped_incarnation(n: usize) -> Context {
        agent(Some(Scope {
            task: None,
            incarnation: Some(IncarnationId(RecordId(n))),
            workspace: None,
        }))
    }

    fn prose(text: &str) -> Prose {
        Prose::new(text.into()).unwrap()
    }

    /// Rows and world fold from one event list, so the fixture cannot
    /// drift between the log's shape and the fold's.
    fn fixture(events: Vec<(Context, Event)>) -> (World, Vec<StoredRecord>) {
        let mut rows = Vec::new();
        let mut records = Vec::new();
        for (n, (context, event)) in events.into_iter().enumerate() {
            let (kind, payload) = wire::disassemble(&event);
            records.push(Record {
                id: RecordId(n),
                timestamp: 1_700_000_000 + n as u64 * 60,
                context: context.clone(),
                event,
            });
            rows.push(StoredRecord {
                seq: n,
                event_time: 1_700_000_000 + n as u64 * 60,
                logged_time: 1_700_000_000 + n as u64 * 60,
                actor: context.actor.as_str().to_string(),
                tier: format!("{:?}", context.tier).to_lowercase(),
                kind,
                payload,
                scope: None,
            });
        }
        let world = World::replay(records).expect("the fixture folds");
        (world, rows)
    }

    fn seqs_of(activity: &[Value]) -> Vec<usize> {
        activity
            .iter()
            .map(|e| e["seq"].as_u64().unwrap() as usize)
            .collect()
    }

    /// The whole inventory in one story: every included kind lands,
    /// every machinery kind is interleaved and stays out.
    fn inventory() -> (World, Vec<StoredRecord>) {
        fixture(vec![
            (
                human("jerry"),
                Event::TaskCreated {
                    name: prose("migrate floop"),
                },
            ),
            (
                human("jerry"),
                Event::TaskCreated {
                    name: prose("other work"),
                },
            ),
            (
                human("jerry"),
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Note {
                        title: prose("the finding"),
                        body: prose("parked finding"),
                    },
                    references: Vec::new(),
                },
            ),
            (
                agent(None),
                Event::TaskClaimed {
                    id: TaskId(0),
                    incarnation: None,
                },
            ),
            (
                human("jerry"),
                Event::CommentRevised {
                    id: CommentId(RecordId(2)),
                    title: None,
                    body: Some(prose("the finding, reworded")),
                },
            ),
            (
                human("jerry"),
                Event::Commented {
                    home: TaskId(1),
                    kind: CommentKind::Offer {
                        title: prose("suite green"),
                        body: prose("suite green"),
                    },
                    references: Vec::new(),
                },
            ),
            (
                human("jerry"),
                Event::Commented {
                    home: TaskId(1),
                    kind: CommentKind::Accept {
                        offer: CommentId(RecordId(5)),
                    },
                    references: Vec::new(),
                },
            ),
            (
                Context::system(),
                Event::IncarnationRegistered {
                    native_id: NativeSessionId::new("a1b2c3d4-evil".into()).unwrap(),
                    saved_session: SessionLocator::new(
                        "/home/story/.pi/sessions/a1b2c3d4.jsonl".into(),
                    )
                    .unwrap(),
                },
            ),
            (
                scoped_incarnation(7),
                Event::MessageSent {
                    from: IncarnationId(RecordId(7)),
                    to: IncarnationId(RecordId(7)),
                    reply_to: None,
                    task: None,
                    body: prose("please look at the failing edge case"),
                },
            ),
            (
                Context::system(),
                Event::WorkspaceCreated {
                    base: GitCommit::new("abc123".into()).unwrap(),
                    branch: GitBranch::new("feature-abc123ab".into()).unwrap(),
                    worktree: WorktreePath::new("/repo/wt/feature-abc123ab".into()).unwrap(),
                },
            ),
            (
                human("jerry"),
                Event::ArtifactAdded {
                    root: TaskId(0),
                    artifact: Artifact {
                        name: prose("sweep figure"),
                        hash: ContentHash::of(b"figure bytes"),
                    },
                },
            ),
            (
                human("jerry"),
                Event::Commented {
                    home: TaskId(0),
                    kind: CommentKind::Use {
                        source: TaskId(1),
                        reason: prose("builds on the landed account"),
                        bound: None,
                    },
                    references: Vec::new(),
                },
            ),
            (
                human("jerry"),
                Event::UseBound {
                    id: CommentId(RecordId(11)),
                    offer: CommentId(RecordId(5)),
                    grounds: prose("the selected account holds"),
                },
            ),
            (
                human("jerry"),
                Event::CommentWithdrawn {
                    id: CommentId(RecordId(2)),
                    note: prose("parked in the wrong place"),
                },
            ),
            (
                human("jerry"),
                Event::TaskCancelled {
                    id: TaskId(0),
                    explanation: prose("superseded by other work"),
                },
            ),
            (
                human("jerry"),
                Event::TaskReopened {
                    id: TaskId(0),
                    explanation: prose("pursued again"),
                },
            ),
            (
                Context::system(),
                Event::WorkspaceEstablished {
                    task_id: TaskId(0),
                    base: GitCommit::new("abc123".into()).unwrap(),
                    branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                    worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
                },
            ),
            (
                Context::system(),
                Event::TaskWorkspaceCheckpointed {
                    task_id: TaskId(0),
                    checkpoint: GitCommit::new("def456".into()).unwrap(),
                },
            ),
            (
                Context::system(),
                Event::WorkspaceChanged {
                    incarnation: IncarnationId(RecordId(7)),
                    worktree: WorktreePath::new("/repo/wt/workspace-deadbeef".into()).unwrap(),
                },
            ),
            (
                scoped_incarnation(7),
                Event::MessageSubmitted {
                    id: MessageId(RecordId(8)),
                    by: IncarnationId(RecordId(7)),
                },
            ),
        ])
    }

    #[test]
    fn recent_activity_carries_the_included_inventory_newest_first() {
        let (world, rows) = inventory();
        let (activity, _) = home_facts(&world, &rows).expect("the home facts project");
        // seqs 7-9 and 16-19 are session, workspace, and message
        // bookkeeping; every other record is activity, newest first
        assert_eq!(
            seqs_of(&activity),
            vec![15, 14, 13, 12, 11, 10, 6, 5, 4, 3, 2, 1, 0]
        );
    }

    #[test]
    fn activity_entries_name_their_home_and_anchor() {
        let (world, rows) = inventory();
        let (activity, _) = home_facts(&world, &rows).expect("the home facts project");
        let by_seq = |seq: usize| {
            activity
                .iter()
                .find(|e| e["seq"].as_u64() == Some(seq as u64))
                .expect("the entry exists")
        };
        let created = by_seq(0);
        assert_eq!(created["event"].as_str(), Some("task_created"));
        assert_eq!(created["home"].as_u64(), Some(0));
        assert_eq!(created["comment"].as_u64(), None);
        let claimed = by_seq(3);
        assert_eq!(claimed["event"].as_str(), Some("task_claimed"));
        assert_eq!(claimed["home"].as_u64(), Some(0));
        let note = by_seq(2);
        assert_eq!(note["event"].as_str(), Some("commented"));
        assert_eq!(note["home"].as_u64(), Some(0));
        assert_eq!(note["comment"].as_u64(), Some(0));
        assert_eq!(note["actor"].as_str(), Some("jerry"));
        assert_eq!(note["tier"].as_str(), Some("human"));
        let revised = by_seq(4);
        assert_eq!(revised["event"].as_str(), Some("comment_revised"));
        assert_eq!(revised["home"].as_u64(), Some(0));
        assert_eq!(revised["comment"].as_u64(), Some(0));
        let offer = by_seq(5);
        assert_eq!(offer["home"].as_u64(), Some(1));
        assert_eq!(offer["comment"].as_u64(), Some(1));
        let accept = by_seq(6);
        assert_eq!(accept["home"].as_u64(), Some(1));
        assert_eq!(accept["comment"].as_u64(), Some(2));
        let artifact = by_seq(10);
        assert_eq!(artifact["event"].as_str(), Some("artifact_added"));
        assert_eq!(artifact["home"].as_u64(), Some(0));
        assert_eq!(artifact["artifact"].as_str(), Some("sweep figure"));
        let binding = by_seq(12);
        assert_eq!(binding["event"].as_str(), Some("use_bound"));
        assert_eq!(binding["home"].as_u64(), Some(0));
        assert_eq!(binding["comment"].as_u64(), Some(3));
        let withdrawal = by_seq(13);
        assert_eq!(withdrawal["event"].as_str(), Some("comment_withdrawn"));
        assert_eq!(withdrawal["home"].as_u64(), Some(0));
        assert_eq!(withdrawal["comment"].as_u64(), Some(0));
        let cancelled = by_seq(14);
        assert_eq!(cancelled["event"].as_str(), Some("task_cancelled"));
        let reopened = by_seq(15);
        assert_eq!(reopened["event"].as_str(), Some("task_reopened"));
        assert_eq!(reopened["home"].as_u64(), Some(0));
    }

    #[test]
    fn recently_active_tasks_follow_the_same_included_records() {
        let (world, rows) = inventory();
        let (_, active) = home_facts(&world, &rows).expect("the home facts project");
        // one entry per task; t-0's latest included record is the
        // reopening, t-1's the accept - the interleaved machinery
        // records order nothing
        let order: Vec<u64> = active.iter().map(|e| e["task"].as_u64().unwrap()).collect();
        assert_eq!(order, vec![0, 1]);
        assert_eq!(active[0]["seq"].as_u64(), Some(15));
        assert_eq!(active[1]["seq"].as_u64(), Some(6));
        assert_eq!(active[1]["at"].as_u64(), Some(1_700_000_000 + 6 * 60));
    }

    #[test]
    fn the_workbench_snapshot_carries_both_home_facts() {
        let (world, rows) = inventory();
        let dir = std::env::temp_dir().join(format!("sac-workbench-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = ArtifactStore::at(dir.clone());
        let value = snapshot(&world, &rows, &store).expect("the snapshot projects");
        assert_eq!(value["recent_activity"].as_array().unwrap().len(), 13);
        assert_eq!(value["recently_active_tasks"].as_array().unwrap().len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
