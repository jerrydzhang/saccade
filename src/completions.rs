//! The completions face: the shell scripts this binary emits, and the
//! hidden candidate read doors the scripts call. The door derives
//! one `value TAB description` line per record fact - path or id
//! first, branch, stage, state, or actor beside it - descriptive
//! facts only, exactly the rendering law the other faces hold.

pub const FISH: &str = include_str!("completions/sac.fish");

use crate::db::StoredRecord;
use crate::objects::incarnation::Incarnation;
use crate::objects::task::TaskState;
use crate::objects::workspace::WorkspaceStage;
use crate::store::{RecordId, World};

/// The hidden candidate doors: one kind per calling position in the
/// completion scripts. The workspace kinds differ only in the stage
/// that leads - clean and restore complete recorded paths only,
/// launch and switch take the plain birth order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Tasks,
    Incarnations,
    Workspaces,
    WorkspacesClean,
    WorkspacesRestore,
}

impl Kind {
    /// The kind name the CLI addresses, and the scripts spell.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Tasks => "tasks",
            Kind::Incarnations => "incarnations",
            Kind::Workspaces => "workspaces",
            Kind::WorkspacesClean => "workspaces-clean",
            Kind::WorkspacesRestore => "workspaces-restore",
        }
    }

    /// The candidates one per line, joined with newlines; the caller
    /// prints and its empty answer is the silent failure, never an
    /// error.
    pub fn candidates(self, world: &World, records: &[StoredRecord]) -> String {
        let lines = match self {
            Kind::Tasks => task_lines(world),
            Kind::Incarnations => incarnation_lines(world, records),
            Kind::Workspaces => workspace_lines(world, None),
            Kind::WorkspacesClean => workspace_lines(world, Some(WorkspaceStage::Available)),
            Kind::WorkspacesRestore => workspace_lines(world, Some(WorkspaceStage::Cleaned)),
        };
        lines.join("\n")
    }
}

fn task_lines(world: &World) -> Vec<String> {
    world
        .tasks
        .iter()
        .enumerate()
        .map(|(position, ctx)| {
            one(
                format!("t-{position}"),
                format!(
                    "{} ({})",
                    ctx.task.name.as_str(),
                    task_state(&ctx.task.state)
                ),
            )
        })
        .collect()
}

/// The state word the board prints, the vocabulary the candidates
/// carry - a cancelled task's explanation stays in its own record.
fn task_state(state: &TaskState) -> &'static str {
    match state {
        TaskState::Open => "open",
        TaskState::Offered => "offered",
        TaskState::Concluded => "concluded",
        TaskState::Cancelled(_) => "cancelled",
    }
}

fn incarnation_lines(world: &World, records: &[StoredRecord]) -> Vec<String> {
    world
        .incarnations
        .iter()
        .map(|(id, ctx)| {
            let number = world
                .incarnation_num(*id)
                .expect("the key comes from the map itself");
            let actor = birth_actor(records, id.0);
            let description = match &ctx.incarnation {
                Incarnation::Native {
                    native_id,
                    saved_session,
                } => format!(
                    "{actor} native session {}, saved {}",
                    native_id.as_str(),
                    saved_session.as_path().display()
                ),
                Incarnation::Established { task, workspace } => format!(
                    "{actor} established t-{}, workspace {}",
                    task.0,
                    workspace.as_path().display()
                ),
            };
            one(format!("i-{number}"), description)
        })
        .collect()
}

/// The actor the birth record carried: identity rides the record,
/// never a current process claim.
fn birth_actor(records: &[StoredRecord], birth: RecordId) -> &str {
    records
        .binary_search_by_key(&birth.0, |r| r.seq)
        .ok()
        .and_then(|i| records.get(i))
        .map(|r| r.actor.as_str())
        .unwrap_or("-")
}

fn workspace_lines(world: &World, lead: Option<WorkspaceStage>) -> Vec<String> {
    let mut rows: Vec<(WorkspaceStage, String)> = world
        .workspace_births()
        .into_iter()
        .filter_map(|birth| {
            let facts = world.workspace_facts(birth)?;
            Some((
                facts.stage,
                one(
                    facts.worktree.as_path().display().to_string(),
                    format!("{}, {}", facts.branch.as_str(), facts.stage.as_str()),
                ),
            ))
        })
        .collect();
    // a stable sort keeps birth order inside each stage group
    if let Some(lead) = lead {
        rows.sort_by_key(|(stage, _)| *stage != lead);
    }
    rows.into_iter().map(|(_, line)| line).collect()
}

/// One candidate line: the value, a tab, the description. Whitespace
/// runs collapse to single spaces so the line contract holds; the
/// bytes between stay the record's own.
fn one(value: String, description: String) -> String {
    format!("{}\t{}", collapse(&value), collapse(&description))
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::events::Event;
    use crate::objects::comment::CommentKind;
    use crate::objects::task::TaskId;
    use crate::store::{Context, Record, Tier};
    use crate::types::actor::ActorName;
    use crate::types::pointers::{
        GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath,
    };
    use crate::types::prose::Prose;

    fn ctx(actor: &str) -> Context {
        Context {
            actor: ActorName::new(actor.into()).unwrap(),
            tier: Tier::Agent,
            scope: None,
        }
    }

    fn record(seq: usize, context: &Context, event: Event) -> Record {
        Record {
            id: RecordId(seq),
            timestamp: seq as u64,
            context: context.clone(),
            event,
        }
    }

    fn stored(seq: usize, actor: &str) -> StoredRecord {
        StoredRecord {
            seq,
            event_time: seq as u64,
            logged_time: seq as u64,
            actor: actor.into(),
            tier: "agent".into(),
            kind: "event".into(),
            payload: "{}".into(),
            scope: None,
        }
    }

    /// One folded world over a small story: two tasks, two workspaces,
    /// one registered native session; the stored records behind them.
    fn world_with_records() -> (World, Vec<StoredRecord>) {
        let events = [
            Event::TaskCreated {
                name: Prose::new("implement floop".into()).unwrap(),
            },
            Event::TaskCreated {
                name: Prose::new("migrate gloom".into()).unwrap(),
            },
            Event::WorkspaceCreated {
                base: GitCommit::new("aaaa1111".into()).unwrap(),
                branch: GitBranch::new("workspace-floop".into()).unwrap(),
                worktree: WorktreePath::new("/home/j/worktrees/floop".into()).unwrap(),
            },
            Event::WorkspaceCreated {
                base: GitCommit::new("bbbb2222".into()).unwrap(),
                branch: GitBranch::new("workspace-gloom".into()).unwrap(),
                worktree: WorktreePath::new("/home/j/worktrees/gloom".into()).unwrap(),
            },
            Event::IncarnationRegistered {
                native_id: NativeSessionId::new("sess-7777".into()).unwrap(),
                saved_session: SessionLocator::new("/home/j/.pi/sessions/7777.json".into())
                    .unwrap(),
            },
        ];
        let mut world = World::new();
        let mut rows = Vec::new();
        for (seq, event) in events.into_iter().enumerate() {
            world = world
                .apply(record(seq, &ctx("pi"), event))
                .expect("the story folds");
            rows.push(stored(seq, "pi"));
        }
        (world, rows)
    }

    #[test]
    fn task_lines_carry_id_name_and_state() {
        let (mut world, mut rows) = world_with_records();
        let offer = record(
            5,
            &ctx("pi"),
            Event::Commented {
                home: TaskId(0),
                kind: CommentKind::Offer {
                    title: Prose::new("floop closes".into()).unwrap(),
                    body: Prose::new("done".into()).unwrap(),
                },
                references: Vec::new(),
            },
        );
        world = world.apply(offer).unwrap();
        rows.push(stored(5, "pi"));

        assert_eq!(
            Kind::Tasks.candidates(&world, &rows),
            "t-0\timplement floop (offered)\nt-1\tmigrate gloom (open)"
        );
    }

    #[test]
    fn a_whitespace_run_in_the_name_collapses_and_the_line_stays_two_fields() {
        let torn = Event::TaskCreated {
            name: Prose::new("name with\ttabs and\nnewlines".into()).unwrap(),
        };
        let mut world = World::new();
        world = world.apply(record(0, &ctx("pi"), torn)).unwrap();
        assert_eq!(
            Kind::Tasks.candidates(&world, &[]),
            "t-0\tname with tabs and newlines (open)"
        );
    }

    #[test]
    fn clean_leads_available_and_restore_leads_cleaned() {
        let (world, mut rows) = world_with_records();
        let fact = record(
            5,
            &ctx("jerry"),
            Event::WorkspaceCleaned {
                workspace: RecordId(3),
            },
        );
        let world = world.apply(fact).unwrap();
        rows.push(stored(5, "jerry"));

        // birth order on the plain door: floop, gloom
        assert_eq!(
            Kind::Workspaces.candidates(&world, &rows),
            "/home/j/worktrees/floop\tworkspace-floop, available\n\
             /home/j/worktrees/gloom\tworkspace-gloom, cleaned"
        );

        // clean leads Available, restore leads Cleaned; birth order
        // holds inside each group
        assert_eq!(
            Kind::WorkspacesClean.candidates(&world, &rows),
            "/home/j/worktrees/floop\tworkspace-floop, available\n\
             /home/j/worktrees/gloom\tworkspace-gloom, cleaned"
        );
        assert_eq!(
            Kind::WorkspacesRestore.candidates(&world, &rows),
            "/home/j/worktrees/gloom\tworkspace-gloom, cleaned\n\
             /home/j/worktrees/floop\tworkspace-floop, available"
        );
    }

    #[test]
    fn incarnation_lines_carry_birth_actor_and_what_the_record_holds() {
        let (world, rows) = world_with_records();
        assert_eq!(
            Kind::Incarnations.candidates(&world, &rows),
            "i-0\tpi native session sess-7777, saved /home/j/.pi/sessions/7777.json"
        );

        // a birth whose record the log rows lack still answers, minus
        // the actor it cannot name
        assert_eq!(
            Kind::Incarnations.candidates(&world, &[]),
            "i-0\t- native session sess-7777, saved /home/j/.pi/sessions/7777.json"
        );
    }

    #[test]
    fn the_kind_names_are_the_door_addresses_the_scripts_spell() {
        assert_eq!(Kind::Tasks.name(), "tasks");
        assert_eq!(Kind::Incarnations.name(), "incarnations");
        assert_eq!(Kind::Workspaces.name(), "workspaces");
        assert_eq!(Kind::WorkspacesClean.name(), "workspaces-clean");
        assert_eq!(Kind::WorkspacesRestore.name(), "workspaces-restore");
    }

    /// The emitted script is authored text: ASCII only, it invokes the
    /// tracker by PATH name, and it addresses the candidate doors by
    /// the kinds this module names.
    #[test]
    fn the_fish_script_stays_ascii_and_calls_the_doors() {
        assert!(
            FISH.is_ascii(),
            "the fish script is authored text and carries ASCII only"
        );
        assert!(!FISH.contains("/nix/store"), "no store path is baked");
        assert!(
            FISH.contains("sac complete $argv"),
            "the door call by PATH name"
        );
        for kind in [
            Kind::Tasks,
            Kind::Incarnations,
            Kind::Workspaces,
            Kind::WorkspacesClean,
            Kind::WorkspacesRestore,
        ] {
            assert!(
                FISH.contains(&format!("__saccade_candidates {}", kind.name())),
                "the script asks for {}",
                kind.name()
            );
        }
    }
}
