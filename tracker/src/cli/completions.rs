//! The completions face: the shell scripts this binary emits, and the
//! hidden candidate read doors the scripts call. The door derives
//! one `value TAB description` line per record fact - path or id
//! first, branch, stage, state, or actor beside it - descriptive
//! facts only, exactly the rendering law the other faces hold.

use super::*;

pub const FISH: &str = include_str!("completions/sac.fish");

use saccade::db::StoredRecord;
use saccade::objects::incarnation::Incarnation;
use saccade::objects::task::TaskState;
use saccade::objects::workspace::WorkspaceStage;
use saccade::store::{RecordId, World};

/// The hidden candidate doors: one kind per calling position in the
/// completion scripts. The workspace kinds differ only in the stage
/// that leads - clean completes recorded paths only,
/// launch takes the plain birth order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Tasks,
    Incarnations,
    Workspaces,
    WorkspacesClean,
    Tags,
}

impl Kind {
    /// The kind name the CLI addresses, and the scripts spell. The
    /// vocabulary is test-pinned: the door itself reads the clap
    /// enum's kebab-case values.
    #[cfg(test)]
    pub(super) fn name(self) -> &'static str {
        match self {
            Kind::Tasks => "tasks",
            Kind::Incarnations => "incarnations",
            Kind::Workspaces => "workspaces",
            Kind::WorkspacesClean => "workspaces-clean",
            Kind::Tags => "tags",
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
            Kind::Tags => tag_lines(world),
        };
        lines.join("\n")
    }
}

pub(super) fn tag_lines(world: &World) -> Vec<String> {
    world
        .tags
        .values()
        .map(|tag| {
            one(
                tag.name.as_str().to_string(),
                tag.description.as_str().to_string(),
            )
        })
        .collect()
}

pub(super) fn task_lines(world: &World) -> Vec<String> {
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
pub(super) fn task_state(state: &TaskState) -> &'static str {
    match state {
        TaskState::Open => "open",
        TaskState::Offered => "offered",
        TaskState::Concluded => "concluded",
        TaskState::Cancelled(_) => "cancelled",
    }
}

pub(super) fn incarnation_lines(world: &World, records: &[StoredRecord]) -> Vec<String> {
    world
        .incarnations
        .iter()
        .map(|(id, ctx)| {
            let number = world
                .incarnation_num(*id)
                .expect("the key comes from the map itself");
            let actor = birth_actor(records, id.0);
            let Incarnation {
                native_id,
                saved_session,
                ..
            } = &ctx.incarnation;
            let saved = saved_session
                .as_ref()
                .map(|l| format!("saved {}", l.as_path().display()))
                .unwrap_or_else(|| "awaiting its first save".into());
            let description = format!("{actor} native session {}, {saved}", native_id.as_str());
            one(format!("i-{number}"), description)
        })
        .collect()
}

/// The actor the birth record carried: identity rides the record,
/// never a current process claim.
pub(super) fn birth_actor(records: &[StoredRecord], birth: RecordId) -> &str {
    records
        .binary_search_by_key(&birth.0, |r| r.seq)
        .ok()
        .and_then(|i| records.get(i))
        .map(|r| r.actor.as_str())
        .unwrap_or("-")
}

pub(super) fn workspace_lines(world: &World, lead: Option<WorkspaceStage>) -> Vec<String> {
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
pub(super) fn one(value: String, description: String) -> String {
    format!("{}\t{}", collapse(&value), collapse(&description))
}

pub(super) fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn complete_candidates(cli: &Cli, kind: CompleteKind) -> String {
    let fold = || -> Result<(World, Vec<db::StoredRecord>), Fail> {
        let destination = resolve_destination(cli)?;
        let loadout = loadout_of(&destination)?;
        let world = match loadout.state {
            LoadState::Full(world) => world,
            LoadState::Degraded(reason) => return Err(Fail::Degraded(reason)),
        };
        Ok((world, loadout.rows))
    };
    match fold() {
        Ok((world, rows)) => Kind::from(kind).candidates(&world, &rows),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use saccade::events::Event;
    use saccade::store::{Context, Record, Tier};
    use saccade::types::actor::ActorName;
    use saccade::types::pointers::{
        GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath,
    };
    use saccade::types::prose::Prose;

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
                tags: Vec::new(),
                name: Prose::new("implement floop".into()).unwrap(),
            },
            Event::TaskCreated {
                tags: Vec::new(),
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
                saved_session: Some(
                    SessionLocator::new("/home/j/.pi/sessions/7777.json".into()).unwrap(),
                ),
                address: None,
                adopted: false,
                harness: saccade::host::harness::Harness::Pi,
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
        let (world, rows) = world_with_records();
        assert_eq!(
            Kind::Tasks.candidates(&world, &rows),
            "t-0\timplement floop (open)\nt-1\tmigrate gloom (open)"
        );
    }

    #[test]
    fn a_whitespace_run_in_the_name_collapses_and_the_line_stays_two_fields() {
        let torn = Event::TaskCreated {
            tags: Vec::new(),
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
    fn clean_leads_available() {
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

        // clean leads Available; birth order holds inside each group
        assert_eq!(
            Kind::WorkspacesClean.candidates(&world, &rows),
            "/home/j/worktrees/floop\tworkspace-floop, available\n\
             /home/j/worktrees/gloom\tworkspace-gloom, cleaned"
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
        assert_eq!(Kind::Tags.name(), "tags");
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
            Kind::Tags,
        ] {
            assert!(
                FISH.contains(&format!("__saccade_candidates {}", kind.name())),
                "the script asks for {}",
                kind.name()
            );
        }
    }

    /// The sweep's flags complete exactly as the CLI defines them, on
    /// the clean verb only: --merged previews, --yes/-y acts on the
    /// preview, and the recorded-path candidates step aside once
    /// --merged is on the line - a path beside --merged is a parser
    /// refusal, never a completion.
    #[test]
    fn the_clean_sweep_flags_complete_on_the_clean_verb_only() {
        assert!(
            FISH.contains("__saccade_verb workspace clean' -l merged"),
            "--merged rides the clean verb: {FISH}"
        );
        let yes_line = FISH
            .lines()
            .find(|line| line.contains("-l yes -s y"))
            .unwrap_or_else(|| panic!("--yes/-y complete: {FISH}"));
        assert!(
            yes_line.contains("__saccade_verb workspace clean")
                && yes_line.contains("__saccade_seen_flag --merged"),
            "--yes acts only on the sweep, beside --merged: {yes_line}"
        );
        let paths_line = FISH
            .lines()
            .find(|line| line.contains("__saccade_candidates workspaces-clean"))
            .unwrap_or_else(|| panic!("the recorded-path candidates complete: {FISH}"));
        assert!(
            paths_line.contains("not __saccade_seen_flag --merged"),
            "the path candidates step aside under --merged: {paths_line}"
        );
    }
}
