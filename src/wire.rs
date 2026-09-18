use crate::events::Event;

use crate::store::Tier;

#[derive(Debug, PartialEq)]
pub enum ParseFail {
    UnknownKind(String),
    UnknownTier(String),
    Malformed { kind: String, detail: String },
}

pub fn tier_of(tier: &Tier) -> &'static str {
    match tier {
        Tier::Human => "human",
        Tier::Agent => "agent",
        Tier::System => "system",
    }
}

pub fn tier_from(s: &str) -> Result<Tier, ParseFail> {
    match s {
        "human" => Ok(Tier::Human),
        "agent" => Ok(Tier::Agent),
        "system" => Ok(Tier::System),
        _ => Err(ParseFail::UnknownTier(s.to_string())),
    }
}

/// Known kinds, anything else in a row is version skew, not corruption
const KINDS: [&str; 18] = [
    "task_created",
    "task_claimed",
    "task_done",
    "task_dropped",
    "task_released",
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "proposal_withdrawn",
    "commented",
    "incarnation_bound",
    "incarnation_prompt_accepted",
    "incarnation_prompt_rejected",
    "incarnation_settled",
    "record_produced_by",
    "task_workspace_created",
    "task_worktree_created",
    "task_workspace_checkpointed",
];

/// Split an event into its db columns. The on-disk shape is serde's
/// externally-tagged enum; the split happens at the tag.
pub fn disassemble(event: &Event) -> (String, String) {
    let tagged = serde_json::to_value(event).expect("in-memory events always serialize");
    let Some(mut entry) = tagged.as_object().map(|m| m.into_iter()) else {
        unreachable!("an event serializes to exactly one keyed object")
    };
    let (kind, payload) = entry
        .next()
        .expect("an event serializes to exactly one keyed object");
    (kind.to_owned(), payload.to_string())
}

pub fn assemble(kind: &str, payload: &str) -> Result<Event, ParseFail> {
    if !KINDS.contains(&kind) {
        return Err(ParseFail::UnknownKind(kind.to_string()));
    }
    let inner: serde_json::Value = serde_json::from_str(payload).map_err(|e| malformed(kind, e))?;
    let mut tagged = serde_json::Map::new();
    tagged.insert(kind.to_string(), inner);
    serde_json::from_value(serde_json::Value::Object(tagged)).map_err(|e| malformed(kind, e))
}

fn malformed(kind: &str, err: serde_json::Error) -> ParseFail {
    ParseFail::Malformed {
        kind: kind.to_string(),
        detail: err.to_string(),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objects::comment::{CommentId, Target};
    use crate::objects::task::TaskId;
    use crate::types::prose::Prose;

    #[test]
    fn tier_tokens_round_trip_all_variants() {
        for tier in [Tier::Human, Tier::Agent, Tier::System] {
            assert_eq!(tier_from(tier_of(&tier)).unwrap(), tier);
        }
    }
    use crate::store::RecordId;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

    #[test]
    fn every_event_kind_round_trips() {
        let samples = [
            Event::TaskCreated {
                name: Prose::new("implement foo".into()).unwrap(),
                parent_id: None,
            },
            Event::TaskCreated {
                name: Prose::new("child".into()).unwrap(),
                parent_id: Some(TaskId(0)),
            },
            Event::TaskClaimed { id: TaskId(0) },
            Event::TaskDone {
                id: TaskId(0),
                receipt: Prose::new("tests green".into()).unwrap(),
            },
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("scope covered elsewhere".into()).unwrap(),
            },
            Event::TaskDropped {
                id: TaskId(0),
                note: Prose::new("superseded by t-2".into()).unwrap(),
            },
            Event::TaskReleased {
                id: TaskId(0),
                note: Prose::new("run dead".into()).unwrap(),
            },
            Event::Commented {
                target: Target::Task(TaskId(0)),
                body: Prose::new("leaning sections, owner: jerry".into()).unwrap(),
                addressee: None,
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(6))),
                body: Prose::new("no - pure tree, here is why".into()).unwrap(),
                addressee: None,
            },
            Event::TaskWorkspaceCreated {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            Event::TaskWorktreeCreated {
                task_id: TaskId(0),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            Event::TaskWorkspaceCheckpointed {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("def456".into()).unwrap(),
            },
        ];

        for event in &samples {
            let (kind, payload) = disassemble(event);
            let back = assemble(&kind, &payload).unwrap_or_else(|e| panic!("{kind}: {e:?}"));
            assert_eq!(&back, event, "different: {kind}");
        }
    }

    #[test]
    fn unknown_kind_and_tier_are_version_skew_not_corruption() {
        assert!(matches!(
            assemble("task_zapped", "{}"),
            Err(ParseFail::UnknownKind(k)) if k == "task_zapped"
        ));
        assert!(matches!(
            tier_from("aliens"),
            Err(ParseFail::UnknownTier(t)) if t == "aliens"
        ));
    }

    #[test]
    fn malformed_payload_of_known_kind_is_loud() {
        assert!(matches!(
            assemble("task_done", "{"),
            Err(ParseFail::Malformed { .. })
        ));
        assert!(matches!(
            assemble("task_claimed", r#"{"id":"three"}"#),
            Err(ParseFail::Malformed { .. })
        ));
    }
}
