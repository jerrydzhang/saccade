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
const KINDS: [&str; 25] = [
    "task_created",
    "task_claimed",
    "task_done",
    "task_delivered",
    "task_accepted",
    "task_dropped",
    "task_released",
    "proposal_created",
    "proposal_accepted",
    "proposal_rejected",
    "proposal_withdrawn",
    "commented",
    "comment_revised",
    "demand_refused",
    "steer_forwarded",
    "artifact_added",
    "incarnation_bound",
    "incarnation_prompt_accepted",
    "incarnation_prompt_rejected",
    "incarnation_settled",
    "incarnation_cancelled",
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
    let payload = if kind == "commented" {
        migrate_commented(payload).map_err(|e| malformed(kind, e))?
    } else {
        payload.to_string()
    };
    let inner: serde_json::Value =
        serde_json::from_str(&payload).map_err(|e| malformed(kind, e))?;
    let mut tagged = serde_json::Map::new();
    tagged.insert(kind.to_string(), inner);
    serde_json::from_value(serde_json::Value::Object(tagged)).map_err(|e| malformed(kind, e))
}

/// Old logs carry the addressee field; the tagged union carries kind.
/// The map is the ratified migration: to:agent becomes a demand,
/// everything else a note.
fn migrate_commented(payload: &str) -> Result<String, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_str(payload)?;
    if let Some(object) = value.as_object_mut()
        && !object.contains_key("kind")
    {
        let kind = match object.get("addressee").and_then(|a| a.as_str()) {
            Some("agent") => "demand",
            _ => "note",
        };
        object.remove("addressee");
        object.insert("kind".into(), kind.into());
    }
    Ok(value.to_string())
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
    use crate::objects::comment::{CommentId, CommentKind, Target};
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::task::TaskId;
    use crate::types::prose::Prose;

    #[test]
    fn tier_tokens_round_trip_all_variants() {
        for tier in [Tier::Human, Tier::Agent, Tier::System] {
            assert_eq!(tier_from(tier_of(&tier)).unwrap(), tier);
        }
    }
    use crate::ContentHash;
    use crate::store::RecordId;
    use crate::types::artifact::Artifact;
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
            Event::TaskDelivered {
                id: TaskId(0),
                receipt: Prose::new("tests green".into()).unwrap(),
            },
            Event::TaskAccepted { id: TaskId(0) },
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
                kind: CommentKind::Note,
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(6))),
                body: Prose::new("no - pure tree, here is why".into()).unwrap(),
                kind: CommentKind::Demand,
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(6))),
                body: Prose::new("change of plan, do it this way".into()).unwrap(),
                kind: CommentKind::Steer,
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(6))),
                body: Prose::new("which way do you want it?".into()).unwrap(),
                kind: CommentKind::Ask,
            },
            Event::CommentRevised {
                id: CommentId(RecordId(6)),
                body: Prose::new("which way do you want it, exactly?".into()).unwrap(),
            },
            Event::SteerForwarded {
                steer: CommentId(RecordId(6)),
            },
            Event::ArtifactAdded {
                root: TaskId(0),
                artifact: Artifact {
                    name: Prose::new("sweep figure".into()).unwrap(),
                    hash: ContentHash::of(b"figure bytes"),
                },
            },
            Event::DemandRefused {
                demand: CommentId(RecordId(6)),
                reason: Prose::new("the worktree is a disk-only leftover".into()).unwrap(),
            },
            Event::TaskWorkspaceCreated {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
            },
            Event::IncarnationCancelled {
                id: IncarnationId(RecordId(0)),
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

    /// The ratified migration: old addressee payloads load unchanged —
    /// to:agent becomes a demand, everything else a note; a payload
    /// that already carries kind passes through untouched.
    #[test]
    fn old_addressee_payloads_load_as_their_kinds() {
        let agent = assemble(
            "commented",
            r#"{"target":{"task":0},"body":"build it","addressee":"agent"}"#,
        )
        .unwrap();
        assert!(
            matches!(
                &agent,
                Event::Commented {
                    kind: CommentKind::Demand,
                    ..
                }
            ),
            "{agent:?}"
        );
        for payload in [
            r#"{"target":{"task":0},"body":"just talking","addressee":"human"}"#.to_string(),
            r#"{"target":{"task":0},"body":"a note","addressee":null}"#.to_string(),
            r#"{"target":{"task":0},"body":"a bare note"}"#.to_string(),
        ] {
            let note = assemble("commented", &payload).unwrap();
            assert!(
                matches!(
                    &note,
                    Event::Commented {
                        kind: CommentKind::Note,
                        ..
                    }
                ),
                "{note:?}"
            );
        }
        // the new shape round-trips through the same door
        let (kind, payload) = disassemble(&agent);
        assert_eq!(kind, "commented");
        assert!(!payload.contains("addressee"), "{payload}");
        assert_eq!(&assemble(&kind, &payload).unwrap(), &agent);
    }
}
