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
const KINDS: [&str; 21] = [
    "task_created",
    "task_claimed",
    "task_cancelled",
    "task_reopened",
    "task_committed",
    "task_offered",
    "task_rejected",
    "task_accepted",
    "workspace_established",
    "task_workspace_checkpointed",
    "workspace_created",
    "workspace_changed",
    "workspace_cleaned",
    "workspace_restored",
    "root_space_bound",
    "root_space_cleared",
    "incarnation_registered",
    "saved_session_bound",
    "host_address_assigned",
    "message_sent",
    "message_submitted",
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
    use crate::objects::incarnation::IncarnationId;
    use crate::objects::message::MessageId;
    use crate::objects::task::TaskId;
    use crate::types::pointers::{NativeSessionId, SessionLocator};
    use crate::types::prose::Prose;

    #[test]
    fn tier_tokens_round_trip_all_variants() {
        for tier in [Tier::Human, Tier::Agent, Tier::System] {
            assert_eq!(tier_from(tier_of(&tier)).unwrap(), tier);
        }
    }
    use crate::harness::Harness;
    use crate::store::RecordId;
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};

    #[test]
    fn every_event_kind_round_trips() {
        let prose = || Prose::new("implement foo".into()).unwrap();
        let samples = [
            Event::TaskCreated { name: prose() },
            Event::TaskClaimed {
                id: TaskId(0),
                incarnation: None,
            },
            Event::TaskCancelled {
                id: TaskId(0),
                explanation: prose(),
            },
            Event::TaskReopened {
                id: TaskId(0),
                explanation: prose(),
            },
            Event::TaskCommitted {
                task: TaskId(0),
                commit: GitCommit::new("a".repeat(40)).unwrap(),
                parent: None,
                incorporations: vec![(TaskId(1), GitCommit::new("b".repeat(40)).unwrap())],
            },
            Event::TaskOffered {
                task: TaskId(0),
                commit: GitCommit::new("a".repeat(40)).unwrap(),
            },
            Event::TaskRejected {
                task: TaskId(0),
                grounds: prose(),
            },
            Event::TaskAccepted { task: TaskId(0) },
            Event::WorkspaceEstablished {
                task_id: TaskId(0),
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("saccade/t-0".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/t-0".into()).unwrap(),
            },
            Event::TaskWorkspaceCheckpointed {
                task_id: TaskId(0),
                checkpoint: GitCommit::new("def456".into()).unwrap(),
            },
            Event::WorkspaceCreated {
                base: GitCommit::new("abc123".into()).unwrap(),
                branch: GitBranch::new("feature-abc123ab".into()).unwrap(),
                worktree: WorktreePath::new("/repo/wt/feature-abc123ab".into()).unwrap(),
            },
            Event::WorkspaceChanged {
                incarnation: IncarnationId(RecordId(16)),
                worktree: WorktreePath::new("/repo/wt/workspace-deadbeef".into()).unwrap(),
            },
            Event::WorkspaceCleaned {
                workspace: RecordId(17),
            },
            Event::WorkspaceRestored {
                workspace: RecordId(17),
            },
            Event::IncarnationRegistered {
                native_id: NativeSessionId::new("a1b2c3d4-evil".into()).unwrap(),
                saved_session: Some(
                    SessionLocator::new("/home/story/.pi/agent/sessions/a1b2c3d4.jsonl".into())
                        .unwrap(),
                ),
                address: None,
                adopted: false,
                // the kebab token rides the same round-trip as the name
                harness: Harness::ClaudeCode,
            },
            Event::MessageSent {
                from: IncarnationId(RecordId(16)),
                to: IncarnationId(RecordId(17)),
                reply_to: None,
                task: Some(TaskId(0)),
                body: Prose::new("please look at the failing edge case".into()).unwrap(),
            },
            Event::MessageSent {
                from: IncarnationId(RecordId(17)),
                to: IncarnationId(RecordId(16)),
                reply_to: Some(MessageId(RecordId(18))),
                task: None,
                body: Prose::new("the edge case held under review".into()).unwrap(),
            },
            Event::MessageSubmitted {
                id: MessageId(RecordId(18)),
                by: IncarnationId(RecordId(17)),
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
            assemble("proposal_created", "{}"),
            Err(ParseFail::UnknownKind(k)) if k == "proposal_created"
        ));
        assert!(matches!(
            tier_from("aliens"),
            Err(ParseFail::UnknownTier(t)) if t == "aliens"
        ));
    }

    // a registration born before the harness field existed folds as
    // pi: the default lives on the payload, so old logs keep loading
    #[test]
    fn a_registration_without_a_harness_field_folds_as_pi() {
        let payload = r#"{"native_id":"aaaa1111","saved_session":"/sessions/aaaa1111.jsonl","address":"/tty3","adopted":false}"#;
        let event = assemble("incarnation_registered", payload).unwrap();
        assert_eq!(
            event,
            Event::IncarnationRegistered {
                native_id: NativeSessionId::new("aaaa1111".into()).unwrap(),
                saved_session: Some(
                    SessionLocator::new("/sessions/aaaa1111.jsonl".into()).unwrap()
                ),
                address: Some("/tty3".into()),
                adopted: false,
                harness: Harness::Pi,
            }
        );
    }

    #[test]
    fn malformed_payload_of_known_kind_is_loud() {
        assert!(matches!(
            assemble("task_cancelled", "{"),
            Err(ParseFail::Malformed { .. })
        ));
        assert!(matches!(
            assemble("task_claimed", r#"{"id":"three"}"#),
            Err(ParseFail::Malformed { .. })
        ));
    }
}
