use serde::Deserialize;

use crate::events::Event;
use crate::objects::comment::{CommentKind, Target};
use crate::store::Tier;
use crate::types::pointers::GitCommit;
use crate::types::prose::Prose;

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
const KINDS: [&str; 26] = [
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
    "comment_withdrawn",
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
    "task_workspace_materialized",
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
    // the workspace rename's ratified migration: logs written before
    // it carry the worktree kind and load unchanged behind this map
    let kind = match kind {
        "task_worktree_created" => "task_workspace_materialized",
        kind => kind,
    };
    if !KINDS.contains(&kind) {
        return Err(ParseFail::UnknownKind(kind.to_string()));
    }
    if kind == "commented" {
        return decode_commented(payload);
    }
    let inner: serde_json::Value = serde_json::from_str(payload).map_err(|e| malformed(kind, e))?;
    let mut tagged = serde_json::Map::new();
    tagged.insert(kind.to_string(), inner);
    serde_json::from_value(serde_json::Value::Object(tagged)).map_err(|e| malformed(kind, e))
}

/// The commented decode. Two migrations for shapes newer writes never
/// make: the addressee field becomes a kind (to:agent a demand,
/// everything else a note), and the bare string kind "demand" — which
/// predates the base field — becomes a demand whose base is the
/// reserved word, minted here at the word's sole origin. An object
/// form claiming the word never reaches this arm: serde hands it to
/// `GitCommit::new`, which rejects it.
fn decode_commented(payload: &str) -> Result<Event, ParseFail> {
    let mut value: serde_json::Value =
        serde_json::from_str(payload).map_err(|e| malformed("commented", e))?;
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
    if value.get("kind").and_then(|k| k.as_str()) == Some("demand") {
        #[derive(Deserialize)]
        struct Shell {
            target: Target,
            body: Prose,
        }
        let mut object = value;
        object
            .as_object_mut()
            .expect("the kind read above saw an object")
            .remove("kind");
        let shell: Shell = serde_json::from_value(object).map_err(|e| malformed("commented", e))?;
        return Ok(Event::Commented {
            target: shell.target,
            body: shell.body,
            kind: CommentKind::Demand {
                base: GitCommit::unrecorded(),
            },
        });
    }
    let mut tagged = serde_json::Map::new();
    tagged.insert("commented".to_string(), value);
    serde_json::from_value(serde_json::Value::Object(tagged)).map_err(|e| malformed("commented", e))
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
                kind: CommentKind::Demand {
                    base: GitCommit::new("a1b2c3".into()).unwrap(),
                },
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
            Event::CommentWithdrawn {
                id: CommentId(RecordId(6)),
                note: Prose::new("parked in the wrong place".into()).unwrap(),
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
            Event::TaskWorkspaceMaterialized {
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

    /// The workspace rename's migration: an old log's worktree kind
    /// loads as the materialized-workspace event, payload untouched.
    #[test]
    fn the_old_worktree_kind_loads_as_the_workspace_event() {
        let event = assemble(
            "task_worktree_created",
            r#"{"task_id":0,"worktree":"/repo/wt/t-0"}"#,
        )
        .unwrap();
        assert!(
            matches!(&event, Event::TaskWorkspaceMaterialized { worktree, .. }
                if worktree.as_path() == std::path::Path::new("/repo/wt/t-0")),
            "{event:?}"
        );
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

    /// The ratified migrations: old addressee payloads load unchanged —
    /// to:agent becomes a demand, everything else a note; a payload
    /// that already carries kind passes through untouched, and the
    /// bare "demand" string predating the base field loads as a
    /// demand whose base is the reserved word.
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
                    kind: CommentKind::Demand { base },
                    ..
                } if base.as_str() == "unrecorded"
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
        // the migration consumed the addressee field; the object form
        // it re-serializes to claims the reserved word, which only the
        // decoder's bare-string map mints — its re-assembly is the
        // malformed pin at this test's tail
        let (_, payload) = disassemble(&agent);
        assert!(!payload.contains("addressee"), "{payload}");
        assert!(payload.contains("unrecorded"), "{payload}");

        // the base-bearing kind keeps its commit through the same door
        let sighted = assemble(
            "commented",
            r#"{"target":{"task":0},"body":"cut at the tip","kind":{"demand":{"base":"a1b2c3"}}}"#,
        )
        .unwrap();
        assert!(
            matches!(
                &sighted,
                Event::Commented {
                    kind: CommentKind::Demand { base: commit },
                    ..
                } if commit.as_str() == "a1b2c3"
            ),
            "{sighted:?}"
        );
        let (kind, payload) = disassemble(&sighted);
        assert_eq!(&assemble(&kind, &payload).unwrap(), &sighted);

        // the bare string kind is the pre-base log shape: it loads as a
        // demand whose base is the reserved word
        let bare = assemble(
            "commented",
            r#"{"target":{"task":0},"body":"fired before bases","kind":"demand"}"#,
        )
        .unwrap();
        assert!(
            matches!(
                &bare,
                Event::Commented {
                    kind: CommentKind::Demand { base },
                    ..
                } if base.as_str() == "unrecorded"
            ),
            "{bare:?}"
        );

        // an object form claiming the reserved word is malformed by
        // `new`'s own rejection: the word enters only through the
        // decoder's bare-string map
        assert!(matches!(
            assemble(
                "commented",
                r#"{"target":{"task":0},"body":"hand-made","kind":{"demand":{"base":"unrecorded"}}}"#,
            ),
            Err(ParseFail::Malformed { .. })
        ));
    }
}
