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
    }
}

pub fn tier_from(s: &str) -> Result<Tier, ParseFail> {
    match s {
        "human" => Ok(Tier::Human),
        "agent" => Ok(Tier::Agent),
        _ => Err(ParseFail::UnknownTier(s.to_string())),
    }
}

/// Known kinds, anything else in a row is version skew, not corruption
const KINDS: [&str; 10] = [
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
    use crate::prose::Prose;
    use crate::store::RecordId;

    #[test]
    fn every_event_kind_round_trips() {
        let samples = [
            Event::TaskCreated {
                id: TaskId(0),
                name: Prose::new("implement foo".into()).unwrap(),
                parent_id: None,
            },
            Event::TaskCreated {
                id: TaskId(1),
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
            },
            Event::Commented {
                target: Target::Comment(CommentId(RecordId(6))),
                body: Prose::new("no - pure tree, here is why".into()).unwrap(),
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
