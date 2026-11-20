use crate::record::store::RecordId;
use crate::types::actor::ActorName;
use crate::types::tag::{TagLine, TagName};

/// A tag born by its own creation record: a flat name, identity at
/// the birth record. Tags are listing facts only; no transition, gate
/// or acceptance reads them.
#[derive(Clone, Debug, PartialEq)]
pub struct TagContext {
    pub name: TagName,
    /// The present line saying what belongs under the tag
    pub description: TagLine,
    pub born_at: u64,
    pub actor: ActorName,
}

/// The tag birth that bears this name, ignoring ASCII case: names are
/// unique under that comparison.
pub fn tag_named(
    tags: &std::collections::BTreeMap<RecordId, TagContext>,
    name: &str,
) -> Option<RecordId> {
    tags.iter()
        .find(|(_, ctx)| ctx.name.as_str().eq_ignore_ascii_case(name))
        .map(|(id, _)| *id)
}
