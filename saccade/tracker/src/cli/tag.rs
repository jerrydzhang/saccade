use super::*;

pub(super) fn tag_name_of(name: &str) -> Result<saccade::record::types::tag::TagName, Fail> {
    Ok(saccade::record::types::tag::TagName::new(name.to_string())?)
}

/// The tag births the names stand for, resolved against the fold
/// ignoring ASCII case; a name no tag bears refuses, naming the tags.
pub(super) fn names_to_tags(world: &World, names: &[String]) -> Result<Vec<RecordId>, Fail> {
    names
        .iter()
        .map(|name| {
            world.tag_named(name).ok_or_else(|| Fail::Taught {
                code: Reject::InvalidTagId.code(),
                text: saccade::refusals::unknown_tag(world, name),
            })
        })
        .collect()
}

pub(super) fn resolve_tags(
    destination: &Destination,
    names: &[String],
) -> Result<Vec<RecordId>, Fail> {
    match loadout_of(destination)?.state {
        LoadState::Full(world) => names_to_tags(&world, names),
        LoadState::Degraded(reason) => Err(Fail::Degraded(reason)),
    }
}

/// A task's tags as the show and edit faces print them: a header,
/// then each tag with its present line.
pub(super) fn task_tag_block(
    world: &World,
    ctx: &saccade::objects::task::TaskContext,
) -> Vec<String> {
    let tags = world.tags_of(ctx);
    if tags.is_empty() {
        return vec!["tags  -".into()];
    }
    let mut out = vec!["tags".to_string()];
    out.extend(
        tags.iter()
            .map(|tag| format!("  {} - {}", tag.name.as_str(), tag.description.as_str())),
    );
    out
}

/// The tags in birth order, one per line, name then line; no counts.
pub(super) fn tag_list(cli: &Cli, destination: &Destination) -> Result<String, Fail> {
    let loadout = loadout_of(destination)?;
    let LoadState::Full(world) = &loadout.state else {
        let LoadState::Degraded(reason) = loadout.state else {
            unreachable!("the state is full or degraded")
        };
        return Err(Fail::Degraded(reason));
    };
    if cli.json {
        let tags: Vec<serde_json::Value> = world
            .tags
            .values()
            .map(|tag| {
                serde_json::json!({
                    "name": tag.name.as_str(),
                    "created_at": tag.born_at,
                    "description": tag.description.as_str(),
                    "by": tag.actor.as_str(),
                })
            })
            .collect();
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "tags": tags,
            "cursor": loadout.rows.last().map(|r| r.seq),
        }))
        .expect("tags are plain data"));
    }
    Ok(world
        .tags
        .values()
        .map(|tag| format!("{} - {}", tag.name.as_str(), tag.description.as_str()))
        .collect::<Vec<_>>()
        .join("\n"))
}
