use super::launch::provision_new;
use super::*;
use std::path::Path;

/// The creation's text face: actual Git facts and the birth record.
pub(super) fn render_birth(birth: &client::WorkspaceBirth) -> String {
    format!(
        "workspace provisioned - record {} (its identity, no numbering)\n  branch {} | base {}\n  path {}",
        birth.seq, birth.branch, birth.base, birth.path
    )
}

/// `sac workspace new`: provision and report; no task, claim, or
/// incarnation rides it.
pub(super) fn workspace_new_door(
    cli: &Cli,
    destination: &Destination,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<String, Fail> {
    let birth = provision_new(destination, label, base)?;
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "created": true,
            "workspace": birth,
        }))
        .expect("plain data"));
    }
    Ok(render_birth(&birth))
}

pub(super) fn workspace_door(
    cli: &Cli,
    destination: &Destination,
    verb: &WorkspaceVerb,
) -> Result<String, Fail> {
    match verb {
        WorkspaceVerb::New { label, base } => {
            workspace_new_door(cli, destination, label.as_deref(), base.as_deref())
        }
        WorkspaceVerb::Clean { path } => workspace_clean_door(cli, destination, path),
        WorkspaceVerb::Root { verb } => match verb {
            RootVerb::Bind => root_bind_door(cli, destination),
            RootVerb::Show => root_show_door(cli, destination),
            RootVerb::Clear => root_clear_door(cli, destination),
        },
    }
}

/// `sac workspace clean`: the ordered cleanup through the server's
/// effects. The reply names every verified step; a failure names the
/// step that stopped, and retry is the recovery path.
pub(super) fn workspace_clean_door(
    cli: &Cli,
    destination: &Destination,
    path: &Path,
) -> Result<String, Fail> {
    let url = server_url(destination, "workspace clean")?;
    client::handshake(&url);
    let reply = client::clean_workspace(&url, path).map_err(Fail::Client)?;
    let record = reply.records.first().map(|r| r.seq);
    let fact = if reply.cleaned {
        format!(
            "cleaned - record {}",
            record.map(|r| r.to_string()).unwrap_or_default()
        )
    } else if reply.already_cleaned {
        "already cleaned - every resource verified absent, no new record".to_string()
    } else {
        "completed".to_string()
    };
    if cli.json {
        return Ok(serde_json::to_string_pretty(&serde_json::json!({
            "cleaned": reply.cleaned,
            "already_cleaned": reply.already_cleaned,
            "workspace": reply.workspace,
            "verified": reply.verified,
            "records": reply.records,
        }))
        .expect("plain data"));
    }
    Ok(format!(
        "workspace {} ({}, branch {})\n  {}\n{}",
        reply.workspace.path,
        reply.workspace.seq,
        reply.workspace.branch,
        fact,
        reply
            .verified
            .iter()
            .map(|step| format!("  - {step}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}
