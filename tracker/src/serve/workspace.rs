use std::path::Path;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;

use crate::db::{self, ExecuteFail};
use crate::events::Event;
use crate::store::Context;
use crate::workspace;

use super::door::{door, ready};
use super::fail::ApiFail;
use super::state::{AppState, Inner, ServerState};
use super::workspace_wire::{
    BranchOutcome, CleanReply, CleanRequest, CreateReply, CreateRequest, WorkspaceBirth,
};

// -- the task-free workspace controller ----------------------------------

/// Provision through the real Git provisioner under the writer lock,
/// then land the birth with the facts the effect actually produced.
fn create_operation(
    app: &AppState,
    guard: &mut ServerState,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<CreateReply, ApiFail> {
    let inner = ready(guard)?;
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    let home = workspace::worktree_home(&inner.db_path);
    let provisioned =
        workspace::provision(&repo_root, &home, label, base).map_err(ApiFail::Workspace)?;
    let drafts = vec![db::RecordDraft {
        context: Context::system(),
        event: Event::WorkspaceCreated {
            base: provisioned.base.clone(),
            branch: provisioned.branch.clone(),
            worktree: provisioned.worktree.clone(),
        },
    }];
    match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
        Ok(records) => {
            let birth = records[0].seq;
            Ok(CreateReply {
                created: true,
                workspace: WorkspaceBirth {
                    seq: birth,
                    base: provisioned.base.as_str().to_string(),
                    branch: provisioned.branch.as_str().to_string(),
                    path: provisioned.worktree.as_path().display().to_string(),
                },
                records,
            })
        }
        // the effect stands: a failed birth is disclosed, never rolled back
        Err(e) => Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage",
            format!(
                "provisioned branch {} at {} but the birth record could not land: {e}; \
                 the branch and directory stand - inspect them before creating another",
                provisioned.branch.as_str(),
                provisioned.worktree.as_path().display()
            ),
        )),
    }
}

pub(super) async fn workspace_create(State(app): State<AppState>, body: Bytes) -> Response {
    door(
        &app,
        body,
        "workspace/create",
        |r: &CreateRequest| r.client.clone(),
        |app, guard, request| {
            create_operation(
                app,
                guard,
                request.label.as_deref(),
                request.base.as_deref(),
            )
        },
    )
}

// -- the workspace lifecycle controller ---------------------------------
// Clean: the ordered
// clean contract runs can-clean checks, the branch decision first
// among the acts (the fate rule a settles, tip and all), terminal
// close, terminal verification, git worktree remove, checkout and
// worktree absence verification, then the branch's compare-and-
// delete against the decided tip as the last act, and only then
// lands the Cleaned fact.

/// The world fold plus the canonical target resolution every lifecycle
/// door shares: exact recorded workspace births at this path, primary
/// and unrecognized directories refused.
fn lifecycle_target(
    inner: &Inner,
    path_str: &str,
) -> Result<(crate::store::World, crate::store::WorkspaceFacts), ApiFail> {
    let world = match db::load(&inner.conn)
        .map_err(|e| ApiFail::Db(e.to_string()))?
        .state
    {
        db::LoadState::Full(world) => world,
        db::LoadState::Degraded(reason) => return Err(ApiFail::Degraded(reason)),
    };
    // the canonical form names the directory in refusal messages; the
    // resolver owns the addressing decision
    let canonical = workspace::canonical_dir(Path::new(path_str)).map_err(ApiFail::Workspace)?;
    let facts = match workspace::resolve_workspace_address(&world, Path::new(path_str))
        .map_err(ApiFail::Workspace)?
    {
        workspace::LifecycleAddress::Single(birth) => world
            .workspace_facts(birth)
            .expect("the resolver named a recorded birth"),
        workspace::LifecycleAddress::Ambiguous(births) => {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_ambiguous_path",
                format!(
                    "{} is the recorded path of {} workspace births ({}); resolve their history by hand before any lifecycle act",
                    canonical.display(),
                    births.len(),
                    births
                        .iter()
                        .map(|b| b.0.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
        workspace::LifecycleAddress::Unrecorded => {
            return Err(ApiFail::Refused(
                StatusCode::BAD_REQUEST,
                "workspace_not_recorded",
                format!(
                    "{} names no recorded workspace birth; clean selects an exact recorded workspace, never a primary checkout or an unrecognized directory",
                    canonical.display()
                ),
            ));
        }
    };
    Ok((world, facts))
}

fn clean_operation(
    app: &AppState,
    guard: &mut ServerState,
    path_str: &str,
) -> Result<CleanReply, ApiFail> {
    let inner = ready(guard)?;
    let Some(repo_root) = app.repo_root().map(Path::to_path_buf) else {
        return Err(ApiFail::NoRepo);
    };
    let (world, facts) = lifecycle_target(inner, path_str)?;
    let birth = facts.birth;
    let canonical =
        workspace::canonical_dir(facts.worktree.as_path()).map_err(ApiFail::Workspace)?;
    let dir_str = canonical.display().to_string();
    if canonical == repo_root {
        return Err(ApiFail::Refused(
            StatusCode::BAD_REQUEST,
            "workspace_primary_refused",
            format!(
                "{} is the primary checkout; cleanup never touches it",
                canonical.display()
            ),
        ));
    }
    let stage = world
        .workspace_stage(birth)
        .expect("a recorded birth holds a stage");
    let mut registrations =
        workspace::registered_worktrees(&repo_root).map_err(ApiFail::Workspace)?;
    let registered = registrations.get(&canonical).cloned();
    let present = canonical.is_dir();

    let mut verified: Vec<String> = Vec::new();
    // recorded cleaned and every resource absent is idempotent success:
    // no duplicate transition, verified through the same doors the
    // cleanup itself ran - terminal list and visible absence alike
    if stage == crate::record::objects::workspace::WorkspaceStage::Cleaned {
        if present || registered.is_some() {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_cleaned_mismatch",
                format!(
                    "{} is recorded Cleaned but {} stands; a recorded cleanup never authorizes deleting unexpected work - inspect it by hand",
                    dir_str,
                    if present {
                        "a directory"
                    } else {
                        "a git worktree registration"
                    }
                ),
            ));
        }
        // one read answers idempotence: the recorded closure holds
        // only while the host names no terminal at the exact path
        let host = crate::herdr::HerdrCore::resolved().map_err(|e| {
            ApiFail::Refused(
                StatusCode::BAD_GATEWAY,
                "workspace_host_unverified",
                format!(
                    "terminal state unverified at a workspace recorded Cleaned: {e}; the idempotence answer names only what verified - retry when the host answers"
                ),
            )
        })?;
        match crate::host::workspace_terminals(&host, &dir_str) {
            Ok(terminals) if terminals.is_empty() => {
                verified.push("verified: the host names no terminals for the workspace".into());
            }
            Ok(terminals) => {
                return Err(ApiFail::Refused(
                    StatusCode::CONFLICT,
                    "workspace_terminals_open",
                    format!(
                        "the host names {} live terminal(s) at a workspace recorded Cleaned; the recorded closure does not hold - inspect and retry",
                        terminals.len()
                    ),
                ));
            }
            Err(e) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "terminal state unverified at a workspace recorded Cleaned: {e}; the idempotence answer names only what verified - retry when the host answers"
                    ),
                ));
            }
        }
        return Ok(CleanReply {
            cleaned: false,
            already_cleaned: true,
            workspace: WorkspaceBirth::of(facts),
            branch: None,
            verified,
            records: Vec::new(),
        });
    }

    let destructive = registered.is_some();
    // a stale git registration at a recorded path is named, never
    // pruned by a door: git's own repair stays a human act
    if registered.is_some() && !canonical.is_dir() {
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_checkout_mismatch",
            format!(
                "git still registers a worktree at {} whose directory is gone; cleanup refuses the stale registration and never prunes it - inspect it and run git worktree prune by hand",
                dir_str
            ),
        ));
    }
    if destructive {
        // the branch is retained until the rule says otherwise: a
        // missing branch refuses the destructive act before anything
        // runs
        let tip = workspace::branch_tip(&repo_root, &facts.branch).map_err(ApiFail::Workspace)?;
        if tip.is_none() {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_branch_missing",
                format!(
                    "the recorded branch {} holds no local tip; a cleanup that would remove its checkout refuses",
                    facts.branch.as_str()
                ),
            ));
        }
        // work preservation: staged, unstaged, and untracked changes
        // refuse the whole door before any host or git act
        if let Some(detail) = workspace::preserved_work(&canonical).map_err(ApiFail::Workspace)? {
            return Err(ApiFail::Refused(
                StatusCode::CONFLICT,
                "workspace_work_present",
                format!(
                    "{} holds staged, unstaged, or untracked work that cleanup refuses to remove:\n{}",
                    dir_str, detail
                ),
            ));
        }
        verified.push(
            "can-clean: recorded Available, branch present, no staged, unstaged, or untracked work"
                .into(),
        );
    } else if present {
        // bytes at the recorded path are not the checkout: a probe
        // finding them deletes nothing
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_checkout_mismatch",
            format!(
                "{} holds a directory git does not register as this repo's worktree; cleanup deletes no unexpected work - inspect it by hand",
                dir_str
            ),
        ));
    } else {
        verified.push("can-clean: recorded Available, checkout already absent".into());
    }

    // the branch decision, read off the pre-clean fold before any
    // effect - first among the acts, ahead of the terminal closes and
    // every git mutation: the merged rule over the standing recorded
    // branches as the exclusion set - this workspace is itself
    // recorded-uncleaned until its own Cleaned fact lands, and the
    // rule's not-the-branch-itself exclusion keeps its own branch
    // from excluding it - so a cleaned workspace's retained branch
    // counts as a candidate and a stacked workspace's standing parent
    // never does. The tip the rule checked rides the decision: the
    // final act deletes against exactly it.
    let decided_tip =
        workspace::branch_tip(&repo_root, &facts.branch).map_err(ApiFail::Workspace)?;
    let branch_outcome = match decided_tip.as_ref() {
        None => BranchOutcome::kept_missing(),
        Some(_) => {
            let recorded = workspace::recorded_workspace_branches(&world);
            let local = workspace::local_branches(&repo_root).map_err(ApiFail::Workspace)?;
            let merged_into =
                workspace::merged_branches(&facts.branch, &recorded, &local, |work, target| {
                    workspace::branch_is_ancestor(&repo_root, work, target)
                })
                .map_err(ApiFail::Workspace)?;
            if merged_into.is_empty() {
                BranchOutcome::kept_unmerged()
            } else {
                BranchOutcome::deleted(merged_into.iter().map(|b| b.as_str().to_string()).collect())
            }
        }
    };

    // the host stage: enumerate by exact path over one read, close
    // each address, and take the finished fact from the close replies
    // themselves - the host's own read of what it ended, never a
    // re-list; a reply without the fact refuses rather than guessing
    let host = crate::herdr::HerdrCore::resolved().map_err(|e| {
        ApiFail::Refused(
            StatusCode::BAD_GATEWAY,
            "workspace_host_unverified",
            format!(
                "terminal closure is unverified: {e}; no checkout removal ran - retry clean when the host answers"
            ),
        )
    })?;
    use crate::host_contract::{CloseFact, HostContract};
    // the enumeration's own face: an unreadable table is the list
    // read's refusal, never the closure's
    let addresses = crate::host::workspace_terminals(&host, &dir_str)?;
    for address in &addresses {
        match host.close(&crate::host_contract::HostAddress(address.clone())) {
            Ok(CloseFact::Finished) => {}
            Ok(CloseFact::Unconfirmed(evidence)) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "the close of terminal {address} named no finished state ({evidence}); no checkout removal ran - retry clean"
                    ),
                ));
            }
            Err(e) => {
                return Err(ApiFail::Refused(
                    StatusCode::BAD_GATEWAY,
                    "workspace_host_unverified",
                    format!(
                        "the host refused the close of terminal {address}: {e}; no checkout removal ran - retry clean"
                    ),
                ));
            }
        }
    }
    if addresses.is_empty() {
        verified.push("verified: the host names no terminals for the workspace".into());
    } else {
        verified.push(format!(
            "closed {} terminal(s); the host named each finished",
            addresses.len()
        ));
    }
    // absence: one more read, a verification read - a terminal born
    // between the enumeration and the closes surfaces here, and the
    // record never lands over an open terminal
    let again = crate::host::workspace_terminals(&host, &dir_str)?;
    if !again.is_empty() {
        return Err(ApiFail::Refused(
            StatusCode::CONFLICT,
            "workspace_terminals_open",
            format!(
                "the host still names {} terminal(s) for {}; no checkout removal ran",
                again.len(),
                dir_str
            ),
        ));
    }
    verified.push("verified: the host names no terminals for the workspace".into());

    // the checkout removal: git's normal protections; an
    // already-absent checkout satisfies. The branch's own step follows
    // the verified removal, so the removal line claims retention only
    // where the branch actually stands
    if destructive {
        workspace::remove_checkout(&repo_root, &canonical).map_err(ApiFail::Workspace)?;
        verified.push(if branch_outcome.deleted {
            "removed the checkout with git worktree remove".to_string()
        } else {
            format!(
                "removed the checkout with git worktree remove; branch {} retained",
                facts.branch.as_str()
            )
        });
    }

    // absence: the path is gone and git no longer registers it
    if canonical.exists() {
        return Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_absence_unverified",
            format!(
                "{} still exists after the removal attempt; the record lands only on verified absence - inspect and retry",
                dir_str
            ),
        ));
    }
    registrations = workspace::registered_worktrees(&repo_root).map_err(ApiFail::Workspace)?;
    if registrations.contains_key(&canonical) {
        return Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "workspace_absence_unverified",
            format!(
                "git still registers a worktree at {} after the removal attempt; the record lands only on verified absence - inspect and retry",
                dir_str
            ),
        ));
    }
    verified.push("verified: the checkout is gone and git registers no worktree there".into());

    // the branch itself, the LAST act, after the verified removal:
    // compare-and-delete against the tip the decision checked - the
    // ref goes only while it still points exactly there, so no
    // failure behind the worktree can leave the branch in doubt, and
    // a concurrent move refuses with the branch standing, the retry
    // re-deciding. The decision, not git's HEAD or checkout reads,
    // is the judge: a branch merged into a rewrite branch cleans
    // from a main checkout. Never a force, never -D.
    if branch_outcome.deleted {
        let tip = decided_tip
            .as_ref()
            .expect("a deleted branch holds a decided tip");
        workspace::delete_branch(&repo_root, &facts.branch, tip).map_err(ApiFail::Workspace)?;
        verified.push(format!(
            "removed the branch {} (merged into {})",
            facts.branch.as_str(),
            branch_outcome
                .merged_into
                .first()
                .map(String::as_str)
                .unwrap_or_default()
        ));
    }

    // every step verified: the completed fact lands, machinery-authored
    let drafts = vec![db::RecordDraft {
        context: Context::system(),
        event: Event::WorkspaceCleaned { workspace: birth },
    }];
    match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
        Ok(records) => Ok(CleanReply {
            cleaned: true,
            already_cleaned: false,
            workspace: WorkspaceBirth::of(facts),
            branch: Some(branch_outcome),
            verified,
            records,
        }),
        Err(ExecuteFail::Reject(crate::Reject::WorkspaceStageMismatch)) => {
            // a concurrent door recorded the same completion: the
            // resources are verified absent either way - idempotent,
            // and this call's own branch act is disclosed as it ran
            Ok(CleanReply {
                cleaned: false,
                already_cleaned: true,
                workspace: WorkspaceBirth::of(facts),
                branch: Some(branch_outcome),
                verified,
                records: Vec::new(),
            })
        }
        Err(e) => Err(ApiFail::Refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage",
            format!(
                "every cleanup step verified but the Cleaned fact could not land: {e}; the resources stand absent - retry clean to record"
            ),
        )),
    }
}

pub(super) async fn workspace_clean(State(app): State<AppState>, body: Bytes) -> Response {
    door(
        &app,
        body,
        "workspace/clean",
        |r: &CleanRequest| r.client.clone(),
        |app, guard, request| clean_operation(app, guard, &request.path),
    )
}
