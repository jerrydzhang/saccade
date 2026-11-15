use std::path::Path;
use std::time::Instant;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::attempts::AsReceived;
use crate::db::{self, ExecuteFail, StoredRecord};
use crate::events::Event;
use crate::store::Context;
use crate::workspace;

use super::door::integration_outcome;
use super::fail::{ApiFail, api_failure};
use super::state::{AppState, Inner, ServerState};

// -- the task-free workspace controller ----------------------------------

/// What a creation provisioned: actual Git facts and the birth record
/// that carries them - the workspace's identity, never a numbering.
#[derive(Serialize)]
pub struct WorkspaceBirth {
    /// The birth record's position: the workspace's stable identity
    pub seq: usize,
    pub base: String,
    pub branch: String,
    pub path: String,
}

#[derive(Serialize)]
pub struct CreateReply {
    pub created: bool,
    pub workspace: WorkspaceBirth,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct CreateRequest {
    /// The descriptive label of the c-366 name: optional, falling
    /// back to workspace-<token>
    #[serde(default)]
    label: Option<String>,
    /// The canonical base: a ref, defaulting to the primary HEAD
    #[serde(default)]
    base: Option<String>,
    #[serde(default)]
    client: Option<String>,
}

/// Provision through the real Git provisioner under the writer lock,
/// then land the birth with the facts the effect actually produced.
fn create_operation(
    app: &AppState,
    guard: &mut ServerState,
    label: Option<&str>,
    base: Option<&str>,
) -> Result<CreateReply, ApiFail> {
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
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
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: CreateRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = create_operation(
        &app,
        &mut guard,
        request.label.as_deref(),
        request.base.as_deref(),
    );
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    if outcome.is_ok() {
        app.wake_landing();
    }
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}

// -- the workspace lifecycle controller ---------------------------------
// Clean: the ordered
// clean contract runs can-clean checks, terminal close, terminal
// verification, git worktree remove (branch retained), checkout and
// worktree absence verification, and the host's visible absence,
// and only then lands the Cleaned fact.

/// The recorded facts of the lifecycle's target: the birth identity
/// and what the birth carried.
#[derive(Serialize)]
pub struct LifecycleWorkspace {
    seq: usize,
    base: String,
    branch: String,
    path: String,
}

impl LifecycleWorkspace {
    fn of(facts: crate::store::WorkspaceFacts) -> Self {
        LifecycleWorkspace {
            seq: facts.birth.0,
            base: facts.base.as_str().to_string(),
            branch: facts.branch.as_str().to_string(),
            path: facts.worktree.as_path().display().to_string(),
        }
    }
}

#[derive(Serialize)]
pub struct CleanReply {
    /// A Cleaned fact landed with this call
    pub cleaned: bool,
    /// Recorded cleaned and every verified resource already absent:
    /// idempotent success, no duplicate transition
    pub already_cleaned: bool,
    pub workspace: LifecycleWorkspace,
    /// The honest step list: what the door asked, what it verified,
    /// and any host ask that failed before the list confirmed closure
    pub verified: Vec<String>,
    pub records: Vec<StoredRecord>,
}

#[derive(Deserialize)]
struct CleanRequest {
    /// The exact recorded workspace path; never inferred from a task
    /// or incarnation
    path: String,
    #[serde(default)]
    client: Option<String>,
}

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
    let ServerState::Ready(inner) = guard else {
        let ServerState::Degraded(reason, _) = guard else {
            unreachable!("the state is one of the two")
        };
        return Err(ApiFail::Degraded(reason.clone()));
    };
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
    if stage == crate::WorkspaceStage::Cleaned {
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
            workspace: LifecycleWorkspace::of(facts),
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
        // the branch is retained: a missing branch refuses the
        // destructive act before anything runs
        if workspace::branch_tip(&repo_root, &facts.branch)
            .map_err(ApiFail::Workspace)?
            .is_none()
        {
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

    // the checkout removal: git's normal protections, the branch and
    // every commit on it retained; an already-absent checkout satisfies
    if destructive {
        workspace::remove_checkout(&repo_root, &canonical).map_err(ApiFail::Workspace)?;
        verified.push(format!(
            "removed the checkout with git worktree remove; branch {} retained",
            facts.branch.as_str()
        ));
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

    // every step verified: the completed fact lands, machinery-authored
    let drafts = vec![db::RecordDraft {
        context: Context::system(),
        event: Event::WorkspaceCleaned { workspace: birth },
    }];
    match db::execute_batch(&mut inner.conn, db::AtomicBatch { drafts }, db::now_epoch()) {
        Ok(records) => Ok(CleanReply {
            cleaned: true,
            already_cleaned: false,
            workspace: LifecycleWorkspace::of(facts),
            verified,
            records,
        }),
        Err(ExecuteFail::Reject(crate::Reject::WorkspaceStageMismatch)) => {
            // a concurrent door recorded the same completion: the
            // resources are verified absent either way - idempotent
            Ok(CleanReply {
                cleaned: false,
                already_cleaned: true,
                workspace: LifecycleWorkspace::of(facts),
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
    let raw = String::from_utf8_lossy(&body).into_owned();
    let request: CleanRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return api_failure(ApiFail::Malformed(e.to_string())),
    };
    let received = AsReceived {
        client: request.client,
        raw,
    };
    let started = Instant::now();
    let mut guard = app.inner.lock().expect("the writer lock is not poisoned");
    let outcome = clean_operation(&app, &mut guard, &request.path);
    integration_outcome(&app, &guard, None, &received, started, &outcome);
    if outcome.is_ok() {
        app.wake_landing();
    }
    drop(guard);
    match outcome {
        Ok(reply) => (
            StatusCode::OK,
            Json(serde_json::to_value(&reply).expect("the reply is plain data")),
        )
            .into_response(),
        Err(fail) => api_failure(fail),
    }
}
