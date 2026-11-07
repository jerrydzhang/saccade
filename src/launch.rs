//! The shared launch composition: one construction and one host
//! submission for every entrance - the CLI host door, the CLI here
//! door, and the workbench door compose and submit identically, so no
//! face can grow its own launch dialect. The composition selects the
//! explicit workspace or the saved session, names the harness, and
//! carries the fresh birth facts; nothing here touches the record -
//! the calling door registers the birth or moves the address, because
//! the CLI reaches the tracker over HTTP and the serving edge writes
//! under its own lock.

use std::path::{Path, PathBuf};

use crate::harness::Harness;
use crate::herdr::HerdrCore;
use crate::host;
use crate::host_contract::{HostAddress, HostContract, Liveness};
use crate::pi_session;
use crate::placement;
use crate::workspace::LifecycleAddress;
use crate::{Incarnation, IncarnationId, WorkspaceStage, World};

/// One launch selection: a fresh session in an explicit existing
/// workspace under the named harness, or a saved session resumed under
/// its recorded harness - an expected harness that mismatches the
/// recorded kind refuses, never silently relaunches as another.
pub enum LaunchArm<'a> {
    Fresh {
        workspace: &'a Path,
        prompt: &'a [String],
        harness: Harness,
    },
    Resume {
        id: IncarnationId,
        prompt: &'a [String],
        expected: Option<Harness>,
    },
}

/// The fresh launch's birth facts, exact at composition time: the
/// pre-allocated native id and the harness it was composed under. The
/// submitting door registers exactly these - a birth is never
/// rediscovered by address or inferred kind.
#[derive(Clone, Debug, PartialEq)]
pub struct FreshBirth {
    pub native_id: String,
    pub harness: Harness,
}

/// The resume arm's target: the incarnation and the recorded harness
/// its saved session resumes under.
#[derive(Clone, Debug, PartialEq)]
pub struct ResumeTarget {
    pub id: IncarnationId,
    pub harness: Harness,
}

/// One composed launch: the routing environment, the exact workspace,
/// the session argv under the arm's harness, the identity facts
/// the submitting door records, and the placement the destination
/// class declares - None for the --here arm, an unbound root, or any
/// unrecorded destination. Mechanics only.
pub struct Composition {
    pub routing: Vec<String>,
    pub dir: PathBuf,
    pub session_argv: Vec<String>,
    pub fresh: Option<FreshBirth>,
    pub resume: Option<ResumeTarget>,
    /// The placement the submit may declare: the bound pair at a
    /// bound primary checkout, the derivation at a recorded
    /// workspace, None elsewhere.
    pub placement: Option<placement::Placement>,
}

/// What a submitted launch produced: the workspace and terminal the
/// host named, the placement the door declared at birth - none for the
/// --here arm - the fresh birth facts for the door's registration, and
/// the resume target for the address move.
pub struct Submitted {
    pub workspace: String,
    pub terminal: String,
    pub space: Option<String>,
    pub tab: Option<String>,
    pub place_refusal: Option<String>,
    pub fresh: Option<FreshBirth>,
    pub resume: Option<ResumeTarget>,
}

/// A launch construction or host-submission refusal. `Refused` carries
/// the wire code beside the prose, so the CLI prints the detail and
/// the serving edge answers with its own status; `Usage` is a
/// door-shaped refusal with no wire consumer (the workbench never
/// resumes).
#[derive(Debug)]
pub enum LaunchFail {
    Refused { code: &'static str, detail: String },
    Usage(String),
    Degraded(String),
}

impl LaunchFail {
    /// The wire refusal code a `Refused` carries; a `Usage` or
    /// `Degraded` has no launch wire face.
    pub fn wire_code(&self) -> Option<&'static str> {
        match self {
            LaunchFail::Refused { code, .. } => Some(code),
            _ => None,
        }
    }
}

impl std::fmt::Display for LaunchFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchFail::Refused { detail, .. } => write!(f, "{detail}"),
            LaunchFail::Usage(detail) => write!(f, "{detail}"),
            LaunchFail::Degraded(reason) => write!(f, "{reason}"),
        }
    }
}

/// Everything one composition reads: the folded world, the repo the
/// sessions tree lives under, the tracker URL the routing names, the
/// CLI path when the door knows its own, and the arm. The host is
/// never passed in: the resume arm resolves it for its own liveness
/// preflight, and the submission resolves it after the direnv equip,
/// so an unequipped launch refuses before any host contact.
pub struct LaunchRequest<'a> {
    pub world: &'a World,
    pub repo: &'a Path,
    pub url: String,
    pub exe: Option<PathBuf>,
    pub arm: LaunchArm<'a>,
    /// Whether the submitted launch declares Herdr placement. True for
    /// every door but --here: the invoking viewer's launch stays
    /// unplaced, reachable by its own recorded host address.
    pub declare_placement: bool,
}

/// Compose one launch: the routing line, the exact workspace and
/// harness argv, and the arm's identity facts. The resume arm runs its
/// strict preflight here - saved locator, header identity, recorded
/// workspace, cleaned guard, and host liveness - so a refused resume
/// contacts no session and changes nothing. The destination class,
/// not the caller, decides the declared placement.
pub fn compose(request: LaunchRequest) -> Result<Composition, LaunchFail> {
    let LaunchRequest {
        world,
        repo,
        url,
        exe,
        arm,
        declare_placement,
    } = request;
    let (dir, session_argv, fresh, resume, label, launched_session) = match arm {
        LaunchArm::Fresh {
            workspace,
            prompt,
            harness,
        } => {
            let dir = canonical_workspace(workspace)?;
            cleaned_guard(world, &dir)?;
            // the pin resolves before anything exists: a miss refuses
            // a launch that never started
            let bin = harness.pinned_binary().map_err(|e| pin_miss(&harness, e))?;
            // the door pre-allocates the native id: identity and
            // address exist at launch time, no attach handshake
            let native_id = pi_session::generate_session_id();
            if harness == Harness::Pi {
                let sessions = pi_session::sessions_dir(repo);
                let _ = std::fs::create_dir_all(&sessions);
            }
            let session_argv = harness
                .fresh_argv(repo, &native_id, prompt)
                .map_err(|e| pin_miss(&harness, e))?;
            (
                dir,
                session_argv,
                Some(FreshBirth {
                    native_id: native_id.clone(),
                    harness,
                }),
                None,
                host::command_label(&bin.display().to_string()),
                native_id,
            )
        }
        LaunchArm::Resume {
            id,
            prompt,
            expected,
        } => {
            let num = world.incarnation_num(id).ok_or_else(|| {
                LaunchFail::Usage("the resume target names no incarnation".into())
            })?;
            let token = format!("i-{num}");
            let ctx = world
                .incarnations
                .get(&id)
                .expect("the fold just resolved this incarnation");
            let Incarnation::Native {
                native_id,
                saved_session,
                address,
                adopted,
                harness,
            } = &ctx.incarnation
            else {
                return Err(LaunchFail::Usage(format!(
                    "{token} carries no saved native session to resume; historical attribution identities have none",
                )));
            };
            // the recorded kind is the authority: an explicit harness
            // is a claim about the session, and a wrong claim refuses
            if let Some(expected) = expected
                && expected != *harness
            {
                return Err(LaunchFail::Usage(format!(
                    "{token} runs {} as recorded; resume starts the recorded harness, and --harness {} would launch another kind",
                    harness.as_str(),
                    expected.as_str(),
                )));
            }
            // strict resume: the saved file must exist and carry the
            // recorded native id, and an unbound birth has nothing
            // saved to resume
            let Some(saved_session) = saved_session else {
                return Err(LaunchFail::Usage(format!(
                    "{token}'s saved session locator is not bound yet: the session has not saved, so there is nothing to resume"
                )));
            };
            let header = harness.read_header(saved_session.as_path()).map_err(|e| {
                LaunchFail::Usage(format!("{token}'s saved session cannot resume: {e}"))
            })?;
            if header.id != native_id.as_str() {
                return Err(LaunchFail::Usage(format!(
                    "{token}'s saved session carries id {}, not {}",
                    header.id,
                    native_id.as_str()
                )));
            }
            // the recorded change is the workspace authority; the
            // header cwd answers only for sessions whose fold holds
            // no change
            let dir = match world.workspace_changes.get(&id) {
                Some(change) => std::path::PathBuf::from(change.worktree.as_path()),
                None => std::path::PathBuf::from(&header.cwd),
            };
            if !dir.is_dir() {
                return Err(LaunchFail::Usage(format!(
                    "the workspace {} is missing; resume starts from the recorded workspace, never a substitute",
                    dir.display()
                )));
            }
            cleaned_guard(world, &dir)?;
            // liveness preflight, only for the launched kind: a
            // launched terminal dies with its session, so a live
            // terminal names a live session; an adopted pane is the
            // caller's own program and outlives the session, so
            // terminal liveness is never session liveness for it - the
            // native duplicate-attach refusal is that kind's backstop.
            // An addressless pre-contract incarnation has no host
            // evidence at all: the resume proceeds, and a live twin is
            // caught at the resumed session's attach - the pre-B
            // migration is exit + resume
            if let Some(carried) = address.as_ref().filter(|_| !*adopted).cloned() {
                // the preflight's own host: resolution failure is the
                // deployment's break, never a free liveness answer
                let host = HerdrCore::resolved().map_err(|e| LaunchFail::Refused {
                    code: "launch_no_herdr",
                    detail: e.to_string(),
                })?;
                let answer = host.live(&HostAddress(carried));
                match answer {
                    Liveness::Live => {
                        return Err(LaunchFail::Usage(format!(
                            "{token}'s session is live at its recorded host address; no launch is submitted; exit that session first"
                        )));
                    }
                    Liveness::Unknown(evidence) => {
                        return Err(LaunchFail::Usage(format!(
                            "{token}'s liveness cannot be read from the host at its recorded address ({evidence}); no launch is submitted - settle the terminal or the record first"
                        )));
                    }
                    Liveness::Gone => {}
                }
            }
            let session_argv = harness
                .resume_argv(native_id.as_str(), saved_session.as_path(), prompt)
                .map_err(|e| pin_miss(harness, e))?;
            (
                dir,
                session_argv,
                None,
                Some(ResumeTarget {
                    id,
                    harness: *harness,
                }),
                host::command_label(
                    &harness
                        .pinned_binary()
                        .map_err(|e| pin_miss(harness, e))?
                        .display()
                        .to_string(),
                ),
                native_id.as_str().to_string(),
            )
        }
    };
    let placement = if declare_placement {
        placement::for_dir(world, repo, &dir)
    } else {
        None
    };
    Ok(Composition {
        routing: routing(url, repo, exe, label, launched_session.as_str()),
        dir,
        session_argv,
        fresh,
        resume,
        placement,
    })
}

/// Submit one composed launch through the host: the direnv equip
/// first - an unequipped launch refuses before any host contact -
/// then the host ask and the placement declaration. Mechanics only -
/// the calling door registers the birth or moves the address, and
/// rendering or any follow-on act belongs to the door.
pub fn submit(composed: Composition) -> Result<Submitted, LaunchFail> {
    let argv = host::composed_launch_argv(&composed.dir, &composed.routing, &composed.session_argv)
        .map_err(|e| LaunchFail::Refused {
            code: e.wire_code(),
            detail: e.to_string(),
        })?;
    // the host ask is one verb: the adapter returns the opaque address
    // synchronously, and the launch never supervises what it launched
    let host = HerdrCore::resolved().map_err(|e| LaunchFail::Refused {
        code: "launch_no_herdr",
        detail: e.to_string(),
    })?;
    let address = host
        .launch(&composed.dir, &argv)
        .map_err(|e| LaunchFail::Refused {
            code: "launch_host_refused",
            detail: e.to_string(),
        })?;
    // placement at birth: the destination class decides what the
    // terminal declares - the bound pair at a bound primary checkout,
    // the derivation at a recorded workspace, and no lookup, no label
    // ask, no placement claim anywhere else. The --here arm composes
    // None the same way: its terminal stays unplaced for explicit
    // later attachment.
    let (space, tab, place_refusal) = match composed.placement {
        Some(placement) => {
            let refusal = host
                .place(&address, &placement.space, &placement.tab)
                .err()
                .map(|e| e.to_string());
            (Some(placement.space), Some(placement.tab), refusal)
        }
        None => (None, None, None),
    };
    Ok(Submitted {
        workspace: composed.dir.display().to_string(),
        terminal: address.0.clone(),
        space,
        tab,
        place_refusal,
        fresh: composed.fresh,
        resume: composed.resume,
    })
}

/// The routing environment words: the acting label (the harness's own
/// binary name), the tracker URL, the repo, and the CLI path when the
/// door knows its own. The words are env assignments, never shell.
fn routing(
    url: String,
    repo: &Path,
    exe: Option<PathBuf>,
    label: String,
    launched_session: &str,
) -> Vec<String> {
    let mut routing = vec![
        format!("SACCADE_ACTOR={label}"),
        format!("SACCADE_SERVER={url}"),
        format!("SACCADE_REPO={}", repo.display()),
        // launch provenance scoped to the exact native identity: a
        // nested manually started session inherits this word but its
        // own session id names another session, so it never reads as
        // the launched one
        format!("SACCADE_LAUNCHED_SESSION={launched_session}"),
    ];
    if let Some(exe) = exe {
        routing.push(format!("SACCADE_CLI={}", exe.display()));
    }
    routing
}

/// The wire code a pin miss carries: the existing launch_no_pi keeps
/// its consumers, and the second harness names its own.
fn pin_miss(harness: &Harness, detail: String) -> LaunchFail {
    let code = match harness {
        Harness::Pi => "launch_no_pi",
        Harness::ClaudeCode => "launch_no_claude_code",
    };
    LaunchFail::Refused { code, detail }
}

/// The explicit existing workspace, canonical: launch selects an
/// existing directory, it never creates one.
fn canonical_workspace(workspace: &Path) -> Result<PathBuf, LaunchFail> {
    let dir = workspace.canonicalize().map_err(|e| LaunchFail::Refused {
        code: "workspace_path_invalid",
        detail: format!("workspace {}: {e}", workspace.display()),
    })?;
    if !dir.is_dir() {
        return Err(LaunchFail::Refused {
            code: "workspace_path_invalid",
            detail: format!(
                "{} is not an existing directory; launch selects an explicit existing workspace, it never creates one",
                dir.display()
            ),
        });
    }
    Ok(dir)
}

/// The cleaned-workspace launch guard: a recorded workspace whose
/// stage is Cleaned refuses ordinary use until an explicit restore,
/// addressed by the one shared resolver - its own stored string or the
/// canonical form it resolves to. Unrecognized directories and
/// primary checkouts are not workspaces and pass untouched; aliased
/// identities refuse rather than being picked.
fn cleaned_guard(world: &World, dir: &Path) -> Result<(), LaunchFail> {
    let canonical =
        crate::workspace::canonical_dir(dir).map_err(|e| LaunchFail::Usage(e.detail()))?;
    match crate::workspace::resolve_workspace_address(world, dir)
        .map_err(|e| LaunchFail::Usage(e.detail()))?
    {
        LifecycleAddress::Single(birth) => {
            if world.workspace_stage(birth) == Some(WorkspaceStage::Cleaned) {
                return Err(LaunchFail::Refused {
                    code: "workspace_cleaned",
                    detail: format!(
                        "{} is recorded Cleaned (record {}); ordinary use waits for an explicit restore: sac workspace restore {}",
                        canonical.display(),
                        birth.0,
                        canonical.display()
                    ),
                });
            }
        }
        LifecycleAddress::Ambiguous(births) => {
            return Err(LaunchFail::Refused {
                code: "workspace_ambiguous_path",
                detail: format!(
                    "{} is the recorded path of {} workspace births; resolve their history by hand before launching there",
                    canonical.display(),
                    births.len()
                ),
            });
        }
        LifecycleAddress::Unrecorded => {}
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_pin_miss_carries_its_own_harnesss_wire_code() {
        let pi = pin_miss(&Harness::Pi, "no pinned pi".into());
        assert_eq!(pi.wire_code(), Some("launch_no_pi"));
        let cc = pin_miss(&Harness::ClaudeCode, "no pinned claude-code".into());
        assert_eq!(cc.wire_code(), Some("launch_no_claude_code"));
        // prose-only refusals carry no wire face
        assert_eq!(LaunchFail::Usage("door-shaped".into()).wire_code(), None);
    }

    #[test]
    fn routing_is_env_words_and_nothing_else() {
        let routing = routing(
            "http://127.0.0.1:1".into(),
            Path::new("/repo/wt"),
            Some(PathBuf::from("/usr/bin/sac")),
            "wrapped-pi".into(),
            "aaaa1111",
        );
        assert_eq!(
            routing,
            vec![
                "SACCADE_ACTOR=wrapped-pi".to_string(),
                "SACCADE_SERVER=http://127.0.0.1:1".to_string(),
                "SACCADE_REPO=/repo/wt".to_string(),
                "SACCADE_LAUNCHED_SESSION=aaaa1111".to_string(),
                "SACCADE_CLI=/usr/bin/sac".to_string(),
            ]
        );
        // a door that does not know its own exe names no CLI word
        let plain = super::routing(
            "http://127.0.0.1:1".into(),
            Path::new("/repo/wt"),
            None,
            "pi".into(),
            "aaaa1111",
        );
        assert_eq!(plain.len(), 4);
        assert!(plain.iter().all(|w| !w.starts_with("SACCADE_CLI=")));
    }
}
