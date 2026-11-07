//! The door's placement mechanics: every terminal a saccade door
//! spawns declares where it lives, at birth - and the destination
//! alone decides what it declares. A recorded workspace declares its
//! stable derived pair; the primary checkout declares the exact pair
//! the human bound (`saccade workspace root bind`); an unbound root
//! or any unrecorded destination declares nothing, and the terminal
//! stays unplaced for human attachment. No launch consults the host's
//! terminal table, lends another terminal's labels, or infers from
//! any cwd: the Herdr app supplies worktree-under-root nesting from
//! its own git facts. Ids wear the app's own shape: a prefix and 32
//! lowercase hex characters.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::World;
use crate::workspace::{LifecycleAddress, resolve_workspace_address};

/// The derived space and tab ids one directory declares.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub space: String,
    pub tab: String,
}

/// The placement a launch declares for `dir`, or None when the
/// destination has no declared placement: the human-bound pair at the
/// primary checkout, the stable derivation for a recorded workspace,
/// unplaced for an unbound root or any unrecorded directory.
pub fn for_dir(world: &World, repo: &Path, dir: &Path) -> Option<Placement> {
    let root_canonical = canonical(dir);
    if root_canonical == canonical(repo) {
        return world.root_space.as_ref().map(|b| Placement {
            space: b.space.clone(),
            tab: b.tab.clone(),
        });
    }
    match resolve_workspace_address(world, dir) {
        Ok(LifecycleAddress::Single(_)) => Some(derived(&root_canonical)),
        _ => None,
    }
}

/// The stable ids for one directory: a domain-separated sha256 over
/// the canonical path, so the space and the tab never collide and the
/// same workspace always derives the same pair.
pub fn derived(dir: &Path) -> Placement {
    let canonical = canonical(dir);
    Placement {
        space: mint("saccade-space", canonical.display(), "spa"),
        tab: mint("saccade-tab", canonical.display(), "tb"),
    }
}

fn canonical(dir: &Path) -> PathBuf {
    dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf())
}

fn mint(domain: &str, path: impl std::fmt::Display, prefix: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0u8]);
    hasher.update(path.to_string().as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
    format!("{prefix}{hex}")
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::objects::workspace::{CreatedWorkspace, CreatedWorkspaceContext, WorkspaceStage};
    use crate::types::pointers::{GitBranch, GitCommit, WorktreePath};
    use crate::{RecordId, WorkspaceId};

    fn world_with_root(space: &str, tab: &str) -> World {
        World {
            root_space: Some(crate::RootSpaceBinding {
                space: space.into(),
                tab: tab.into(),
            }),
            ..World::new()
        }
    }

    fn world_with_workspace(path: &Path) -> World {
        let mut world = World::new();
        world.workspaces.insert(
            WorkspaceId(RecordId(7)),
            CreatedWorkspaceContext {
                workspace: CreatedWorkspace {
                    base: GitCommit::new("0f9e8d7c".into()).unwrap(),
                    branch: GitBranch::new("workspace-gloom".into()).unwrap(),
                    worktree: WorktreePath::new(path.to_path_buf()).unwrap(),
                    stage: WorkspaceStage::Available,
                },
                born_at: 0,
                stage_fact: None,
            },
        );
        world
    }

    #[test]
    fn the_derivation_is_stable_shaped_and_distinct() {
        let dir = std::env::temp_dir().join(format!("sac-place-stable-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let one = derived(&dir);
        let again = derived(&dir);
        assert_eq!(one, again, "the same directory derives the same ids");
        let trailing = derived(&dir.join("sub"));
        assert_ne!(one, trailing, "another directory derives other ids");
        for (id, prefix) in [(one.space.as_str(), "spa"), (one.tab.as_str(), "tb")] {
            assert!(id.starts_with(prefix), "{id}");
            assert_eq!(id.len(), prefix.len() + 32, "{id}");
            assert!(
                id[prefix.len()..].bytes().all(|b| b.is_ascii_hexdigit())
                    && id[prefix.len()..].bytes().all(|b| !b.is_ascii_uppercase()),
                "{id} is lowercase hex"
            );
        }
        assert_ne!(one.space, one.tab, "the space and tab never collide");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bound_root_declares_the_bound_pair_and_no_host_read() {
        let repo = std::env::temp_dir().join(format!("sac-place-root-{}", std::process::id()));
        std::fs::create_dir_all(&repo).unwrap();
        let world = world_with_root("spaManual", "tbManual");
        assert_eq!(
            for_dir(&world, &repo, &repo),
            Some(Placement {
                space: "spaManual".into(),
                tab: "tbManual".into()
            })
        );
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn an_unbound_root_is_unplaced_never_a_derived_space() {
        let repo = std::env::temp_dir().join(format!("sac-place-noroot-{}", std::process::id()));
        std::fs::create_dir_all(&repo).unwrap();
        let world = World::new();
        assert_eq!(for_dir(&world, &repo, &repo), None);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn a_recorded_workspace_declares_its_derivation() {
        let ws = std::env::temp_dir().join(format!("sac-place-ws-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let repo = std::env::temp_dir().join(format!("sac-place-repo-{}", std::process::id()));
        std::fs::create_dir_all(&repo).unwrap();
        let world = world_with_workspace(&ws);
        assert_eq!(for_dir(&world, &repo, &ws), Some(derived(&ws)));
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[test]
    fn an_unrecorded_non_root_destination_is_unplaced() {
        let repo = std::env::temp_dir().join(format!("sac-place-repo2-{}", std::process::id()));
        let other = std::env::temp_dir().join(format!("sac-place-other-{}", std::process::id()));
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        // even a bound root does not lend its pair to another directory
        let world = world_with_root("spaManual", "tbManual");
        assert_eq!(for_dir(&world, &repo, &other), None);
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&other);
    }
}
