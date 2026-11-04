//! The door's placement mechanics: every terminal a saccade door
//! spawns declares where it lives, at birth. The space id is stable
//! per workspace identity - the canonical directory - so every launch
//! and every resume of one workspace declares the same space, and an
//! already-standing space for that identity keeps its terminals. When
//! a live terminal of the workspace carries a placement, the door
//! declares that one (the standing space survives); otherwise the
//! stable derivation stands. Ids wear the app's own shape: a prefix
//! and 32 lowercase hex characters.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::herdr::{HerdrCore, HostTerminal};
use crate::host_contract::HostAddress;

/// The derived space and tab ids one directory declares.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub space: String,
    pub tab: String,
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

/// The placement a launch declares for `dir`: the standing space of a
/// live terminal already sitting in the workspace, else the stable
/// derivation. A lookup miss (no table, no live candidates, no
/// labels) is the derivation - the declaration is always defined.
pub fn for_dir(host: &HerdrCore, dir: &Path) -> Placement {
    let canonical = canonical(dir);
    let fallback = derived(&canonical);
    let Ok(rows) = host.terminals() else {
        return fallback;
    };
    let mut candidates: Vec<&HostTerminal> = rows
        .iter()
        .filter(|row| row.state == "live" && under(&canonical, &row.cwd))
        .collect();
    candidates.sort_by(|a, b| a.address.cmp(&b.address));
    for row in candidates {
        if let Some(standing) = host.placement_of(&HostAddress(row.address.clone())) {
            return Placement {
                space: standing.space,
                tab: standing.tab.unwrap_or_else(|| fallback.tab.clone()),
            };
        }
    }
    fallback
}

/// One directory contains another cwd: equal, or a path below it.
pub fn under(dir: &Path, cwd: &str) -> bool {
    let dir = canonical(dir).to_string_lossy().into_owned();
    let dir = dir.trim_end_matches('/');
    if dir.is_empty() {
        return false;
    }
    cwd == dir
        || cwd
            .strip_prefix(dir)
            .is_some_and(|rest| rest.starts_with('/'))
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

    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    fn fake(name: &str, rows: &str, gets: &[(&str, &str)]) -> (HerdrCore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("sac-place-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("hc");
        std::fs::write(
            &script,
            "#!/bin/sh\nhere=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nprintf '%s\\n' \"$@\" >> \"$here/argv-$1.log\"\nif [ \"$1\" = ls ]; then\n  printf '{\"terminals\":%s}\\n' \"$(cat \"$here/rows.json\" 2>/dev/null)\"\n  exit 0\nfi\nif [ \"$1\" = get ] && [ -f \"$here/get-$2.json\" ]; then\n  cat \"$here/get-$2.json\"; printf '\\n'; exit 0\nfi\nif [ \"$1\" = get ]; then\n  printf '{\"error\":{\"kind\":\"not_found\",\"message\":\"no terminal\"}}\\n' >&2\n  exit 3\nfi\nprintf '{}\\n'\nexit 0\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(dir.join("rows.json"), rows).unwrap();
        for (address, record) in gets {
            std::fs::write(dir.join(format!("get-{address}.json")), record).unwrap();
        }
        for _ in 0..50 {
            match std::process::Command::new(&script).arg("--warmup").output() {
                Ok(_) => break,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => panic!("the fake script never became spawnable: {e}"),
            }
        }
        (HerdrCore::over(&script), dir)
    }

    fn argv_log(dir: &Path, verb: &str) -> Vec<String> {
        std::fs::read_to_string(dir.join(format!("argv-{verb}.log")))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
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
    fn under_matches_equal_and_nested_paths_never_prefix_stretches() {
        let dir = Path::new("/home/x/ws");
        assert!(under(dir, "/home/x/ws"));
        assert!(under(dir, "/home/x/ws/sub/deeper"));
        assert!(!under(dir, "/home/x/ws-other"));
        assert!(!under(dir, "/home/x"));
        assert!(!under(dir, "/elsewhere"));
        assert!(under(Path::new("/home/x/ws/"), "/home/x/ws/nested"));
        // an unresolvable dir still matches by its own string
        assert!(under(Path::new("/no/such/dir"), "/no/such/dir/inside"));
        assert!(!under(Path::new("/no/such/dir"), "/no/such/dir-other"));
    }

    #[test]
    fn a_live_labeled_candidate_lends_its_standing_placement() {
        let ws = std::env::temp_dir().join(format!("sac-place-ws-{}", std::process::id()));
        std::fs::create_dir_all(ws.join("sub")).unwrap();
        let rows = format!(
            "[{{\"address\":\"t5\",\"cwd\":\"{}\",\"state\":\"live\"}},\
             {{\"address\":\"t6\",\"cwd\":\"/elsewhere\",\"state\":\"live\"}},\
             {{\"address\":\"t7\",\"cwd\":\"{}\",\"state\":\"finished\"}}]",
            ws.join("sub").display(),
            ws.display(),
        );
        let (host, dir) = fake(
            "standing",
            &rows,
            &[(
                "t5",
                "{\"state\":\"live\",\"labels\":{\"herdr.space\":\"spaStanding\",\"herdr.tab\":\"tbStanding\"}}",
            )],
        );
        let placed = for_dir(&host, &ws);
        assert_eq!(placed.space, "spaStanding");
        assert_eq!(placed.tab, "tbStanding");
        // only the live under-dir candidate was asked: the outside and
        // finished rows never reached the record read
        assert_eq!(
            argv_log(&dir, "get"),
            vec!["get".to_string(), "t5".to_string(), "--json".to_string()]
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn a_standing_space_without_a_tab_takes_the_derived_tab() {
        let ws = std::env::temp_dir().join(format!("sac-place-notab-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let rows = format!(
            "[{{\"address\":\"t5\",\"cwd\":\"{}\",\"state\":\"live\"}}]",
            ws.display()
        );
        let (host, dir) = fake(
            "standing-notab",
            &rows,
            &[(
                "t5",
                "{\"state\":\"live\",\"labels\":{\"herdr.space\":\"spaStanding\"}}",
            )],
        );
        let placed = for_dir(&host, &ws);
        assert_eq!(placed.space, "spaStanding");
        assert_eq!(placed.tab, derived(&ws).tab);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn the_lowest_address_wins_among_standing_candidates() {
        let ws = std::env::temp_dir().join(format!("sac-place-order-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let rows = format!(
            "[{{\"address\":\"t9\",\"cwd\":\"{}\",\"state\":\"live\"}},\
             {{\"address\":\"t2\",\"cwd\":\"{}\",\"state\":\"live\"}}]",
            ws.display(),
            ws.display(),
        );
        let (host, dir) = fake(
            "standing-order",
            &rows,
            &[
                (
                    "t2",
                    "{\"state\":\"live\",\"labels\":{\"herdr.space\":\"spaEarly\"}}",
                ),
                (
                    "t9",
                    "{\"state\":\"live\",\"labels\":{\"herdr.space\":\"spaLate\"}}",
                ),
            ],
        );
        assert_eq!(for_dir(&host, &ws).space, "spaEarly");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn every_lookup_miss_falls_back_to_the_derivation() {
        let ws = std::env::temp_dir().join(format!("sac-place-miss-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let expect = derived(&ws);
        // no rows at all
        let (host, dir) = fake("miss-empty", "", &[]);
        assert_eq!(for_dir(&host, &ws), expect);
        let _ = std::fs::remove_dir_all(&dir);
        // live candidates under the dir, none labeled
        let rows = format!(
            "[{{\"address\":\"t5\",\"cwd\":\"{}\",\"state\":\"live\"}}]",
            ws.display()
        );
        let (host, dir) = fake("miss-unlabeled", &rows, &[]);
        assert_eq!(for_dir(&host, &ws), expect);
        let _ = std::fs::remove_dir_all(&dir);
        // the table itself unreadable
        let (host, dir) = fake("miss-broken", "", &[]);
        std::fs::write(dir.join("hc"), "#!/bin/sh\nexit 1\n").unwrap();
        std::fs::set_permissions(dir.join("hc"), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(for_dir(&host, &ws), expect);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&ws);
    }
}
