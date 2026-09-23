//! The binary's self-carried curriculum. The skill files on disk are
//! the source; the embeds here are their pinned copy inside the
//! binary, and the `skill` verbs deploy and verify that copy in any
//! directory the binary serves.

use std::path::Path;

/// The skill's home, relative to the directory a session works in.
pub const HOME: &str = ".agents/skills/saccade";

/// The refusal at a differing copy: deletion is the force flag.
pub const DIFFERS: &str = ".agents/skills/saccade differs from this binary's skill — delete the directory and run sac skill install to deploy this version";

/// The deployed version line's key, as it stands in SKILL.md's frontmatter.
const VERSION_KEY: &str = "x-saccade-version:";

/// The embedded curriculum: file names as they deploy, bytes as compiled.
pub fn embedded() -> [(&'static str, &'static str); 3] {
    [
        (
            "SKILL.md",
            include_str!("../.agents/skills/saccade/SKILL.md"),
        ),
        (
            "errors.md",
            include_str!("../.agents/skills/saccade/errors.md"),
        ),
        (
            "diagnosing.md",
            include_str!("../.agents/skills/saccade/diagnosing.md"),
        ),
    ]
}

/// The frontmatter block: the keys between the fences, and the text
/// from the closing fence onward.
fn frontmatter(text: &str) -> Option<(&str, &str)> {
    let body = text.strip_prefix("---\n")?;
    let close = body.find("\n---\n")?;
    Some((&body[..close], &body[close..]))
}

/// The bytes one skill file deploys as: the embedded copy, SKILL.md
/// carrying the deploying binary's version inside its frontmatter.
fn deploy_bytes(name: &str, text: &str, version: &str) -> String {
    if name != "SKILL.md" {
        return text.to_string();
    }
    match frontmatter(text) {
        Some((keys, rest)) => format!("---\n{keys}\n{VERSION_KEY} {version}{rest}"),
        None => text.to_string(),
    }
}

/// The curriculum beneath the stamp: the version line names the deploy,
/// not the teaching, so comparison drops it.
fn without_version_line(text: &str) -> String {
    match frontmatter(text) {
        Some((keys, rest)) => format!(
            "---\n{}{rest}",
            keys.lines()
                .filter(|l| !l.starts_with(VERSION_KEY))
                .collect::<Vec<_>>()
                .join("\n")
        ),
        None => text.to_string(),
    }
}

/// The version a deployed SKILL.md's frontmatter carries, if any.
fn carried_version(text: &str) -> Option<String> {
    let (keys, _) = frontmatter(text)?;
    keys.lines()
        .find_map(|l| l.strip_prefix(VERSION_KEY))
        .map(|v| v.trim().to_string())
}

/// One file's copy against its embedded source; SKILL.md's comparison
/// ignores the deployed version line.
fn teaches(name: &str, copy: &str, source: &str) -> bool {
    if name == "SKILL.md" {
        without_version_line(copy) == source
    } else {
        copy == source
    }
}

fn read(home: &Path, name: &str) -> std::io::Result<String> {
    std::fs::read_to_string(home.join(name))
}

/// What `skill check` can say about the copy under a directory: the
/// version line the copy carries, when it carries one.
pub enum Verdict {
    InSync(Option<String>),
    Drifted(Option<String>),
    Absent,
}

pub fn check(cwd: &Path) -> Verdict {
    let home = cwd.join(HOME);
    if !home.is_dir() {
        return Verdict::Absent;
    }
    let carries = read(&home, "SKILL.md")
        .ok()
        .and_then(|t| carried_version(&t));
    let in_sync = embedded()
        .iter()
        .all(|(name, source)| read(&home, name).is_ok_and(|copy| teaches(name, &copy, source)));
    if in_sync {
        Verdict::InSync(carries)
    } else {
        Verdict::Drifted(carries)
    }
}

/// What `skill install` did.
pub enum Deployed {
    /// The three files landed, SKILL.md stamped with this binary's version.
    Wrote,
    /// A byte-identical copy already stood; nothing was touched.
    Untouched,
}

/// Deploy the embedded skill into `cwd`'s skill home: a byte-identical
/// copy is a no-op, a differing one refuses.
pub fn install(cwd: &Path, version: &str) -> Result<Deployed, String> {
    let home = cwd.join(HOME);
    if home.exists() {
        let exact = embedded().iter().all(|(name, source)| {
            read(&home, name).is_ok_and(|copy| copy == deploy_bytes(name, source, version))
        });
        return if exact {
            Ok(Deployed::Untouched)
        } else {
            Err(DIFFERS.to_string())
        };
    }
    std::fs::create_dir_all(&home).map_err(|e| format!("creating {}: {e}", home.display()))?;
    for (name, source) in embedded() {
        std::fs::write(home.join(name), deploy_bytes(name, source, version))
            .map_err(|e| format!("writing {}: {e}", home.join(name).display()))?;
    }
    Ok(Deployed::Wrote)
}
