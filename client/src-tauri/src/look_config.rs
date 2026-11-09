//! The look's host side (t-24@r3): the user's base16 scheme and text
//! weight live in a small configuration file in the platform's config
//! directory, read at start and re-read when it (or the scheme file it
//! names) changes. The host reads files and follows changes; the
//! frontend owns parsing, the contrast floor and the fallback notice.
//! Read-only: the only filesystem door here opens the human's own
//! config and the scheme file it names - never renderer-chosen paths.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

/// What the host answers about the local configuration. `generation`
/// advances on every reload; a look wait against a stale generation
/// answers with the current payload either way.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LookPayload {
    pub generation: u64,
    pub config_found: bool,
    pub config_path: Option<String>,
    pub scheme_text: Option<String>,
    pub weight: Option<String>,
    pub problem: Option<String>,
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(rename = "scheme-path")]
    scheme_path: Option<String>,
    #[serde(rename = "scheme-inline")]
    scheme_inline: Option<String>,
    weight: Option<String>,
}

/// What a configuration load resolved to, before publication.
struct Loaded {
    config_found: bool,
    scheme_text: Option<String>,
    weight: Option<String>,
    problem: Option<String>,
}

const WEIGHTS: [&str; 3] = ["regular", "medium", "semibold"];

/// Read and validate the configuration file at `path`. Missing file is
/// a normal answer (config_found false); an unreadable or invalid file
/// is a problem, stated; a scheme file that cannot be read is a
/// problem, stated.
fn load_config(path: &Path) -> Loaded {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Loaded {
                config_found: false,
                scheme_text: None,
                weight: None,
                problem: None,
            };
        }
        Err(error) => {
            return Loaded {
                config_found: true,
                scheme_text: None,
                weight: None,
                problem: Some(format!("look configuration unreadable: {error}")),
            };
        }
    };
    let config: ConfigFile = match toml::from_str(&text) {
        Ok(config) => config,
        Err(error) => {
            return Loaded {
                config_found: true,
                scheme_text: None,
                weight: None,
                problem: Some(format!("look configuration invalid: {error}")),
            };
        }
    };
    let mut problem: Option<String> = None;
    if config.scheme_path.is_some() && config.scheme_inline.is_some() {
        problem = Some("look configuration names both scheme-path and scheme-inline".into());
    }
    let weight = match config.weight.as_deref() {
        None => None,
        Some(w) if WEIGHTS.contains(&w) => Some(w.to_string()),
        Some(other) => {
            problem = Some(format!(
                "look configuration weight '{other}' is not one of regular, medium, semibold"
            ));
            None
        }
    };
    let scheme_text = match (&config.scheme_path, &config.scheme_inline) {
        (Some(path), _) if problem.is_none() => match std::fs::read_to_string(path) {
            Ok(text) => Some(text),
            Err(error) => {
                problem = Some(format!("scheme file unreadable at {path}: {error}"));
                None
            }
        },
        (None, Some(inline)) if problem.is_none() => Some(inline.clone()),
        _ => None,
    };
    Loaded {
        config_found: true,
        scheme_text,
        weight,
        problem,
    }
}

fn payload_of(path: Option<&Path>, loaded: Loaded, generation: u64) -> LookPayload {
    LookPayload {
        generation,
        config_found: loaded.config_found,
        config_path: path.map(|p| p.display().to_string()),
        scheme_text: loaded.scheme_text,
        weight: loaded.weight,
        problem: loaded.problem,
    }
}

/// The shared look state: the current payload, and the condvar the
/// look wait parks on. The payload's own generation is the truth.
struct Shared {
    payload: Mutex<LookPayload>,
    changed: Condvar,
}

static LOOK: OnceLock<Shared> = OnceLock::new();
static CONFIG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

fn shared() -> &'static Shared {
    LOOK.get_or_init(|| Shared {
        payload: Mutex::new(LookPayload {
            generation: 0,
            config_found: false,
            config_path: None,
            scheme_text: None,
            weight: None,
            problem: Some("the host has no look state yet".into()),
        }),
        changed: Condvar::new(),
    })
}

/// Load now and publish; returns true when the payload changed. The
/// comparison ignores the generation: identical content publishes
/// nothing, so a follow wait only wakes on a real change.
fn reload(path: Option<&Path>) -> bool {
    let shared = shared();
    let loaded = match path {
        Some(path) => load_config(path),
        None => Loaded {
            config_found: false,
            scheme_text: None,
            weight: None,
            problem: None,
        },
    };
    let mut guard = shared.payload.lock().expect("look payload mutex");
    let next = payload_of(path, loaded, guard.generation + 1);
    let same = guard.config_found == next.config_found
        && guard.config_path == next.config_path
        && guard.scheme_text == next.scheme_text
        && guard.weight == next.weight
        && guard.problem == next.problem;
    if same {
        return false;
    }
    *guard = next;
    drop(guard);
    shared.changed.notify_all();
    true
}

/// The bounded look wait: answers as soon as the generation passes
/// `since`, or with the current payload after the hold expires. The
/// frontend compares generations and ignores a no-change answer.
fn wait_for_change(since: u64, hold: Duration) -> LookPayload {
    let shared = shared();
    let mut guard = shared.payload.lock().expect("look payload mutex");
    let deadline = std::time::Instant::now() + hold;
    while guard.generation <= since {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return guard.clone();
        }
        let (guard2, _timeout) = shared
            .changed
            .wait_timeout(guard, left)
            .expect("look condvar");
        guard = guard2;
    }
    guard.clone()
}

/// A change fingerprint of the files that feed the look: the config
/// file and the scheme file it names. Mtime and size; content is read
/// only when the fingerprint moves. A file that cannot be stat'ed is
/// None on both sides, so recovery reads as a change like anything
/// else.
type Stat = Option<(i64, u64)>;

fn stat(path: &Path) -> Stat {
    let meta = std::fs::metadata(path).ok()?;
    use std::os::unix::fs::MetadataExt;
    Some((meta.mtime(), meta.len()))
}

fn fingerprint(path: Option<&Path>) -> (Stat, Stat) {
    let config = path.and_then(stat);
    let scheme = path
        .and_then(load_config_public)
        .and_then(|loaded| loaded.scheme_path)
        .and_then(|named| stat(Path::new(&named)));
    (config, scheme)
}

// test seam: load_config with the scheme path kept for fingerprinting
struct LoadedPublic {
    scheme_path: Option<String>,
}

fn load_config_public(path: &Path) -> Option<LoadedPublic> {
    let text = std::fs::read_to_string(path).ok()?;
    let config: ConfigFile = toml::from_str(&text).ok()?;
    Some(LoadedPublic {
        scheme_path: config.scheme_path,
    })
}

/// Initialise the look state and start the follow thread. Called once
/// from the shell's setup, before the webview loads.
pub fn init(config_dir: Option<PathBuf>) {
    let path = config_dir.map(|dir| dir.join("config.toml"));
    let _ = CONFIG_PATH.set(path.clone());
    reload(path.as_deref());
    std::thread::Builder::new()
        .name("look-follow".into())
        .spawn(move || follow_loop(path))
        .expect("look follow thread");
}

fn follow_loop(path: Option<PathBuf>) {
    let mut last = fingerprint(path.as_deref());
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let now = fingerprint(path.as_deref());
        if now != last {
            last = now;
            reload(path.as_deref());
        }
    }
}

#[tauri::command]
pub fn look_config() -> LookPayload {
    shared().payload.lock().expect("look payload mutex").clone()
}

#[tauri::command]
pub async fn wait_look_config(since: u64) -> LookPayload {
    // the bounded hold rides a blocking thread; the async command never
    // blocks the runtime's workers
    let handle = tauri::async_runtime::spawn_blocking(move || {
        wait_for_change(since, Duration::from_secs(25))
    });
    handle.await.expect("look wait task")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("saccade-look-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn write(path: &Path, text: &str) {
        fs::write(path, text).expect("write");
    }

    #[test]
    fn a_missing_config_is_a_normal_answer() {
        let dir = temp_dir("missing");
        let loaded = load_config(&dir.join("config.toml"));
        assert!(!loaded.config_found);
        assert!(loaded.problem.is_none());
        assert!(loaded.scheme_text.is_none());
    }

    #[test]
    fn a_path_config_reads_the_scheme_file() {
        let dir = temp_dir("path");
        let scheme = dir.join("scheme.yaml");
        write(&scheme, "scheme: \"Mountain\"\nbase00: \"100f0e\"\n");
        let config = dir.join("config.toml");
        write(
            &config,
            &format!(
                "scheme-path = \"{}\"\nweight = \"medium\"\n",
                scheme.display()
            ),
        );
        let loaded = load_config(&config);
        assert!(loaded.config_found);
        assert_eq!(
            loaded.scheme_text.as_deref(),
            Some("scheme: \"Mountain\"\nbase00: \"100f0e\"\n")
        );
        assert_eq!(loaded.weight.as_deref(), Some("medium"));
        assert!(loaded.problem.is_none());
    }

    #[test]
    fn an_inline_config_carries_the_text() {
        let dir = temp_dir("inline");
        let config = dir.join("config.toml");
        write(
            &config,
            "scheme-inline = \"\"\"\nscheme: \"Paper\"\nbase00: \"f4f1ea\"\n\"\"\"\n",
        );
        let loaded = load_config(&config);
        assert_eq!(
            loaded.scheme_text.as_deref(),
            Some("scheme: \"Paper\"\nbase00: \"f4f1ea\"\n")
        );
        assert_eq!(loaded.weight, None);
        assert!(loaded.problem.is_none());
    }

    #[test]
    fn naming_both_scheme_keys_is_a_stated_problem() {
        let dir = temp_dir("both");
        let config = dir.join("config.toml");
        write(
            &config,
            "scheme-path = \"/a.yaml\"\nscheme-inline = \"base00: \\\"000000\\\"\"\n",
        );
        let loaded = load_config(&config);
        assert!(loaded.config_found);
        assert!(loaded.scheme_text.is_none());
        assert!(loaded
            .problem
            .as_deref()
            .unwrap()
            .contains("both scheme-path and scheme-inline"));
    }

    #[test]
    fn an_unknown_key_and_a_bad_weight_state_their_problem() {
        let dir = temp_dir("bad");
        let config = dir.join("config.toml");
        write(&config, "scheme-path = \"/a.yaml\"\nsheme = \"typo\"\n");
        let loaded = load_config(&config);
        assert!(loaded.problem.as_deref().unwrap().contains("unknown field"));

        let config2 = dir.join("config2.toml");
        write(&config2, "scheme-path = \"/a.yaml\"\nweight = \"bold\"\n");
        let loaded2 = load_config(&config2);
        assert!(loaded2.problem.as_deref().unwrap().contains("weight"));
        assert_eq!(loaded2.weight, None);
    }

    #[test]
    fn a_missing_scheme_file_is_a_stated_problem() {
        let dir = temp_dir("noscheme");
        let config = dir.join("config.toml");
        write(&config, "scheme-path = \"/nonexistent/scheme.yaml\"\n");
        let loaded = load_config(&config);
        assert!(loaded.config_found);
        assert!(loaded.scheme_text.is_none());
        assert!(loaded
            .problem
            .as_deref()
            .unwrap()
            .contains("scheme file unreadable"));
    }

    #[test]
    fn reload_publishes_once_per_change_and_the_wait_wakes() {
        let dir = temp_dir("reload");
        let config = dir.join("config.toml");
        write(&config, "scheme-inline = \"base00: \\\"101010\\\"\"\n");
        let path = Some(config.as_path());
        assert!(reload(path), "first publish");
        let first = shared().payload.lock().unwrap().clone();
        assert_eq!(first.generation, 1);
        assert_eq!(first.scheme_text.as_deref(), Some("base00: \"101010\""));
        assert!(!reload(path), "identical reload publishes nothing");
        assert_eq!(shared().payload.lock().unwrap().generation, 1);

        // a wait against the current generation holds; a change wakes it
        let waiter = {
            let handle = std::thread::spawn(|| wait_for_change(1, Duration::from_secs(10)));
            std::thread::sleep(Duration::from_millis(80));
            write(&config, "scheme-inline = \"base00: \\\"202020\\\"\"\n");
            // the fingerprint's mtime granularity can swallow a fast
            // rewrite; force the reload the watcher would do
            reload(path);
            handle
        };
        let woke = waiter.join().expect("waiter");
        assert_eq!(woke.generation, 2);
        assert_eq!(woke.scheme_text.as_deref(), Some("base00: \"202020\""));

        // an up-to-date wait answers after its hold, unchanged
        let start = std::time::Instant::now();
        let held = wait_for_change(woke.generation, Duration::from_millis(120));
        assert_eq!(held.generation, 2);
        assert!(start.elapsed() >= Duration::from_millis(100));
    }
}
