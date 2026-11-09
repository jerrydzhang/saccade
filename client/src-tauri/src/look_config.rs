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
    /// The Herdr machine that hosts this tracker, as the local core
    /// lists it (t-24@r12): a bare path reads there through the local
    /// core. None on the hub, where the local core's own machine is
    /// read.
    pub tracker_machine: Option<String>,
    /// The publish identity override (t-24@r11): the actor name the
    /// suite's one record write records instead of the device account
    /// name. None follows sac's own rule (the USER environment
    /// variable, else "human"), and publish never refuses for lack of
    /// it.
    pub actor: Option<String>,
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(rename = "scheme-path")]
    scheme_path: Option<String>,
    #[serde(rename = "scheme-inline")]
    scheme_inline: Option<String>,
    weight: Option<String>,
    #[serde(rename = "tracker-machine")]
    tracker_machine: Option<String>,
    actor: Option<String>,
}

/// What a configuration load resolved to, before publication.
struct Loaded {
    config_found: bool,
    scheme_text: Option<String>,
    weight: Option<String>,
    problem: Option<String>,
    tracker_machine: Option<String>,
    actor: Option<String>,
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
                tracker_machine: None,
                actor: None,
            };
        }
        Err(error) => {
            return Loaded {
                config_found: true,
                scheme_text: None,
                weight: None,
                problem: Some(format!("look configuration unreadable: {error}")),
                tracker_machine: None,
                actor: None,
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
                tracker_machine: None,
                actor: None,
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
        tracker_machine: config
            .tracker_machine
            .map(|machine| machine.trim().to_string())
            .filter(|machine| !machine.is_empty()),
        actor: config
            .actor
            .map(|actor| actor.trim().to_string())
            .filter(|actor| !actor.is_empty()),
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
        tracker_machine: loaded.tracker_machine,
        actor: loaded.actor,
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
            tracker_machine: None,
            actor: None,
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
            tracker_machine: None,
            actor: None,
        },
    };
    let mut guard = shared.payload.lock().expect("look payload mutex");
    let next = payload_of(path, loaded, guard.generation + 1);
    let same = guard.config_found == next.config_found
        && guard.config_path == next.config_path
        && guard.scheme_text == next.scheme_text
        && guard.weight == next.weight
        && guard.problem == next.problem
        && guard.tracker_machine == next.tracker_machine
        && guard.actor == next.actor;
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

/// A change fingerprint of the files that feed the look: a content
/// hash of the config file and of the scheme file it names. Identity
/// by content, not stat: same-length edits inside one mtime second and
/// symlink retargets between identical-length, equal-mtime files (a
/// Nix store swap) both change the hash, and reading follows the link.
/// An unreadable file is None, so recovery reads as a change like
/// anything else.
type Fingerprint = Option<u64>;

fn content_hash(path: &Path) -> Fingerprint {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = DefaultHasher::new();
    hasher.write(&bytes);
    hasher.write(&(bytes.len() as u64).to_le_bytes());
    Some(hasher.finish())
}

fn fingerprint(path: Option<&Path>) -> (Fingerprint, Fingerprint) {
    let config = path.and_then(content_hash);
    let scheme = path
        .and_then(load_config_public)
        .and_then(|loaded| loaded.scheme_path)
        .and_then(|named| content_hash(Path::new(&named)));
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

/// One poll of the follow loop: sample, compare against the last
/// fingerprint, reload when it moved. Returns whether a reload ran, so
/// the tests can hold actual follow behaviour without the thread.
fn follow_tick(last: &mut (Fingerprint, Fingerprint), path: Option<&Path>) -> bool {
    let now = fingerprint(path);
    let changed = now != *last;
    *last = now;
    if changed {
        reload(path);
    }
    changed
}

/// Initialise the look state and start the follow thread. Called once
/// from the shell's setup, before the webview loads. The first
/// fingerprint is sampled before the initial reload, so a change racing
/// the start lands in the follower's `last` as already-seen only after
/// the reload published it - anything written between sample and
/// publish moves the fingerprint again and the first tick reloads.
pub fn init(config_dir: Option<PathBuf>) {
    let path = config_dir.map(|dir| dir.join("config.toml"));
    let _ = CONFIG_PATH.set(path.clone());
    let first = fingerprint(path.as_deref());
    reload(path.as_deref());
    std::thread::Builder::new()
        .name("look-follow".into())
        .spawn(move || follow_loop(first, path))
        .expect("look follow thread");
}

fn follow_loop(mut last: (Fingerprint, Fingerprint), path: Option<PathBuf>) {
    loop {
        std::thread::sleep(Duration::from_secs(1));
        follow_tick(&mut last, path.as_deref());
    }
}

#[tauri::command]
pub fn look_config() -> LookPayload {
    shared().payload.lock().expect("look payload mutex").clone()
}

/// The declared tracker machine, read on demand: a bare path resolves
/// through it (t-24@r12). None without a declaration - the hub case
/// reads the local core's own machine.
pub fn current_tracker_machine() -> Option<String> {
    shared()
        .payload
        .lock()
        .expect("look payload mutex")
        .tracker_machine
        .clone()
}

/// The publish identity override, read on demand (t-24@r11). None
/// follows sac's own rule on the caller side.
pub fn current_actor_override() -> Option<String> {
    shared()
        .payload
        .lock()
        .expect("look payload mutex")
        .actor
        .clone()
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

    // the tests that publish through the module's shared state run in
    // one process and must not interleave their generation assertions
    static SHARED_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn temp_dir(name: &str) -> PathBuf {
        // unique creation, artifacts retained: no preemptive recursive
        // delete, no predictable PID-keyed path (process id, a counter,
        // and nanos make collisions implausible; a collision just fails
        // the test loudly instead of reusing another run's files)
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "saccade-look-test-{name}-{}-{n}-{nanos}",
            std::process::id()
        ));
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
    fn the_tracker_machine_declaration_rides_the_same_file() {
        let dir = temp_dir("tracker-machine");
        let config = dir.join("config.toml");
        write(
            &config,
            "scheme-path = \"/a.yaml\"\ntracker-machine = \" hub-core \"\n",
        );
        let loaded = load_config(&config);
        assert!(loaded.config_found);
        assert_eq!(
            loaded.tracker_machine.as_deref(),
            Some("hub-core"),
            "the declaration is trimmed"
        );
        write(&config, "tracker-machine = \"   \"\n");
        let loaded = load_config(&config);
        assert_eq!(loaded.tracker_machine, None, "an empty declaration is none");
        write(&config, "scheme-path = \"/a.yaml\"\n");
        let loaded = load_config(&config);
        assert_eq!(
            loaded.tracker_machine, None,
            "no declaration is none, never a guess"
        );
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
        let _guard = SHARED_TEST_LOCK.lock().unwrap();
        let dir = temp_dir("reload");
        let config = dir.join("config.toml");
        write(&config, "scheme-inline = \"base00: \\\"101010\\\"\"\n");
        let path = Some(config.as_path());
        // generations accumulate across the process's shared look state;
        // the assertions are relative to this test's starting point
        let before = shared().payload.lock().unwrap().generation;
        assert!(reload(path), "first publish");
        let first = shared().payload.lock().unwrap().clone();
        assert_eq!(first.generation, before + 1);
        assert_eq!(first.scheme_text.as_deref(), Some("base00: \"101010\""));
        assert!(!reload(path), "identical reload publishes nothing");
        assert_eq!(shared().payload.lock().unwrap().generation, before + 1);

        // a wait against the current generation holds; a change wakes it.
        // The test drives the reload the follower's tick would do.
        let waiter = {
            let since = before + 1;
            let handle =
                std::thread::spawn(move || wait_for_change(since, Duration::from_secs(10)));
            std::thread::sleep(Duration::from_millis(80));
            write(&config, "scheme-inline = \"base00: \\\"202020\\\"\"\n");
            reload(path);
            handle
        };
        let woke = waiter.join().expect("waiter");
        assert_eq!(woke.generation, before + 2);
        assert_eq!(woke.scheme_text.as_deref(), Some("base00: \"202020\""));

        // an up-to-date wait answers after its hold, unchanged
        let start = std::time::Instant::now();
        let held = wait_for_change(woke.generation, Duration::from_millis(120));
        assert_eq!(held.generation, before + 2);
        assert!(start.elapsed() >= Duration::from_millis(100));
    }

    #[test]
    fn the_follow_tick_publishes_same_length_replacements_and_symlink_swaps() {
        let _guard = SHARED_TEST_LOCK.lock().unwrap();
        let dir = temp_dir("follow");

        // a same-length replacement: identity is content, so an edit
        // inside one mtime second moves the fingerprint
        let config = dir.join("config.toml");
        write(&config, "scheme-inline = \"base00: \\\"101010\\\"\"\n");
        let mut last = fingerprint(Some(config.as_path()));
        reload(Some(config.as_path()));
        let generation = shared().payload.lock().unwrap().generation;
        write(&config, "scheme-inline = \"base00: \\\"202020\\\"\"\n");
        assert!(
            follow_tick(&mut last, Some(config.as_path())),
            "same-length edit must publish"
        );
        assert_eq!(shared().payload.lock().unwrap().generation, generation + 1);
        assert_eq!(
            shared().payload.lock().unwrap().scheme_text.as_deref(),
            Some("base00: \"202020\"")
        );

        // an unchanged poll publishes nothing
        assert!(!follow_tick(&mut last, Some(config.as_path())));

        // a symlink retarget between identical-length files (the Nix
        // store swap pattern): reading follows the link, so the hash
        // moves whatever the mtimes and the inode say
        let scheme_dir = dir.join("schemes");
        fs::create_dir_all(&scheme_dir).expect("schemes dir");
        let a = scheme_dir.join("a.yaml");
        let b = scheme_dir.join("b.yaml");
        write(&a, "scheme: \"A\"\nbase00: \"111111\"\n");
        write(&b, "scheme: \"B\"\nbase00: \"222222\"\n");
        let link = dir.join("scheme.yaml");
        std::os::unix::fs::symlink(&a, &link).expect("symlink");
        write(&config, &format!("scheme-path = \"{}\"\n", link.display()));
        assert!(
            follow_tick(&mut last, Some(config.as_path())),
            "new scheme path must publish"
        );
        assert_eq!(
            shared().payload.lock().unwrap().scheme_text.as_deref(),
            Some("scheme: \"A\"\nbase00: \"111111\"\n")
        );

        // the swap itself: rename the new link over the old one, the
        // way a generation switch does
        let link2 = dir.join("scheme.yaml.new");
        std::os::unix::fs::symlink(&b, &link2).expect("symlink");
        fs::rename(&link2, &link).expect("rename over link");
        assert!(
            follow_tick(&mut last, Some(config.as_path())),
            "symlink swap must publish"
        );
        assert_eq!(
            shared().payload.lock().unwrap().scheme_text.as_deref(),
            Some("scheme: \"B\"\nbase00: \"222222\"\n")
        );
    }

    #[test]
    fn a_change_racing_the_start_is_caught_on_the_first_tick() {
        let _guard = SHARED_TEST_LOCK.lock().unwrap();
        let dir = temp_dir("race");
        let config = dir.join("config.toml");
        write(&config, "scheme-inline = \"base00: \\\"101010\\\"\"\n");
        // init's order: sample the fingerprint first, then load and
        // publish; the write races between the sample and the publish
        let mut last = fingerprint(Some(config.as_path()));
        reload(Some(config.as_path()));
        write(&config, "scheme-inline = \"base00: \\\"202020\\\"\"\n");
        // the first tick sees the moved fingerprint and reloads
        assert!(follow_tick(&mut last, Some(config.as_path())));
        assert_eq!(
            shared().payload.lock().unwrap().scheme_text.as_deref(),
            Some("base00: \"202020\"")
        );
    }
}
