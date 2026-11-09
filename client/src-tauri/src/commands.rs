//! Read-only declaration and asset reads, plus the suite's one write:
//! the explainer annotation submit (t-24@r21). The guard mirrors the
//! frontend resolve rules exactly: relative segments only, no parent
//! traversal, no backslash separators, no absolute paths, no NUL bytes.
//! Canonicalized
//! prefix checking runs second so a guard bypass still cannot escape.
//!
//! Local locations read through std::fs exactly as before. A location
//! that names a machine reads through the LOCAL herdr-core's face
//! instead (file.read for text; fixed-argv exec for bytes and remote
//! path resolution; file.edit for the one annotation submit -
//! exact-once, all-or-none), with containment proven on the remote side
//! and every machine state surfacing as a structured error. The core
//! socket is the only door here, and it opens only the calls named
//! here; the renderer gets no socket, no shell and no machine
//! management.

use board_viewer_core::{
    guard_relative, CoreClient, CoreFail, FileEdit, SocketTransport, Subscription,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::look_config;

/// The lanes' test-open channel (BOARD_VIEWER_TEST_OPEN): the one
/// location the app opens at startup, riding the same model.open the
/// inspection surface uses. Test-only machinery, named as such
/// (t-24@r11); a human run has the variable unset and starts empty -
/// the launch argument and the local-device path form left the human
/// surface with it.
#[tauri::command]
pub fn test_open() -> Option<String> {
    let spec = std::env::var("BOARD_VIEWER_TEST_OPEN")
        .ok()
        .filter(|s| !s.is_empty());
    eprintln!("[bridge] test_open invoked: {:?}", spec);
    spec
}

/// The declared tracker machine (t-24@r12): the Herdr machine that
/// hosts this tracker, read on demand by a bare-path open. None on the
/// hub, where the local core's own machine is read.
#[tauri::command]
pub fn tracker_machine() -> Option<String> {
    look_config::current_tracker_machine()
}

/// The identity the suite's one record write records (t-24@r11): the
/// human tier and, as the actor, the configured override, else the
/// device account name as `sac` derives it without SACCADE_ACTOR
/// ($USER), else "human". Publish never refuses for lack of it.
/// Whether the shell feasibility probe is active (t-19, test-only
/// machinery): the main window probes its localStorage only when the
/// lane set BOARD_VIEWER_SHELL_PROBE, so a human run writes nothing.
#[tauri::command]
pub fn shell_probe_active() -> bool {
    std::env::var("BOARD_VIEWER_SHELL_PROBE")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

/// The shell feasibility probe's report (t-19, test-only machinery):
/// one line per window with the localStorage probe keys that window
/// sees, read from the native lane's app log as the storage-sharing
/// evidence.
#[derive(Deserialize)]
pub struct ShellProbeReport {
    pub label: String,
    pub sees_main: Option<String>,
    pub sees_second: Option<String>,
}

#[tauri::command]
pub fn shell_probe_report(report: ShellProbeReport) {
    eprintln!(
        "[shell-probe] window {} sees main={:?} second={:?}",
        report.label, report.sees_main, report.sees_second
    );
}

#[derive(Serialize)]
pub struct PublishIdentity {
    pub actor: String,
    pub tier: String,
}

#[tauri::command]
pub fn publish_identity() -> PublishIdentity {
    let actor = look_config::current_actor_override()
        .or_else(|| std::env::var("USER").ok())
        .filter(|name| !name.trim().is_empty())
        .map(|name| name.trim().to_string())
        .unwrap_or_else(|| "human".into());
    eprintln!("[bridge] publish_identity: {actor}/human");
    PublishIdentity {
        actor,
        tier: "human".into(),
    }
}

/// This device's own machine id (t-24@r14): the trimmed
/// `/etc/machine-id`, read on demand for the hub proof against the
/// tracker root page's fact. None when unreadable or empty, so an
/// unavailable id never proves anything.
#[tauri::command]
pub fn machine_id() -> Option<String> {
    let id = std::fs::read_to_string("/etc/machine-id")
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|id| !id.is_empty());
    eprintln!(
        "[bridge] machine_id invoked: {}",
        if id.is_some() { "present" } else { "absent" }
    );
    id
}

#[derive(Serialize, Deserialize)]
pub struct DeclaredText {
    pub text: String,
}

/// What one landed edit answers: the file's new size. The frontend
/// re-opens the file through its read path after a landed edit, so no
/// text rides back here.
#[derive(Serialize, Deserialize, Debug)]
pub struct EditedText {
    pub bytes: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BridgeError {
    pub code: String,
    pub message: String,
}

fn deny(reason: &str) -> BridgeError {
    BridgeError {
        code: "asset-denied".into(),
        message: reason.into(),
    }
}

fn missing(path: &str) -> BridgeError {
    BridgeError {
        code: "asset-missing".into(),
        message: format!("not found or unreadable: {path}"),
    }
}

fn core_failure(fail: CoreFail) -> BridgeError {
    BridgeError {
        code: fail.code().into(),
        message: fail.to_string(),
    }
}

/// The one file write the suite makes (t-14 ratified item 3, t-24@r21):
/// an explainer annotation submit. One exact-text replacement through
/// the local core's `file.edit` - all-or-none, old found exactly once -
/// on the machine the caller resolved (the same rule the bare-path
/// reads ride: the local core on the hub, the declared tracker machine
/// elsewhere). Identity-free: a file write lands no record, so the
/// log line below is the act's trace.
#[tauri::command]
pub fn edit_declaration(
    path: String,
    old: String,
    new: String,
    on: Option<String>,
) -> Result<EditedText, BridgeError> {
    if path.is_empty() {
        return Err(deny("empty path"));
    }
    if !path.starts_with('/') {
        return Err(deny("the edit path must be absolute"));
    }
    if path.contains('\0') || old.contains('\0') || new.contains('\0') {
        return Err(deny("NUL byte in the edit"));
    }
    let machine = match on.as_deref() {
        None => return Err(deny("the explainer edit names no machine; bare paths only")),
        Some(machine) => machine,
    };
    let client = core_client()?;
    let outcome = match client.edit_text(machine, &path, &[FileEdit { old, new }]) {
        Ok(outcome) => outcome,
        Err(fail) => {
            let _ = log_bridge(
                "edit_declaration",
                &format!("{}{path}", core_machine_label(machine)),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
    let _ = log_bridge(
        "edit_declaration",
        &format!("{}{path}", core_machine_label(machine)),
        &format!("{} bytes", outcome.bytes),
    );
    Ok(EditedText {
        bytes: outcome.bytes,
    })
}

fn core_client() -> Result<CoreClient<SocketTransport>, BridgeError> {
    let path = SocketTransport::default_path().map_err(core_failure)?;
    Ok(CoreClient::new(SocketTransport::new(path)))
}

/// The Herdr machine a command rides. `Some("")` is the local core's
/// own machine (t-24@r12: the hub's bare-path reads go through the
/// local core, never std::fs); `Some(name)` is a named machine; None
/// is the host file reader, test-only machinery since t-24@r11.
enum CoreRead {
    Local,
    Named(String),
    HostFile,
}

fn core_read_of(on: &Option<String>) -> CoreRead {
    match on.as_deref() {
        None => CoreRead::HostFile,
        Some("") => CoreRead::Local,
        Some(machine) => CoreRead::Named(machine.to_string()),
    }
}

/// The frontend already applied these rules; repeat them here so the
/// host boundary holds even against a tampered renderer.
fn canonical_prefix_holds(base: &std::path::Path, joined: &std::path::Path) -> bool {
    match (base.canonicalize(), joined.canonicalize()) {
        (Ok(base), Ok(joined)) => joined.starts_with(base),
        _ => false,
    }
}

fn log_bridge(operation: &str, path: &str, outcome: &str) -> std::io::Result<()> {
    // A useful outcome line on the app's own stderr: the human running
    // the foreground app sees what the bridge actually read.
    eprintln!("[bridge] {operation} {path}: {outcome}");
    Ok(())
}

fn read_text_resolved(base: &str, relative: Option<&str>) -> Result<DeclaredText, BridgeError> {
    let base_path = std::path::Path::new(base);
    let target = match relative {
        None => base_path.to_path_buf(),
        Some(rel) => {
            guard_relative(rel).map_err(deny)?;
            base_path.join(rel)
        }
    };
    if !canonical_prefix_holds(base_path, &target) {
        return Err(deny("resolved path left the declaration directory"));
    }
    let bytes = std::fs::read(&target).map_err(|_| {
        let _ = log_bridge("read_declaration", &target.to_string_lossy(), "missing");
        missing(&target.to_string_lossy())
    })?;
    let _ = log_bridge(
        "read_declaration",
        &target.to_string_lossy(),
        &format!("{} bytes", bytes.len()),
    );
    String::from_utf8(bytes)
        .map(|text| DeclaredText { text })
        .map_err(|_| BridgeError {
            code: "asset-denied".into(),
            message: "declaration bytes are not UTF-8 text".into(),
        })
}

fn read_declaration_remote(machine: &str, path: &str) -> Result<DeclaredText, BridgeError> {
    if !path.starts_with('/') {
        return Err(deny("remote declaration path must be absolute"));
    }
    if path.contains('\0') {
        return Err(deny("NUL byte in path"));
    }
    let client = core_client()?;
    let text = match client.read_text(machine, path) {
        Ok(text) => text,
        Err(fail) => {
            let _ = log_bridge(
                "read_declaration",
                &format!("{}{path}", core_machine_label(machine)),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
    let _ = log_bridge(
        "read_declaration",
        &format!("{}{path}", core_machine_label(machine)),
        &format!("{} bytes", text.len()),
    );
    Ok(DeclaredText { text })
}

/// The log label of a core machine: the local core's own machine has
/// no name, and the log says so plainly.
fn core_machine_label(machine: &str) -> String {
    if machine.is_empty() {
        "(local core)".to_string()
    } else {
        format!("{machine}:")
    }
}

#[tauri::command]
pub fn read_declaration(path: String, on: Option<String>) -> Result<DeclaredText, BridgeError> {
    if path.is_empty() {
        return Err(deny("empty path"));
    }
    match core_read_of(&on) {
        CoreRead::Named(machine) => read_declaration_remote(&machine, &path),
        CoreRead::Local => read_declaration_remote("", &path),
        CoreRead::HostFile => read_text_resolved(&path, None),
    }
}

fn read_asset_remote(machine: &str, base: &str, relative: &str) -> Result<Vec<u8>, BridgeError> {
    if !base.starts_with('/') {
        return Err(deny("remote base path must be absolute"));
    }
    if base.contains('\0') {
        return Err(deny("NUL byte in path"));
    }
    let client = core_client()?;
    let bytes = match client.asset_bytes(machine, base, relative) {
        Ok(bytes) => bytes,
        Err(fail) => {
            let _ = log_bridge(
                "read_asset",
                &format!("{machine}:{base}/{relative}"),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
    let _ = log_bridge(
        "read_asset",
        &format!("{machine}:{base}/{relative}"),
        &format!("{} bytes", bytes.len()),
    );
    Ok(bytes)
}

#[tauri::command]
pub fn read_asset(
    base: String,
    relative: String,
    on: Option<String>,
) -> Result<Vec<u8>, BridgeError> {
    guard_relative(&relative).map_err(deny)?;
    match core_read_of(&on) {
        CoreRead::Named(machine) => read_asset_remote(&machine, &base, &relative),
        CoreRead::Local => read_asset_local_core(&base, &relative),
        CoreRead::HostFile => {
            let base_path = std::path::Path::new(&base);
            let target = base_path.join(&relative);
            if !canonical_prefix_holds(base_path, &target) {
                return Err(deny("resolved path left the declaration directory"));
            }
            let bytes = std::fs::read(&target).map_err(|_| {
                let _ = log_bridge("read_asset", &target.to_string_lossy(), "missing");
                missing(&target.to_string_lossy())
            })?;
            let _ = log_bridge(
                "read_asset",
                &target.to_string_lossy(),
                &format!("{} bytes", bytes.len()),
            );
            Ok(bytes)
        }
    }
}

/// One asset through the local core's own machine (t-24@r12): the same
/// fixed-argv read the named machines use, with containment proven on
/// the core side, never a std::fs shortcut.
fn read_asset_local_core(base: &str, relative: &str) -> Result<Vec<u8>, BridgeError> {
    if !base.starts_with('/') {
        return Err(deny("remote base path must be absolute"));
    }
    if base.contains('\0') {
        return Err(deny("NUL byte in base path"));
    }
    let client = core_client()?;
    let bytes = match client.asset_bytes("", base, relative) {
        Ok(bytes) => bytes,
        Err(fail) => {
            let _ = log_bridge(
                "read_asset",
                &format!("(local core){base}/{relative}"),
                &format!("failed: {}", fail.code()),
            );
            return Err(core_failure(fail));
        }
    };
    let _ = log_bridge(
        "read_asset",
        &format!("(local core){base}/{relative}"),
        &format!("{} bytes", bytes.len()),
    );
    Ok(bytes)
}

/// A static identity statement for the machine-readable surface: what this
/// binary is, never a claim about any session or tracker state.
#[tauri::command]
pub fn viewer_identity() -> serde_json::Value {
    serde_json::json!({
        "component": "saccade-board-viewer",
        "version": env!("CARGO_PKG_VERSION"),
        "read_only": true,
    })
}

#[cfg(test)]
mod tests {
    use super::{core_read_of, guard_relative, CoreRead, DeckFeedMessage};

    #[test]
    fn an_edit_refuses_non_absolute_and_nul_paths_before_any_wire() {
        for (path, old, new) in [
            ("", "a".to_string(), "b".to_string()),
            ("relative/path.yaml", "a".to_string(), "b".to_string()),
            ("/abs\0path", "a".to_string(), "b".to_string()),
            ("/abs", "a\0".to_string(), "b".to_string()),
            ("/abs", "a".to_string(), "b\0".to_string()),
        ] {
            let failure = super::edit_declaration(path.into(), old, new, Some("".into()))
                .expect_err("refused");
            assert_eq!(failure.code, "asset-denied", "{failure:?}");
        }
    }

    #[test]
    fn an_edit_without_a_resolved_machine_is_refused() {
        let failure = super::edit_declaration("/abs/x.yaml".into(), "a".into(), "b".into(), None)
            .expect_err("refused");
        assert_eq!(failure.code, "asset-denied");
        assert!(failure.message.contains("bare paths"), "{failure:?}");
    }

    #[test]
    fn a_deck_name_must_be_the_viewer_s_own_form() {
        let good = format!("deck-{}", "ab".repeat(12));
        assert!(super::deck_name_guard(&good).is_ok());
        let upper = format!("deck-{}", "AB".repeat(12));
        let short = format!("deck-{}", "ab".repeat(11));
        let long = format!("deck-{}", "ab".repeat(13));
        let other = format!("other-{}", "ab".repeat(12));
        for bad in [
            upper,
            short,
            long,
            "deck-xyxyxyxyxyxyxyxyxyxyxyxy".to_string(),
            other,
            String::new(),
        ] {
            let failure = super::deck_name_guard(&bad).expect_err("refused");
            assert_eq!(failure.code, "asset-denied", "{bad:?}");
            assert!(failure.message.contains("24 lowercase hex"), "{failure:?}");
        }
    }

    #[test]
    fn a_deck_machine_must_be_a_core_machine_name_or_the_hub() {
        assert!(super::deck_machine_guard("").is_ok(), "the hub's own core");
        assert!(super::deck_machine_guard("tired").is_ok());
        for bad in ["../x", "a b", "-lead", ".dot", ""] {
            if bad.is_empty() {
                continue; // empty is the hub and already admitted
            }
            let failure = super::deck_machine_guard(bad).expect_err("refused");
            assert_eq!(failure.code, "asset-denied", "{bad:?}");
        }
    }

    #[test]
    fn the_feed_message_tags_are_lowercase_on_the_wire() {
        let doc = serde_json::to_value(DeckFeedMessage::Doc { version: 2 }).unwrap();
        assert_eq!(doc["kind"], "doc");
        assert_eq!(doc["version"], 2);
        let lost = serde_json::to_value(DeckFeedMessage::Lost { reason: "x".into() }).unwrap();
        assert_eq!(lost["kind"], "lost");
        let quiet = serde_json::to_value(DeckFeedMessage::Quiet).unwrap();
        assert_eq!(quiet["kind"], "quiet");
    }

    #[test]
    fn the_read_target_follows_the_machine_argument() {
        assert!(matches!(core_read_of(&None), CoreRead::HostFile));
        assert!(matches!(core_read_of(&Some("".into())), CoreRead::Local));
        assert!(
            matches!(core_read_of(&Some("tired".into())), CoreRead::Named(name) if name == "tired")
        );
    }

    #[test]
    fn parent_traversal_is_denied() {
        for attempt in ["../secret", "assets/../../x", "a/../.."] {
            assert!(guard_relative(attempt).is_err(), "{attempt}");
        }
    }

    #[test]
    fn absolute_and_windows_paths_are_denied() {
        for attempt in ["/etc/passwd", "C:/x", "~/.ssh/id_ed25519", "a\\b"] {
            assert!(guard_relative(attempt).is_err(), "{attempt}");
        }
    }

    #[test]
    fn ordinary_relative_paths_pass() {
        assert!(guard_relative("assets/figure.svg").is_ok());
        assert!(guard_relative("./nested/./fig.png").is_ok());
    }

    #[test]
    fn nul_bytes_are_denied() {
        assert!(guard_relative("a\0b").is_err());
    }
}

/// The system clipboard write, through the app's own GTK display
/// connection. The webview's copy event never dispatches under the
/// native WebKitGTK (execCommand copy returns false even for plain
/// prose selections), so the frontend routes its copy payload here on
/// Ctrl+C and suppresses the native command.
#[tauri::command]
pub fn write_clipboard(window: tauri::Window, text: String) -> Result<usize, String> {
    use gtk::traits::WidgetExt;
    let display = window
        .gtk_window()
        .map_err(|_| "no gtk window".to_string())?
        .display();
    let selection = gtk::gdk::Atom::from("CLIPBOARD");
    let clipboard = gtk::Clipboard::for_display(&display, &selection);
    clipboard.set_text(&text);
    eprintln!("[bridge] clipboard set: {} chars", text.chars().count());
    Ok(text.chars().count())
}

// -- the deck (t-24@r21 B1, B2) ----------------------------------------

/// The deck this window offered, for the close-time delete: the name
/// and the machine it lives on, recorded when the create lands. Never
/// a second copy of deck state - the document is the truth; this is
/// only the address the shell must retire at close.
static DECK_ADDRESS: std::sync::Mutex<Option<(String, String)>> = std::sync::Mutex::new(None);

/// The address the shell must retire at close, for lib.rs's window
/// hook: (name, machine) once a deck was offered, else None.
pub fn deck_address() -> Option<(String, String)> {
    DECK_ADDRESS.lock().ok().and_then(|slot| slot.clone())
}

fn deck_machine_guard(on: &str) -> Result<(), BridgeError> {
    // the deck machine is the caller's resolution of the bare-path
    // rule: "" for the hub's own core, else a declared machine name.
    // Anything else has not gone through that rule.
    if on.is_empty() {
        return Ok(());
    }
    if !on
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        || on.starts_with('-')
        || on.starts_with('.')
    {
        return Err(deny("the deck machine is not a core machine name"));
    }
    Ok(())
}

/// The deck's name must be the viewer's own form: `deck-` plus 24
/// lowercase hex characters - 96 bits inside the core's name rule
/// (t-24@r21 B2, probed against the pinned core). A name that does
/// not match is refused before the wire: an address that can be
/// guessed is not a permission.
fn deck_name_guard(name: &str) -> Result<(), BridgeError> {
    let valid = name.len() == 29
        && name.starts_with("deck-")
        && name[5..].chars().all(|c| c.is_ascii_hexdigit())
        && name[5..].chars().all(|c| !c.is_ascii_uppercase());
    if !valid {
        return Err(deny("the deck name is not deck- plus 24 lowercase hex"));
    }
    Ok(())
}

/// Offer the deck: create the document as an empty list at version 0.
/// A conflict means the name is not fresh - the frontend generates a
/// new address rather than adopting a document it did not create.
#[tauri::command]
pub fn deck_create(name: String, on: String) -> Result<u64, BridgeError> {
    deck_name_guard(&name)?;
    deck_machine_guard(&on)?;
    let client = core_client()?;
    let version = client
        .doc_set(&on, &name, &serde_json::json!([]), 0)
        .map_err(core_failure)?;
    let recorded = DECK_ADDRESS.lock().map(|mut slot| {
        *slot = Some((name.clone(), on));
    });
    if recorded.is_err() {
        return Err(deny("the deck address could not be recorded"));
    }
    eprintln!("[bridge] deck_create {name}: version {version}");
    Ok(version)
}

/// The deck's read-back: the document as it stands (locations only,
/// never the human's position) with its version.
#[derive(Serialize)]
pub struct DeckDoc {
    pub value: serde_json::Value,
    pub version: u64,
}

#[tauri::command]
pub fn deck_read(name: String, on: String) -> Result<DeckDoc, BridgeError> {
    deck_name_guard(&name)?;
    deck_machine_guard(&on)?;
    let client = core_client()?;
    let doc = client.doc_get(&on, &name).map_err(core_failure)?;
    Ok(DeckDoc {
        value: doc.value,
        version: doc.version,
    })
}

/// Retire the address: the document is deleted (null value, retained
/// version) when the window closes or its state is discarded. A lost
/// race reads the current version and deletes once more; a document
/// already gone is success - the address is dead either way. Best
/// effort and bounded: the close path never hangs on it.
#[tauri::command]
pub fn deck_delete(name: String, on: String) -> Result<(), BridgeError> {
    deck_name_guard(&name)?;
    deck_machine_guard(&on)?;
    let client = core_client()?;
    let mut version = client.doc_get(&on, &name).map_err(core_failure)?.version;
    for _ in 0..3 {
        match client.doc_set(&on, &name, &serde_json::json!(null), version) {
            Ok(_) => {
                eprintln!("[bridge] deck_delete {name}: deleted");
                return Ok(());
            }
            Err(CoreFail::Conflict { .. }) => {
                version = client.doc_get(&on, &name).map_err(core_failure)?.version;
            }
            Err(fail) => return Err(core_failure(fail)),
        }
    }
    Err(deny("the delete kept losing its race"))
}

/// The deck's open feed: the subscription the watch holds. One feed -
/// this viewer is one window.
static DECK_FEED: std::sync::Mutex<Option<Subscription>> = std::sync::Mutex::new(None);

/// One feed answer: a document landing for the deck's name, the feed
/// ending, or the hold expiring quiet (a normal answer - the cue feed
/// waits with the window, never a timer). The tags are lowercase: the
/// frontend's watch loop matches these exact strings.
#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum DeckFeedMessage {
    Doc { version: u64 },
    Lost { reason: String },
    Quiet,
}

/// Watch the deck: open the persistent `events.subscribe` connection
/// the holds read from. The cue feed is wake-driven: `deck_wait`
/// parks on the subscription with a bounded read (25 s, the hold
/// pattern wait_look_config rides), answers on a landing, and the
/// frontend re-enters the hold after every answer. No timer poll: the
/// only timer is the hold's own deadline, and the core's heartbeat
/// sits far inside it.
#[tauri::command]
pub fn deck_watch(name: String, on: String) -> Result<(), BridgeError> {
    deck_name_guard(&name)?;
    deck_machine_guard(&on)?;
    let socket = SocketTransport::default_path().map_err(core_failure)?;
    let subscription = Subscription::open(&socket, &on).map_err(core_failure)?;
    let mut slot = DECK_FEED
        .lock()
        .map_err(|_| deny("the deck feed lock poisoned"))?;
    *slot = Some(subscription);
    eprintln!("[bridge] deck_watch {name}: subscribed");
    Ok(())
}

const DECK_HOLD_SECONDS: u64 = 25;

fn deck_hold(name: &str) -> DeckFeedMessage {
    let mut slot = match DECK_FEED.lock() {
        Ok(slot) => slot,
        Err(_) => {
            return DeckFeedMessage::Lost {
                reason: "the feed lock is poisoned".into(),
            }
        }
    };
    let subscription = match slot.as_mut() {
        Some(subscription) => subscription,
        None => {
            return DeckFeedMessage::Lost {
                reason: "no feed is open".into(),
            }
        }
    };
    match subscription.next_doc_event(name, Duration::from_secs(DECK_HOLD_SECONDS)) {
        Ok(Some(event)) => {
            eprintln!(
                "[bridge] deck_hold {name}: doc event v{} forwarded",
                event.version
            );
            DeckFeedMessage::Doc {
                version: event.version,
            }
        }
        Ok(None) => {
            // the core closed the stream: the feed is over
            *slot = None;
            DeckFeedMessage::Lost {
                reason: "the core closed the feed".into(),
            }
        }
        Err(fail) if matches!(fail, CoreFail::HoldQuiet) => {
            // the hold's own deadline: quiet is a normal answer
            DeckFeedMessage::Quiet
        }
        Err(fail) => {
            *slot = None;
            DeckFeedMessage::Lost {
                reason: fail.to_string(),
            }
        }
    }
}

/// One bounded hold on the cue feed (t-24@r21 B1: read by the viewer
/// through events.subscribe, never a timer). Answers on a landing for
/// the deck's name, on the feed ending, or quiet at the deadline.
#[tauri::command]
pub async fn deck_wait(name: String) -> Result<DeckFeedMessage, BridgeError> {
    deck_name_guard(&name)?;
    let held_name = name;
    let handle = tauri::async_runtime::spawn_blocking(move || deck_hold(&held_name));
    handle.await.map_err(|_| deny("the deck hold task failed"))
}
