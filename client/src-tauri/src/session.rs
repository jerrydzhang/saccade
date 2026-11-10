//! Launch restore (t-34@r6 change 4): the window states this device
//! had open when the app last quit, so a launch reopens them. The
//! shell's own memory, one small file in the app's data directory, never
//! a record fact and never on the tracker: a window's state lives in the
//! window store; this only remembers which states were on screen.
//!
//! A bound window adds its state; a closed window removes it unless it
//! was the last window, because closing the last window is quitting and
//! what was open then is what the next launch restores.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

struct Memory {
    file: Option<PathBuf>,
    /// The states open in this run, in the order they were bound.
    open: Vec<String>,
    /// The states open when the previous run quit; handed out once.
    last: Option<Vec<String>>,
}

static MEMORY: OnceLock<Mutex<Memory>> = OnceLock::new();

fn memory() -> &'static Mutex<Memory> {
    MEMORY.get_or_init(|| {
        Mutex::new(Memory {
            file: None,
            open: Vec::new(),
            last: None,
        })
    })
}

fn read(file: &PathBuf) -> Vec<String> {
    std::fs::read_to_string(file)
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<String>>(&text).ok())
        .unwrap_or_default()
}

fn write(memory: &Memory) {
    let Some(file) = &memory.file else { return };
    if let Some(parent) = file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let text = serde_json::to_string(&memory.open).unwrap_or_else(|_| "[]".into());
    if let Err(error) = std::fs::write(file, text) {
        eprintln!("[shell] session memory not written: {error}");
    }
}

/// Load the previous run's open states from the data directory; none
/// when there is no directory (the app then starts as a first launch).
pub fn init(data_dir: Option<PathBuf>) {
    let file = data_dir.map(|dir| dir.join("open-windows.json"));
    let last = file.as_ref().map(read).unwrap_or_default();
    eprintln!(
        "[shell] session memory: {} window(s) open at the last quit",
        last.len()
    );
    let mut memory = memory().lock().expect("the session memory holds");
    memory.file = file;
    memory.last = Some(last);
}

/// A window bound its state.
pub fn bound(state_id: &str) {
    let mut memory = memory().lock().expect("the session memory holds");
    if !memory.open.iter().any(|id| id == state_id) {
        memory.open.push(state_id.to_string());
    }
    write(&memory);
}

/// A window closed. With other windows still open it is a close: the
/// state leaves the memory. As the last window it is the quit: the
/// memory stays as the next launch's restore.
pub fn closed(state_id: &str, windows_remaining: usize) {
    if windows_remaining == 0 {
        return;
    }
    let mut memory = memory().lock().expect("the session memory holds");
    memory.open.retain(|id| id != state_id);
    write(&memory);
}

/// The states open at the last quit, once per run: the launch window
/// asks, takes the first, and opens a window for each of the rest.
#[tauri::command]
pub fn last_session() -> Vec<String> {
    memory()
        .lock()
        .expect("the session memory holds")
        .last
        .take()
        .unwrap_or_default()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_quit_keeps_what_was_open_and_a_close_forgets_it() {
        let dir = std::env::temp_dir().join(format!("sac-session-{}", std::process::id()));
        let file = dir.join("open-windows.json");
        let _ = std::fs::remove_file(&file);
        init(Some(dir.clone()));
        assert!(last_session().is_empty(), "a first launch restores nothing");
        bound("a");
        bound("b");
        bound("c");
        closed("b", 2);
        assert_eq!(
            read(&file),
            vec!["a".to_string(), "c".to_string()],
            "a close forgets its state"
        );
        closed("c", 1);
        closed("a", 0);
        assert_eq!(
            read(&file),
            vec!["a".to_string()],
            "the last window's close is the quit: it stays"
        );
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }
}
