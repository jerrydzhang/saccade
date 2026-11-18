//! The window store's file (t-71): the app's device-local window
//! state, one document at <app-data-dir>/window-store.json. The shell
//! is the file's dumb persistence: it hands out the document's text
//! and stores the document's text; the schema, the states and every
//! rule about them live in the frontend's window-store module. Never
//! a record fact and never on the tracker: the document belongs to
//! this device alone.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

struct Memory {
    file: Option<PathBuf>,
}

static MEMORY: OnceLock<Mutex<Memory>> = OnceLock::new();

fn memory() -> &'static Mutex<Memory> {
    MEMORY.get_or_init(|| Mutex::new(Memory { file: None }))
}

/// The atomic write: a temp file in the same directory, then the
/// rename - a crash mid-write leaves the previous document, never a
/// torn one.
fn write(file: &std::path::Path, text: &str) -> std::io::Result<()> {
    let dir = file.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(dir)?;
    let name = file
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("window-store.json");
    let temp = dir.join(format!(".{name}.tmp"));
    std::fs::write(&temp, text)?;
    std::fs::rename(&temp, file)
}

/// Point the store at the app's data directory; none when there is no
/// directory, and every save then logs and answers.
pub fn init(data_dir: Option<PathBuf>) {
    let file = data_dir.map(|dir| dir.join("window-store.json"));
    eprintln!(
        "[shell] window store: {}",
        file.as_ref()
            .map(|file| file.display().to_string())
            .unwrap_or_else(|| "unavailable".into())
    );
    let mut memory = memory().lock().expect("the window store holds");
    memory.file = file;
}

/// The stored document's text; None when there is no file yet (a
/// first launch, or the directory went away).
#[tauri::command]
pub fn window_store_load() -> Option<String> {
    let memory = memory().lock().expect("the window store holds");
    let Some(file) = &memory.file else {
        return None;
    };
    match std::fs::read_to_string(file) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            eprintln!("[shell] window store not read: {error}");
            None
        }
    }
}

/// Store the document's text. A failure logs and answers: the store
/// stays local, and a window keeps working without it.
#[tauri::command]
pub fn window_store_save(text: String) {
    let memory = memory().lock().expect("the window store holds");
    let Some(file) = &memory.file else {
        eprintln!("[shell] window store not written: no data directory");
        return;
    };
    if let Err(error) = write(file, &text) {
        eprintln!("[shell] window store not written: {error}");
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn an_atomic_write_lands_whole_and_leaves_no_temp_behind() {
        let dir = std::env::temp_dir().join(format!("sac-window-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the temp directory is made");
        let file = dir.join("window-store.json");
        write(&file, "{\"schema\":1,\"states\":{}}").expect("the write lands");
        assert_eq!(
            std::fs::read_to_string(&file).expect("the document reads back"),
            "{\"schema\":1,\"states\":{}}"
        );
        let second = write(&file, "{\"schema\":1,\"states\":{\"s-1\":{}}}");
        assert!(second.is_ok(), "a second write over the same file lands");
        assert_eq!(
            std::fs::read_to_string(&file).expect("the document reads back"),
            "{\"schema\":1,\"states\":{\"s-1\":{}}}"
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("the directory reads")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name())
            .filter(|name| name.to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no temp file is left behind");
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn the_store_round_trips_through_the_commands_and_answers_none_before_any_save() {
        let dir = std::env::temp_dir().join(format!("sac-window-store-cmd-{}", std::process::id()));
        let file = dir.join("window-store.json");
        let _ = std::fs::remove_file(&file);
        init(Some(dir.clone()));
        assert_eq!(
            window_store_load(),
            None,
            "a first launch reads no document"
        );
        window_store_save(
            "{\"schema\":1,\"states\":{\"s-1\":{\"location\":\"board.yaml\"}}}".into(),
        );
        assert_eq!(
            window_store_load().expect("the document reads back"),
            "{\"schema\":1,\"states\":{\"s-1\":{\"location\":\"board.yaml\"}}}"
        );
        // no data directory: the save logs and answers, the load
        // answers none
        init(None);
        window_store_save("{}".into());
        assert_eq!(window_store_load(), None);
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }
}
