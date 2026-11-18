//! The suite's window registry (t-71@r3): the shell's one memory of
//! its windows. Window identity is the LABEL. An entry carries the two
//! facts a run needs - which window this one links to (the receiver
//! `o` delivers into) and what place it was born to show - and the
//! registry is their only owner, so two windows can never overwrite
//! each other. Nothing here is written anywhere and nothing survives
//! the process: a quit leaves no window state behind, and a launch
//! reads none.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;

#[derive(Clone, Default)]
pub struct WindowRegistry {
    entries: Arc<Mutex<HashMap<String, RegistryEntry>>>,
}

/// One live window's facts: the label it links to, and the location it
/// was born to show (the o act's new window births its receiver at the
/// spec; the launch window was born to no place).
#[derive(Clone, Default)]
pub struct RegistryEntry {
    pub link: Option<String>,
    pub opens_at: Option<String>,
}

#[derive(Serialize)]
pub struct Binding {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opens_at: Option<String>,
}

impl WindowRegistry {
    pub fn new() -> WindowRegistry {
        WindowRegistry::default()
    }

    /// A window is born: its entry exists from the moment the shell
    /// creates it, so the suite's links and lists can name it at once.
    pub fn born(&self, label: &str, opens_at: Option<String>) {
        self.entries
            .lock()
            .expect("the window registry holds")
            .insert(
                label.to_string(),
                RegistryEntry {
                    link: None,
                    opens_at,
                },
            );
    }

    /// The place a window was born to show; none when it was born to
    /// no place (the launch window) or is not live.
    pub fn opens_at_of(&self, label: &str) -> Option<String> {
        self.entries
            .lock()
            .expect("the window registry holds")
            .get(label)
            .and_then(|entry| entry.opens_at.clone())
    }

    /// The window this one links to, when it carries a link.
    pub fn link_of(&self, label: &str) -> Option<String> {
        self.entries
            .lock()
            .expect("the window registry holds")
            .get(label)
            .and_then(|entry| entry.link.clone())
    }

    /// Set the from window's link: one out of a window, only to a live
    /// window, never to itself (the honest refusals name the reason).
    pub fn set_link(&self, from: &str, to: &str) -> Result<(), String> {
        if from == to {
            return Err(format!("a window does not link to itself ({from})"));
        }
        let mut entries = self.entries.lock().expect("the window registry holds");
        if !entries.contains_key(from) {
            return Err(format!("no live window carries {from}"));
        }
        if !entries.contains_key(to) {
            return Err(format!("no live window carries {to}"));
        }
        if let Some(entry) = entries.get_mut(from) {
            entry.link = Some(to.to_string());
        }
        Ok(())
    }

    /// Unlink; unlinking a window that carries no link (or is not
    /// live) changes nothing.
    pub fn clear_link(&self, from: &str) {
        if let Some(entry) = self
            .entries
            .lock()
            .expect("the window registry holds")
            .get_mut(from)
        {
            entry.link = None;
        }
    }

    /// A window is destroyed: its entry goes, and every link pointing
    /// at its label lapses in the same step - a dead window is no
    /// receiver (the old detach invariant, shell-side, race-free).
    /// True when the window carried an entry (it was live).
    pub fn forget(&self, label: &str) -> bool {
        let mut entries = self.entries.lock().expect("the window registry holds");
        let was_live = entries.remove(label).is_some();
        for entry in entries.values_mut() {
            if entry.link.as_deref() == Some(label) {
                entry.link = None;
            }
        }
        was_live
    }
}

/// The calling window's binding: its label and, when it was born to
/// show a place, the place. Absent opens_at means no birth place - the
/// window's boot falls through to the test-open channel and the E4
/// floor.
#[tauri::command]
pub fn window_binding(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
) -> Binding {
    let label = window.label().to_string();
    let opens_at = registry.opens_at_of(&label);
    eprintln!(
        "[bridge] window_binding {}: {}",
        label,
        opens_at
            .as_deref()
            .map(|place| format!("born at {place}"))
            .unwrap_or_else(|| "born to no place".to_string())
    );
    Binding { label, opens_at }
}

/// Set the calling-surface link between two live windows, by label.
#[tauri::command]
pub fn set_window_link(
    registry: tauri::State<'_, WindowRegistry>,
    from: String,
    to: String,
) -> Result<(), String> {
    eprintln!("[bridge] set_window_link {from} -> {to}");
    registry.set_link(&from, &to)
}

/// Unlink a window; idempotent.
#[tauri::command]
pub fn clear_window_link(registry: tauri::State<'_, WindowRegistry>, from: String) {
    eprintln!("[bridge] clear_window_link {from}");
    registry.clear_link(&from);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_window_is_born_links_and_lapses_on_close() {
        let registry = WindowRegistry::new();
        // the launch window: live from the shell's setup, born to no place
        registry.born("main", None);
        // the o act's new window: born at a place
        registry.born("w-1", Some("host/repo/t-1@r2".into()));
        assert_eq!(
            registry.opens_at_of("w-1").as_deref(),
            Some("host/repo/t-1@r2")
        );
        assert_eq!(
            registry.opens_at_of("main"),
            None,
            "the launch window was born to no place"
        );
        // the link: between live windows only, never to itself
        assert!(
            registry.set_link("main", "w-9").is_err(),
            "a target no live window carries refuses"
        );
        assert!(
            registry.set_link("w-9", "main").is_err(),
            "a source no live window carries refuses"
        );
        assert!(
            registry.set_link("main", "main").is_err(),
            "a window does not link to itself"
        );
        registry.set_link("main", "w-1").expect("live windows link");
        assert_eq!(registry.link_of("main").as_deref(), Some("w-1"));
        // the close: the entry goes and the link pointing at the
        // closed window lapses in the same step
        assert!(registry.forget("w-1"), "the destroyed window was live");
        assert_eq!(registry.link_of("main"), None, "the dead link lapsed");
        assert!(!registry.forget("w-1"), "the forget is once");
        // unlinking without a link changes nothing
        registry.clear_link("main");
        assert_eq!(registry.link_of("main"), None);
        // a link survives its source's own close only in life: while
        // both windows live, the link stands
        registry.born("w-2", None);
        registry.set_link("main", "w-2").expect("live windows link");
        registry.clear_link("main");
        assert_eq!(registry.link_of("main"), None);
    }
}
