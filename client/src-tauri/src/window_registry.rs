//! The window binding: which window state each webview window of this
//! process shows. The registry is the shell's own memory, one entry
//! per window label; the window state itself lives in the app's own
//! device-local store (t-71, revising t-14's tracker-side placement),
//! reached by the frontend through the store's two commands. A window
//! that has not bound yet answers its label alone, and the frontend
//! binds the state it created or resumed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;

#[derive(Clone, Default)]
pub struct WindowRegistry {
    entries: Arc<Mutex<HashMap<String, RegistryEntry>>>,
}

/// One window's pairing: the window store state it shows.
#[derive(Clone, Default)]
pub struct RegistryEntry {
    pub state_id: String,
}

#[derive(Serialize)]
pub struct Binding {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_id: Option<String>,
}

impl WindowRegistry {
    pub fn new() -> WindowRegistry {
        WindowRegistry::default()
    }

    /// The state a window is known to show: a pre-seeded resume or a
    /// completed binding.
    pub fn state_id_of(&self, label: &str) -> Option<String> {
        self.entries
            .lock()
            .expect("the window registry holds")
            .get(label)
            .map(|entry| entry.state_id.clone())
    }

    /// Pre-seed a window's state before the window exists: a resume
    /// finds its state at the first binding query.
    pub fn seed_state(&self, label: &str, state_id: &str) {
        self.entries
            .lock()
            .expect("the window registry holds")
            .insert(
                label.to_string(),
                RegistryEntry {
                    state_id: state_id.to_string(),
                },
            );
    }

    /// Complete a window's binding: the state it created or resumed.
    pub fn bind(&self, label: &str, state_id: &str) {
        self.entries
            .lock()
            .expect("the window registry holds")
            .insert(
                label.to_string(),
                RegistryEntry {
                    state_id: state_id.to_string(),
                },
            );
    }

    /// Take a window's binding: the close forget fires once, on the
    /// destruction that ends the window.
    pub fn take_binding(&self, label: &str) -> Option<RegistryEntry> {
        self.entries
            .lock()
            .expect("the window registry holds")
            .remove(label)
    }
}

/// The calling window's binding: its label and the window store state it
/// shows, when one is known.
#[tauri::command]
pub fn window_binding(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
) -> Binding {
    let label = window.label().to_string();
    let state_id = registry.state_id_of(&label);
    eprintln!(
        "[bridge] window_binding {}: {}",
        label,
        state_id
            .as_deref()
            .map(|id| format!("bound to {id}"))
            .unwrap_or_else(|| "unbound".to_string())
    );
    Binding { label, state_id }
}

/// Bind the calling window to the window store state it created or
/// resumed; the state lives in the app's own store (t-71), and the
/// binding is what suite_windows and the close forget read.
#[tauri::command]
pub fn bind_window_state(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
    state_id: String,
) {
    let label = window.label().to_string();
    eprintln!("[bridge] bind_window_state {label} -> {state_id}");
    registry.bind(&label, &state_id);
    crate::session::bound(&state_id);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_window_seeds_binds_and_takes_its_binding_once() {
        let registry = WindowRegistry::new();
        // the resume path: the state is seeded before the window exists
        registry.seed_state("w-1", "s-9");
        assert_eq!(registry.state_id_of("w-1").as_deref(), Some("s-9"));
        // the frontend completes the binding
        registry.bind("w-1", "s-9");
        let entry = registry.take_binding("w-1").expect("the binding stands");
        assert_eq!(entry.state_id, "s-9");
        // the take is once: the destruction that ends the window
        assert!(registry.take_binding("w-1").is_none());
        assert!(registry.state_id_of("w-1").is_none());
        // a window that never bound answers none
        assert!(registry.state_id_of("main").is_none());
    }
}
