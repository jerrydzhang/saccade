//! The window binding: which pool state each webview window of this
//! process shows, and - once the window's frontend has bound - where
//! the pool lives and which device this is, so the shell can detach
//! the state when the window closes. The registry is the shell's own
//! memory, one entry per window label; the pool state itself lives on
//! the tracker's server (t-14, t-24@r20). A window that has not bound
//! yet answers its label alone, and the frontend binds the state it
//! created or resumed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;

#[derive(Clone, Default)]
pub struct WindowRegistry {
    entries: Arc<Mutex<HashMap<String, RegistryEntry>>>,
}

/// One window's pairing: the pool state it shows and, when the
/// frontend has bound, the server and device its detach needs.
#[derive(Clone, Default)]
pub struct RegistryEntry {
    pub state_id: String,
    pub server: Option<String>,
    pub device: Option<String>,
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
                    server: None,
                    device: None,
                },
            );
    }

    /// Complete a window's binding: the state, where the pool lives,
    /// which device this is.
    pub fn bind(&self, label: &str, state_id: &str, server: &str, device: &str) {
        self.entries
            .lock()
            .expect("the window registry holds")
            .insert(
                label.to_string(),
                RegistryEntry {
                    state_id: state_id.to_string(),
                    server: Some(server.to_string()),
                    device: Some(device.to_string()),
                },
            );
    }

    /// Take a window's binding: the close detach fires once, on the
    /// destruction that ends the window.
    pub fn take_binding(&self, label: &str) -> Option<RegistryEntry> {
        self.entries
            .lock()
            .expect("the window registry holds")
            .remove(label)
    }
}

/// The calling window's binding: its label and the pool state it
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

/// Bind the calling window to the pool state it created or resumed,
/// naming where the pool lives and which device this is - the facts
/// the close detach needs.
#[tauri::command]
pub fn bind_window_state(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
    state_id: String,
    server: String,
    device: String,
) {
    let label = window.label().to_string();
    eprintln!("[bridge] bind_window_state {label} -> {state_id}");
    registry.bind(&label, &state_id, &server, &device);
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
        // the frontend completes the binding with the detach facts
        registry.bind("w-1", "s-9", "http://127.0.0.1:8811", "dev-1");
        let entry = registry.take_binding("w-1").expect("the binding stands");
        assert_eq!(entry.state_id, "s-9");
        assert_eq!(entry.server.as_deref(), Some("http://127.0.0.1:8811"));
        assert_eq!(entry.device.as_deref(), Some("dev-1"));
        // the take is once: the destruction that ends the window
        assert!(registry.take_binding("w-1").is_none());
        assert!(registry.state_id_of("w-1").is_none());
        // a window that never bound answers none
        assert!(registry.state_id_of("main").is_none());
    }
}
