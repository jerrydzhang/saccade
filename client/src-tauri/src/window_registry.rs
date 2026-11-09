//! The window binding: which pool state each webview window of this
//! process shows. The registry is the shell's own memory, one entry
//! per window label, so a window's state survives a frontend reload
//! and is known to the shell that will close, jump and link windows.
//! The pool state itself lives on the tracker's server (t-14,
//! t-24@r20); the shell records only the pairing. A window that has
//! not bound yet answers no state, and the frontend binds the state it
//! created.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;

#[derive(Default)]
pub struct WindowRegistry {
    bindings: Mutex<HashMap<String, String>>,
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

    fn binding(&self, label: &str) -> Option<String> {
        self.bindings
            .lock()
            .expect("the window registry holds")
            .get(label)
            .cloned()
    }

    fn bind(&self, label: &str, state_id: &str) {
        self.bindings
            .lock()
            .expect("the window registry holds")
            .insert(label.to_string(), state_id.to_string());
    }

    fn unbind(&self, label: &str) {
        self.bindings
            .lock()
            .expect("the window registry holds")
            .remove(label);
    }
}

/// The calling window's binding: its label and the pool state it
/// shows, when one is bound.
#[tauri::command]
pub fn window_binding(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
) -> Binding {
    let label = window.label().to_string();
    let state_id = registry.binding(&label);
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

/// Bind the calling window to the pool state it created or resumed.
#[tauri::command]
pub fn bind_window_state(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
    state_id: String,
) {
    let label = window.label().to_string();
    eprintln!("[bridge] bind_window_state {label} -> {state_id}");
    registry.bind(&label, &state_id);
}

/// Drop the calling window's binding (a window that stopped showing
/// its state without closing).
#[tauri::command]
pub fn unbind_window_state(
    registry: tauri::State<'_, WindowRegistry>,
    window: tauri::WebviewWindow,
) {
    let label = window.label().to_string();
    registry.unbind(&label);
}
