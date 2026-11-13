//! The suite shell's window acts (t-14 "The suite"): one app, many
//! first-class windows, the window manager owning all layout. The
//! shell creates windows (`w`, the ask-once answer's new-window act,
//! and resuming a detached window state here), focuses one only when the
//! human asks (the window jump, through the platform's activation
//! request), broadcasts the window jump's labels to every suite
//! window, and detaches a closing window's state - closing
//! detaches, the state stays (t-24@r20). Placement is never set: a
//! new window gets whatever the window manager gives, and a new
//! window never asks for focus.

use std::thread;
use std::time::Duration;

use serde::Deserialize;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

use crate::window_registry::WindowRegistry;

/// The detach the shell fires when a bound window is destroyed:
/// fire-and-retry-once, a lost detach logged - the attachment is the
/// only fact a lost close leaves behind, and the next local act
/// (attaching again) corrects it.
fn detach_async(server: String, device: String, state: String) {
    thread::spawn(move || {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(3)))
            .build()
            .new_agent();
        let url = format!(
            "{}/api/v1/windows/devices/{}/detach",
            server.trim_end_matches('/'),
            device
        );
        let body = serde_json::json!({ "state": state });
        for attempt in 0..2 {
            let outcome = agent.post(&url).send_json(&body);
            match outcome {
                Ok(_) => return,
                Err(error) => {
                    if attempt == 1 {
                        eprintln!("[shell] detach of {state} failed after the retry: {error}");
                    }
                }
            }
            thread::sleep(Duration::from_secs(2));
        }
    });
}

/// One window event, hooked for every window of this process: a bound
/// window's destruction detaches its state on this device. The
/// binding names the server and the device, so the shell can fire the
/// detach after the webview is gone.
pub fn on_window_event(
    window: &tauri::Window<tauri::Wry>,
    event: &WindowEvent,
    registry: &WindowRegistry,
) {
    if let WindowEvent::Destroyed = event {
        let label = window.label().to_string();
        if let Some(entry) = registry.take_binding(&label) {
            eprintln!(
                "[shell] window {label} destroyed; detaching {}",
                entry.state_id
            );
            use tauri::Manager;
            let remaining = window
                .app_handle()
                .webview_windows()
                .into_keys()
                .filter(|other| *other != label)
                .count();
            crate::session::closed(&entry.state_id, remaining);
            if let (Some(server), Some(device)) = (entry.server, entry.device) {
                detach_async(server, device, entry.state_id);
            } else {
                eprintln!(
                    "[shell] window {label} carried no server or device; its state {} stays attached",
                    entry.state_id
                );
            }
        }
    }
}

/// A fresh, unique window label: the windows this shell creates are
/// the suite's own, and their labels are opaque to the window store.
fn fresh_label(app: &tauri::AppHandle) -> String {
    let mut counter = 0u64;
    loop {
        let candidate = format!("w-{}{:x}", seconds_now(), counter);
        if app.get_webview_window(&candidate).is_none() {
            return candidate;
        }
        counter += 1;
    }
}

fn seconds_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// What one suite window carries: its label and the window store state it
/// shows, when known.
#[derive(serde::Serialize)]
pub struct SuiteWindow {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_id: Option<String>,
}

/// The suite's windows in this process, label order.
#[tauri::command]
pub fn suite_windows(
    app: tauri::AppHandle,
    registry: tauri::State<'_, WindowRegistry>,
) -> Vec<SuiteWindow> {
    let mut windows: Vec<SuiteWindow> = app
        .webview_windows()
        .into_keys()
        .map(|label| SuiteWindow {
            state_id: registry.state_id_of(&label),
            label,
        })
        .collect();
    windows.sort_by(|a, b| a.label.cmp(&b.label));
    windows
}

/// Open a suite window: the window manager gives it what it gives,
/// focus is never requested, and the state it should resume - when
/// the caller names one - is pre-seeded so the new window's bootstrap
/// finds the binding and applies the state.
#[tauri::command]
pub fn open_suite_window(
    app: tauri::AppHandle,
    registry: tauri::State<'_, WindowRegistry>,
    state_id: Option<String>,
) -> Result<String, String> {
    let label = fresh_label(&app);
    if let Some(seed) = &state_id {
        registry.seed_state(&label, seed);
    }
    eprintln!(
        "[shell] opening window {label}{}",
        state_id
            .as_deref()
            .map(|id| format!(" resuming {id}"))
            .unwrap_or_else(|| " empty (a new window state)".to_string())
    );
    let built_label = label.clone();
    WebviewWindowBuilder::new(&app, &label, WebviewUrl::default())
        .title("Saccade")
        // An explicit inner size at creation: an unmapped webview
        // sizes its viewport to CONTENT until the compositor maps
        // it, and an unconstrained viewport cannot scroll at all
        // (clientHeight == scrollHeight) - the first-use verdict's
        // defect 1, where j/k and paging were dead in every window
        // the WM had not yet activated. The size forces the layout
        // constraint at creation; focus is still never requested.
        .inner_size(1024.0, 768.0)
        .focused(false)
        .build()
        .and_then(|window| {
            // The builder's size is a request; an unactivated GTK
            // window can defer its widget allocation, leaving the
            // webview's viewport unconstrained (content-sized, so
            // nothing can scroll - defect 1's shape). An explicit
            // resize after build forces the allocation now.
            window.set_size(tauri::LogicalSize::new(1024.0, 768.0))?;
            // focused(false) builds the GTK window with accept_focus
            // false, and tao 0.37.2 restores it only on the toplevel's
            // FIRST draw - which on niri (Wayland) never came: the
            // compositor showed the window activated and GTK dropped
            // every key (m-537/538). "Never steal focus" is focus on
            // map, not accepting focus: refuse the first, allow the
            // second, now. tao's late draw hook only sets true again.
            {
                use gtk::prelude::GtkWindowExt;
                let gtk_window = window.gtk_window()?;
                gtk_window.set_focus_on_map(false);
                gtk_window.set_accept_focus(true);
            }
            // m-540 (trace 2): niri activated the born window and GTK
            // marked it active, yet keys reached nothing until a click
            // inside: the WebKit view was not the toplevel's focus
            // widget. Make it the focus widget now - grab_focus on a
            // window that is not active sets the focus widget without
            // asking for activation (never Tauri's set_focus, which
            // requests it) - and again whenever the window becomes
            // active without the view holding focus.
            window.with_webview(|platform| {
                use gtk::prelude::{Cast, GtkWindowExt, WidgetExt};
                let view = platform.inner();
                view.grab_focus();
                if let Some(toplevel) = view.toplevel() {
                    if let Ok(toplevel) = toplevel.downcast::<gtk::Window>() {
                        let view = view.clone();
                        toplevel.connect_is_active_notify(move |toplevel| {
                            let holds = toplevel
                                .focused_widget()
                                .is_some_and(|widget| widget == *view.upcast_ref::<gtk::Widget>());
                            if toplevel.is_active() && !holds {
                                view.grab_focus();
                            }
                        });
                    }
                }
            })?;
            Ok(window)
        })
        .map(|_| built_label)
        .map_err(|error| {
            registry.take_binding(&label);
            format!("the window manager refused: {error}")
        })
}

/// GTK's own focus facts for the calling window (m-537/538): whether
/// the toplevel accepts keyboard focus at all, whether it takes focus
/// when mapped, and whether it is the active window now. Read from GTK
/// directly - never inferred from focus events - because a window can
/// be shown activated by the compositor while GTK drops its keys.
#[derive(serde::Serialize)]
pub struct FocusState {
    pub accepts_focus: bool,
    pub focus_on_map: bool,
    pub is_active: bool,
    /// The toplevel's focus widget's GType name ("WebKitWebView" when
    /// keys reach the page), or None when the window has none.
    pub focus_widget: Option<String>,
}

#[tauri::command]
pub fn window_focus_state(window: tauri::WebviewWindow) -> Result<FocusState, String> {
    use gtk::prelude::GtkWindowExt;
    let gtk_window = window.gtk_window().map_err(|error| error.to_string())?;
    Ok(FocusState {
        accepts_focus: gtk_window.accepts_focus(),
        focus_on_map: gtk_window.gets_focus_on_map(),
        is_active: gtk_window.is_active(),
        focus_widget: {
            use gtk::glib::prelude::ObjectExt;
            gtk_window
                .focused_widget()
                .map(|widget| widget.type_().name().to_string())
        },
    })
}

/// The window jump's focus move: the human asked, so the activation
/// request goes out. Settled focus on a real desktop is the window
/// manager's answer to that request.
#[tauri::command]
pub fn focus_suite_window(app: tauri::AppHandle, label: String) -> Result<(), String> {
    let window = app
        .get_webview_window(&label)
        .ok_or_else(|| format!("no suite window {label}"))?;
    eprintln!("[shell] window jump: activation request to {label}");
    window.set_focus().map_err(|error| error.to_string())
}

/// The label assignments the window jump hands out: each suite window
/// shows its chip while the mode is on.
#[derive(Deserialize)]
pub struct JumpLabel {
    pub window: String,
    pub chip: String,
}

/// Broadcast the labels: every suite window shows the chip the acting
/// window assigned it, through the shell's own eval door - the
/// windows stay opaque to each other; the shell is the messenger.
#[tauri::command]
pub fn broadcast_jump_labels(app: tauri::AppHandle, labels: Vec<JumpLabel>) {
    for entry in &labels {
        let Some(window) = app.get_webview_window(&entry.window) else {
            continue;
        };
        let chip = serde_json::to_string(&entry.chip).unwrap_or_else(|_| "\"?\"".to_string());
        let script = format!("window.__suiteJumpShow && window.__suiteJumpShow({chip});");
        if let Err(error) = window.eval(&script) {
            eprintln!(
                "[shell] label broadcast to {} failed: {error}",
                entry.window
            );
        }
    }
}

/// The elsewhere door: one window opens the location another chose -
/// the shell evals the receiver, focus never moves. The window is
/// opaque; the shell is the messenger, exactly as for the labels.
#[tauri::command]
pub fn open_in_window(
    app: tauri::AppHandle,
    label: String,
    location: String,
) -> Result<(), String> {
    let window = app
        .get_webview_window(&label)
        .ok_or_else(|| format!("no suite window {label}"))?;
    let spec = serde_json::to_string(&location).map_err(|error| error.to_string())?;
    let script = format!("window.__suiteOpenAt && window.__suiteOpenAt({spec});");
    window
        .eval(&script)
        .map_err(|error| format!("the open could not reach {label}: {error}"))
}

/// Clear the labels everywhere (the jump answered, or escape).
#[tauri::command]
pub fn broadcast_jump_labels_clear(app: tauri::AppHandle) {
    for (label, window) in app.webview_windows() {
        if let Err(error) = window.eval("window.__suiteJumpHide && window.__suiteJumpHide();") {
            eprintln!("[shell] label clear to {label} failed: {error}");
        }
    }
}
