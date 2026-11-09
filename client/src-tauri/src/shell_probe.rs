//! The shell feasibility probe (t-19): test-only machinery, named as
//! such, never presented as human functionality. The native lane sets
//! BOARD_VIEWER_SHELL_PROBE and this module then creates one second
//! webview window beside the main one and issues one focus activation
//! request, so the lane's app log answers the two feasibility facts the
//! shell stands on: a second first-class window comes up over
//! WebKitGTK inside the lane's private Weston, and the windows share
//! one origin-scoped localStorage (the frontend reports what each
//! window sees through the shell_probe_report bridge command). A human
//! run has the variable unset: no second window is ever created.

use std::time::Duration;

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

const PROBE_LABEL: &str = "probe-second";

pub fn maybe_start(handle: tauri::AppHandle) {
    let active = std::env::var("BOARD_VIEWER_SHELL_PROBE")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    if !active {
        return;
    }
    eprintln!("[shell-probe] active: the second window comes up in 6s");
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(6));
        let builder_handle = handle.clone();
        let scheduled = handle.run_on_main_thread(move || {
            // window creation rides the main thread (GTK); focused(false)
            // so the probe never moves focus - placement stays the
            // window manager's
            match WebviewWindowBuilder::new(&builder_handle, PROBE_LABEL, WebviewUrl::default())
                .title("shell probe second")
                .focused(false)
                .build()
            {
                Ok(_window) => eprintln!("[shell-probe] second window built: {PROBE_LABEL}"),
                Err(error) => eprintln!("[shell-probe] second window FAILED: {error}"),
            }
        });
        if let Err(error) = scheduled {
            eprintln!("[shell-probe] main-thread scheduling FAILED: {error}");
            return;
        }
        std::thread::sleep(Duration::from_secs(3));
        let focus_handle = handle.clone();
        let requested = handle.run_on_main_thread(move || {
            if let Some(main) = focus_handle.get_webview_window("main") {
                // the window jump's focus move: the activation request
                // itself is the evidence the lane pins; settled focus on
                // a real desktop stays a separate gate
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| main.set_focus())) {
                    Ok(Ok(())) => eprintln!("[shell-probe] activation request sent to main: ok"),
                    Ok(Err(error)) => {
                        eprintln!("[shell-probe] activation request FAILED: {error}")
                    }
                    Err(_) => eprintln!("[shell-probe] activation request panicked"),
                }
            } else {
                eprintln!("[shell-probe] main window not found for the activation request");
            }
        });
        if let Err(error) = requested {
            eprintln!("[shell-probe] activation scheduling FAILED: {error}");
        }
    });
}
