//! The narrow native bridge. Read-only file operations with structural
//! path guards, structured errors, no shell, no eval, no generic
//! filesystem surface. The one local door beyond std::fs is the local
//! herdr-core's read-only socket face for machine-named locations; the
//! frontend stays renderer-hostile to everything else.

pub mod board_validate;
pub mod commands;
pub mod look_config;
pub mod shell_probe;
pub mod suite;
pub mod window_registry;
pub mod words;

pub fn run() {
    // The human surface opens through the header's location field and
    // the picker (t-24@r11): the launch argument and the local-device
    // path form left with them. The lanes open their boards through the
    // test-open channel (SAC_APP_TEST_OPEN), test-only machinery
    // named as such.

    // Webview console observability is opt-in per environment: only the
    // native lane sets SAC_APP_WEBVIEW_CONSOLE, so a human run's
    // webview stays silent. weston runs with --renderer=pixman in the
    // lane; this setting only routes console lines to stdout for the
    // app.log evidence.
    let console_requested = std::env::var("SAC_APP_WEBVIEW_CONSOLE").is_ok();
    println!(
        "[bridge] webview console-to-stdout: {}",
        if console_requested { "on" } else { "off" }
    );

    // the registry before the builder: the close detach's hook needs
    // it, and the windows need the same registry managed
    let registry = window_registry::WindowRegistry::new();
    tauri::Builder::default()
        .on_window_event({
            let registry = registry.clone();
            move |window, event| suite::on_window_event(window, event, &registry)
        })
        .invoke_handler(tauri::generate_handler![
            commands::read_declaration,
            commands::read_asset,
            commands::edit_declaration,
            commands::viewer_identity,
            commands::test_open,
            commands::tracker_machine,
            commands::machine_id,
            commands::publish_identity,
            commands::write_clipboard,
            commands::deck_create,
            commands::deck_read,
            commands::deck_delete,
            commands::deck_watch,
            commands::deck_wait,
            commands::shell_probe_active,
            commands::shell_probe_report,
            window_registry::window_binding,
            window_registry::bind_window_state,
            suite::suite_windows,
            suite::open_suite_window,
            suite::focus_suite_window,
            suite::broadcast_jump_labels,
            suite::broadcast_jump_labels_clear,
            suite::open_in_window,
            board_validate::validate_board,
            look_config::look_config,
            look_config::wait_look_config,
            words::words_wait,
        ])
        // the deck's address dies with the window (t-24@r21 B1): on
        // close the document is deleted through the same core face it
        // was created on - bounded, best effort, logged; the window
        // still closes if the core is gone.
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                if let Some((name, on)) = commands::deck_address() {
                    let deletion = std::panic::catch_unwind(|| commands::deck_delete(name, on));
                    eprintln!(
                        "[bridge] deck delete at close: {}",
                        match deletion {
                            Ok(Ok(())) => "deleted".to_string(),
                            Ok(Err(error)) => format!("failed: {} {}", error.code, error.message),
                            Err(_) => "panicked".to_string(),
                        }
                    );
                }
            }
        })
        .setup(move |app| {
            // the look follows the human's own configuration file in the
            // platform config directory (t-24@r3); the directory may not
            // exist yet, in which case the built-in scheme is in use and
            // the follow thread picks the file up the moment it appears
            use tauri::Manager;
            let dir = app.path().app_config_dir().ok();
            eprintln!(
                "[bridge] look config dir: {}",
                dir.as_ref()
                    .map(|d| d.display().to_string())
                    .unwrap_or_else(|| "unavailable".into())
            );
            look_config::init(dir);
            // the app's ONE Herdr event feed (t-24@r22): the socket
            // resolved once here; none, it dials nothing and the
            // instruments read an empty table
            words::start(
                saccade_app_core::SocketTransport::default_path()
                    .map(|path| path.to_path_buf())
                    .ok(),
            );
            // the window registry: which window state each window shows
            // (t-19); the shell's own memory, never a record fact
            app.manage(registry);
            // the shell feasibility probe (t-19): test-only machinery,
            // inert unless the lane sets SAC_APP_SHELL_PROBE
            shell_probe::maybe_start(app.handle().clone());
            if console_requested {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    window
                        .with_webview(|webview| {
                            #[cfg(target_os = "linux")]
                            {
                                use webkit2gtk::{SettingsExt, WebViewExt};
                                let inner = webview.inner();
                                if let Some(settings) = WebViewExt::settings(&inner) {
                                    settings.set_enable_write_console_messages_to_stdout(true);
                                }
                                // the engine's context menu carries a Copy
                                // entry that runs the engine copy command -
                                // which loses math natively (no copy event
                                // dispatches); copying is Ctrl+C through the
                                // bridge, so the menu is suppressed wholesale
                                inner.connect_context_menu(|_webview, _menu, _event, _hit_test| {
                                    true
                                });
                                eprintln!(
                                    "[bridge] FONTCONFIG_FILE: {}",
                                    std::env::var("FONTCONFIG_FILE")
                                        .unwrap_or_else(|_| "unset".into())
                                );
                                // WebKitGTK derives the page scale from the
                                // GTK gtk-xft-dpi setting (WebKit PR 28691:
                                // scale = dpi/96). Without a desktop
                                // environment providing it, the setting
                                // carries a negative sentinel, and the page
                                // scale becomes -1/96 - corrupting every
                                // measurement and painting SVG text into
                                // nothing. A negative DPI is never real, so
                                // the shell substitutes the 96-dpi standard;
                                // a desktop-provided value is left alone.
                                use gtk::prelude::*;
                                if let Some(gtk_settings) = gtk::Settings::default() {
                                    let xft_dpi: i32 = gtk_settings.property("gtk-xft-dpi");
                                    eprintln!("[bridge] gtk-xft-dpi was: {}", xft_dpi);
                                    if xft_dpi <= 0 {
                                        gtk_settings.set_property("gtk-xft-dpi", 98304);
                                        eprintln!("[bridge] gtk-xft-dpi set: 98304 (96 dpi)");
                                    }
                                }
                            }
                        })
                        .expect("webview console setting failed");
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the Saccade app");
}
