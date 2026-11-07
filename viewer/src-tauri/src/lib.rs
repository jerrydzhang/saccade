//! The narrow native bridge. Read-only file operations with structural
//! path guards, structured errors, no shell, no eval, no generic
//! filesystem surface. The one local door beyond std::fs is the local
//! herdr-core's read-only socket face for machine-named locations; the
//! frontend stays renderer-hostile to everything else.

pub mod commands;

pub fn run() {
    // The explicit board location handed to the foreground launch, if
    // any: `saccade-board-viewer <board.yaml>` or
    // `saccade-board-viewer --on <machine> <board.yaml>`. The composed
    // location string is the same grammar the open control parses. Never
    // a registry, never a fallback - without it the app starts empty.
    let mut args = std::env::args().skip(1);
    let mut on: Option<String> = None;
    let mut board: Option<String> = None;
    while let Some(arg) = args.next() {
        if arg == "--on" {
            match args.next() {
                Some(machine) if !machine.is_empty() => on = Some(machine),
                _ => break, // malformed launch: never guess a location
            }
        } else if !arg.starts_with('-') && !arg.is_empty() {
            board = Some(arg);
            break;
        }
    }
    let location = board.map(|path| match &on {
        Some(machine) => format!("{machine}:{path}"),
        None => path,
    });
    commands::set_initial_board(location);

    // Webview console observability is opt-in per environment: only the
    // native lane sets BOARD_VIEWER_WEBVIEW_CONSOLE, so a human run's
    // webview stays silent. weston runs with --renderer=pixman in the
    // lane; this setting only routes console lines to stdout for the
    // app.log evidence.
    let console_requested = std::env::var("BOARD_VIEWER_WEBVIEW_CONSOLE").is_ok();
    println!(
        "[bridge] webview console-to-stdout: {}",
        if console_requested { "on" } else { "off" }
    );

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::read_declaration,
            commands::read_asset,
            commands::viewer_identity,
            commands::initial_board,
            commands::write_clipboard,
        ])
        .setup(move |app| {
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
        .expect("error while running the board viewer");
}
