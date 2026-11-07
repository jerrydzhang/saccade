//! The narrow native bridge. Read-only file operations with structural
//! path guards, structured errors, no shell, no eval, no network, no
//! generic filesystem surface. The frontend stays renderer-hostile to
//! everything else.

pub mod commands;

pub fn run() {
    // The explicit board path handed to the foreground launch, if any:
    // `saccade-board-viewer <board.yaml>`. Never a registry, never a
    // fallback - without it the app starts empty.
    let mut args = std::env::args().skip(1);
    let board = args
        .find(|arg| !arg.starts_with('-'))
        .filter(|arg| !arg.is_empty());
    commands::set_initial_board(board);

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
            commands::read_clipboard,
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
