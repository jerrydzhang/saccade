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
                                if let Some(settings) = webview.inner().settings() {
                                    settings.set_enable_write_console_messages_to_stdout(true);
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
