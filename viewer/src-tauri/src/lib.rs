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

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::read_declaration,
            commands::read_asset,
            commands::viewer_identity,
            commands::initial_board,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the board viewer");
}
