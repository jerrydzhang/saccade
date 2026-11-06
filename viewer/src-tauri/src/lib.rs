//! The narrow native bridge. Read-only file operations with structural
//! path guards, structured errors, no shell, no eval, no network, no
//! generic filesystem surface. The frontend stays renderer-hostile to
//! everything else.

pub mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::read_declaration,
            commands::read_asset,
            commands::viewer_identity,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the board viewer");
}
