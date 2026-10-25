use std::path::Path;
use std::process::Command;

/// The workbench assets are a build product, never source: web/dist
/// regenerates from web/src via `just web-build` (npm run build). The
/// crate embeds whatever dist holds, so a missing dist must stop the
/// build with the one instruction that fixes it - a silent empty
/// bundle would be worse than a loud refusal.
fn main() {
    let dist = Path::new("web/dist");
    let assets = ["index.html", "app.js", "style.css"];
    for asset in assets {
        println!("cargo:rerun-if-changed=web/src/{asset}");
    }
    println!("cargo:rerun-if-changed=web/index.html");
    println!("cargo:rerun-if-changed=web/style.css");
    let missing: Vec<&str> = assets
        .iter()
        .filter(|asset| !dist.join(asset).is_file())
        .copied()
        .collect();
    if !missing.is_empty() {
        // rebuilding from source is the reliable path; a fallback write
        // of placeholder bytes would ship a workbench that renders
        // nothing, so the prerequisite is explicit instead
        let rebuilt = Command::new("npm")
            .args(["run", "build"])
            .current_dir("web")
            .output();
        let ok = matches!(&rebuilt, Ok(out) if out.status.success())
            && assets.iter().all(|asset| dist.join(asset).is_file());
        if !ok {
            panic!(
                "workbench assets are missing ({}); run `just web-build` (or cd web && npm run \
                 build) before building the binary - dist is a generated build product, never \
                 committed",
                missing.join(", ")
            );
        }
    }
}
