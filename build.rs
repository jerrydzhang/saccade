use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// The crate embeds whatever this script stages into
/// OUT_DIR/workbench: `SACCADE_WEB_ASSETS=<dir>` stages that
/// directory verbatim (a Nix derivation owns the parity proof), and
/// otherwise any build input newer than a staged target reruns the
/// npm build - a stale embed is a refused build, never a silent
/// fallback.
fn main() {
    println!("cargo:rerun-if-env-changed=SACCADE_WEB_ASSETS");
    let web = Path::new("web");
    let mut inputs: Vec<PathBuf> = glob_mjs(&web.join("src"));
    for rel in [
        "index.html",
        "style.css",
        "package.json",
        "package-lock.json",
        "ascii.mjs",
        "copy-assets.mjs",
    ] {
        inputs.push(web.join(rel));
    }
    for input in &inputs {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    let assets = ["index.html", "app.js", "style.css"];
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let staged = out_dir.join("workbench");
    std::fs::create_dir_all(&staged).expect("stage directory is writable");

    if let Ok(provided) = std::env::var("SACCADE_WEB_ASSETS") {
        for asset in assets {
            let from = Path::new(&provided).join(asset);
            let bytes = std::fs::read(&from).unwrap_or_else(|error| {
                panic!("provided asset {} is unreadable: {error}", from.display())
            });
            std::fs::write(staged.join(asset), bytes).expect("stage directory is writable");
        }
        return;
    }

    let fresh = |asset: &str| {
        let target = staged.join(asset);
        std::fs::metadata(&target)
            .map(|meta| meta.modified().unwrap_or(SystemTime::UNIX_EPOCH))
            .ok()
    };
    let input_times: Vec<SystemTime> = inputs
        .iter()
        .map(|input| {
            std::fs::metadata(input)
                .unwrap_or_else(|error| {
                    panic!("build input {} is unreadable: {error}", input.display())
                })
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH)
        })
        .collect();
    let stale = assets.iter().any(|asset| match fresh(asset) {
        None => true,
        Some(target_time) => input_times.iter().any(|time| *time > target_time),
    });
    if stale {
        let built = Command::new("npm")
            .args(["run", "build"])
            .current_dir(web)
            .output();
        let ok = matches!(&built, Ok(out) if out.status.success())
            && assets
                .iter()
                .all(|asset| web.join("dist").join(asset).is_file());
        if !ok {
            let detail = match built {
                Ok(out) => String::from_utf8_lossy(&out.stderr).into_owned(),
                Err(error) => error.to_string(),
            };
            panic!(
                "embedded workbench assets are stale or missing and the npm build failed; run \
                 `just web-build` (cd web && npm ci --offline && npm run build) before building \
                 the binary - dist is a generated build product, never committed. npm detail: \
                 {}",
                detail.trim()
            );
        }
    }
    for asset in assets {
        let bytes = std::fs::read(web.join("dist").join(asset))
            .unwrap_or_else(|error| panic!("built asset {asset} is unreadable: {error}"));
        std::fs::write(staged.join(asset), bytes).expect("stage directory is writable");
    }
}

/// Every mjs source the bundle reads, so an added module joins the
/// freshness set by existing.
fn glob_mjs(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("web/src is a directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "mjs"))
        .collect();
    found.sort();
    found
}
