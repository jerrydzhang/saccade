# The board viewer's packaging: the frontend dist, its checks, and the
# retained native Tauri app - one callPackage-able file whose
# dependencies come from the pkgs it is called on. The flake's
# overlays.default calls it against a consumer's own nixpkgs: the
# packaged binary then shares the host's libglvnd/mesa stack and the
# system /run/opengl-driver EGL vendors, where this flake's own pin
# produces a foreign stack that aborts the app on a NixOS client.
{
  lib,
  stdenv,
  buildNpmPackage,
  rustPlatform,
  fetchCrate,
  pkg-config,
  makeWrapper,
  webkitgtk_4_1,
  gtk3,
  cairo,
  gdk-pixbuf,
  glib,
  pango,
  dbus,
  alsa-lib,
  # the repository root, passed in by every call site: the app's cargo
  # tree spans client/src-tauri and the shared grammar crate, and a
  # relative climb out of client/ would stay an unnormalized ".."
  # path inside flake evaluation
  repoRoot,
}: let
  clientRoot = repoRoot + "/client";
  # the viewer's own frontend and npm graph: separate sources,
  # lock and hash, never merged into the tracker crate or the workbench
  # assets. Its checks are plain tool runs (typecheck, build, contract
  # tests); they claim nothing about service isolation - the enforced
  # bwrap boundary is the harness's job and applies to the service
  # lanes only.
  src = lib.sourceByRegex clientRoot [
    "^src$"
    "^src/.*$"
    "^tests$"
    "^tests/.*$"
    "^fixtures$"
    "^fixtures/.*$"
    "^package\\.json$"
    "^package-lock\\.json$"
    "^index\\.html$"
    "^svelte\\.config\\.js$"
    "^vite\\.config\\.ts$"
    "^tsconfig\\.json$"
  ];
  npm = {
    pname = "saccade-board-viewer";
    version = "0.1.0";
    inherit src;
    npmDepsHash = "sha256-v7PrzqHrGOLwK5xyq4AQvhMb8Va0MomRzI1zjaOEpcM=";
  };
  webDist = buildNpmPackage (npm
    // {
      installPhase = ''
        mkdir -p $out
        cp -r dist/. $out/
      '';
    });
in {
  web = webDist;

  # the retained native app: the Tauri shell built against the
  # bundled frontend dist, wrapped with its runtime libraries - a
  # foreground binary a human runs with an explicit board path. No
  # deployment, install or remote transfer.
  app = let
    webkitLibs = lib.makeLibraryPath [
      webkitgtk_4_1
      gtk3
      cairo
      gdk-pixbuf
      glib
      pango
      dbus
      alsa-lib
    ];
    # the app's cargo tree: client/src-tauri plus the shared board-grammar
    # crate it validates through, kept at their real tree shape so the
    # path dependency "../../grammar" resolves exactly as in the working
    # copy. The fixture corpus is dead weight for the build and stays
    # only while the source shape is unchanged; no nested lock files.
    cargoSrc = lib.sourceByRegex repoRoot [
      "^Cargo\.toml$"
      "^Cargo\.lock$"
      "^tracker$"
      "^tracker/Cargo\.toml$"
      "^client$"
      "^client/src-tauri$"
      "^client/src-tauri/Cargo\.toml$"
      "^Cargo\.lock$"
      "^client/src-tauri/build\.rs$"
      "^client/src-tauri/tauri\.conf\.json$"
      "^client/src-tauri/src$"
      "^client/src-tauri/src/.*$"
      "^client/src-tauri/core-client$"
      "^client/src-tauri/core-client/Cargo\\.toml$"
      "^client/src-tauri/core-client/src$"
      "^client/src-tauri/core-client/src/.*$"
      "^client/src-tauri/icons$"
      "^client/src-tauri/icons/.*$"
      "^client/fixtures$"
      "^client/fixtures/.*$"
      "^grammar$"
      "^grammar/Cargo\.toml$"
      "^grammar/src$"
      "^grammar/src/.*$"
    ];
  in
    rustPlatform.buildRustPackage {
      pname = "saccade-board-viewer-app";
      version = "0.1.0";
      src = cargoSrc;
      # no tests inside nix: the parity suite runs in the devshell
      # (viewer-native-check); a sandbox cold compile puts the cheap
      # failure behind the expensive step
      doCheck = false;
      # cargoRoot names the manifest for the lockfile check and vendor
      # paths; buildAndTestSubdir is what moves cargo into it for the
      # build itself in this nixpkgs
      cargoRoot = ".";
      buildAndTestSubdir = "client/src-tauri";
      cargoLock = {
        lockFile = repoRoot + "/Cargo.lock";
      };
      # without tauri's custom-protocol feature the release binary
      # dials the dev server instead of serving the embedded dist
      buildFeatures = "custom-protocol";
      nativeBuildInputs = [pkg-config makeWrapper];
      buildInputs = [
        webkitgtk_4_1
        gtk3
        cairo
        gdk-pixbuf
        glib
        pango
        dbus
      ];
      # the embedded frontend: tauri.conf reads frontendDist
      # ../dist, one level above the manifest, and preBuild runs from
      # the source root before the build hook enters the manifest dir
      preBuild = ''
        mkdir -p client/dist
        cp -r ${webDist}/. client/dist/
      '';
      postInstall = ''
        wrapProgram $out/bin/saccade-board-viewer \
          --prefix LD_LIBRARY_PATH : ${webkitLibs}
      '';
      meta.mainProgram = "saccade-board-viewer";
    };

  # the WebDriver intermediary node for the native interaction lane:
  # nixpkgs does not package tauri-driver at this pin, so it builds
  # from the crates.io crate with the upstream Cargo.lock committed
  # beside it (tauri-driver/). Pure Rust (hyper/tokio; no GUI
  # libraries), binds 127.0.0.1 only, and proxies to the
  # WebKitWebDriver already inside the app's webkitgtk closure.
  driver = rustPlatform.buildRustPackage {
    pname = "tauri-driver";
    version = "2.1.0";
    src = fetchCrate {
      pname = "tauri-driver";
      version = "2.1.0";
      hash = "sha256-Ttik3t/IUnze3SHcpH9zPS1UwW9B/V70Wvel+eBHhOE=";
    };
    cargoLock.lockFile = clientRoot + "/tauri-driver/Cargo.lock";
    meta.mainProgram = "tauri-driver";
  };
}
