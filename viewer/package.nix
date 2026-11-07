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
}: let
  # the viewer's own frontend and npm graph: separate sources,
  # lock and hash, never merged into the tracker crate or the workbench
  # assets. Its checks are plain tool runs (typecheck, build, contract
  # tests); they claim nothing about service isolation - the enforced
  # bwrap boundary is the harness's job and applies to the service
  # lanes only.
  src = lib.sourceByRegex ./. [
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
  npmCheck = name: command: extra:
    buildNpmPackage (npm
      // {
        pname = "board-viewer-${name}";
        buildPhase = command;
        installPhase = ''
          touch $out
        '';
      }
      // extra);
  webDist = buildNpmPackage (npm
    // {
      installPhase = ''
        mkdir -p $out
        cp -r dist/. $out/
      '';
    });
in {
  web = webDist;
  checks = lib.optionalAttrs stdenv.hostPlatform.isLinux {
    board-viewer-typecheck = npmCheck "typecheck" "npm run check" {};
    board-viewer-build = npmCheck "build" "npm run build" {};
    board-viewer-contracts = npmCheck "contracts" "npm run test" {};
  };

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
    cargoSrc = lib.sourceByRegex ./src-tauri [
      "^src$"
      "^src/.*$"
      "^Cargo\\.toml$"
      "^Cargo\\.lock$"
      "^build\\.rs$"
      "^tauri\\.conf\\.json$"
      "^icons$"
      "^icons/.*$"
    ];
  in
    rustPlatform.buildRustPackage {
      pname = "saccade-board-viewer-app";
      version = "0.1.0";
      src = cargoSrc;
      cargoLock = {
        lockFile = ./src-tauri/Cargo.lock;
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
      # ../dist, one level above the source root in the build tree
      preBuild = ''
        mkdir -p ../dist
        cp -r ${webDist}/. ../dist/
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
    cargoLock.lockFile = ./tauri-driver/Cargo.lock;
    meta.mainProgram = "tauri-driver";
  };
}
