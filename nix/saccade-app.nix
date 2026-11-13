# The Saccade app's packaging: the frontend dist (client/port), the
# native binary (the client/port-shell crate over the shared shell
# library in client/src-tauri/src) and the tauri-driver the native
# test project boots - one callPackage-able file taking every
# dependency from the pkgs it is called on, so overlays.default can
# build it against a consumer's own nixpkgs. (The port took the app's
# name when the old frontend retired, t-36 m-554.)
#
# The npm build runs vite against the port tree's own vite config from
# the client root's npm graph; the cargo build compiles the shared
# shell through the port crate with its tauri conf embedding this
# dist.
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
  # tree spans client/port-shell, client/src-tauri (shared shell source
  # and core-client) and the shared grammar crate, and a relative climb
  # out of client/ would stay an unnormalized ".." path inside flake
  # evaluation
  repoRoot,
}: let
  clientRoot = repoRoot + "/client";
  # the frontend and its npm graph: the client's package-lock, the port
  # tree, and the fixture corpus its bundled globs reach
  src = lib.sourceByRegex clientRoot [
    "^port$"
    "^port/.*$"
    "^fixtures$"
    "^fixtures/.*$"
    "^package\\.json$"
    "^package-lock\\.json$"
  ];
  npm = {
    pname = "saccade-app-web";
    version = "0.1.0";
    inherit src;
    npmDepsHash = "sha256-Sn8IYlaL6ODCPZTqSQ/AEy+E7+DQkonskNljcO+eus8=";
  };
  webDist = buildNpmPackage (npm
    // {
      # not `npm run build`: the port builds through its own vite
      # config from the client root, reusing the client's node_modules
      dontNpmBuild = true;
      buildPhase = ''
        runHook preBuild
        node_modules/.bin/vite build -c port/vite.config.ts
        runHook postBuild
      '';
      installPhase = ''
        mkdir -p $out
        cp -r port/dist/. $out/
      '';
    });
in {
  web = webDist;

  # the native binary: the shared Tauri shell compiled through the port
  # crate, wrapped with its runtime libraries - a foreground binary the
  # harness boots by store path. No deployment, install or remote
  # transfer.
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
    # the cargo tree: client/port-shell plus everything the shared shell
    # compile reads (the lib source under client/src-tauri/src,
    # core-client, the grammar crate). The root workspace manifests are
    # not part of this tree: the port crate is its own workspace root.
    cargoSrc = lib.sourceByRegex repoRoot [
      "^client$"
      "^client/port-shell$"
      "^client/port-shell/Cargo\\.toml$"
      "^client/port-shell/Cargo\\.lock$"
      "^client/port-shell/build\\.rs$"
      "^client/port-shell/tauri\\.conf\\.json$"
      "^client/port-shell/src$"
      "^client/port-shell/src/.*$"
      "^client/src-tauri$"
      "^client/src-tauri/src$"
      "^client/src-tauri/src/.*$"
      "^client/src-tauri/icons$"
      "^client/src-tauri/icons/.*$"
      "^client/src-tauri/core-client$"
      "^client/src-tauri/core-client/Cargo\\.toml$"
      "^client/src-tauri/core-client/src$"
      "^client/src-tauri/core-client/src/.*$"
      "^grammar$"
      "^grammar/Cargo\\.toml$"
      "^grammar/src$"
      "^grammar/src/.*$"
    ];
  in
    rustPlatform.buildRustPackage {
      pname = "saccade-app";
      version = "0.1.0";
      src = cargoSrc;
      # no tests inside nix: the parity suite runs in the devshell; a
      # sandbox cold compile puts the cheap failure behind the
      # expensive step
      doCheck = false;
      # cargoRoot names the manifest dir for the lockfile check and the
      # vendor paths: the port crate carries its own Cargo.lock
      cargoRoot = "client/port-shell";
      buildAndTestSubdir = "client/port-shell";
      cargoLock = {
        lockFile = clientRoot + "/port-shell/Cargo.lock";
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
      # the embedded frontend: the port tauri conf reads frontendDist
      # ../port/dist, one level above the port crate's manifest, and
      # preBuild runs from the source root before the build hook
      # enters the crate dir
      preBuild = ''
        mkdir -p client/port/dist
        cp -r ${webDist}/. client/port/dist/
      '';
      postInstall = ''
        wrapProgram $out/bin/saccade-app \
          --prefix LD_LIBRARY_PATH : ${webkitLibs}
      '';
      meta.mainProgram = "saccade-app";
    };

  # the WebDriver intermediary node for the native test project:
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
