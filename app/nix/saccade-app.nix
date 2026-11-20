# The Saccade app's packaging: the frontend dist (app/port), the
# native binary (the app/port-shell crate over the shared shell
# library in app/src-tauri/src) and the tauri-driver the native
# test project boots - one callPackage-able file taking every
# dependency from the pkgs it is called on, so overlays.default can
# build it against a consumer's own nixpkgs. (The port took the app's
# name when the old frontend retired, t-36 m-554.)
#
# The npm build runs vite against the port tree's own vite config from
# the app root's npm graph; the cargo build compiles the shared
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
  # tree spans app/port-shell, app/src-tauri (shared shell source and
  # core-client) and the shared grammar crate, and a relative climb
  # out of app/ would stay an unnormalized ".." path inside flake
  # evaluation
  repoRoot,
}:
let
  clientRoot = repoRoot + "/app";
  # the frontend and its npm graph: the app's package-lock, the port
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
    version = "0.2.11";
    inherit src;
    npmDepsHash = "sha256-XsZzfaWncPRaoul4xtDjiu2M5UKvINtW8DunQw80pfc=";
  };
  webDist = buildNpmPackage (
    npm
    // {
      # not `npm run build`: the port builds through its own vite
      # config from the app root, reusing the app's node_modules
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
    }
  );
in
{
  web = webDist;

  # the native binary: the shared Tauri shell compiled through the port
  # crate, wrapped with its runtime libraries - a foreground binary the
  # harness boots by store path. No deployment, install or remote
  # transfer.
  app =
    let
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
      # the cargo tree: the app workspace (app/Cargo.toml, its lock),
      # app/port-shell, everything the shared shell compile reads (the
      # lib source under app/src-tauri/src, core-client) and the grammar
      # crate from saccade/grammar. The saccade workspace manifests are
      # not part of this tree.
      cargoSrc = lib.sourceByRegex repoRoot [
        "^app$"
        "^app/Cargo\\.toml$"
        "^app/Cargo\\.lock$"
        "^app/port-shell$"
        "^app/port-shell/Cargo\\.toml$"
        "^app/port-shell/build\\.rs$"
        "^app/port-shell/tauri\\.conf\\.json$"
        "^app/port-shell/src$"
        "^app/port-shell/src/.*$"
        "^app/src-tauri$"
        "^app/src-tauri/src$"
        "^app/src-tauri/src/.*$"
        "^app/src-tauri/icons$"
        "^app/src-tauri/icons/.*$"
        "^app/src-tauri/core-client$"
        "^app/src-tauri/core-client/Cargo\\.toml$"
        "^app/src-tauri/core-client/src$"
        "^app/src-tauri/core-client/src/.*$"
        "^saccade$"
        "^saccade/grammar$"
        "^saccade/grammar/Cargo\\.toml$"
        "^saccade/grammar/src$"
        "^saccade/grammar/src/.*$"
      ];
    in
    rustPlatform.buildRustPackage {
      pname = "saccade-app";
      version = "0.2.11";
      src = cargoSrc;
      # no tests inside nix: the parity suite runs in the devshell; a
      # sandbox cold compile puts the cheap failure behind the
      # expensive step
      doCheck = false;
      # cargoRoot names the manifest dir for the lockfile check and the
      # vendor paths: the app workspace carries its own Cargo.lock
      cargoRoot = "app";
      buildAndTestSubdir = "app/port-shell";
      cargoLock = {
        lockFile = clientRoot + "/Cargo.lock";
      };
      # without tauri's custom-protocol feature the release binary
      # dials the dev server instead of serving the embedded dist
      buildFeatures = "custom-protocol";
      nativeBuildInputs = [
        pkg-config
        makeWrapper
      ];
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
        mkdir -p app/port/dist
        cp -r ${webDist}/. app/port/dist/
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
