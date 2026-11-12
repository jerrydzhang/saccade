# The Saccade port's packaging: the port frontend dist and the port's
# own native binary - one callPackage-able file, shaped like
# saccade-app.nix but pointing at the port tree (client/port frontend,
# client/port-shell crate). Additive: the existing saccade-app
# packaging in nix/saccade-app.nix is untouched, and this file's
# source lists are its own - nothing here feeds the old derivation.
#
# The npm build reuses the client's package-lock (one node graph for
# both trees) and runs vite against the port tree's own vite config;
# the cargo build compiles the shared Rust shell through the port's
# own crate (client/port-shell, lib path into client/src-tauri/src)
# with the port tauri conf embedding this dist.
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
  # the port's frontend and npm graph: the SAME package-lock as the old
  # tree (one node graph, reused), while the sources are the port tree
  # plus the shared fixture corpus its bundled globs reach
  src = lib.sourceByRegex clientRoot [
    "^port$"
    "^port/.*$"
    "^fixtures$"
    "^fixtures/.*$"
    "^package\\.json$"
    "^package-lock\\.json$"
  ];
  npm = {
    pname = "saccade-app-port-web";
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

  # the port's native binary: the shared Tauri shell compiled through
  # the port crate, wrapped with its runtime libraries - a foreground
  # binary the harness boots by store path. No deployment, install or
  # remote transfer.
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
    # the port's cargo tree: client/port-shell plus everything the
    # shared shell compile reads (the lib source under
    # client/src-tauri/src, core-client, the grammar crate). The old
    # crate's own manifest and the root workspace manifests are not
    # part of this tree: the port crate is its own workspace root.
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
      pname = "saccade-app-port";
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
        wrapProgram $out/bin/saccade-app-port \
          --prefix LD_LIBRARY_PATH : ${webkitLibs}
      '';
      meta.mainProgram = "saccade-app-port";
    };
}
