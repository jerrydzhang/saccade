{
  nixpkgs,
  nixpkgs-client,
  rust-overlay,
  saccade,
  wrapped-pi,
  devenv,
  herdr-core-x86_64-linux,
  herdr-core-aarch64-linux,
  claude-code-x86_64-linux,
  claude-code-aarch64-linux,
  ...
}@inputs:
let
  inherit (nixpkgs) lib;
  forAllSystems = lib.genAttrs lib.systems.flakeExposed;
  rustOverlays = [ rust-overlay.overlays.default ];
  # the pinned release bytes per linux target: the release manifest
  # hashes the tarball, which the lock entry pins by narHash; the
  # binarySha256 here is the derived assertion over the unpacked
  # binary, checked at build, completing the chain to the tested bytes
  herdrPin = {
    x86_64-linux = {
      src = herdr-core-x86_64-linux;
      binarySha256 = "028fe4d21bcdf2fe0c1d187dead79800aab85edbf14f3c0dcc115fcc74f589d4";
    };
    aarch64-linux = {
      src = herdr-core-aarch64-linux;
      binarySha256 = "713eed55c78d32d4994b9cdb2e36293bf6ba2a17c14bf8143b14bd0033b76dc2";
    };
  };
  mkHerdrCore =
    {
      pkgs,
      src,
      binarySha256,
    }:
    pkgs.runCommand "herdr-core-pinned" { } ''
      set -o errexit
      echo "${binarySha256}  ${src}/herdr-core" | ${pkgs.coreutils}/bin/sha256sum --check --strict -
      mkdir -p $out/bin
      ${pkgs.coreutils}/bin/install -m 0555 ${src}/herdr-core $out/bin/herdr-core
      ${pkgs.coreutils}/bin/install -m 0444 ${src}/LICENSE ${src}/THIRD-PARTY-NOTICES $out/
    '';
  # the development arm: scratch HOME/XDG plus hc's own session
  # scoping (HC_SESSION), so a dev core and its sockets live apart
  # from the machine server even with the ambient install present
  mkHerdrDev =
    {
      pkgs,
      herdrCore,
    }:
    pkgs.writeShellApplication {
      name = "herdr-core";
      runtimeInputs = [ pkgs.coreutils ];
      text = ''
        dev=''${XDG_STATE_HOME:-''${HOME:-/tmp}/.local/state}/saccade/hcd
        mkdir -p "$dev/h" "$dev/c" "$dev/d" "$dev/s"
        export HOME="$dev/h"
        export XDG_CONFIG_HOME="$dev/c"
        export XDG_DATA_HOME="$dev/d"
        export XDG_STATE_HOME="$dev/s"
        # short names keep the derived socket path under the kernel's
        # 107-byte limit even in deep build-sandbox HOMEs
        export HC_SESSION=dev
        exec ${herdrCore}/bin/herdr-core "$@"
      '';
    };
  herdrCore =
    system: pkgs:
    let
      pin = herdrPin.${system} or null;
    in
    if pin == null then
      null
    else
      mkHerdrCore {
        inherit pkgs;
        inherit (pin) src binarySha256;
      };
  herdrDev =
    system: pkgs:
    let
      core = herdrCore system pkgs;
    in
    if core == null then
      null
    else
      mkHerdrDev {
        inherit pkgs;
        herdrCore = core;
      };
  # the pinned Claude Code: the official npm platform package carries
  # the whole product as one native binary per target, so the wrapper
  # npm package is not needed. The lock entry pins the tarball bytes
  # (narHash) and the derivation asserts the unpacked binary against
  # its sha256 before anything touches it; the shipped copy then gets
  # the standard interpreter and rpath, so it runs under plain Nix
  # with no nix-ld dependence and no fork of the program. The license
  # is Anthropic's proprietary one: the flake consumes the official
  # distribution at pinned bytes for the person's own use, never
  # redistributes or forks it, and the subscription login stays in
  # the user's own config outside the closure.
  claudeCodePin = {
    x86_64-linux = {
      src = claude-code-x86_64-linux;
      binarySha256 = "708c77b204db30086ff6f1ed34577bda03e1be74ced80557cbf7b54249ebf0bd";
    };
    aarch64-linux = {
      src = claude-code-aarch64-linux;
      binarySha256 = "53a0d7e219d7b0804ecba5dbc002745e89f786cc946d2de85a6470d5b2d1fb06";
    };
  };
  # the package's own version line, read straight from Cargo.toml:
  # the hook's expected server version is generated, never hardcoded
  cargoVersion =
    let
      line = lib.findFirst (l: builtins.match ''version = "[0-9]+\.[0-9]+\.[0-9]+"'' l != null) null (
        lib.splitString "\n" (builtins.readFile ../saccade/tracker/Cargo.toml)
      );
    in
    if line == null then
      throw "Cargo.toml carries no readable version line"
    else
      builtins.substring 11 (builtins.stringLength line - 12) line;
  mkClaudeCode =
    {
      pkgs,
      src,
      binarySha256,
    }:
    pkgs.runCommand "claude-code-pinned"
      {
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [ pkgs.glibc ];
      }
      ''
        set -o errexit
        echo "${binarySha256}  ${src}/claude" | ${pkgs.coreutils}/bin/sha256sum --check --strict -
        mkdir -p $out/bin $out/share
        ${pkgs.coreutils}/bin/install -m 0755 ${src}/claude $out/bin/claude
        autoPatchelf $out/bin/claude
        chmod 0555 $out/bin/claude
        ${pkgs.coreutils}/bin/install -m 0444 ${src}/LICENSE.md ${src}/README.md $out/share/
      '';
  # the SessionStart identity hook: the repo template with the
  # expected server line generated from Cargo.toml, shellchecked
  # before it ships
  mkAttachHook =
    { pkgs }:
    let
      substituted = pkgs.replaceVars ../saccade/tracker/native/claude-code/saccade-cc-attach.template {
        expectMajor = lib.versions.major cargoVersion;
        expectMinor = lib.versions.minor cargoVersion;
      };
    in
    pkgs.runCommand "saccade-cc-attach"
      {
        nativeBuildInputs = [ pkgs.shellcheck ];
      }
      ''
        set -o errexit
        ${pkgs.shellcheck}/bin/shellcheck ${substituted}
        ${pkgs.bash}/bin/bash -n ${substituted}
        mkdir -p $out/bin
        ${pkgs.coreutils}/bin/install -m 0555 ${substituted} $out/bin/saccade-cc-attach
      '';
  # the worktree hooks of the composed Claude Code harness: one
  # script, the mode (create or remove) as its first argument, read
  # from the same template the shellchecked packaged copy ships
  # from - no placeholder rides it, so the composition is the
  # packaging: shellcheck, a bash parse, then the install
  mkWorktreeHook =
    { pkgs }:
    let
      substituted =
        pkgs.replaceVars ../saccade/tracker/native/claude-code/saccade-cc-worktree.template
          { };
    in
    pkgs.runCommand "saccade-cc-worktree"
      {
        nativeBuildInputs = [ pkgs.shellcheck ];
      }
      ''
        set -o errexit
        ${pkgs.shellcheck}/bin/shellcheck ${substituted}
        ${pkgs.bash}/bin/bash -n ${substituted}
        mkdir -p $out/bin
        ${pkgs.coreutils}/bin/install -m 0555 ${substituted} $out/bin/saccade-cc-worktree
      '';
  mkCCSettings =
    {
      pkgs,
      attachHook,
      worktreeHook,
    }:
    pkgs.runCommand "saccade-claude-code-settings.json" { } ''
      set -o errexit
      export attachHook="${attachHook}"
      export worktreeHook="${worktreeHook}"
      substituteAll ${../saccade/tracker/native/claude-code/settings.json.template} $out
      # the packaged file must parse as settings: a template that no
      # longer carries the hook wire refuses the build here
      ${pkgs.jq}/bin/jq -e .hooks.SessionStart $out >/dev/null
    '';
  # the composed harness: the pinned binary with the packaged
  # per-launch settings injected ahead of the caller's argv, which
  # passes through untouched
  mkSaccadeClaudeCode =
    {
      pkgs,
      claude,
      attachHook,
      worktreeHook,
    }:
    let
      settings = mkCCSettings {
        inherit pkgs attachHook worktreeHook;
      };
      wrapper = pkgs.replaceVars ../saccade/tracker/native/claude-code/saccade-claude-code.template {
        inherit claude settings;
        jq = pkgs.jq;
        curl = pkgs.curl;
      };
    in
    pkgs.runCommand "saccade-claude-code"
      {
        nativeBuildInputs = [ pkgs.shellcheck ];
      }
      ''
        set -o errexit
        ${pkgs.shellcheck}/bin/shellcheck ${wrapper}
        ${pkgs.bash}/bin/bash -n ${wrapper}
        mkdir -p $out/bin
        ${pkgs.coreutils}/bin/install -m 0555 ${wrapper} $out/bin/saccade-claude-code
      '';
  saccadeClaudeCodeFor =
    system: pkgs:
    let
      pin = claudeCodePin.${system} or null;
      claude =
        if pin == null then
          null
        else
          mkClaudeCode {
            inherit pkgs;
            inherit (pin) src binarySha256;
          };
    in
    if claude == null then
      null
    else
      mkSaccadeClaudeCode {
        inherit pkgs;
        claude = "${claude}/bin/claude";
        attachHook = "${mkAttachHook { inherit pkgs; }}/bin/saccade-cc-attach";
        worktreeHook = "${mkWorktreeHook { inherit pkgs; }}/bin/saccade-cc-worktree";
      };
  # the native integration the composed harness loads
  saccadePiExtensions = [ "saccade/tracker/native/saccade.ts" ];
  # the Saccade app's packaging lives in app/nix/saccade-app.nix: one
  # callPackage-able file taking every dependency from the pkgs it
  # is called on, so overlays.default can build it against a
  # consumer's own nixpkgs - the packaged binary then shares the
  # host's GL stack instead of this flake's pin. The repo root rides
  # in explicitly: the app's cargo tree spans app/port-shell,
  # app/src-tauri and the shared grammar crate, and a relative
  # climb would stay an unnormalized ".." path inside flake evaluation.
  saccadeApp = pkgs: root: pkgs.callPackage ../app/nix/saccade-app.nix { repoRoot = root; };
in
{
  # the consumer-facing overlay: a NixOS configuration sets
  # nixpkgs.overlays = [ inputs.saccade.overlays.default ] and gets
  # pkgs.saccade-app built against its own nixpkgs
  overlays.default = final: prev: {
    saccade-app = (final.callPackage ../app/nix/saccade-app.nix { repoRoot = ../.; }).app;
  };

  devShells = forAllSystems (
    system:
    let
      pkgs = import nixpkgs {
        inherit system;
        overlays = rustOverlays;
      };
      clientPkgs = nixpkgs-client.legacyPackages.${system};

      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [
          "rust-src"
          "rust-analyzer"
        ];
      };
      devHerdrCore = herdrCore system pkgs;
    in
    {
      # the Saccade app's native dev shell: its own dependency graph,
      # separate from the tracker devshell and the tracker crate. The
      # WebKitGTK/GTK build inputs here serve the viewer's Tauri crate
      # only; nothing here touches the tracker's build.
      # the browser lane's fontconfig: a generated config over pinned
      # DejaVu fonts, both pure store paths - the sandbox already binds
      # /nix/store read-only, so no host /etc/fonts or font dirs are
      # ever mounted
      saccade-app =
        let
          laneFontsConf = pkgs.writeText "saccade-lane-fonts.conf" ''
            <?xml version="1.0"?>
            <!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
            <fontconfig>
              <dir>${pkgs.dejavu_fonts}</dir>
              <include>${pkgs.fontconfig.out}/etc/fonts/conf.d</include>
              <cachedir prefix="xdg">fontconfig</cachedir>
            </fontconfig>
          '';
        in
        pkgs.mkShell {
          # the pinned Playwright pair (python package and browser
          # package) from this same nixpkgs revision: no browser
          # install or download anywhere
          packages = [
            rustToolchain
            pkgs.nodejs
            pkgs.pkg-config
            (pkgs.python3.withPackages (ps: [
              ps.playwright
              ps.pyflakes
            ]))
            pkgs.bubblewrap
            pkgs.weston
            pkgs.mesa
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            (saccadeApp pkgs ../.).driver
            # the Tauri/GTK block rides nixpkgs-client so the shell's
            # WebKitWebDriver matches the packaged app's WebKitGTK
            # (the two pins carry different webkitgtk releases);
            # the lane tools above stay on nixpkgs
            clientPkgs.webkitgtk_4_1
            clientPkgs.gtk3
            clientPkgs.cairo
            clientPkgs.gdk-pixbuf
            clientPkgs.glib
            clientPkgs.dbus
            pkgs.dejavu_fonts
          ];
          env.PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
          # the sandbox has no /etc/fonts, so the makeFontsConf include
          # of conf.d resolves to nothing and generic families match no
          # font; this conf carries the pinned fontconfig's own conf.d
          # (45/60-generic+latin assign sans-serif to DejaVu Sans)
          env.FONTCONFIG_FILE = laneFontsConf;
          # the native lane's software-GL pin: glvnd vendor dir whose
          # 50_mesa.json carries the absolute store path of
          # libEGL_mesa.so.0 (llvmpipe/swrast live in the same output)
          env.SAC_APP_MESA_EGL_VENDOR_DIR = "${pkgs.mesa}/share/glvnd/egl_vendor.d";
          # the remote lane's pinned core: the flake's 0.1.17 binary,
          # sha-checked at build; the lane verifies --version again
          env.SAC_APP_HERDR_CORE_BIN = "${devHerdrCore}/bin/herdr-core";
        };
      default = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          (
            { config, ... }:
            let
              # the dev wrapper: wrapped-pi pointed at this repo's
              # working-tree extension, no adaptation, no preload
              devPi =
                let
                  wrappedPiBin = "${wrapped-pi.packages.${system}.default}/bin/wrapped-pi";
                  extraArgs = lib.escapeShellArgs (
                    lib.concatMap (path: [
                      "-e"
                      "${config.devenv.root}/${path}"
                    ]) saccadePiExtensions
                  );
                in
                pkgs.lib.hiPrio (
                  pkgs.writeShellApplication {
                    name = "saccade-pi";
                    runtimeInputs = [ pkgs.util-linux ];
                    text = ''
                      exec ${wrappedPiBin} ${extraArgs} "$@"
                    '';
                  }
                );
              devHerdr = herdrDev system pkgs;
              # the composed Claude Code harness the devshell names and
              # carries: the same package the release bundles, so the
              # development override and the compile pin are one line
              devCC = saccadeClaudeCodeFor system pkgs;
            in
            {
              enterShell = ''
                if [ ! -f .agents/skills/saccade/SKILL.md ]; then
                  cargo run --quiet --manifest-path saccade/Cargo.toml -p saccade --bin sac -- skill install
                fi
              '';

              packages =
                with pkgs;
                [
                  rustToolchain
                  (pkgs.python3.withPackages (ps: [
                    ps.playwright
                    ps.pyflakes
                  ]))
                  prek
                  just
                  nixfmt
                  typescript
                  nodejs
                  direnv
                  jq
                  curl
                  saccade.packages.${system}.default
                  devPi
                ]
                ++ lib.optional (devCC != null) devCC;

              env = {
                SACCADE_PI = "${devPi}/bin/saccade-pi";
                PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
                # no SACCADE_HERDR here, by ruling: interactive use hits the
                # ambient production core (the orca shape); the sandboxed
                # dev arm is the opt-in flake output herdr-core-dev, never a
                # PATH entry - a wrapper named herdr-core would shadow
                # ambient resolution and recreate the posture seam
              }
              // lib.optionalAttrs (devCC != null) {
                # the runtime override naming the composed harness: the
                # pinned release binary resolves it, and a dev build
                # bakes it as its compile pin
                SACCADE_CLAUDE_CODE = "${devCC}/bin/saccade-claude-code";
              };

              process.manager.implementation = "native";
              processes.sac = {
                # production posture, deterministic: the override arm wins over
                # any PATH order (the devshell PATH carries a dev-arm wrapper
                # also named herdr-core; empty-string resolution lost that
                # race at runtime), so the server always joins the machine
                # core through the system binary
                env.SACCADE_HERDR = "/etc/profiles/per-user/jerry/bin/herdr-core";
                exec = "${
                  saccade.packages.${system}.default
                }/bin/sac serve --repo /home/jerry/Projects/saccade --bind 0.0.0.0 --port 8811";
                restart = {
                  on = "on_failure";
                  max = 10;
                };
                ready.http.get = {
                  port = 8811;
                  path = "/";
                };
              };
            }
          )
        ];
      };
    }
  );

  packages = forAllSystems (
    system:
    let
      pkgs = import nixpkgs {
        inherit system;
        overlays = rustOverlays;
      };
      clientPkgs = nixpkgs-client.legacyPackages.${system};
      devHerdrCore = herdrCore system pkgs;
      devHerdrPin = herdrDev system pkgs;
      saccadeClaudeCode = saccadeClaudeCodeFor system pkgs;
      pi = wrapped-pi.packages.${system}.default.override {
        extraExtensions = map (path: ../. + "/${path}") saccadePiExtensions;
      };
      # the composed harness: wrapped-pi with the saccade extension,
      # nothing interposed. util-linux rides the closure: the
      # integration's guard helper is flock, never an assumed
      # ambient binary.
      saccadePi = pkgs.writeShellApplication {
        name = "saccade-pi";
        runtimeInputs = [ pkgs.util-linux ];
        text = ''
          exec ${pi}/bin/wrapped-pi "$@"
        '';
      };
      sac = import ../saccade/nix/sac.nix {
        inherit
          pkgs
          lib
          saccadePi
          devHerdrPin
          saccadeClaudeCode
          ;
        root = ../saccade;
      };
      # the pinned archive: main's 0.18.0 tip, the release the live
      # tracker ran; the rev is the pin, the use is read-only.
      sac-archive = import ../saccade/nix/sac-archive.nix {
        inherit pkgs lib;
        archiveRev = "189894f54f4c747ad937493f34c5e526a76726b8";
        archiveSrc = builtins.fetchGit {
          url = "https://github.com/jerrydzhang/saccade";
          rev = "189894f54f4c747ad937493f34c5e526a76726b8";
          allRefs = true;
        };
      };
      # the grammar crate as its own package
      board-grammar = pkgs.rustPlatform.buildRustPackage {
        pname = "board-grammar";
        version = "0.1.0";
        # no tests inside nix: the grammar suite runs in the devshell
        doCheck = false;
        # built as the workspace member it is, with the workspace
        # lock: the grammar-local Cargo.lock is not tracked, so a
        # git-sourced build cannot use it (this derivation failed
        # evaluation from any git checkout since the lock left git)
        src = ../saccade;
        cargoRoot = ".";
        buildAndTestSubdir = "grammar";
        cargoLock.lockFile = ../saccade/Cargo.lock;
        meta.mainProgram = "board-grammar";
      };
    in
    let
      pkgSet = {
        sac = sac;

        sac-archive = sac-archive;

        board-grammar = board-grammar;

        saccade-pi = saccadePi;

        # the app the client runs: built against nixpkgs-client, the
        # revision the human's system flake is built from
        saccade-app = (saccadeApp clientPkgs ../.).app;
        saccade-app-web = (saccadeApp pkgs ../.).web;
        saccade-app-tauri-driver = (saccadeApp pkgs ../.).driver;

        default = pkgs.symlinkJoin {
          name = "saccade";
          paths = [
            sac
            saccadePi
          ]
          ++ lib.optional (saccadeClaudeCode != null) saccadeClaudeCode;
          meta.mainProgram = "sac";
        };
      }
      // lib.optionalAttrs (saccadeClaudeCode != null) {
        # the composed Claude Code harness: the pinned binary, the
        # SessionStart identity hook, the injected settings
        saccade-claude-code = saccadeClaudeCode;
      }
      // lib.optionalAttrs (devHerdrCore != null) {
        # the claim itself: the unpatched pinned herdr-core, runnable
        # by anyone with the flake; no source patches, no fork
        herdr-core = devHerdrCore;
      }
      // lib.optionalAttrs (devHerdrPin != null) {
        # the development arm: the same bytes behind the dev sandbox
        herdr-core-dev = devHerdrPin;
      };
      # one installable that builds every package: the merge gate
      # and the offer-time build name it, so nothing reaches a
      # review unbuilt. Heavier than its parts alone: it carries
      # the pins and the archive too.
      all = pkgs.releaseTools.aggregate {
        name = "saccade-all";
        constituents = lib.attrValues pkgSet;
      };
    in
    pkgSet
    // {
      inherit all;
    }
  );

  apps = forAllSystems (
    system:
    let
      saccadeAppBin = (saccadeApp nixpkgs-client.legacyPackages.${system} ../.).app;
    in
    {
      saccade-app = {
        type = "app";
        program = "${saccadeAppBin}/bin/saccade-app";
      };
      default = {
        type = "app";
        program = "${saccadeAppBin}/bin/saccade-app";
      };
    }
  );

  # no tests inside nix: the suites, the frontend checks and the
  # contract tests run in the devshell and the lanes, never behind a
  # sandbox cold compile; the flake carries no checks output

  formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);
}
