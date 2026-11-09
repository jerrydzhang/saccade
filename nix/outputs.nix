{
    nixpkgs,
    nixpkgs-client,
    rust-overlay,
    saccade,
    wrapped-pi,
    devenv,
    herdr-core-x86_64-linux,
    herdr-core-aarch64-linux,
    herdr-app-x86_64-linux,
    herdr-app-aarch64-linux,
    claude-code-x86_64-linux,
    claude-code-aarch64-linux,
    ...
  } @ inputs: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    rustOverlays = [rust-overlay.overlays.default];
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
    # the three X runtime libs the release binary dynamically links
    # and the tarball does not carry
    # the libraries the release binary needs, split by how it finds
    # them: libxcb and libxkbcommon are DT_NEEDED (ldd sees them); the
    # wayland client stack and the vulkan loader are dlopened by gpui's
    # linux platform at init and surface creation (ldd never shows
    # them, and a clean env panics without them); mesa carries libGL,
    # libEGL, the software vulkan driver, and the ICD files
    herdrAppLibs = pkgs:
      pkgs.lib.makeLibraryPath [
        pkgs.libxcb
        pkgs.libxkbcommon
        pkgs.wayland
        pkgs.vulkan-loader
        pkgs.mesa
      ];
    # the app release's own manifest hashes its tarball; the lock pins
    # those bytes and the derivation asserts the unpacked binary
    herdrAppPin = {
      x86_64-linux = {
        src = herdr-app-x86_64-linux;
        binarySha256 = "80bcd0405e59834f4c76c8458551d3f5134cde650651e4ad917c51e9366d3b42";
      };
      aarch64-linux = {
        src = herdr-app-aarch64-linux;
        binarySha256 = "db7ddf3cb0b8519573490b729abf7eee7a2472436e2ddb490ad0686504edbe17";
      };
    };
    mkHerdrApp = {
      pkgs,
      src,
      binarySha256,
    }:
      pkgs.runCommand "herdr-app-pinned" {} ''
        set -o errexit
        echo "${binarySha256}  ${src}/herdr-app" | ${pkgs.coreutils}/bin/sha256sum --check --strict -
        mkdir -p $out/bin $out/share
        ${pkgs.coreutils}/bin/install -m 0555 ${src}/herdr-app $out/bin/herdr-app
        ${pkgs.coreutils}/bin/install -m 0444 ${src}/LICENSE \
          ${src}/THIRD-PARTY-NOTICES ${src}/herdr-app.svg $out/share/
      '';
    mkHerdrCore = {
      pkgs,
      src,
      binarySha256,
    }:
      pkgs.runCommand "herdr-core-pinned" {} ''
        set -o errexit
        echo "${binarySha256}  ${src}/herdr-core" | ${pkgs.coreutils}/bin/sha256sum --check --strict -
        mkdir -p $out/bin
        ${pkgs.coreutils}/bin/install -m 0555 ${src}/herdr-core $out/bin/herdr-core
        ${pkgs.coreutils}/bin/install -m 0444 ${src}/LICENSE ${src}/THIRD-PARTY-NOTICES $out/
      '';
    # the development arm: scratch HOME/XDG plus hc's own session
    # scoping (HC_SESSION), so a dev core and its sockets live apart
    # from the machine server even with the ambient install present
    mkHerdrDev = {
      pkgs,
      herdrCore,
    }:
      pkgs.writeShellApplication {
        name = "herdr-core";
        runtimeInputs = [pkgs.coreutils];
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
    herdrCore = system: pkgs: let
      pin = herdrPin.${system} or null;
    in
      if pin == null
      then null
      else
        mkHerdrCore {
          inherit pkgs;
          inherit (pin) src binarySha256;
        };
    # the app as saccade ships it: a wrapper carrying the runtime
    # libs the release links, nothing seeded - placement is the
    # door's declaration, and no mod rides between them
    mkHerdrAppSaccade = {
      pkgs,
      herdrApp,
    }:
      pkgs.writeShellApplication {
        name = "herdr-app-saccade";
        runtimeInputs = [pkgs.coreutils];
        text = ''
          export LD_LIBRARY_PATH=${herdrAppLibs pkgs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          # the vulkan loader's ICD discovery rides the host's data dirs
          # first (a real GPU on a desktop keeps its drivers); mesa's
          # share is appended so a headless box still finds lavapipe
          export XDG_DATA_DIRS=${pkgs.mesa}/share:''${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}
          # HERDR_APP_HOME is the app's whole home when set, used raw;
          # otherwise the config dir's herdr-app folder
          home=''${HERDR_APP_HOME:-''${XDG_CONFIG_HOME:-''${HOME:-/nonexistent}/.config}/herdr-app}
          # the retired mod's seeded copy leaves with the component:
          # the wrapper removes only what it seeded (the saccade folder
          # carrying its own stamp) - anything else under mods is the
          # person's and stays, and the app starts on main when the
          # selected layout file is gone
          if [ -f "$home/mods/saccade/.version" ]; then
            rm -rf "$home/mods/saccade"
            rm -f "$home/layouts/saccade.json"
          fi
          exec ${herdrApp}/bin/herdr-app "$@"
        '';
      };
    mkHerdrUi = args: import ./herdr-ui.nix (args // {
      inherit mkHerdrAppSaccade;
      script = ../tools/herdr-ui/herdr_ui.py;
    });
    herdrApp = system: pkgs: let
      pin = herdrAppPin.${system} or null;
    in
      if pin == null
      then null
      else
        mkHerdrApp {
          inherit pkgs;
          inherit (pin) src binarySha256;
        };
    herdrDev = system: pkgs: let
      core = herdrCore system pkgs;
    in
      if core == null
      then null
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
    cargoVersion = let
      line = lib.findFirst
        (l: builtins.match ''version = "[0-9]+\.[0-9]+\.[0-9]+"'' l != null)
        null
        (lib.splitString "\n" (builtins.readFile ../tracker/Cargo.toml));
    in
      if line == null
      then throw "Cargo.toml carries no readable version line"
      else builtins.substring 11 (builtins.stringLength line - 12) line;
    mkClaudeCode = {
      pkgs,
      src,
      binarySha256,
    }:
      pkgs.runCommand "claude-code-pinned" {
        nativeBuildInputs = [pkgs.autoPatchelfHook];
        buildInputs = [pkgs.glibc];
      } ''
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
    mkAttachHook = {pkgs}: let
      substituted = pkgs.replaceVars ../tracker/native/claude-code/saccade-cc-attach.template {
        expectMajor = lib.versions.major cargoVersion;
        expectMinor = lib.versions.minor cargoVersion;
      };
    in
      pkgs.runCommand "saccade-cc-attach" {
        nativeBuildInputs = [pkgs.shellcheck];
      } ''
        set -o errexit
        ${pkgs.shellcheck}/bin/shellcheck ${substituted}
        ${pkgs.bash}/bin/bash -n ${substituted}
        mkdir -p $out/bin
        ${pkgs.coreutils}/bin/install -m 0555 ${substituted} $out/bin/saccade-cc-attach
      '';
    mkCCSettings = {pkgs, attachHook}:
      pkgs.runCommand "saccade-claude-code-settings.json" {} ''
        set -o errexit
        export attachHook="${attachHook}"
        substituteAll ${../tracker/native/claude-code/settings.json.template} $out
        # the packaged file must parse as settings: a template that no
        # longer carries the hook wire refuses the build here
        ${pkgs.jq}/bin/jq -e .hooks.SessionStart $out >/dev/null
      '';
    # the composed harness: the pinned binary with the packaged
    # per-launch settings injected ahead of the caller's argv, which
    # passes through untouched
    mkSaccadeClaudeCode = {
      pkgs,
      claude,
      attachHook,
    }: let
      settings = mkCCSettings {inherit pkgs attachHook;};
      wrapper = pkgs.replaceVars ../tracker/native/claude-code/saccade-claude-code.template {
        inherit claude settings;
        jq = pkgs.jq;
        curl = pkgs.curl;
      };
    in
      pkgs.runCommand "saccade-claude-code" {
        nativeBuildInputs = [pkgs.shellcheck];
      } ''
        set -o errexit
        ${pkgs.shellcheck}/bin/shellcheck ${wrapper}
        ${pkgs.bash}/bin/bash -n ${wrapper}
        mkdir -p $out/bin
        ${pkgs.coreutils}/bin/install -m 0555 ${wrapper} $out/bin/saccade-claude-code
      '';
    saccadeClaudeCodeFor = system: pkgs: let
      pin = claudeCodePin.${system} or null;
      claude =
        if pin == null
        then null
        else
          mkClaudeCode {
            inherit pkgs;
            inherit (pin) src binarySha256;
          };
    in
      if claude == null
      then null
      else
        mkSaccadeClaudeCode {
          inherit pkgs;
          claude = "${claude}/bin/claude";
          attachHook = "${mkAttachHook {inherit pkgs;}}/bin/saccade-cc-attach";
        };
    # the native integration the composed harness loads
    saccadePiExtensions = ["tracker/native/saccade.ts"];
    # the Saccade app's packaging lives in nix/saccade-app.nix: one
    # callPackage-able file taking every dependency from the pkgs it
    # is called on, so overlays.default can build it against a
    # consumer's own nixpkgs - the packaged binary then shares the
    # host's GL stack instead of this flake's pin. The repo root rides
    # in explicitly: the app's cargo tree spans client/src-tauri and
    # the shared grammar crate, and package paths are rooted explicitly
    # would stay an unnormalized ".." path inside flake evaluation.
    saccadeApp =
      pkgs: root:
        pkgs.callPackage ./saccade-app.nix {repoRoot = root;};
  in {
    # the consumer-facing overlay: a NixOS configuration sets
    # nixpkgs.overlays = [ inputs.saccade.overlays.default ] and gets
    # pkgs.saccade-app built against its own nixpkgs
    overlays.default = final: prev: {
      saccade-app = (final.callPackage ./saccade-app.nix {repoRoot = ../.;}).app;
    };

    devShells = forAllSystems (
      system: let
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
      in {
        # the Saccade app's native dev shell: its own dependency graph,
        # separate from the tracker devshell and the tracker crate. The
        # WebKitGTK/GTK build inputs here serve the viewer's Tauri crate
        # only; nothing here touches the tracker's build.
        # the browser lane's fontconfig: a generated config over pinned
        # DejaVu fonts, both pure store paths - the sandbox already binds
        # /nix/store read-only, so no host /etc/fonts or font dirs are
        # ever mounted
        saccade-app = let
          laneFontsConf = pkgs.writeText "saccade-lane-fonts.conf" ''
            <?xml version="1.0"?>
            <!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
            <fontconfig>
              <dir>${pkgs.dejavu_fonts}</dir>
              <include>${pkgs.fontconfig.out}/etc/fonts/conf.d</include>
              <cachedir prefix="xdg">fontconfig</cachedir>
            </fontconfig>
          '';
        in pkgs.mkShell {
          # the pinned Playwright pair (python package and browser
          # package) from this same nixpkgs revision: no browser
          # install or download anywhere
          packages =
            [
              rustToolchain
              pkgs.nodejs
              pkgs.pkg-config
              (pkgs.python3.withPackages (ps: [ps.playwright ps.pyflakes]))
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
            ({config, ...}: let
              # the dev wrapper: wrapped-pi pointed at this repo's
              # working-tree extension, no adaptation, no preload
              devPi = let
                wrappedPiBin = "${wrapped-pi.packages.${system}.default}/bin/wrapped-pi";
                extraArgs = lib.escapeShellArgs (lib.concatMap (path: ["-e" "${config.devenv.root}/${path}"]) saccadePiExtensions);
              in
                pkgs.lib.hiPrio (pkgs.writeShellApplication {
                  name = "saccade-pi";
                  runtimeInputs = [pkgs.util-linux];
                  text = ''
                    exec ${wrappedPiBin} ${extraArgs} "$@"
                  '';
                });
              devHerdr = herdrDev system pkgs;
              # the composed Claude Code harness the devshell names and
              # carries: the same package the release bundles, so the
              # development override and the compile pin are one line
              devCC = saccadeClaudeCodeFor system pkgs;
            in {
              enterShell = ''
                if [ ! -f .agents/skills/saccade/SKILL.md ]; then
                  cargo run --quiet -p saccade --bin sac -- skill install
                fi
              '';

              packages =
                with pkgs; [
                  rustToolchain
                  (pkgs.python3.withPackages (ps: [ps.playwright ps.pyflakes]))
                  prek
                  just
                  typescript
                  nodejs
                  direnv
                  jq
                  curl
                  saccade.packages.${system}.default
                  devPi
                ]
                ++ lib.optional (devCC != null) devCC;

              env =
                {
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
                exec = "${saccade.packages.${system}.default}/bin/sac serve --repo /home/jerry/Projects/saccade --bind 0.0.0.0 --port 8811";
                restart = {
                  on = "on_failure";
                  max = 10;
                };
                ready.http.get = {
                  port = 8811;
                  path = "/";
                };
              };
            })
          ];
        };
      }
    );

    packages = forAllSystems (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = rustOverlays;
        };
        clientPkgs = nixpkgs-client.legacyPackages.${system};
        devHerdrCore = herdrCore system pkgs;
        devHerdrPin = herdrDev system pkgs;
        devHerdrApp = herdrApp system pkgs;
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
          runtimeInputs = [pkgs.util-linux];
          text = ''
            exec ${pi}/bin/wrapped-pi "$@"
          '';
        };
        sac = import ./sac.nix {
          inherit pkgs lib saccadePi devHerdrPin saccadeClaudeCode;
          root = ../.;
        };
        # the pinned archive: main's 0.18.0 tip, the release the live
        # tracker ran; the rev is the pin, the use is read-only.
        sac-archive = import ./sac-archive.nix {
          inherit pkgs lib;
          archiveRev = "189894f54f4c747ad937493f34c5e526a76726b8";
          archiveSrc =
            builtins.fetchGit {
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
          src = ../.;
          cargoRoot = ".";
          buildAndTestSubdir = "grammar";
          cargoLock.lockFile = ../Cargo.lock;
          meta.mainProgram = "board-grammar";
        };
        # the agents companion: same workspace-member shape as the
        # grammar, one binary per hub
        companion = pkgs.rustPlatform.buildRustPackage {
          pname = "saccade-companion";
          version = "0.1.0";
          doCheck = false;
          src = ../.;
          cargoRoot = ".";
          buildAndTestSubdir = "companion";
          cargoLock.lockFile = ../Cargo.lock;
          meta.mainProgram = "sac-companion";
        };
      in
        let
          pkgSet =
            {
              sac = sac;

              sac-archive = sac-archive;

              board-grammar = board-grammar;

              companion = companion;

              saccade-pi = saccadePi;

              # the app the client runs: built against nixpkgs-client, the
              # revision the human's system flake is built from
              saccade-app = (saccadeApp clientPkgs ../.).app;
              saccade-app-web = (saccadeApp pkgs ../.).web;
              saccade-app-tauri-driver = (saccadeApp pkgs ../.).driver;

              default = pkgs.symlinkJoin {
                name = "saccade";
                paths =
                  [sac saccadePi]
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
            }
            // lib.optionalAttrs (devHerdrApp != null) {
              # the pairing's app side: the pinned app and the wrapper that
              # carries its runtime libs - runnable with nix, no manual
              # file installs, nothing seeded
              herdr-app = devHerdrApp;
              herdr-app-saccade = mkHerdrAppSaccade {
                inherit pkgs;
                herdrApp = devHerdrApp;
              };
            }
            // lib.optionalAttrs (pkgs.stdenv.hostPlatform.isLinux && devHerdrApp != null && devHerdrCore != null) {
              herdr-ui = mkHerdrUi {
                inherit pkgs;
                herdrApp = devHerdrApp;
                herdrCore = devHerdrCore;
              };
            };
          # one installable that builds every package: the merge gate
          # and the offer-time build name it, so nothing reaches a
          # review unbuilt. Heavier than its parts alone: it carries
          # the desktop wrapper, the pins and the archive too.
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

    apps = forAllSystems (system: let
      saccadeAppBin = (saccadeApp nixpkgs-client.legacyPackages.${system} ../.).app;
    in {
      saccade-app = {
        type = "app";
        program = "${saccadeAppBin}/bin/saccade-app";
      };
      default = {
        type = "app";
        program = "${saccadeAppBin}/bin/saccade-app";
      };
    });

    # no tests inside nix: the suites, the frontend checks and the
    # contract tests run in the devshell and the lanes, never behind a
    # sandbox cold compile; the flake carries no checks output

    formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.alejandra);
}
