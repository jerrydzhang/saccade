{
  description = "Saccade devshell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    saccade = {
      url = "github:jerrydzhang/saccade";
      # the pinned release's self-input would nest one generation per
      # release bump; it aliases the same pin, and only the release's
      # devShell wants it
      inputs.saccade.follows = "saccade";
    };
    wrapped-pi.url = "github:jerrydzhang/wrapped-pi";
    devenv.url = "github:cachix/devenv";
    # the dev host: one herdr-core release per tracker version. The
    # lock entry pins the tarball bytes (narHash) and the derivation
    # asserts the binary against the release manifest's sha256, so the
    # tested claim stays the exact tested bytes. The curl installer is
    # not a door; this is a fixed derivation per target.
    herdr-core-x86_64-linux = {
      url = "https://releases.herdr.dev/core/0.1.17/herdr-core-linux-x86_64.tar.gz";
      flake = false;
    };
    herdr-core-aarch64-linux = {
      url = "https://releases.herdr.dev/core/0.1.17/herdr-core-linux-aarch64.tar.gz";
      flake = false;
    };
    # the desktop app, the second side of the spaces realization: the
    # 0.1.11 line the laptop acceptance client runs, pinned the same
    # way the core is, completing the versioned triple (app 0.1.11 +
    # core 0.1.17 + the mod's own version)
    herdr-app-x86_64-linux = {
      url = "https://releases.herdr.dev/app/0.1.11/herdr-app-linux-x86_64.tar.gz";
      flake = false;
    };
    herdr-app-aarch64-linux = {
      url = "https://releases.herdr.dev/app/0.1.11/herdr-app-linux-aarch64.tar.gz";
      flake = false;
    };
  };

  nixConfig = {
    extra-substituters = ["https://devenv.cachix.org"];
    extra-trusted-public-keys = ["devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="];
  };

  outputs = {
    nixpkgs,
    rust-overlay,
    saccade,
    wrapped-pi,
    devenv,
    herdr-core-x86_64-linux,
    herdr-core-aarch64-linux,
    herdr-app-x86_64-linux,
    herdr-app-aarch64-linux,
    ...
  } @ inputs: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    overlays = [rust-overlay.overlays.default];
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
    # the spaces mod's own version, the triple's third leg; a change
    # here re-seeds the user's copy through the wrapper's stamp
    saccadeModVersion = "0.1.0";
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
    # the spaces mod's store path: the mod folder plus the derived
    # layout (main plus the saccade panel), one seeding bundle
    mkSaccadeSpacesMod = {pkgs}:
      pkgs.runCommand "saccade-spaces-mod-${saccadeModVersion}" {} ''
        mkdir -p $out/mods/saccade $out/mods/tmux $out/layouts
        for f in mod.json main.js plan.js; do
          ${pkgs.coreutils}/bin/install -m 0444 ${./mods/saccade}/$f $out/mods/saccade/$f
        done
        for f in mod.json main.js keys.json; do
          ${pkgs.coreutils}/bin/install -m 0444 ${./mods/tmux}/$f $out/mods/tmux/$f
        done
        ${pkgs.coreutils}/bin/install -m 0444 ${./mods/saccade-layout.json} $out/layouts/saccade.json
        echo -n ${saccadeModVersion} > $out/mods/saccade/.version
      '';
    # the app with the mod: a first-run seeding wrapper, the app's own
    # sanctioned config path (mods the app scans and runs live). The
    # stamp re-seeds when the mod's version moves; a changed mod file
    # with a current stamp is a human edit and stays.
    mkHerdrAppSaccade = {
      pkgs,
      herdrApp,
      mod,
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
          # a deployment-owned home: the app's HERDR_APP_HOME is used raw
          # when the caller sets one, and otherwise this deployment's own
          # folder - never the shared ~/.config/herdr-app, which belongs to
          # the person (their hand-made mods, layouts and keys stay theirs,
          # and this wrapper never writes there)
          home=''${HERDR_APP_HOME:-''${XDG_CONFIG_HOME:-''${HOME:-/nonexistent}/.config}/herdr-app-saccade}
          export HERDR_APP_HOME="$home"
          mods=$home/mods
          stamp=$mods/saccade/.version
          # the stamp is the mod package's store path - its content hash:
          # a rebuilt mod re-seeds, a human-edited one whose stamp matches
          # stays
          want=${mod}
          if [ "$stamp" != "$want" ]; then
            mkdir -p "$mods/saccade" "$mods/tmux" "$home/layouts"
            for f in mod.json main.js plan.js; do
              rm -f "$mods/saccade/$f"
              install -m 0644 "${mod}/mods/saccade/$f" "$mods/saccade/$f"
            done
            for f in mod.json main.js keys.json; do
              rm -f "$mods/tmux/$f"
              install -m 0644 "${mod}/mods/tmux/$f" "$mods/tmux/$f"
            done
            # the deployment no longer ships a config override: a stale
            # one from an older stamp would pin a wrong root
            rm -f "$mods/saccade/config.json"
            printf '%s' "$want" > "$stamp"
            # the derived layout seeds always; the selection seeds only a
            # fresh home - an existing home keeps its own layout choice,
            # and switching once is the app's own documented act
            install -m 0444 "${mod}/layouts/saccade.json" "$home/layouts/saccade.json"
            if [ ! -f "$home/shared.json" ]; then
              printf '{"layout":"saccade"}' > "$home/shared.json"
            fi
          fi
          exec ${herdrApp}/bin/herdr-app "$@"
        '';
      };
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
    # the native integration the composed harness loads
    saccadePiExtensions = ["native/saccade.ts"];
  in {

    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
        };
      in {
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
                pkgs.hiPrio (pkgs.writeShellApplication {
                  name = "saccade-pi";
                  runtimeInputs = [pkgs.util-linux];
                  text = ''
                    exec ${wrappedPiBin} ${extraArgs} "$@"
                  '';
                });
              devHerdr = herdrDev system pkgs;
            in {
              packages = with pkgs; [
                rustToolchain
                (pkgs.python3.withPackages (ps: [ps.playwright]))
                prek
                just
                typescript
                nodejs
                direnv
                saccade.packages.${system}.default
                devPi
              ];

              env = {
                SACCADE_PI = "${devPi}/bin/saccade-pi";
                PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
                # no SACCADE_HERDR here, by ruling: interactive use hits the
                # ambient production core (the orca shape); the sandboxed
                # dev arm is the opt-in flake output herdr-core-dev, never a
                # PATH entry - a wrapper named herdr-core would shadow
                # ambient resolution and recreate the posture seam
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
          inherit system overlays;
        };
        devHerdrCore = herdrCore system pkgs;
        devHerdrPin = herdrDev system pkgs;
        devHerdrApp = herdrApp system pkgs;
        devMod = mkSaccadeSpacesMod {inherit pkgs;};
        pi = wrapped-pi.packages.${system}.default.override {
          extraExtensions = map (path: ./. + "/${path}") saccadePiExtensions;
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
        # the workbench frontend, built once from the committed sources
        # and lock; the binary embeds these three files through the
        # SACCADE_WEB_ASSETS build interface
        webAssets = pkgs.buildNpmPackage {
          pname = "saccade-workbench-web";
          version = "0.16.2";
          src = pkgs.lib.sourceByRegex ./web [
            "^src$"
            "^src/.*\\.mjs$"
            "^package\\.json$"
            "^package-lock\\.json$"
            "^index\\.html$"
            "^style\\.css$"
            "^ascii\\.mjs$"
            "^copy-assets\\.mjs$"
          ];
          npmDepsHash = "sha256-vtELfcqrPnAcODazFjmSQCx0gjcp/JzABR+hcTiOFjI=";
          installPhase = ''
            mkdir -p $out
            cp dist/index.html dist/app.js dist/style.css $out/
          '';
        };
        sac = pkgs.rustPlatform.buildRustPackage ({
          name = "saccade";
          src = ./.;
          buildInputs = [];
          # the api tests exercise real git worktrees under a
          # writable home, through a real direnv layer
          nativeBuildInputs = [
            pkgs.git
            pkgs.direnv
            # flock guards the native-door tests' scratch holders
            pkgs.util-linux
          ];
          # the pinned harness this binary's start door launches;
          # SACCADE_PI remains the development override
          SACCADE_PI = "${saccadePi}/bin/saccade-pi";
          # the launch doors equip the workspace through this direnv; the
          # baked store path is both the resolution and the closure edge,
          # never a runtime override
          SACCADE_DIRENV = "${pkgs.direnv}/bin/direnv";
          # the embedded workbench assets ride the build interface,
          # never a runtime path
          SACCADE_WEB_ASSETS = webAssets;
          # the fish script the binary itself emits installs as a vendor
          # completion, so fish picks it up with zero user config and it
          # updates atomically with the binary; the script invokes sac
          # by PATH name, never a baked store path
          postInstall = ''
            mkdir -p $out/share/fish/vendor_completions.d
            $out/bin/sac completions fish > $out/share/fish/vendor_completions.d/sac.fish
          '';
          # the unattended-prose door alone drives a real host pty;
          # the nix sandbox mounts no /dev/pts and it cannot open the
          # slave there, while the full suite passes in the devShell
          checkFlags = [
            "--skip"
            "piped_human_prose_passes_and_unattended_refuses"
          ];
          preCheck = ''
            export HOME=$(mktemp -d)
          '';
          cargoLock.lockFile = ./Cargo.lock;
          meta.mainProgram = "sac";
        }
        // lib.optionalAttrs (devHerdrPin != null) {
          # the pinned dev host this binary resolves last; SACCADE_HERDR
          # remains the explicit override and the ambient binary the
          # production arm
          SACCADE_HERDR_PIN = "${devHerdrPin}/bin/herdr-core";
        });
      in
        {
          saccade-pi = saccadePi;

          saccade-web-assets = webAssets;

          default = pkgs.symlinkJoin {
            name = "saccade";
            paths = [sac saccadePi];
            meta.mainProgram = "sac";
          };
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
          # the spaces realization's second side: the pinned app, the
          # mod folder alone, and the app with the mod seeded - the
          # versioned triple's app side, runnable with nix, no manual
          # file installs
          herdr-app = devHerdrApp;
          saccade-spaces-mod = devMod;
          herdr-app-saccade = mkHerdrAppSaccade {
            inherit pkgs;
            herdrApp = devHerdrApp;
            mod = devMod;
          };
        }
    );

    formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
