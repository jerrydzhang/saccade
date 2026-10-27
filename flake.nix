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
    ...
  } @ inputs: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    overlays = [rust-overlay.overlays.default];
    # the native integration the composed harness loads
    saccadePiExtensions = ["native/saccade.ts"];
    # the composition's own receipt preload, copied into the store as a
    # pair so the registration path is a space-free store path
    saccadeCompositionPreloadFiles = ["native/saccade-composition-shim.mjs" "native/saccade-composition-hooks.mjs"];
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
              # the composition's receipt preload, copied into the
              # store as a pair: the registration path stays a
              # space-free store path, and the hook transforms nothing
              # without the pinned anchor, so inherited registrations
              # in child processes stay inert
              compositionPreload = pkgs.runCommand "saccade-composition-preload" {} ''
                mkdir -p $out
                ${lib.concatMapStrings (path: "cp ${./. + "/${path}"} $out/${baseNameOf path}\n") saccadeCompositionPreloadFiles}
              '';
              devPi = lib.hiPrio (pkgs.writeShellApplication {
                name = "saccade-pi";
                text = ''
                  export NODE_OPTIONS="--import ${compositionPreload}/saccade-composition-shim.mjs''${NODE_OPTIONS:+ $NODE_OPTIONS}"
                  exec ${wrapped-pi.packages.${system}.default}/bin/wrapped-pi ${lib.escapeShellArgs (lib.concatMap (path: ["-e" "${config.devenv.root}/${path}"]) saccadePiExtensions)} "$@"
                '';
              });
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
              };

              process.manager.implementation = "native";
              processes.sac = {
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
        pi = wrapped-pi.packages.${system}.default.override {
          extraExtensions = map (path: ./. + "/${path}") saccadePiExtensions;
        };
        compositionPreload = pkgs.runCommand "saccade-composition-preload" {} ''
          mkdir -p $out
          ${lib.concatMapStrings (path: "cp ${./. + "/${path}"} $out/${baseNameOf path}\n") saccadeCompositionPreloadFiles}
        '';
        saccadePi = pkgs.writeShellApplication {
          name = "saccade-pi";
          # util-linux rides the closure: the integration's guard helper
          # is flock, never an assumed ambient binary
          runtimeInputs = [pkgs.util-linux];
          text = ''
            export NODE_OPTIONS="--import ${compositionPreload}/saccade-composition-shim.mjs"
            exec ${pi}/bin/wrapped-pi "$@"
          '';
        };
        # the workbench frontend, built once from the committed sources
        # and lock; the binary embeds these three files through the
        # SACCADE_WEB_ASSETS build interface
        webAssets = pkgs.buildNpmPackage {
          pname = "saccade-workbench-web";
          version = "0.15.0";
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
        sac = pkgs.rustPlatform.buildRustPackage {
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
        };
      in {
        saccade-pi = saccadePi;

        saccade-web-assets = webAssets;

        default = pkgs.symlinkJoin {
          name = "saccade";
          paths = [sac saccadePi];
          meta.mainProgram = "sac";
        };
      }
    );

    formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
