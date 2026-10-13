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
    saccadePiExtensions = [];
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
              devPi = pkgs.writeShellApplication {
                name = "wrapped-pi";
                text = ''
                  exec ${wrapped-pi.packages.${system}.default}/bin/wrapped-pi ${lib.escapeShellArgs (lib.concatMap (path: ["-e" "${config.devenv.root}/${path}"]) saccadePiExtensions)} "$@"
                '';
              };
            in {
              packages = with pkgs; [
                rustToolchain
                (pkgs.python3.withPackages (ps: [ps.playwright]))
                prek
                just
                saccade.packages.${system}.default
                devPi
              ];

              env = {
                SACCADE_PI = "${devPi}/bin/wrapped-pi";
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
      in
        {
          default = pkgs.rustPlatform.buildRustPackage {
            name = "saccade";
            src = ./.;
            buildInputs = [];
            # the api tests exercise real git worktrees under a
            # writable home, through a real direnv layer
            nativeBuildInputs = [
              pkgs.git
              pkgs.direnv
            ];
            # the pinned harness this binary's start door launches;
            # SACCADE_PI remains the development override
            SACCADE_PI = "${pi}/bin/wrapped-pi";
            preCheck = ''
              export HOME=$(mktemp -d)
            '';
            cargoLock.lockFile = ./Cargo.lock;
            meta.mainProgram = "sac";
          };
        }
    );

    formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
