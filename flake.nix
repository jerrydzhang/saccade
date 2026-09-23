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
    llm-agents.url = "github:numtide/llm-agents.nix";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-substituters = ["https://devenv.cachix.org"];
    extra-trusted-public-keys = ["devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="];
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    saccade,
    llm-agents,
    devenv,
    ...
  } @ inputs: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    overlays = [rust-overlay.overlays.default];
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
            ({...}: {
              packages = with pkgs; [
                rustToolchain
                (pkgs.python3.withPackages (ps: [ps.playwright]))
                prek
                just
                saccade.packages.${system}.default
                llm-agents.packages.${system}.pi
              ];

              env = {
                PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
                # development override for the pinned executor; the
                # package bakes SACCADE_PI_PATH instead
                SACCADE_PI = "${llm-agents.packages.${system}.pi}/bin/pi";
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
      in {
        default = pkgs.rustPlatform.buildRustPackage {
          name = "saccade";
          src = ./.;
          buildInputs = [];
          # the suite's runner tests exercise git worktrees and the
          # state root under a writable home; the executor tests need
          # python3 for the fake-Pi stub
          nativeBuildInputs = [
            pkgs.git
            pkgs.python3
            llm-agents.packages.${system}.pi
          ];
          # the executor pin: never the ambient binary. The check phase
          # runs the real-pi smoke and the fake-Pi stub against this path
          env.SACCADE_PI_PATH = "${llm-agents.packages.${system}.pi}/bin/pi";
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
