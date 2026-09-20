{
  description = "Saccade devshell setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    ...
  }: let
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
        default = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain
            (pkgs.python3.withPackages (ps: [ps.playwright]))
            prek
            just
          ];

          shellHook = ''
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
          '';
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
          nativeBuildInputs = [];
          cargoLock.lockFile = ./Cargo.lock;
        };
      }
    );

    formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
