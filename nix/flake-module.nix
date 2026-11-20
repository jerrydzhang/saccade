# The host tooling both projects use: the package set with the rust
# overlay, the rust toolchain the dev shells carry, the pinned
# herdr-core and its development arm, the formatter, and the one
# installable that builds every package. The projects' own modules
# (saccade/flake-module.nix, app/flake-module.nix) read the shared
# pieces from the perSystem argument `host`.
{ inputs, ... }:
let
  inherit (inputs)
    nixpkgs
    rust-overlay
    herdr-core-x86_64-linux
    herdr-core-aarch64-linux
    ;
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
in
{
  systems = nixpkgs.lib.systems.flakeExposed;

  perSystem =
    {
      system,
      config,
      pkgs,
      lib,
      ...
    }:
    let
      herdrCore =
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
        if herdrCore == null then
          null
        else
          mkHerdrDev {
            inherit pkgs herdrCore;
          };
    in
    {
      _module.args.pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      _module.args.host = {
        inherit herdrCore herdrDev;
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
        };
      };

      packages =
        lib.optionalAttrs (herdrCore != null) {
          # the claim itself: the unpatched pinned herdr-core, runnable
          # by anyone with the flake; no source patches, no fork
          herdr-core = herdrCore;
        }
        // lib.optionalAttrs (herdrDev != null) {
          # the development arm: the same bytes behind the dev sandbox
          herdr-core-dev = herdrDev;
        }
        // {
          # one installable that builds every package: the merge gate
          # and the offer-time build name it, so nothing reaches a
          # review unbuilt. Heavier than its parts alone: it carries
          # the pins and the archive too.
          all = pkgs.releaseTools.aggregate {
            name = "saccade-all";
            constituents = [
              config.packages.saccade
              config.packages.app
            ]
            ++ lib.optional (herdrCore != null) herdrCore
            ++ lib.optional (herdrDev != null) herdrDev;
          };
        };

      # no tests inside nix: the suites, the frontend checks and the
      # contract tests run in the devshells and the lanes, never behind
      # a sandbox cold compile; the flake carries no checks output

      formatter = nixpkgs.legacyPackages.${system}.nixfmt-tree;
    };
}
