{
  description = "Saccade devshell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # the client nixpkgs: the revision the human's system flake is
    # built from. The board-viewer package and app build against it so
    # the build agents test is the build the client runs; the tracker's
    # own packages, checks, and devshells stay on nixpkgs.
    nixpkgs-client.url = "github:NixOS/nixpkgs/c59305bab2065cfecc4944690d9eedbb56f3a9fa";
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
    # the desktop app, the second side of the pairing: the 0.1.11
    # line the laptop acceptance client runs, pinned the same way the
    # core is (app 0.1.11 + core 0.1.17 - placement belongs to the
    # door, no mod between them)
    herdr-app-x86_64-linux = {
      url = "https://releases.herdr.dev/app/0.1.11/herdr-app-linux-x86_64.tar.gz";
      flake = false;
    };
    herdr-app-aarch64-linux = {
      url = "https://releases.herdr.dev/app/0.1.11/herdr-app-linux-aarch64.tar.gz";
      flake = false;
    };
    # the pinned Claude Code, one official npm platform package per
    # target; see claudeCodePin beside the derivations
    claude-code-x86_64-linux = {
      url = "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.318.tgz";
      flake = false;
    };
    claude-code-aarch64-linux = {
      url = "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-arm64/-/claude-code-linux-arm64-2.1.318.tgz";
      flake = false;
    };
  };

  nixConfig = {
    extra-substituters = ["https://devenv.cachix.org"];
    extra-trusted-public-keys = ["devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="];
  };

  outputs = inputs: import ./nix/outputs.nix inputs;

}
