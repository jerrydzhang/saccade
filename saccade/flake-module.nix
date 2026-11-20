# The saccade project's flake outputs: the tracker (sac), the board
# grammar crate, the pinned archive, the composed harnesses (pi and
# Claude Code), the default package the release consumers pin, the
# project's aggregate, and its devenv shell. Every build here takes its
# source from this directory only.
{ inputs, ... }:
let
  inherit (inputs)
    nixpkgs
    saccade
    wrapped-pi
    devenv
    claude-code-x86_64-linux
    claude-code-aarch64-linux
    ;
  inherit (nixpkgs) lib;
  # the cargo workspace the tracker and the grammar build from: the
  # workspace manifest, its lock and the two crates, nothing else of
  # the repository
  cargoSrc = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./Cargo.toml
      ./Cargo.lock
      ./tracker
      ./grammar
    ];
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
        lib.splitString "\n" (builtins.readFile ./tracker/Cargo.toml)
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
      substituted = pkgs.replaceVars ./tracker/native/claude-code/saccade-cc-attach.template {
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
      substituted = pkgs.replaceVars ./tracker/native/claude-code/saccade-cc-worktree.template { };
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
      substituteAll ${./tracker/native/claude-code/settings.json.template} $out
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
      wrapper = pkgs.replaceVars ./tracker/native/claude-code/saccade-claude-code.template {
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
  # the native integration the composed harness loads, relative to
  # this directory
  saccadePiExtensions = [ "tracker/native/saccade.ts" ];
  # the packaged copy of each extension: only the extension file,
  # at its own relative path, so the harness never depends on the
  # rest of the repository
  packagedExtension =
    path:
    "${
      lib.fileset.toSource {
        root = ./.;
        fileset = ./. + "/${path}";
      }
    }/${path}";
in
{
  perSystem =
    {
      system,
      config,
      pkgs,
      host,
      ...
    }:
    let
      saccadeClaudeCode = saccadeClaudeCodeFor system pkgs;
      pi = wrapped-pi.packages.${system}.default.override {
        extraExtensions = map packagedExtension saccadePiExtensions;
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
      sac = import ./nix/sac.nix {
        inherit
          pkgs
          lib
          saccadePi
          saccadeClaudeCode
          ;
        devHerdrPin = host.herdrDev;
        src = cargoSrc;
        cargoLockFile = ./Cargo.lock;
      };
      # the pinned archive: main's 0.18.0 tip, the release the live
      # tracker ran; the rev is the pin, the use is read-only.
      sac-archive = import ./nix/sac-archive.nix {
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
        src = cargoSrc;
        cargoRoot = ".";
        buildAndTestSubdir = "grammar";
        cargoLock.lockFile = ./Cargo.lock;
        meta.mainProgram = "board-grammar";
      };
      default = pkgs.symlinkJoin {
        name = "saccade";
        paths = [
          sac
          saccadePi
        ]
        ++ lib.optional (saccadeClaudeCode != null) saccadeClaudeCode;
        meta.mainProgram = "sac";
      };
      projectPackages = {
        inherit
          sac
          sac-archive
          board-grammar
          default
          ;
        saccade-pi = saccadePi;
      }
      // lib.optionalAttrs (saccadeClaudeCode != null) {
        # the composed Claude Code harness: the pinned binary, the
        # SessionStart identity hook, the injected settings
        saccade-claude-code = saccadeClaudeCode;
      };
    in
    {
      packages = projectPackages // {
        # the project's aggregate: every saccade package in one build
        saccade = pkgs.releaseTools.aggregate {
          name = "saccade-project";
          constituents = lib.attrValues projectPackages;
        };
      };

      devShells.saccade = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          (
            { config, ... }:
            let
              # the dev wrapper: wrapped-pi pointed at this checkout's
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
              devHerdr = host.herdrDev;
              # the composed Claude Code harness the devshell names and
              # carries: the same package the release bundles, so the
              # development override and the compile pin are one line
              devCC = saccadeClaudeCodeFor system pkgs;
            in
            {
              # the skill deploys at the repository's top level, where
              # sessions start
              enterShell = ''
                top=$(${pkgs.git}/bin/git rev-parse --show-toplevel)
                if [ ! -f "$top/.agents/skills/saccade/SKILL.md" ]; then
                  (cd "$top" && cargo run --quiet --manifest-path saccade/Cargo.toml -p saccade --bin sac -- skill install)
                fi
              '';

              packages =
                with pkgs;
                [
                  host.rustToolchain
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
                # the served repository is the checkout this shell sits
                # in, read at start: a moved or renamed checkout needs no
                # edit here
                exec = ''
                  ${
                    saccade.packages.${system}.default
                  }/bin/sac serve --repo "$(${pkgs.git}/bin/git -C ${lib.escapeShellArg config.devenv.root} rev-parse --show-toplevel)" --bind 0.0.0.0 --port 8811
                '';
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
      # the name the repository's shell carried before the projects
      # split: the saccade shell
      devShells.default = config.devShells.saccade;
    };
}
