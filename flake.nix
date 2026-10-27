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
    # the receipt preload pair the generated wrapper registers
    saccadeCompositionPreloadFiles = ["native/saccade-composition-shim.mjs" "native/saccade-composition-hooks.mjs"];
    mkCompositionLaunchAdapter = {
      pkgs,
      wrappedPiPkg,
      shimSrc,
      hooksSrc,
      extraArgs ? "",
    }: let
      wrappedPiBin = "${wrappedPiPkg}/bin/wrapped-pi";
    in
      pkgs.runCommand "saccade-pi-composition"
        {
          passthru.preloadDir = "preload";
        } ''
        set -o errexit
        # wrapped-pi's exec line names the pinned pi bin: one anchor
        piBin=$(${pkgs.gnugrep}/bin/grep -oP '^exec \K\S+' ${wrappedPiBin})
        test -n "$piBin" && test -f "$piBin"
        # the pinned runtime's own module tree carries the two receipt
        # targets: the readable module and the minified bundle chunk,
        # each asserted to hold its anchor exactly once - a moved or
        # drift shape fails this build visibly
        dist=$(dirname "$(dirname "$piBin")")/lib/pi/node_modules/@earendil-works/pi-coding-agent/dist
        readable=$dist/core/agent-session.js
        test -f "$readable"
        test "$(${pkgs.gnugrep}/bin/grep -cF 'streamingBehavior: options?.deliverAs,' $readable)" -eq 1
        chunk=$(${pkgs.gnugrep}/bin/grep -rlF 'streamingBehavior:options?.deliverAs,images,source:"extension"' $dist/bundle/chunks/ | head -1)
        test -n "$chunk"
        test "$(${pkgs.gnugrep}/bin/grep -cF 'streamingBehavior:options?.deliverAs,images,source:"extension"' $chunk)" -eq 1
        # the baked receipt sources: the exact pinned module URLs the
        # hook guards on, written without shell-quoting games
        mkdir -p $out/preload $out/bin
        printf '["file://%s", "file://%s"]\n' "$readable" "$chunk" > $out/preload/receipt-sources.json
        cp ${shimSrc} $out/preload/saccade-composition-shim.mjs
        cp ${hooksSrc} $out/preload/saccade-composition-hooks.mjs
        # the pinned pi bin execs the cli bundle on exactly one line:
        # insert the receipt preload before the bundle path, so the
        # flag scopes to this Node invocation alone
        ${pkgs.gnused}/bin/sed "s|^exec \"\([^\"]*\)\" |exec \"\1\" --import \"$out/preload\"/saccade-composition-shim.mjs |" "$piBin" > adapted-pi
        grep -q -- '--import' adapted-pi
        test "$(${pkgs.gnugrep}/bin/grep -c -- '--import' adapted-pi)" -eq 1
        # saccade-pi: wrapped-pi verbatim, pointed at the adapted pi
        ${pkgs.gnused}/bin/sed "s|^exec ''${piBin} |exec $out/bin/adapted-pi ${extraArgs} |" ${wrappedPiBin} > saccade-pi
        grep -q "adapted-pi ${extraArgs}" saccade-pi
        chmod +x adapted-pi saccade-pi
        cp adapted-pi $out/bin/adapted-pi
        cp saccade-pi $out/bin/saccade-pi
      '';

    # The composition launch adapter: generated at build time from the
    # pinned wrapped-pi and pi launch scripts, interposing the receipt
    # preload as an explicit --import on the composition's own Node
    # invocation. Argument-scoped: child processes the session spawns
    # run their own Node invocations and never carry the hook. Each
    # generation step asserts its anchor, failing the build loudly when
    # a pin moves the launch shape.
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
              # store as a pair: the adapter interposes it on the
              # composition's own Node invocation as an explicit
              # --import, arg-scoped so child processes never carry it
              compositionPreload = pkgs.runCommand "saccade-composition-preload"
                {
                  shimSrc = ./native/saccade-composition-shim.mjs;
                  hooksSrc = ./native/saccade-composition-hooks.mjs;
                  passthru.preloadDir = "preload";
                }
                ''
                  mkdir -p $out
                  ${lib.concatMapStrings (path: "cp ${./. + "/${path}"} $out/${baseNameOf path}\n") saccadeCompositionPreloadFiles}
                '';
              devPi = lib.hiPrio (mkCompositionLaunchAdapter {
                inherit pkgs;
                wrappedPiPkg = wrapped-pi.packages.${system}.default;
                inherit (compositionPreload) shimSrc hooksSrc;
                extraArgs = lib.escapeShellArgs (lib.concatMap (path: ["-e" "${config.devenv.root}/${path}"]) saccadePiExtensions);
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
                SACCADE_COMPOSITION_PRELOAD = "${devPi}/preload";
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
        compositionPreload = pkgs.runCommand "saccade-composition-preload"
          {
            shimSrc = ./native/saccade-composition-shim.mjs;
            hooksSrc = ./native/saccade-composition-hooks.mjs;
            passthru.preloadDir = "preload";
          }
          ''
            mkdir -p $out
            ${lib.concatMapStrings (path: "cp ${./. + "/${path}"} $out/${baseNameOf path}\n") saccadeCompositionPreloadFiles}
          '';
        # the generated launch adapter is the wrapper's payload; the
        # outer script only carries util-linux in the runtime closure
        launchAdapter = mkCompositionLaunchAdapter {
          inherit pkgs;
          wrappedPiPkg = pi;
          inherit (compositionPreload) shimSrc hooksSrc;
        };
        saccadePi = pkgs.writeShellApplication {
          name = "saccade-pi";
          # util-linux rides the closure: the integration's guard helper
          # is flock, never an assumed ambient binary
          runtimeInputs = [pkgs.util-linux];
          text = ''
            exec ${launchAdapter}/bin/saccade-pi "$@"
          '';
        };
        # the workbench frontend, built once from the committed sources
        # and lock; the binary embeds these three files through the
        # SACCADE_WEB_ASSETS build interface
        webAssets = pkgs.buildNpmPackage {
          pname = "saccade-workbench-web";
          version = "0.16.0";
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
