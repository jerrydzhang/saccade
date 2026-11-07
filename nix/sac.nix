{pkgs, lib, saccadePi, devHerdrPin, saccadeClaudeCode, root}:
pkgs.rustPlatform.buildRustPackage ({
          name = "saccade";
          src = root;
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
          cargoLock.lockFile = root + "/Cargo.lock";
          meta.mainProgram = "sac";
        }
        // lib.optionalAttrs (devHerdrPin != null) {
          # the pinned dev host this binary resolves last; SACCADE_HERDR
          # remains the explicit override and the ambient binary the
          # production arm
          SACCADE_HERDR_PIN = "${devHerdrPin}/bin/herdr-core";
        }
        // lib.optionalAttrs (saccadeClaudeCode != null) {
          # the composed harness this binary's start door launches;
          # the devshell's SACCADE_CLAUDE_CODE remains the development
          # override
          SACCADE_CLAUDE_CODE = "${saccadeClaudeCode}/bin/saccade-claude-code";
        })
