# snapshots come from the pinned saccade input, never the live working tree
{
  lib,
  pkgs,
  saccade,
}: let
  # explicit list: extensions/ also carries tests and docs
  piExtensionFiles = [
    "websearch.ts"
    "bash-background.ts"
  ];
  piExtensionInstallLines =
    lib.concatMapStrings (f: ''
      install -m 0444 "${saccade}/extensions/${f}" "$out/${f}"
    '')
    piExtensionFiles;

  pi-extensions-snapshot = pkgs.runCommand "saccade-pi-extensions" {} ''
    mkdir -p $out
    ${piExtensionInstallLines}
  '';

  install-pi-extensions = pkgs.writeShellApplication {
    name = "install-pi-extensions";
    runtimeInputs = with pkgs; [coreutils jq gnugrep];
    text = let
      nixBin = "${pkgs.nix}/bin/nix";
      snapshot = "${pi-extensions-snapshot}";
      filesJson = builtins.toJSON piExtensionFiles;
      currentFiles = lib.concatStringsSep "\n" piExtensionFiles;
    in ''
      : "''${HOME:?install-pi-extensions: HOME is not set}"
      target="$HOME/.pi/agent/extensions"
      root="$target/.saccade-pi-extensions-snapshot"
      manifest="$target/.saccade-pi-extensions.json"
      files=(${lib.concatStringsSep " " piExtensionFiles})

      die() {
        echo "install-pi-extensions: $*" >&2
        exit 1
      }

      if [[ -e $target && ! -d $target ]]; then
        die "$target exists and is not a directory"
      fi
      mkdir -p "$target"

      # the out-link is created only by this installer; anything else at
      # that name is not ours to replace
      if [[ -e $root || -L $root ]]; then
        if [[ ! -L $root ]]; then
          die "$root exists and is not a symlink"
        fi
        if [[ ! -f $manifest ]]; then
          die "$root exists but $manifest does not; not an installation of this installer"
        fi
        if [[ "$(readlink "$root")" != "$(jq -r '.snapshot' "$manifest")" ]]; then
          die "$root does not resolve to the recorded snapshot; refusing to touch it"
        fi
      fi

      previous=""
      if [[ -f $manifest ]]; then
        previous="$(jq -r '.files[]' "$manifest")" || die "$manifest is not installer state; remove it and its files by hand"
      fi
      while IFS= read -r f; do
        [[ -z $f ]] && continue
        if ! grep -qxE '[A-Za-z0-9._-]+[.]ts' <<<"$f"; then
          die "$manifest names '$f', which this installer does not install"
        fi
      done <<<"$previous"

      # the manifest names what a previous run installed, never permission to
      # replace bytes now on disk: every existing managed or stale entry must
      # be this installer's own symlink to the recorded root
      while IFS= read -r f; do
        [[ -z $f ]] && continue
        if [[ -e $target/$f || -L $target/$f ]]; then
          if [[ ! -L $target/$f || "$(readlink "$target/$f")" != "$root/$f" ]]; then
            die "$target/$f was replaced and is not this installer's symlink; refusing to touch it"
          fi
        fi
      done <<<"$(printf '%s\n' "$previous" "''${files[@]}" | sort -u)"

      # root the new snapshot before anything points through the out-link
      "${nixBin}" --extra-experimental-features 'nix-command flakes' build --out-link "$root" "${snapshot}"

      # a previous generation may have installed files this one does not
      while IFS= read -r f; do
        [[ -z $f ]] && continue
        grep -qxF "$f" <<<"${currentFiles}" || rm -f "$target/$f"
      done <<<"$previous"

      for f in "''${files[@]}"; do
        ln -sfn "$root/$f" "$target/$f"
      done

      tmp="$manifest.tmp"
      jq -n --arg snapshot "${snapshot}" --arg root "$root" --argjson files '${filesJson}' \
        '{snapshot: $snapshot, gc_root: $root, files: $files}' >"$tmp"
      mv "$tmp" "$manifest"

      echo "installed ${lib.concatStringsSep ", " piExtensionFiles} in $target"
      echo "snapshot ${snapshot} rooted at $root (survives nix gc while present)"
      echo "update: bump the saccade input pin, then run this again"
      echo "remove: rm $target/{${lib.concatStringsSep "," piExtensionFiles}} \"$manifest\" \"$root\""
    '';
  };
in {
  inherit pi-extensions-snapshot install-pi-extensions;
}
