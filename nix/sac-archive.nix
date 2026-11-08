# The pinned archive: the sac release that loads the pre-board log, for
# reading the old database only. The revision is main's 0.18.0 tip, the
# release the live tracker ran before the board cutover; the derivation
# builds it from that exact tree, workbench frontend included, and the
# use contract is read-only: `sac clone` against the old database.
# Nothing in this flake serves the old format.
{
  pkgs,
  lib,
  archiveRev,
  archiveSrc,
}:
let
  # the workbench frontend the old binary embeds through
  # SACCADE_WEB_ASSETS; the hash is the old flake's own pin, unchanged
  # since 0.16.2
  webAssets = pkgs.buildNpmPackage {
    pname = "saccade-archive-web";
    version = "0.18.0";
    src = pkgs.lib.sourceByRegex (archiveSrc + "/web") [
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
in
  pkgs.rustPlatform.buildRustPackage {
    pname = "sac-archive";
    version = "0.18.0";
    src = archiveSrc;
    nativeBuildInputs = [
      pkgs.git
      pkgs.direnv
      pkgs.util-linux
    ];
    SACCADE_DIRENV = "${pkgs.direnv}/bin/direnv";
    SACCADE_WEB_ASSETS = webAssets;
    # the archive's use is reading the old log: clone, list, show. Its
    # own suite is not the pin - the environment it assumed has moved -
    # so the build runs no tests and the verification is the version
    # and the clone dry run.
    doCheck = false;
    cargoLock.lockFile = "${archiveSrc}/Cargo.lock";
    meta.mainProgram = "sac";
  }
