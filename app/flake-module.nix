# The app project's flake outputs: the Saccade app (the native binary,
# the frontend dist, the tauri-driver the native test project boots),
# the project's aggregate, its dev shell, the runnable apps and the
# consumer overlay. The builds take their source from this directory
# plus saccade/grammar, the one crate the app depends on.
{ inputs, ... }:
let
  inherit (inputs) nixpkgs-client;
  # the Saccade app's packaging lives in app/nix/saccade-app.nix: one
  # callPackage-able file taking every dependency from the pkgs it
  # is called on, so overlays.default can build it against a
  # consumer's own nixpkgs - the packaged binary then shares the
  # host's GL stack instead of this flake's pin. The repo root rides
  # in explicitly: the app's cargo tree spans app/port-shell,
  # app/src-tauri and the shared grammar crate under saccade/, and a
  # relative climb would stay an unnormalized ".." path inside flake
  # evaluation.
  repoRoot = ./..;
  saccadeApp = pkgs: pkgs.callPackage ./nix/saccade-app.nix { inherit repoRoot; };
in
{
  # the consumer-facing overlay: a NixOS configuration sets
  # nixpkgs.overlays = [ inputs.saccade.overlays.default ] and gets
  # pkgs.saccade-app built against its own nixpkgs
  flake.overlays.default = final: prev: {
    saccade-app = (saccadeApp final).app;
  };

  perSystem =
    {
      system,
      config,
      pkgs,
      lib,
      host,
      ...
    }:
    let
      clientPkgs = nixpkgs-client.legacyPackages.${system};
      projectPackages = {
        # the app the client runs: built against nixpkgs-client, the
        # revision the human's system flake is built from
        saccade-app = (saccadeApp clientPkgs).app;
        saccade-app-web = (saccadeApp pkgs).web;
        saccade-app-tauri-driver = (saccadeApp pkgs).driver;
      };
      saccadeAppBin = projectPackages.saccade-app;
    in
    {
      packages = projectPackages // {
        # the project's aggregate: every app package in one build
        app = pkgs.releaseTools.aggregate {
          name = "saccade-app-project";
          constituents = lib.attrValues projectPackages;
        };
      };

      apps = {
        saccade-app = {
          type = "app";
          program = "${saccadeAppBin}/bin/saccade-app";
        };
        default = {
          type = "app";
          program = "${saccadeAppBin}/bin/saccade-app";
        };
      };

      # the Saccade app's native dev shell: its own dependency graph,
      # separate from the saccade devshell and the tracker crate. The
      # WebKitGTK/GTK build inputs here serve the viewer's Tauri crate
      # only; nothing here touches the tracker's build.
      # the browser lane's fontconfig: a generated config over pinned
      # DejaVu fonts, both pure store paths - the sandbox already binds
      # /nix/store read-only, so no host /etc/fonts or font dirs are
      # ever mounted
      devShells.app =
        let
          laneFontsConf = pkgs.writeText "saccade-lane-fonts.conf" ''
            <?xml version="1.0"?>
            <!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
            <fontconfig>
              <dir>${pkgs.dejavu_fonts}</dir>
              <include>${pkgs.fontconfig.out}/etc/fonts/conf.d</include>
              <cachedir prefix="xdg">fontconfig</cachedir>
            </fontconfig>
          '';
        in
        pkgs.mkShell {
          # the pinned Playwright pair (python package and browser
          # package) from this same nixpkgs revision: no browser
          # install or download anywhere
          packages = [
            host.rustToolchain
            pkgs.nodejs
            pkgs.pkg-config
            (pkgs.python3.withPackages (ps: [
              ps.playwright
              ps.pyflakes
            ]))
            pkgs.bubblewrap
            pkgs.weston
            pkgs.mesa
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            (saccadeApp pkgs).driver
            # the Tauri/GTK block rides nixpkgs-client so the shell's
            # WebKitWebDriver matches the packaged app's WebKitGTK
            # (the two pins carry different webkitgtk releases);
            # the lane tools above stay on nixpkgs
            clientPkgs.webkitgtk_4_1
            clientPkgs.gtk3
            clientPkgs.cairo
            clientPkgs.gdk-pixbuf
            clientPkgs.glib
            clientPkgs.dbus
            pkgs.dejavu_fonts
          ];
          env.PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
          # the sandbox has no /etc/fonts, so the makeFontsConf include
          # of conf.d resolves to nothing and generic families match no
          # font; this conf carries the pinned fontconfig's own conf.d
          # (45/60-generic+latin assign sans-serif to DejaVu Sans)
          env.FONTCONFIG_FILE = laneFontsConf;
          # the native lane's software-GL pin: glvnd vendor dir whose
          # 50_mesa.json carries the absolute store path of
          # libEGL_mesa.so.0 (llvmpipe/swrast live in the same output)
          env.SAC_APP_MESA_EGL_VENDOR_DIR = "${pkgs.mesa}/share/glvnd/egl_vendor.d";
          # the remote lane's pinned core: the flake's 0.1.17 binary,
          # sha-checked at build; the lane verifies --version again
          env.SAC_APP_HERDR_CORE_BIN = "${host.herdrCore}/bin/herdr-core";
        };
      # the name the app's shell carried before the projects split
      devShells.saccade-app = config.devShells.app;
    };
}
