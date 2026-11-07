{pkgs, herdrApp, herdrCore, mkHerdrAppSaccade, script}: let
  python = pkgs.python3.withPackages (ps: [ps.python-xlib]);
  app = mkHerdrAppSaccade {inherit pkgs herdrApp;};
in
  pkgs.writeShellApplication {
    name = "herdr-ui";
    runtimeInputs = [pkgs.weston pkgs.xwayland];
    text = ''
      exec ${python}/bin/python -B ${script} \
        --app ${app}/bin/herdr-app-saccade \
        --core ${herdrCore}/bin/herdr-core \
        --weston ${pkgs.weston}/bin/weston \
        --xwayland ${pkgs.xwayland}/bin/Xwayland "$@"
    '';
  }
