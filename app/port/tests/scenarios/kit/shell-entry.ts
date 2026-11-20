// The bridged app's entry (kit/shell-app.html): the shell mock is
// installed BEFORE the port's own main.ts boots - module imports
// evaluate in order, depth first (a dynamic import would be rewritten
// by vitest for a runner this nested page does not have). The lane's
// asks install lands first of all (this lane has no native shell
// either; the mock answers the shell's own commands).
import "./lane-asks.ts";
import "./shell-install.ts";
import "../../../src/main.ts";
