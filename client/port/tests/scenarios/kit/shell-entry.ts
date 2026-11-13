// The bridged app's entry (kit/shell-app.html): the shell mock is
// installed BEFORE the port's own main.ts boots - module imports
// evaluate in order, depth first (a dynamic import would be rewritten
// by vitest for a runner this nested page does not have).
import "./shell-install.ts";
import "../../../src/main.ts";
