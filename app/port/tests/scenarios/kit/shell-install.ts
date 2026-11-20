// Installs the shell mock in this realm at module evaluation, from the
// world the test page holds (kit/shell-entry.ts imports it first).
import { installShell, SHELL_WORLD_KEY, type ShellWorld } from "./shell-mock.ts";

const world = (window.parent as unknown as Record<string, ShellWorld | undefined>)[SHELL_WORLD_KEY];
if (world === undefined) throw new Error("the bridged app needs the test page's shell world");
installShell(world);
