import { mount } from "svelte";
import App from "./App.svelte";
import { startShellProbe } from "./lib/shellprobe.ts";
// The ledger's faces ship with the app as local files (t-34@r6 change
// A): IBM Plex Mono for the tools, IBM Plex Sans for boards, in the
// look's weights (regular 400, medium 500, semibold 600; lookVars.ts).
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/400-italic.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans/600-italic.css";
import "./styles/tokens.css";
import "./styles/board.css";

const target = document.getElementById("app");
if (!target) {
  throw new Error("the viewer expects an #app mount point");
}

startShellProbe();
mount(App, { target });
