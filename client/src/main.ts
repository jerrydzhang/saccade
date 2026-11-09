import { mount } from "svelte";
import App from "./App.svelte";
import { startShellProbe } from "./lib/shellprobe.ts";
import "./styles/tokens.css";
import "./styles/board.css";

const target = document.getElementById("app");
if (!target) {
  throw new Error("the viewer expects an #app mount point");
}

startShellProbe();
mount(App, { target });
