import { mount } from "svelte";
import App from "./App.svelte";
import "./styles/tokens.css";
import "./styles/app.css";
// the theme overlay: the ONE design-language injection point (r93/r94
// - swappable by construction; remove this import to restore the
// placement-only base exactly)
import "./styles/theme.css";

const target = document.getElementById("app");
if (!target) {
  throw new Error("the port expects an #app mount point");
}

mount(App, { target });
