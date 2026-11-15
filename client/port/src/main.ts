import { mount } from "svelte";
import App from "./App.svelte";
import "./fonts.ts";
import "./styles/tokens.css";
import "./styles/app.css";
// the language: the ONE place token values are chosen (light and dark)
import "./styles/theme.css";
import { followThemePreference } from "./lib/theme-mode.ts";

followThemePreference(document.documentElement, window.matchMedia("(prefers-color-scheme: dark)"));

const target = document.getElementById("app");
if (!target) {
  throw new Error("the port expects an #app mount point");
}

mount(App, { target });
