// The specimen page (dev only: the build's input is index.html alone):
// every block of the language, rendered by the app's own components
// from fixtures, so an accepted look is already the shipped code.
// ?theme=light|dark pins the mode; otherwise it follows the system.
// ?section=<name> shows one section alone.
import { mount } from "svelte";
import Specimen from "./Specimen.svelte";
import "../fonts.ts";
import "../styles/tokens.css";
import "../styles/app.css";
import "../styles/theme.css";
import { applyThemeMode, followThemePreference } from "../lib/theme-mode.ts";

const params = new URLSearchParams(window.location.search);
const pinned = params.get("theme");
if (pinned === "light" || pinned === "dark") {
  applyThemeMode(document.documentElement, pinned);
} else {
  followThemePreference(document.documentElement, window.matchMedia("(prefers-color-scheme: dark)"));
}

const target = document.getElementById("specimen");
if (!target) throw new Error("the specimen expects a #specimen mount point");
mount(Specimen, { target, props: { only: params.get("section") } });
