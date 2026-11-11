import { mount } from "svelte";
import App from "./App.svelte";
import "./styles/tokens.css";
import "./styles/app.css";

const target = document.getElementById("app");
if (!target) {
  throw new Error("the port expects an #app mount point");
}

mount(App, { target });
