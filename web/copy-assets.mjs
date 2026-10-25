import { copyFileSync } from "node:fs";
copyFileSync("index.html", "dist/index.html");
copyFileSync("style.css", "dist/style.css");
