// The bundle's one honest escaping pass: esbuild --charset=ascii still
// emits literal control bytes inside RegExp character classes, and the
// repository holds authored text to ASCII. Each offending byte becomes
// its \xHH escape - identical meaning in every legal position (string,
// template, regex) a minified bundle can hold.
import { readFileSync, writeFileSync } from "node:fs";

const path = process.argv[2];
const bytes = readFileSync(path);
let out = "";
for (const byte of bytes) {
  if ((byte >= 0x20 && byte <= 0x7e) || byte === 0x09 || byte === 0x0a) {
    out += String.fromCharCode(byte);
  } else {
    out += "\\x" + byte.toString(16).padStart(2, "0");
  }
}
writeFileSync(path, out);
