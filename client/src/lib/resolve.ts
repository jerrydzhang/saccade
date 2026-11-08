// Declaration-relative asset path resolution. Pure string semantics so the
// same rules run in the browser, node tests, and the Tauri host adapter
// (defense in depth: the host adapter re-checks before every read).
//
// Unicode paths are data and stay data: these rules never transliterate.

const WINDOWS_DRIVE = /^[A-Za-z]:/;

export type ResolveFailure = { code: "asset-denied"; reason: string };

export type ResolveResult =
  | { ok: true; joined: string }
  | { ok: false; failure: ResolveFailure };

export function resolveRelative(baseDir: string, relative: string): ResolveResult {
  if (relative.length === 0) {
    return { ok: false, failure: { code: "asset-denied", reason: "empty path" } };
  }
  if (relative.includes("\0")) {
    return { ok: false, failure: { code: "asset-denied", reason: "NUL byte in path" } };
  }
  if (relative.includes("\\")) {
    return {
      ok: false,
      failure: { code: "asset-denied", reason: "backslash is not a separator" },
    };
  }
  if (relative.startsWith("/") || WINDOWS_DRIVE.test(relative)) {
    return { ok: false, failure: { code: "asset-denied", reason: "absolute path" } };
  }
  const segments = relative.split("/");
  const out: string[] = [];
  for (const segment of segments) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") {
      return {
        ok: false,
        failure: { code: "asset-denied", reason: "parent traversal" },
      };
    }
    out.push(segment);
  }
  if (out.length === 0) {
    return { ok: false, failure: { code: "asset-denied", reason: "empty path" } };
  }
  const base = baseDir.endsWith("/") ? baseDir.slice(0, -1) : baseDir;
  // an empty declaration base is repo-relative (tracker boards): the
  // join must not grow a leading slash the tree listing never carries
  const prefix = base === "" ? "" : `${base}/`;
  return { ok: true, joined: `${prefix}${out.join("/")}` };
}
