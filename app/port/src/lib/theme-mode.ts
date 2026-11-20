// The window's theme mode: light or dark, from the system's preference
// (prefers-color-scheme, which WebKitGTK reads from the GTK settings).
// The language assigns light on :root and dark on :root[data-theme=
// "dark"] (styles/theme.css); this module only keeps the attribute in
// step with the preference. A setting of its own can come later.

export type ThemeMode = "light" | "dark";

/** The minimal face of a MediaQueryList this module reads. */
export type PreferenceQuery = {
  readonly matches: boolean;
  addEventListener(type: "change", listener: () => void): void;
  removeEventListener(type: "change", listener: () => void): void;
};

export function applyThemeMode(root: { dataset: DOMStringMap }, mode: ThemeMode): void {
  root.dataset.theme = mode;
}

/** Follow the preference: set the mode now and on every change; the
 * returned function stops following. */
export function followThemePreference(root: { dataset: DOMStringMap }, query: PreferenceQuery): () => void {
  const apply = (): void => applyThemeMode(root, query.matches ? "dark" : "light");
  apply();
  query.addEventListener("change", apply);
  return () => query.removeEventListener("change", apply);
}
