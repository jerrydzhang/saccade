// The strip's presentation (t-59, "Strip: app, location, link; load
// status only when loading or failed"): which of the strip view's
// statuses (App.ts stripView) the strip shows, and how. Pure, so the
// rule is tested apart from the window.

/** The statuses the strip keeps quiet about: a loaded view (the
 * location says it) and an empty window (nothing to say). */
const QUIET = new Set(["loaded", "empty"]);

/** What the strip says about the load: null when the view is simply
 * loaded or empty; otherwise the status, marked failed unless it is
 * still loading. */
export function stripStatus(status: string): { text: string; failed: boolean } | null {
  if (QUIET.has(status)) return null;
  return { text: status, failed: status !== "loading" };
}
