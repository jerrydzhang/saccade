// The rail's kept views, the frame's half: a view is a kept location
// of a registered repository. Its raw spelling comes from the
// repository's declaration (offered by the file) or from the device's
// own store (kept by Space k); either way the frame reads it through
// the claims seam - the claiming instrument canonicalizes it and
// words its label - and holds only canonical locations. The frame
// never reads what a location means.

import { locationTitleFor } from "../instruments/claims.ts";
import { canonicalAddress, instrumentForAddress } from "../instruments/registry.ts";
import { parseLocation } from "../location.ts";

/** A raw view read: the canonical path inside the repository and the
 * canonical location with the repository's name, or why it cannot be
 * read. */
export type ViewVerdict =
  | Readonly<{ ok: true; path: string; location: string }>
  | Readonly<{ ok: false; refusal: string }>;

/** One kept view as the rail draws it. */
export type RailView = Readonly<{
  /** The rail id of the entry. */
  part: string;
  /** The canonical location ("name/path"): what choosing it opens. */
  location: string;
  /** The canonical path inside the repository, as the device stores it. */
  path: string;
  /** The words drawn on the entry. */
  label: string;
  /** Who keeps it: the declaration file offers it, or this device kept it. */
  source: "file" | "device";
}>;

export const viewPart = (location: string): string => `rail/view/${location}`;

/** Read a view's raw spelling (a path inside the repository, written
 * without its name) into its canonical form. */
export function readView(name: string, raw: string): ViewVerdict {
  const address = parseLocation(`${name}/${raw.trim()}`);
  if (address === null || address.machine !== `tracker:${name}`) {
    return { ok: false, refusal: `'${raw}' is not a location inside ${name}` };
  }
  const canonical = canonicalAddress(address);
  if (!canonical.ok) return canonical;
  if (instrumentForAddress(canonical.address) === null) {
    return { ok: false, refusal: `no instrument shows '${raw}'` };
  }
  return { ok: true, path: canonical.address.path, location: `${name}/${canonical.address.path}` };
}

/** A project's views: the file's first, in the file's order, then the
 * device's in the order kept, each canonical location once (a view the
 * device kept that the file also offers shows as the file's). A file
 * view that cannot be read is a problem line naming it; the others
 * still stand. A device view that no longer reads is dropped from the
 * list, never from the store. */
export function resolveViews(
  name: string,
  fileViews: readonly string[],
  deviceViews: readonly string[],
): { views: RailView[]; problems: string[] } {
  const views: RailView[] = [];
  const problems: string[] = [];
  const add = (verdict: Extract<ViewVerdict, { ok: true }>, source: RailView["source"]): void => {
    if (views.some((view) => view.location === verdict.location)) return;
    const title = locationTitleFor({ machine: `tracker:${name}`, path: verdict.path });
    views.push({
      part: viewPart(verdict.location),
      location: verdict.location,
      path: verdict.path,
      label: title ?? verdict.path,
      source,
    });
  };
  for (const raw of fileViews) {
    const verdict = readView(name, raw);
    if (verdict.ok) add(verdict, "file");
    else problems.push(`workbench.toml view '${raw}' cannot be read: ${verdict.refusal}`);
  }
  for (const raw of deviceViews) {
    const verdict = readView(name, raw);
    if (verdict.ok) add(verdict, "device");
  }
  return { views, problems };
}
