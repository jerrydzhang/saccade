// The task list's narrowing: the one place that spells it. A tasks
// location may carry "?tag=auth,ui" (every tag named) or "?untagged"
// (tasks carrying no tag); the frame holds that string opaque and asks
// this module to validate it and to give its canonical form. The
// canonical form is lowercase, deduplicated and sorted, so two
// spellings of one narrowing are one location. A different spelling
// changes this file and its tests and nothing else.

export type Narrowing = Readonly<{
  /** Lowercase, sorted, no repeats; empty when untagged. */
  tags: readonly string[];
  untagged: boolean;
}>;

export type NarrowingVerdict =
  | Readonly<{ ok: true; narrowing: Narrowing; canonical: string }>
  | Readonly<{ ok: false; refusal: string }>;

// a tag name as the tracker admits one
const TAG_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

/** Parse the opaque string after the question mark. */
export function parseNarrowing(opaque: string): NarrowingVerdict {
  if (opaque === "untagged") {
    return { ok: true, narrowing: { tags: [], untagged: true }, canonical: "untagged" };
  }
  if (opaque.startsWith("untagged")) {
    return { ok: false, refusal: "untagged stands alone; it takes no tag names" };
  }
  if (!opaque.startsWith("tag=")) {
    return {
      ok: false,
      refusal: `'${opaque}' is not a narrowing; the task list narrows by tag=<name>[,<name>...] or untagged`,
    };
  }
  const names = opaque.slice("tag=".length).split(",");
  if (names.some((name) => name.includes("=") || name.includes("&") || name.includes("?"))) {
    return { ok: false, refusal: "tag= names tags separated by commas, and nothing else" };
  }
  if (names.length === 1 && names[0] === "") {
    return { ok: false, refusal: "tag= names no tag; name at least one, or leave the narrowing off" };
  }
  for (const name of names) {
    if (!TAG_NAME.test(name)) {
      return { ok: false, refusal: `'${name}' is not a tag name; names are letters, digits, '.', '_' and '-'` };
    }
  }
  const tags = [...new Set(names.map((name) => name.toLowerCase()))].sort();
  return { ok: true, narrowing: { tags, untagged: false }, canonical: `tag=${tags.join(",")}` };
}

/** The tasks path a narrowing spells: the canonical location tail. */
export function narrowedPath(narrowing: Narrowing): string {
  return narrowing.untagged ? "tasks?untagged" : `tasks?tag=${narrowing.tags.join(",")}`;
}

/** The query the tracker's tasks page takes for a narrowing. */
export function narrowingQuery(narrowing: Narrowing): string {
  return narrowing.untagged ? "untagged" : narrowing.tags.map((tag) => `tag=${tag}`).join("&");
}

/** Whether the page's echo is exactly the narrowing asked for: the
 * tracker's recorded names compare as the canonical form does. An
 * absent echo is a tracker that did not narrow. */
export function echoMatches(
  asked: Narrowing,
  echo: { tags: readonly string[]; untagged: boolean } | undefined,
): boolean {
  if (echo === undefined) return false;
  if (echo.untagged !== asked.untagged) return false;
  const echoed = [...new Set(echo.tags.map((tag) => tag.toLowerCase()))].sort();
  return echoed.length === asked.tags.length && echoed.every((tag, i) => tag === asked.tags[i]);
}

/** What a tasks path asks: no narrowing (the whole list), a
 * narrowing with the path in its canonical spelling, or a refusal. */
export function narrowingOfPath(
  path: string,
):
  | Readonly<{ ok: true; narrowing: Narrowing | null; path: string }>
  | Readonly<{ ok: false; refusal: string }> {
  const at = path.indexOf("?");
  if (at < 0) return { ok: true, narrowing: null, path };
  const verdict = parseNarrowing(path.slice(at + 1));
  if (!verdict.ok) return verdict;
  return { ok: true, narrowing: verdict.narrowing, path: narrowedPath(verdict.narrowing) };
}

export const NOT_NARROWED_BY_THIS_TRACKER =
  "this tracker does not narrow by tag; move the binaries in step (sac 0.22.2 or later)";
