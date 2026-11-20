// What a figure shows, decided from its file (D3): the prototype shows
// a figure of any type (inst16.html:211-214) and enlarges what is
// shown (:392-400). Pure: the hydration renders by this answer and the
// inspection enlarges the same thing.

const IMAGE_TYPES: Readonly<Record<string, string>> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  svg: "image/svg+xml",
  bmp: "image/bmp",
  avif: "image/avif",
};

/** What a figure shows, by its file: an image type by extension; any
 * other file that decodes as UTF-8 without NULs is text; anything
 * else is named as binary - never a broken image. */
export function figureShape(
  input: string,
  bytes: Uint8Array,
): { kind: "image"; mime: string } | { kind: "text"; text: string } | { kind: "binary" } {
  const extension = /\.([^./]+)$/.exec(input)?.[1]?.toLowerCase() ?? "";
  const mime = IMAGE_TYPES[extension];
  if (mime !== undefined) return { kind: "image", mime };
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (!text.includes("\u0000")) return { kind: "text", text };
  } catch {
    // not UTF-8: binary
  }
  return { kind: "binary" };
}
