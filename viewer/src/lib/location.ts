// The explicit location grammar, one parser for every input surface
// (launch argument, open control, inspection open): "machine:/absolute/
// path" names a machine from the local core's list, "/absolute/path" is
// the machine this viewer runs on, and nothing else is a location. A
// bare relative path is refused, loudly, never guessed.
//
// Machine names may carry colons themselves (user@host:port), so the
// separator is the first colon whose remainder starts with "/".

export type ParsedLocation = { machine: string; path: string } | null;

export function parseLocation(input: string): ParsedLocation {
  const raw = input.trim();
  if (raw.length === 0) return null;
  if (raw.startsWith("/")) return { machine: "host", path: raw };
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === ":" && raw[i + 1] === "/") {
      const machine = raw.slice(0, i);
      const path = raw.slice(i + 1);
      if (machine.length > 0 && !machine.includes("\0")) {
        return { machine, path };
      }
      return null;
    }
  }
  return null;
}
