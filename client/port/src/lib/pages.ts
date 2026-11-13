// The /view/v1 page client: the instruments' read surface (t-14
// "Pages", t-24@r2). Every answer is a view document in
// application/json; only the file family answers with bytes. Refusals
// are documents (kind "refusal", a fact naming the code, a text naming
// what would make it answerable) under an informative HTTP status.
// Reading is lenient inside the supported major: a part kind a build
// does not know rides through as raw source beside a diagnostic and
// never refuses the page; the page kind itself is never switched on -
// consumers read the fields they need.

import { SourceUnavailableError, type LaneCode } from "./board/errors.ts";

// One wire request's whole patience: past this, the server is down and
// the answer is a structured refusal.
const REQUEST_BOUND_MS = 10_000;

/** The fetch the page client uses: real fetch by default, injected in
 * tests so the wire shapes are pinned without a server. */
export type Fetcher = (url: string) => Promise<Response>;

async function boundedFetch(fetcher: Fetcher, url: string): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_BOUND_MS);
  try {
    return await fetcher(url);
  } catch (error) {
    throw new SourceUnavailableError(
      "core-unavailable",
      `the tracker server did not answer at ${url}: ${String(error)}`,
    );
  } finally {
    clearTimeout(timer);
  }
}

// -- view grammar v1 ---------------------------------------------------

export type TextPart = { kind: "text"; text: string; location?: string };
export type GroupPart = {
  kind: "group";
  title?: string;
  content: ViewPart[];
  location?: string;
};
export type FactPart = {
  kind: "fact";
  label: string;
  value: string;
  location?: string;
};
export type TimePart = {
  kind: "time";
  at: number;
  label?: string;
  location?: string;
};
export type RefPart = { kind: "ref"; location: string; label: string };
export type FigurePart = {
  kind: "figure";
  caption: string;
  file: string;
  location?: string;
};
/** The pipeline page's rows (the view grammar's seventh kind, landed
 * with the pipeline page, t-24@r2). Cells are plain text; a row
 * carries the location its primary act follows and the moment its age
 * renders from. */
export type TableRowPart = {
  cells: string[];
  location?: string;
  at?: number;
};
export type TablePart = {
  kind: "table";
  columns: string[];
  rows: TableRowPart[];
  location?: string;
};
/** A kind this build does not know: the raw authored fields stay, and
 * the consumer falls back instead of breaking. */
export type OtherPart = {
  kind: "other";
  name: string;
  raw: Record<string, unknown>;
};

export type ViewPart =
  TextPart | GroupPart | FactPart | TimePart | RefPart | FigurePart | TablePart | OtherPart;

export type HistoryEntryRef = { kind: "ref"; location: string; label: string };

export type HistoryEntry = {
  seq: number;
  event: string;
  actor: string;
  tier: string;
  at: number;
  revision: number | null;
  commit: string | null;
  message: string | null;
  grounds: string | null;
  explanation: string | null;
  session: HistoryEntryRef | null;
  refs: HistoryEntryRef[];
};

export type GrammarRange = { min: number; max: number };

/** One page document. The kind names the family ("root", "document",
 * "timeline", "diff", "component", "refusal", ...); a later minor may
 * add kinds and fields, so nothing here refuses on either. */
export type ViewPage = {
  view: { major: number };
  kind: string;
  location: string;
  seq: number | null;
  tracker?: string;
  /** The tracker host's trimmed /etc/machine-id (t-24@r14): the fact
   * a bare-path read compares with this device's own machine id.
   * Absent means "cannot prove same machine". */
  machineId?: string;
  /** The tracker's repository path (t-24@r16), beside the machine id. */
  repoPath?: string;
  viewMajors?: number[];
  boardGrammar?: GrammarRange;
  code?: string;
  task?: string;
  revision?: number;
  content?: ViewPart[];
  entries?: HistoryEntry[];
};

/** A refusal document, raised: the code names the refusal, the detail
 * is the text naming what would make it answerable. */
export class PageRefusal extends SourceUnavailableError {
  readonly status: number;
  constructor(code: string, detail: string, status: number) {
    super(refusalLaneCode(status), `${code}: ${detail}`);
    this.status = status;
  }
}

function refusalLaneCode(status: number): LaneCode {
  if (status === 400) return "core-protocol";
  if (status === 404) return "asset-missing";
  if (status === 503) return "lane-unavailable";
  return "core-protocol";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function optionalString(raw: Record<string, unknown>, key: string): string | undefined {
  const value = raw[key];
  return typeof value === "string" ? value : undefined;
}

function decodePart(raw: Record<string, unknown>): ViewPart {
  switch (raw.kind) {
    case "text":
      return {
        kind: "text",
        text: typeof raw.text === "string" ? raw.text : "",
        location: optionalString(raw, "location"),
      };
    case "group": {
      const children = Array.isArray(raw.content) ? raw.content : [];
      return {
        kind: "group",
        title: optionalString(raw, "title"),
        content: children.map((child) => {
          const record = asRecord(child);
          return record === null
            ? ({
                kind: "other",
                name: "malformed",
                raw: {},
              } satisfies OtherPart)
            : decodePart(record);
        }),
        location: optionalString(raw, "location"),
      };
    }
    case "fact":
      return {
        kind: "fact",
        label: typeof raw.label === "string" ? raw.label : "",
        value: typeof raw.value === "string" ? raw.value : "",
        location: optionalString(raw, "location"),
      };
    case "time":
      return {
        kind: "time",
        at: typeof raw.at === "number" ? raw.at : 0,
        label: optionalString(raw, "label"),
        location: optionalString(raw, "location"),
      };
    case "ref":
      return {
        kind: "ref",
        location: typeof raw.location === "string" ? raw.location : "",
        label: typeof raw.label === "string" ? raw.label : "",
      };
    case "figure":
      return {
        kind: "figure",
        caption: typeof raw.caption === "string" ? raw.caption : "",
        file: typeof raw.file === "string" ? raw.file : "",
        location: optionalString(raw, "location"),
      };
    case "table": {
      const columns = Array.isArray(raw.columns)
        ? raw.columns.filter((c): c is string => typeof c === "string")
        : [];
      const rows = Array.isArray(raw.rows) ? raw.rows : [];
      return {
        kind: "table",
        columns,
        rows: rows.map((row) => {
          const record = asRecord(row);
          if (record === null) return { cells: [], location: undefined, at: undefined };
          const cells = Array.isArray(record.cells)
            ? record.cells.filter((c): c is string => typeof c === "string")
            : [];
          return {
            cells,
            location: optionalString(record, "location"),
            at: typeof record.at === "number" ? record.at : undefined,
          };
        }),
        location: optionalString(raw, "location"),
      };
    }
    default:
      return { kind: "other", name: String(raw.kind ?? "unknown"), raw };
  }
}

function decodeRef(raw: Record<string, unknown>): HistoryEntryRef {
  return {
    kind: "ref",
    location: typeof raw.location === "string" ? raw.location : "",
    label: typeof raw.label === "string" ? raw.label : "",
  };
}

function decodeHistoryEntry(raw: Record<string, unknown>): HistoryEntry {
  const session = asRecord(raw.session);
  const refs = Array.isArray(raw.refs) ? raw.refs : [];
  return {
    seq: typeof raw.seq === "number" ? raw.seq : 0,
    event: typeof raw.event === "string" ? raw.event : "",
    actor: typeof raw.actor === "string" ? raw.actor : "",
    tier: typeof raw.tier === "string" ? raw.tier : "",
    at: typeof raw.at === "number" ? raw.at : 0,
    revision: typeof raw.revision === "number" ? raw.revision : null,
    commit: typeof raw.commit === "string" ? raw.commit : null,
    message: typeof raw.message === "string" ? raw.message : null,
    grounds: typeof raw.grounds === "string" ? raw.grounds : null,
    explanation: typeof raw.explanation === "string" ? raw.explanation : null,
    session: session !== null && typeof session.location === "string" ? decodeRef(session) : null,
    refs: refs.map((ref) => {
      const record = asRecord(ref);
      return record === null
        ? { kind: "ref" as const, location: "", label: "" }
        : decodeRef(record);
    }),
  };
}

export function decodePage(value: unknown): ViewPage {
  const raw = asRecord(value);
  if (raw === null) {
    throw new SourceUnavailableError("core-protocol", "the view answer carries no document");
  }
  const view = asRecord(raw.view);
  const content = Array.isArray(raw.content) ? raw.content : undefined;
  const entries = Array.isArray(raw.entries) ? raw.entries : undefined;
  const grammar = asRecord(raw.board_grammar);
  const majors = Array.isArray(raw.view_majors) ? raw.view_majors : undefined;
  return {
    view: {
      major: view !== null && typeof view.major === "number" ? view.major : 0,
    },
    kind: typeof raw.kind === "string" ? raw.kind : "",
    location: typeof raw.location === "string" ? raw.location : "",
    seq: typeof raw.seq === "number" ? raw.seq : null,
    tracker: optionalString(raw, "tracker"),
    machineId: optionalString(raw, "machine_id"),
    repoPath: optionalString(raw, "repo_path"),
    viewMajors:
      majors === undefined
        ? undefined
        : majors.filter((major): major is number => typeof major === "number"),
    boardGrammar:
      grammar !== null && typeof grammar.min === "number" && typeof grammar.max === "number"
        ? { min: grammar.min, max: grammar.max }
        : undefined,
    code: optionalString(raw, "code"),
    task: optionalString(raw, "task"),
    revision: typeof raw.revision === "number" ? raw.revision : undefined,
    content:
      content === undefined
        ? undefined
        : content.map((part) => {
            const record = asRecord(part);
            return record === null
              ? ({
                  kind: "other",
                  name: "malformed",
                  raw: {},
                } satisfies OtherPart)
              : decodePart(record);
          }),
    entries:
      entries === undefined
        ? undefined
        : entries.map((entry) => {
            const record = asRecord(entry);
            return record === null
              ? {
                  seq: 0,
                  event: "",
                  actor: "",
                  tier: "",
                  at: 0,
                  revision: null,
                  commit: null,
                  message: null,
                  grounds: null,
                  explanation: null,
                  session: null,
                  refs: [],
                }
              : decodeHistoryEntry(record);
          }),
  };
}

/** One page document off the wire: a refusal document raises, any
 * other kind answers decoded. */
export async function fetchPage(fetcher: Fetcher, url: string): Promise<ViewPage> {
  const response = await boundedFetch(fetcher, url);
  let value: unknown;
  try {
    value = await response.json();
  } catch (error) {
    throw new SourceUnavailableError(
      "core-protocol",
      `the tracker server answered ${url} with non-JSON: ${String(error)}`,
    );
  }
  const page = decodePage(value);
  if (page.kind === "refusal") {
    const detail = (page.content ?? [])
      .filter((part): part is TextPart => part.kind === "text")
      .map((part) => part.text)
      .join(" ");
    throw new PageRefusal(page.code ?? "refused", detail, response.status);
  }
  return page;
}

/** Raw bytes off the file family: no JSON, no document, just the file
 * with the type its extension names. */
export async function fetchFileBytes(fetcher: Fetcher, url: string): Promise<Uint8Array> {
  const response = await boundedFetch(fetcher, url);
  if (!response.ok) {
    // the file family refuses as a document too
    let code = `status ${response.status}`;
    let detail = "";
    try {
      const page = decodePage(await response.json());
      if (page.kind === "refusal") {
        code = page.code ?? code;
        detail = (page.content ?? [])
          .filter((part): part is TextPart => part.kind === "text")
          .map((part) => part.text)
          .join(" ");
      }
    } catch {
      // a non-JSON body still refuses with the status
    }
    throw new PageRefusal(code, detail || `the file route refused ${url}`, response.status);
  }
  try {
    return new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    throw new SourceUnavailableError(
      "core-protocol",
      `the file read failed at ${url}: ${String(error)}`,
    );
  }
}

/** The facts a board page carries about its task: what the picker and
 * the header show without decoding the board. */
export function pageFact(page: ViewPage, label: string): string | null {
  for (const part of page.content ?? []) {
    if (part.kind === "fact" && part.label === label) return part.value;
  }
  return null;
}
