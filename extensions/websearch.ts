/**
 * Native Pi websearch capability: web_search + web_fetch.
 *
 * Providers, fixed order: keyless Exa MCP primary, ZAI MCP fallback (only
 * when ZAI_API_KEY is set). No other providers, no model calls, no caching,
 * no configuration surface. Explicit-failure discipline: provider attempts
 * are reported honestly; cancellation never triggers fallback.
 *
 * Authored for the pinned native Pi extension API (pi.registerTool,
 * TypeBox parameters). The default factory is the only pi-facing code;
 * everything below it is pure logic exercised directly by tests.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Bounds — stated in tool descriptions and echoed in output when truncating.
const DEFAULT_NUM_RESULTS = 5;
const MAX_NUM_RESULTS = 20;
const EXCERPT_CHARS = 240;
const DEFAULT_FETCH_CHARS = 20_000;
const MAX_FETCH_CHARS = 200_000;
const PROVIDER_TIMEOUT_MS = 30_000;
const RAW_TEXT_CAP = 20_000;

const EXA_MCP_URL = "https://mcp.exa.ai/mcp";
const ZAI_SEARCH_URL = "https://api.z.ai/api/mcp/web_search_prime/mcp";
const ZAI_READER_URL = "https://api.z.ai/api/mcp/web_reader/mcp";

export interface SearchParams {
	query: string;
	numResults?: number;
}

export interface FetchParams {
	url: string;
	maxChars?: number;
}

export interface WebSearchDeps {
	fetchImpl?: typeof fetch;
	env?: Record<string, string | undefined>;
	tmpDir?: string;
	timeoutMs?: number;
	signal?: AbortSignal;
}

export interface ToolOutcome {
	text: string;
	details: Record<string, unknown>;
}

interface ParsedResult {
	title: string;
	url: string;
	publishedDate?: string;
	excerpt?: string;
}

class AbortedError extends Error {
	constructor() {
		super("Cancelled");
		this.name = "AbortedError";
	}
}

function throwIfAborted(signal: AbortSignal | undefined): void {
	if (signal?.aborted) throw new AbortedError();
}

/** Compose the caller signal with the per-provider hard timeout. */
function boundedSignal(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
 const timeout = AbortSignal.timeout(timeoutMs);
	return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

/** Keep credential bytes out of any text we surface; never prints. */
function scrub(text: string, secrets: (string | undefined)[]): string {
	let out = text;
	for (const s of secrets) {
		if (s && s.length > 8 && out.includes(s)) out = out.split(s).join("[redacted]");
	}
	return out;
}

function clampInt(value: number | undefined, min: number, max: number, fallback: number): number {
	if (value === undefined || !Number.isFinite(value)) return fallback;
	return Math.min(max, Math.max(min, Math.floor(value)));
}

function truncate(text: string, max: number): { text: string; truncated: boolean } {
	return text.length <= max ? { text, truncated: false } : { text: text.slice(0, max), truncated: true };
}

// ── Exa MCP (stateless, keyless) ────────────────────────────────────────────

async function exaMcpCall(
	tool: string,
	args: Record<string, unknown>,
	deps: WebSearchDeps,
): Promise<Record<string, any>> {
	const fetchImpl = deps.fetchImpl ?? fetch;
	const signal = boundedSignal(deps.signal, deps.timeoutMs ?? PROVIDER_TIMEOUT_MS);
	const url = `${EXA_MCP_URL}?tools=${tool}`;
	const response = await fetchImpl(url, {
		method: "POST",
		headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream" },
		body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: tool, arguments: args } }),
		signal,
	});
	const raw = await response.text();
	if (!response.ok) {
		throw new Error(`HTTP ${response.status}: ${truncate(raw, 300).text}`);
	}
	// The endpoint answers either plain JSON or SSE; both carry one JSON body.
	const data = extractJsonRpcPayload(raw);
	if (!data) throw new Error("unparseable MCP response (no JSON-RPC body)");
	if (data.error) throw new Error(`MCP error: ${JSON.stringify(data.error)}`);
	const result = data.result;
	if (!result || typeof result !== "object") throw new Error("MCP response missing result");
	if (result.isError === true) {
		const text = mcpTextBlocks(result).join(" ").trim();
		throw new Error(truncate(text || "MCP tool reported an error", 300).text);
	}
	return result;
}

/** Pull the JSON-RPC object out of a raw JSON or SSE response body. */
function extractJsonRpcPayload(raw: string): Record<string, any> | null {
	const trimmed = raw.trim();
	if (trimmed.startsWith("{")) {
		try {
			return JSON.parse(trimmed);
		} catch {
			return null;
		}
	}
	// SSE: take the last data line carrying a JSON-RPC response (result or
	// error). Request ids vary per call (1 = initialize, 2 = tools/call), so
	// id equality is not a usable filter.
	let found: Record<string, any> | null = null;
	for (const line of trimmed.split("\n")) {
		const m = line.match(/^data:\s*(\{.*\})\s*$/);
		if (!m) continue;
		try {
			const parsed = JSON.parse(m[1]);
			if (parsed.jsonrpc === "2.0" && (parsed.result !== undefined || parsed.error !== undefined)) found = parsed;
		} catch {
			// ignore non-JSON data lines
		}
	}
	return found;
}

function mcpTextBlocks(result: Record<string, any>): string[] {
	const content = result.content;
	if (!Array.isArray(content)) return [];
	return content
		.filter((c: any) => c && c.type === "text" && typeof c.text === "string")
		.map((c: any) => c.text as string);
}

/**
 * Parse Exa's text sections (`Title:`/`URL:`/`Author:`/`Published Date:`/
 * `Text:`) into results. Returns null unless at least one section carries
 * both a title and a URL — partial guesses are never promoted to sources.
 */
export function parseExaTextSections(text: string): ParsedResult[] | null {
	const sections = text
		.replace(/\r\n?/g, "\n")
		.split(/\n{2,}(?=Title:)/)
		.map((s) => s.trim())
		.filter((s) => s.startsWith("Title:"));
	const results: ParsedResult[] = [];
	for (const section of sections) {
		const field = (name: string): string | undefined => {
			const m = section.match(new RegExp(`^${name}:\\s*(.+)$`, "m"));
			return m ? m[1].trim() : undefined;
		};
		const title = field("Title");
		const url = field("URL");
		if (!title || !url) continue;
		const body = section.split(/^Text:\s*/m)[1];
		results.push({
			title,
			url,
			publishedDate: field("Published Date"),
			excerpt: body ? body.trim() : undefined,
		});
	}
	return results.length > 0 ? results : null;
}

interface ExaOutcome {
	kind: "structured" | "raw";
	results?: ParsedResult[];
	rawText?: string;
}

function normalizeExaSearch(result: Record<string, any>): ExaOutcome {
	// Some MCP clients receive structuredContent; prefer it when present.
	const structured = result.structuredContent;
	if (structured && Array.isArray(structured.results)) {
		const results: ParsedResult[] = [];
		for (const r of structured.results) {
			if (r && typeof r.url === "string" && r.url) {
				results.push({
					title: typeof r.title === "string" && r.title ? r.title : r.url,
					url: r.url,
					publishedDate: typeof r.publishedDate === "string" ? r.publishedDate : undefined,
					excerpt:
						typeof r.summary === "string"
							? r.summary
							: typeof r.text === "string"
								? r.text
								: Array.isArray(r.highlights)
									? r.highlights.join(" ")
									: undefined,
				});
			}
		}
		if (results.length > 0) return { kind: "structured", results };
	}
	const text = mcpTextBlocks(result).join("\n\n");
	const parsed = parseExaTextSections(text);
	if (parsed) return { kind: "structured", results: parsed };
	if (text.trim()) return { kind: "raw", rawText: text };
	throw new Error("Exa returned no parsable results or text");
}

async function exaSearch(params: SearchParams, deps: WebSearchDeps): Promise<ExaOutcome> {
	const numResults = clampInt(params.numResults, 1, MAX_NUM_RESULTS, DEFAULT_NUM_RESULTS);
	// The current Exa schema requires `objective`; the supplied query is the
	// deterministic adaptation — no second model authors it.
	const result = await exaMcpCall("web_search_exa", { query: params.query, objective: params.query, numResults }, deps);
	return normalizeExaSearch(result);
}

async function exaFetch(url: string, deps: WebSearchDeps): Promise<string> {
	const result = await exaMcpCall(
		"web_fetch_exa",
		{ urls: [url], maxCharacters: MAX_FETCH_CHARS },
		deps,
	);
	const text = mcpTextBlocks(result).join("\n\n").trim();
	if (!text) throw new Error("Exa fetch returned no content");
	return text;
}

// ── ZAI MCP (session-based, bearer key) ─────────────────────────────────────

async function zaiSession(url: string, key: string, deps: WebSearchDeps): Promise<string> {
	const fetchImpl = deps.fetchImpl ?? fetch;
	const response = await fetchImpl(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Accept: "application/json, text/event-stream",
			Authorization: `Bearer ${key}`,
		},
		body: JSON.stringify({
			jsonrpc: "2.0",
			id: 1,
			method: "initialize",
			params: {
				protocolVersion: "2024-11-05",
				capabilities: {},
				clientInfo: { name: "saccade-websearch", version: "0.1.0" },
			},
		}),
		signal: boundedSignal(deps.signal, deps.timeoutMs ?? PROVIDER_TIMEOUT_MS),
	});
	if (!response.ok) throw new Error(`initialize HTTP ${response.status}`);
	const sessionId = response.headers.get("mcp-session-id");
	if (!sessionId) throw new Error("no mcp-session-id on initialize response");
	// Notifications/initialized is protocol-politeness; its failure is not
	// fatal — the tools/list exchange worked without it during verification.
	await fetchImpl(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Accept: "application/json, text/event-stream",
			Authorization: `Bearer ${key}`,
			"mcp-session-id": sessionId,
		},
		body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
		signal: boundedSignal(deps.signal, deps.timeoutMs ?? PROVIDER_TIMEOUT_MS),
	}).catch(() => undefined);
	return sessionId;
}

async function zaiToolCall(
	baseUrl: string,
	toolName: string,
	args: Record<string, unknown>,
	key: string,
	sessionId: string,
	deps: WebSearchDeps,
): Promise<Record<string, any>> {
	const fetchImpl = deps.fetchImpl ?? fetch;
	const response = await fetchImpl(baseUrl, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Accept: "application/json, text/event-stream",
			Authorization: `Bearer ${key}`,
			"mcp-session-id": sessionId,
		},
		body: JSON.stringify({
			jsonrpc: "2.0",
			id: 2,
			method: "tools/call",
			params: { name: toolName, arguments: args },
		}),
		signal: boundedSignal(deps.signal, deps.timeoutMs ?? PROVIDER_TIMEOUT_MS),
	});
	const raw = await response.text();
	if (!response.ok) throw new Error(`HTTP ${response.status}: ${truncate(raw, 300).text}`);
	const data = extractJsonRpcPayload(raw);
	if (!data) throw new Error("unparseable MCP response (no JSON-RPC body)");
	if (data.error) {
		const err = new Error(`MCP error: ${JSON.stringify(data.error)}`);
		(err as any).shapeError = true;
		throw err;
	}
	const result = data.result;
	if (!result || typeof result !== "object") throw new Error("MCP response missing result");
	if (result.isError === true) {
		const text = mcpTextBlocks(result).join(" ").trim();
		const err = new Error(truncate(text || "MCP tool reported an error", 300).text);
		(err as any).shapeError = true;
		throw err;
	}
	return result;
}

function isShapeError(err: unknown): boolean {
	const msg = err instanceof Error ? err.message : String(err);
	const parsed = msg.match(/MCP error: (\{.*\})/);
	if (parsed) {
		try {
			const code = JSON.parse(parsed[1])?.code;
			// -32602 (invalid params) and -32601/unknown-tool style protocol errors
			return code === -32602 || code === -32601;
		} catch {
			return false;
		}
	}
	return (err as any)?.shapeError === true;
}

async function zaiSearch(
	query: string,
	numResults: number | undefined,
	key: string,
	deps: WebSearchDeps,
): Promise<ExaOutcome> {
	// One session spans the bounded shape attempts; a shape error must not
	// multiply session setups.
	const sessionId = await zaiSession(ZAI_SEARCH_URL, key, deps);
	// Attempt shapes, bounded: the skill-proven shape first, then the minimal
	// schema shape. Only argument-shape failures advance; real failures stop.
	const shapes: Record<string, unknown>[] = [
		{ search_query: query, location: "us", content_size: "high" },
		{ search_query: query },
	];
	let lastError: unknown;
	for (const args of shapes) {
		try {
			const result = await zaiToolCall(ZAI_SEARCH_URL, "web_search_prime", args, key, sessionId, deps);
			const text = mcpTextBlocks(result).join("\n\n").trim();
			if (!text) throw new Error("ZAI search returned no content");
			return parseZaiSearchPayload(text);
		} catch (err) {
			lastError = err;
			if (!isShapeError(err)) throw err;
		}
	}
	throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/**
 * ZAI returns a JSON array (often JSON-encoded inside the text block) of
 * {title, link, content}. Parsed only when it round-trips cleanly; the
 * domain-level links are shown exactly as the provider returns them.
 */
export function parseZaiSearchPayload(text: string): ExaOutcome {
	try {
		let value: any = JSON.parse(text);
		if (typeof value === "string") value = JSON.parse(value);
		if (Array.isArray(value)) {
			const results: ParsedResult[] = [];
			for (const item of value) {
				if (item && typeof item === "object" && typeof item.title === "string" && typeof item.link === "string") {
					results.push({
						title: item.title,
						url: item.link,
						excerpt: typeof item.content === "string" ? item.content : undefined,
					});
				}
			}
			if (results.length > 0) return { kind: "structured", results };
		}
	} catch {
		// fall through to raw
	}
	return { kind: "raw", rawText: text };
}

async function zaiFetch(url: string, key: string, deps: WebSearchDeps): Promise<string> {
	const sessionId = await zaiSession(ZAI_READER_URL, key, deps);
	const result = await zaiToolCall(ZAI_READER_URL, "webReader", { url, return_format: "markdown" }, key, sessionId, deps);
	const text = mcpTextBlocks(result).join("\n\n").trim();
	if (!text) throw new Error("ZAI reader returned no content");
	return unwrapZaiReaderPayload(text);
}

/** The reader wraps the page in a JSON string even in markdown mode; unwrap
 * to the content field when it round-trips, else pass the text through. */
function unwrapZaiReaderPayload(text: string): string {
	try {
		let value: any = JSON.parse(text);
		if (typeof value === "string") value = JSON.parse(value);
		if (value && typeof value === "object" && typeof value.content === "string" && value.content.trim()) {
			return value.content;
		}
	} catch {
		// not JSON — return as-is
	}
	return text;
}

// ── Orchestration: Exa first, ZAI when keyed; cancel never falls back ──────

function secretsOf(env: Record<string, string | undefined>): string[] {
	return [env.ZAI_API_KEY];
}

async function runWithFallback(
	deps: WebSearchDeps,
	runExa: () => Promise<ToolOutcome>,
	runZai: () => Promise<ToolOutcome>,
	toolName: string,
): Promise<ToolOutcome> {
	const env = deps.env ?? process.env;
	const secrets = secretsOf(env);
	const attempts: string[] = [];
	try {
		return await runExa();
	} catch (err) {
		throwIfAborted(deps.signal);
		attempts.push(`exa-mcp: ${scrub(err instanceof Error ? err.message : String(err), secrets)}`);
	}
	const key = env.ZAI_API_KEY;
	if (!key) {
		throw new Error(`${toolName} failed — ${attempts.join("; ")}; zai: unavailable (ZAI_API_KEY not set)`);
	}
	try {
		return await runZai();
	} catch (err) {
		// A caller cancellation arriving during the fallback attempt is not a
		// provider failure — surface it as cancellation, never as an error.
		throwIfAborted(deps.signal);
		attempts.push(`zai: ${scrub(err instanceof Error ? err.message : String(err), secrets)}`);
	}
	throw new Error(`${toolName} failed — ${attempts.join("; ")}`);
}

export async function runWebSearch(params: SearchParams, deps: WebSearchDeps = {}): Promise<ToolOutcome> {
	throwIfAborted(deps.signal);
	const numResults = clampInt(params.numResults, 1, MAX_NUM_RESULTS, DEFAULT_NUM_RESULTS);
	return runWithFallback(
		deps,
		async () => formatSearchOutcome("exa-mcp", await exaSearch(params, deps), numResults),
		async () => {
			const key = (deps.env ?? process.env).ZAI_API_KEY!;
			const outcome = await zaiSearch(params.query, numResults, key, deps);
			return formatSearchOutcome("zai", outcome, numResults);
		},
		"web_search",
	);
}

function formatSearchOutcome(provider: string, outcome: ExaOutcome, numResults: number): ToolOutcome {
	if (outcome.kind === "structured") {
		const results = outcome.results!.slice(0, numResults);
		const lines: string[] = [`${results.length} result${results.length === 1 ? "" : "s"} via ${provider}`, ""];
		if (provider === "zai") lines.push(`(zai links are domain-level, exactly as the provider returns them)`, "");
		results.forEach((r, i) => {
			const date = r.publishedDate ? ` (${r.publishedDate})` : "";
			lines.push(`[${i + 1}] ${r.title}${date}`);
			lines.push(`    ${r.url}`);
			if (r.excerpt) {
				const excerpt = truncate(r.excerpt.replace(/\s+/g, " ").trim(), EXCERPT_CHARS);
				lines.push(`    ${excerpt.text}${excerpt.truncated ? "…" : ""}`);
			}
			lines.push("");
		});
		return {
			text: lines.join("\n").trimEnd(),
			details: { provider, resultCount: results.length, truncatedTo: numResults },
		};
	}
	const capped = truncate(outcome.rawText!, RAW_TEXT_CAP);
	return {
		text:
			`results via ${provider} (raw provider content; source structure not confidently parseable — URLs appear inline)\n\n` +
			capped.text +
			(capped.truncated ? `\n\n[truncated at ${RAW_TEXT_CAP} characters]` : ""),
		details: { provider, kind: "raw" },
	};
}

export async function runWebFetch(params: FetchParams, deps: WebSearchDeps = {}): Promise<ToolOutcome> {
	throwIfAborted(deps.signal);
	if (!params.url || !/^https?:\/\//i.test(params.url)) {
		throw new Error("web_fetch requires an http(s) URL");
	}
	const maxChars = clampInt(params.maxChars, 1, MAX_FETCH_CHARS, DEFAULT_FETCH_CHARS);
	const outcome = await runWithFallback(
		deps,
		async (): Promise<ToolOutcome> => {
			const content = await exaFetch(params.url, deps);
			return formatFetchOutcome(params.url, "exa-mcp", content, maxChars, deps);
		},
		async (): Promise<ToolOutcome> => {
			const key = (deps.env ?? process.env).ZAI_API_KEY!;
			const content = await zaiFetch(params.url, key, deps);
			return formatFetchOutcome(params.url, "zai", content, maxChars, deps);
		},
		"web_fetch",
	);
	return outcome;
}

async function formatFetchOutcome(
	url: string,
	provider: string,
	content: string,
	maxChars: number,
	deps: WebSearchDeps,
): Promise<ToolOutcome> {
	const header = `Content from ${url} via ${provider}\n\n`;
	const capped = truncate(content, maxChars);
	if (!capped.truncated) {
		return { text: header + content, details: { provider, url, chars: content.length } };
	}
	const path = await persistFullContent(url, content, deps);
	return {
		text:
			header +
			`Showing first ${maxChars} of ${content.length} characters. Full content saved to:\n${path}\n` +
			`(read it with the read tool using offset/limit)\n\n---\n\n` +
			capped.text,
		details: { provider, url, chars: content.length, inlineChars: maxChars, fullPath: path },
	};
}

async function persistFullContent(url: string, content: string, deps: WebSearchDeps): Promise<string> {
	const { mkdir, writeFile } = await import("node:fs/promises");
	const { tmpdir } = await import("node:os");
	const { join } = await import("node:path");
	const { createHash } = await import("node:crypto");
	const dir = deps.tmpDir ?? join(tmpdir(), "saccade-websearch");
	await mkdir(dir, { recursive: true });
	let host = "page";
	try {
		host = new URL(url).hostname.replace(/[^a-z0-9.-]/gi, "_") || "page";
	} catch {
		// keep default
	}
	const hash = createHash("sha1").update(url).digest("hex").slice(0, 10);
	const path = join(dir, `${host}-${hash}.md`);
	await writeFile(path, content, "utf-8");
	return path;
}

// ── Pi registration ─────────────────────────────────────────────────────────

export default async function (pi: ExtensionAPI): Promise<void> {
	const { Type } = await import("typebox");
	pi.registerTool({
		name: "web_search",
		label: "Web Search",
		description:
			`Search the web and return source titles, URLs and excerpts with provider attribution. ` +
			`Providers: keyless Exa MCP first, ZAI fallback when ZAI_API_KEY is set. ` +
			`numResults: results per query (default ${DEFAULT_NUM_RESULTS}, max ${MAX_NUM_RESULTS}). ` +
			`Excerpts are capped at ${EXCERPT_CHARS} characters; pass recency/site operators inside the query text itself ` +
			`(forwarded verbatim, backend filtering not guaranteed). One bounded attempt per provider, ${PROVIDER_TIMEOUT_MS / 1000}s timeout each.`,
		promptSnippet:
			"Use for current web information; returns sourced titles/URLs/excerpts, not full pages — follow up with web_fetch on promising URLs.",
		parameters: Type.Object({
			query: Type.String({ description: "Search query; recency/site operators may be included verbatim." }),
			numResults: Type.Optional(
				Type.Number({ description: `Results per query (default ${DEFAULT_NUM_RESULTS}, max ${MAX_NUM_RESULTS}).` }),
			),
		}),
		async execute(_toolCallId, params, signal) {
			try {
				const outcome = await runWebSearch(
					{ query: params.query, numResults: params.numResults },
					{ signal },
				);
				return { content: [{ type: "text" as const, text: outcome.text }], details: outcome.details };
			} catch (err) {
				if (err instanceof AbortedError || (err as any)?.name === "AbortError") {
					return { content: [{ type: "text" as const, text: "Cancelled" }], details: { cancelled: true } };
				}
				throw err;
			}
		},
	});

	pi.registerTool({
		name: "web_fetch",
		label: "Web Fetch",
		description:
			`Fetch a URL and return its readable content (markdown where the provider supplies it) with the source URL stated. ` +
			`Providers: keyless Exa MCP first, ZAI reader fallback when ZAI_API_KEY is set. ` +
			`maxChars: inline character bound (default ${DEFAULT_FETCH_CHARS}, max ${MAX_FETCH_CHARS}); larger content is ` +
			`written to a temporary file whose path is returned for ordinary reads. Extraction is capped at ${MAX_FETCH_CHARS} characters. ` +
			`JavaScript-heavy pages may fail — the error says which provider and why.`,
		promptSnippet:
			"Use to read a specific page found via web_search; oversized content lands in a temp file you can read with offset/limit.",
		parameters: Type.Object({
			url: Type.String({ description: "The http(s) URL to fetch." }),
			maxChars: Type.Optional(
				Type.Number({
					description: `Inline character bound (default ${DEFAULT_FETCH_CHARS}, max ${MAX_FETCH_CHARS}); larger content goes to a temp file.`,
				}),
			),
		}),
		async execute(_toolCallId, params, signal) {
			try {
				const outcome = await runWebFetch({ url: params.url, maxChars: params.maxChars }, { signal });
				return { content: [{ type: "text" as const, text: outcome.text }], details: outcome.details };
			} catch (err) {
				if (err instanceof AbortedError || (err as any)?.name === "AbortError") {
					return { content: [{ type: "text" as const, text: "Cancelled" }], details: { cancelled: true } };
				}
				throw err;
			}
		},
	});
}
