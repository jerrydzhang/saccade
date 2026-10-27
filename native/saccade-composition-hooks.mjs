// The load hook half of the composition preload: the receipt forward
// applies ONLY to the exact pinned module URLs baked into the
// receiptSources list at build time; every other module - foreign
// same-anchor code, child SDK copies - passes through untouched. The
// registration is argument-scoped, so unrelated processes never run
// this hook at all. An intended target carrying anything but exactly
// one anchor fails its load visibly: silence would leave sends stuck
// unacknowledged.
import { readFileSync } from "node:fs";

const receiptSources = JSON.parse(
	readFileSync(new URL("receipt-sources.json", import.meta.url), "utf8"),
);
const intended = new Set(receiptSources);

// The readable form shipped in dist/core/agent-session.js.
const READABLE_ANCHOR = `        await this.prompt(text, {
            expandPromptTemplates: options?.expandPromptTemplates ?? false,
            streamingBehavior: options?.deliverAs,
            images,
            source: "extension",
        });`;
const READABLE_FORWARDED = `        await this.prompt(text, {
            expandPromptTemplates: options?.expandPromptTemplates ?? false,
            streamingBehavior: options?.deliverAs,
            images,
            source: "extension",
            preflightResult: options?.preflightResult,
        });`;

// The minified form shipped in the cli bundle chunk.
const MINIFIED_ANCHOR =
	'await this.prompt(text,{expandPromptTemplates:options?.expandPromptTemplates??!1,streamingBehavior:options?.deliverAs,images,source:"extension"})';
const MINIFIED_FORWARDED =
	'await this.prompt(text,{expandPromptTemplates:options?.expandPromptTemplates??!1,streamingBehavior:options?.deliverAs,images,source:"extension",preflightResult:options?.preflightResult})';

// The intended target's shipped form follows its path: the readable
// module carries the spaced anchor, the minified bundle chunk carries
// the condensed one.
function anchorFor(url) {
	if (url.endsWith("/dist/core/agent-session.js")) return READABLE_ANCHOR;
	if (url.includes("/dist/bundle/chunks/")) return MINIFIED_ANCHOR;
	return undefined;
}

const decoder = new TextDecoder();

export async function load(url, context, nextLoad) {
	if (!intended.has(url)) {
		return nextLoad(url, context);
	}
	const anchor = anchorFor(url);
	const result = await nextLoad(url, context);
	// newer runtimes hand the default load's source over as a typed
	// array, older ones as a string; both decode to the same text
	if (result.source === null || result.source === undefined) {
		throw new Error(
			`saccade composition preload: the intended target ${url} surfaced no source to transform`,
		);
	}
	const text = typeof result.source === "string" ? result.source : decoder.decode(result.source);
	const hits = text.split(anchor).length - 1;
	if (hits !== 1) {
		throw new Error(
			`saccade composition preload: the pinned sendUserMessage forward anchor appears ${hits} times in the intended target ${url}; the receipt forward cannot apply - failing visibly instead of leaving sends stuck unacknowledged`,
		);
	}
	const forwarded = anchor === READABLE_ANCHOR ? READABLE_FORWARDED : MINIFIED_FORWARDED;
	return {...result, source: text.replace(anchor, forwarded)};
}
