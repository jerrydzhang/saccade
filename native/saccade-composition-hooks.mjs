// The load hook half of the composition preload: transforms exactly one
// anchored property forward in the pinned runtime's sendUserMessage, so
// a delivery that passes preflightResult receives its own per-call
// acceptance outcome (queued at steer insertion, started at run start,
// handled on interception). The pinned runtime ships the same forward
// in two forms - the readable package module and the minified cli
// bundle - so both anchors are carried, each independently asserted to
// appear exactly once before it is rewritten. Any source without an
// exact anchor passes through untouched, which is the confinement
// proof: arbitrary processes that inherit the registration never
// contain an anchor and are never modified, and a pinned-runtime change
// that moves an anchor fails the session loudly instead of silently
// no-oping.

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

const FORMS = [
	[READABLE_ANCHOR, READABLE_FORWARDED],
	[MINIFIED_ANCHOR, MINIFIED_FORWARDED],
];

const decoder = new TextDecoder();

function transform(source, url) {
	for (const [anchor, forwarded] of FORMS) {
		const hits = source.split(anchor).length - 1;
		if (hits === 0) continue;
		if (hits !== 1) {
			throw new Error(
				`saccade composition shim: the sendUserMessage forward anchor appears ${hits} times in ${url}; the pinned runtime moved and the receipt forward cannot be applied safely`,
			);
		}
		return source.replace(anchor, forwarded);
	}
	return source;
}

export async function load(url, context, nextLoad) {
	const result = await nextLoad(url, context);
	// newer runtimes hand the default load's source over as a typed
	// array, older ones as a string; both decode to the same text
	if (result.source === null || result.source === undefined) {
		return result;
	}
	const text = typeof result.source === "string" ? result.source : decoder.decode(result.source);
	const transformed = transform(text, url);
	if (transformed === text) {
		return result;
	}
	return { ...result, source: transformed };
}
