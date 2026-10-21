// The pure mechanics of the native integration, pinned without pi:
// addresses, the envelope, and the addressee filter. Run with
// `node --experimental-strip-types --test native/saccade.test.ts`.
import assert from "node:assert/strict";
import test from "node:test";
import {
	envelopeText,
	incomingFor,
	numToken,
	userText,
} from "./saccade.ts";

function mail(overrides: Record<string, unknown>) {
	return {
		seq: 7,
		num: 3,
		from: { seq: 1, num: 1, actor: "pi" },
		to: { seq: 0, num: 0 },
		reply_to: null,
		task: null,
		state: "pending",
		body: "the smoke request",
		born_at: 1,
		...overrides,
	};
}

test("an address is exactly the prefix and ASCII digits", () => {
	assert.equal(numToken("i-12", "i-"), 12);
	assert.equal(numToken(" m-0 ", "m-"), 0);
	// no bare number, no missing prefix, no empty tail
	assert.throws(() => numToken("12", "i-"));
	assert.throws(() => numToken("i-", "i-"));
	assert.throws(() => numToken("i", "i-"));
	assert.throws(() => numToken("", "i-"));
	// no coercive forms: signs, exponents, hex, decimals, whitespace
	// inside the digits
	assert.throws(() => numToken("i-+5", "i-"));
	assert.throws(() => numToken("i--5", "i-"));
	assert.throws(() => numToken("i-1e3", "i-"));
	assert.throws(() => numToken("i-0x10", "i-"));
	assert.throws(() => numToken("i-1.5", "i-"));
	assert.throws(() => numToken("i- 5", "i-"));
	assert.throws(() => numToken("t-1_0", "t-"));
	// beyond the safe integers an address could round: refuse
	assert.throws(() => numToken("i-9007199254740993", "i-"));
	assert.equal(numToken("i-9007199254740991", "i-"), 9007199254740991);
});

test("the envelope carries the machine-supplied origin in ASCII", () => {
	const plain = mail({});
	assert.equal(
		envelopeText(plain),
		"[saccade message m-3 | from i-1]\n\nthe smoke request",
	);
	const full = mail({ reply_to: 5, task: 2, num: 4 });
	assert.equal(
		envelopeText(full),
		"[saccade message m-4 | from i-1 | replying to m-5 | task t-2]\n\nthe smoke request",
	);
	// the body is user data and rides byte for byte
	// the body is user data and rides byte for byte: Unicode data
	// pinned through the envelope, the source staying ASCII
	const unicode = mail({ body: "r\u00e9sum\u00e9 \u2014 \u2713" });
	assert.ok(envelopeText(unicode).endsWith("r\u00e9sum\u00e9 \u2014 \u2713"));
});

test("delivery filters on the addressee, self-addressed included", () => {
	const incoming = [
		mail({ seq: 1, state: "pending", to: { seq: 0, num: 0 } }),
		// outgoing archive: mine as sender, not mine to deliver
		mail({ seq: 2, state: "pending", from: { seq: 0, num: 0, actor: "me" }, to: { seq: 1, num: 1 } }),
		// submitted: already handed over
		mail({ seq: 3, state: "submitted", to: { seq: 0, num: 0 } }),
		// another incarnation's mail entirely
		mail({ seq: 4, state: "pending", to: { seq: 2, num: 2 } }),
		// self-addressed: from me and to me is still incoming mail
		mail({ seq: 5, state: "pending", from: { seq: 0, num: 0, actor: "me" }, to: { seq: 0, num: 0 } }),
	];
	const mine = incomingFor(0, incoming);
	assert.deepEqual(
		mine.map((m) => m.seq),
		[1, 5],
	);
});

test("user text flattens string and block content", () => {
	assert.equal(userText({ role: "user", content: "plain" }), "plain");
	assert.equal(
		userText({
			role: "user",
			content: [
				{ type: "text", text: "[saccade message m-3" },
				{ type: "text", text: " | from i-1]" },
			],
		}),
		"[saccade message m-3 | from i-1]",
	);
});
