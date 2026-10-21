// The integration's mechanics and queue contract, pinned against a stateful fake of the native followUp queue.
import assert from "node:assert/strict";
import test from "node:test";
import {
	envelopeText,
	incomingFor,
	numToken,
	userText,
	default as saccadeIntegration,
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
	// coercive forms refuse: signs, exponents, hex, decimals, inner whitespace
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
	// the body is user data riding byte for byte
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

interface SessionEntryLike {
	type: string;
	message?: { role: string; content: unknown };
}

class FakeSession {
	entries: SessionEntryLike[] = [];
	nativeId: string;
	file: string;
	constructor(nativeId: string, file: string) {
		this.nativeId = nativeId;
		this.file = file;
	}
	getSessionId() {
		return this.nativeId;
	}
	getSessionFile() {
		return this.file;
	}
	getEntries() {
		return this.entries;
	}
}

// The fake queue drains only after a run ends, firing events before the append.
class FakePi {
	tools = new Map<string, any>();
	handlers = new Map<string, Array<(event: any, ctx: any) => unknown>>();
	followUps: string[] = [];
	session: FakeSession;

	constructor(nativeId: string, file: string) {
		this.session = new FakeSession(nativeId, file);
	}

	registerTool(def: any) {
		this.tools.set(def.name, def);
	}
	on(event: string, handler: (event: any, ctx: any) => unknown) {
		const list = this.handlers.get(event) ?? [];
		list.push(handler);
		this.handlers.set(event, list);
		return () => {
			const at = list.indexOf(handler);
			if (at >= 0) list.splice(at, 1);
		};
	}
	sendUserMessage(content: string, _opts?: unknown) {
		this.followUps.push(content);
	}
	appendEntry(_type: string, _data?: unknown) {}

	async emit(event: string, extra: Record<string, unknown> = {}) {
		for (const handler of this.handlers.get(event) ?? []) {
			await handler({ type: event, ...extra }, { sessionManager: this.session });
		}
	}

	// the queue drains after the tool call returned, events before the append
	async runQueue() {
		const queued = this.followUps.splice(0);
		for (const text of queued) {
			const message = { role: "user", content: text };
			await this.emit("message_start", { message });
			await this.emit("message_end", { message });
			this.session.entries.push({ type: "message", message });
		}
		await this.emit("agent_end", {});
		await this.emit("agent_settled", {});
	}
}

class FakeTracker {
	attachments = new Map<string, { incarnation: number; num: number }>();
	mailboxes = new Map<number, any[]>();
	submissions: Array<{ incarnation: number; message: number }> = [];
	sent: any[] = [];
	attachBodies: any[] = [];
	attachFails = false;
	failNext = 0;
	// parked bridge reads: held until a record lands or the test wakes them
	parked: Array<() => void> = [];

	url() {
		return "http://tracker.test";
	}

	private reply(body: unknown, status = 200) {
		return { status, json: async () => body };
	}

	// a landed record wakes every parked read, like the server's notify
	land(incarnation: number, message: any) {
		const box = this.mailboxes.get(incarnation) ?? [];
		box.push(message);
		this.mailboxes.set(incarnation, box);
		for (const wake of this.parked.splice(0)) wake();
	}

	handle = async (rawUrl: any, init?: any) => {
		const path = new URL(String(rawUrl)).pathname;
		const body = init?.body ? JSON.parse(init.body) : {};
		if (this.failNext > 0) {
			this.failNext -= 1;
			throw new Error("tracker unreachable");
		}
		if (path === "/api/v1/native/attach") {
			this.attachBodies.push(body);
			if (this.attachFails) {
				return this.reply({ error: { code: "degraded", detail: "down" } }, 503);
			}
			const known = this.attachments.get(body.native_id);
			if (known) {
				return this.reply({ registered: false, ...known });
			}
			const fresh = { incarnation: this.attachments.size, num: this.attachments.size };
			this.attachments.set(body.native_id, fresh);
			return this.reply({ registered: true, ...fresh });
		}
		if (path === "/api/v1/messages/pending") {
			const url = new URL(String(rawUrl));
			const incarnation = Number(url.searchParams.get("incarnation"));
			const mine = (this.mailboxes.get(incarnation) ?? []).filter(
				(m: any) => m.state === "pending" && m.to.seq === incarnation,
			);
			const waitMs = Number(url.searchParams.get("wait_ms") ?? 0);
			if (waitMs > 0 && mine.length === 0) {
				// the parked read: held until a record lands
				await new Promise<void>((resolve) => this.parked.push(resolve));
			}
			return this.reply({ messages: this.mailboxes.get(incarnation) ?? [] });
		}
		if (path === "/api/v1/messages/send") {
			this.sent.push(body);
			const seq = 100 + this.sent.length;
			return this.reply({
				message: { seq, num: seq },
				from: { seq: body.context?.scope?.incarnation ?? 0, num: body.context?.scope?.incarnation ?? 0 },
			});
		}
		if (path === "/api/v1/messages/submit") {
			this.submissions.push({ incarnation: body.incarnation, message: body.message });
			return this.reply({});
		}
		return this.reply({ error: { code: "not_found", detail: path } }, 404);
	};

	install() {
		globalThis.fetch = this.handle as any;
	}
}

interface Harness {
	pi: FakePi;
	tracker: FakeTracker;
	tools: (name: string) => any;
}

function harness(nativeId: string, file: string, actor?: string): Harness {
	if (actor === undefined) {
		delete process.env.SACCADE_ACTOR;
	} else {
		process.env.SACCADE_ACTOR = actor;
	}
	const pi = new FakePi(nativeId, file);
	saccadeIntegration(pi as any);
	const tracker = new FakeTracker();
	tracker.install();
	process.env.SACCADE_SERVER = tracker.url();
	return {
		pi,
		tracker,
		tools: (name: string) => pi.tools.get(name),
	};
}

const boundary = <T>(promise: Promise<T>, ms: number, what: string) =>
	Promise.race([
		promise,
		new Promise<T>((_resolve, reject) => {
			const timer = setTimeout(
				() => reject(new Error(`${what} never returned`)),
				ms,
			);
			// a lost race must not hold the test process open
			timer.unref();
		}),
	]);

const tick = () => new Promise((resolve) => setTimeout(resolve, 20));

test("a released wait queues and returns; the submission confirms from saved history", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("aaaa1111", me);
	h.tracker.mailboxes.set(0, [mail({ seq: 50, to: { seq: 0, num: 0 } })]);
	// the start drain queues the mail natively; nothing submitted yet
	await h.pi.emit("session_start");
	assert.equal(h.pi.followUps.length, 1);
	assert.equal(h.tracker.submissions.length, 0);

	// events before the append submit nothing
	const envelope = h.pi.followUps[0];
	const message = { role: "user", content: envelope };
	await h.pi.emit("message_start", { message });
	await h.pi.emit("message_end", { message });
	await tick();
	assert.equal(h.tracker.submissions.length, 0, "events precede insertion");

	// the entry lands, and a later boundary confirms against it
	h.pi.session.entries.push({ type: "message", message });
	await h.pi.emit("agent_settled");
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 50 }]);
});

test("the wait tool returns while the queue is still undrained", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("aaaa1111", me);
	await h.pi.emit("session_start");
	h.tracker.mailboxes.set(0, [mail({ seq: 51, num: 9, to: { seq: 0, num: 0 } })]);

	const wait = h.tools("saccade_wait");
	const controller = new AbortController();
	const result = await boundary(
		wait.execute("t1", {}, controller.signal, { sessionManager: h.pi.session }),
		8000,
		"saccade_wait",
	);
	// the tool returned before any queue processing
	assert.ok(String(result.content[0].text).includes("received m-9"));
	assert.equal(h.pi.followUps.length, 1);
	assert.equal(h.tracker.submissions.length, 0);
	// and the queued delivery still confirms after the run
	await h.pi.runQueue();
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 51 }]);
});

test("the wait finishes cancelled when the call aborts, queueing nothing", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("aaaa1111", me);
	await h.pi.emit("session_start");

	const wait = h.tools("saccade_wait");
	const controller = new AbortController();
	const running = wait.execute("t1", {}, controller.signal, {
		sessionManager: h.pi.session,
	});
	controller.abort();
	const result = await boundary(running, 8000, "aborted saccade_wait");
	assert.ok(
		String(result.content[0].text).includes("the call was cancelled"),
	);
	assert.equal(h.pi.followUps.length, 0);
	await h.pi.runQueue();
	await tick();
	assert.equal(h.tracker.submissions.length, 0);
});

test("reassociation cancels a queued delivery: no stale recipient submits", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("aaaa1111", me);
	h.tracker.mailboxes.set(0, [mail({ seq: 52, to: { seq: 0, num: 0 } })]);
	await h.pi.emit("session_start");
	assert.equal(h.pi.followUps.length, 1);

	// a new native session cancels the old association's queued delivery
	h.pi.session = new FakeSession("bbbb2222", "/sessions/bbbb2222.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.tracker.attachments.size, 2);

	// a later boundary never submits the cancelled delivery
	h.pi.followUps = [];
	h.pi.session.entries.push({
		type: "message",
		message: { role: "user", content: envelopeText(mail({ seq: 52, to: { seq: 0, num: 0 } })) },
	});
	await h.pi.emit("agent_settled");
	await tick();
	assert.equal(h.tracker.submissions.length, 0);
});

test("agent possession survives a tracker outage; injected actors win", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	assert.equal(process.env.SACCADE_ACTOR, "saccade-pi");
	h.tracker.attachFails = true;
	await h.pi.emit("session_start");
	// a failed attach still leaves nested sac calls agent-tier
	assert.equal(process.env.SACCADE_ACTOR, "saccade-pi");
	assert.equal(
		Object.prototype.hasOwnProperty.call(process.env, "SACCADE_INCARNATION_SEQ"),
		false,
	);

	// an injected actor is never overwritten
	const injected = harness("cccc3333", "/sessions/cccc3333.jsonl", "worker");
	assert.equal(process.env.SACCADE_ACTOR, "worker");
	injected.tracker.attachFails = true;
	await injected.pi.emit("session_start");
	assert.equal(process.env.SACCADE_ACTOR, "worker");
});

test("a running registered peer receives without any wait tool", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.pi.followUps.length, 0);
	// the bridge holds a parked read; the record's landing wakes it
	h.tracker.land(0, mail({ seq: 60, num: 5, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 1);
	assert.ok(h.pi.followUps[0].includes("[saccade message m-5"));
	assert.equal(h.tracker.submissions.length, 0);
	// the queued envelope still confirms from saved history
	await h.pi.runQueue();
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 60 }]);
});

test("background arrival and the wait tool queue a message once", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.tracker.land(0, mail({ seq: 61, num: 6, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 1);
	// the wait tool sees the same pending message and releases, but
	// its own queueing is a no-op: one envelope, one path
	const wait = h.tools("saccade_wait");
	const result = await boundary(
		wait.execute("t1", {}, new AbortController().signal, { sessionManager: h.pi.session }),
		8000,
		"saccade_wait",
	);
	assert.ok(String(result.content[0].text).includes("received m-6"));
	assert.equal(h.pi.followUps.length, 1);
});

test("a failed bridge read reconnects and keeps delivering", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	// the read after the parked one fails, then transport recovers
	h.tracker.failNext = 1;
	h.tracker.land(0, mail({ seq: 62, num: 7, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 1, "the landing delivered before the failure");
	// mail landing while the bridge is in its backoff still arrives
	const later = mail({ seq: 63, num: 8, to: { seq: 0, num: 0 } });
	const before = h.pi.followUps.length;
	const deadline = Date.now() + 8000;
	while (h.pi.followUps.length === before && Date.now() < deadline) {
		h.tracker.land(0, later);
		await new Promise((resolve) => setTimeout(resolve, 150));
	}
	assert.ok(h.pi.followUps.length > before, "the reconnected bridge delivered");
});

test("session shutdown stops the bridge: nothing delivers after it", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	await h.pi.emit("session_shutdown");
	assert.equal(h.pi.followUps.length, 0);
	h.tracker.land(0, mail({ seq: 64, num: 9, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 0);
	assert.equal(h.tracker.submissions.length, 0);
});

test("reassociation strands the old bridge: no stale delivery, no stale submit", async () => {
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	// a new native session reassociates; the old bridge's parked read dies
	h.pi.session = new FakeSession("bbbb2222", "/sessions/bbbb2222.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.tracker.attachments.size, 2);
	// mail for the OLD incarnation lands and wakes every parked read
	h.tracker.land(0, mail({ seq: 65, num: 1, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 0, "the old bridge delivered nothing");
	assert.equal(h.tracker.submissions.length, 0);
	// and the new association still delivers its own
	h.tracker.land(1, mail({ seq: 66, num: 2, to: { seq: 1, num: 1 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 1);
});

test("the attach carries the launch token it was launched with", async () => {
	process.env.SACCADE_LAUNCH_TOKEN = "launch-abc123";
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.tracker.attachBodies.length, 1);
	assert.equal(h.tracker.attachBodies[0].launch_token, "launch-abc123");
	delete process.env.SACCADE_LAUNCH_TOKEN;
});

test("the declared task anchors scope; an explicit task argument is message context", async () => {
	process.env.SACCADE_ACTOR = "pi";
	process.env.SACCADE_TASK = "t-9";
	delete process.env.SACCADE_WORKSPACE;
	const h = harness("aaaa1111", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("aaaa1111", me);
	await h.pi.emit("session_start");

	const send = h.tools("saccade_send");
	await send.execute(
		"t1",
		{ to: "i-1", body: "the answer", task: "t-2" },
		undefined,
		{ sessionManager: h.pi.session },
	);
	const sent = h.tracker.sent[h.tracker.sent.length - 1];
	// scope.task reads the declared environment alone
	assert.equal(sent.context.scope.task, 9);
	// the argument rides the message, never the scope
	assert.equal(sent.task, 2);

	// no declared task: the scope stays task-free even with an argument
	delete process.env.SACCADE_TASK;
	await send.execute(
		"t2",
		{ to: "i-1", body: "another", task: "t-2" },
		undefined,
		{ sessionManager: h.pi.session },
	);
	const second = h.tracker.sent[h.tracker.sent.length - 1];
	assert.equal(second.context.scope.task, null);
	assert.equal(second.task, 2);
	delete process.env.SACCADE_TASK;
});
