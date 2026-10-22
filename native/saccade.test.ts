// The integration's mechanics and queue contract, pinned against a stateful fake of the native followUp queue.
import assert from "node:assert/strict";
import test from "node:test";
import { join } from "node:path";
import {
	chmodSync,
	closeSync,
	existsSync,
	mkdtempSync,
	openSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionToolContext,
	SessionEntry,
	Theme,
	ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import {
	envelopeText,
	incomingFor,
	numToken,
	takeGuardLock,
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
		state: "pending" as const,
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

type SessionEntryLike = { type: string; message?: { role: string; content: unknown } };

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
class FakePi implements ExtensionAPI {
	tools = new Map<string, ToolDefinition>();
	handlers = new Map<string, Array<(event: any, ctx: any) => unknown>>();
	followUps: string[] = [];
	session: FakeSession;
	// the next sendUserMessage is refused before queue or history
	failNextSend = 0;
	idle = false;
	pendingNative = false;

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
	// the pinned contract: void, and internal failures vanish into the
	// runtime's own error sink - the extension cannot observe them
	errors: string[] = [];
	shutdowns = 0;
	sendUserMessage(content: string, _opts?: unknown) {
		if (this.failNextSend > 0) {
			this.failNextSend -= 1;
			this.errors.push("send_user_message: no auth: the provider refused");
			return;
		}
		this.followUps.push(content);
	}
	appendEntry<T = unknown>(_customType: string, _data?: T): void {}
	// every other ExtensionAPI member: present on the real surface,
	// never part of this test's exercised surface
	exec(): never {
		throw new Error("fake pi: exec");
	}
	getActiveTools(): never {
		throw new Error("fake pi: getActiveTools");
	}
	getAllTools(): never {
		throw new Error("fake pi: getAllTools");
	}
	getCommands(): never {
		throw new Error("fake pi: getCommands");
	}
	getFlag(): never {
		throw new Error("fake pi: getFlag");
	}
	getMcpServers(): never {
		throw new Error("fake pi: getMcpServers");
	}
	getSessionName(): never {
		throw new Error("fake pi: getSessionName");
	}
	getSettings(): never {
		throw new Error("fake pi: getSettings");
	}
	getThinkingLevel(): never {
		throw new Error("fake pi: getThinkingLevel");
	}
	registerCommand(): never {
		throw new Error("fake pi: registerCommand");
	}
	registerEntryRenderer(): never {
		throw new Error("fake pi: registerEntryRenderer");
	}
	registerFlag(): never {
		throw new Error("fake pi: registerFlag");
	}
	registerMarkdownTransformer(): never {
		throw new Error("fake pi: registerMarkdownTransformer");
	}
	registerMcpServer(): never {
		throw new Error("fake pi: registerMcpServer");
	}
	registerMessageRenderer(): never {
		throw new Error("fake pi: registerMessageRenderer");
	}
	registerProvider(): never {
		throw new Error("fake pi: registerProvider");
	}
	registerShortcut(): never {
		throw new Error("fake pi: registerShortcut");
	}
	registerToolRenderer(): never {
		throw new Error("fake pi: registerToolRenderer");
	}
	registerVirtualModel(): never {
		throw new Error("fake pi: registerVirtualModel");
	}
	sendMessage(): never {
		throw new Error("fake pi: sendMessage");
	}
	setActiveTools(): never {
		throw new Error("fake pi: setActiveTools");
	}
	setLabel(): never {
		throw new Error("fake pi: setLabel");
	}
	setModel(): never {
		throw new Error("fake pi: setModel");
	}
	setSessionName(): never {
		throw new Error("fake pi: setSessionName");
	}
	setThinkingLevel(): never {
		throw new Error("fake pi: setThinkingLevel");
	}
	unregisterMcpServer(): never {
		throw new Error("fake pi: unregisterMcpServer");
	}
	unregisterProvider(): never {
		throw new Error("fake pi: unregisterProvider");
	}
	unregisterVirtualModel(): never {
		throw new Error("fake pi: unregisterVirtualModel");
	}
	events = {
		emit: (): void => {},
		on: (): (() => void) => () => {},
	};

	shutdown() {
		this.shutdowns += 1;
	}

	async emit(event: string, extra: Record<string, unknown> = {}) {
		const ctx = this.context();
		for (const handler of this.handlers.get(event) ?? []) {
			await handler({ type: event, ...extra }, ctx);
		}
	}

	context(): ExtensionContext {
		return fakeContext(this, this.session);
	}

	toolContext(): ExtensionToolContext {
		return this.context() as ExtensionToolContext;
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

	// wake parked reads without landing anything
	wake() {
		for (const resolve of this.parked.splice(0)) resolve();
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
			const guard = "/tmp/saccade-fake-guard-" + body.native_id + ".guard";
			const known = this.attachments.get(body.native_id);
			if (known) {
				return this.reply({ registered: false, ...known, guard });
			}
			const fresh = { incarnation: this.attachments.size, num: this.attachments.size };
			this.attachments.set(body.native_id, fresh);
			return this.reply({ registered: true, ...fresh, guard });
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
	tools: (name: string) => ToolDefinition;
	ctx: () => ExtensionContext;
}

// The context the fake hands to handlers and tools: the faithful
// ExtensionContext surface (typed against the pinned SDK), with the
// members the tests control wired and everything else refusing.
function fakeContext(pi: FakePi, session: FakeSession): ExtensionContext {
	const unimplemented = (name: string) => (): never => {
		throw new Error(`fake context: ${name} is not part of this test`);
	};
	return {
		sessionManager: {
			getCwd: () => "/",
			getSessionDir: () => "/sessions",
			getSessionId: () => session.nativeId,
			getSessionFile: () => session.file,
			getLeafId: () => null,
			getLeafEntry: () => undefined,
			getEntry: () => undefined,
			getLabel: () => undefined,
			getBranch: () => [],
			buildContextEntries: () => [],
			buildSessionProjection: () => unimplemented("buildSessionProjection")(),
			getHeader: () => null,
			getEntries: () => session.entries as unknown as SessionEntry[],
			getTree: () => unimplemented("getTree")(),
			getSessionName: () => undefined,
		},
		ui: {
			select: () => unimplemented("ui.select")(),
			confirm: () => unimplemented("ui.confirm")(),
			input: () => unimplemented("ui.input")(),
			notify: () => unimplemented("ui.notify")(),
			onTerminalInput: () => unimplemented("ui.onTerminalInput")(),
			setStatus: () => unimplemented("ui.setStatus")(),
			setWorkingMessage: () => unimplemented("ui.setWorkingMessage")(),
			setWorkingVisible: () => unimplemented("ui.setWorkingVisible")(),
			setWorkingIndicator: () => unimplemented("ui.setWorkingIndicator")(),
			setHiddenThinkingLabel: () =>
				unimplemented("ui.setHiddenThinkingLabel")(),
			setWidget: () => unimplemented("ui.setWidget")(),
			setFooter: () => unimplemented("ui.setFooter")(),
			setHeader: () => unimplemented("ui.setHeader")(),
			setTitle: () => unimplemented("ui.setTitle")(),
			custom: () => unimplemented("ui.custom")(),
			pasteToEditor: () => unimplemented("ui.pasteToEditor")(),
			setEditorText: () => unimplemented("ui.setEditorText")(),
			getEditorText: () => unimplemented("ui.getEditorText")(),
			editor: () => unimplemented("ui.editor")(),
			addAutocompleteProvider: () =>
				unimplemented("ui.addAutocompleteProvider")(),
			setEditorComponent: () => unimplemented("ui.setEditorComponent")(),
			getEditorComponent: () => unimplemented("ui.getEditorComponent")(),
			theme: undefined as unknown as Theme,
			getAllThemes: () => unimplemented("ui.getAllThemes")(),
			getTheme: () => unimplemented("ui.getTheme")(),
			setTheme: () => unimplemented("ui.setTheme")(),
			getToolsExpanded: () => unimplemented("ui.getToolsExpanded")(),
			setToolsExpanded: () => unimplemented("ui.setToolsExpanded")(),
		},
		mode: "tui",
		hasUI: true,
		cwd: "/",
		modelRegistry: {} as never,
		model: undefined,
		scopedModels: [],
		getSystemPrompt: () => "",
		isIdle: () => pi.idle,
		isProjectTrusted: () => true,
		signal: undefined,
		abort: () => unimplemented("abort")(),
		hasPendingMessages: () => pi.pendingNative,
		shutdown: () => {
			pi.shutdowns += 1;
		},
		getContextUsage: () => undefined,
		compact: () => unimplemented("compact")(),
	};
}

function harness(
	nativeId: string,
	file: string,
	actor?: string,
	shared?: FakeTracker,
): Harness {
	if (actor === undefined) {
		delete process.env.SACCADE_ACTOR;
	} else {
		process.env.SACCADE_ACTOR = actor;
	}
	const pi = new FakePi(nativeId, file);
	saccadeIntegration(pi);
	const tracker = shared ?? new FakeTracker();
	tracker.install();
	process.env.SACCADE_SERVER = tracker.url();
	return {
		pi,
		tracker,
		tools: (name: string) => {
			const def = pi.tools.get(name);
			assert.ok(def, `the fake registered ${name}`);
			return def;
		},
		ctx: () => fakeContext(pi, pi.session),
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

// The text of a tool result: results arrive typed as the SDK's
// AgentToolResult and this is the one place their content is unpacked.
function resultText(result: unknown): string {
	const r = result as { content: Array<{ type: string; text: string }> };
	return r.content
		.filter((c) => c.type === "text")
		.map((c) => c.text)
		.join("");
}

// End a harness's session: loops stop, guards release, and no leaked
// activity from one test reaches another through the global fetch.
async function stop(h: Harness): Promise<void> {
	await h.pi.emit("session_shutdown");
}

test("a released wait queues and returns; the submission confirms from saved history", async () => {
	const h = harness("sac-test-01", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-01", me);
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
	await stop(h);
});

test("the wait tool returns while the queue is still undrained", async () => {
	const h = harness("sac-test-02", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-02", me);
	await h.pi.emit("session_start");
	h.tracker.mailboxes.set(0, [mail({ seq: 51, num: 9, to: { seq: 0, num: 0 } })]);

	const wait = h.tools("saccade_wait");
	const controller = new AbortController();
	const result = await boundary(
		wait.execute("t1", {}, controller.signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) }),
		8000,
		"saccade_wait",
	);
	// the tool returned before any queue processing
	assert.ok(resultText(result).includes("received m-9"));
	assert.equal(h.pi.followUps.length, 1);
	assert.equal(h.tracker.submissions.length, 0);
	// and the queued delivery still confirms after the run
	await h.pi.runQueue();
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 51 }]);
	await stop(h);
});

test("the wait finishes cancelled when the call aborts, queueing nothing", async () => {
	const h = harness("sac-test-03", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-03", me);
	await h.pi.emit("session_start");

	const wait = h.tools("saccade_wait");
	const controller = new AbortController();
	const running = wait.execute(
		"t1",
		{},
		controller.signal,
		undefined,
		{ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
	);
	controller.abort();
	const result = await boundary(running, 8000, "aborted saccade_wait");
	assert.ok(
		resultText(result).includes("the call was cancelled"),
	);
	assert.equal(h.pi.followUps.length, 0);
	await h.pi.runQueue();
	await tick();
	assert.equal(h.tracker.submissions.length, 0);
	await stop(h);
});

test("reassociation cancels a queued delivery: no stale recipient submits", async () => {
	const h = harness("sac-test-04", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-04", me);
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
	await stop(h);
});

test("agent possession survives a tracker outage; injected actors win", async () => {
	const h = harness("sac-test-05", "/sessions/aaaa1111.jsonl");
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
	await stop(h);
});

test("a running registered peer receives without any wait tool", async () => {
	const h = harness("sac-test-06", "/sessions/aaaa1111.jsonl");
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
	await stop(h);
});

test("background arrival and the wait tool queue a message once", async () => {
	const h = harness("sac-test-07", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.tracker.land(0, mail({ seq: 61, num: 6, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 1);
	// the wait tool sees the same pending message and releases, but
	// its own queueing is a no-op: one envelope, one path
	const wait = h.tools("saccade_wait");
	const result = await boundary(
		wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) }),
		8000,
		"saccade_wait",
	);
	assert.ok(resultText(result).includes("received m-6"));
	assert.equal(h.pi.followUps.length, 1);
	await stop(h);
});

test("a failed bridge read reconnects and keeps delivering", async () => {
	const h = harness("sac-test-08", "/sessions/aaaa1111.jsonl");
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
	await stop(h);
});

test("session shutdown stops the bridge: nothing delivers after it", async () => {
	const h = harness("sac-test-09", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	await h.pi.emit("session_shutdown");
	assert.equal(h.pi.followUps.length, 0);
	h.tracker.land(0, mail({ seq: 64, num: 9, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 0);
	assert.equal(h.tracker.submissions.length, 0);
});

test("reassociation strands the old bridge: no stale delivery, no stale submit", async () => {
	const h = harness("sac-test-10", "/sessions/aaaa1111.jsonl");
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
	await stop(h);
});

test("an extension-source input releases the wait as the message, not as human input", async () => {
	const h = harness("sac-test-11", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	const wait = h.tools("saccade_wait");
	const running = wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) });
	h.tracker.land(0, mail({ seq: 70, num: 4, to: { seq: 0, num: 0 } }));
	await tick();
	// the queued envelope enters the session as an extension-source input
	const envelope = h.pi.followUps[0];
	assert.ok(envelope, "the bridge queued the delivery");
	await h.pi.emit("input", { source: "extension", text: envelope });
	const result = await boundary(running, 8000, "saccade_wait");
	const text = resultText(result);
	assert.ok(text.includes("received m-4"), text);
	assert.ok(!text.includes("human"), `mislabelled as human: ${text}`);
	await stop(h);
});

test("interactive input keeps the human wording and no correlation", async () => {
	const h = harness("sac-test-12", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	const wait = h.tools("saccade_wait");
	const running = wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) });
	await tick();
	await h.pi.emit("input", { source: "interactive", text: "typing over it" });
	const result = await boundary(running, 8000, "saccade_wait");
	const text = resultText(result);
	assert.ok(text.includes("native human input arrived"), text);
	assert.equal(h.pi.followUps.length, 0);
	await stop(h);
});

test("a refused native send recovers at the human-input boundary", async () => {
	const h = harness("sac-test-13", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.pi.failNextSend = 1;
	h.tracker.land(0, mail({ seq: 71, num: 5, to: { seq: 0, num: 0 } }));
	await tick();
	await tick();
	// the refusal is invisible to the extension (void + error sink):
	// nothing queued, and no promise ever resolved or rejected here
	assert.equal(h.pi.followUps.length, 0, "the refused send queued nothing");
	assert.ok(h.pi.errors.length > 0, "the runtime saw the failure");
	// wakes alone cannot deliver: the guard honestly holds, because
	// no supported observation says the delivery was lost
	h.tracker.wake();
	await tick();
	await tick();
	assert.equal(h.pi.followUps.length, 0);
	// the human correction is the recovery boundary: the session is
	// idle with nothing natively pending, the input event reconciles,
	// and the boundary's own read delivers again
	h.pi.idle = true;
	h.pi.pendingNative = false;
	await h.pi.emit("input", { source: "interactive", text: "fixed the auth" });
	const deadline = Date.now() + 8000;
	while (h.pi.followUps.length === 0 && Date.now() < deadline) {
		h.tracker.wake();
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	assert.equal(h.pi.followUps.length, 1, "the reconciled guard re-delivered");
	assert.equal(h.tracker.submissions.length, 0);
	await stop(h);
});

test("a discarded queued delivery reconciles at an idle settle and redelivers", async () => {
	const h = harness("sac-test-14", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.tracker.land(0, mail({ seq: 72, num: 6, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.followUps.length, 1);
	// user interruption discarded the queued input: idle, nothing
	// natively pending, and the envelope never reached saved history
	h.pi.idle = true;
	h.pi.pendingNative = false;
	await h.pi.emit("agent_settled");
	await tick();
	assert.equal(h.tracker.submissions.length, 0);
	// the released guard lets the next read deliver again, after the
	// settle cadence
	h.tracker.wake();
	const redeliver = Date.now() + 8000;
	while (h.pi.followUps.length < 2 && Date.now() < redeliver) {
		h.tracker.wake();
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	assert.equal(h.pi.followUps.length, 2);
	// and the redelivered envelope still confirms from saved history
	h.pi.session.entries.push({
		type: "message",
		message: { role: "user", content: h.pi.followUps[1] },
	});
	await h.pi.emit("agent_settled");
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 72 }]);
	await stop(h);
});

test("missing-auth failure with no settle recovers at an explicit wait", async () => {
	const h = harness("sac-test-15", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.pi.failNextSend = 1;
	h.tracker.land(0, mail({ seq: 73, num: 7, to: { seq: 0, num: 0 } }));
	await tick();
	await tick();
	// no agent_settled ever fires - the send failed before any run.
	// The explicit wait is the recovery boundary: it reconciles, then
	// its own first read delivers
	const wait = h.tools("saccade_wait");
	h.pi.idle = true;
	h.pi.pendingNative = false;
	const running = wait.execute("t1", {}, new AbortController().signal, undefined, {
		...h.ctx(),
		tools: [],
		executeTool: () => Promise.resolve({} as never),
	});
	h.tracker.wake();
	const deadline = Date.now() + 8000;
	while (h.pi.followUps.length === 0 && Date.now() < deadline) {
		h.tracker.wake();
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	assert.equal(h.pi.followUps.length, 1, "the explicit wait re-delivered");
	const result = await boundary(running, 8000, "saccade_wait");
	assert.ok(resultText(result).includes("received m-7"));
	await stop(h);
});

test("another extension's input yields our wait with generic native wording", async () => {
	const h = harness("sac-test-16", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	const wait = h.tools("saccade_wait");
	const running = wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) });
	await tick();
	// not our envelope: another native extension's message
	await h.pi.emit("input", {
		source: "extension",
		text: "[some other extension] probe",
	});
	const result = await boundary(running, 8000, "saccade_wait");
	const text = resultText(result);
	assert.ok(text.includes("native input arrived through the integration"), text);
	assert.ok(!text.includes("human"), `fabricated human origin: ${text}`);
	assert.ok(!text.includes("received m-"), `fabricated correlation: ${text}`);
	assert.equal(h.pi.followUps.length, 0);
	await stop(h);
});

test("the attach carries the launch token it was launched with", async () => {
	process.env.SACCADE_LAUNCH_TOKEN = "launch-abc123";
	const h = harness("sac-test-17", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.tracker.attachBodies.length, 1);
	assert.equal(h.tracker.attachBodies[0].launch_token, "launch-abc123");
	delete process.env.SACCADE_LAUNCH_TOKEN;
	await stop(h);
});

test("the declared task anchors scope; an explicit task argument is message context", async () => {
	process.env.SACCADE_ACTOR = "pi";
	process.env.SACCADE_TASK = "t-9";
	delete process.env.SACCADE_WORKSPACE;
	const h = harness("sac-test-18", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-18", me);
	await h.pi.emit("session_start");

	const send = h.tools("saccade_send");
	await send.execute(
		"t1",
		{ to: "i-1", body: "the answer", task: "t-2" },
		undefined,
		undefined,
		{ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
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
		undefined,
		{ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
	);
	const second = h.tracker.sent[h.tracker.sent.length - 1];
	assert.equal(second.context.scope.task, null);
	assert.equal(second.task, 2);
	delete process.env.SACCADE_TASK;
	await stop(h);
});

test("the guard take is kernel-real: held while the fd lives, free after death", async () => {
	// the EXACT production helper, against the kernel: retain the fd ->
	// an independent probe must fail; close it -> the probe must succeed
	const guard = "/tmp/saccade-guard-kernel-" + process.pid + ".guard";
	const fd = openSync(guard, "a+");
	let closed = false;
	try {
		assert.equal(takeGuardLock(fd), true, "the production take succeeds on a free guard");
		// an independent process probes with its own descriptor: its
		// flock must fail while the holder lives
		const probeFd = openSync(guard, "a+");
		const heldProbe = spawnSync("flock", ["-n", "3"], {
			stdio: ["ignore", "ignore", "ignore", probeFd],
		});
		closeSync(probeFd);
		assert.notEqual(heldProbe.status, 0, "an independent probe sees the lock held");
		closeSync(fd);
		closed = true;
		const freeFd = openSync(guard, "a+");
		const freeProbe = spawnSync("flock", ["-n", "3"], {
			stdio: ["ignore", "ignore", "ignore", freeFd],
		});
		closeSync(freeFd);
		assert.equal(freeProbe.status, 0, "death releases the lock");
		// and the fd-form lock never materializes a file named 3
		assert.ok(!existsSync("3"), "no file named 3 may appear");
	} finally {
		if (!closed) closeSync(fd);
		rmSync(guard, { force: true });
		rmSync("3", { force: true });
	}
});

test("the guard take classifies by the documented flock contract", () => {
	const guard = "/tmp/saccade-guard-classify-" + process.pid + ".guard";
	const fd = openSync(guard, "a+");
	const shim = mkdtempSync("/tmp/saccade-flock-shim-");
	const savedPath = process.env.PATH;
	try {
		// exit 1 is the documented held-lock conflict: busy
		writeFileSync(join(shim, "flock"), "#!/bin/sh\nexit 1\n");
		chmodSync(join(shim, "flock"), 0o755);
		process.env.PATH = shim;
		assert.equal(takeGuardLock(fd), false, "exit 1 is the busy conflict");
		// every other exit, signal death, and spawn failure stay unknown
		writeFileSync(join(shim, "flock"), "#!/bin/sh\nexit 16\n");
		assert.equal(takeGuardLock(fd), undefined, "exit 16 is unknown");
		writeFileSync(join(shim, "flock"), "#!/bin/sh\nkill -TERM $$\n");
		assert.equal(takeGuardLock(fd), undefined, "signal death is unknown");
		process.env.PATH = "";
		assert.equal(takeGuardLock(fd), undefined, "a missing flock is unknown");
	} finally {
		if (savedPath === undefined) delete process.env.PATH;
		else process.env.PATH = savedPath;
		closeSync(fd);
		rmSync(guard, { force: true });
		rmSync(shim, { recursive: true, force: true });
	}
});

test("a held guard shuts the duplicate down before any model work", async () => {
	const shared = new FakeTracker();
	// the original attaches and holds its session's guard
	const original = harness("dup-original0", "/sessions/dup-original.jsonl", "pi", shared);
	await original.pi.emit("session_start");
	assert.equal(original.pi.shutdowns, 0);
	// a real duplicate start over the same native session: the held
	// guard is detected, the duplicate names its refusal, and pi is
	// shut down before any model work - never a silent null association
	const duplicate = harness("dup-original0", "/sessions/dup-original.jsonl", "pi", shared);
	await duplicate.pi.emit("session_start");
	assert.equal(duplicate.pi.shutdowns, 1, "the duplicate shuts down");
	// the duplicate never participated: its identity names the held
	// guard, and nothing registered
	const identity = duplicate.tools("saccade_identity");
	const result = await identity.execute(
		"t1",
		{},
		undefined,
		undefined,
		{ ...duplicate.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
	);
	const text = resultText(result);
	assert.ok(text.includes("not registered"), text);
	assert.ok(text.includes("duplicate"), `the refusal names the held guard: ${text}`);
	// the original is untouched and still holds its guard
	assert.equal(original.pi.followUps.length, 0);
	await stop(original);
	await stop(duplicate);
});

test("an unknown guard take refuses the association and shuts the session down", async () => {
	const shared = new FakeTracker();
	// a flock that fails outside the documented contract: the take is
	// unknown, and an unverified session must not keep running unguarded
	const shim = mkdtempSync("/tmp/saccade-flock-shim-");
	writeFileSync(join(shim, "flock"), "#!/bin/sh\nexit 16\n");
	chmodSync(join(shim, "flock"), 0o755);
	const savedPath = process.env.PATH;
	process.env.PATH = shim;
	let h: ReturnType<typeof harness>;
	try {
		h = harness("unknown-guard0", "/sessions/unknown-guard.jsonl", "pi", shared);
		await h.pi.emit("session_start");
		assert.equal(h.pi.shutdowns, 1, "the unverified session shuts down");
		// still inside the broken-flock window: a retried attach meets
		// the same unknown and refuses again - never a live association
		const identity = h.tools("saccade_identity");
		const result = await identity.execute(
			"t1",
			{},
			undefined,
			undefined,
			{ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
		);
		const text = resultText(result);
		assert.ok(text.includes("not registered"), text);
		assert.ok(text.includes("cannot be probed"), `the refusal names the unknown: ${text}`);
		assert.equal(h.pi.shutdowns, 2, "every attempt refuses and shuts down");
	} finally {
		if (savedPath === undefined) delete process.env.PATH;
		else process.env.PATH = savedPath;
	}
	await stop(h);
	rmSync(shim, { recursive: true, force: true });
});
