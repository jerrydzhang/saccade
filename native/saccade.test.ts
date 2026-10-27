// The integration's mechanics and queue contract, pinned against a stateful fake of the native steer queue.
import assert from "node:assert/strict";
import test from "node:test";
import { join } from "node:path";
import {
	chmodSync,
	closeSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	openSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import type {
	CustomEntry,
	EntryRenderer,
	ExtensionAPI,
	ExtensionContext,
	ExtensionToolContext,
	SessionEntry,
	Theme,
	ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import {
	envelopeText,
	handoffEnvelope,
	handoffHelperScript,
	headerCwd,
	incomingFor,
	numToken,
	takeGuardLock,
	userText,
	validJobsSnapshot,
	type HandoffDoneData,
	type HandoffEntryData,
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

test("arrival filters on the addressee and the confirmed submission, self-addressed included", () => {
	const incoming = [
		// a confirmed arrival to me
		mail({ seq: 1, state: "submitted", to: { seq: 0, num: 0 } }),
		// outgoing archive: mine as sender, not mine to receive
		mail({ seq: 2, state: "submitted", from: { seq: 0, num: 0, actor: "me" }, to: { seq: 1, num: 1 } }),
		// recorded without a confirmed submission: evidence, never an
		// arrival
		mail({ seq: 3, state: "pending", to: { seq: 0, num: 0 } }),
		// another incarnation's mail entirely
		mail({ seq: 4, state: "submitted", to: { seq: 2, num: 2 } }),
		// self-addressed: from me and to me is still incoming mail
		mail({ seq: 5, state: "submitted", from: { seq: 0, num: 0, actor: "me" }, to: { seq: 0, num: 0 } }),
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
	file: string | undefined;
	constructor(nativeId: string, file: string | undefined) {
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

// The fake queue drains when the test fires the next turn boundary, events before the append.
class FakePi implements ExtensionAPI {
	tools = new Map<string, ToolDefinition>();
	handlers = new Map<string, Array<(event: any, ctx: any) => unknown>>();
	steers: string[] = [];
	// the deliverAs mode each accepted send carried, in push order
	modes: Array<"steer" | "followUp" | undefined> = [];
	session: FakeSession;
	// the next sendUserMessage is refused before queue or history
	failNextSend = 0;
	// the honest resting state: a session nobody is running
	idle = true;
	pendingNative = false;
	// when set, an accepted input stays unobservable: the queue holds
	// nothing the acceptance probes can see, so the confirmation waits
	// for the later boundaries - the held-path seam
	silentAccept = false;

	hasUI = true;
	constructor(nativeId: string, file: string | undefined) {
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
	sendUserMessage(
		content: string,
		opts?: { deliverAs?: "steer" | "followUp"; expandPromptTemplates?: boolean },
	) {
		if (this.failNextSend > 0) {
			this.failNextSend -= 1;
			this.errors.push("send_user_message: no auth: the provider refused");
			return;
		}
		this.steers.push(content);
		this.modes.push(opts?.deliverAs);
		// the runtime's own truth: an accepted input is either pending
		// natively or already consuming a run
		if (!this.silentAccept) {
			this.pendingNative = true;
			this.idle = false;
		}
	}
	appendEntries: Array<{ type: string; data: unknown }> = [];
	// when set, appendEntry throws for this custom type: the persistence
	// failure seam
	failAppend: string | null = null;
	// components the registered renderer painted, in append order
	rendered: unknown[] = [];
	entryRenderers = new Map<string, EntryRenderer>();
	registerEntryRenderer<T = unknown>(
		customType: string,
		renderer: EntryRenderer<T>,
	): void {
		this.entryRenderers.set(customType, renderer as EntryRenderer);
	}
	appendEntry<T = unknown>(customType: string, data?: T): void {
		if (this.failAppend === customType) {
			throw new Error("session storage refused the append");
		}
		this.appendEntries.push({ type: customType, data });
		// faithful to interactive mode: a registered renderer paints the
		// entry into chat; without one the append stays invisible
		const renderer = this.entryRenderers.get(customType);
		if (renderer) {
			this.rendered.push(
				renderer(
					{ type: "custom", customType, data } as CustomEntry<T>,
					{ expanded: false },
					undefined as unknown as Theme,
				),
			);
		}
	}
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
	// the shared event bus under pi.events, with every emission recorded
	busHandlers = new Map<string, Array<(data: any) => void>>();
	busEmitted: Array<{ channel: string; payload: any }> = [];
	events = {
		emit: (channel: string, data: any) => {
			this.busEmitted.push({ channel, payload: data });
			for (const handler of this.busHandlers.get(channel) ?? []) handler(data);
		},
		on: (channel: string, handler: (data: any) => void) => {
			const list = this.busHandlers.get(channel) ?? [];
			list.push(handler);
			this.busHandlers.set(channel, list);
			return () => {
				const at = list.indexOf(handler);
				if (at >= 0) list.splice(at, 1);
			};
		},
	} as unknown as ExtensionAPI["events"];

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
		const queued = this.steers.splice(0);
		this.pendingNative = false;
		this.idle = true;
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
	// the sender attempts' staged tasks, handed exactly once by the
	// transport read - the wire's stage registry, mirrored
	stages = new Map<number, any[]>();
	submissions: Array<{ incarnation: number; message: number }> = [];
	sent: any[] = [];
	attachBodies: any[] = [];
	attachFails = false;
	// guard paths are pure in the native id within one tracker; the
	// salt keeps different tests' retained guards from colliding
	guardSalt = Math.random().toString(36).slice(2);
	failNext = 0;
	// parked bridge reads: held until a record lands or the test wakes them
	parked: Array<() => void> = [];
	// the records fold the /api/v1/records read returns, and the path the
	// next provision births; payloads travel as the JSON itself
	records: Array<{ seq: number; kind: string; payload: unknown }> = [];
	creates: any[] = [];
	createPaths: string[] = [];
	// holds the next workspace create mid-flight, like the parked reads
	private createGate: { open: () => void; promise: Promise<void> } | null = null;

	holdNextCreate(): void {
		let open!: () => void;
		const promise = new Promise<void>((resolve) => {
			open = resolve;
		});
		this.createGate = { open, promise };
	}

	releaseCreate(): void {
		const gate = this.createGate;
		this.createGate = null;
		gate?.open();
	}

	url() {
		return "http://tracker.test";
	}

	private reply(body: unknown, status = 200) {
		return { status, json: async () => body };
	}

	// wake parked transport reads without staging anything
	wake() {
		for (const resolve of this.parked.splice(0)) resolve();
	}

	// a landed record joins the mailbox: a descriptive read only, and
	// no machinery delivers it - arrivals release waits, nothing more
	land(incarnation: number, message: any) {
		const box = this.mailboxes.get(incarnation) ?? [];
		box.push(message);
		this.mailboxes.set(incarnation, box);
	}

	// one sender attempt's staged task, woken immediately: the transport
	// half of record-then-attempt
	stage(incarnation: number, message: any) {
		const box = this.stages.get(incarnation) ?? [];
		box.push({ token: `tok-${message.seq}-${box.length}`, message });
		this.stages.set(incarnation, box);
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
			const guard =
				"/tmp/saccade-fake-guard-" + this.guardSalt + "-" + body.native_id + ".guard";
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
			return this.reply({ messages: this.mailboxes.get(incarnation) ?? [] });
		}
		if (path === "/api/v1/messages/deliver") {
			const url = new URL(String(rawUrl));
			const incarnation = Number(url.searchParams.get("incarnation"));
			const waitMs = Number(url.searchParams.get("wait_ms") ?? 0);
			const taken = (this.stages.get(incarnation) ?? []).splice(0);
			this.stages.set(incarnation, []);
			if (taken.length === 0 && waitMs > 0) {
				// the parked transport read: held until a task is staged
				await new Promise<void>((resolve) => this.parked.push(resolve));
				// a woken read can still fail mid-cycle, like any
				// transport: the stage survives for the reconnected read
				if (this.failNext > 0) {
					this.failNext -= 1;
					throw new Error("tracker unreachable");
				}
				const again = (this.stages.get(incarnation) ?? []).splice(0);
				this.stages.set(incarnation, []);
				return this.reply({ deliveries: again });
			}
			return this.reply({ deliveries: taken });
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
		if (path === "/api/v1/records") {
			return this.reply({ records: this.records });
		}
		if (path === "/api/v1/workspace/create") {
			this.creates.push(body);
			const seq = 300 + this.creates.length;
			if (this.createGate !== null) {
				const gate = this.createGate;
				await gate.promise;
			}
			return this.reply({
				created: true,
				workspace: {
					seq,
					base: "ba5eba11",
					branch: `scratch-branch-${seq}`,
					path: this.createPaths.shift() ?? "/nonexistent-saccade-provision",
				},
			});
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
		hasUI: pi.hasUI,
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
	file: string | undefined,
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

// A reloaded runtime runs a FRESH factory over the same tracker. The
// guard registry is process-global, so the fresh instance adopts the
// held exact-session guard instead of refusing as a duplicate.
function reloadedHarness(
	h: Harness,
	nativeId: string,
	file: string | undefined,
): Harness {
	const pi = new FakePi(nativeId, file);
	saccadeIntegration(pi);
	return {
		pi,
		tracker: h.tracker,
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

// An independent kernel probe of a guard file: true when the lock is
// held by some live descriptor, as activation would see it.
function probeHeld(guardPath: string): boolean {
	const fd = openSync(guardPath, "a+");
	const probe = spawnSync("flock", ["-n", "3"], {
		stdio: ["ignore", "ignore", "ignore", fd],
	});
	closeSync(fd);
	return probe.status !== 0;
}

// End a harness's session: loops stop, queued work dies, and no
// leaked activity from one test reaches another through the global
// fetch. The guard rides the process: the fake tracker's salted paths
// keep one test's retained lock from reaching another.
async function stop(h: Harness): Promise<void> {
	await h.pi.emit("session_shutdown");
}

test("a start or resume drains nothing: recorded pending is evidence, never backlog", async () => {
	const h = harness("sac-test-25", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-25", me);
	// recorded mail without a confirmed submission sits in the mailbox
	h.tracker.land(0, mail({ seq: 80, num: 9, state: "pending", to: { seq: 0, num: 0 } }));
	h.tracker.land(0, mail({ seq: 81, num: 10, state: "pending", to: { seq: 0, num: 0 } }));
	await h.pi.emit("session_start");
	await tick();
	await tick();
	// the start queued nothing and confirmed nothing: no machinery
	// carries a recorded-but-unsubmitted message, on resume or ever
	assert.equal(h.pi.steers.length, 0);
	assert.equal(h.tracker.submissions.length, 0);
	// and a later boundary never revisits it
	await h.pi.emit("agent_settled");
	await tick();
	assert.equal(h.pi.steers.length, 0);
	assert.equal(h.tracker.submissions.length, 0);
	await stop(h);
});

test("a staged delivery rides the steer queue, so it lands at the next turn boundary", async () => {
	const h = harness("sac-test-24", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-24", me);
	await h.pi.emit("session_start");
	// the sender's attempt stages its task; the transport hands it once
	h.tracker.stage(0, mail({ seq: 60, to: { seq: 0, num: 0 } }));
	await tick();
	// the assertion pins the delivery mode: steer at the next turn boundary, never the run-end followUp
	assert.deepEqual(h.pi.modes, ["steer"]);
	// the queue's acceptance confirmed the sender's attempt
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 60 }]);
	await stop(h);
});

test("a released wait reports confirmed arrivals and queues nothing", async () => {
	const h = harness("sac-test-01", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-01", me);
	await h.pi.emit("session_start");
	// the arrival already rode its sender's submission into the queue:
	// a submitted record the mailbox carries
	h.tracker.land(0, mail({ seq: 50, state: "submitted", to: { seq: 0, num: 0 } }));
	const wait = h.tools("saccade_wait");
	const result = await boundary(
		wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) }),
		8000,
		"saccade_wait",
	);
	assert.ok(resultText(result).includes("received m-3"), resultText(result));
	// the wait only reports: the sender's attempt did the queueing
	assert.equal(h.pi.steers.length, 0);
	assert.equal(h.tracker.submissions.length, 0);
	// and a pending record never releases a wait: it is evidence
	h.tracker.land(0, mail({ seq: 51, num: 4, state: "pending", to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 0);
	await stop(h);
});

test("the attempt confirms at queue acceptance, before any turn boundary", async () => {
	const h = harness("sac-test-02", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-02", me);
	await h.pi.emit("session_start");
	h.tracker.stage(0, mail({ seq: 51, num: 9, to: { seq: 0, num: 0 } }));
	await tick();
	// the queue accepted the input: the submission is already recorded,
	// with no boundary fired and nothing inserted into history
	assert.equal(h.pi.steers.length, 1);
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 51 }]);
	assert.equal(h.pi.session.entries.length, 0, "no saved history yet");
	// the later run confirms nothing twice
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
	assert.equal(h.pi.steers.length, 0);
	await h.pi.runQueue();
	await tick();
	assert.equal(h.tracker.submissions.length, 0);
	await stop(h);
});

test("reassociation cancels a held delivery: no stale recipient submits", async () => {
	const h = harness("sac-test-04", "/sessions/aaaa1111.jsonl");
	const me = { incarnation: 0, num: 0 };
	h.tracker.attachments.set("sac-test-04", me);
	await h.pi.emit("session_start");
	// the queue holds the input, but no acceptance evidence is
	// observable: the confirmation stays held
	h.pi.silentAccept = true;
	h.tracker.stage(0, mail({ seq: 52, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 1);
	assert.equal(h.tracker.submissions.length, 0);

	// a new native session cancels the old association's held delivery
	h.pi.session = new FakeSession("bbbb2222", "/sessions/bbbb2222.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.tracker.attachments.size, 2);

	// a later boundary never submits the cancelled delivery
	h.pi.steers = [];
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
	assert.equal(h.pi.steers.length, 0);
	// the bridge holds a parked transport read; the sender's staged
	// task wakes it
	h.tracker.stage(0, mail({ seq: 60, num: 5, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 1);
	assert.ok(h.pi.steers[0].includes("[saccade message m-5"));
	// the queue's acceptance confirmed the attempt with no boundary
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 60 }]);
	await h.pi.runQueue();
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 60 }]);
	await stop(h);
});

test("a staged delivery queues once and the wait never queues a copy", async () => {
	const h = harness("sac-test-07", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.tracker.stage(0, mail({ seq: 61, num: 6, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 1);
	// the confirmed arrival releases the wait, whose report queues
	// nothing: one envelope, one path
	h.tracker.land(0, mail({ seq: 61, num: 6, state: "submitted", to: { seq: 0, num: 0 } }));
	const wait = h.tools("saccade_wait");
	const result = await boundary(
		wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) }),
		8000,
		"saccade_wait",
	);
	assert.ok(resultText(result).includes("received m-6"));
	assert.equal(h.pi.steers.length, 1);
	await stop(h);
});

test("a failed transport read reconnects and the stage still delivers", async () => {
	const h = harness("sac-test-08", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	// two deliveries ride woken parked reads
	h.tracker.stage(0, mail({ seq: 62, num: 7, to: { seq: 0, num: 0 } }));
	await tick();
	h.tracker.stage(0, mail({ seq: 63, num: 8, to: { seq: 0, num: 0 } }));
	await tick();
	await tick();
	assert.equal(h.pi.steers.length, 2);
	assert.deepEqual(h.tracker.submissions, [
		{ incarnation: 0, message: 62 },
		{ incarnation: 0, message: 63 },
	]);
	// the loop's next read fails at entry, then the backoff runs: the
	// attempt staged mid-failure still delivers on the reconnected read
	h.tracker.failNext = 1;
	await tick();
	h.tracker.stage(0, mail({ seq: 64, num: 9, to: { seq: 0, num: 0 } }));
	const deadline = Date.now() + 8000;
	while (h.pi.steers.length < 3 && Date.now() < deadline) {
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	assert.equal(h.pi.steers.length, 3, "the reconnected transport delivered");
	await tick();
	await tick();
	assert.deepEqual(h.tracker.submissions, [
		{ incarnation: 0, message: 62 },
		{ incarnation: 0, message: 63 },
		{ incarnation: 0, message: 64 },
	]);
	await stop(h);
});
test("session shutdown stops the bridge: nothing delivers after it", async () => {
	const h = harness("sac-test-09", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	await h.pi.emit("session_shutdown");
	assert.equal(h.pi.steers.length, 0);
	h.tracker.stage(0, mail({ seq: 64, num: 9, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 0);
	assert.equal(h.tracker.submissions.length, 0);
});

test("reassociation strands the old bridge: no stale delivery, no stale submit", async () => {
	const h = harness("sac-test-10", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	// a new native session reassociates; the old bridge's parked read dies
	h.pi.session = new FakeSession("bbbb2222", "/sessions/bbbb2222.jsonl");
	await h.pi.emit("session_start");
	assert.equal(h.tracker.attachments.size, 2);
	// a staged task for the OLD incarnation finds no old bridge
	h.tracker.stage(0, mail({ seq: 65, num: 1, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 0, "the old bridge delivered nothing");
	assert.equal(h.tracker.submissions.length, 0);
	// and the new association still delivers its own
	h.tracker.stage(1, mail({ seq: 66, num: 2, to: { seq: 1, num: 1 } }));
	await tick();
	assert.equal(h.pi.steers.length, 1);
	await stop(h);
});

test("an extension-source input releases the wait as the message, not as human input", async () => {
	const h = harness("sac-test-11", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	const wait = h.tools("saccade_wait");
	const running = wait.execute("t1", {}, new AbortController().signal, undefined, { ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) });
	h.tracker.stage(0, mail({ seq: 70, num: 4, to: { seq: 0, num: 0 } }));
	const queued = Date.now() + 8000;
	while (h.pi.steers.length === 0 && Date.now() < queued) {
		await new Promise((resolve) => setTimeout(resolve, 50));
	}
	// the queued envelope enters the session as an extension-source input
	const envelope = h.pi.steers[0];
	assert.ok(envelope, "the transport queued the delivery");
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
	assert.equal(h.pi.steers.length, 0);
	await stop(h);
});

test("a refused send drops at a boundary: no confirmation, no redelivery", async () => {
	const h = harness("sac-test-13", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.pi.failNextSend = 1;
	h.tracker.stage(0, mail({ seq: 71, num: 5, to: { seq: 0, num: 0 } }));
	await tick();
	await tick();
	// the refusal is invisible to the extension (void + error sink):
	// nothing queued, and no promise ever resolved or rejected here
	assert.equal(h.pi.steers.length, 0, "the refused send queued nothing");
	assert.ok(h.pi.errors.length > 0, "the runtime saw the failure");
	assert.equal(h.tracker.submissions.length, 0, "no evidence, no confirmation");
	// the session is idle with nothing natively pending: the boundary
	// settles the held delivery as refused, and nothing redelivers it
	await h.pi.emit("input", { source: "interactive", text: "fixed the auth" });
	await tick();
	await tick();
	assert.equal(h.pi.steers.length, 0, "a refused delivery never requeues");
	assert.equal(h.tracker.submissions.length, 0);
	await stop(h);
});

test("a confirmed delivery stays submitted when the runtime later discards it", async () => {
	const h = harness("sac-test-14", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	h.tracker.stage(0, mail({ seq: 72, num: 6, to: { seq: 0, num: 0 } }));
	await tick();
	assert.equal(h.pi.steers.length, 1);
	// the queue's acceptance already confirmed the attempt
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 72 }]);
	// the runtime discards the input before any run: idle, nothing
	// pending, and the envelope never reaches saved history
	h.pi.steers = [];
	h.pi.idle = true;
	h.pi.pendingNative = false;
	await h.pi.emit("agent_settled");
	await tick();
	// the record stands - the recipient reads it from the mailbox -
	// and nothing redelivers or retracts
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 72 }]);
	assert.equal(h.pi.steers.length, 0);
	await stop(h);
});

test("unobservable acceptance holds the confirmation; saved evidence confirms later", async () => {
	const h = harness("sac-test-15", "/sessions/aaaa1111.jsonl");
	await h.pi.emit("session_start");
	// the queue took the input, but no acceptance evidence is
	// observable: the attempt is never optimistically confirmed
	h.pi.silentAccept = true;
	h.tracker.stage(0, mail({ seq: 73, num: 7, to: { seq: 0, num: 0 } }));
	await tick();
	await tick();
	assert.equal(h.pi.steers.length, 1);
	assert.equal(h.tracker.submissions.length, 0, "no evidence, no optimistic submit");
	// the envelope lands in saved history, which only an accepted
	// input reaches: the later boundary confirms from that evidence
	h.pi.session.entries.push({
		type: "message",
		message: { role: "user", content: h.pi.steers[0] },
	});
	await h.pi.emit("agent_settled");
	await tick();
	assert.deepEqual(h.tracker.submissions, [{ incarnation: 0, message: 73 }]);
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
	assert.equal(h.pi.steers.length, 0);
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
	const sentResult = await send.execute(
		"t1",
		{ to: "i-1", body: "the answer", task: "t-2" },
		undefined,
		undefined,
		{ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
	);
	// the success face names the queue's acceptance, never a pending promise
	assert.ok(
		resultText(sentResult).includes("accepted into their session's input queue"),
		resultText(sentResult),
	);
	assert.ok(!resultText(sentResult).includes("pending"), resultText(sentResult));
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

test("a transient tracker outage across reload keeps the exact-session guard held", async () => {
	const h = harness("reload-guard0", "/sessions/reload-guard.jsonl", "pi", new FakeTracker());
	const guard =
		"/tmp/saccade-fake-guard-" + h.tracker.guardSalt + "-reload-guard0.guard";
	await h.pi.emit("session_start");
	assert.equal(probeHeld(guard), true, "the first attach holds the session's guard");
	// the real reload sequence: shutdown, then a start whose attach
	// hits a transient tracker outage
	await h.pi.emit("session_shutdown");
	h.tracker.attachFails = true;
	await h.pi.emit("session_start");
	assert.equal(probeHeld(guard), true, "the outage must not release the guard");
	assert.equal(h.pi.shutdowns, 0, "the process is guarded: it stays alive");
	const identity = h.tools("saccade_identity");
	const toolCtx = () => ({ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) });
	const refused = resultText(await identity.execute("t1", {}, undefined, undefined, toolCtx()));
	assert.ok(refused.includes("not registered"), refused);
	// the tracker returns: the retained lock is reused, never retaken
	h.tracker.attachFails = false;
	const ok = resultText(await identity.execute("t2", {}, undefined, undefined, toolCtx()));
	assert.ok(ok.includes("i-0"), ok);
	assert.equal(probeHeld(guard), true, "the retained guard is still the one held");
	assert.equal(h.pi.shutdowns, 0, "no refusal ever fired");
	await stop(h);
});

test("a protocol-launched boot that cannot verify refuses before model work", async () => {
	process.env.SACCADE_LAUNCH_TOKEN = "boot-token";
	const h = harness("launch-outage0", "/sessions/launch-outage.jsonl", "pi", new FakeTracker());
	try {
		h.tracker.attachFails = true;
		await h.pi.emit("session_start");
		assert.equal(h.pi.shutdowns, 1, "the unverified launched boot shuts down");
	} finally {
		delete process.env.SACCADE_LAUNCH_TOKEN;
	}
	await stop(h);
});

test("a reload to a different native session releases the old session's guard", async () => {
	const h = harness("switch-old0", "/sessions/switch-old.jsonl", "pi", new FakeTracker());
	const guardOf = (id: string) =>
		"/tmp/saccade-fake-guard-" + h.tracker.guardSalt + "-" + id + ".guard";
	await h.pi.emit("session_start");
	assert.equal(probeHeld(guardOf("switch-old0")), true, "the old session's guard is held");
	h.pi.session = new FakeSession("switch-new0", "/sessions/switch-new.jsonl");
	await h.pi.emit("session_start");
	assert.equal(
		probeHeld(guardOf("switch-old0")),
		false,
		"the old session's guard released with the identity change",
	);
	assert.equal(probeHeld(guardOf("switch-new0")), true, "the new session's guard is held");
	await stop(h);
});

test("a fresh factory after shutdown adopts the held exact-session guard", async () => {
	// proof level: fresh factory, same module (cached-factory reload path); kernel evidence
	const shared = new FakeTracker();
	const first = harness("adopt-guard0", "/sessions/adopt-guard.jsonl", "pi", shared);
	const guard =
		"/tmp/saccade-fake-guard-" + shared.guardSalt + "-adopt-guard0.guard";
	await first.pi.emit("session_start");
	assert.equal(probeHeld(guard), true, "the first instance holds the guard");
	await first.pi.emit("session_shutdown", { reason: "reload" });
	// a reloaded runtime: a new factory invocation, fresh closures
	const second = reloadedHarness(first, "adopt-guard0", "/sessions/adopt-guard.jsonl");
	await second.pi.emit("session_start");
	assert.equal(probeHeld(guard), true, "the reloaded runtime keeps the guard held");
	assert.equal(second.pi.shutdowns, 0, "the reloaded runtime is not refused");
	const identity = second.tools("saccade_identity");
	const text = resultText(
		await identity.execute(
			"t1",
			{},
			undefined,
			undefined,
			{ ...second.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
		),
	);
	assert.ok(text.includes("i-0"), text);
	assert.ok(text.includes("already registered"), text);
	await stop(second);
});

test("the guard registry survives a full module re-evaluation", async () => {
	// proof level: fresh module re-evaluation (jiti moduleCache:false shape); kernel evidence
	const shared = new FakeTracker();
	const guard =
		"/tmp/saccade-fake-guard-" + shared.guardSalt + "-reeval-guard0.guard";
	process.env.SACCADE_ACTOR = "pi";
	shared.install();
	process.env.SACCADE_SERVER = shared.url();
	const firstPi = new FakePi("reeval-guard0", "/sessions/reeval-guard.jsonl");
	saccadeIntegration(firstPi);
	await firstPi.emit("session_start");
	assert.equal(probeHeld(guard), true);
	await firstPi.emit("session_shutdown", { reason: "reload" });
	// a different specifier means a different module record: fresh
	// module scope, factory re-evaluated from the source text
	const reevaluated = (await import(`./saccade.ts?reeval=${Date.now()}`)) as
		typeof import("./saccade.ts");
	const secondPi = new FakePi("reeval-guard0", "/sessions/reeval-guard.jsonl");
	reevaluated.default(secondPi);
	await secondPi.emit("session_start");
	assert.equal(probeHeld(guard), true, "the lock survives module re-evaluation");
	assert.equal(secondPi.shutdowns, 0, "the re-evaluated runtime is not refused");
	await stop({ pi: secondPi, tracker: shared, tools: () => undefined as never, ctx: () => undefined as never } as Harness);
});

test("a real second process refuses the attach and its death releases the lock", async () => {
	// proof level: kernel, cross-process - refusal, single registration, release on real group death
	const shared = new FakeTracker();
	const guard =
		"/tmp/saccade-fake-guard-" + shared.guardSalt + "-crossproc0.guard";
	const holder = spawn("flock", [guard, "sleep", "30"], {
		stdio: ["ignore", "ignore", "ignore"],
		detached: true,
	});
	try {
		const deadline = Date.now() + 5000;
		while (!probeHeld(guard) && Date.now() < deadline) await tick();
		assert.equal(probeHeld(guard), true, "the second process holds the lock");
		const dup = harness("crossproc0", "/sessions/crossproc.jsonl", "pi", shared);
		await dup.pi.emit("session_start");
		assert.ok(
			dup.pi.shutdowns >= 1,
			"the duplicate refusal requests exit (refusal and start-rule agree)",
		);
		const identity = dup.tools("saccade_identity");
		const text = resultText(
			await identity.execute(
				"t1",
				{},
				undefined,
				undefined,
				{ ...dup.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
			),
		);
		assert.ok(text.includes("not registered"), text);
		assert.ok(text.includes("duplicate"), `the refusal names the held guard: ${text}`);
		// real process-group death: the kernel releases, and a fresh
		// instance of the integration attaches onto the freed lock
		process.kill(-holder.pid!, "SIGKILL");
		const freeDeadline = Date.now() + 5000;
		while (probeHeld(guard) && Date.now() < freeDeadline) await tick();
		assert.equal(probeHeld(guard), false, "process death released the lock");
		const fresh = reloadedHarness(dup, "crossproc0", "/sessions/crossproc.jsonl");
		await fresh.pi.emit("session_start");
		assert.equal(fresh.pi.shutdowns, 0, "the freed guard admits the session");
		assert.equal(shared.attachments.size, 1, "one registration, resolved twice");
		await stop(fresh);
	} finally {
		try {
			process.kill(-holder.pid!, "SIGKILL");
		} catch {
			// the group is already gone
		}
	}
});

test("a human-started session that cannot verify exits, explaining the failure", async () => {
	// ratified offline rule: no token, tracker down, no held guard - explain and exit
	const shared = new FakeTracker();
	delete process.env.SACCADE_LAUNCH_TOKEN;
	const h = harness("offline-gap0", "/sessions/offline-gap.jsonl", "pi", shared);
	shared.attachFails = true;
	await h.pi.emit("session_start");
	assert.equal(h.pi.shutdowns, 1, "the unverified start exits");
	const note = h.pi.appendEntries[h.pi.appendEntries.length - 1];
	assert.ok(note, "the failure is explained, not silent");
	assert.equal(note.type, "saccade_attach");
	assert.match(
		String((note.data as { error?: string }).error),
		/degraded/,
	);
	// the explanation paints: the registered renderer rendered the
	// visible card, not just the private record
	const painted = h.pi.rendered[h.pi.rendered.length - 1] as {
		render: (width: number) => string[];
	};
	assert.ok(painted, "the attach entry has a reader and painted");
	assert.match(
		painted.render(80).join("\n"),
		/degraded/,
		"the visible card names the tracker failure",
	);
	shared.attachFails = false;
	await stop(h);
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
		assert.ok(
			h.pi.shutdowns >= 1,
			"the unverified session requests exit (refusal and start-rule agree)",
		);
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
		assert.ok(
			h.pi.shutdowns > 1,
			"every attempt refuses and requests exit",
		);
	} finally {
		if (savedPath === undefined) delete process.env.PATH;
		else process.env.PATH = savedPath;
	}
	await stop(h);
	rmSync(shim, { recursive: true, force: true });
});

test("an in-memory helper start stays alive, unregistered, parent env cleared", async () => {
	// intentional in-memory helper: alive, unregistered, no inherited incarnation
	const shared = new FakeTracker();
	delete process.env.SACCADE_INCARNATION_SEQ;
	const h = harness("helper-run0", undefined, "pi", shared);
	process.env.SACCADE_INCARNATION_SEQ = "0";
	await h.pi.emit("session_start");
	// the inherited parent incarnation is gone before any helper work
	assert.equal(
		process.env.SACCADE_INCARNATION_SEQ,
		undefined,
		"the helper never inherits the parent incarnation",
	);
	assert.equal(h.pi.shutdowns, 0, "a healthy tracker does not kill the helper");
	assert.equal(shared.attachments.size, 0, "nothing registered without a session");
	const identity = h.tools("saccade_identity");
	const text = resultText(
		await identity.execute(
			"t1",
			{},
			undefined,
			undefined,
			{ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) },
		),
	);
	assert.ok(text.includes("not registered"), text);
	assert.equal(process.env.SACCADE_INCARNATION_SEQ, undefined);
	await stop(h);
});

test("a headless unverifiable start explains on stderr before exiting", async () => {
	// no UI to paint: the explanation must still be observable
	const shared = new FakeTracker();
	const h = harness("headless-off0", "/sessions/headless-off.jsonl", "pi", shared);
	h.pi.hasUI = false;
	shared.attachFails = true;
	const written: string[] = [];
	const original = process.stderr.write.bind(process.stderr);
	process.stderr.write = ((chunk: string | Uint8Array) => {
		written.push(String(chunk));
		return true;
	}) as typeof process.stderr.write;
	try {
		await h.pi.emit("session_start");
	} finally {
		process.stderr.write = original;
	}
	assert.equal(h.pi.shutdowns >= 1, true, "the unverified start exits");
	assert.match(
		written.join(""),
		/degraded/,
		"stderr names the tracker failure",
	);
	shared.attachFails = false;
	await stop(h);
});

// -- the explicit own-session workspace handoff ------------------------------

// The fake wrapper side of the public contract: a jobs tool served
// through the nested dispatch the pinned SDK actually runs, plus
// admission dispatches of the same shape - tool_execution_start before
// the tool_call hooks, the end always, blocked calls never execute.
interface JobsExecute {
	isError: boolean;
	text: string;
	structured?: unknown;
}

class FakeWrapper {
	pi: FakePi;
	// the registered live jobs, returned as the snapshot's payload
	jobs: Array<{ id: string; kind: string; state: string }> = [];
	// when set, the jobs outcome is an error result carrying this text
	jobsError: string | null = null;
	// when set, replaces the structuredContent the jobs tool returns
	payload: { value: unknown } | null = null;
	// when true, no jobs tool answers the nested call at all
	noJobsTool = false;
	queries = 0;
	// the admission attempts the tool_call hook blocked, in order
	blocked: string[] = [];
	private seq = 0;
	// holds the next jobs query mid-dispatch, before its result
	private gate: { open: () => void; promise: Promise<void> } | null = null;

	constructor(pi: FakePi) {
		this.pi = pi;
	}

	holdNextQuery(): void {
		let open!: () => void;
		const promise = new Promise<void>((resolve) => {
			open = resolve;
		});
		this.gate = { open, promise };
	}

	releaseQuery(): void {
		const gate = this.gate;
		this.gate = null;
		gate?.open();
	}

	private jobsExecute(): Promise<JobsExecute> {
		// the read counts once dispatched, even when nothing answers
		this.queries += 1;
		if (this.noJobsTool) return Promise.resolve({ isError: true, text: "Tool jobs not found" });
		if (this.gate !== null) {
			// the gate stays until released: a later holdNextQuery replaces it
			const gate = this.gate;
			return gate.promise.then(() => this.jobsAnswer());
		}
		return Promise.resolve(this.jobsAnswer());
	}

	private jobsAnswer(): JobsExecute {
		if (this.jobsError !== null) return { isError: true, text: this.jobsError };
		const structured = this.payload !== null ? this.payload.value : { jobs: this.jobs };
		return { isError: false, text: "jobs snapshot", structured };
	}

	// the pinned hook order: the first block wins and stops the chain
	private async hooksBlocked(toolName: string, args: unknown, id: string): Promise<boolean> {
		for (const handler of this.pi.handlers.get("tool_call") ?? []) {
			const result = (await handler(
				{ type: "tool_call", toolCallId: id, toolName, input: args },
				this.pi.context(),
			)) as { block?: boolean } | undefined;
			if (result?.block) return true;
		}
		return false;
	}

	private endEvent(toolName: string, id: string, outcome: JobsExecute) {
		return {
			toolCallId: id,
			toolName,
			result: {
				content: [{ type: "text", text: outcome.text }],
				details: undefined,
				...(outcome.structured !== undefined ? { structuredContent: outcome.structured } : {}),
			},
			isError: outcome.isError,
		};
	}

	// one full nested dispatch: start, hooks, execute, end
	private async dispatch(toolName: string, args: unknown, execute: () => Promise<JobsExecute>): Promise<JobsExecute> {
		const id = `fake-${toolName}-${this.seq++}`;
		await this.pi.emit("tool_execution_start", { toolCallId: id, toolName, args });
		let outcome: JobsExecute;
		if (await this.hooksBlocked(toolName, args, id)) {
			this.blocked.push(toolName);
			outcome = { isError: true, text: `blocked: ${toolName}` };
		} else {
			outcome = await execute();
		}
		await this.pi.emit("tool_execution_end", this.endEvent(toolName, id, outcome));
		return outcome;
	}

	// an admission attempt: a background bash start or a subagent start
	async startAdmission(toolName: "bash" | "subagent", args: Record<string, unknown>): Promise<void> {
		await this.dispatch(toolName, args, async () => ({ isError: false, text: `${toolName} started` }));
	}

	// an admission whose dispatch has started but whose end has not
	// fired: the in-flight shape the registry cannot show yet. The
	// returned settle fires the end; settle(true) ends it failed.
	async holdAdmission(
		toolName: "bash" | "subagent",
		args: Record<string, unknown>,
	): Promise<(failed?: boolean) => Promise<void>> {
		const id = `fake-${toolName}-${this.seq++}`;
		await this.pi.emit("tool_execution_start", { toolCallId: id, toolName, args });
		if (await this.hooksBlocked(toolName, args, id)) {
			this.blocked.push(toolName);
			await this.pi.emit("tool_execution_end", {
				...this.endEvent(toolName, id, { isError: true, text: "blocked" }),
			});
			return async () => {};
		}
		return (failed = false) =>
			this.pi
				.emit("tool_execution_end", {
					...this.endEvent(toolName, id, { isError: failed, text: failed ? "failed" : "ended" }),
				})
				.then(() => {});
	}

	// the tool context the integration sees: executeTool rides the same
	// dispatch and the jobs tool answers through it
	ctxOf(h: Harness): ExtensionToolContext {
		return {
			...h.ctx(),
			tools: [],
			executeTool: async (name: string, args: unknown) => {
				const outcome = await this.dispatch(name, args, () => this.jobsExecute());
				return {
					toolCall: { type: "toolCall", id: "fake-nested", name, arguments: {} },
					result: {
						content: [{ type: "text", text: outcome.text }],
						details: undefined,
						...(outcome.structured !== undefined ? { structuredContent: outcome.structured } : {}),
					},
					isError: outcome.isError,
				};
			},
		} as unknown as ExtensionToolContext;
	}
}

// The workspace arm's environment: routing and the controller door named.
async function withHandoffEnv<T>(fn: () => Promise<T>): Promise<T> {
	const savedCli = process.env.SACCADE_CLI;
	const savedRepo = process.env.SACCADE_REPO;
	process.env.SACCADE_CLI = "/bin/true";
	process.env.SACCADE_REPO = "/story/tracker";
	try {
		return await fn();
	} finally {
		if (savedCli === undefined) delete process.env.SACCADE_CLI;
		else process.env.SACCADE_CLI = savedCli;
		if (savedRepo === undefined) delete process.env.SACCADE_REPO;
		else process.env.SACCADE_REPO = savedRepo;
	}
}

function scratchDir(tag: string): string {
	const dir = mkdtempSync(join(tmpdir(), `saccade-handoff-${tag}-`));
	return dir;
}

test("the handoff envelope carries the machine provenance in ASCII with the target as data", () => {
	const data: HandoffEntryData = {
		token: "handoff-t",
		nativeId: "sess-1",
		sessionFile: "/sessions/sess-1.jsonl",
		toolCall: "call_9",
		incarnation: 3,
		target: "/story/worktrees/\u{e9}tude",
		created: null,
		log: "/tmp/handoff.log",
		scope: { actor: "pi", task: "t-9" },
	};
	const envelope = handoffEnvelope(data, 41);
	assert.match(envelope, /^\[saccade workspace handoff \| tool call_9 \| change record 41\]/);
	assert.match(envelope, /kept its id, saved history, and declared scope/);
	// the moved-to directory rides byte for byte, never transliterated
	assert.ok(envelope.includes("/story/worktrees/\u{e9}tude"));
	assert.doesNotMatch(envelope.split("\n")[0] ?? "", /[^\x20-\x7e]/);
});

test("headerCwd reads the saved header the one way the controller does", () => {
	const dir = scratchDir("header");
	const file = join(dir, "session.jsonl");
	writeFileSync(
		file,
		`{"type":"session","version":3,"id":"handoff-smoke-1a2b3c4d","cwd":"/story/wt-a"}\n{"type":"message","id":"ab12cd34","message":{"role":"user","content":"implement the handoff"}}\n`,
	);
	assert.equal(headerCwd(file), "/story/wt-a");
	writeFileSync(file, `{"type":"message","id":"ab12cd34"}\n`);
	assert.equal(headerCwd(file), null, "a conversation line first is no header");
	writeFileSync(file, `{"type":"session","id":"handoff-smoke-1a2b3c4d"}\n`);
	assert.equal(headerCwd(file), null, "a header without a cwd reads null");
	writeFileSync(file, `{broken\n`);
	assert.equal(headerCwd(file), null, "an unreadable file reads null");
	assert.equal(headerCwd(join(dir, "absent.jsonl")), null);
	rmSync(dir, { recursive: true, force: true });
});

test("the helper script waits on the exact guard then calls the controller door by name", () => {
	const script = handoffHelperScript({
		guard: "/state/act/s1234abcd.guard",
		cli: "/bin/sac",
		server: "http://127.0.0.1:8811",
		repo: "/story/tracker",
		num: 7,
		target: "/story/wt/it's moved",
		cycles: 480,
	});
	assert.match(script, /flock -n '\/state\/act\/s1234abcd\.guard' true/);
	assert.match(script, /while \[ "\$i" -lt 480 \]/);
	// the resume is the stopped-session controller's own door, addressed
	// by the public incarnation number and resumed in place
	assert.match(
		script,
		/exec '\/bin\/sac' --server 'http:\/\/127\.0\.0\.1:8811' --repo '\/story\/tracker' workspace switch 'i-7' --to '\/story\/wt\/it'\\''s moved' --resume/,
	);
});

test("the workspace tool refuses an unsaved session before any status read", async () => {
	const h = harness("mem0", undefined, "pi");
	const wrapper = new FakeWrapper(h.pi);
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /unsaved \(in-memory\)/);
	});
	assert.equal(wrapper.queries, 0, "an unsaved session never reads the wrapper");
	// no admission gate leaked: a background start goes through untouched
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, []);
	await stop(h);
});

test("the workspace tool refuses as unknown when no jobs tool answers", async () => {
	const h = harness("noprov1", "/sessions/noprov1.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	wrapper.noJobsTool = true;
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /unverifiable activity is never idle/);
	});
	assert.equal(wrapper.queries, 1, "the read went out once and was refused");
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false, "nothing armed on an unknown wrapper");
	assert.equal(h.pi.shutdowns, 0, "an unknown wrapper never shuts the session down");
	// the hold went back with the refusal: admissions run untouched
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, []);
	await stop(h);
});

test("a jobs answer that errors or is malformed refuses as unknown, never as idle", async () => {
	const h = harness("badjobs1", "/sessions/badjobs1.jsonl", "pi");
	await h.pi.emit("session_start");
	const variants: Array<{ label: string; shape: (w: FakeWrapper) => void }> = [
		{ label: "errored", shape: (w) => (w.jobsError = "the poller died") },
		{ label: "no structuredContent", shape: (w) => (w.payload = { value: undefined }) },
		{ label: "not an object", shape: (w) => (w.payload = { value: "jobs" }) },
		{ label: "no jobs field", shape: (w) => (w.payload = { value: {} }) },
		{ label: "jobs not an array", shape: (w) => (w.payload = { value: { jobs: "all clear" } }) },
		{ label: "entry not an object", shape: (w) => (w.payload = { value: { jobs: [7] } }) },
		{ label: "id missing", shape: (w) => (w.payload = { value: { jobs: [{ kind: "bash", state: "running" }] } }) },
		{ label: "kind unknown", shape: (w) => (w.payload = { value: { jobs: [{ id: "bg-1", kind: "powershell", state: "running" }] } }) },
		{ label: "state unseen", shape: (w) => (w.payload = { value: { jobs: [{ id: "bg-1", kind: "bash", state: "finishing" }] } }) },
	];
	await withHandoffEnv(async () => {
		for (const variant of variants) {
			const wrapper = new FakeWrapper(h.pi);
			variant.shape(wrapper);
			const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
			assert.equal((out as any).isError, true, `${variant.label} must refuse`);
			assert.match((out as any).content[0].text, /never idle/, `${variant.label} refuses as unknown`);
			assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false, `${variant.label} armed nothing`);
			await wrapper.startAdmission("subagent", { task: "later work" });
			assert.deepEqual(wrapper.blocked, [], `${variant.label} released the hold`);
		}
	});
	await stop(h);
});

test("the workspace tool refuses a busy snapshot, stopping nothing", async () => {
	const h = harness("busy2", "/sessions/busy2.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	wrapper.jobs = [{ id: "bg-1", kind: "bash", state: "running" }];
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((out as any).content[0].text, /1 reported background job\(s\) \(running, stopping, or unresolved\): bg-1 \(bash, running\)/);
		assert.match((out as any).content[0].text, /nothing was stopped and nothing changed/);
	});
	assert.equal(wrapper.queries, 1);
	assert.equal(h.pi.shutdowns, 0);
	// the hold released at once: the session stays usable
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, []);
	await stop(h);
});

test("stopping and unknown jobs refuse exactly like running ones", async () => {
	const h = harness("busy2b", "/sessions/busy2b.jsonl", "pi");
	await h.pi.emit("session_start");
	await withHandoffEnv(async () => {
		for (const job of [
			{ id: "bg-2", kind: "bash", state: "stopping" },
			{ id: "sa-1", kind: "subagent", state: "unknown" },
		]) {
			const wrapper = new FakeWrapper(h.pi);
			wrapper.jobs = [job];
			const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
			assert.equal((out as any).isError, true, `${job.state} must refuse`);
			assert.match((out as any).content[0].text, new RegExp(`${job.id} \\(${job.kind}, ${job.state}\\)`));
			assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false);
		}
	});
	await stop(h);
});

test("the workspace tool names the missing helper inputs and reads nothing", async () => {
	const h = harness("nocli3", "/sessions/nocli3.jsonl", "pi");
	await h.pi.emit("session_start");
	delete process.env.SACCADE_CLI;
	const wrapper = new FakeWrapper(h.pi);
	try {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((out as any).content[0].text, /SACCADE_CLI/);
	} finally {
		delete process.env.SACCADE_CLI;
	}
	assert.equal(wrapper.queries, 0, "an unlaunchable helper never reads the wrapper");
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "no gate leaked from the refused request");
	await stop(h);
});

test("the workspace tool's param discipline leaves no default and no mixed shape", async () => {
	const h = harness("params4", "/sessions/params4.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	await withHandoffEnv(async () => {
		const missing = await h.tools("saccade_workspace").execute("c1", { choice: "existing" }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((missing as any).content[0].text, /names exactly a path/);
		const mixed = await h.tools("saccade_workspace").execute("c2", { choice: "existing", path: "/tmp", label: "nope" }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((mixed as any).content[0].text, /names exactly a path/);
		const pathed = await h.tools("saccade_workspace").execute("c3", { choice: "new", path: "/tmp" }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((pathed as any).content[0].text, /takes no path/);
	});
	assert.equal(wrapper.queries, 0, "param failures read nothing");
	await stop(h);
});

test("a failed preparation releases the hold and keeps the session usable", async () => {
	const h = harness("giveup5", "/sessions/giveup5.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const missing = join(scratchDir("giveup5x"), "not-there");
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: missing }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((out as any).isError, true);
	});
	assert.equal(wrapper.queries, 1, "the status read happened before the path failed");
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false);
	assert.equal(h.pi.shutdowns, 0, "a failed preparation never shuts the session down");
	await wrapper.startAdmission("subagent", { task: "carry on" });
	assert.deepEqual(wrapper.blocked, [], "the failed preparation released its own hold");
	await stop(h);
});

test("an armed handoff persists its facts, blocks admissions, and shuts the session down at settle", async () => {
	const h = harness("arm6", "/sessions/arm6.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("arm6-target");
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("call_arm", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		const body = (out as any).content[0].text;
		assert.match(body, /workspace handoff armed/);
		assert.match(body, /background admissions are held/);
		assert.ok(body.includes(target), "the armed reply names the target");
		assert.match(body, /helper's outcome log is (\S+\.handoff\.log)/);
		assert.match(body, /if the session does not come back, that file names the failure/);
	});
	assert.equal(wrapper.queries, 1, "one valid empty snapshot armed the handoff");
	const armed = h.pi.appendEntries.find((e) => e.type === "saccade_handoff");
	assert.ok(armed, "the arm is a saved fact");
	const data = armed!.data as HandoffEntryData;
	assert.equal(data.nativeId, "arm6");
	assert.equal(data.sessionFile, "/sessions/arm6.jsonl");
	assert.equal(data.toolCall, "call_arm");
	assert.equal(data.target, target);
	assert.equal(data.created, null, "an existing choice provisions nothing");
	// the harness injected this actor; the session carries no task scope
	assert.deepEqual(data.scope, { actor: "pi", task: null });
	assert.match(data.log, /\.handoff\.log$/);
	// while armed, background starts refuse and foreground work does not
	await wrapper.startAdmission("bash", { command: "sleep 5", background: true });
	await wrapper.startAdmission("subagent", { task: "more work" });
	assert.deepEqual(wrapper.blocked, ["bash", "subagent"], "the armed handoff holds admissions");
	await wrapper.startAdmission("bash", { command: "ls" });
	assert.equal(wrapper.blocked.length, 2, "foreground bash is not an admission");
	// the settle boundary shuts the session down with the helper launched
	await h.pi.emit("agent_settled");
	assert.equal(h.pi.shutdowns, 1, "the armed settle is the shutdown");
	assert.ok(existsSync(data.log), "the helper launched with its log file");
	// the helper is /bin/true under the test's SACCADE_CLI: it exits at once
	await new Promise((resolve) => setTimeout(resolve, 400));
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal(done, undefined, "a launched handoff is not cancelled afterwards");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("a failed arm append leaves nothing for the settle boundary to launch", async () => {
	const h = harness("appendfail19", "/sessions/appendfail19.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("appendfail19-target");
	await withHandoffEnv(async () => {
		h.pi.failAppend = "saccade_handoff";
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /hold is released and the session stays/);
		h.pi.failAppend = null;
		// the arm never went live: the settle boundary launches and shuts
		// down nothing, and the admission hold went back with the token
		await h.pi.emit("agent_settled");
	});
	assert.equal(h.pi.shutdowns, 0, "an unpersisted arm never shuts the session down");
	const armed = h.pi.appendEntries.find((e) => e.type === "saccade_handoff");
	assert.equal(armed, undefined, "the failed append persisted nothing");
	// the helper's log would sit beside the fake guard; nothing created it
	const guard = "/tmp/saccade-fake-guard-" + h.tracker.guardSalt + "-appendfail19.guard";
	assert.equal(existsSync(guard.replace(/\.guard$/, ".handoff.log")), false, "no helper launched for the failed arm");
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the failed append released the hold");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("an aborted run cancels only its armed handoff: no launch, no shutdown, admissions restored", async () => {
	const h = harness("abort20", "/sessions/abort20.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("abort20-target");
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((out as any).content[0].text, /handoff armed/);
		// the pinned AgentSettledEvent carries the run's own abort
		await h.pi.emit("agent_settled", { aborted: true });
	});
	assert.equal(h.pi.shutdowns, 0, "an aborted run never shuts the session down");
	const armed = h.pi.appendEntries.find((e) => e.type === "saccade_handoff")!;
	const log = (armed.data as HandoffEntryData).log;
	assert.equal(existsSync(log), false, "no helper launched on abort");
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "cancelled");
	assert.match((done!.data as HandoffDoneData).reason ?? "", /aborted/);
	// admissions restored: the released gate admits a fresh background start
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the abort released the hold");
	await stop(h);
});

test("a helper that cannot launch cancels the handoff and keeps the session alive", async () => {
	const h = harness("launchfail7", "/sessions/launchfail7.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("launchfail7-target");
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((out as any).content[0].text, /handoff armed/);
		const armed = h.pi.appendEntries.find((e) => e.type === "saccade_handoff")!;
		// the log path is occupied by a directory: the launch itself fails
		mkdirSync((armed.data as HandoffEntryData).log, { recursive: true });
		await h.pi.emit("agent_settled");
	});
	assert.equal(h.pi.shutdowns, 0, "a failed launch keeps the session usable");
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.ok(done, "the failed launch is a recorded outcome");
	assert.equal((done!.data as HandoffDoneData).outcome, "cancelled");
	assert.match((done!.data as HandoffDoneData).reason ?? "", /helper could not launch/);
	// the gate went with the cancelled arm
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the cancellation released the hold");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("native human input cancels an armed handoff; extension input does not", async () => {
	const h = harness("input8", "/sessions/input8.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("input8-target");
	await withHandoffEnv(async () => {
		await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		await h.pi.emit("input", { source: "extension", text: "[saccade message m-9 | from i-2]\n\nhello" });
		assert.equal(
			h.pi.appendEntries.some((e) => e.type === "saccade_handoff_done"),
			false,
			"an arriving delivery is not a takeover",
		);
		// still armed: admissions stay held through an arriving delivery
		await wrapper.startAdmission("bash", { command: "sleep 5", background: true });
		assert.deepEqual(wrapper.blocked, ["bash"]);
		await h.pi.emit("input", { source: "user", text: "wait, stay here" });
	});
	assert.equal(h.pi.shutdowns, 0);
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "cancelled");
	assert.match((done!.data as HandoffDoneData).reason ?? "", /human input/);
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, ["bash"], "the takeover released the hold");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("a session end before the settle cancels the armed handoff", async () => {
	const h = harness("end9", "/sessions/end9.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("end9-target");
	await withHandoffEnv(async () => {
		await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		await h.pi.emit("session_shutdown");
	});
	assert.equal(h.pi.shutdowns, 0);
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "cancelled");
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the session end released the hold");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("the jobs snapshot validator reads the pinned shape and tolerates compatible extras", () => {
	assert.deepEqual(validJobsSnapshot({ jobs: [] }), []);
	assert.deepEqual(validJobsSnapshot({ jobs: [{ id: "bg-1", kind: "bash", state: "running" }] }), [
		{ id: "bg-1", kind: "bash", state: "running" },
	]);
	assert.deepEqual(
		validJobsSnapshot({
			jobs: [
				{ id: "bg-2", kind: "bash", state: "stopping", startedAt: 9 },
				{ id: "sa-1", kind: "subagent", state: "unknown" },
			],
			checkedAt: "now",
		}),
		[
			{ id: "bg-2", kind: "bash", state: "stopping" },
			{ id: "sa-1", kind: "subagent", state: "unknown" },
		],
	);
	for (const bad of [
		null,
		undefined,
		"jobs",
		[],
		{},
		{ jobs: null },
		{ jobs: "none" },
		{ jobs: [null] },
		{ jobs: [{ id: 3, kind: "bash", state: "running" }] },
		{ jobs: [{ id: "", kind: "bash", state: "running" }] },
		{ jobs: [{ id: "bg-1", kind: "shell", state: "running" }] },
		{ jobs: [{ id: "bg-1", kind: "bash", state: "exited" }] },
		{ jobs: [{ id: "bg-1", kind: "bash", state: "finishing" }] },
	]) {
		assert.equal(validJobsSnapshot(bad), null, `a malformed snapshot (${JSON.stringify(bad)}) is unknown, never empty`);
	}
});

test("background admissions block while the handoff prepares; foreground work and jobs do not", async () => {
	const h = harness("prepare21", "/sessions/prepare21.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("prepare21-target");
	await withHandoffEnv(async () => {
		wrapper.holdNextQuery();
		const running = h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		await tick();
		assert.equal(wrapper.queries, 1, "the status read is in flight");
		// the gate is up: background bash and subagent starts refuse
		await wrapper.startAdmission("bash", { command: "sleep 5", background: true });
		await wrapper.startAdmission("subagent", { task: "sneaky work" });
		assert.deepEqual(wrapper.blocked, ["bash", "subagent"], "the preparing hold blocks admissions");
		// foreground bash, and the jobs query itself, are not admissions
		await wrapper.startAdmission("bash", { command: "ls" });
		assert.equal(wrapper.blocked.length, 2, "foreground bash passes the gate");
		// the blocked attempts left nothing in flight: the empty snapshot arms
		wrapper.releaseQuery();
		const out = await running;
		assert.match((out as any).content[0].text, /handoff armed/);
		await h.pi.emit("input", { source: "user", text: "stay" });
	});
	const armed = h.pi.appendEntries.find((e) => e.type === "saccade_handoff");
	assert.ok(armed, "the valid empty snapshot armed the handoff");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("an admission still in flight refuses the move before any query, and its end frees it", async () => {
	const h = harness("inflight22", "/sessions/inflight22.jsonl", "pi");
	await h.pi.emit("session_start");
	const target = scratchDir("inflight22-target");
	await withHandoffEnv(async () => {
		// a start whose spawn has not been acknowledged: no end yet
		const settle = await new FakeWrapper(h.pi).holdAdmission("bash", { command: "sleep 5", background: true });
		const busy = new FakeWrapper(h.pi);
		const refused = await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, busy.ctxOf(h));
		assert.equal((refused as any).isError, true);
		assert.match((refused as any).content[0].text, /1 background start\(s\) still in flight/);
		assert.equal(busy.queries, 0, "the in-flight admission is visible before the read");
		// a failed end (a refused spawn) frees the count the same way
		await settle(true);
		const ok = new FakeWrapper(h.pi);
		const armed = await h.tools("saccade_workspace").execute("c2", { choice: "existing", path: target }, undefined, undefined, ok.ctxOf(h));
		assert.match((armed as any).content[0].text, /handoff armed/);
		await h.pi.emit("input", { source: "user", text: "stay" });
	});
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("a second handoff request refuses while one is preparing or armed", async () => {
	const h = harness("second23", "/sessions/second23.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("second23-target");
	await withHandoffEnv(async () => {
		wrapper.holdNextQuery();
		const first = h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		await tick();
		const second = await h.tools("saccade_workspace").execute("c2", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((second as any).isError, true);
		assert.match((second as any).content[0].text, /already preparing or armed/);
		wrapper.releaseQuery();
		await first;
		const third = await h.tools("saccade_workspace").execute("c3", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((third as any).isError, true);
		assert.match((third as any).content[0].text, /already preparing or armed/, "an armed handoff refuses another");
		await h.pi.emit("input", { source: "user", text: "stay" });
	});
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("human input during the status read cancels the preparation; the stale read never arms", async () => {
	const h = harness("staleq24", "/sessions/staleq24.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("staleq24-target");
	await withHandoffEnv(async () => {
		wrapper.holdNextQuery();
		const running = h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		await tick();
		// the human takes the session back while the read is pending
		await h.pi.emit("input", { source: "user", text: "actually, stay" });
		wrapper.releaseQuery();
		const out = await running;
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /cancelled or the session changed while its status was read/);
	});
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "cancelled");
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false, "the stale read armed nothing");
	// the session is usable: a fresh request goes all the way
	const again = new FakeWrapper(h.pi);
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c2", { choice: "existing", path: target }, undefined, undefined, again.ctxOf(h));
		assert.match((out as any).content[0].text, /handoff armed/);
		await h.pi.emit("input", { source: "user", text: "stay" });
	});
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("an abort during the status read releases the hold and refuses the move", async () => {
	const h = harness("staleq25", "/sessions/staleq25.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("staleq25-target");
	const controller = new AbortController();
	await withHandoffEnv(async () => {
		wrapper.holdNextQuery();
		const running = h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, controller.signal, undefined, wrapper.ctxOf(h));
		await tick();
		controller.abort();
		wrapper.releaseQuery();
		const out = await running;
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /the call was cancelled; the hold is released and the session stays/);
	});
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false, "an aborted read never arms");
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the abort released the hold");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("a session end during the status read cancels the preparation; the read never arms", async () => {
	const h = harness("staleq26", "/sessions/staleq26.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("staleq26-target");
	await withHandoffEnv(async () => {
		wrapper.holdNextQuery();
		const running = h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
		await tick();
		await h.pi.emit("session_shutdown");
		wrapper.releaseQuery();
		const out = await running;
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /cancelled or the session changed while its status was read/);
	});
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "cancelled");
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false, "a dead preparation never arms");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("a provision awaited while the human cancels never arms and keeps the birth pointer", async () => {
	const h = harness("staleq27", "/sessions/staleq27.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const provisioned = scratchDir("staleq27-ws");
	h.tracker.createPaths.push(provisioned);
	await withHandoffEnv(async () => {
		h.tracker.holdNextCreate();
		const running = h.tools("saccade_workspace").execute("c1", { choice: "new", label: "late" }, undefined, undefined, wrapper.ctxOf(h));
		await tick();
		assert.equal(h.tracker.creates.length, 1, "the provision call is in flight");
		// the human cancels during the awaited provision call
		await h.pi.emit("input", { source: "user", text: "no, stay" });
		h.tracker.releaseCreate();
		const out = await running;
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /cancelled or the session changed while preparing/);
		// a provision already landed: the face must not claim nothing changed
		assert.doesNotMatch((out as any).content[0].text, /nothing changed/);
		// the birth already landed: the pointer survives the cancel
		assert.match((out as any).content[0].text, new RegExp(`birth record 301 at ${provisioned}; no move happened`));
	});
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false, "the cancelled preparation armed nothing");
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the cancel released the hold");
	rmSync(provisioned, { recursive: true, force: true });
	await stop(h);
});

// A saved session file with a story header and one entry, carrying the
// armed handoff's expected world.
function savedSession(id: string, cwd: string): string {
	const dir = scratchDir(`saved-${id}`);
	const file = join(dir, `${id}.jsonl`);
	writeFileSync(
		file,
		`{"type":"session","version":3,"id":"${id}","cwd":"${cwd}"}\n{"type":"message","id":"ab12cd34","message":{"role":"user","content":"continue the story work"}}\n`,
	);
	return file;
}

function armedEntry(overrides: Record<string, unknown>) {
	return {
		type: "custom",
		id: "feed1234",
		customType: "saccade_handoff",
		data: {
			token: "handoff-arm9",
			nativeId: "resume10",
			sessionFile: "/sessions/resume10.jsonl",
			toolCall: "call_res",
			incarnation: 0,
			target: "/story/wt-2",
			created: null,
			log: "/tmp/handoff.log",
			scope: { actor: null, task: null },
			...overrides,
		},
	};
}

function changedRecord(seq: number, incarnation: number, target: string) {
	return { seq, kind: "workspace_changed", payload: { incarnation, worktree: target } };
}

async function startWith(h: Harness) {
	await h.pi.emit("session_start");
}

test("a resumed session consumes the armed handoff and reactivates the own request with machine provenance", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume10", target);
	const h = harness("resume10", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(411, 0, target));
	h.pi.session.entries.push(
		armedEntry({ sessionFile: file, scope: { actor: "injected-scope-actor", task: "t-9" } }) as any,
	);
	await startWith(h);
	// the validated arm restores the session's own injected scope before
	// anything can write under the launch's default identity
	assert.equal(process.env.SACCADE_ACTOR, "injected-scope-actor");
	assert.equal(process.env.SACCADE_TASK, "t-9");
	// queue first: the done entry is NOT terminal at send time
	assert.equal(h.pi.steers.length, 1, "the continuation entered the native queue");
	assert.equal(
		h.pi.appendEntries.some((e) => e.type === "saccade_handoff_done"),
		false,
		"activation is not terminal before the saved insert",
	);
	const envelope = h.pi.steers[0] ?? "";
	assert.match(envelope, /^\[saccade workspace handoff \| tool call_res \| change record 411\]/);
	assert.match(envelope, /Continue the goal you were pursuing/);
	// the run saves the envelope; the boundary completes the arm terminally
	await h.pi.runQueue();
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.ok(done, "the saved insert is a recorded fact");
	assert.equal((done!.data as HandoffDoneData).outcome, "activated");
	assert.equal((done!.data as HandoffDoneData).record, 411);
	delete process.env.SACCADE_TASK;
	await stop(h);
});

test("a forked or foreign session never consumes the armed handoff", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume11", target);
	const h = harness("forked11", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(412, 0, target));
	// the arm names another native session: the fork carries the history
	// but not the request
	h.pi.session.entries.push(armedEntry({ nativeId: "the-original", sessionFile: file }) as any);
	await startWith(h);
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "stale");
	assert.match((done!.data as HandoffDoneData).reason ?? "", /not this session's/);
	assert.equal(h.pi.steers.length, 0);
	await stop(h);
});

test("an unmoved or re-moved header marks the arm stale without redirecting it", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume12", "/story/wt-elsewhere");
	const h = harness("resume12", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(413, 0, target));
	h.pi.session.entries.push(armedEntry({ nativeId: "resume12", sessionFile: file }) as any);
	await startWith(h);
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "stale");
	assert.match((done!.data as HandoffDoneData).reason ?? "", /not the armed target/);
	assert.equal(h.pi.steers.length, 0);
	await stop(h);
});

test("a missing confirmed change record marks the arm stale", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume13", target);
	const h = harness("resume13", file, "pi", new FakeTracker());
	// the record names another incarnation: nothing confirms this move
	h.tracker.records.push(changedRecord(414, 5, target));
	h.pi.session.entries.push(armedEntry({ nativeId: "resume13", sessionFile: file }) as any);
	await startWith(h);
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "stale");
	assert.match((done!.data as HandoffDoneData).reason ?? "", /no confirmed workspace change record/);
	await stop(h);
});

test("a tracker outage leaves the arm unconsumed for the next start", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume14", target);
	const h = harness("resume14", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(415, 0, target));
	h.pi.session.entries.push(armedEntry({ nativeId: "resume14", sessionFile: file }) as any);
	h.tracker.failNext = 1;
	await startWith(h);
	assert.equal(
		h.pi.appendEntries.some((e) => e.type === "saccade_handoff_done"),
		false,
		"an unread correlation is never a stale guess",
	);
	assert.equal(h.pi.steers.length, 0);
	// the next start reads the record and queues; the run saves the
	// envelope and the boundary completes the arm
	await startWith(h);
	assert.equal(h.pi.steers.length, 1);
	await h.pi.runQueue();
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "activated");
	await stop(h);
});

test("a refused continuation insertion stays unconsumed and re-queues at a later boundary", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume16", target);
	const h = harness("resume16", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(417, 0, target));
	h.pi.session.entries.push(armedEntry({ sessionFile: file, nativeId: "resume16" }) as any);
	// the queue refuses this insertion: nothing saved, nothing terminal
	h.pi.failNextSend = 1;
	await startWith(h);
	assert.equal(h.pi.steers.length, 0, "the refused insertion queued nothing");
	assert.equal(
		h.pi.appendEntries.some((e) => e.type === "saccade_handoff_done"),
		false,
		"an unsaved envelope never completes the arm",
	);
	// an idle boundary re-queues the same envelope - the delivery
	// discipline's recover-over-once-flag, no timers
	h.pi.idle = true;
	h.pi.pendingNative = false;
	await h.pi.emit("agent_settled");
	assert.equal(h.pi.steers.length, 1, "the boundary re-queued the envelope");
	await h.pi.runQueue();
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "activated");
	assert.equal((done!.data as HandoffDoneData).record, 417);
	await stop(h);
});

test("an envelope already in the saved history completes the arm without re-queueing", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume17", target);
	const h = harness("resume17", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(418, 0, target));
	const data = armedEntry({ sessionFile: file, nativeId: "resume17" }).data as HandoffEntryData;
	const envelope = handoffEnvelope(data, 418);
	// the crash-and-recover shape: the envelope was saved but the done
	// entry never landed
	h.pi.session.entries.push(armedEntry({ sessionFile: file, nativeId: "resume17" }) as any);
	h.pi.session.entries.push({ type: "message", message: { role: "user", content: envelope } } as any);
	await startWith(h);
	const done = h.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.ok(done, "the saved envelope completes the arm at once");
	assert.equal((done!.data as HandoffDoneData).outcome, "activated");
	assert.equal((done!.data as HandoffDoneData).record, 418);
	assert.equal(h.pi.steers.length, 0, "no duplicate queue for a saved envelope");
	await stop(h);
});

test("a reload between queue and save re-queues at the next start, never a silent loss", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume18", target);
	const tracker = new FakeTracker();
	tracker.records.push(changedRecord(419, 0, target));
	const h = harness("resume18", file, "pi", tracker);
	h.pi.session.entries.push(armedEntry({ sessionFile: file, nativeId: "resume18" }) as any);
	await startWith(h);
	assert.equal(h.pi.steers.length, 1);
	assert.equal(
		h.pi.appendEntries.some((e) => e.type === "saccade_handoff_done"),
		false,
	);
	// the reload: a fresh module over the same session file and tracker;
	// the queued envelope was never saved, the arm is still unconsumed
	const second = reloadedHarness(h, "resume18", file);
	second.tracker.records = tracker.records;
	second.pi.session.entries.push(...h.pi.session.entries);
	await second.pi.emit("session_start");
	assert.equal(second.pi.steers.length, 1, "the next start re-queues the unsaved envelope");
	await second.pi.runQueue();
	const done = second.pi.appendEntries.find((e) => e.type === "saccade_handoff_done");
	assert.equal((done!.data as HandoffDoneData).outcome, "activated");
	assert.equal((done!.data as HandoffDoneData).record, 419);
});

test("an already-consumed arm stays consumed and an older arm is superseded", async () => {
	const target = "/story/wt-2";
	const file = savedSession("resume15", target);
	const h = harness("resume15", file, "pi", new FakeTracker());
	h.tracker.records.push(changedRecord(416, 0, target));
	h.pi.session.entries.push(armedEntry({ token: "handoff-old", nativeId: "resume15", sessionFile: file }) as any);
	h.pi.session.entries.push(armedEntry({ token: "handoff-new", nativeId: "resume15", sessionFile: file }) as any);
	h.pi.session.entries.push({
		type: "custom",
		id: "cafe5678",
		customType: "saccade_handoff_done",
		data: { token: "handoff-old", outcome: "cancelled", reason: "earlier cancel" },
	} as any);
	await startWith(h);
	assert.equal(h.pi.steers.length, 1, "the latest arm queued its continuation");
	await h.pi.runQueue();
	const dones = h.pi.appendEntries.filter((e) => e.type === "saccade_handoff_done");
	assert.deepEqual(
		dones.map((e) => (e.data as HandoffDoneData).token),
		["handoff-new"],
		"the consumed arm stays consumed; the latest arm activates",
	);
	assert.equal((dones[0]!.data as HandoffDoneData).outcome, "activated");
	await stop(h);
});

test("the handoff entries paint through registered renderers", async () => {
	const h = harness("paint16", "/sessions/paint16.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const target = scratchDir("paint16-target");
	await withHandoffEnv(async () => {
		await h.tools("saccade_workspace").execute("c1", { choice: "existing", path: target }, undefined, undefined, wrapper.ctxOf(h));
	});
	assert.ok(h.pi.entryRenderers.get("saccade_handoff"), "the armed entry has a renderer");
	assert.ok(h.pi.entryRenderers.get("saccade_handoff_done"), "the done entry has a renderer");
	const painted = h.pi.rendered.at(-1) as any;
	assert.ok(String(painted?.text ?? painted).includes(target), "the arm painted with its target");
	rmSync(target, { recursive: true, force: true });
	await stop(h);
});

test("the new choice provisions through the server and arms the birth record", async () => {
	const h = harness("fresh17", "/sessions/fresh17.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	const provisioned = scratchDir("fresh17-ws");
	h.tracker.createPaths.push(provisioned);
	await withHandoffEnv(async () => {
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "new", label: "\u{e9}tude" }, undefined, undefined, wrapper.ctxOf(h));
		assert.match((out as any).content[0].text, /provisioned now, birth record 301/);
	});
	const created = h.tracker.creates[0];
	assert.equal(created.label, "\u{e9}tude", "the label rides as data, never transliterated");
	const armed = h.pi.appendEntries.find((e) => e.type === "saccade_handoff");
	assert.equal((armed!.data as HandoffEntryData).created, 301);
	assert.equal((armed!.data as HandoffEntryData).target, provisioned);
	rmSync(provisioned, { recursive: true, force: true });
	await stop(h);
});

test("a provision that comes back with no usable path releases the hold and keeps the birth pointer", async () => {
	const h = harness("fresh18", "/sessions/fresh18.jsonl", "pi");
	await h.pi.emit("session_start");
	const wrapper = new FakeWrapper(h.pi);
	await withHandoffEnv(async () => {
		// no createPaths queued: the fake births /nonexistent-saccade-provision
		const out = await h.tools("saccade_workspace").execute("c1", { choice: "new" }, undefined, undefined, wrapper.ctxOf(h));
		assert.equal((out as any).isError, true);
		assert.match((out as any).content[0].text, /hold is released and the session stays/);
		// the birth already landed: the failure face carries the true
		// provision pointer, never a rollback or a success claim
		assert.match((out as any).content[0].text, /birth record 301 at \/nonexistent-saccade-provision; no move happened/);
	});
	await wrapper.startAdmission("bash", { command: "ls", background: true });
	assert.deepEqual(wrapper.blocked, [], "the failed provision released the hold");
	assert.equal(h.pi.appendEntries.some((e) => e.type === "saccade_handoff"), false);
	// nothing is live for the settle boundary after the failure
	await h.pi.emit("agent_settled");
	assert.equal(h.pi.shutdowns, 0);
	await stop(h);
});
