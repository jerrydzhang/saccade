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

// The guard sidecar lives beside the session file, so every fake
// session file must sit in a real directory. One memoized mapping per
// literal path: reload sequences name the same file and must land on
// the same sidecar.
const sessionFiles = new Map<string, string>();
function realSessionFile(file: string): string {
	// a path that already exists is its own real session file (the
	// consumption fixtures write real ones); only the /sessions
	// literals need a home
	if (existsSync(file)) return file;
	let real = sessionFiles.get(file);
	if (real === undefined) {
		const dir = mkdtempSync(join(tmpdir(), "sac-session-"));
		real = join(dir, file.replaceAll("/", "_"));
		sessionFiles.set(file, real);
	}
	return real;
}

// The sidecar guard path for a fake session file.
function guardOf(file: string): string {
	return `${realSessionFile(file)}.guard`;
}

class FakeSession {
	entries: SessionEntryLike[] = [];
	nativeId: string;
	file: string | undefined;
	constructor(nativeId: string, file: string | undefined) {
		this.nativeId = nativeId;
		this.file = file === undefined ? undefined : realSessionFile(file);
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
	// when set, an accepted input stays unobservable: the receipt
	// callback never fires, so the confirmation waits for the later
	// boundaries - the lost-receipt seam
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
		opts?: {
			deliverAs?: "steer" | "followUp";
			expandPromptTemplates?: boolean;
			preflightResult?: (outcome: "queued" | "started" | "handled") => void;
		},
	) {
		if (this.failNextSend > 0) {
			this.failNextSend -= 1;
			this.errors.push("send_user_message: no auth: the provider refused");
			return;
		}
		this.steers.push(content);
		this.modes.push(opts?.deliverAs);
		// the runtime's own truth: an accepted input is either pending
		// natively or already consuming a run, and the per-call receipt
		// names which acceptance happened
		if (!this.silentAccept) {
			const started = this.idle;
			this.pendingNative = true;
			this.idle = false;
			opts?.preflightResult?.(started ? "started" : "queued");
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
	sent: any[] = [];
	attachBodies: any[] = [];
	attachFails = false;
	failNext = 0;
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

	// a landed record joins the mailbox: a descriptive read only, and
	// no machinery delivers it
	land(incarnation: number, message: any) {
		const box = this.mailboxes.get(incarnation) ?? [];
		box.push(message);
		this.mailboxes.set(incarnation, box);
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
		if (path === "/api/v1/messages") {
			const url = new URL(String(rawUrl));
			const incarnation = Number(url.searchParams.get("incarnation"));
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
	const guard = guardOf("/sessions/reload-guard.jsonl");
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


test("a reload to a different native session releases the old session's guard", async () => {
	const h = harness("switch-old0", "/sessions/switch-old.jsonl", "pi", new FakeTracker());
	const oldGuard = guardOf("/sessions/switch-old.jsonl");
	const newGuard = guardOf("/sessions/switch-new.jsonl");
	await h.pi.emit("session_start");
	assert.equal(probeHeld(oldGuard), true, "the old session's guard is held");
	h.pi.session = new FakeSession("switch-new0", "/sessions/switch-new.jsonl");
	await h.pi.emit("session_start");
	assert.equal(
		probeHeld(oldGuard),
		false,
		"the old session's guard released with the identity change",
	);
	assert.equal(probeHeld(newGuard), true, "the new session's guard is held");
	await stop(h);
});

test("a fresh factory after shutdown adopts the held exact-session guard", async () => {
	// proof level: fresh factory, same module (cached-factory reload path); kernel evidence
	const shared = new FakeTracker();
	const first = harness("adopt-guard0", "/sessions/adopt-guard.jsonl", "pi", shared);
	const guard = guardOf("/sessions/adopt-guard.jsonl");
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
	const guard = guardOf("/sessions/reeval-guard.jsonl");
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
	const guard = guardOf("/sessions/crossproc.jsonl");
	const holder = spawn("flock", [guard, "sleep", "30"], {
		stdio: ["ignore", "ignore", "ignore"],
		detached: true,
	});
	try {
		const deadline = Date.now() + 5000;
		while (!probeHeld(guard) && Date.now() < deadline) await tick();
		assert.equal(probeHeld(guard), true, "the second process holds the lock");
		const dup = harness("crossproc0", "/sessions/crossproc.jsonl", "pi", shared);
		const dupErr: string[] = [];
		const bodiesBefore = shared.attachBodies.length;
		const originalErr = process.stderr.write.bind(process.stderr);
		process.stderr.write = ((chunk: string | Uint8Array) => {
			dupErr.push(String(chunk));
			return true;
		}) as typeof process.stderr.write;
		try {
			await dup.pi.emit("session_start");
		} finally {
			process.stderr.write = originalErr;
		}
		assert.ok(
			dup.pi.shutdowns >= 1,
			"the duplicate refusal requests exit (refusal and start-rule agree)",
		);
		// the refusal stays local: no history write, stderr carries it
		assert.equal(
			dup.pi.appendEntries.length,
			0,
			"a refused duplicate writes no history",
		);
		assert.match(
			dupErr.join(""),
			/duplicate/,
			"the refusal names the held guard on stderr",
		);
		// the guard-refused duplicate posts nothing: no attach, no reassignment
		assert.equal(
			shared.attachBodies.length,
			bodiesBefore,
			"a refused duplicate posts no attach",
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

test("a tracker outage at boot keeps the session alive and guarded: registration retries lazily", async () => {
	// the guard take precedes any tracker ask, so identity is verified
	// even when the tracker is down: the session stays, the failure is
	// painted, and the next tool call registers lazily
	const shared = new FakeTracker();
	const h = harness("offline-gap0", "/sessions/offline-gap.jsonl", "pi", shared);
	shared.attachFails = true;
	await h.pi.emit("session_start");
	assert.equal(h.pi.shutdowns, 0, "a verified identity never exits for a tracker outage");
	assert.equal(probeHeld(guardOf("/sessions/offline-gap.jsonl")), true, "the guard is held");
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
	const identity = h.tools("saccade_identity");
	const toolCtx = () => ({ ...h.ctx(), tools: [], executeTool: () => Promise.resolve({} as never) });
	const refused = resultText(await identity.execute("t1", {}, undefined, undefined, toolCtx()));
	assert.ok(refused.includes("not registered"), refused);
	// the tracker returns: the lazy retry registers without a restart
	shared.attachFails = false;
	const ok = resultText(await identity.execute("t2", {}, undefined, undefined, toolCtx()));
	assert.ok(ok.includes("i-0"), ok);
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
		// an unprobed guard owns nothing: the refusal writes no history
		assert.equal(
			h.pi.appendEntries.length,
			0,
			"an unverified refusal writes no history",
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
	// an in-memory helper owns no saved history to write
	assert.equal(h.pi.appendEntries.length, 0, "the helper writes no history");
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
	// no UI to paint: an identity that cannot verify (the guard take
	// is unknown) must still be observable before the exit
	const shared = new FakeTracker();
	const shim = mkdtempSync("/tmp/saccade-flock-shim-");
	writeFileSync(join(shim, "flock"), "#!/bin/sh\nexit 16\n");
	chmodSync(join(shim, "flock"), 0o755);
	const savedPath = process.env.PATH;
	process.env.PATH = shim;
	let h: ReturnType<typeof harness>;
	const written: string[] = [];
	const original = process.stderr.write.bind(process.stderr);
	process.stderr.write = ((chunk: string | Uint8Array) => {
		written.push(String(chunk));
		return true;
	}) as typeof process.stderr.write;
	try {
		h = harness("headless-off0", "/sessions/headless-off.jsonl", "pi", shared);
		h.pi.hasUI = false;
		await h.pi.emit("session_start");
	} finally {
		process.stderr.write = original;
		if (savedPath === undefined) delete process.env.PATH;
		else process.env.PATH = savedPath;
	}
	assert.equal(h.pi.shutdowns >= 1, true, "the unverified start exits");
	assert.match(
		written.join(""),
		/cannot be probed/,
		"stderr names the identity failure",
	);
	await stop(h);
	rmSync(shim, { recursive: true, force: true });
});

test("the presented terminal pairs host disclosure with session-scoped launch provenance", async () => {
	const saved = [process.env.HC_TERMINAL_ID, process.env.SACCADE_LAUNCHED_SESSION];
	const cases = [
		// the door's own word: launched
		{ id: "presenter0001", word: "presenter0001", adopted: false },
		// no launch word: a manual start in a host pane reads adopted
		{ id: "presenter0002", word: undefined, adopted: true },
		// a word naming another session: a nested start reads adopted
		{ id: "presenter0003", word: "older9999", adopted: true },
	];
	try {
		for (const c of cases) {
			process.env.HC_TERMINAL_ID = "t9";
			if (c.word === undefined) delete process.env.SACCADE_LAUNCHED_SESSION;
			else process.env.SACCADE_LAUNCHED_SESSION = c.word;
			const h = harness(c.id, `/sessions/${c.id}.jsonl`, "pi");
			await h.pi.emit("session_start");
			const body = h.tracker.attachBodies[h.tracker.attachBodies.length - 1];
			assert.deepEqual(
				body.terminal,
				{ address: "t9", adopted: c.adopted },
				`the pair rides the attach for ${c.id}`,
			);
			await stop(h);
		}
		// no injected address presents nothing: outside the host, the
		// association stays as the record holds it
		process.env.HC_TERMINAL_ID = "";
		const h = harness("presenter0004", "/sessions/presenter0004.jsonl", "pi");
		await h.pi.emit("session_start");
		const body = h.tracker.attachBodies[h.tracker.attachBodies.length - 1];
		assert.equal(body.terminal, undefined, "no disclosure sends no pair");
		await stop(h);
	} finally {
		if (saved[0] === undefined) delete process.env.HC_TERMINAL_ID;
		else process.env.HC_TERMINAL_ID = saved[0];
		if (saved[1] === undefined) delete process.env.SACCADE_LAUNCHED_SESSION;
		else process.env.SACCADE_LAUNCHED_SESSION = saved[1];
	}
});
