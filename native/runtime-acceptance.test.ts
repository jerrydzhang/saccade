// The acceptance-callback contract against the real pinned runtime:
// this file runs under the composition preload, so the package's
// sendUserMessage carries the forwarded preflight receipt, and every
// scenario below observes what the sender's wire attempt would observe
// - queued at steer insertion, started at run start, handled on
// interception, silence on a pre-acceptance throw, and an ordinary
// option-less call that queues exactly as before. The model stream
// itself is expected to fail against the smoke key; acceptance precedes
// the stream, which is the bar under test.
import { strict as assert } from "node:assert";
import test from "node:test";
import {
	createAgentSession,
	DefaultResourceLoader,
	getAgentDir,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

interface SessionHarness {
	session: any;
	dispose: () => void;
}

async function harness(extensions: Array<(pi: any) => void>): Promise<SessionHarness> {
	const home = mkdtempSync(join(tmpdir(), "saccade-acceptance-"));
	const modelRuntime = await ModelRuntime.create({
		authPath: join(home, "auth.json"),
		modelsPath: join(home, "models.json"),
	});
	const model = modelRuntime.getModel("anthropic", "claude-sonnet-4-5");
	assert.ok(model, "the pinned registry lists the smoke model");
	// a smoke key: auth checks pass, the stream itself fails, and the
	// failure is downstream of the acceptance under test
	modelRuntime.setRuntimeApiKey(model.provider, "smoke-key");
	const resourceLoader = new DefaultResourceLoader({
		cwd: home,
		agentDir: getAgentDir(),
		extensionFactories: extensions,
	});
	await resourceLoader.reload();
	const { session } = await createAgentSession({
		cwd: home,
		agentDir: getAgentDir(),
		model,
		thinkingLevel: "off",
		modelRuntime,
		settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
		resourceLoader,
		sessionManager: SessionManager.inMemory(home),
	});
	return {
		session,
		dispose: () => {
			session.abortCompaction();
			session.dispose();
		},
	};
}

// The per-call receipt as a promise: it settles with the outcome when
// the runtime reports this message's acceptance or interception, and
// rejects when the runtime throws before any outcome exists - exactly
// the two shapes the sender's wire attempt distinguishes.
function receipt(session: any, envelope: string): Promise<string> {
	return new Promise((resolve, reject) => {
		const options = {
			deliverAs: "steer",
			preflightResult: (outcome: string) => resolve(outcome),
		} as unknown as Parameters<typeof session.sendUserMessage>[1];
		const pending = session.sendUserMessage(envelope, options);
		if (pending && typeof pending.catch === "function") {
			pending.catch((error: unknown) => reject(error));
		}
	});
}

test("idle acceptance reports started without waiting for the turn", { timeout: 30000 }, async () => {
	const { session, dispose } = await harness([]);
	try {
		const outcome = await receipt(session, "saccade smoke: idle start");
		assert.equal(outcome, "started");
		assert.equal(
			session.getSteeringMessages().length,
			0,
			"an idle-start input enters the run, not the steer queue",
		);
	} finally {
		dispose();
	}
});

test("busy acceptance reports queued at steer insertion, before the turn", { timeout: 30000 }, async () => {
	const { session, dispose } = await harness([]);
	try {
		const active = new Promise<void>((resolve) => {
			session.subscribe((event: any) => {
				if (event.type === "agent_start") resolve();
			});
		});
		const settling = session.prompt("saccade smoke: hold the run open").catch(() => {
			// the smoke key cannot stream; the failure is downstream of
			// the acceptance under test
		});
		await active;
		const outcome = await receipt(session, "saccade smoke: busy steer");
		assert.equal(outcome, "queued");
		assert.ok(
			session.getSteeringMessages().includes("saccade smoke: busy steer"),
			"the steer sits in the queue at receipt time",
		);
		session.abort();
		await settling;
	} finally {
		dispose();
	}
});

test("an input-handler interception reports handled", { timeout: 30000 }, async () => {
	const { session, dispose } = await harness([
		(pi: any) => {
			pi.on("input", () => ({ action: "handled" }));
		},
	]);
	try {
		const outcome = await receipt(session, "saccade smoke: intercepted");
		assert.equal(outcome, "handled");
		assert.equal(session.getSteeringMessages().length, 0);
	} finally {
		dispose();
	}
});

test("a compaction-window rejection never fires the receipt", { timeout: 30000 }, async () => {
	const { session, dispose } = await harness([]);
	try {
		// enough fodder that compaction preparation finds a cut point:
		// keepRecentTokens defaults well above one short message, so the
		// appended bodies carry real bulk
		for (const index of [0, 1, 2]) {
			session.sessionManager.appendMessage({
				role: "user",
				content: `saccade smoke: compaction fodder ${index} ${"fodder ".repeat(12000)}`,
				timestamp: Date.now(),
			});
		}
		const inWindow = new Promise<void>((resolve) => {
			session.subscribe((event: any) => {
				if (event.type === "compaction_start") resolve();
			});
		});
		const settling = session.compact().catch(() => {
			// the summarization cannot succeed against the smoke key
		});
		await inWindow;
		await assert.rejects(
			receipt(session, "saccade smoke: rejected in the window"),
			/compaction is in progress/,
			"the compaction window rejects before queueing",
		);
		session.abortCompaction();
		await settling;
	} finally {
		dispose();
	}
});

test("the adaptation never leaks into child Node programs importing the same SDK", { timeout: 30000 }, async () => {
	const { session, dispose } = await harness([]);
	try {
		// the parent process carries the composition's --import: its
		// sendUserMessage is the adapted one
		const outcome = await receipt(session, "saccade smoke: parent carries the preload");
		assert.equal(outcome, "started");
		// a child Node program importing the SAME pinned SDK runs with
		// a clean environment - the preload is argument-scoped, so the
		// child's own module copy is untransformed
		const probe = "import(\"@earendil-works/pi-coding-agent\").then((m) => { const s = m.AgentSession.prototype.sendUserMessage.toString(); process.stdout.write(String(s.includes(\"preflightResult: options?.preflightResult\"))); });";
		const childOptions = {
			cwd: dirname(fileURLToPath(import.meta.url)),
			env: { ...process.env, NODE_OPTIONS: "" },
			// the pinned SDK's ambient node:child_process shim types the
			// spawn minimally; this smoke needs the pipes
			stdio: ["pipe", "pipe", "pipe"],
		};
		const child: any = spawn(process.execPath, ["--input-type=module", "--eval", probe], childOptions);
		let out = "";
		let errOut = "";
		child.stdout.on("data", (chunk: unknown) => {
			out += String(chunk);
		});
		child.stderr.on("data", (chunk: unknown) => {
			errOut += String(chunk);
		});
		await new Promise<void>((resolve, reject) => {
			child.on("exit", (code: number) =>
				code === 0
					? resolve()
					: reject(new Error(`child exited ${code}: ${errOut.slice(0, 300)}`)),
			);
		});
		assert.equal(out, "false", "the child's SDK copy must be untransformed");
	} finally {
		dispose();
	}
});

test("an ordinary option-less call queues exactly as before", { timeout: 30000 }, async () => {
	const { session, dispose } = await harness([]);
	try {
		// the option-less call fires in the same microtask window as the
		// busy receipt scenario: immediately after the run activates,
		// before the smoke key's stream failure can end it. The queue's
		// own update event is the durable observation - the steering
		// snapshot drains within the same tick
		const queuedUpdates = new Promise<void>((resolve) => {
			session.subscribe((event: any) => {
				if (event.type === "queue_update") resolve();
			});
		});
		const settling = session.prompt("saccade smoke: hold for the plain call").catch(() => {});
		const activated = new Promise<void>((resolve) => {
			session.subscribe((event: any) => {
				if (event.type === "agent_start") resolve();
			});
		});
		await activated;
		session.sendUserMessage("saccade smoke: plain call", { deliverAs: "steer" });
		await queuedUpdates;
		session.abort();
		await settling;
	} finally {
		dispose();
	}
});
