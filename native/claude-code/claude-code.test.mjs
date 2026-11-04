// Focused tests for the composed Claude Code harness mechanics: the
// SessionStart identity hook against a fake tracker (version proof,
// attach, env channel), and the wrapper template's settings injection
// and collision refusal against a fake claude recorder. Run inside the
// dev shell (jq, curl, node on PATH):
//   node --test native/claude-code/claude-code.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import {
	chmodSync,
	existsSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const hookTemplate = readFileSync(
	join(here, "saccade-cc-attach.template"),
	"utf8",
);
const wrapperTemplate = readFileSync(
	join(here, "saccade-claude-code.template"),
	"utf8",
);
const settingsTemplate = readFileSync(
	join(here, "settings.json.template"),
	"utf8",
);

// the same placeholder substitution the flake's replaceVars performs
function substitute(template, replacements) {
	return template.replace(/@([a-zA-Z]+)@/g, (whole, name) =>
		Object.hasOwn(replacements, name) ? replacements[name] : whole,
	);
}

function scratch() {
	return mkdtempSync(join(tmpdir(), "saccade-cc-test-"));
}

// the packaged hook for one expected server version line
function buildHook(dir, { major = "0", minor = "18" } = {}) {
	const path = join(dir, "saccade-cc-attach");
	writeFileSync(
		path,
		substitute(hookTemplate, { expectMajor: major, expectMinor: minor }),
	);
	chmodSync(path, 0o755);
	return path;
}

const attachReply = {
	registered: true,
	incarnation: 42,
	num: 7,
	records: [],
};

const versionReply = { version: "0.18.0", db: "/db/saccade.db" };

const sessionStartInput = {
	session_id: "a1b2c3d4-1111-4222-8333-444455556666",
	transcript_path: "/home/someone/.claude/projects/p/a1b2c3d4.jsonl",
	cwd: "/home/someone/ws",
	hook_event_name: "SessionStart",
	source: "startup",
};

// one fake tracker: routes /api/v1/version and the attach POST,
// records every request, replies with the programmed status and body
function fakeTracker({ version = [200, versionReply], attach = [200, attachReply] } = {}) {
	let server;
	const requests = [];
	const ready = new Promise((resolve) => {
		server = createServer((req, res) => {
			const chunks = [];
			req.on("data", (chunk) => chunks.push(chunk));
			req.on("end", () => {
				const isVersion = req.url === "/api/v1/version";
				const [status, body] = isVersion ? version : attach;
				requests.push({
					method: req.method,
					url: req.url,
					body: isVersion
						? undefined
						: JSON.parse(Buffer.concat(chunks).toString("utf8")),
				});
				res.writeHead(status, { "content-type": "application/json" });
				res.end(JSON.stringify(body));
			});
		});
		server.listen(0, "127.0.0.1", () => resolve());
	});
	return ready.then(
		() =>
			new Promise((resolveDone) => {
				const { port } = server.address();
				resolveDone({
					url: `http://127.0.0.1:${port}`,
					requests,
					attachRequests: () => requests.filter((r) => r.url !== "/api/v1/version"),
					versionRequests: () => requests.filter((r) => r.url === "/api/v1/version"),
					close: () =>
						new Promise((done) => {
							server.close(() => done());
						}),
				});
			}),
	);
}

const INVAL = "export SACCADE_INCARNATION_SEQ=''\n";

// spawn, never spawnSync: the fake tracker answers on this same
// event loop, which a synchronous wait would block forever
function runHook(hook, input, env) {
	return new Promise((resolve) => {
		const child = spawn(hook, {
			env: { ...process.env, ...env },
		});
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (chunk) => (stdout += chunk));
		child.stderr.on("data", (chunk) => (stderr += chunk));
		child.stdin.on("error", () => {});
		child.on("close", (code) =>
			resolve({ status: code, stdout, stderr }),
		);
		child.stdin.end(typeof input === "string" ? input : JSON.stringify(input));
	});
}

test("a proven server gets the version request then the attach, and the env channel gets the seq", async () => {
	const tracker = await fakeTracker();
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 0, run.stderr);
	assert.deepEqual(tracker.requests.map((r) => [r.method, r.url]), [
		["GET", "/api/v1/version"],
		["POST", "/api/v1/native/attach"],
	]);
	assert.deepEqual(tracker.attachRequests()[0].body, {
		native_id: sessionStartInput.session_id,
		session_file: sessionStartInput.transcript_path,
		harness: "claude-code",
	});
	assert.equal(
		readFileSync(envFile, "utf8"),
		INVAL + "export SACCADE_INCARNATION_SEQ='42'\n",
	);
	const out = JSON.parse(run.stdout);
	assert.match(out.systemMessage, /attached as i-7$/);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a stale export from a prior run is invalidated, then rebinds on success", async () => {
	const tracker = await fakeTracker();
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	writeFileSync(envFile, "export SACCADE_INCARNATION_SEQ='9'\n");
	const run = await runHook(
		hook,
		{ ...sessionStartInput, source: "clear" },
		{
			SACCADE_SERVER: tracker.url,
			CLAUDE_ENV_FILE: envFile,
		},
	);
	assert.equal(run.status, 0, run.stderr);
	assert.equal(
		readFileSync(envFile, "utf8"),
		"export SACCADE_INCARNATION_SEQ='9'\n" +
			INVAL +
			"export SACCADE_INCARNATION_SEQ='42'\n",
	);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a failed rebind leaves the stale export invalidated with no numeric line", async () => {
	const tracker = await fakeTracker({
		attach: [
			409,
			{
				error: {
					code: "saved_session_mismatch",
					detail: "native session a1 is registered elsewhere",
				},
			},
		],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	writeFileSync(envFile, "export SACCADE_INCARNATION_SEQ='9'\n");
	const run = await runHook(
		hook,
		{ ...sessionStartInput, source: "fork" },
		{
			SACCADE_SERVER: tracker.url,
			CLAUDE_ENV_FILE: envFile,
		},
	);
	assert.equal(run.status, 2);
	assert.match(run.stderr, /saved_session_mismatch/);
	assert.match(run.stderr, /registered elsewhere/);
	assert.match(run.stderr, /source fork/);
	assert.equal(
		readFileSync(envFile, "utf8"),
		"export SACCADE_INCARNATION_SEQ='9'\n" + INVAL,
	);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a missing CLAUDE_ENV_FILE fails before any request", async () => {
	const tracker = await fakeTracker();
	const dir = scratch();
	const hook = buildHook(dir);
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /CLAUDE_ENV_FILE is not set/);
	assert.equal(run.stdout, "");
	assert.equal(tracker.requests.length, 0);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("an unwritable CLAUDE_ENV_FILE fails before any request", async () => {
	const tracker = await fakeTracker();
	const dir = scratch();
	const hook = buildHook(dir);
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: join(dir, "missing-dir", "env"),
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /could not be written/);
	assert.equal(tracker.requests.length, 0);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("no SACCADE_SERVER routing fails after invalidation only", async () => {
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: "",
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /SACCADE_SERVER/);
	assert.equal(readFileSync(envFile, "utf8"), INVAL);
	rmSync(dir, { recursive: true, force: true });
});

test("an unreachable tracker fails with no attach", async () => {
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		// port 1 on loopback refuses; nothing listens there
		SACCADE_SERVER: "http://127.0.0.1:1",
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /saccade:/);
	assert.equal(readFileSync(envFile, "utf8"), INVAL);
	rmSync(dir, { recursive: true, force: true });
});

test("a server on a different minor is rejected before the attach", async () => {
	const tracker = await fakeTracker({
		version: [200, { version: "0.17.1", db: "/db" }],
	});
	const dir = scratch();
	const hook = buildHook(dir, { major: "0", minor: "18" });
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /different minor/);
	assert.equal(tracker.attachRequests().length, 0);
	assert.equal(readFileSync(envFile, "utf8"), INVAL);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a server on a different major line is rejected as unproven", async () => {
	const tracker = await fakeTracker({
		version: [200, { version: "1.18.0", db: "/db" }],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /no compatibility rule spans major lines/);
	assert.equal(tracker.attachRequests().length, 0);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("an unreadable version string is rejected as unproven", async () => {
	const tracker = await fakeTracker({
		version: [200, { version: "dev" }],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: join(dir, "env"),
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /does not read as major\.minor\.patch/);
	assert.equal(tracker.attachRequests().length, 0);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a missing version field is rejected as unproven", async () => {
	const tracker = await fakeTracker({
		version: [200, { db: "/db" }],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: join(dir, "env"),
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /carries no version string/);
	assert.equal(tracker.attachRequests().length, 0);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a failing version route is rejected before the attach", async () => {
	const tracker = await fakeTracker({ version: [404, {}] });
	const dir = scratch();
	const hook = buildHook(dir);
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: join(dir, "env"),
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /unproven/);
	assert.equal(tracker.attachRequests().length, 0);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a same-minor different-patch server is compatible", async () => {
	const tracker = await fakeTracker({
		version: [200, { version: "0.18.9", db: "/db" }],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 0, run.stderr);
	assert.equal(tracker.attachRequests().length, 1);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("malformed hook input fails closed", async () => {
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, "definitely not json", {
		SACCADE_SERVER: "http://127.0.0.1:1",
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /hook payload/);
	assert.equal(existsSync(envFile), false);
	rmSync(dir, { recursive: true, force: true });
});

test("a payload without session_id fails closed", async () => {
	const tracker = await fakeTracker();
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(
		hook,
		{ ...sessionStartInput, session_id: "" },
		{
			SACCADE_SERVER: tracker.url,
			CLAUDE_ENV_FILE: envFile,
		},
	);
	assert.equal(run.status, 2);
	assert.match(run.stderr, /no session_id/);
	assert.equal(existsSync(envFile), false);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a relative transcript_path fails closed with the session evidence", async () => {
	const tracker = await fakeTracker();
	const dir = scratch();
	const hook = buildHook(dir);
	const run = await runHook(
		hook,
		{ ...sessionStartInput, transcript_path: "relative/x.jsonl" },
		{ SACCADE_SERVER: tracker.url },
	);
	assert.equal(run.status, 2);
	assert.match(run.stderr, /not absolute/);
	assert.match(run.stderr, /source startup/);
	assert.equal(tracker.requests.length, 0, "no request left the machine");
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a reply without an incarnation fails with the identity invalidated", async () => {
	const tracker = await fakeTracker({
		attach: [200, { registered: true, num: 7 }],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /no incarnation/);
	assert.equal(readFileSync(envFile, "utf8"), INVAL);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

test("a fractional incarnation never reaches the env file", async () => {
	const tracker = await fakeTracker({
		attach: [200, { ...attachReply, incarnation: 42.5 }],
	});
	const dir = scratch();
	const hook = buildHook(dir);
	const envFile = join(dir, "env");
	const run = await runHook(hook, sessionStartInput, {
		SACCADE_SERVER: tracker.url,
		CLAUDE_ENV_FILE: envFile,
	});
	assert.equal(run.status, 2);
	assert.match(run.stderr, /not a record position/);
	assert.equal(readFileSync(envFile, "utf8"), INVAL);
	await tracker.close();
	rmSync(dir, { recursive: true, force: true });
});

// -- the wrapper ------------------------------------------------------

function buildWrapper(dir, { settings, recorder }) {
	const wrapperPath = join(dir, "saccade-claude-code");
	const text = substitute(wrapperTemplate, {
		claude: recorder,
		settings,
		jq: "/nix/store/fake-jq",
		curl: "/nix/store/fake-curl",
	});
	writeFileSync(wrapperPath, text);
	chmodSync(wrapperPath, 0o755);
	return wrapperPath;
}

function fakeClaude(dir, script) {
	const recorder = join(dir, "fake-claude");
	writeFileSync(recorder, `#!/usr/bin/env bash\n${script}\n`);
	chmodSync(recorder, 0o755);
	return recorder;
}

test("the wrapper injects the settings ahead of argv preserved verbatim", () => {
	const dir = scratch();
	const settings = join(dir, "settings.json");
	const argvFile = join(dir, "argv");
	const recorder = fakeClaude(dir, `printf '%s\\n' "$@" > "$RECORD"`);
	const wrapperPath = buildWrapper(dir, { settings, recorder });
	const run = spawnSync(wrapperPath, ["--model", "opus", "hello world"], {
		env: { ...process.env, RECORD: argvFile },
		encoding: "utf8",
	});
	assert.equal(run.status, 0, run.stderr);
	assert.deepEqual(readFileSync(argvFile, "utf8").split("\n"), [
		"--settings",
		settings,
		"--model",
		"opus",
		"hello world",
		"",
	]);
	rmSync(dir, { recursive: true, force: true });
});

test("the wrapper refuses an argv carrying --settings", () => {
	const dir = scratch();
	const recorder = fakeClaude(dir, "exit 9");
	const wrapperPath = buildWrapper(dir, {
		settings: join(dir, "settings.json"),
		recorder,
	});
	const run = spawnSync(wrapperPath, ["--settings", "/tmp/x.json"], {
		encoding: "utf8",
	});
	assert.equal(run.status, 1);
	assert.match(run.stderr, /--settings/);
	rmSync(dir, { recursive: true, force: true });
});

test("the wrapper refuses the --settings= form too", () => {
	const dir = scratch();
	const recorder = fakeClaude(dir, "exit 9");
	const wrapperPath = buildWrapper(dir, {
		settings: join(dir, "settings.json"),
		recorder,
	});
	const run = spawnSync(wrapperPath, ["--settings=/tmp/x.json"], {
		encoding: "utf8",
	});
	assert.equal(run.status, 1);
	assert.match(run.stderr, /--settings/);
	rmSync(dir, { recursive: true, force: true });
});

test("the packaged settings wire the hook on every SessionStart source", () => {
	const hookPath = "/nix/store/fake-hook/bin/saccade-cc-attach";
	const settings = JSON.parse(
		substitute(settingsTemplate, { attachHook: hookPath }),
	);
	const groups = settings.hooks.SessionStart;
	assert.equal(groups.length, 1);
	assert.equal(groups[0].matcher, "startup|resume|clear|compact|fork");
	const entry = groups[0].hooks[0];
	assert.equal(entry.type, "command");
	assert.equal(entry.command, hookPath);
	assert.deepEqual(entry.args, []);
	assert.equal(entry.timeout, 30);
	assert.equal(entry.onFailure, "block");
});

test("the substituted hook script parses as bash and leaves no placeholders", async () => {
	const dir = scratch();
	const hook = buildHook(dir);
	const text = readFileSync(hook, "utf8");
	assert.equal(/@[a-zA-Z]+@/.test(text), false);
	const probe = spawnSync("bash", ["-n", hook], { encoding: "utf8" });
	assert.equal(probe.status, 0, probe.stderr);
	rmSync(dir, { recursive: true, force: true });
});
