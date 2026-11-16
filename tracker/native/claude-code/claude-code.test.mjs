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
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	symlinkSync,
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
const worktreeTemplate = readFileSync(
	join(here, "saccade-cc-worktree.template"),
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

test("the hook presents the host-injected terminal pair with session-scoped provenance", async () => {
	// the pair mirrors the pi face: adopted is false only when the
	// launch word names this exact session - a nested session in the
	// same pane inherits the word and still reads adopted
	const cases = [
		{ word: sessionStartInput.session_id, adopted: false },
		{ word: "older9999", adopted: true },
	];
	for (const c of cases) {
		const tracker = await fakeTracker();
		const dir = scratch();
		const hook = buildHook(dir);
		const envFile = join(dir, "env");
		const run = await runHook(hook, sessionStartInput, {
			SACCADE_SERVER: tracker.url,
			CLAUDE_ENV_FILE: envFile,
			HC_TERMINAL_ID: "t7",
			SACCADE_LAUNCHED_SESSION: c.word,
		});
		assert.equal(run.status, 0, run.stderr);
		assert.deepEqual(tracker.attachRequests()[0].body.terminal, {
			address: "t7",
			adopted: c.adopted,
		});
		await tracker.close();
		rmSync(dir, { recursive: true, force: true });
	}
});

// spawn, never spawnSync: the fake tracker answers on this same
// event loop, which a synchronous wait would block forever. The
// worktree hooks take their mode as argv; the SessionStart hook
// takes none.
function runHook(hook, input, env, args = []) {
	const argv = typeof args === "string" ? [args] : args;
	return new Promise((resolve) => {
		const child = spawn(hook, argv, {
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
		// the pair is presenter-controlled: bare means no host disclosure
		HC_TERMINAL_ID: "",
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

// a bare PATH dir: only the tools named (bash always, for the
// wrapper's own shebang) and never sac - the root ask fails quietly
// and the wrapper must still launch exactly as it did without the
// grant
function emptyBin(dir, tools = []) {
	const bin = join(dir, "empty-bin");
	mkdirSync(bin, { recursive: true });
	for (const tool of ["bash", ...tools]) {
		const found = spawnSync("sh", ["-c", `command -v ${tool}`], {
			encoding: "utf8",
		});
		assert.equal(found.status, 0, `the dev shell carries ${tool}`);
		symlinkSync(found.stdout.trim(), join(bin, tool));
	}
	return bin;
}

// a fake sac on a PATH the wrapper resolves first: records its argv,
// answers what the scripted run demands - the list's json face
// carrying the root field, or a refusal. The worktree hooks resolve
// sac from the same convention.
function fakeSac(dir, script) {
	const bin = join(dir, "bin");
	mkdirSync(bin, { recursive: true });
	const sac = join(bin, "sac");
	writeFileSync(sac, `#!/usr/bin/env bash\n${script}\n`);
	chmodSync(sac, 0o755);
	return bin;
}

function sacPathEnv(bin) {
	return { PATH: `${bin}:${process.env.PATH}` };
}

test("the wrapper injects the settings ahead of argv preserved verbatim", () => {
	const dir = scratch();
	const settings = join(dir, "settings.json");
	const argvFile = join(dir, "argv");
	const recorder = fakeClaude(dir, `printf '%s\\n' "$@" > "$RECORD"`);
	const wrapperPath = buildWrapper(dir, { settings, recorder });
	// a PATH with no sac on it: the root ask fails quietly, no flag is
	// added, and the recorded argv is exactly the packaged injection
	const run = spawnSync(wrapperPath, ["--model", "opus", "hello world"], {
		env: { ...process.env, RECORD: argvFile, PATH: emptyBin(dir) },
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
	const settingsText = substitute(settingsTemplate, {
		attachHook: hookPath,
		worktreeHook: "/nix/store/fake-hook/bin/saccade-cc-worktree",
	});
	const settings = JSON.parse(settingsText);
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

test("the substituted settings wire the worktree hooks and carry no permissions grant", () => {
	const worktreePath = "/nix/store/fake-hook/bin/saccade-cc-worktree";
	const settingsText = substitute(settingsTemplate, {
		attachHook: "/nix/store/fake-hook/bin/saccade-cc-attach",
		worktreeHook: worktreePath,
	});
	const settings = JSON.parse(settingsText);
	// both worktree events: one group each, no matcher (the events
	// filter nothing), the packaged script carrying the mode as argv
	for (const [event, mode] of [
		["WorktreeCreate", "create"],
		["WorktreeRemove", "remove"],
	]) {
		const groups = settings.hooks[event];
		assert.equal(groups.length, 1);
		assert.equal(groups[0].matcher, undefined);
		const entry = groups[0].hooks[0];
		assert.equal(entry.type, "command");
		assert.equal(entry.command, worktreePath);
		assert.deepEqual(entry.args, [mode]);
		assert.equal(entry.timeout, 30);
	}
	// no pre-authorized directories: the workspace root is the launch
	// wrapper's --add-dir ask, asked of sac per launch, never a baked
	// permission block
	assert.equal(settings.permissions, undefined);
	assert.equal(settingsText.includes("@"), false, settingsText);
});

// -- the wrapper's workspace-root grant -------------------------------

// one launch against a scripted sac: the fake records the ask and the
// fake claude records the argv it received
function launchWrapper(dir, sacScript) {
	const settings = join(dir, "settings.json");
	const argvFile = join(dir, "argv");
	const sacArgvFile = join(dir, "sac-argv");
	const bin = fakeSac(dir, sacScript);
	const recorder = fakeClaude(dir, `printf '%s\\n' "$@" > "$RECORD"`);
	const wrapperPath = buildWrapper(dir, { settings, recorder });
	const run = spawnSync(wrapperPath, ["--model", "opus"], {
		env: {
			...process.env,
			RECORD: argvFile,
			SAC_RECORD: sacArgvFile,
			...sacPathEnv(bin),
		},
		encoding: "utf8",
	});
	return {
		run,
		settings,
		claudeArgv: () => readFileSync(argvFile, "utf8").split("\n"),
		sacArgv: () => readFileSync(sacArgvFile, "utf8").split("\n"),
	};
}

const stateDir = "/home/someone/state/saccade/saccade-f1f3b490";

const rootedAnswer = (root) =>
	`printf '%s\\n' "$@" > "$SAC_RECORD"\nprintf '%s\\n' '{"workspaces":[],"cursor":null,"root":"${root}"}'`;

const noAddDir = (argv) => {
	assert.ok(!argv.includes("--add-dir"), `no --add-dir in: ${argv.join(" ")}`);
};

const neverStateDir = (argv) => {
	const added = argv[argv.indexOf("--add-dir") + 1];
	assert.ok(
		added === undefined || !added.startsWith(stateDir) || added.endsWith("worktrees"),
		`never the state directory itself: ${argv.join(" ")}`,
	);
};

test("inside a tracked repository the wrapper adds the named root as --add-dir", () => {
	const dir = scratch();
	const root = `${stateDir}/worktrees`;
	const { run, settings, claudeArgv, sacArgv } = launchWrapper(dir, rootedAnswer(root));
	assert.equal(run.status, 0, run.stderr);
	assert.equal(run.stderr, "", "the launch adds no noise of its own");
	// the small read: workspace list's json face, nothing else
	assert.deepEqual(sacArgv(), ["workspace", "list", "--json", ""]);
	// the grant rides ahead of the caller's argv, after the settings
	assert.deepEqual(claudeArgv(), [
		"--settings",
		settings,
		"--add-dir",
		root,
		"--model",
		"opus",
		"",
	]);
	// the root is the worktrees home, never the state directory that
	// holds every tracker's database
	neverStateDir(claudeArgv());
	rmSync(dir, { recursive: true, force: true });
});

test("outside a tracked repository the wrapper adds nothing and stays silent", () => {
	const dir = scratch();
	const { run, settings, claudeArgv, sacArgv } = launchWrapper(
		dir,
		`printf '%s\\n' "$@" > "$SAC_RECORD"
echo 'fatal: not a git repository' >&2
exit 1`,
	);
	assert.equal(run.status, 0, run.stderr);
	// the read failed: no flag, and the wrapper's own stderr stays
	// empty - a launch never carries the refusal's noise
	assert.equal(run.stderr, "");
	assert.deepEqual(claudeArgv(), [
		"--settings",
		settings,
		"--model",
		"opus",
		"",
	]);
	noAddDir(claudeArgv());
	assert.deepEqual(sacArgv(), ["workspace", "list", "--json", ""]);
	rmSync(dir, { recursive: true, force: true });
});

test("a rootless json answer adds no --add-dir", () => {
	const dir = scratch();
	const { run, claudeArgv } = launchWrapper(
		dir,
		`printf '%s\\n' '{"workspaces":[],"cursor":null}'`,
	);
	assert.equal(run.status, 0, run.stderr);
	assert.deepEqual(claudeArgv(), [
		"--settings",
		join(dir, "settings.json"),
		"--model",
		"opus",
		"",
	]);
	noAddDir(claudeArgv());
	neverStateDir(claudeArgv());
	rmSync(dir, { recursive: true, force: true });
});

test("a non-json answer adds no --add-dir", () => {
	// a sac that answers garbage where the root belongs: jq reads no
	// string, the wrapper adds nothing
	const dir = scratch();
	const { run, claudeArgv } = launchWrapper(dir, `printf '%s\\n' 'not json at all'`);
	assert.equal(run.status, 0, run.stderr);
	assert.deepEqual(claudeArgv(), [
		"--settings",
		join(dir, "settings.json"),
		"--model",
		"opus",
		"",
	]);
		noAddDir(claudeArgv());
	rmSync(dir, { recursive: true, force: true });
});

test("a missing sac never breaks the launch", () => {
	const dir = scratch();
	const settings = join(dir, "settings.json");
	const argvFile = join(dir, "argv");
	const recorder = fakeClaude(dir, `printf '%s\\n' "$@" > "$RECORD"`);
	const wrapperPath = buildWrapper(dir, { settings, recorder });
	// jq alone on the PATH: the ask fails for want of sac, the launch
	// proceeds with no --add-dir
	const bin = emptyBin(dir, ["jq"]);
	const run = spawnSync(wrapperPath, ["--model", "opus"], {
		env: { ...process.env, RECORD: argvFile, PATH: bin },
		encoding: "utf8",
	});
	assert.equal(run.status, 0, run.stderr);
	assert.equal(run.stderr, "");
	noAddDir(readFileSync(argvFile, "utf8").split("\n"));
	rmSync(dir, { recursive: true, force: true });
});

// -- the worktree hooks ----------------------------------------------

const worktreeCreateInput = {
	session_id: "a1b2c3d4-1111-4222-8333-444455556666",
	transcript_path: "/home/someone/.claude/projects/p/a1b2c3d4.jsonl",
	cwd: "/home/someone/ws",
	hook_event_name: "WorktreeCreate",
	name: "trial-name",
};

const worktreeRemoveInput = {
	session_id: "a1b2c3d4-1111-4222-8333-444455556666",
	transcript_path: "/home/someone/.claude/projects/p/a1b2c3d4.jsonl",
	cwd: "/home/someone/ws",
	hook_event_name: "WorktreeRemove",
	worktree_path: "/home/someone/state/saccade/saccade-f1f3b490/worktrees/ws-a1b2c3",
};

// the same placeholder substitution, applied to the worktree script:
// the template carries no packaging-time placeholder, so the
// substituted copy is the template byte for byte
function buildWorktreeHook(dir) {
	const path = join(dir, "saccade-cc-worktree");
	writeFileSync(path, substitute(worktreeTemplate, {}));
	chmodSync(path, 0o755);
	return path;
}

test("the create hook provisions through sac and prints the path as the only stdout line", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const argvFile = join(dir, "argv");
	const created = join(
		"/home/someone/state/saccade/saccade-f1f3b490/worktrees",
		"ws-a1b2c3",
	);
	const bin = fakeSac(
		dir,
		`printf '%s\\n' "$@" > "$RECORD"\nprintf '%s\\n' '{"created":true,"workspace":{"path":"${created}"}}'`,
	);
	const run = await runHook(hook, worktreeCreateInput, {
		...sacPathEnv(bin),
		RECORD: argvFile,
	}, "create");
	assert.equal(run.status, 0, run.stderr);
	// the door the trial verified: workspace new with the event's name,
	// the json face asked for, nothing else
	assert.deepEqual(readFileSync(argvFile, "utf8").split("\n"), [
		"workspace",
		"new",
		"trial-name",
		"--json",
		"",
	]);
	assert.equal(run.stdout, `${created}\n`);
	rmSync(dir, { recursive: true, force: true });
});

test("a refused creation fails closed with sac's words and no path on stdout", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const bin = fakeSac(dir, `echo 'workspace label refused' >&2; exit 9`);
	const run = await runHook(hook, worktreeCreateInput, sacPathEnv(bin), "create");
	assert.equal(run.status, 2);
	assert.match(run.stderr, /workspace label refused/);
	assert.match(run.stderr, /no worktree is provisioned/);
	assert.equal(run.stdout, "");
	rmSync(dir, { recursive: true, force: true });
});

test("a create payload without a name fails closed before sac runs", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const argvFile = join(dir, "argv");
	const bin = fakeSac(
		dir,
		`printf '%s\\n' "$@" > "$RECORD"\nexit 0`,
	);
	const run = await runHook(
		hook,
		{ ...worktreeCreateInput, name: "" },
		{
			...sacPathEnv(bin),
			RECORD: argvFile,
		},
		"create",
	);
	assert.equal(run.status, 2);
	assert.match(run.stderr, /carries no worktree name/);
	assert.equal(run.stdout, "");
	assert.equal(existsSync(argvFile), false, "sac never ran");
	rmSync(dir, { recursive: true, force: true });
});

test("a malformed create payload fails closed with no provisioning", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const bin = fakeSac(dir, "exit 0");
	const run = await runHook(hook, "definitely not json", sacPathEnv(bin), "create");
	assert.equal(run.status, 2);
	assert.match(run.stderr, /not a hook payload/);
	assert.equal(run.stdout, "");
	rmSync(dir, { recursive: true, force: true });
});

test("the remove hook cleans through sac and rides its exit code", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const argvFile = join(dir, "argv");
	const bin = fakeSac(
		dir,
		`printf '%s\\n' "$@" > "$RECORD"\necho 'workspace w-9 cleaned'\nexit 0`,
	);
	const run = await runHook(
		hook,
		worktreeRemoveInput,
		{
			...sacPathEnv(bin),
			RECORD: argvFile,
		},
		"remove",
	);
	assert.equal(run.status, 0, run.stderr);
	assert.deepEqual(readFileSync(argvFile, "utf8").split("\n"), [
		"workspace",
		"clean",
		worktreeRemoveInput.worktree_path,
		"",
	]);
	assert.match(run.stdout, /workspace w-9 cleaned/);
	rmSync(dir, { recursive: true, force: true });
});

test("a refused clean rides sac's own code with its words on stderr", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const bin = fakeSac(dir, `echo 'workspace_work_present' >&2; exit 6`);
	const run = await runHook(hook, worktreeRemoveInput, sacPathEnv(bin), "remove");
	assert.equal(run.status, 6);
	assert.match(run.stderr, /workspace_work_present/);
	rmSync(dir, { recursive: true, force: true });
});

test("a remove payload without a worktree_path fails closed before sac runs", async () => {
	const dir = scratch();
	const hook = buildWorktreeHook(dir);
	const argvFile = join(dir, "argv");
	const bin = fakeSac(
		dir,
		`printf '%s\\n' "$@" > "$RECORD"\nexit 0`,
	);
	const run = await runHook(
		hook,
		{ ...worktreeRemoveInput, worktree_path: "" },
		{
			...sacPathEnv(bin),
			RECORD: argvFile,
		},
		"remove",
	);
	assert.equal(run.status, 2);
	assert.match(run.stderr, /carries no worktree_path/);
	assert.equal(existsSync(argvFile), false, "sac never ran");
	rmSync(dir, { recursive: true, force: true });
});

test("the substituted hook script parses as bash and leaves no placeholders", async () => {
	const dir = scratch();
	const hook = buildHook(dir);
	const text = readFileSync(hook, "utf8");
	assert.equal(/@[a-zA-Z]+@/.test(text), false);
	const probe = spawnSync("bash", ["-n", hook], { encoding: "utf8" });
	assert.equal(probe.status, 0, probe.stderr);
	const worktree = buildWorktreeHook(dir);
	const worktreeText = readFileSync(worktree, "utf8");
	assert.equal(/@[a-zA-Z]+@/.test(worktreeText), false);
	const worktreeProbe = spawnSync("bash", ["-n", worktree], {
		encoding: "utf8",
	});
	assert.equal(worktreeProbe.status, 0, worktreeProbe.stderr);
	rmSync(dir, { recursive: true, force: true });
});
