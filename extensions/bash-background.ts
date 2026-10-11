/**
 * Background execution for the bash tool: `background: true` runs a command
 * detached, returns immediately, and delivers the result as a follow-up
 * message when it finishes. Ordinary calls delegate to pi's built-in bash
 * unchanged. `bash_stop {id}` stops a command this session launched.
 *
 * Loaded explicitly — see extensions/README.md. Nothing here installs itself.
 */

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { closeSync, fstatSync, openSync, readSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Type } from "typebox";
import {
	createBashToolDefinition,
	getShellConfig,
	type ExtensionAPI,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";

const MARKER = "__BASH_BG_EXIT__";
const POLL_MS = 500;
const STOP_GRACE_MS = 2000;
const STOP_POLL_STEP_MS = 100;
const AFTER_KILL_MS = 1000;
const TAIL_BYTES = 4096;
const TAIL_LINES = 20;
const MAX_TIMEOUT_MS = 2_147_483_647;
const SESSION_ENV_KEYS = ["PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"] as const;

// The wrapper runs the command in the caller's shell (argv transport, never
// interpolated) and appends the nonce-tagged exit line, so the exit code
// survives extension reload: the reloaded instance holds no ChildProcess, it
// finds the marker in the log.
const WRAPPER = [
	"cmd=$1; nonce=$2; shift 2",
	'"$@" "$cmd"; code=$?',
	`printf "\\n${MARKER}:%s:%s\\n" "$nonce" "$code"`,
].join("\n");

type JobState = "running" | "stopping" | "exited" | "unknown" | "stopped";

// Plain data only: this map is the reload continuity, so a reloaded instance
// can take running jobs over from marker reads alone. Never store a
// ChildProcess or a captured pi here.
interface BgJob {
	id: string;
	command: string;
	logPath: string;
	pid: number;
	nonce: string;
	startedAt: number;
	deadlineAt?: number;
	state: JobState;
	stopReason?: "stopped" | "timeout";
	exitCode?: number;
	endedAt?: number;
	notified: boolean;
	stopping: boolean;
	groupGone: boolean;
}

interface Registry {
	jobs: Map<string, BgJob>;
	nextId: number;
	poller?: ReturnType<typeof setInterval>;
}

// globalThis under a Symbol.for key because pi loads extensions with
// moduleCache: false — module state dies on /reload, this survives it within
// the same process. It holds no meaning across processes.
const REGISTRY_KEY = Symbol.for("bash-background.jobs");

function registry(): Registry {
	const holder = globalThis as typeof globalThis & { [REGISTRY_KEY]?: Registry };
	const reg = (holder[REGISTRY_KEY] ??= { jobs: new Map<string, BgJob>(), nextId: 1 });
	return reg;
}

let pi: ExtensionAPI | undefined;

function reached(send: () => void): boolean {
	try {
		send();
		return true;
	} catch (error) {
		const code = (error as NodeJS.ErrnoException).code;
		if (code === "ESRCH" || code === "EPERM") return false;
		throw error;
	}
}

function pidAlive(pid: number): boolean {
	return reached(() => process.kill(pid, 0));
}

function groupAlive(pid: number): boolean {
	return reached(() => process.kill(-pid, 0));
}

// Pinned bash convention: invalid and out-of-range timeouts throw the same
// errors the built-in throws; omission means no deadline.
function validateTimeout(timeout: number | undefined): void {
	if (timeout === undefined) return;
	if (!Number.isFinite(timeout) || timeout <= 0) {
		throw new Error("Invalid timeout: must be a finite number of seconds");
	}
	if (timeout * 1000 > MAX_TIMEOUT_MS) {
		throw new Error(`Invalid timeout: maximum is ${MAX_TIMEOUT_MS / 1000} seconds`);
	}
}

function shellEnv(ctx: ExtensionContext | undefined): NodeJS.ProcessEnv {
	const env = { ...process.env };
	for (const key of SESSION_ENV_KEYS) delete env[key];
	if (ctx) {
		env.PI_SESSION_ID = ctx.sessionManager.getSessionId();
		const file = ctx.sessionManager.getSessionFile();
		if (file) env.PI_SESSION_FILE = file;
		if (ctx.model) {
			env.PI_PROVIDER = ctx.model.provider;
			env.PI_MODEL = ctx.model.id;
		}
		if (ctx.thinkingLevel) env.PI_REASONING_LEVEL = ctx.thinkingLevel;
	}
	return env;
}

async function spawnJob(command: string, timeout: number | undefined, ctx: ExtensionContext | undefined): Promise<BgJob> {
	const cwd = ctx?.cwd ?? process.cwd();
	const logPath = join(tmpdir(), `bash-bg-${randomBytes(6).toString("hex")}.log`);
	const fd = openSync(logPath, "wx", 0o600);
	const nonce = randomBytes(8).toString("hex");
	try {
		const { shell, args } = getShellConfig();
		const child = spawn("sh", ["-c", WRAPPER, "bash-bg", command, nonce, shell, ...args], {
			cwd,
			env: shellEnv(ctx),
			detached: true,
			stdio: ["ignore", fd, fd],
		});
		// The armed reply must not lie: spawn failures (missing cwd, ENOENT)
		// surface here as a thrown error, not as a started job. A later error
		// event gets a no-op listener so a dropped handle cannot crash the host.
		await new Promise<void>((resolve, reject) => {
			child.once("spawn", resolve);
			child.once("error", reject);
		});
		child.on("error", () => {});
		child.unref();
		const reg = registry();
		const job: BgJob = {
			id: `bg-${reg.nextId++}`,
			command,
			logPath,
			pid: child.pid!,
			nonce,
			startedAt: Date.now(),
			deadlineAt: timeout !== undefined ? Date.now() + timeout * 1000 : undefined,
			state: "running",
			notified: false,
			stopping: false,
			groupGone: false,
		};
		reg.jobs.set(job.id, job);
		ensurePoller();
		return job;
	} finally {
		closeSync(fd);
	}
}

// Bounded tail read: a log can grow without limit, so nothing here reads the
// whole file. Returns the last bytes plus the exit marker when present.
function readTail(job: BgJob, bytes: number): string {
	let fd: number;
	try {
		fd = openSync(job.logPath, "r");
	} catch {
		return "";
	}
	try {
		const size = fstatSync(fd).size;
		const start = Math.max(0, size - bytes);
		const buffer = Buffer.alloc(size - start);
		readSync(fd, buffer, 0, buffer.length, start);
		return buffer.toString("utf8");
	} finally {
		closeSync(fd);
	}
}

function readMarker(job: BgJob): number | undefined {
	const tail = readTail(job, TAIL_BYTES * 2);
	const line = tail.split("\n").reverse().find((l) => l.startsWith(`${MARKER}:`));
	if (!line) return undefined;
	const parts = line.split(":");
	if (parts[1] !== job.nonce) return undefined;
	const code = Number(parts[2]);
	return Number.isInteger(code) ? code : undefined;
}

function outputTail(job: BgJob): { text: string; cut: boolean } {
	const tail = readTail(job, TAIL_BYTES)
		.split("\n")
		.filter((l) => !l.startsWith(`${MARKER}:`))
		.join("\n")
		.replace(/^\n+/, "")
		.replace(/\n+$/, "");
	if (tail === "") return { text: "", cut: false };
	const lines = tail.split("\n");
	const kept = lines.slice(-TAIL_LINES);
	const cut = lines.length > TAIL_LINES || Buffer.byteLength(tail) > TAIL_BYTES;
	return { text: kept.join("\n"), cut };
}

function oneLine(command: string): string {
	const line = command.replace(/\s*\n\s*/g, " ");
	return line.length > 120 ? `${line.slice(0, 117)}...` : line;
}

function outcome(job: BgJob): string {
	if (job.state === "exited") {
		if (job.exitCode !== 0) return `failed: exited with code ${job.exitCode}`;
		return "exited with code 0";
	}
	if (job.state === "stopped" && job.stopReason === "timeout") {
		return `was stopped: its timeout of ${Math.round(((job.deadlineAt ?? 0) - job.startedAt) / 100) / 10}s elapsed`;
	}
	if (job.state === "unknown") return "ended without reporting an exit status";
	return `was stopped (${job.stopReason ?? "by request"})`;
}

function completionText(job: BgJob): string {
	const seconds = Math.round(((job.endedAt ?? Date.now()) - job.startedAt) / 100) / 10;
	const { text, cut } = outputTail(job);
	const lines = [
		`Background command ${job.id} ${outcome(job)}, ran ${seconds}s.`,
		`Command: ${oneLine(job.command)}`,
	];
	if (text === "") lines.push("(no output)");
	else {
		lines.push(cut ? `Output, last ${TAIL_LINES} lines — full output: ${job.logPath}` : "Output:");
		lines.push(text);
	}
	if (text !== "" && !cut) lines.push(`Full output: ${job.logPath}`);
	return lines.join("\n");
}

function deliver(job: BgJob): void {
	if (!pi || job.notified || job.state === "running" || job.state === "stopping") return;
	job.notified = true;
	try {
		pi.sendMessage(
			{
				customType: "bash-background-result",
				content: completionText(job),
				display: true,
				details: {
					id: job.id,
					state: job.state,
					exitCode: job.exitCode ?? null,
					durationMs: (job.endedAt ?? Date.now()) - job.startedAt,
					logPath: job.logPath,
				},
			},
			// followUp never interrupts a running turn; triggerTurn wakes an idle
			// session. This is the completion carrier, not a steer.
			{ deliverAs: "followUp", triggerTurn: true },
		);
	} catch {
		// A stale handle (mid-reload) must not consume the result: the next
		// poller tick retries through the rebound instance.
		job.notified = false;
	}
}

const killsInFlight = new Map<string, Promise<string>>();

// Returns the observed outcome; the caller words it. Sets terminal state.
function stopJob(job: BgJob, reason: "stopped" | "timeout"): Promise<string> {
	const existing = killsInFlight.get(job.id);
	if (existing) return existing;
	const run = (async (): Promise<string> => {
		job.stopping = true;
		job.stopReason = reason;
		// Marked before the kill so the poller can never deliver a completion
		// for a command the agent explicitly stopped; the stop result itself
		// carries the observed outcome.
		if (reason === "stopped") job.notified = true;
		const marker = readMarker(job);
		if (marker !== undefined) {
			job.state = "exited";
			job.exitCode = marker;
			job.endedAt = Date.now();
			return `finished on its own while being stopped: exited with code ${marker}`;
		}
		if (!groupAlive(job.pid)) {
			job.groupGone = true;
			const late = readMarker(job);
			if (late !== undefined) {
				job.state = "exited";
				job.exitCode = late;
			} else {
				job.state = "unknown";
			}
			job.endedAt = Date.now();
			return job.state === "exited" ? `had already finished: exited with code ${late}` : "had already ended (exit status unknown)";
		}
		reached(() => process.kill(-job.pid, "SIGTERM"));
		let killed = false;
		for (let waited = 0; waited < STOP_GRACE_MS; waited += STOP_POLL_STEP_MS) {
			await new Promise((r) => setTimeout(r, STOP_POLL_STEP_MS));
			if (!groupAlive(job.pid)) {
				killed = true;
				break;
			}
		}
		if (!killed) {
			reached(() => process.kill(-job.pid, "SIGKILL"));
			for (let waited = 0; waited < AFTER_KILL_MS; waited += STOP_POLL_STEP_MS) {
				await new Promise((r) => setTimeout(r, STOP_POLL_STEP_MS));
				if (!groupAlive(job.pid)) {
					killed = true;
					break;
				}
			}
		}
		job.groupGone = killed;
		const late = readMarker(job);
		if (late !== undefined) {
			job.state = "exited";
			job.exitCode = late;
			job.endedAt = Date.now();
			return `finished on its own while being stopped: exited with code ${late}`;
		}
		job.endedAt = Date.now();
		if (killed) {
			job.state = "stopped";
			return reason === "timeout"
				? "stopped at its deadline (command group gone)"
				: "stopped (command group gone)";
		}
		job.state = "stopped";
		return "could not be stopped: the command group survived SIGKILL and may still be running";
	})();
	const clear = (): void => {
		job.stopping = false;
		killsInFlight.delete(job.id);
	};
	killsInFlight.set(job.id, run.then(clear, clear));
	return run;
}

function stopAllJobs(): void {
	for (const job of registry().jobs.values()) {
		if (job.state === "running") void stopJob(job, "stopped");
	}
}

function ensurePoller(): void {
	const reg = registry();
	if (reg.poller !== undefined) return;
	reg.poller = setInterval(pollTick, POLL_MS);
	reg.poller.unref();
}

function stopPoller(): void {
	const reg = registry();
	if (reg.poller !== undefined) clearInterval(reg.poller);
	reg.poller = undefined;
}

function pollTick(): void {
	const reg = registry();
	for (const job of reg.jobs.values()) {
		if (job.state !== "running" || job.stopping) continue;
		if (job.deadlineAt !== undefined && Date.now() >= job.deadlineAt) {
			void stopJob(job, "timeout");
			continue;
		}
		const marker = readMarker(job);
		if (marker !== undefined) {
			job.state = "exited";
			job.exitCode = marker;
			job.endedAt = Date.now();
		} else if (!pidAlive(job.pid)) {
			job.state = "unknown";
			job.endedAt = Date.now();
		}
	}
	for (const job of reg.jobs.values()) {
		deliver(job);
	}
	const pending = [...reg.jobs.values()].some((job) => job.state === "running" || job.state === "stopping" || !job.notified);
	if (!pending) stopPoller();
}

// A reload re-runs the factory against the same registry: drop this module
// copy's stale timer and reconnect delivery to the live handle.
function attach(handle: ExtensionAPI): void {
	pi = handle;
	stopPoller();
	ensurePoller();
}

const BACKGROUND_CONTRACT =
	"Set background:true to run a command detached and continue other work: the call returns at once with a job id and a log path, and the result is delivered to you automatically as a follow-up message when the command finishes — do not poll or sleep-wait; read the log anytime for progress (reading never consumes the result). timeout, if set, limits total runtime including time in the background. Stop a running background command with bash_stop.";

export default function (handle: ExtensionAPI) {
	// Bound at factory time, not only at session_start: a completion may land
	// between the two, and delivery must never wait on lifecycle ordering.
	pi = handle;
	const builtin = createBashToolDefinition(process.cwd());
	handle.registerTool({
		...builtin,
		description: `${builtin.description} ${BACKGROUND_CONTRACT}`,
		parameters: Type.Object({
			...builtin.parameters.properties,
			background: Type.Optional(
				Type.Boolean({ description: "Run detached and deliver the result as an automatic follow-up message when it finishes" }),
			),
		}),
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const input = params as { command: string; timeout?: number; background?: boolean };
			if (input.background !== true) {
				return builtin.execute(toolCallId, { command: input.command, timeout: input.timeout }, signal, onUpdate, ctx);
			}
			validateTimeout(input.timeout);
			// The call's abort signal is deliberately unused: it aborts the tool
			// call, and a background command must outlive the call that started it.
			const job = await spawnJob(input.command, input.timeout, ctx);
			const seconds = input.timeout !== undefined ? input.timeout : undefined;
			return {
				content: [
					{
						type: "text",
						text: [
							`Background command ${job.id} started (pid ${job.pid}).`,
							`Command: ${oneLine(input.command)}`,
							`Output streams to ${job.logPath} — read it anytime for progress; reading never consumes the result.`,
							seconds !== undefined ? `Deadline: ${seconds}s total runtime.` : "No deadline.",
							"The result will be delivered here automatically when it finishes — continue other work instead of polling.",
						].join("\n"),
					},
				],
				details: undefined,
			};
		},
	});
	handle.registerTool({
		name: "bash_stop",
		label: "Stop background command",
		description:
			"Stop a background command (bash with background:true) that this session started, by its job id (bg-N). Terminates the command's whole process group and reports the observed outcome; the stop result is the outcome — no completion message follows for a command you stopped.",
		parameters: Type.Object({
			id: Type.String({ description: "Job id returned by a background bash call, e.g. bg-1" }),
		}),
		async execute(_toolCallId, params) {
			const job = registry().jobs.get(params.id);
			if (!job) {
				return {
					content: [{ type: "text", text: `No background command ${params.id} in this session. Only commands started here can be stopped.` }],
					details: { id: params.id, found: false },
				};
			}
			if (job.state !== "running" && job.state !== "stopping") {
				const seconds = Math.round(((job.endedAt ?? Date.now()) - job.startedAt) / 100) / 10;
				return {
					content: [
						{
							type: "text",
							text: `${job.id} already finished: ${outcome(job)}, ran ${seconds}s. Output: ${job.logPath}`,
						},
					],
					details: { id: job.id, state: job.state },
				};
			}
			const how = await stopJob(job, "stopped");
			return {
				content: [{ type: "text", text: `${job.id} ${how}.` }],
				details: { id: job.id, state: job.state, outcome: how },
			};
		},
	});
	handle.on("session_start", (event) => {
		// A replaced session (new/resume/fork) never adopts the old session's
		// jobs; a reload takes them over. Cleanup here is belt-and-braces for a
		// shutdown that did not run.
		if (event.reason !== "reload" && event.reason !== "startup") {
			stopAllJobs();
			registry().jobs.clear();
			stopPoller();
		}
		attach(handle);
	});
	handle.on("session_shutdown", (event) => {
		if (event.reason === "reload") return;
		stopAllJobs();
		registry().jobs.clear();
		stopPoller();
		pi = undefined;
	});
}
