// Saccade's native integration, loaded by the saccade-pi composition:
// the session attaches as an incarnation, sends durable addressed
// messages, and waits that incoming messages or native human input
// release - releasing only this integration's own wait, never
// interrupting other work.
import { closeSync, openSync, readFileSync, realpathSync, statSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { Text } from "@earendil-works/pi-tui";
import { Type, type Static } from "typebox";
import type {
	AgentSettledEvent,
	ExtensionContext,
	ExtensionToolContext,
	ExtensionAPI,
	InputEvent,
	SessionStartEvent,
	ToolCallEvent,
	ToolExecutionEndEvent,
	ToolExecutionStartEvent,
} from "@earendil-works/pi-coding-agent";

// The outcome of a supported nested call, taken from the pinned
// executeTool signature itself.
type NestedToolOutcome = Awaited<ReturnType<ExtensionToolContext["executeTool"]>>;

const POLL_MS = 1500;
// the automatic actor fallback: agent possession when the launch did
// not inject a name; an injected SACCADE_ACTOR always wins
const FALLBACK_ACTOR = "saccade-pi";

interface AttachReply {
	registered: boolean;
	incarnation: number;
	num: number;
	/** This session's launch guard: the server-authoritative path this
	 * integration flock-locks from association until death - the
	 * per-native-session runtime evidence activation reads. */
	guard?: string;
}

interface SendReply {
	message: { seq: number; num: number | null };
	from: { seq: number; num: number | null };
}

interface MailboxMessage {
	seq: number;
	num: number | null;
	from: { seq: number; num: number | null; actor: string };
	to: { seq: number; num: number | null };
	reply_to: number | null;
	task: number | null;
	state: "pending" | "submitted";
	body: string;
	born_at: number;
}

interface Association extends AttachReply {
	nativeId: string;
	sessionFile: string;
	/** The open descriptor whose shared open-file-description holds the
	 * guard's kernel lock: closed on association change and death, so
	 * any release is observed by the activator's probe. */
	guardFd: number | null;
}

// One wait tool call owns one of these; nothing about a wait lives in
// shared state, so a stale poll can never observe or strand a newer
// wait's release.
type Outcome = "message" | "input" | "extension" | "cancelled" | "detached";

interface WaitState {
	timer: ReturnType<typeof setInterval> | null;
	offSignal: (() => void) | null;
	done: boolean;
	mail: MailboxMessage[] | null;
	settle: (outcome: Outcome) => void;
}

// An envelope queued natively whose submission still awaits confirmed
// insertion into the saved session history. Bound to the association
// it was queued under, so a reassociated session never submits as a
// stale recipient.
interface QueuedDelivery {
	envelope: string;
	messageSeq: number;
	incarnation: number;
	sessionFile: string;
}

// Take the guard's kernel lock on the open file description this fd
// shares with a short-lived helper: the helper exits, the lock stays
// held because this process's descriptor remains open, and any death
// releases it. true = taken; false = held by a live holder; undefined
// = the host offers no flock, which is unknown, not quiet.
export function takeGuardLock(fd: number): boolean | undefined {
	try {
		const run = spawnSync("flock", ["-n", "3"], {
			stdio: ["ignore", "ignore", "ignore", fd],
		});
		// the documented flock contract: 0 = taken, 1 = the held-lock
		// conflict (busy), every other exit, signal death, or spawn
		// failure is an unknown state - never folded into busy
		if (run.error) return undefined;
		if (run.signal !== null) return undefined;
		if (run.status === 0) return true;
		if (run.status === 1) return false;
		return undefined;
	} catch {
		return undefined;
	}
}

// The held guard's descriptor plus its exact session identity. On
// globalThis under a registered symbol because pi reloads extensions
// with moduleCache: false: module state dies on reload while the
// process-held descriptor lives. Operational state, never lifecycle
// or authority.
interface GuardRecord {
	fd: number;
	path: string;
	sessionId: string;
	sessionFile: string;
}
const GUARD_REGISTRY_KEY = Symbol.for("saccade.guard");
function heldGuard(): GuardRecord | undefined {
	const holder = globalThis as typeof globalThis & {
		[GUARD_REGISTRY_KEY]?: GuardRecord;
	};
	return holder[GUARD_REGISTRY_KEY];
}
function holdGuard(record: GuardRecord | undefined) {
	const holder = globalThis as typeof globalThis & {
		[GUARD_REGISTRY_KEY]?: GuardRecord | undefined;
	};
	holder[GUARD_REGISTRY_KEY] = record;
}

const IdentityParams = Type.Object({});
const SendParams = Type.Object({
	to: Type.String({ description: "recipient incarnation, i-<n>" }),
	body: Type.String({ description: "the message body" }),
	reply_to: Type.Optional(
		Type.String({ description: "the message answered, m-<n>" }),
	),
	task: Type.Optional(Type.String({ description: "task context, t-<n>" })),
});
const WaitParams = Type.Object({
	reply_to: Type.Optional(
		Type.String({
			description: "only messages answering m-<n> release the wait",
		}),
	),
});
const WorkspaceParams = Type.Object({
	choice: Type.Union([Type.Literal("existing"), Type.Literal("new")], {
		description: "the explicit choice: move to an existing directory, or provision a new workspace",
	}),
	path: Type.Optional(
		Type.String({
			description: "existing: the directory to move this session's saved workspace to; it must already exist",
		}),
	),
	label: Type.Optional(
		Type.String({ description: "new: the descriptive label; omit for workspace-<token>" }),
	),
	base: Type.Optional(
		Type.String({ description: "new: the Git base ref; defaults to the primary checkout's current HEAD" }),
	),
});

interface WorkspaceBirthReply {
	created: boolean;
	workspace: { seq: number; base: string; branch: string; path: string };
}

// -- the explicit own-session workspace handoff --------------------------

// One live row of the wrapper's public jobs snapshot: the shape the
// pinned contract declares for the read-only `jobs` tool - registered
// jobs whose state is running, stopping, or unknown; terminal jobs are
// absent by contract. Compatible extra fields are ignored.
export interface JobsSnapshotEntry {
	id: string;
	kind: "bash" | "subagent";
	state: "running" | "stopping" | "unknown";
}

// The snapshot's validation: exactly {jobs: [...]}, each entry carrying
// an id, a kind, and one of the live states. Any deviation - missing,
// malformed, an unseen state - reads as unknown, never as empty: future
// shapes are unverifiable, not silently skipped.
export function validJobsSnapshot(structured: unknown): JobsSnapshotEntry[] | null {
	if (typeof structured !== "object" || structured === null || Array.isArray(structured)) {
		return null;
	}
	const jobs = (structured as { jobs?: unknown }).jobs;
	if (!Array.isArray(jobs)) return null;
	const live: JobsSnapshotEntry[] = [];
	for (const entry of jobs) {
		if (typeof entry !== "object" || entry === null || Array.isArray(entry)) return null;
		const { id, kind, state } = entry as Record<string, unknown>;
		if (typeof id !== "string" || id === "") return null;
		if (kind !== "bash" && kind !== "subagent") return null;
		if (state !== "running" && state !== "stopping" && state !== "unknown") return null;
		live.push({ id, kind, state });
	}
	return live;
}

// The wrapper admissions a handoff must freeze: a background bash start
// and any subagent start. Foreground work, stop, wait, and the jobs
// query itself are not admissions.
function isAdmission(toolName: string, args: unknown): boolean {
	if (toolName === "subagent") return true;
	if (toolName === "bash") {
		return (args as { background?: unknown } | undefined)?.background === true;
	}
	return false;
}

// The saved-session facts an armed handoff persists: the exact identity
// the resume-side correlation validates, the injected self scope it must
// carry across the launch, the helper's inputs, and the human-followable
// failure location.
export interface HandoffEntryData {
	token: string;
	nativeId: string;
	sessionFile: string;
	toolCall: string;
	incarnation: number;
	target: string;
	created: number | null;
	log: string;
	scope: { actor: string | null; task: string | null };
}

// What became of an armed handoff: the token is the join - entries carry
// machine-assigned ids only after persistence, so a done entry names the
// armed entry's token, never an id it cannot know.
export interface HandoffDoneData {
	token: string;
	outcome: "activated" | "stale" | "cancelled";
	reason?: string;
	record?: number;
}

// The handoff this process armed and has not yet shut down for. Module
// state on purpose: any reload or session replacement kills it with the
// tool call's run, and the gate goes with it.
interface PendingHandoff {
	token: string;
	target: string;
	toolCall: string;
	log: string;
	created: number | null;
	script: string;
}

// The helper launch seam: exported so the contract test can pin the
// script's words, and overridden nowhere in production.
export function handoffHelperScript(spec: {
	guard: string;
	cli: string;
	server: string;
	repo: string;
	num: number;
	target: string;
	cycles: number;
}): string {
	const quote = (word: string): string => `'${word.replaceAll("'", `'\\''`)}'`;
	const cycles = String(spec.cycles);
	return [
		"i=0",
		`while [ "$i" -lt ${cycles} ]; do`,
		`\tif flock -n ${quote(spec.guard)} true 2>/dev/null; then break; fi`,
		"	sleep 0.25",
		"	i=$((i+1))",
		"done",
		`if ! flock -n ${quote(spec.guard)} true 2>/dev/null; then`,
		`\techo ${quote("the session's guard stayed held past the quiescence bound; no switch or resume was attempted")}`,
		"\texit 3",
		"fi",
		`exec ${quote(spec.cli)} --server ${quote(spec.server)} --repo ${quote(spec.repo)} workspace switch ${quote(`i-${spec.num}`)} --to ${quote(spec.target)} --resume`,
	].join("\n");
}

// Launch the detached helper: a shell that waits for this process's own
// death to release the guard, then calls the existing stopped-session
// controller door. The helper inherits no descriptor of this process -
// only the log file it writes - so its own wait cannot block the
// release it waits for. Detached and unref'd: it outlives this process
// by design and bounds itself.
function launchHandoffHelper(script: string, log: string): void {
	const fd = openSync(log, "a", 0o600);
	try {
		const child = spawn("sh", ["-c", script], {
			detached: true,
			stdio: ["ignore", fd, fd],
		});
		child.on("error", () => {});
		child.unref();
	} finally {
		closeSync(fd);
	}
}

// The machine envelope that reactivates the explicit own request after
// the confirmed move: original tool reference and the change record's
// storage seq, never human-shaped text. All words are authored ASCII; the
// target is data. Record seqs render as bare numbers: c-N is the comment
// numbering, never record provenance.
export function handoffEnvelope(data: HandoffEntryData, record: number): string {
	return [
		`[saccade workspace handoff | tool ${data.toolCall} | change record ${record}]`,
		"",
		`Your explicit own-session workspace handoff is confirmed: this native session kept its id, saved history, and declared scope; the saved working directory moved to ${data.target} (record ${record}).`,
		"Continue the goal you were pursuing when you called saccade_workspace - no new goal was created and nothing else changed.",
	].join("\n");
}

// The saved header's cwd, read the one way the controller does: first
// line, session header, its cwd field. Anything else is unreadable.
export function headerCwd(sessionFile: string): string | null {
	try {
		const line = readFileSync(sessionFile, "utf8").split("\n", 1)[0] ?? "";
		const header = JSON.parse(line) as { type?: string; cwd?: unknown };
		if (header.type !== "session") return null;
		if (typeof header.cwd !== "string" || header.cwd === "") return null;
		return header.cwd;
	} catch {
		return null;
	}
}

export default function saccadeIntegration(pi: ExtensionAPI) {
	// the attach record's reader: interactive chat paints custom
	// entries only through a registered renderer
	pi.registerEntryRenderer("saccade_attach", (entry) => {
		const data = (entry.data ?? {}) as { error?: string };
		return new Text(data.error ? `saccade: ${data.error}` : "saccade: attached");
	});
	// the armed handoff and its outcome, painted for the session's human
	pi.registerEntryRenderer("saccade_handoff", (entry) => {
		const data = (entry.data ?? {}) as HandoffEntryData;
		const provisioned = data.created !== null ? ` (provisioned, birth record ${data.created})` : "";
		return new Text(
			`saccade: handoff armed - saved workspace moves to ${data.target}${provisioned} when this run settles; helper log ${data.log}`,
		);
	});
	pi.registerEntryRenderer("saccade_handoff_done", (entry) => {
		const data = (entry.data ?? {}) as HandoffDoneData;
		const line =
			data.outcome === "activated"
				? `handoff activated, resumed there (record ${data.record})`
				: data.outcome === "stale"
					? `handoff not activated: ${data.reason ?? "stale"}`
					: `handoff cancelled: ${data.reason ?? "cancelled"}`;
		return new Text(`saccade: ${line}`);
	});
	// the session's association: resolved or registered fresh at every
	// session start by the session's actual native id and locator
	let association: Association | null = null;
	let attachError: string | null = null;
	// this start is the intentional in-memory mode: the pinned contract
	// leaves getSessionFile undefined exactly when persist is false, so
	// there is no saved session to verify and no startup exit
	let unpersistedSession = false;
	// the waits this integration currently owns
	const waits = new Set<WaitState>();
	// envelopes queued natively, submissions not yet confirmed
	const deliveries = new Set<QueuedDelivery>();
	// message seqs already queued under this association, with the
	// message itself for correlation: startup, the wait tool, and the
	// background bridge all drain the same mailbox, and a message
	// enters the native queue exactly once - until reality disagrees
	// (the send was refused, or the queued input was discarded), at
	// which point the guard releases and the next drain delivers again
	const queued = new Map<number, MailboxMessage>();
	// the armed own-session handoff this process has not yet shut down
	// for: null once the helper is launched (the shutdown is the armed
	// run's own act), and cancelled with its gate on any other boundary
	let pendingHandoff: PendingHandoff | null = null;
	// continuations queued natively whose terminal done awaits the saved
	// envelope, bound to the arm they came from; association-bound like
	// deliveries - a reassociated session never completes a stale one
	const continuations = new Set<{ data: HandoffEntryData; record: number; envelope: string }>();
	// The own admission gate, set while a handoff request prepares and
	// then owned by its armed handoff: background bash and subagent
	// starts refuse at the public tool_call hook. Module state on
	// purpose, like the arm itself - a reload or session replacement
	// kills it with the tool call's run.
	let admissionHold: { token: string } | null = null;
	// Admissions dispatched and not yet finished, keyed by tool call id:
	// the gap between a start and its registry entry, which the jobs
	// snapshot cannot see yet. Bounded native call state, never a model
	// of ongoing jobs - the registered truth stays the jobs read.
	const admissions = new Set<string>();

	// The public admission guard: while this session's handoff prepares
	// or is armed, background bash and subagent starts refuse before
	// execution - stop, wait, jobs, and foreground work stay allowed,
	// and nothing already running is touched.
	pi.on("tool_call", (event: ToolCallEvent) => {
		if (admissionHold === null && pendingHandoff === null) return undefined;
		if (!isAdmission(event.toolName, event.input)) return undefined;
		return {
			block: true,
			reason:
				"a saccade workspace handoff is preparing: no new background bash or subagent may start; nothing already running is stopped - retry the move later",
		};
	});
	pi.on("tool_execution_start", (event: ToolExecutionStartEvent) => {
		if (isAdmission(event.toolName, event.args)) admissions.add(event.toolCallId);
	});
	// the end is unconditional on the pinned runtime, blocked attempts
	// included: a finished dispatch never strands its token here
	pi.on("tool_execution_end", (event: ToolExecutionEndEvent) => {
		admissions.delete(event.toolCallId);
	});

	// The awaited status read: the wrapper's public jobs tool through the
	// supported nested dispatch, so disabled tools and permissions apply
	// unchanged. A missing tool, an error, and any malformed answer
	// return unknown - never empty, never idle.
	async function queryJobs(
		ctx: ExtensionToolContext,
		signal?: AbortSignal,
	): Promise<JobsSnapshotEntry[] | null> {
		if (typeof ctx.executeTool !== "function") return null;
		try {
			const outcome: NestedToolOutcome = await ctx.executeTool("jobs", {}, { signal });
			if (outcome.isError) return null;
			return validJobsSnapshot(outcome.result?.structuredContent);
		} catch {
			// a dispatch that refuses to answer is the same unknown as a
			// missing tool
			return null;
		}
	}

	// agent possession is set at load, before any attach: a tracker
	// outage must still leave nested sac calls acting as the agent, and
	// an injected name always wins
	const injectedActor = process.env.SACCADE_ACTOR;
	// whether this session's actor was injected (launch-time possession)
	// or is only the automatic fallback: an injected actor rides the
	// self-handoff's scope, the fallback does not pretend to be one
	const actorInjected = injectedActor !== undefined && injectedActor.trim() !== "";
	if (injectedActor === undefined || injectedActor.trim() === "") {
		process.env.SACCADE_ACTOR = FALLBACK_ACTOR;
	}

	const server = () => {
		const url = process.env.SACCADE_SERVER;
		return url && url.trim() !== "" ? url.trim() : undefined;
	};

	const actor = () => process.env.SACCADE_ACTOR || FALLBACK_ACTOR;

	async function tracker<T>(path: string, init?: RequestInit): Promise<T> {
		const url = server();
		if (!url) throw new Error("no SACCADE_SERVER routing in this environment");
		const response = await fetch(`${url}${path}`, {
			...init,
			headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
		});
		const body: unknown = await response.json();
		if (response.status !== 200) {
			const code = (body as any)?.error?.code ?? response.status;
			const detail = (body as any)?.error?.detail;
			throw new Error(
				`${code}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`,
			);
		}
		return body as T;
	}

	// Release the held guard, if any: closing the descriptor is the
	// observed kernel release activation reads.
	function releaseGuard() {
		const held = heldGuard();
		if (held) {
			try {
				closeSync(held.fd);
			} catch {
				// already closed: the release is the same either way
			}
			holdGuard(undefined);
		}
	}

	// Suspend the association runtime without touching the guard: the
	// identity channel goes with the association, queued deliveries die
	// unsubmitted, and the bridge loop strands - but an exact-session
	// guard stays kernel-held. Process death releases it; a reload's
	// next start reuses it. Releasing before a reattach reply would let
	// a transient tracker outage leave this live process unguarded
	// while activation sees a free session and resumes another copy.
	function suspendAssociation() {
		association = null;
		attachError = null;
		deliveries.clear();
		queued.clear();
		delete process.env.SACCADE_INCARNATION_SEQ;
	}

	// Resolve-or-register this session's incarnation from the session's
	// actual native id and saved locator - never from a remembered
	// attachment. The reply is the machine identity, and the env channel
	// rides it for `sac`.
	async function attach(ctx: ExtensionContext): Promise<Association | null> {
		const sessionId = ctx.sessionManager.getSessionId();
		const sessionFile = ctx.sessionManager.getSessionFile();
		suspendAssociation();
		const held = heldGuard();
		if (
			held &&
			(held.sessionId !== sessionId || held.sessionFile !== sessionFile)
		) {
			// a different native session's guard cannot be reused: this
			// process no longer runs it, so the release is honest
			releaseGuard();
		}
		if (sessionFile === undefined && sessionId) {
			// the pinned in-memory mode: the contract leaves the session
			// file undefined exactly when nothing persists, and the id is
			// always generated - an intentional helper has no saved
			// session to verify, so it stays alive unregistered, checked
			// before routing so server presence cannot change its shape
			unpersistedSession = true;
			attachError = "the session is unsaved (in-memory); nothing to register";
			return null;
		}
		if (!server()) {
			attachError = "no SACCADE_SERVER routing; the session stays unregistered";
			return null;
		}
		if (!sessionId || !sessionFile) {
			// a persisted session without its file is invalid identity:
			// cannot verify, so the startup exit rule applies
			attachError = "the session carries no saved native file; nothing to register";
			return null;
		}
		try {
			const reply = await tracker<AttachReply>("/api/v1/native/attach", {
				method: "POST",
				body: JSON.stringify({
					native_id: sessionId,
					session_file: sessionFile,
					launch_token: process.env.SACCADE_LAUNCH_TOKEN || undefined,
				}),
			});
			// the session's own runtime guard: kernel-held from here to
			// association change or death. Held = a live process already
			// owns this native session: this process is a duplicate - it
			// shuts down before any model work. Unknown = the host
			// offers no flock: fail safe, no association, no
			// participation - an unverified session must never look
			// free to the backend's probe. A guard held for this exact
			// session from a previous start is reused, never reopened:
			// the lock is already ours.
			if (!reply.guard) {
				releaseGuard();
			} else if (heldGuard()?.path === reply.guard) {
				// retained exact-session guard
			} else {
				releaseGuard();
				let fd: number;
				try {
					fd = openSync(reply.guard, "a+");
				} catch (error) {
					// an unopenable guard is as unsafe as an unknown take:
					// the process exits rather than continue unguarded
					attachError = `the guard path is unopenable: ${errorText(error)}`;
					ctx.shutdown();
					return null;
				}
				const taken = takeGuardLock(fd);
				if (taken === false) {
					closeSync(fd);
					attachError =
						"duplicate: this native session's guard is held by a live process";
					// the supported termination path: the duplicate exits
					// before any model work, on the context, not the pi
					ctx.shutdown();
					return null;
				}
				if (taken === undefined) {
					// unknown refuses the association AND the session: no
					// unguarded tracked process may keep running
					closeSync(fd);
					attachError =
						"the guard cannot be probed here; refusing an unverified association";
					ctx.shutdown();
					return null;
				}
				holdGuard({
					fd,
					path: reply.guard,
					sessionId,
					sessionFile,
				});
			}
			association = {
				...reply,
				nativeId: sessionId,
				sessionFile,
				guardFd: heldGuard()?.fd ?? null,
			};
			process.env.SACCADE_INCARNATION_SEQ = String(reply.incarnation);
			// the launch token's bootstrap is done: a later fork or
			// reload in this process reattaches tokenless, so no other
			// association can ever be retargeted by this one
			delete process.env.SACCADE_LAUNCH_TOKEN;
			startBridge(association);
			return association;
		} catch (error) {
			attachError = errorText(error);
			// a retained exact-session guard stays held across the failed
			// attempt: this live process never continues unguarded, and
			// the next attach attempt reuses the lock
			return null;
		}
	}

	// An attached session stays attached; a failed attach retries here,
	// so explicit later tools recover when the tracker returns.
	async function ensureAttached(ctx: ExtensionContext): Promise<Association | null> {
		return association ?? (await attach(ctx));
	}

	async function mailbox(incarnation: number): Promise<MailboxMessage[]> {
		const reply = await tracker<{ messages: MailboxMessage[] }>(
			`/api/v1/messages/pending?incarnation=${incarnation}`,
		);
		return reply.messages;
	}

	// Filter to the addressee and queue each new message natively - the
	// queueing half of the one incoming path.
	function queueIncoming(messages: MailboxMessage[]): MailboxMessage[] {
		const who = association;
		if (!who) return [];
		const incoming = incomingFor(who.incarnation, messages);
		for (const message of incoming) {
			queueDelivery(message);
		}
		return incoming;
	}

	// The one incoming read: startup, the wait tool, and the background
	// bridge all come through here; the queued-once guard keeps a
	// message to a single envelope.
	async function drainIncoming(): Promise<MailboxMessage[]> {
		const who = association;
		if (!who) return [];
		return queueIncoming(await mailbox(who.incarnation));
	}

	// The background bridge: one parked read per association, woken by
	// the record's landing server-side, delivering at whatever native
	// boundary the session supports - into the running queue, or as a
	// turn when idle. The loop dies with its association, so a
	// reassociated session never delivers as a stale recipient, and a
	// failed read reconnects with backoff - transport, nothing more.
	const bridgeLoops = new Set<Association>();
	function startBridge(assoc: Association): void {
		if (bridgeLoops.has(assoc)) return;
		bridgeLoops.add(assoc);
		void (async () => {
			let backoff = 500;
			while (association === assoc) {
				try {
					const reply = await tracker<{ messages: MailboxMessage[] }>(
						`/api/v1/messages/pending?incarnation=${assoc.incarnation}&wait_ms=25000&bridge=1`,
					);
					backoff = 500;
					const delivered = queueIncoming(reply.messages);
					if (delivered.length > 0) {
						// the parked read signals landings; pending mail that
						// awaits its own submission re-reads at the ordinary
						// cadence, never a hot loop against the tracker
						await new Promise((resolve) => setTimeout(resolve, POLL_MS));
					}
				} catch {
					await new Promise((resolve) => setTimeout(resolve, backoff));
					backoff = Math.min(backoff * 2, 15000);
				}
			}
			bridgeLoops.delete(assoc);
		})();
	}

	// Queue one message natively and return: never block on what the
	// native queue does after the current run. The submission is
	// confirmed separately, by the delivery observers. Pinned 1.2.0:
	// sendUserMessage returns void and its internal failures are
	// invisible here (the runtime routes them to its own error sink) -
	// a refused or discarded delivery is recovered at the next
	// supported boundary, never by watching a promise.
	function queueDelivery(message: MailboxMessage): void {
		if (association === null) return;
		if (queued.has(message.seq)) return;
		queued.set(message.seq, message);
		const delivery: QueuedDelivery = {
			envelope: envelopeText(message),
			messageSeq: message.seq,
			incarnation: association.incarnation,
			sessionFile: association.sessionFile,
		};
		deliveries.add(delivery);
		pi.sendUserMessage(delivery.envelope, { deliverAs: "followUp" });
	}

	// Release the guards of deliveries that reality no longer holds:
	// neither persisted in saved history nor natively pending while the
	// session is truly idle means the queued input was refused before
	// insertion or discarded after queueing. The next drain delivers
	// again - recoverability over the once-flag, no timers.
	function reconcileDiscarded(ctx: ExtensionContext): boolean {
		if (ctx.isIdle?.() !== true || ctx.hasPendingMessages?.() !== false) {
			return false;
		}
		let released = false;
		for (const delivery of [...deliveries]) {
			if (!savedEntryHas(ctx, delivery.envelope)) {
				deliveries.delete(delivery);
				queued.delete(delivery.messageSeq);
				released = true;
			}
		}
		return released;
	}

	// Extension message events are not proof of insertion - the
	// message_end hook runs before the SessionManager append - so
	// confirmation reads the saved entries themselves.
	function savedEntryHas(ctx: ExtensionContext, envelope: string): boolean {
		for (const entry of ctx.sessionManager.getEntries()) {
			if (entry.type !== "message") continue;
			const message = entry.message;
			if (message && message.role === "user" && userText(message) === envelope) {
				return true;
			}
		}
		return false;
	}

	// Confirm what the saved history actually holds: a delivery whose
	// envelope is in the saved entries submits; a delivery bound to an
	// association that is no longer live cancels. A failed submit - or
	// a success whose response is lost - goes back to the queue and may
	// re-post on later boundaries until the next resume clears it:
	// limited retry-on-later-boundary bookkeeping, not convergence and
	// not an exactly-once promise.
	function confirmDeliveries(ctx: ExtensionContext): void {
		reconcileDiscarded(ctx);
		for (const delivery of [...deliveries]) {
			if (
				association === null ||
				association.incarnation !== delivery.incarnation ||
				association.sessionFile !== delivery.sessionFile
			) {
				deliveries.delete(delivery);
				continue;
			}
			if (!savedEntryHas(ctx, delivery.envelope)) continue;
			deliveries.delete(delivery);
			tracker("/api/v1/messages/submit", {
				method: "POST",
				body: JSON.stringify({
					incarnation: delivery.incarnation,
					message: delivery.messageSeq,
				}),
			}).catch(() => {
				deliveries.add(delivery);
			});
		}
	}

	// The confirmation boundaries: any later message end, run end, or
	// settle - each checks the saved entries, never an event alone.
	pi.on("message_end", (_event: unknown, ctx: ExtensionContext) => {
		confirmDeliveries(ctx);
		confirmContinuations(ctx);
		return undefined;
	});
	pi.on("agent_end", (_event: unknown, ctx: ExtensionContext) => {
		confirmDeliveries(ctx);
		confirmContinuations(ctx);
		return undefined;
	});
	pi.on("agent_settled", (event: AgentSettledEvent, ctx: ExtensionContext) => {
		confirmDeliveries(ctx);
		confirmContinuations(ctx);
		if (event.aborted === true) {
			// the run was aborted: only this handoff cancels, with its
			// gate - no launch, no shutdown, no undo of a provision that
			// already landed, no touching of any job
			cancelPendingHandoff("the run was aborted; the session stays");
			return undefined;
		}
		settlePendingHandoff(ctx);
		return undefined;
	});

	// Release the armed handoff - or a still-preparing request, whose
	// gate has not reached a persisted arm - without shutting down: the
	// outcome lands as a fact and the session stays exactly as it was.
	// A preparation cancelled here observes its hold gone at its next
	// await and never arms.
	function cancelPendingHandoff(reason: string) {
		const armed = pendingHandoff;
		if (armed !== null) {
			pendingHandoff = null;
			pi.appendEntry("saccade_handoff_done", {
				token: armed.token,
				outcome: "cancelled",
				reason,
			} satisfies HandoffDoneData);
			return;
		}
		const preparing = admissionHold;
		if (preparing !== null) {
			admissionHold = null;
			pi.appendEntry("saccade_handoff_done", {
				token: preparing.token,
				outcome: "cancelled",
				reason,
			} satisfies HandoffDoneData);
		}
	}

	// The settle boundary is the safe native act: the run's entries - the
	// tool result among them - are saved before this fires, so the old
	// process stops with its history complete. The helper is launched
	// first (detached, unref'd, no descriptor of this process) and the
	// supported shutdown follows; a launch failure releases the hold and
	// keeps the session usable.
	function settlePendingHandoff(ctx: ExtensionContext) {
		const armed = pendingHandoff;
		if (armed === null) return;
		try {
			launchHandoffHelper(armed.script, armed.log);
		} catch (error) {
			pendingHandoff = null;
			pi.appendEntry("saccade_handoff_done", {
				token: armed.token,
				outcome: "cancelled",
				reason: `the helper could not launch: ${errorText(error)}`,
			} satisfies HandoffDoneData);
			return;
		}
		pendingHandoff = null;
		ctx.shutdown();
	}

	// The armed handoff entries a saved history carries, minus the ones a
	// done entry already resolved: the token is the join.
	function unconsumedHandoffs(ctx: ExtensionContext): HandoffEntryData[] {
		const done = new Set<string>();
		const armed: HandoffEntryData[] = [];
		for (const entry of ctx.sessionManager.getEntries()) {
			if (entry.type !== "custom") continue;
			const shaped = entry as { customType?: string; data?: unknown };
			if (shaped.customType === "saccade_handoff_done") {
				const token = (shaped.data ?? {}) as HandoffDoneData;
				if (typeof token.token === "string") done.add(token.token);
				continue;
			}
			if (shaped.customType !== "saccade_handoff") continue;
			const data = shaped.data as HandoffEntryData | undefined;
			if (data && typeof data.token === "string") armed.push(data);
		}
		return armed.filter((a) => !done.has(a.token));
	}

	// The confirmed change record for this incarnation and target: actual
	// tracker facts scanned once, the latest match wins. The payload
	// travels as the JSON itself - an object on the wire, a string only
	// in older storage shapes - and an unreadable payload is skipped,
	// never guessed.
	async function confirmedChangeRecord(incarnation: number, target: string): Promise<number | null> {
		const reply = await tracker<{ records: Array<{ seq: number; kind: string; payload: unknown }> }>(
			"/api/v1/records",
		);
		let found: number | null = null;
		for (const record of reply.records) {
			if (record.kind !== "workspace_changed") continue;
			try {
				const payload = (
					typeof record.payload === "string" ? JSON.parse(record.payload) : record.payload
				) as { incarnation?: unknown; worktree?: unknown };
				if (payload.incarnation === incarnation && payload.worktree === target) found = record.seq;
			} catch {
				continue;
			}
		}
		return found;
	}

	// Correlate the saved armed entry against the exact current native
	// association, the actual saved header, and the confirmed record
	// before consuming it. Startup data is correlation, never authority:
	// a fork, another session, an unmoved or re-moved header, or a
	// missing record marks the entry stale instead of redirecting it, and
	// a tracker outage leaves it unconsumed for the next start.
	async function consumeHandoff(ctx: ExtensionContext, who: Association): Promise<void> {
		const candidates = unconsumedHandoffs(ctx);
		if (candidates.length === 0) return;
		for (const older of candidates.slice(0, -1)) {
			pi.appendEntry("saccade_handoff_done", {
				token: older.token,
				outcome: "stale",
				reason: "superseded by a later arm",
			} satisfies HandoffDoneData);
		}
		const data = candidates[candidates.length - 1]!;
		const stale = (reason: string) => {
			pi.appendEntry("saccade_handoff_done", {
				token: data.token,
				outcome: "stale",
				reason,
			} satisfies HandoffDoneData);
		};
		if (data.nativeId !== ctx.sessionManager.getSessionId()) {
			return stale("the armed native id is not this session's");
		}
		if (data.sessionFile !== ctx.sessionManager.getSessionFile()) {
			return stale("the armed session file is not this session's");
		}
		if (headerCwd(data.sessionFile) !== data.target) {
			return stale("the saved session's working directory is not the armed target");
		}
		const record = await confirmedChangeRecord(who.incarnation, data.target);
		if (record === null) {
			return stale("no confirmed workspace change record matches this incarnation and target");
		}
		// the validated arm restores this session's own injected scope: the
		// values captured at arm time, applied in the resumed process before
		// any write can ride the wrong identity channel
		if (data.scope?.actor) process.env.SACCADE_ACTOR = data.scope.actor;
		if (data.scope?.task) process.env.SACCADE_TASK = data.scope.task;
		// Activation is not terminal until the continuation envelope is in
		// the saved history: sendUserMessage returns void and its failures
		// are invisible, so the done entry lands only at the saved-insertion
		// confirmation, exactly like a message delivery. Until then the arm
		// stays unconsumed - a crash, refusal, or reload re-queues at the
		// next supported boundary instead of silencing the request forever.
		const envelope = handoffEnvelope(data, record);
		if (savedEntryHas(ctx, envelope)) {
			pi.appendEntry("saccade_handoff_done", {
				token: data.token,
				outcome: "activated",
				record,
			} satisfies HandoffDoneData);
			return;
		}
		if (![...continuations].some((c) => c.data.token === data.token)) {
			continuations.add({ data, record, envelope });
			pi.sendUserMessage(envelope, { deliverAs: "followUp" });
		}
	}

	// The saved-insertion correlation for queued continuations, riding the
	// same boundaries as delivery confirmation: a saved envelope completes
	// the arm terminally; an envelope the queue dropped while the session
	// is idle re-queues; a busy session just waits for its own boundary.
	function confirmContinuations(ctx: ExtensionContext): void {
		for (const continuation of [...continuations]) {
			if (savedEntryHas(ctx, continuation.envelope)) {
				continuations.delete(continuation);
				pi.appendEntry("saccade_handoff_done", {
					token: continuation.data.token,
					outcome: "activated",
					record: continuation.record,
				} satisfies HandoffDoneData);
				continue;
			}
			if (ctx.isIdle?.() === true && ctx.hasPendingMessages?.() === false) {
				pi.sendUserMessage(continuation.envelope, { deliverAs: "followUp" });
			}
		}
	}

	function text(content: string) {
		return {
			content: [{ type: "text" as const, text: content }],
			details: undefined,
		};
	}

	function failed(content: string) {
		return { ...text(content), isError: true } as const;
	}

	// Finish a wait exactly once and take its machinery down with it:
	// timer, signal listener, and the active set. A finished wait's
	// callbacks are inert, so it can never consume a newer wait's
	// release.
	function finish(state: WaitState, outcome: Outcome, mail: MailboxMessage[]) {
		if (state.done) return;
		state.done = true;
		if (state.timer !== null) {
			clearInterval(state.timer);
			state.timer = null;
		}
		if (state.offSignal !== null) {
			const off = state.offSignal;
			state.offSignal = null;
			off();
		}
		waits.delete(state);
		state.mail = mail;
		state.settle(outcome);
	}

	function cancelAllWaits() {
		for (const state of [...waits]) {
			finish(state, "detached", []);
		}
	}

	pi.registerTool({
		name: "saccade_identity",
		label: "Saccade identity",
		description:
			"Report this session's registered saccade identity: the incarnation's public number i-<n> that others address, and whether it registered now or resolved an existing registration.",
		parameters: IdentityParams,
		execute: async (_id, _params: Static<typeof IdentityParams>, _signal, _onUpdate, ctx: ExtensionToolContext) => {
			const who = await ensureAttached(ctx);
			if (!who) return failed(`not registered: ${attachError}`);
			return text(
				`this session is i-${who.num} (birth seq ${who.incarnation}${who.registered ? ", registered now" : ", already registered"}); others address it as i-${who.num}`,
			);
		},
	});

	// The environment-declared task: context.scope.task reads this
	// alone. A tool argument names the message's task context, never a
	// scope rebind.
	function declaredTask(): number | null {
		const value = process.env.SACCADE_TASK;
		return value && value.trim() !== "" ? numToken(value, "t-") : null;
	}

	pi.registerTool({
		name: "saccade_send",
		label: "Saccade send",
		description:
			"Send a durable message to another registered incarnation. Addressing uses public numbers: the recipient i-<n>, an optional reply_to m-<n> naming the message this one answers, and an optional task t-<n> riding as context without publishing to its thread. Nonblocking; pair with saccade_wait when the reply matters now.",
		parameters: SendParams,
		execute: async (_id, params: Static<typeof SendParams>, _signal, _onUpdate, ctx: ExtensionToolContext) => {
			const who = await ensureAttached(ctx);
			if (!who) return failed(`not sent: ${attachError}`);
			try {
				const to = numToken(params.to, "i-");
				const reply = await tracker<SendReply>("/api/v1/messages/send", {
					method: "POST",
					body: JSON.stringify({
						context: {
							actor: actor(),
							tier: "agent",
							scope: {
								task: declaredTask(),
								incarnation: who.incarnation,
								workspace: process.env.SACCADE_WORKSPACE ?? null,
							},
						},
						to,
						reply_to: params.reply_to ? numToken(params.reply_to, "m-") : null,
						task: params.task ? numToken(params.task, "t-") : null,
						body: params.body,
					}),
				});
				return text(
					`sent m-${reply.message.num} to i-${to}` +
						(params.reply_to
							? ` replying to m-${numToken(params.reply_to, "m-")}`
							: "") +
						"; it stays pending until their integration submits it",
				);
			} catch (error) {
				return failed(`not sent: ${errorText(error)}`);
			}
		},
	});

	pi.registerTool({
		name: "saccade_wait",
		label: "Saccade wait",
		description:
			"Wait for incoming saccade messages addressed to this incarnation. Released by a matching message (delivered natively into your session) or by native human input; the result names which happened, never whether a request was answered. No messages and no input: it keeps waiting.",
		parameters: WaitParams,
		execute: async (_id, params: Static<typeof WaitParams>, signal, _onUpdate, ctx: ExtensionToolContext) => {
			const who = await ensureAttached(ctx);
			if (!who) return failed(`cannot wait: ${attachError}`);
			// an explicit wait is a supported recovery boundary: refused
			// or discarded deliveries release before this wait reads
			reconcileDiscarded(ctx);
			const wantReply = params.reply_to ? numToken(params.reply_to, "m-") : null;
			const incarnation = who.incarnation;
			const state: WaitState = {
				timer: null,
				offSignal: null,
				done: false,
				mail: null,
				settle: () => {},
			};
			const released = await new Promise<Outcome>((resolve) => {
				state.settle = (outcome) => resolve(outcome);
				waits.add(state);
				// the tool call's own cancellation finishes the wait; the
				// listener comes off in finish, so no path leaks it
				if (signal) {
					const onAbort = () => finish(state, "cancelled", []);
					if (signal.aborted) {
						onAbort();
						return;
					}
					signal.addEventListener("abort", onAbort);
					state.offSignal = () => signal.removeEventListener("abort", onAbort);
				}
				const poll = async () => {
					if (state.done) return;
					try {
						const incoming = incomingFor(
							incarnation,
							await mailbox(incarnation),
						);
						const mine =
							wantReply === null
								? incoming
								: incoming.filter((m) => m.reply_to === wantReply);
						if (mine.length > 0) finish(state, "message", mine);
					} catch {
						// a poll that cannot reach the tracker keeps waiting
					}
				};
				state.timer = setInterval(poll, POLL_MS);
			});
			// whatever the path, the wait's machinery is already down
			if (released === "extension") {
				return text(
					"the wait ended because native input arrived through the integration; any pending messages stay in the mailbox - saccade_wait again to receive them",
				);
			}
			if (released === "input") {
				return text(
					"the wait ended because native human input arrived; any pending messages stay in the mailbox - saccade_wait again to receive them",
				);
			}
			if (released === "cancelled") {
				return text(
					"the wait ended without a message (the call was cancelled); any pending messages stay in the mailbox - saccade_wait again to receive them",
				);
			}
			if (released === "detached") {
				return text(
					"the wait ended because this session's saccade association changed; nothing was delivered - saccade_identity to re-check, saccade_wait to receive",
				);
			}
			// queue and return: the native queue drains after this tool
			// call is done, and the submission confirms asynchronously
			// from the saved history, never inside this call
			const delivered = state.mail ?? [];
			for (const message of delivered) {
				queueDelivery(message);
			}
			const named = delivered
				.map((m) => `m-${m.num} from i-${m.from.num}`)
				.join(", ");
			return text(
				`received ${named}; delivered into your session as your next input - reply through saccade_send with reply_to`,
			);
		},
	});

	pi.registerTool({
		name: "saccade_workspace",
		label: "Saccade workspace",
		description:
			"Move THIS session to another workspace, by explicit choice only: an existing directory, or a newly provisioned one. The saved working directory changes at a safe native boundary - this run settles, the session shuts down, and a bounded helper moves the saved session and resumes it there with the same native id, history, and declared scope. Refuses while background jobs or subagents are active or unverifiable (nothing is stopped), and blocks new background starts until the handoff resolves.",
		parameters: WorkspaceParams,
		execute: async (
			id,
			params: Static<typeof WorkspaceParams>,
			signal,
			_onUpdate,
			ctx: ExtensionToolContext,
		) => {
			const who = await ensureAttached(ctx);
			if (!who) return failed(`not moved: ${attachError}`);
			const held = heldGuard();
			const cli = process.env.SACCADE_CLI?.trim() || undefined;
			const repo = process.env.SACCADE_REPO?.trim() || undefined;
			const url = server();
			if (!held || !cli || !repo || !url) {
				const missing = [
					!held ? "this session's guard" : null,
					!url ? "SACCADE_SERVER" : null,
					!repo ? "SACCADE_REPO" : null,
					!cli ? "SACCADE_CLI (the controller door's path)" : null,
				]
					.filter((part) => part !== null)
					.join(", ");
				return failed(`not moved: the handoff helper needs ${missing}; nothing changed`);
			}
			const token = `handoff-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
			// param discipline before any hold: a request that can never arm
			// holds nothing and reads nothing
			if (
				params.choice === "existing" &&
				(params.path === undefined || params.label !== undefined || params.base !== undefined)
			) {
				return failed(
					"not moved: the existing choice names exactly a path and nothing else; nothing changed",
				);
			}
			if (params.choice === "new" && params.path !== undefined) {
				return failed("not moved: the new choice provisions a workspace and takes no path; nothing changed");
			}
			if (admissionHold !== null || pendingHandoff !== null) {
				return failed(
					"not moved: a workspace handoff is already preparing or armed for this session; nothing changed",
				);
			}
			// the admission gate goes up before any awaited read, so a new
			// background start cannot slip past the status check while it
			// runs; every exit below releases this request's hold, and a
			// refused or failed preparation leaves the session as it was
			admissionHold = { token };
			// a provision that already landed keeps its pointer through any
			// later failure: the birth stands, the error face names it, and
			// nothing is rolled back or claimed moved
			let created: number | null = null;
			let createdPath: string | null = null;
			const giveUp = (why: string) => {
				// only this request's hold: a newer request's, or an already
				// cancelled, hold is never released here
				if (admissionHold?.token === token) admissionHold = null;
				// only this call's token-owned runtime arm is cleared: the
				// append may have failed before the arm persisted, and the
				// settle boundary must never launch an unpersisted handoff
				if (pendingHandoff?.token === token) pendingHandoff = null;
				const pointer =
					created !== null && createdPath !== null
						? ` The provisioned workspace stands: birth record ${created} at ${createdPath}; no move happened.`
						: "";
				return failed(`${why}${pointer}`);
			};
			// after every awaited step the request continues only while its
			// own hold is still current under the same association: human
			// input, abort, reload, or session replacement cancels the
			// preparation, and a stale completion must not re-arm
			const still = () => admissionHold?.token === token && association === who;
			try {
				// an admission already executing when the gate went up - a
				// start whose spawn has not landed in the registry - is
				// exactly what the jobs snapshot cannot see yet
				if (admissions.size > 0) {
					return giveUp(
						`not moved: ${admissions.size} background start(s) still in flight; nothing was stopped and nothing changed`,
					);
				}
				const jobs = await queryJobs(ctx, signal);
				if (!still()) {
					return giveUp(
						"not moved: the handoff was cancelled or the session changed while its status was read; nothing changed",
					);
				}
				if (signal?.aborted) {
					return giveUp("not moved: the call was cancelled; the hold is released and the session stays");
				}
				if (jobs === null) {
					return giveUp(
						"not moved: no valid jobs snapshot answered - the tool is missing, errored, or malformed, and unverifiable activity is never idle; nothing changed",
					);
				}
				if (jobs.length > 0) {
					const listed = jobs.map((j) => `${j.id} (${j.kind}, ${j.state})`).join(", ");
					return giveUp(
						`not moved: ${jobs.length} reported background job(s) (running, stopping, or unresolved): ${listed}; nothing was stopped and nothing changed`,
					);
				}
				if (admissions.size > 0) {
					return giveUp(
						"not moved: a background start raced the status read; nothing was stopped and nothing changed",
					);
				}
				let target: string;
				if (params.choice === "existing") {
					const resolved = realpathSync(params.path!);
					if (!statSync(resolved).isDirectory()) {
						return giveUp(`not moved: ${resolved} is not a directory; nothing changed`);
					}
					target = resolved;
				} else {
					const reply = await tracker<WorkspaceBirthReply>("/api/v1/workspace/create", {
						method: "POST",
						body: JSON.stringify({
							label: params.label ?? null,
							base: params.base ?? null,
							client: "saccade-native-tool",
						}),
					});
					created = reply.workspace.seq;
					createdPath = reply.workspace.path;
					target = realpathSync(reply.workspace.path);
				}
				if (!still()) {
					return giveUp(
						"not moved: the handoff was cancelled or the session changed while preparing; no workspace move happened",
					);
				}
				if (signal?.aborted) {
					return giveUp("not moved: the call was cancelled; the hold is released and the session stays");
				}
				const script = handoffHelperScript({
					guard: held.path,
					cli,
					server: url,
					repo,
					num: who.num,
					target,
					cycles: 480,
				});
				const log = /\.guard$/.test(held.path)
					? held.path.replace(/\.guard$/, ".handoff.log")
					: `${held.path}.handoff.log`;
				// the session's actual injected scope rides the arm: what the
				// launch injected is what the resumed session must carry, never
				// a value inferred from a task, message, or name
				const scope = {
					actor: actorInjected ? (process.env.SACCADE_ACTOR ?? null) : null,
					task: process.env.SACCADE_TASK?.trim() ? process.env.SACCADE_TASK.trim() : null,
				};
				// the arm goes live only after its entry is persisted: a failed
				// append must leave nothing for the settle boundary to launch
				pi.appendEntry("saccade_handoff", {
					token,
					nativeId: who.nativeId,
					sessionFile: who.sessionFile,
					toolCall: id,
					incarnation: who.incarnation,
					target,
					created,
					log,
					scope,
				} satisfies HandoffEntryData);
				pendingHandoff = { token, target, toolCall: id, log, created, script };
				// the armed handoff owns the admissions gate from here: the
				// preparing hold transfers, so the guard never blinks
				admissionHold = null;
				const provisioned = created !== null ? ` (provisioned now, birth record ${created})` : "";
				return text(
					[
						`workspace handoff armed: background admissions are held, target ${target}${provisioned}.`,
						"When this run settles, the session shuts down and a bounded helper moves the saved session's working directory and resumes it there - same native id, history, and declared scope.",
						`The helper's outcome log is ${log}; if the session does not come back, that file names the failure.`,
						"Do not start background work. Native human input or a reload cancels the handoff and releases the hold.",
					].join("\n"),
				);
			} catch (error) {
				return giveUp(`not moved: ${errorText(error)}; the hold is released and the session stays`);
			}
		},
	});

	pi.on("session_start", async (_event: SessionStartEvent, ctx: ExtensionContext) => {
		// every start reassociates by the session's actual native id and
		// locator - new, fork, resume, and reload alike - and the old
		// association's waits and queued deliveries die with it, never
		// submitting or delivering into a session this incarnation is not
		cancelAllWaits();
		unpersistedSession = false;
		const who = await attach(ctx);
		if (!who) {
			if (attachError) pi.appendEntry("saccade_attach", { error: attachError });
			// startup that cannot verify association exits, explaining
			// itself above - unless an applicable exact-session guard is
			// held: a guarded reload or reassociation retries lazily
			if (!heldGuard() && !unpersistedSession) {
				// headless runs paint nothing: the explanation goes to
				// stderr so the failure is observable before the exit
				if (!ctx.hasUI) {
					process.stderr.write(`saccade: ${attachError}\n`);
				}
				ctx.shutdown();
			}
			return;
		}
		// activation at a supported boundary: the mailbox drains at start
		try {
			await drainIncoming();
		} catch (error) {
			pi.appendEntry("saccade_attach", {
				incarnation: who.incarnation,
				error: errorText(error),
			});
		}
		// a saved armed handoff correlates against this exact association;
		// an unread correlation stays unconsumed, never stale by guess
		try {
			await consumeHandoff(ctx, who);
		} catch (error) {
			pi.appendEntry("saccade_attach", {
				incarnation: who.incarnation,
				error: `handoff correlation unread: ${errorText(error)}`,
			});
		}
	});

	pi.on("session_shutdown", async () => {
		// idempotent teardown: waits finish, queued deliveries cancel,
		// the association and its env channel go with the session - and
		// the exact-session guard stays kernel-held: process death is
		// its release, and a reload's next start reuses it. A handoff
		// still armed here never launched its helper: reload and session
		// replacement cancel it with its gate; on the quit path the
		// settle boundary already emptied it.
		cancelPendingHandoff("the session ended before the handoff's run settled");
		cancelAllWaits();
		suspendAssociation();
		return undefined;
	});

	// native input releases this integration's own waits, before the
	// input queues; the input itself proceeds untouched. Source matters:
	// an extension-source input IS this integration's own delivery
	// entering the session - an agent message arriving, never human
	// input - so the wait releases with the message's correlation; a
	// human typing keeps the human wording. Neither reading claims a
	// reply was satisfied.
	pi.on("input", async (event: InputEvent, ctx: ExtensionContext) => {
		if (event.source === "extension") {
			const text = event.text;
			const correlated = [...queued.values()].find(
				(message) => envelopeText(message) === text,
			);
			if (correlated) {
				// our own envelope: the wait releases with the real
				// message correlation
				for (const state of [...waits]) {
					finish(state, "message", [correlated]);
				}
			} else {
				// another extension's input is native input too: it
				// yields our wait under a generic wording - never
				// fabricated human origin, never a guessed correlation
				for (const state of [...waits]) {
					finish(state, "extension", []);
				}
			}
			return { action: "continue" };
		}
		// a human correction is a supported recovery boundary: guards
		// for refused or discarded deliveries release here, so an
		// explicit wait or the bridge's next read delivers again
		reconcileDiscarded(ctx);
		// the human took the session back: an armed handoff cancels with
		// its gate, and the input proceeds in the staying session
		cancelPendingHandoff("native human input arrived; the session stays");
		for (const state of [...waits]) {
			finish(state, "input", []);
		}
		return { action: "continue" };
	});
}

// The envelope a delivery injects: machine-supplied origin, ASCII
// separators, the body untouched.
export function envelopeText(message: MailboxMessage): string {
	let head = `[saccade message m-${message.num} | from i-${message.from.num}`;
	if (message.reply_to !== null) head += ` | replying to m-${message.reply_to}`;
	if (message.task !== null) head += ` | task t-${message.task}`;
	head += "]";
	return `${head}\n\n${message.body}`;
}

// The user-facing text of a native user message, whatever content
// shape pi handed in.
export function userText(message: { role: string; content: unknown }): string {
	if (typeof message.content === "string") return message.content;
	if (Array.isArray(message.content)) {
		return message.content
			.filter((block: any) => block?.type === "text")
			.map((block: any) => String(block.text ?? ""))
			.join("");
	}
	return "";
}

// The addressee's mail alone: the mailbox reads both directions, and
// delivery filters on the recipient - a self-addressed message is
// incoming mail like any other.
export function incomingFor(
	incarnation: number,
	messages: MailboxMessage[],
): MailboxMessage[] {
	return messages.filter((m) => m.state === "pending" && m.to.seq === incarnation);
}

// An address is exactly <prefix><ASCII digits>: no bare number, no
// empty tail, no signs, exponents, or other Number coercions, and the
// value must be a safe integer so no address ever rounds or rebinds.
export function numToken(token: string, prefix: string): number {
	const trimmed = token.trim();
	if (!trimmed.startsWith(prefix)) {
		throw new Error(`'${token}' is not an ${prefix}<n> reference`);
	}
	const digits = trimmed.slice(prefix.length);
	if (!/^[0-9]+$/.test(digits)) {
		throw new Error(`'${token}' is not an ${prefix}<n> reference`);
	}
	const n = Number(digits);
	if (!Number.isSafeInteger(n) || n < 0) {
		throw new Error(`'${token}' is not an ${prefix}<n> reference`);
	}
	return n;
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
