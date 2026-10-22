// Saccade's native integration, loaded by the saccade-pi composition:
// the session attaches as an incarnation, sends durable addressed
// messages, and waits that incoming messages or native human input
// release - releasing only this integration's own wait, never
// interrupting other work.
import { closeSync, openSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { Text } from "@earendil-works/pi-tui";
import { Type, type Static } from "typebox";
import type {
	ExtensionContext,
	ExtensionToolContext,
	ExtensionAPI,
	InputEvent,
	SessionStartEvent,
} from "@earendil-works/pi-coding-agent";

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

export default function saccadeIntegration(pi: ExtensionAPI) {
	// the attach record's reader: interactive chat paints custom
	// entries only through a registered renderer
	pi.registerEntryRenderer("saccade_attach", (entry) => {
		const data = (entry.data ?? {}) as { error?: string };
		return new Text(data.error ? `saccade: ${data.error}` : "saccade: attached");
	});
	// the session's association: resolved or registered fresh at every
	// session start by the session's actual native id and locator
	let association: Association | null = null;
	let attachError: string | null = null;
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

	// agent possession is set at load, before any attach: a tracker
	// outage must still leave nested sac calls acting as the agent, and
	// an injected name always wins
	const injectedActor = process.env.SACCADE_ACTOR;
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
		if (!server()) {
			attachError = "no SACCADE_SERVER routing; the session stays unregistered";
			return null;
		}
		if (!sessionId || !sessionFile) {
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
		return undefined;
	});
	pi.on("agent_end", (_event: unknown, ctx: ExtensionContext) => {
		confirmDeliveries(ctx);
		return undefined;
	});
	pi.on("agent_settled", (_event: unknown, ctx: ExtensionContext) => {
		confirmDeliveries(ctx);
		return undefined;
	});

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

	pi.on("session_start", async (_event: SessionStartEvent, ctx: ExtensionContext) => {
		// every start reassociates by the session's actual native id and
		// locator - new, fork, resume, and reload alike - and the old
		// association's waits and queued deliveries die with it, never
		// submitting or delivering into a session this incarnation is not
		cancelAllWaits();
		const who = await attach(ctx);
		if (!who) {
			if (attachError) pi.appendEntry("saccade_attach", { error: attachError });
			// startup that cannot verify association exits, explaining
			// itself above - unless an applicable exact-session guard is
			// held: a guarded reload or reassociation retries lazily
			if (!heldGuard()) {
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
	});

	pi.on("session_shutdown", async () => {
		// idempotent teardown: waits finish, queued deliveries cancel,
		// the association and its env channel go with the session - and
		// the exact-session guard stays kernel-held: process death is
		// its release, and a reload's next start reuses it
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
