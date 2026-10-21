// Saccade's native integration, loaded by the saccade-pi composition:
// the session attaches as an incarnation, sends durable addressed
// messages, and waits that incoming messages or native human input
// release - releasing only this integration's own wait, never
// interrupting other work.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const POLL_MS = 1500;
// the automatic actor fallback: agent possession when the launch did
// not inject a name; an injected SACCADE_ACTOR always wins
const FALLBACK_ACTOR = "saccade-pi";

interface AttachReply {
	registered: boolean;
	incarnation: number;
	num: number;
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
}

// One wait tool call owns one of these; nothing about a wait lives in
// shared state, so a stale poll can never observe or strand a newer
// wait's release.
type Outcome = "message" | "input" | "cancelled" | "detached";

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

export default function saccadeIntegration(pi: ExtensionAPI) {
	// the session's association: resolved or registered fresh at every
	// session start by the session's actual native id and locator
	let association: Association | null = null;
	let attachError: string | null = null;
	// the waits this integration currently owns
	const waits = new Set<WaitState>();
	// envelopes queued natively, submissions not yet confirmed
	const deliveries = new Set<QueuedDelivery>();

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

	// Detach whatever this session carried: the identity channel goes
	// with it, and queued deliveries die unsubmitted - a reassociated
	// session never submits as the previous incarnation.
	function detach() {
		association = null;
		attachError = null;
		deliveries.clear();
		delete process.env.SACCADE_INCARNATION_SEQ;
	}

	// Resolve-or-register this session's incarnation from the session's
	// actual native id and saved locator - never from a remembered
	// attachment. The reply is the machine identity, and the env channel
	// rides it for `sac`.
	async function attach(ctx: {
		sessionManager: { getSessionId(): string | undefined; getSessionFile(): string | undefined };
	}): Promise<Association | null> {
		detach();
		if (!server()) {
			attachError = "no SACCADE_SERVER routing; the session stays unregistered";
			return null;
		}
		const sessionId = ctx.sessionManager.getSessionId();
		const sessionFile = ctx.sessionManager.getSessionFile();
		if (!sessionId || !sessionFile) {
			attachError = "the session carries no saved native file; nothing to register";
			return null;
		}
		try {
			const reply = await tracker<AttachReply>("/api/v1/native/attach", {
				method: "POST",
				body: JSON.stringify({ native_id: sessionId, session_file: sessionFile }),
			});
			association = { ...reply, nativeId: sessionId, sessionFile };
			process.env.SACCADE_INCARNATION_SEQ = String(reply.incarnation);
			return association;
		} catch (error) {
			attachError = errorText(error);
			return null;
		}
	}

	// An attached session stays attached; a failed attach retries here,
	// so explicit later tools recover when the tracker returns.
	async function ensureAttached(ctx: {
		sessionManager: { getSessionId(): string | undefined; getSessionFile(): string | undefined };
	}): Promise<Association | null> {
		return association ?? (await attach(ctx));
	}

	async function mailbox(incarnation: number): Promise<MailboxMessage[]> {
		const reply = await tracker<{ messages: MailboxMessage[] }>(
			`/api/v1/messages/pending?incarnation=${incarnation}`,
		);
		return reply.messages;
	}

	// Queue one message natively and return: the wait tool must never
	// block on what the native queue does after the current run. The
	// submission is confirmed separately, by the delivery observers.
	function queueDelivery(message: MailboxMessage): void {
		if (association === null) return;
		const delivery: QueuedDelivery = {
			envelope: envelopeText(message),
			messageSeq: message.seq,
			incarnation: association.incarnation,
			sessionFile: association.sessionFile,
		};
		deliveries.add(delivery);
		pi.sendUserMessage(delivery.envelope, { deliverAs: "followUp" });
	}

	// Extension message events are not proof of insertion - the
	// message_end hook runs before the SessionManager append - so
	// confirmation reads the saved entries themselves.
	function savedEntryHas(ctx: {
		sessionManager: { getEntries(): Array<{ type: string; message?: { role: string; content: unknown } }> };
	}, envelope: string): boolean {
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
	function confirmDeliveries(ctx: {
		sessionManager: { getEntries(): Array<{ type: string; message?: { role: string; content: unknown } }> };
	}): void {
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
	pi.on("message_end", (_event, ctx) => {
		confirmDeliveries(ctx);
		return undefined;
	});
	pi.on("agent_end", (_event, ctx) => {
		confirmDeliveries(ctx);
		return undefined;
	});
	pi.on("agent_settled", (_event, ctx) => {
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
		parameters: { type: "object", properties: {} },
		execute: async (_id, _params, _signal, ctx) => {
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
		parameters: {
			type: "object",
			properties: {
				to: { type: "string", description: "recipient incarnation, i-<n>" },
				body: { type: "string", description: "the message body" },
				reply_to: { type: "string", description: "the message answered, m-<n>" },
				task: { type: "string", description: "task context, t-<n>" },
			},
			required: ["to", "body"],
		},
		execute: async (_id, params, _signal, ctx) => {
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
		parameters: {
			type: "object",
			properties: {
				reply_to: {
					type: "string",
					description: "only messages answering m-<n> release the wait",
				},
			},
		},
		execute: async (_id, params, signal, ctx) => {
			const who = await ensureAttached(ctx);
			if (!who) return failed(`cannot wait: ${attachError}`);
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

	pi.on("session_start", async (_event, ctx) => {
		// every start reassociates by the session's actual native id and
		// locator - new, fork, resume, and reload alike - and the old
		// association's waits and queued deliveries die with it, never
		// submitting or delivering into a session this incarnation is not
		cancelAllWaits();
		const who = await attach(ctx);
		if (!who) {
			if (attachError) pi.appendEntry("saccade_attach", { error: attachError });
			return;
		}
		// activation at a supported boundary: the mailbox drains at start
		try {
			const incoming = incomingFor(
				who.incarnation,
				await mailbox(who.incarnation),
			);
			for (const message of incoming) {
				queueDelivery(message);
			}
		} catch (error) {
			pi.appendEntry("saccade_attach", {
				incarnation: who.incarnation,
				error: errorText(error),
			});
		}
	});

	pi.on("session_shutdown", async () => {
		// idempotent teardown: waits finish, queued deliveries cancel,
		// the association and its env channel go with the session
		cancelAllWaits();
		detach();
		return undefined;
	});

	// native human input releases this integration's own waits, before
	// the input queues; the input itself proceeds untouched
	pi.on("input", async () => {
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
