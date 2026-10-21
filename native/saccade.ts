// Saccade's native integration, loaded by the saccade-pi composition:
// the session attaches as an incarnation, sends durable addressed
// messages, and waits that incoming messages or native human input
// release — releasing only this integration's own wait, never
// interrupting other work.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const POLL_MS = 1500;

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

export default function saccadeIntegration(pi: ExtensionAPI) {
	// attachment state: one incarnation per native session, resolved or
	// registered at session start
	let attached: AttachReply | null = null;
	let attachError: string | null = null;
	// the integration's own wait: released by incoming mail or native
	// human input, never by aborting other work
	let waitRelease: ((reason: "message" | "input") => void) | null = null;
	let waitTimer: ReturnType<typeof setInterval> | null = null;
	// the mail a released wait holds for delivery
	let held: MailboxMessage[] = [];

	const server = () => {
		const url = process.env.SACCADE_SERVER;
		return url && url.trim() !== "" ? url.trim() : undefined;
	};

	const actor = () => process.env.SACCADE_ACTOR || "saccade-pi";

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

	/// Resolve-or-register this session's incarnation; the reply is
	/// the machine identity, and the env channel rides it for `sac`.
	async function attach(ctx: {
		sessionManager: { getSessionId(): string | undefined; getSessionFile(): string | undefined };
	}): Promise<AttachReply | null> {
		if (attached || attachError) return attached;
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
			attached = await tracker<AttachReply>("/api/v1/native/attach", {
				method: "POST",
				body: JSON.stringify({ native_id: sessionId, session_file: sessionFile }),
			});
			process.env.SACCADE_INCARNATION_SEQ = String(attached.incarnation);
			return attached;
		} catch (error) {
			attachError = errorText(error);
			return null;
		}
	}

	function envelopeText(message: MailboxMessage): string {
		let head = `[saccade message m-${message.num} · from i-${message.from.num}`;
		if (message.reply_to !== null) head += ` · replying to m-${message.reply_to}`;
		if (message.task !== null) head += ` · task t-${message.task}`;
		head += "]";
		return `${head}\n\n${message.body}`;
	}

	/// Deliver one message natively, then record the submission: the
	/// bookkeeping never claims reading, agreement, or completion.
	async function deliver(message: MailboxMessage): Promise<void> {
		pi.sendUserMessage(envelopeText(message), { deliverAs: "followUp" });
		await tracker("/api/v1/messages/submit", {
			method: "POST",
			body: JSON.stringify({
				incarnation: attached!.incarnation,
				message: message.seq,
			}),
		});
	}

	async function mailbox(): Promise<MailboxMessage[]> {
		const reply = await tracker<{ messages: MailboxMessage[] }>(
			`/api/v1/messages/pending?incarnation=${attached!.incarnation}`,
		);
		return reply.messages;
	}

	function requires(): string | null {
		return attached ? null : (attachError ?? "not attached yet");
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

	function stopWaiting() {
		if (waitTimer !== null) {
			clearInterval(waitTimer);
			waitTimer = null;
		}
	}

	pi.registerTool({
		name: "saccade_identity",
		label: "Saccade identity",
		description:
			"Report this session's registered saccade identity: the incarnation's public number i-<n> that others address, and whether it registered now or resolved an existing registration.",
		parameters: { type: "object", properties: {} },
		execute: async (_id, _params, _signal, ctx) => {
			const who = await attach(ctx);
			if (!who) return failed(`not registered: ${attachError}`);
			return text(
				`this session is i-${who.num} (birth seq ${who.incarnation}${who.registered ? ", registered now" : ", already registered"}); others address it as i-${who.num}`,
			);
		},
	});

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
			const who = await attach(ctx);
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
								task: null,
								incarnation: who.incarnation,
								workspace: null,
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
		execute: async (_id, params, _signal, ctx) => {
			const who = await attach(ctx);
			if (!who) return failed(`cannot wait: ${attachError}`);
			const wantReply = params.reply_to ? numToken(params.reply_to, "m-") : null;
			const released = await new Promise<"message" | "input">((resolve) => {
				waitRelease = resolve;
				waitTimer = setInterval(() => {
					mailbox()
						.then((incoming) => {
							const mine = incoming.filter(
								(m) =>
									m.state === "pending" &&
									m.from.seq !== who.incarnation &&
									(wantReply === null || m.reply_to === wantReply),
							);
							if (mine.length > 0 && waitRelease) {
								held = mine;
								stopWaiting();
								waitRelease = null;
								resolve("message");
							}
						})
						.catch(() => {
							// a poll that cannot reach the tracker keeps waiting
						});
				}, POLL_MS);
			});
			if (released === "input") {
				return text(
					"the wait ended because native human input arrived; any pending messages stay in the mailbox — saccade_wait again to receive them",
				);
			}
			const delivered = held;
			held = [];
			for (const message of delivered) {
				await deliver(message);
			}
			const named = delivered
				.map((m) => `m-${m.num} from i-${m.from.num}`)
				.join(", ");
			return text(
				`received ${named}; delivered into your session as your next input — reply through saccade_send with reply_to`,
			);
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		const who = await attach(ctx);
		if (!who) {
			if (attachError) pi.appendEntry("saccade_attach", { error: attachError });
			return;
		}
		// activation at a supported boundary: the mailbox drains at start
		try {
			const incoming = (await mailbox()).filter(
				(m) => m.state === "pending" && m.from.seq !== who.incarnation,
			);
			for (const message of incoming) {
				await deliver(message);
			}
		} catch (error) {
			pi.appendEntry("saccade_attach", {
				incarnation: who.incarnation,
				error: errorText(error),
			});
		}
	});

	// native human input releases only this integration's own wait,
	// before the input queues; the input itself proceeds untouched
	pi.on("input", async () => {
		if (waitRelease) {
			const release = waitRelease;
			waitRelease = null;
			stopWaiting();
			release("input");
		}
		return { action: "continue" };
	});
}

function numToken(token: string, prefix: string): number {
	const trimmed = token.trim();
	const bare = trimmed.startsWith(prefix) ? trimmed.slice(prefix.length) : trimmed;
	const n = Number(bare);
	if (!Number.isInteger(n) || n < 0) {
		throw new Error(`'${token}' is not a ${prefix}<n> reference`);
	}
	return n;
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
