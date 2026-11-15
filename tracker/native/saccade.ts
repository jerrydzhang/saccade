// Saccade's native integration, loaded by the saccade-pi composition:
// the session attaches as an incarnation and sends durable addressed
// messages. Arrival is the host's: senders submit into this
// session's terminal through the host adapter, and this integration
// never carries, queues, or confirms deliveries - recorded mail is
// evidence, and `sac messages` is its read.
import { closeSync, openSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { Text } from "@earendil-works/pi-tui";
import { Type, type Static } from "typebox";
import type {
	ExtensionContext,
	ExtensionToolContext,
	ExtensionAPI,
	SessionStartEvent,
} from "@earendil-works/pi-coding-agent";

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

interface Association extends AttachReply {
	nativeId: string;
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
	// this start is the intentional in-memory mode: the pinned contract
	// leaves getSessionFile undefined exactly when persist is false, so
	// there is no saved session to verify and no startup exit
	let unpersistedSession = false;
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
	// identity channel goes with the association - but an exact-session
	// guard stays kernel-held. Process death releases it; a reload's
	// next start reuses it. Releasing before a reattach reply would let
	// a transient tracker outage leave this live process unguarded
	// while a duplicate takes the session's place.
	function suspendAssociation() {
		association = null;
		attachError = null;
		delete process.env.SACCADE_INCARNATION_SEQ;
	}

	// The session's own runtime guard: a sidecar beside the saved
	// session file, keyed by the exact file this process runs. The
	// duplicate-attach backstop is entirely extension-owned - the
	// server never names guard paths.
	function guardSidecar(sessionFile: string): string {
		return `${sessionFile}.guard`;
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
		const guard = sessionFile !== undefined ? guardSidecar(sessionFile) : undefined;
		if (
			held &&
			guard !== undefined &&
			(held.sessionId !== sessionId || held.path !== guard)
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
		if (!sessionId || !sessionFile || guard === undefined) {
			// a persisted session without its file is invalid identity:
			// cannot verify, so the startup exit rule applies
			attachError = "the session carries no saved native file; nothing to register";
			return null;
		}
		// the session's own runtime guard: kernel-held from here to
		// association change or death. Held = a live process already
		// owns this native session: this process is a duplicate - it
		// shuts down before any model work. Unknown = the host offers
		// no flock: fail safe, no association, no participation - an
		// unverified session must never look free. A guard held for
		// this exact session from a previous start is reused, never
		// reopened: the lock is already ours.
		if (heldGuard()?.path === guard) {
			// retained exact-session guard
		} else {
			releaseGuard();
			let fd: number;
			try {
				fd = openSync(guard, "a+");
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
				path: guard,
				sessionId,
				sessionFile,
			});
		}
		// the presented terminal is host disclosure plus door
		// provenance: the pinned host dialect injects the running
		// terminal's own address (HC_TERMINAL_ID), and the launch word
		// names this exact session only when the door launched it - a
		// nested manually started session inherits both and still reads
		// adopted, because its own id names another session
		const disclosed = process.env.HC_TERMINAL_ID;
		const launchedFor = process.env.SACCADE_LAUNCHED_SESSION;
		const terminal = disclosed
			? { address: disclosed, adopted: launchedFor !== sessionId }
			: undefined;
		try {
			const reply = await tracker<AttachReply>("/api/v1/native/attach", {
				method: "POST",
				body: JSON.stringify({
					native_id: sessionId,
					session_file: sessionFile,
					...(terminal ? { terminal } : {}),
				}),
			});
			association = {
				...reply,
				nativeId: sessionId,
				sessionFile,
			};
			process.env.SACCADE_INCARNATION_SEQ = String(reply.incarnation);
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

	function text(content: string) {
		return {
			content: [{ type: "text" as const, text: content }],
			details: undefined,
		};
	}

	function failed(content: string) {
		return { ...text(content), isError: true } as const;
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
			"Send a durable message to another registered incarnation. Addressing uses public numbers: the recipient i-<n>, an optional reply_to m-<n> naming the message this one answers, and an optional task t-<n> riding as context without publishing to its thread. The message records first, then the attempt waits briefly for the recipient's queue to accept it - success means accepted, a failure names the recorded m-N and why; pair with `sac wait i-<n> --timeout <secs>` when the reply matters now.",
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
						"; accepted into their session's input queue",
				);
			} catch (error) {
				return failed(`not sent: ${errorText(error)}`);
			}
		},
	});

	pi.on("session_start", async (_event: SessionStartEvent, ctx: ExtensionContext) => {
		// every start reassociates by the session's actual native id and
		// locator - new, fork, resume, and reload alike - and the old
		// association dies with it
		unpersistedSession = false;
		const who = await attach(ctx);
		if (!who) {
			// a process holding no guard never writes the history it could not claim
			if (attachError && heldGuard()) {
				pi.appendEntry("saccade_attach", { error: attachError });
			}
			// startup that cannot verify association exits, explaining
			// itself above - unless an applicable exact-session guard is
			// held: a guarded reload or reassociation retries lazily
			if (!heldGuard() && !unpersistedSession) {
				// stderr carries the refusal locally even when a UI exists
				process.stderr.write(`saccade: ${attachError}\n`);
				ctx.shutdown();
			}
			return;
		}
		// no mailbox drain at start: recorded mail without a confirmed
		// submission is evidence, and nothing delivers it later
	});

	pi.on("session_shutdown", async () => {
		// idempotent teardown: the association and its env channel go
		// with the session - and the exact-session guard stays
		// kernel-held: process death is its release, and a reload's
		// next start reuses it.
		suspendAssociation();
		return undefined;
	});
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
