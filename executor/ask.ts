/**
 * The ask door — saccade's only extension surface. Loaded by the runner
 * at spawn (--extension), it registers the one tool an incarnation has
 * for reaching its human: a blocking question.
 *
 * The tool writes an Ask comment through the tracker's existing command
 * door under the run's own actor (SACCADE_ACTOR possession names agent
 * tier), blocks on `sac wait` until the answer lands, and returns the
 * release — which carries the answer — as the tool result. The answer is
 * an authored comment on the thread; nothing middlemans.
 *
 * Environment the runner provides: SACCADE_TASK (the task being served),
 * SACCADE_SERVER (the tracker's command door), SACCADE_ACTOR (the run's
 * derived attribution), SACCADE_SAC (this binary's path).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "ask",
    label: "Ask",
    description:
      "Ask the human a blocking question about the task's work. The question " +
      "lands on the task's thread as an ask; the tool blocks until the human " +
      "answers; the answer returns as the tool result. Use it when a decision, " +
      "a missing fact, or a judgment call blocks the work — not for " +
      "observations, which belong on the thread as notes.",
    parameters: Type.Object({
      question: Type.String({
        description: "The question, self-contained — it stands on the thread",
      }),
    }),
    async execute(toolCallId, params, signal) {
      const task = process.env.SACCADE_TASK;
      if (!task || Number.isNaN(Number(task))) {
        throw new Error(
          "ask: no SACCADE_TASK in the environment; this tool works only inside a saccade run",
        );
      }
      const server = process.env.SACCADE_SERVER ?? "http://127.0.0.1:8811";
      const actor = process.env.SACCADE_ACTOR ?? "run";
      const res = await fetch(`${server}/api/v1/command`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          context: { actor, tier: "agent" },
          at: null,
          command: {
            comment: {
              target: { task: Number(task) },
              body: params.question,
              kind: "ask",
            },
          },
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`ask: the comment door refused (${res.status}): ${detail}`);
      }
      const reply = (await res.json()) as { records?: Array<{ seq?: number }> };
      const seq = reply.records?.[reply.records.length - 1]?.seq;
      if (seq === undefined) {
        throw new Error("ask: the comment door wrote no record");
      }

      // block on the wait: the release carries the answer
      const sac = process.env.SACCADE_SAC ?? "sac";
      return await new Promise((resolve, reject) => {
        const child = spawn(sac, ["wait", `c-${seq}`], { signal });
        let out = "";
        child.stdout.on("data", (d) => (out += d));
        child.on("error", reject);
        child.on("close", (code) => {
          if (code === 0 && out.trim()) {
            resolve({ content: [{ type: "text", text: out.trim() }], details: {} });
          } else {
            reject(new Error(`ask: sac wait c-${seq} exited ${code}: ${out.trim()}`));
          }
        });
      });
    },
  });
}
