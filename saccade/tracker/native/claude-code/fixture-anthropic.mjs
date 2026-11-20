// The fixture Anthropic endpoint: loopback only, minimal messages
// replies so one turn completes and saves a transcript. No model, no
// auth service; any key is accepted.
import { createServer } from "node:http";
import { appendFileSync } from "node:fs";

const logPath = process.argv[2];
const port = Number(process.argv[3]);

const reply = {
	id: "msg_fixture_1",
	type: "message",
	role: "assistant",
	model: "claude-fake",
	content: [{ type: "text", text: "fixture-ok" }],
	stop_reason: "end_turn",
	stop_sequence: null,
	usage: { input_tokens: 1, output_tokens: 1 },
};

function log(entry) {
	appendFileSync(logPath, JSON.stringify({ at: Date.now(), ...entry }) + "\n");
}

createServer((req, res) => {
	const chunks = [];
	req.on("data", (c) => chunks.push(c));
	req.on("end", () => {
		log({ url: req.url, method: req.method });
		const path = new URL(req.url, "http://fixture").pathname;
		if (path.endsWith("/messages") || path.endsWith("/count_tokens")) {
			if (path.endsWith("/count_tokens")) {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ input_tokens: 1 }));
				return;
			}
			// echo the requested model: the pinned client rejects a reply
			// whose model does not match the request
			let requested = "claude-sonnet-4-5";
			try {
				requested = JSON.parse(Buffer.concat(chunks).toString("utf8")).model;
			} catch {}
			res.writeHead(200, { "content-type": "application/json" });
			res.end(JSON.stringify({ ...reply, model: requested }));
			return;
		}
		res.writeHead(404, { "content-type": "application/json" });
		res.end(JSON.stringify({ error: { type: "not_found_error" } }));
	});
}).listen(port, "127.0.0.1", () => log({ listening: port }));
