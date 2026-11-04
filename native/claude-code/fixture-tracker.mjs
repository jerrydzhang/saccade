// The fixture tracker for isolated pinned-CC gates: serves the
// version line and a valid attach reply, and logs every request to a
// file the PTY driver reads. Delays and refusals are programmable so
// startup, failure, and duplicate runs share one harness. Loopback
// only; nothing here is the real tracker.
import { createServer } from "node:http";
import { appendFileSync, writeFileSync } from "node:fs";

const logPath = process.argv[2];
const port = Number(process.argv[3]);
// delay version replies (ms) to hold SessionStart open for raw-input
// evidence; 0 for normal runs
const versionDelay = Number(process.argv[4] || 0);
// attach refusal: "ok" or a refusal status code
const attachMode = process.argv[5] || "ok";

const attachReply = {
	registered: true,
	incarnation: 42,
	num: 7,
	records: [],
};

function log(entry) {
	appendFileSync(logPath, JSON.stringify({ at: Date.now(), ...entry }) + "\n");
}

writeFileSync(logPath, "");

const server = createServer((req, res) => {
	const chunks = [];
	req.on("data", (c) => chunks.push(c));
	req.on("end", () => {
		const body = Buffer.concat(chunks).toString("utf8");
		log({ url: req.url, method: req.method, body });
		const finish = () => {
			if (req.url === "/api/v1/version") {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ version: "0.18.0", db: "/db" }));
				return;
			}
			if (req.url === "/api/v1/native/attach") {
				if (attachMode === "ok") {
					res.writeHead(200, { "content-type": "application/json" });
					res.end(JSON.stringify(attachReply));
				} else {
					res.writeHead(Number(attachMode), { "content-type": "application/json" });
					res.end(
						JSON.stringify({
							error: { code: "saved_session_mismatch", detail: "fixture refusal" },
						}),
					);
				}
				return;
			}
			res.writeHead(404, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: { code: "not_found", detail: req.url } }));
		};
		if (req.url === "/api/v1/version" && versionDelay > 0) {
			setTimeout(finish, versionDelay);
		} else {
			finish();
		}
	});
});

server.listen(port, "127.0.0.1", () => log({ listening: port }));
