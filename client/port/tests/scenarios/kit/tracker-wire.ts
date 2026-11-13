// The scenarios' door into the running throwaway tracker: plain fetch
// over its HTTP wire, so both realms use it (the browser test page is
// an allowed origin; the native project runs in node). Writes commit
// boards mid-scenario; the window-state read checks what the app
// stored (a birth location, a link) on the store it really talks to.
import type { TrackerDoor, WindowStateView } from "./driver.ts";

const CONTEXT = { actor: "port-suite", tier: "agent", scope: null };
const CLIENT = "port-suite";

function base64(text: string): string {
  let binary = "";
  for (const byte of new TextEncoder().encode(text)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function base64Of(content: string | Uint8Array): string {
  if (typeof content === "string") return base64(content);
  let binary = "";
  for (const byte of content) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export function trackerDoor(base: string): TrackerDoor {
  const call = async (method: string, route: string, payload?: unknown): Promise<Record<string, unknown>> => {
    const reply = await fetch(`${base}${route}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    const body = (await reply.json()) as Record<string, unknown>;
    if (!reply.ok) throw new Error(`${method} ${route} answered ${reply.status}: ${JSON.stringify(body).slice(0, 300)}`);
    return body;
  };
  return {
    async createTask(name) {
      const reply = await call("POST", "/api/v1/command", {
        context: CONTEXT,
        command: { create_task: { name } },
        at: null,
        client: CLIENT,
      });
      return String(reply.id);
    },
    async commit(task, board, message, files = []) {
      const reply = await call("POST", "/api/v1/boards/commit", {
        context: CONTEXT,
        task: Number(task.replace(/^t-/, "")),
        files: {
          files: [
            { path: "board.yaml", content: base64(board) },
            ...files.map((file) => ({ path: file.path, content: base64Of(file.content) })),
          ],
        },
        message,
        client: CLIENT,
      });
      return { revision: Number(reply.revision) };
    },
    async windowState(id, device) {
      const reply = await call("GET", `/api/v1/windows/states/${encodeURIComponent(id)}?device=${encodeURIComponent(device)}`);
      return reply.state as WindowStateView;
    },
    async attachedWindow(id, device) {
      // a window this device already shows, born and attached the way
      // a window's own boot does it (client/port/src/lib/windows.ts
      // create + attach): the shell mock's other windows exist here
      const created = await fetch(`${base}/api/v1/windows/states`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, device }),
      });
      if (created.status !== 200 && created.status !== 409) throw new Error(`window state ${id}: ${created.status}`);
      await call("POST", `/api/v1/windows/devices/${encodeURIComponent(device)}/attach`, { state: id });
    },
  };
}
