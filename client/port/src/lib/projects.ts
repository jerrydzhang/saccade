// The device's project registry: the list of registered paths -
// "machine:/path/to/repository", the repository whose declaration file
// names the project - and nothing else about a project. The
// declaration (workbench.toml at the repository's root) is read
// through the one core module at every act that needs its facts, so
// what the device keeps can never disagree with what the file says.
// Registering is its own act: it reads the file there and has the
// project; opening a path never registers one.

import type { Machine, MachineAsks } from "./asks.ts";
import { SourceUnavailableError } from "./board/errors.ts";

const PROJECTS_KEY = "saccade.projects";

/** What a declaration file says: the project's name, and its
 * tracker's port when it declares one - a project with no [tracker]
 * section registers and lists, it just has no tracker source. */
export type Declaration = {
  name: string;
  port: number | null;
};

/** The declaration file's one shape (the ratified single-project
 * form): `name = "..."`, then an optional `[tracker]` section
 * carrying `port = N`. Anything else is ignored; a file that names no
 * project is not a declaration. */
export function parseDeclaration(text: string): Declaration | null {
  let name: string | null = null;
  let port: number | null = null;
  let inTracker = false;
  for (const rawLine of text.split("\n")) {
    const line = rawLine.replace(/(^|\s)#.*$/, "").trim();
    if (line.length === 0) continue;
    if (line === "[tracker]") {
      inTracker = true;
      continue;
    }
    if (line.startsWith("[") && line.endsWith("]")) {
      inTracker = false;
      continue;
    }
    const assign = /^([A-Za-z0-9_-]+)\s*=\s*(.+)$/.exec(line);
    if (!assign || assign[1] === undefined || assign[2] === undefined) continue;
    const key = assign[1];
    const value = assign[2].trim();
    if (key === "name" && name === null) {
      const quoted = /^"([^"]*)"$/.exec(value);
      if (quoted && quoted[1] !== undefined) name = quoted[1];
      continue;
    }
    if (key === "port" && inTracker && port === null) {
      const number = /^\d+$/.exec(value);
      if (number) {
        const parsed = Number(value);
        if (Number.isFinite(parsed) && parsed <= 65535) port = parsed;
      }
    }
  }
  if (name === null || name.trim().length === 0) return null;
  return { name, port };
}

function readStore(): string[] {
  try {
    const raw = localStorage.getItem(PROJECTS_KEY);
    const parsed: unknown = raw === null ? [] : JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((entry): entry is string => typeof entry === "string" && entry.includes(":/"));
  } catch {
    return [];
  }
}

function writeStore(paths: readonly string[]): void {
  try {
    localStorage.setItem(PROJECTS_KEY, JSON.stringify([...paths]));
  } catch {
    // storage refused: the project stays registered for this session
  }
}

/** The address a registration or a project fact carries: the machine
 * and the repository's absolute path, never joined - the machine name
 * may carry colons itself. */
export type RegisteredPath = { machine: Machine; path: string };

export function formatRegisteredPath(registered: RegisteredPath): string {
  return `${registered.machine}:${registered.path}`;
}

export function parseRegisteredPath(input: string): RegisteredPath | null {
  const raw = input.trim();
  const cut = raw.indexOf(":/");
  if (cut <= 0) return null;
  const machine = raw.slice(0, cut);
  const path = raw.slice(cut + 1);
  if (machine.length === 0 || !path.startsWith("/") || path.includes("\0")) return null;
  return { machine, path };
}

/** The facts a project's declaration carries beside its address:
 * everything the rail and the tracker reads need, read fresh from the
 * file through the module (the session cache only spares re-reads
 * within an unchanged store). */
export type Project = RegisteredPath & { declaration: Declaration };

export class ProjectRegistry {
  // session memory of the declarations behind the stored paths; the
  // device store keeps the paths alone
  private cache = new Map<string, Declaration>();
  private storeEpoch = 0;
  private cacheEpoch = -1;

  constructor(private readonly asks: MachineAsks) {}

  private changed(): void {
    this.storeEpoch += 1;
  }

  /** The registered paths, in registration order. */
  registeredPaths(): RegisteredPath[] {
    return readStore().flatMap((entry) => {
      const parsed = parseRegisteredPath(entry);
      return parsed ? [parsed] : [];
    });
  }

  private async declarationOf(registered: RegisteredPath): Promise<Declaration> {
    const key = formatRegisteredPath(registered);
    if (this.cacheEpoch !== this.storeEpoch) {
      this.cache.clear();
      this.cacheEpoch = this.storeEpoch;
    }
    const cached = this.cache.get(key);
    if (cached) return cached;
    // the registered path names the repository; the declaration is the
    // file at its root
    const file = `${registered.path.replace(/\/+$/, "")}/workbench.toml`;
    let text: string;
    try {
      text = await this.asks.readFile(registered.machine, file);
    } catch (error) {
      throw new SourceUnavailableError(
        "lane-unavailable",
        `the declaration at ${key} cannot be read: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const declaration = parseDeclaration(text);
    if (declaration === null) {
      throw new SourceUnavailableError(
        "asset-missing",
        `${key} carries no project: workbench.toml must name the project (name = "...")`,
      );
    }
    this.cache.set(key, declaration);
    return declaration;
  }

  /** Register the project whose repository lives at the path: the
   * declaration there must name the project. Registering opens
   * nothing; the registered path joins the device's list. */
  async register(registered: RegisteredPath): Promise<Declaration> {
    const declaration = await this.declarationOf(registered);
    const key = formatRegisteredPath(registered);
    const paths = readStore().filter((entry) => entry !== key);
    paths.push(key);
    writeStore(paths);
    this.changed();
    return declaration;
  }

  /** Forget a registered project: its path leaves the device's list;
   * the repository and its record are untouched. */
  forget(registered: RegisteredPath): void {
    const key = formatRegisteredPath(registered);
    writeStore(readStore().filter((entry) => entry !== key));
    this.cache.delete(key);
    this.changed();
  }

  /** Every registered project whose declaration reads: the rail's
   * list. A path whose file went missing or names nothing is skipped
   * here (its problem is registration's to say, not the rail's). */
  async projects(): Promise<Project[]> {
    const out: Project[] = [];
    for (const registered of this.registeredPaths()) {
      try {
        const declaration = await this.declarationOf(registered);
        out.push({ ...registered, declaration });
      } catch {
        // an unreadable declaration is no project
      }
    }
    return out;
  }

  /** The registered project naming itself `name`: the address a
   * "name/t-N" location resolves to. */
  async projectByName(name: string): Promise<Project | null> {
    const projects = await this.projects();
    return projects.find((project) => project.declaration.name === name) ?? null;
  }
}
