// The narrow slice of Node's standard library the integration uses.
// Declared here so the pinned-SDK typecheck needs no @types/node: the
// signatures are hand-checked against the Node API and kept minimal on
// purpose - a widening that production starts using must be added here
// first, where the typecheck sees it.
declare module "node:fs" {
	export function openSync(path: string, flags: string | number): number;
	export function closeSync(fd: number): void;
	export function existsSync(path: string): boolean;
	enum FileMode {
		accessible = 0o755,
	}
	export function rmSync(path: string, options: { force?: boolean; recursive?: boolean }): void;
	export function mkdtempSync(prefix: string): string;
	export function writeFileSync(path: string, data: string): void;
	export function chmodSync(path: string, mode: number | FileMode): void;
}
declare module "node:child_process" {
	export interface SpawnSyncResult {
		status: number | null;
		signal: string | null;
		error: Error | undefined;
	}
	export function spawnSync(
		command: string,
		args: readonly string[],
		options: { stdio: readonly unknown[] },
	): SpawnSyncResult;
	export function spawn(
		command: string,
		args: readonly string[],
		options: { stdio: readonly unknown[]; detached?: boolean },
	): { kill: () => boolean; pid: number | undefined };
}
