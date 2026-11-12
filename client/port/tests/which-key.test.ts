// PENDING (batch 3, red before the code): which-key is a property of the
// matcher's tables, not a space special case (human ruling r110). These
// read a PROPOSED pure projection the feature adds:
//
//   src/lib/input/which-key.ts
//     continuations(context: InputContext, prefix: readonly KeyInput[])
//       -> readonly { keys: string; label: string; actionId: string }[]
//     whichKeyTunables: { delayMs: number; suspendsTimeout: boolean }
//
// keys/label/actionId use describeCommands' own projection
// (formatSequence labels), so help and which-key cannot drift from the
// table. The tunables are named and read, never pinned here.
import { expect, test } from "vitest";
import {
  commandContext,
  describeCommands,
  key,
  sequence,
  shift,
  type CommandBinding,
  type InputContext,
  type KeyInput,
} from "../src/lib/input/input-context.ts";

type Entry = { keys: string; label: string; actionId: string };
type WhichKeyModule = {
  continuations(context: InputContext, prefix: readonly KeyInput[]): readonly Entry[];
  whichKeyTunables: { delayMs: number; suspendsTimeout: boolean };
};

// a runtime path, so the suite collects (and typechecks) before the
// module exists
const MODULE = "../src/lib/input/which-key.ts";
const whichKey = (): Promise<WhichKeyModule> => import(/* @vite-ignore */ MODULE) as Promise<WhichKeyModule>;

const stroke = (k: string, over: Partial<KeyInput> = {}): KeyInput =>
  ({ key: k, ctrl: false, shift: false, alt: false, meta: false, ...over });

const bind = (id: string, label: string, ...keys: string[]): CommandBinding => ({
  action: { id, label, run: () => {} },
  keys: [sequence(key(keys[0]!), ...keys.slice(1).map((k) => key(k)))],
});

function table(): { context: InputContext; bindings: CommandBinding[] } {
  const bindings: CommandBinding[] = [
    bind("t.top", "scroll to the top", "g", "g"),
    bind("t.down", "scroll down", "j"),
    bind("t.picker", "open a location", " ", " "),
    bind("t.find", "find", " ", "/"),
    { action: { id: "t.bottom", label: "scroll to the bottom", run: () => {} }, keys: [sequence(shift("G"))] },
  ];
  // the table is read at call time: adding to it is the only change
  return { context: commandContext(() => bindings), bindings };
}

test("a pending prefix lists exactly its continuations from the table", { tags: ["pending"] }, async () => {
  const { continuations } = await whichKey();
  const { context } = table();
  expect(continuations(context, [stroke("g")])).toEqual([{ keys: "g g", label: "scroll to the top", actionId: "t.top" }]);
  const space = continuations(context, [stroke(" ")]).map((entry) => entry.keys).sort();
  expect(space).toEqual(["spc /", "spc spc"]);
});

test("composition: a command added to the table appears with no other change", { tags: ["pending"] }, async () => {
  const { continuations } = await whichKey();
  const { context, bindings } = table();
  bindings.push(bind("t.trail", "the trail", "g", "t"));
  expect(continuations(context, [stroke("g")]).map((entry) => entry.actionId).sort()).toEqual(["t.top", "t.trail"]);
});

test("help: the empty prefix lists every bound command with its keys", { tags: ["pending"] }, async () => {
  const { continuations } = await whichKey();
  const { context } = table();
  const listed = continuations(context, []);
  for (const command of describeCommands(context)) {
    for (const keys of command.keyLabels) {
      expect(listed).toContainEqual({ keys, label: command.label, actionId: command.actionId });
    }
  }
});

test("the popup delay and the timeout-suspend are named tunables", { tags: ["pending"] }, async () => {
  const { whichKeyTunables } = await whichKey();
  expect(whichKeyTunables.delayMs).toBeGreaterThan(0);
  expect(typeof whichKeyTunables.suspendsTimeout).toBe("boolean");
});
