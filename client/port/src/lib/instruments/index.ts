// The instruments' index: the ONE file that names the instruments.
// Each instrument module carries its own registration; this file calls
// them in claim order - the narrower claims first (a project's task
// list before the board's everything-else) - and re-exports the faces
// the frame reads. Adding an instrument touches this file and the
// instrument's own module, no frame file.
import {
  instrumentOfKind,
  instrumentForAddress,
  registeredInstruments,
  constructInstrument,
  type RegisteredInstrument,
  type RunningInstrument,
  type WindowEnvironment,
  type WindowServices,
  type RenderPair,
} from "./registry.ts";
import { register as registerBoard } from "../../instruments/board.ts";
import { register as registerTasks } from "../../instruments/tasks.ts";

const tasks = registerTasks();
const board = registerBoard();

export {
  tasks,
  board,
  instrumentOfKind,
  instrumentForAddress,
  registeredInstruments,
  constructInstrument,
};
export type { RegisteredInstrument, RunningInstrument, WindowEnvironment, WindowServices, RenderPair };
