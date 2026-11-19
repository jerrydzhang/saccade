// The control plane's add field: registering a project is its
// own act, separate from opening anything. `a` (the plane's reading
// context, and a project's act menu) opens a small inline field; the
// human types machine:/path/to/repository, Enter registers the
// repository whose declaration file names the project (the registry
// reads the file through the one core module) and the rail lists it -
// nothing opens, the shown project stays. Escape leaves the field.
//
// The shape is Find's (lib/find/interaction.ts): a pure session plus
// a text-entry context of declared field commands, captured through
// the instrument's own lease while the field is open, so typing stays
// native and the lease returns keys to the CURRENT base.

import { key, sequence, textEntryContext } from "../contracts/input.ts";
import type { InputContext } from "../contracts/input.ts";
import { parseRegisteredPath, type RegisteredPath } from "../projects.ts";

export type AddView = Readonly<{
  open: boolean;
  /** What the field holds. */
  value: string;
  /** Why the last submit registered nothing; null otherwise. */
  problem: string | null;
}>;

export const ADD_CLOSED: AddView = { open: false, value: "", problem: null };

export interface AddProjectDeps {
  capture(context: InputContext): { release(): void };
  /** Register the repository at the path (it joins the rail).
   * Refusals arrive as the field's problem; resolution reads the
   * declaration through the module, so this is async. */
  register(registered: RegisteredPath): Promise<void>;
}

export function createAddProject(deps: AddProjectDeps) {
  let view: AddView = ADD_CLOSED;
  let lease: { release(): void } | null = null;
  const observers = new Set<(view: AddView) => void>();
  const notify = (): void => {
    for (const observer of [...observers]) observer(view);
  };

  function close(): void {
    if (!view.open) return;
    view = ADD_CLOSED;
    lease?.release();
    lease = null;
    notify();
  }

  let submitting = false;

  async function submit(): Promise<void> {
    if (!view.open || submitting) return;
    const registered = parseRegisteredPath(view.value);
    if (registered === null) {
      view = {
        ...view,
        problem: `'${view.value.trim()}' is not a repository path; type machine:/path/to/repository`,
      };
      notify();
      return;
    }
    submitting = true;
    try {
      await deps.register(registered);
    } catch (error) {
      submitting = false;
      view = { ...view, problem: error instanceof Error ? error.message : String(error) };
      notify();
      return;
    }
    submitting = false;
    close();
  }

  const context = textEntryContext(() => [
    { action: { id: "tasks.add.submit", label: "add the project", run: () => void submit() }, keys: [sequence(key("Enter"))] },
    { action: { id: "tasks.add.close", label: "leave the field", run: close }, keys: [sequence(key("Escape"))] },
  ]);

  return {
    view: (): AddView => view,
    enter(): void {
      if (view.open) return;
      view = { open: true, value: "", problem: null };
      lease = deps.capture(context);
      notify();
    },
    type(value: string): void {
      if (!view.open) return;
      view = { open: true, value, problem: null };
      notify();
    },
    submit,
    close,
    subscribe(observer: (view: AddView) => void): () => void {
      observers.add(observer);
      observer(view);
      return () => {
        observers.delete(observer);
      };
    },
  };
}

export type AddProject = ReturnType<typeof createAddProject>;
