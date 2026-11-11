// The open-location picker controller: the window's open surface.
// Text-entry semantics per scenario 10: typing edits the field
// natively (the window key seam skips field targets), Enter submits
// through the ONE open funnel (openSpec - the same path every control
// uses), Escape closes. Submission of an unparsable spec leaves the
// refusal to the open funnel's own diagnostics; the picker closes
// either way - the strip carries the outcome.

export interface PickerState {
  readonly open: boolean;
}

export interface PickerController {
  open(): void;
  close(): void;
  submit(spec: string): void;
  snapshot(): PickerState;
  subscribe(observer: (state: PickerState) => void): () => void;
}

export function createPickerController(openSpec: (spec: string) => void): PickerController {
  let state: PickerState = { open: false };
  const observers = new Set<(state: PickerState) => void>();

  function publish(next: PickerState): void {
    state = next;
    for (const observer of [...observers]) observer(state);
  }

  return {
    open() {
      publish({ open: true });
    },
    close() {
      publish({ open: false });
    },
    submit(spec: string) {
      const trimmed = spec.trim();
      publish({ open: false });
      if (trimmed.length > 0) openSpec(trimmed);
    },
    snapshot() {
      return state;
    },
    subscribe(observer) {
      observers.add(observer);
      observer(state);
      return () => {
        observers.delete(observer);
      };
    },
  };
}
