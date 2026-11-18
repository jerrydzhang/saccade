// The frame's half of the instrument contract: the presentation
// of the buffer's published targets. An instrument publishes its
// targets with its data; this module presents them with no code in the
// instrument - select mode over the targets in view (the geometry
// bounds decide in-view; nested labels per the target's `inside`), the
// act menu built from the acts list in order, the click paths (plain
// click = the first act; a modified click = the act its key names;
// right-click = the act list at the pointer), the frame's reading
// context (the shared keys, defined once in lib/input/contexts.ts),
// and the anchors - the reading position's, for Back, and the
// selection's, which follows a target across a refresh by its durable
// anchor and clears when no published target carries it any more.
//
// Pure logic over injected ports: no DOM, no Svelte, no transport. The
// targets port reads the buffer's CURRENT publication at ask time, so
// the sessions here outlive refreshes the way an instrument's own
// sessions did (the task list's allTargets() read the shown page and
// the live rail alike); a lease resolves its target when the gesture
// presents it, and an element the gesture cannot measure means the
// target has left - the lease acts nothing, never a silently wrong
// act. The frame surface component renders the views this module
// notifies.

import type { Bounds, Point, ReadingAnchor, TargetDefinition, TargetPresentations, Unsubscribe, Viewport } from "../contracts/buffer.ts";
import type { CommandBinding, InputContext } from "../contracts/input.ts";
import type { AnchorLocation } from "../contracts/places.ts";
import { frameReadingContext } from "../input/contexts.ts";
import { createSelectSession, type SelectableTarget, type SelectSessionView } from "../select/session.ts";
import { createActMenu, type ActMenu } from "../menu/create.ts";
import type { MenuAct, MenuSession, MenuTarget, MenuView } from "../menu/session.ts";

/** The ports the frame's presentation reads, all injected: the
 * published targets (the buffer's current publication, read at ask
 * time), the geometry service's content-coordinate bounds, the input
 * capture funnel, and the viewport the reading keys serve. */
export interface TargetPresentationPorts {
  targets(): readonly TargetDefinition[];
  bounds(id: string): Bounds | null;
  capture(context: InputContext): { release(): void };
  viewport: Viewport;
}

/** The key a modified click runs (the built mapping in
 * boardController/hydrate: shift and middle are the `o` act); a kind
 * that declares no `o` act acts nothing - never a hidden default. */
const MODIFIED_CLICK_KEY = "o";

export function createTargetPresentations(ports: TargetPresentationPorts): TargetPresentations {
  const observers = new Set<() => void>();
  function notify(): void {
    for (const observer of [...observers]) observer();
  }

  const targetOf = (id: string): TargetDefinition | undefined =>
    ports.targets().find((entry) => entry.id === id);

  const menuTargetOf = (def: TargetDefinition): MenuTarget => ({
    part: def.id,
    kind: def.kind,
    label: def.label,
  });

  // The menu's acts are the published acts, resolved when the menu
  // asks - the same array the lease and Enter run, in the published
  // order (MenuAct is the contract's TargetAct).
  const actsOf = (target: MenuTarget): readonly MenuAct[] =>
    targetOf(target.part ?? "")?.acts() ?? [];

  const menu: ActMenu = createActMenu(actsOf, { capture: ports.capture });
  menu.setOnNotify(notify);

  // The selection's anchor: what the readout's selection fact and a
  // refresh follow. A target without an anchor (a rail entry) selects
  // without recording one.
  let selected: string | null = null;

  const select = createSelectSession({
    // the select machinery's TargetKind is the app's target kinds; the
    // published kind names its own (the same narrowing the registry's
    // reviewed cast makes at one boundary)
    targets: () =>
      ports.targets().map((def): SelectableTarget => ({
        part: def.id,
        kind: def.kind as SelectableTarget["kind"],
        text: def.label,
        within: def.inside ?? undefined,
      })),
    visible(target) {
      // the in-view idiom: content-coordinate bounds against the
      // viewport's window, the check both instruments kept by hand
      const metrics = ports.viewport.metrics();
      const viewTop = metrics.position.top;
      const viewBottom = viewTop + metrics.height;
      const box = ports.bounds(target.part);
      if (box === null) return false;
      return box.top < viewBottom && box.bottom > viewTop;
    },
    capture: ports.capture,
    onSelected(target) {
      selected = targetOf(target.part)?.anchor ?? null;
      // scenario 9: the selection opens the act menu
      const def = targetOf(target.part);
      if (def === undefined) return;
      menu.openFor(menuTargetOf(def));
    },
  });
  select.subscribe(() => notify());

  function menuView(): MenuView | null {
    if (!menu.session.isOpen()) return null;
    const target = menu.session.target();
    if (target === null) return null;
    return {
      open: true,
      label: target.label,
      acts: menu.session.acts().map((act) => ({ key: act.key, label: act.action.label })),
      at: menu.session.at(),
      part: target.part,
    };
  }

  return {
    presentTarget(id, measure) {
      const def = targetOf(id) ?? null;
      let released = false;
      // The gesture's own measurement, taken with the lease: an
      // element it cannot measure has left the surface, and a stale
      // lease acts nothing.
      const alive = (): TargetDefinition | null => {
        if (released || def === null) return null;
        return measure() === null ? null : def;
      };
      return {
        activatePrimary(): void {
          const current = alive();
          if (current === null) return;
          // the first act is the primary: what a plain click and Enter
          // run
          current.acts()[0]?.action.run();
        },
        activatePointer(kind) {
          const current = alive();
          if (current === null) return false;
          const target = menuTargetOf(current);
          const outcome =
            kind === "plain"
              ? menu.session.primaryClick(target)
              : menu.session.keyClick(target, MODIFIED_CLICK_KEY);
          return outcome.kind === "act";
        },
        showActions(at: Point): void {
          const current = alive();
          if (current === null) return;
          menu.openFor(menuTargetOf(current), { x: at.left, y: at.top });
        },
        release(): void {
          released = true;
        },
      };
    },

    selectView(): SelectSessionView | null {
      return select.view();
    },

    menuView,

    menuSession(): MenuSession {
      return menu.session;
    },

    subscribe(observer: () => void): Unsubscribe {
      observers.add(observer);
      observer();
      return () => {
        observers.delete(observer);
      };
    },

    readingContext(extra: readonly CommandBinding[] = []): InputContext {
      return frameReadingContext(
        {
          scrollDown: () => ports.viewport.scrollLines(1),
          scrollUp: () => ports.viewport.scrollLines(-1),
          scrollToTop: () => ports.viewport.start(),
          scrollToBottom: () => ports.viewport.end(),
          scrollFraction: (direction, fraction) => ports.viewport.scrollFraction(direction, fraction),
          enterSelect: () => select.enter(),
        },
        extra,
      );
    },

    readingAnchor(): ReadingAnchor {
      // the reading-position idiom: the published target whose bounds
      // hold the viewport's top edge, named by its durable anchor (a
      // row's task); targets without an anchor cannot carry a return
      // place and are skipped, never guessed
      const viewTop = ports.viewport.metrics().position.top;
      for (const def of ports.targets()) {
        if (def.anchor === null) continue;
        const box = ports.bounds(def.id);
        if (box === null) continue;
        if (viewTop >= box.top && viewTop < box.bottom) {
          return { kind: "anchor", anchor: def.anchor, bounds: box };
        }
      }
      return { kind: "default" };
    },

    locateAnchor(anchor: string): AnchorLocation {
      const def = ports.targets().find((entry) => entry.anchor === anchor);
      if (def === undefined) return { kind: "missing" };
      const box = ports.bounds(def.id);
      if (box === null) return { kind: "pending" };
      return { kind: "found", bounds: box };
    },

    selectedAnchor(): string | null {
      // the selection follows its anchor across a refresh: it stands
      // while a published target still carries it, and clears - never
      // a silent neighbour - when none does (scenario 12's identity
      // rule, the frame's own now)
      if (selected === null) return null;
      return ports.targets().some((entry) => entry.anchor === selected) ? selected : null;
    },
  };
}
