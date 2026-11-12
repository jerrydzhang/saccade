// The board's menu wiring: the per-kind act declarations (the staged
// KINDS shape) and the board's menu host - selection opens, keys
// route through the shared matcher, close-before-execute comes from
// the session module. Pure: the controller supplies fold state, the
// model's citation sources, and the window services the reference
// acts ride.
//
// The reference acts (t-36@r90, the prototype's KINDS at inst16.html
// :36-46 as the contract): a citation's first act is 'f' "follow
// here" - so a plain click (primary, act 0) follows here - then 'y'
// "copy reference". The copy payload is the prototype-parity
// encoding: the bare t-N@rK, with the canonical part path appended
// (:partPath) when the citation names a part. The OPEN o/i/w acts
// are the elsewhere slice, not this one.

import type { Action, InputContext, KeyInput, KeyResolution } from "../input/input-context.ts";
import { selectToPart } from "../board/targets.ts";
import { formatLocation } from "../location.ts";
import { createActMenu } from "./create.ts";
import type { MenuAct, MenuSession, MenuTarget } from "./session.ts";
import type { ElsewhereAct } from "../elsewhere.ts";

/** A model-authored citation source (the incorporation block's). */
export type ReferenceSource = Readonly<{
  task: string;
  revision: number;
  select: readonly string[] | null;
}>;

/** The window services the reference acts ride: opening goes through
 * the window's ONE open funnel (trail + anchored restore), copying
 * reports its payload and attempts the real clipboard. The window
 * injects them; until then no reference acts are offered. */
export type ReferencePorts = Readonly<{
  open(spec: string, part: string | null): void;
  copy(payload: string): void;
  /** The elsewhere acts (o/i/w, S18): present only when the window
   * has bound its window-store state - the acts appear with it. */
  elsewhere?: (spec: string, act: ElsewhereAct) => void;
  /** The figure's enlarge act (S04): opens the window's inspection
   * surface for the part. */
  figureEnlarge?: (part: string) => void;
}>;

export interface BoardMenuDeps {
  /** Whether a group part path is folded now. */
  isFolded(part: string): boolean;
  /** Fold or unfold a group part path. */
  toggleFold(part: string): void;
  /** The citation source of an incorporation part (model-derived), or
   * null when the part names no citation. */
  referenceOf(part: string): ReferenceSource | null;
  /** The window's reference services, or null before injection. */
  referencePorts(): ReferencePorts | null;
  /** The machine the shown frame resolved at: the follow act composes
   * the open spec against it (formatLocation), never a colon join. */
  referenceMachine(): string | null;
  /** A part's own citation payload (the SHOWN task@revision plus the
   * part's full path), or null when the shown frame carries no
   * pinned revision to cite. */
  partCitation(part: string): string | null;
  /** Capture input while open. Lease release returns to the CURRENT
   * base; a menu never reinstalls its own saved reading context. */
  capture(context: InputContext): { release(): void };
}

export interface BoardMenu {
  session: MenuSession;
  /** The context installed while the menu is open (null closed). */
  context(): InputContext | null;
  /** Selection (or a right-click) opens the menu when the target has
   * acts; a no-op for act-less kinds (plain blocks). */
  openFor(target: Readonly<{ part: string | null; kind: string; label: string }>, at?: { x: number; y: number }): void;
  /** The view-change notification (open, act, close). */
  setOnNotify(callback: () => void): void;
}

export function createBoardMenu(deps: BoardMenuDeps): BoardMenu {
  return createActMenu(
    (target) => actsFor(target, deps),
    { capture: (context) => deps.capture(context) },
  );
}

/** The per-kind act declarations. Groups fold/unfold (chunk C); a
 * citation's reference target follows here first (so a plain click
 * follows) and copies its payload (t-36@r90). Every other kind
 * carries no acts yet. */
export function actsFor(
  target: MenuTarget,
  deps: Pick<
    BoardMenuDeps,
    "isFolded" | "toggleFold" | "referenceOf" | "referencePorts" | "referenceMachine" | "partCitation"
  >,
): readonly MenuAct[] {
  if (target.kind === "group" && target.part !== null) {
    const folded = deps.isFolded(target.part);
    const acts: MenuAct[] = [
      {
        key: "z",
        action: {
          id: "board.fold",
          label: folded ? "unfold" : "fold",
          run: () => deps.toggleFold(target.part as string),
        },
      },
    ];
    // the group's own citation (the prototype's group kind: z Fold, y
    // Copy citation): the SHOWN revision plus the group's full path
    const citation = deps.partCitation(target.part);
    const ports = deps.referencePorts();
    if (citation !== null && ports !== null) {
      acts.push({
        key: "y",
        action: {
          id: "board.copy-citation",
          label: "copy citation",
          run: () => ports.copy(citation),
        },
      });
    }
    return acts;
  }
  if (target.kind === "figure" && target.part !== null) {
    // the prototype's figure kind (inst16 :47): 'x' Enlarge FIRST -
    // a plain click enlarges - then 'a' Annotate and the OPEN/copy
    // acts. 'a' waits for the drafts surface (the port has none yet -
    // recorded, never silently dropped); 'o' opens through a link the
    // port's board figures do not carry yet (offered when they do)
    const ports = deps.referencePorts();
    if (ports === null) return [];
    const citation = deps.partCitation(target.part);
    const acts: MenuAct[] = [
      {
        key: "x",
        action: {
          id: "figure.enlarge",
          label: "enlarge",
          run: () => ports.figureEnlarge?.(target.part ?? ""),
        },
      },
    ];
    if (citation !== null) {
      acts.push({
        key: "y",
        action: {
          id: "figure.copy-citation",
          label: "copy citation",
          run: () => ports.copy(citation),
        },
      });
    }
    return acts;
  }
  if (target.kind === "text" && target.part !== null) {
    // F5 + B10 (ruled): every part's menu OPENS - a text block's
    // act is y 'copy citation' (the shown revision plus the part's
    // full canonical path). This is the block's PRIMARY while
    // annotate (a) and history (t) are unbuilt: Enter or a plain
    // click on a paragraph copies, never annotates. The ruled
    // difference from the prototype's a-first order is recorded.
    const citation = deps.partCitation(target.part);
    const ports = deps.referencePorts();
    if (citation === null || ports === null) return [];
    return [
      {
        key: "y",
        action: {
          id: "block.copy-citation",
          label: "copy citation",
          run: () => ports.copy(citation),
        },
      },
    ];
  }
  if (target.kind === "reference" && target.part !== null) {
    const source = deps.referenceOf(target.part);
    const ports = deps.referencePorts();
    const machine = deps.referenceMachine();
    if (source === null || ports === null || machine === null) return [];
    // the follow composes the parseable location form (a tracker
    // machine renders name/spec - never a colon join, 5a44c71), and a
    // citation naming a part anchors the open there (selectToPart of
    // the authored grammar select)
    const path = `${source.task}@r${source.revision}`;
    const select = source.select && source.select.length > 0 ? source.select.join("/") : null;
    const anchorPart = select === null ? null : selectToPart(select);
    const spec = formatLocation({ machine, path });
    const copyPayload = anchorPart === null ? path : `${path}:${anchorPart}`;
    return [
      {
        key: "f",
        action: {
          id: "reference.follow",
          label: "follow here",
          run: () => ports.open(spec, anchorPart === "" ? null : anchorPart),
        },
      },
      // the OPEN acts (the r90 KINDS order: follow-here first, then
      // open elsewhere, then copy): o delivers at the link, i asks
      // once without touching the link, w opens a new viewer
      ...(ports.elsewhere
        ? ([
            {
              key: "o",
              action: {
                id: "reference.open-at-link",
                label: "open at the link",
                run: () => ports.elsewhere?.(spec, "link"),
              },
            },
            {
              key: "i",
              action: {
                id: "reference.open-once",
                label: "open elsewhere once",
                run: () => ports.elsewhere?.(spec, "once"),
              },
            },
            {
              key: "w",
              action: {
                id: "reference.open-new",
                label: "open in a new viewer",
                run: () => ports.elsewhere?.(spec, "new"),
              },
            },
          ] as const)
        : []),
      {
        key: "y",
        action: {
          id: "reference.copy",
          label: "copy reference",
          run: () => ports.copy(copyPayload),
        },
      },
    ];
  }
  return [];
}
