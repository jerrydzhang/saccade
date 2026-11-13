// The act menu's page-side dismissal: the pointer and wheel events
// that close an open menu without acting. The session owns what a
// close means (session.ts); this binds the page to it, once, for
// every instrument that paints a menu.
//
// - A pointer-down outside the menu closes it (the prototype's :517).
//   A right-click elsewhere is a pointer-down first, so it closes this
//   menu before its contextmenu opens the next one.
// - A wheel outside the menu closes it: the menu is anchored to a
//   point on the page, and scrolling would carry it away from what it
//   names. Only the wheel counts - programmatic scrolls (a reveal, a
//   follow refresh restoring the reading place) never dismiss.

export interface MenuDismissal {
  outsidePointerDown(): void;
  scrolledAway(): void;
}

/** Bind dismissal while `menu` is painted; returns the unbind. */
export function bindMenuDismissal(menu: HTMLElement, session: MenuDismissal): () => void {
  const view = menu.ownerDocument.defaultView;
  if (view === null) return () => {};
  const outside = (event: Event): boolean =>
    !(event.target instanceof Node && menu.contains(event.target));
  const onPointerDown = (event: PointerEvent): void => {
    if (outside(event)) session.outsidePointerDown();
  };
  const onWheel = (event: WheelEvent): void => {
    if (outside(event)) session.scrolledAway();
  };
  view.addEventListener("pointerdown", onPointerDown, true);
  view.addEventListener("wheel", onWheel, { capture: true, passive: true });
  return () => {
    view.removeEventListener("pointerdown", onPointerDown, true);
    view.removeEventListener("wheel", onWheel, { capture: true });
  };
}
