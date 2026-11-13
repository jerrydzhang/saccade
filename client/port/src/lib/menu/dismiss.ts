// The act menu's page-side dismissal: the pointer and wheel events
// that close an open menu without acting. The session owns what a
// close means (session.ts); this binds the page to it, once, for
// every instrument that paints a menu.
//
// - A mouse-down outside the menu closes it (the prototype's :517).
//   A right-click elsewhere is a mouse-down first, so it closes this
//   menu before its contextmenu opens the next one. mousedown, not
//   pointerdown: in the native app (WebKitGTK) a left click's
//   pointerdown never reached a window listener in the native lane,
//   while browser WebKit delivered it; mousedown reaches both.
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
  const onMouseDown = (event: MouseEvent): void => {
    if (outside(event)) session.outsidePointerDown();
  };
  const onWheel = (event: WheelEvent): void => {
    if (outside(event)) session.scrolledAway();
  };
  view.addEventListener("mousedown", onMouseDown, true);
  view.addEventListener("wheel", onWheel, { capture: true, passive: true });
  return () => {
    view.removeEventListener("mousedown", onMouseDown, true);
    view.removeEventListener("wheel", onWheel, { capture: true });
  };
}
