// Context menus (t-14 "Interaction"): typing a label selects a target
// and opens its one context menu beside it, each act showing its key;
// right-click opens the same menu at the pointer; every act is
// reachable by the pointer too. One floating panel, ruled at the edge
// (the design language: a menu is a structure that needs an edge), no
// chrome beyond the acts.

export type MenuAct = {
  /** The key that fires the act while the menu is open. */
  key: string;
  label: string;
  onAct: () => void;
};

export type OpenMenu = {
  x: number;
  y: number;
  acts: MenuAct[];
  onClose: () => void;
  /** A heading line above the acts (the prototype's menu head). */
  title?: string;
  /** Enter fires the first act (the prototype's select menus). */
  enterFirst?: boolean;
  /** Any other key closes the menu (the prototype's which-key: space
   * then a key that is no command does nothing and closes). */
  closeOnOtherKey?: boolean;
  /** A data-control id for the panel, for the lanes. */
  control?: string;
};

/** How a key shows in a menu: the space key reads "spc". */
export function keyLabel(key: string): string {
  return key === " " ? "spc" : key;
}

/** Open one menu at the pointer (or beside a selection). The panel
 * closes on Escape, on any click outside, or on an act; the caller's
 * onClose runs on every close so the selection state follows. */
export function openContextMenu(open: OpenMenu): () => void {
  const panel = document.createElement("div");
  panel.className = "ctx-menu";
  panel.setAttribute("role", "menu");
  if (open.control) panel.dataset.control = open.control;
  if (open.title) {
    const head = document.createElement("div");
    head.className = "ctx-menu-head";
    head.textContent = open.title;
    panel.append(head);
  }
  const handlers: Array<[HTMLElement, () => void]> = [];
  for (const act of open.acts) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "ctx-menu-act";
    row.setAttribute("role", "menuitem");
    const key = document.createElement("span");
    key.className = "ctx-menu-key";
    key.textContent = keyLabel(act.key);
    const label = document.createElement("span");
    label.className = "ctx-menu-label";
    label.textContent = act.label;
    row.append(key, label);
    const fire = () => {
      close();
      act.onAct();
    };
    row.addEventListener("click", fire);
    handlers.push([row, fire]);
    panel.append(row);
  }
  document.body.append(panel);

  // keep the panel on screen: flip above/left when it would overflow
  const rect = panel.getBoundingClientRect();
  const maxX = window.innerWidth - rect.width - 8;
  const maxY = window.innerHeight - rect.height - 8;
  panel.style.left = `${Math.max(8, Math.min(open.x, maxX))}px`;
  panel.style.top = `${Math.max(8, Math.min(open.y, maxY))}px`;

  const onKey = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
      return;
    }
    const act =
      open.acts.find((candidate) => candidate.key === event.key) ??
      (open.enterFirst && event.key === "Enter" ? open.acts[0] : undefined);
    if (act) {
      event.preventDefault();
      event.stopPropagation();
      close();
      act.onAct();
      return;
    }
    // a lone modifier is not a key choice
    if (open.closeOnOtherKey && !["Shift", "Control", "Alt", "Meta"].includes(event.key)) {
      event.preventDefault();
      event.stopPropagation();
      close();
    }
  };
  const onOutside = (event: MouseEvent) => {
    if (!panel.contains(event.target as Node)) close();
  };
  window.addEventListener("keydown", onKey, true);
  window.addEventListener("mousedown", onOutside, true);

  function close(): void {
    window.removeEventListener("keydown", onKey, true);
    window.removeEventListener("mousedown", onOutside, true);
    panel.remove();
    open.onClose();
  }

  return close;
}
