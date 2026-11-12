// The act menu's placement model (order of work step 3, M3): the
// prototype's openAt rules (inst16.html:61-64) as pure math - at the
// pointer when a right-click carries coordinates, else beside the
// target (its left + 10px, 4px below); flipped ABOVE the target when
// the menu would not fit below; clamped to the viewport with a 4px
// margin. Testable in node; the views apply the result.

export interface PlacementPoint {
  readonly x: number;
  readonly y: number;
}

export interface PlacementBox {
  readonly left: number;
  readonly top: number;
  readonly bottom: number;
}

export interface PlacementViewport {
  readonly width: number;
  readonly height: number;
}

export interface MenuPlacementInput {
  /** The right-click point, if any (viewport coordinates). */
  readonly at?: PlacementPoint | null;
  /** The target's box (viewport coordinates); absent when the menu
   * opened without one. */
  readonly target?: PlacementBox | null;
  /** The menu's own measured size. */
  readonly menu: Readonly<{ width: number; height: number }>;
  readonly viewport: PlacementViewport;
}

export function menuPlacement(input: MenuPlacementInput): PlacementPoint {
  const { menu, viewport } = input;
  const margin = 4;
  const gap = 4;
  let left: number;
  let top: number;
  if (input.at !== null && input.at !== undefined) {
    left = input.at.x;
    top = input.at.y;
  } else if (input.target !== null && input.target !== undefined) {
    const target = input.target;
    left = target.left + 10;
    // beside-below when it fits, flipped above when it does not
    const below = target.bottom + gap;
    top = below + menu.height + margin <= viewport.height ? below : target.top - gap - menu.height;
  } else {
    // no anchor: the viewport's top-left corner, clamped
    left = margin;
    top = margin;
  }
  // clamp to the viewport (inst16's Math.max/min dance, 64)
  left = Math.max(margin, Math.min(left, viewport.width - menu.width - margin));
  top = Math.max(margin, Math.min(top, viewport.height - menu.height - margin));
  return { x: left, y: top };
}
