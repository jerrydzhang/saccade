// The browser project's pointer commands (vitest browser commands, run
// in node beside Playwright): a real mouse drag and a real wheel notch
// on a port control, through Playwright's page.mouse. Needed because
// vitest's userEvent has no raw drag-by-offset or wheel; the control's
// box comes from the app's iframe, so the coordinates are the page's.
import type { BrowserCommand } from "vitest/node";

type Context = Parameters<BrowserCommand<[]>>[0] & {
  page: { mouse: { move(x: number, y: number, o?: { steps?: number }): Promise<void>; down(): Promise<void>; up(): Promise<void>; wheel(dx: number, dy: number): Promise<void> } };
  frame(): Promise<{ frameLocator(selector: string): { locator(selector: string): { boundingBox(): Promise<{ x: number; y: number; width: number; height: number } | null> } } }>;
};

async function centerOf(context: Context, control: string): Promise<{ x: number; y: number }> {
  const frame = await context.frame();
  const box = await frame
    .frameLocator('iframe[data-port-suite="app"]')
    .locator(`[data-port-control=${JSON.stringify(control)}]`)
    .boundingBox();
  if (box === null) throw new Error(`the control ${control} is not rendered`);
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

export const portDrag: BrowserCommand<[control: string, dx: number, dy: number]> = async (context, control, dx, dy) => {
  const ctx = context as unknown as Context;
  const start = await centerOf(ctx, control);
  await ctx.page.mouse.move(start.x, start.y);
  await ctx.page.mouse.down();
  await ctx.page.mouse.move(start.x + dx, start.y + dy, { steps: 5 });
  await ctx.page.mouse.up();
};

export const portWheel: BrowserCommand<[control: string, deltaY: number]> = async (context, control, deltaY) => {
  const ctx = context as unknown as Context;
  const start = await centerOf(ctx, control);
  await ctx.page.mouse.move(start.x, start.y);
  await ctx.page.mouse.wheel(0, deltaY);
};
