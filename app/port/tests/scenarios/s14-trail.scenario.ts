// S14: the trail records explicit openings only
// (t36_differential._s14_port). Two picker openings land two trail
// entries with the cursor on the newest; a fold through the act menu
// leaves the trail untouched.
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

test("S14 the trail records explicit openings only", async ({ app, tracker }) => {
  const { citing, landing } = tracker.tasks;
  await openLocation(app, spec(tracker, citing));
  const obs = await openLocation(app, spec(tracker, landing));

  const trail = obs.trail;
  expect(trail.depth).toBeGreaterThanOrEqual(2);
  expect(trail.cursor).toBe(trail.depth - 1);
  const locations = trail.entries.map((entry) => entry.location).join(" ");
  expect(locations).toContain(`/${citing}`);
  expect(locations).toContain(`/${landing}`);
  expect(trail).toMatchObject({ canBack: true, canForward: false });

  // the fold act must not record an opening
  const group = obs.content?.parts.find((part) => part.kind === "group");
  expect(group, "the landing board carries a group to fold").toBeDefined();
  await app.clickPart(group!.part, { button: "right", at: "header" });
  await expect.poll(() => app.menuOpen()).toBe(true);
  await app.press("z");
  await expect.poll(() => app.menuOpen()).toBe(false);
  expect((await app.readout()).trail).toEqual(trail);
});

// t-43: Back restores what was asked. A board opened bare follows; left
// and returned to with Back, it comes back bare (never the revision it
// resolved to), shows the revision committed meanwhile, keeps
// following, and its window place stays bare.
test("S14 Back onto a board opened bare lands it bare and following", async ({ app, tracker }) => {
  const board = (task: string, text: string) => `grammar: 1\ntask: ${task}\ncontent:\n  - text: ${text}\n`;
  const task = await tracker.createTask("t-43: back onto a bare board");
  await tracker.commit(task, board(task, "the first revision"), "the first revision");
  const bare = `${tracker.repo}/${task}`;

  const opened = await openLocation(app, spec(tracker, task));
  expect(opened.board).toMatchObject({ asked: bare, location: `${bare}@r1` });
  await openLocation(app, spec(tracker, tracker.tasks.landing));
  await tracker.commit(task, board(task, "the second revision"), "the second revision");

  await app.press("Ctrl-o");
  await expect
    .poll(async () => {
      const obs = await app.readout();
      return { asked: obs.board.asked, location: obs.board.location, status: obs.board.status };
    })
    .toEqual({ asked: bare, location: `${bare}@r2`, status: "loaded" });
  const back = await app.readout();
  expect(back.follow).toMatchObject({ armed: true, address: bare });
  expect(back.window.place?.location ?? bare).toBe(bare);

  // and it keeps following: a later commit shows without any open
  await tracker.commit(task, board(task, "the third revision"), "the third revision");
  await expect
    .poll(async () => (await app.readout()).board.location, { timeout: 20_000 })
    .toBe(`${bare}@r3`);
});
