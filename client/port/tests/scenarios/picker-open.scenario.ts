// The picker opens a location: space space, the typed spec, Enter;
// the shown board answers the location that was asked for
// (t36_port.open_via_port_picker).
import { expect, test } from "#scenario";
import { openLocation, spec } from "./kit/helpers.ts";

test("the picker opens a typed location", { tags: ["native"] }, async ({ app, tracker }) => {
  const obs = await openLocation(app, spec(tracker, tracker.tasks.cited));
  expect(obs.surface).toBe("board");
  expect(obs.board).toMatchObject({
    status: "loaded",
    asked: `${tracker.repo}/${tracker.tasks.cited}`,
    location: `${tracker.repo}/${tracker.tasks.cited}@r1`,
    task: tracker.tasks.cited,
  });
  expect(obs.content?.parts.map((part) => part.kind)).toContain("group");
});
