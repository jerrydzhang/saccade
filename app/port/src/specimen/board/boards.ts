// The specimen's boards: authored YAML, as a tracker would serve them.

export const DOCUMENT = `grammar: 1
task: t-2
content:
  - text: |
      # Numerically check the hand-estimated atlas walk of the covered-volume form

      **Brief, written by the t-0 conversation. Undertaking open. No result yet.**
      Keep this brief intact; write the account in the group at the bottom,
      and state any relaxation of the brief before the result.
  - text: |
      ## Why this is being asked

      The parent discussion (t-0@r9, first group) walked its behavioural
      atlas through a working form of a loss **by hand estimate only**.
      Those estimates are now being leaned on. Your job is to replace them
      with checked numbers, or to show where they are wrong. The method
      follows [the quadrature notes](https://example.org/quadrature).

      - **Worst case:** score = m log V(c max a), at c = 1.
      - **Per observation:** score = sum log V(c, a).

      \`\`\`sh
      $ uv run python -m atlas.walk --form covered-volume --c 1
      walked 14 regions, 3 disagree with the hand estimate
      \`\`\`

      A table that disagrees with the hand estimates is a fully successful
      result. Report the three disagreeing regions first, with \`atlas.walk\`
      at its default tolerance.
  - text: |
      ## The form to compute

      Region S (a box), observations $x_1, \\dots, x_m$, expression $f$,
      buffer $\\delta$ in the units of the data:

      $$V(c) = \\int_S \\mathbf{1}[\\, d(x, f) \\le c\\,\\delta \\,] \\, dx$$

      ### Refinement

      Compute volumes by deterministic quadrature fine enough that the
      reported digits are stable, and show that with a refinement check.
  - figure:
      input: figures/loop.svg
      caption: The walk, from read to accept.
  - annotation: |
      The buffer bound is a placeholder until t-6 lands; read the
      table's last column as provisional.
  - incorporation:
      source:
        task: t-24
        revision: 23
        select: [/content/0]
      content:
        - text: |
            The parent's worst case, restated: the sum over regions
            double-counted the buffer where two regions touch.
  - group:
      title: Account
      content:
        - text: |
            Fourteen regions walked at c = 1; three disagree with the
            hand estimate, all where two regions share a face.
        - incorporation:
            source:
              task: t-24
              revision: 23
              select: [/content/1]
            content:
              - text: |
                  The tolerance the walk ran at, as the parent set it.
        - text: |
            ## Open

            Whether the buffer maximum needs a certified optimizer, or
            dense sampling is enough, is t-6's to settle.
`;

export const CITED = `grammar: 1
task: t-24
content:
  - text: |
      The worst case at c = 1 is the only quantity the brief can ask for;
      the sum over regions counts the shared buffer twice.
  - text: |
      Tolerance: 1e-6 absolute, refined until three digits hold.
`;

const history = (worst: string, per: string, report: string, extra: string) => `grammar: 1
task: t-2
content:
  - text: |
      # Numerically check the hand-estimated atlas walk of the covered-volume form

      **Brief, written by the t-0 conversation. Undertaking open. No result yet.**
      Keep this brief intact; write the account in the group at the bottom,
      and state any relaxation of the brief before the result.
  - text: |
      ## Why this is being asked

      The parent discussion (t-0@r9, first group) walked its behavioural
      atlas through a working form of a loss **by hand estimate only**.
      Those estimates are now being leaned on. Your job is to replace them
      with checked numbers, or to show where they are wrong.
  - text: |
      - **Worst case:** score = m log V(c max a) ${worst}.
      - **Per observation:** score = sum log V(c, a) ${per}.
  - text: |
      \`\`\`sh
      $ uv run python -m atlas.walk --form covered-volume --c 1
      walked 14 regions, 3 disagree with the hand estimate
      \`\`\`
  - text: |
      A table that disagrees with the hand estimates is a fully successful
      result.${report}
  - text: |
      ## The form to compute

      Region S (a box), observations x1 ... xm, expression f, buffer d in
      the units of the data. Compute volumes by deterministic quadrature
      fine enough that the reported digits are stable, and show that with
      a refinement check.
${extra}  - text: |
      Compute the buffer maximum reliably (dense sampling of the ball or a
      bounded optimizer; say which). Estimator noise is not acceptable in
      a table meant to settle the hand estimates.
`;

export const HISTORY_BEFORE = history(
  "for each c",
  "summed over regions",
  "",
  `  - text: |
      Report each region's volume to three digits.
`,
);
export const HISTORY_AFTER = history("at c = 1", "per region", " Report the three disagreeing regions first.", "");

export const HISTORY_CHANGES = [
  { change: "changed", from: "/content/2", to: "/content/2" },
  { change: "changed", from: "/content/4", to: "/content/4" },
  { change: "removed", from: "/content/6" },
] as const;

export const HISTORY_MESSAGE =
  "Pin c = 1 in the worst-case form and ask for the disagreeing regions first; drop the per-region sum, which the brief never used. The parent discussion leans on the worst case at c = 1 only, and the sum over regions double-counted the buffer where two regions touch, so it was never a quantity the brief could ask for.";
