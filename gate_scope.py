#!/usr/bin/env python3
"""Which project suites a push runs: the merge gate's scope rule.

usage: gate_scope.py <from-sha> <to-sha>

Prints the justfile recipes to run, one per line, in order:
saccade::test, app::test, or both.

- A path under app/ runs the app's suites.
- A path under saccade/ runs both: the app depends on saccade (its
  grammar crate, and the sac its suites serve).
- Any other path (a root file: the flake, the justfiles' root, the
  hooks, the law) runs both.
- No changed path runs neither: nothing the suites read moved.
- A push whose old side is absent (a new branch) or unreadable runs
  both: the rule never narrows on a diff it could not compute.
"""

import subprocess
import sys

SACCADE = "saccade::test"
APP = "app::test"
# a push's old side when the remote had no such ref
ABSENT = "0" * 40


def suites(paths: list[str]) -> list[str]:
    """The recipes the changed paths call for, saccade's first."""
    wanted: set[str] = set()
    for path in paths:
        if path.startswith("app/"):
            wanted.add(APP)
        else:
            wanted.update((SACCADE, APP))
    return [recipe for recipe in (SACCADE, APP) if recipe in wanted]


def changed_paths(old: str, new: str) -> list[str] | None:
    """The paths between the two commits, or None when the diff cannot
    be read (no old side, or a commit this checkout does not have)."""
    if old in ("", ABSENT) or new in ("", ABSENT):
        return None
    run = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{old}..{new}"],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        return None
    return [line for line in run.stdout.split("\n") if line != ""]


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: gate_scope.py <from-sha> <to-sha>", file=sys.stderr)
        sys.exit(2)
    paths = changed_paths(sys.argv[1], sys.argv[2])
    recipes = [SACCADE, APP] if paths is None else suites(paths)
    for recipe in recipes:
        print(recipe)


if __name__ == "__main__":
    main()
