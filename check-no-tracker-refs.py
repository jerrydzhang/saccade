#!/usr/bin/env python3
"""Code comments carry no tracker references: the repository check.

Reads the staged diff (git diff --cached, hunks only) and refuses a
commit whose added or changed lines put a task, message, session or
revision number into a comment of a code file.

- Only the lines a commit adds or changes: the repository's existing
  references do not block, and a moved or renamed file adds no lines.
- Comments, not data: a line is a comment only when, stripped, it
  begins with the file's own comment marker - so a location string in
  a test, a pattern that matches task numbers, and a fixture board
  naming its task all pass by construction.
- Code, not prose: markdown is out of scope; the READMEs may still
  point at the tracker. Extensions the table below does not know are
  not scanned.
- The commit message is untouched: the trail from a line to its
  decision lives there.

Known gaps, deliberately kept and not assumed away:

- A reference in a trailing comment after code on the same line is
  not caught: whole-line detection is what keeps strings out.
- A block comment's interior is caught only at its opener and at
  continuation lines that start with the marker; an interior line
  opened by an unchanged line above it has no marker to see.
- An extension outside the table is not scanned.
- A number written so the patterns do not see it is not caught: a
  kind the rule does not name, or a value spelled around ("the
  seven-oh-one ruling").

The rule itself lives in AGENTS.md under "Layout and docs".
"""

import re
import subprocess
import sys

# A line is treated as a comment when, stripped, it begins with one of
# the file extension's markers. "*" covers a block comment's
# continuation lines in the languages that use /* */ blocks.
COMMENT_MARKERS = {
    ".ts": ("//",),
    ".js": ("//",),
    ".mjs": ("//",),
    ".cjs": ("//",),
    ".jsx": ("//",),
    ".tsx": ("//",),
    ".svelte": ("//", "/*", "*", "<!--", "-->"),
    ".rs": ("//", "/*", "*"),
    ".sh": ("#",),
    ".bash": ("#",),
    ".py": ("#",),
    ".toml": ("#",),
    ".yaml": ("#",),
    ".yml": ("#",),
    ".nix": ("#", "//", "/*", "*"),
    ".css": ("/*", "*"),
    ".html": ("<!--", "-->"),
    ".md": None,  # prose, out of scope by the rule itself
}

# The four kinds the rule names: a task, a message, a session, a
# revision. \b keeps compounds ("r2d2", "ps-2") out; @rK of a location
# matches through the same revision arm.
REFERENCE = re.compile(
    r"(?P<task>\bt-\d+\b)"
    r"|(?P<message>\bm-\d+\b)"
    r"|(?P<session>\bs-\d+\b)"
    r"|(?P<revision>\br\d+\b)"
)

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def staged_diff() -> str:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    run = subprocess.run(
        ["git", "-C", top, "diff", "--cached", "--no-color", "--unified=0"],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        print(f"no-tracker-refs: could not scan the staged diff: {run.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    return run.stdout


def scan() -> list[tuple[str, int, str, str]]:
    offenders: list[tuple[str, int, str, str]] = []
    path: str | None = None
    lineno = 0
    for line in staged_diff().split("\n"):
        if line.startswith("+++ b/"):
            path = line[6:].strip('"')
            continue
        if line.startswith("--- ") or line.startswith("index ") or line.startswith("diff --git "):
            continue
        hunk = HUNK.match(line)
        if hunk is not None:
            lineno = int(hunk.group(1))
            continue
        if not line.startswith("+") or path is None:
            continue
        at = lineno
        lineno += 1
        added = line[1:].strip()
        if added == "":
            continue
        extension = "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
        markers = COMMENT_MARKERS.get(extension)
        if not markers:  # unknown extension, or prose
            continue
        if not any(added.startswith(marker) for marker in markers):
            continue
        match = REFERENCE.search(added)
        if match is None:
            continue
        kind = next(name for name, hit in match.groupdict().items() if hit)
        offenders.append((path, at, added, kind))
    return offenders


def main() -> None:
    offenders = scan()
    if offenders == []:
        return
    print("no-tracker-refs: a code comment carries a tracker reference", file=sys.stderr)
    for path, lineno, added, kind in offenders:
        print(f"  {path}:{lineno}: {kind}: {added}", file=sys.stderr)
    print("  a comment states the rule itself, never where it was decided: no task,", file=sys.stderr)
    print("  message, session or revision numbers; data that names a task (a fixture", file=sys.stderr)
    print("  board, a sample location in a test) is not a reference; the trail from a", file=sys.stderr)
    print("  line to its decision is the commit message, which names its task", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
