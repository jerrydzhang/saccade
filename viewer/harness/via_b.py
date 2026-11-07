"""The remote mesh lane's via program: what core A runs when it dials
core B. machine.add --via hands the program one shell line ("a shell
line to run there as its last argument, like ssh"). The observed
shapes from the run budget: the dial-time probe "echo ok", and a
sh -c line that locates herdr-core on PATH, sets HC_SESSION=b and
execs `connect`.

The program is therefore the sandbox's faithful remote shell: it runs
the line whole under the pinned /bin/sh - which the lane binds - with
the pinned core's bin directory first on PATH, so the line's
`command -v herdr-core` resolves to exactly the tested bytes. Nothing
is second-guessed and no shell is spawned by this program: the line is
the contract, and the sandbox is the containment.

Runs inside the sandbox as
[<BOARD_VIEWER_VIA_PY>, /pin/via_b.py, <line words...>]. The decision
lives in shell_argv so the contract runs against fakes in the fast
check loop.
"""

from __future__ import annotations

import os
import sys


class ViaRefused(Exception):
    """The handover is not a shell line at all; A surfaces the
    refusal, and the message quotes what arrived."""


def line_from_argv(words: list[str]) -> str:
    """The composed line, however the core hands it over: the via
    command's own words are dropped, the rest is the line - joined
    when the core appends it word by word, intact when it appends one
    quoted argument."""
    return " ".join(words[1:])


def shell_argv(line: str, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """(argv, child env) for the line: the pinned /bin/sh -c, with the
    pinned core's bin directory leading PATH so the line's own
    `command -v herdr-core` finds the tested bytes. The caller's env is
    never mutated."""
    if "\0" in line:
        raise ViaRefused(f"via: refusing line with a NUL byte: {line!r}")
    if not line.strip():
        raise ViaRefused(f"via: refusing an empty handover: {line!r}")
    child_env = dict(env)
    core_bin = env.get("BOARD_VIEWER_HERDR_CORE_BIN", "")
    if not core_bin.startswith("/"):
        raise ViaRefused("via: BOARD_VIEWER_HERDR_CORE_BIN must be an absolute pinned path")
    child_env["PATH"] = f"{os.path.dirname(core_bin)}:{child_env.get('PATH', '')}"
    return ["/bin/sh", "-c", line], child_env


def main() -> int:
    # a diagnostic breadcrumb in the sandbox's own tmp dir: the exact
    # argv the core handed over, so a composition surprise is evidence,
    # not a guess
    try:
        trail = os.path.join(os.environ.get("TMPDIR", "/tmp"), "via-args.log")
        with open(trail, "a", encoding="utf-8") as log:
            log.write(repr(sys.argv) + "\n")
    except OSError:
        pass
    line = line_from_argv(sys.argv)
    try:
        argv, child_env = shell_argv(line, dict(os.environ))
    except ViaRefused as refusal:
        print(refusal, file=sys.stderr)
        return 2
    os.execve(argv[0], argv, child_env)


if __name__ == "__main__":
    sys.exit(main())
