"""The remote mesh lane's via program: what core A runs when it dials
core B. machine.add --via hands the program one shell line ("a shell
line to run there as its last argument, like ssh"); the line's head
must be exactly hc - it is rewritten to the pinned binary by path and
executed against session b, overriding the inherited session a. The
stdin/stdout pipe to A's core is the transport; no shell is ever
spawned, nothing asks, nothing retries.

Runs inside the sandbox as [<HERDR_VIA_PY>, /pin/via_b.py, <line>].
The decision lives in resolve_line so the contract runs against fakes
in the fast check loop.
"""

from __future__ import annotations

import os
import shlex
import sys


class ViaRefused(Exception):
    """The composed line is not the connect shape this program exists
    for; A surfaces the refusal, and the message quotes the line."""


def resolve_line(line: str, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """The whole decision, pure: (argv, child env) for the exec."""
    if "\0" in line:
        raise ViaRefused(f"via: refusing line with a NUL byte: {line!r}")
    try:
        argv = shlex.split(line)
    except ValueError as error:
        raise ViaRefused(f"via: unparseable line {line!r}: {error}") from error
    if not argv or argv[0] != "hc":
        raise ViaRefused(f"via: refusing line whose head is not hc: {line!r}")
    core = env.get("HERDR_CORE_BIN", "")
    if not core.startswith("/"):
        raise ViaRefused("via: HERDR_CORE_BIN must be an absolute pinned path")
    rewritten = [core, *argv[1:]]
    child_env = dict(env)
    child_env["HC_SESSION"] = "b"
    return rewritten, child_env


def main() -> int:
    line = sys.argv[-1] if len(sys.argv) > 1 else ""
    try:
        argv, child_env = resolve_line(line, dict(os.environ))
    except ViaRefused as refusal:
        print(refusal, file=sys.stderr)
        return 2
    os.execve(argv[0], argv, child_env)


if __name__ == "__main__":
    sys.exit(main())
