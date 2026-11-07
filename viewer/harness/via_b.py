"""The remote mesh lane's via program: what core A runs when it dials
core B. machine.add --via hands the program one shell line ("a shell
line to run there as its last argument, like ssh"); two shapes exist,
and everything else refuses:

- the reachability probe the core dials with, exactly "echo ok": the
  pipe itself answers "ok" - bytes pass, nothing asked;
- a connect line whose head is exactly "hc": rewritten to the pinned
  binary by path and executed against session b, overriding the
  inherited session a.

The stdin/stdout pipe to A's core is the transport; no shell is ever
spawned, nothing retries. Runs inside the sandbox as
[<BOARD_VIEWER_VIA_PY>, /pin/via_b.py, <line>]. The decision lives in
decide() so the contract runs against fakes in the fast check loop.
"""

from __future__ import annotations

import os
import shlex
import sys

PROBE_ARGV = ["echo", "ok"]


class ViaRefused(Exception):
    """The composed line is neither the probe nor a connect line; A
    surfaces the refusal, and the message quotes the line."""


def decide(line: str, env: dict[str, str]) -> tuple[str, object, object]:
    """('print', text, None) for the probe, ('exec', argv, child_env)
    for a connect line."""
    if "\0" in line:
        raise ViaRefused(f"via: refusing line with a NUL byte: {line!r}")
    try:
        argv = shlex.split(line)
    except ValueError as error:
        raise ViaRefused(f"via: unparseable line {line!r}: {error}") from error
    if argv == PROBE_ARGV:
        return "print", PROBE_ARGV[1], None
    if argv and argv[0] == "hc":
        core = env.get("BOARD_VIEWER_HERDR_CORE_BIN", "")
        if not core.startswith("/"):
            raise ViaRefused("via: BOARD_VIEWER_HERDR_CORE_BIN must be an absolute pinned path")
        rewritten = [core, *argv[1:]]
        child_env = dict(env)
        child_env["HC_SESSION"] = "b"
        return "exec", rewritten, child_env
    raise ViaRefused(f"via: refusing line that is neither the probe nor a connect line: {line!r}")


def line_from_argv(words: list[str]) -> str:
    """The composed line, however the core hands it over: the via
    command's own words are dropped, the rest is the line - joined
    when the core appends it word by word, intact when it appends one
    quoted argument."""
    return " ".join(words[1:])


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
        action, payload, child_env = decide(line, dict(os.environ))
    except ViaRefused as refusal:
        print(refusal, file=sys.stderr)
        return 2
    if action == "print":
        print(payload)
        return 0
    os.execve(payload[0], payload, child_env)


if __name__ == "__main__":
    sys.exit(main())
