#!/usr/bin/env python3
"""Raw probe: run one command under a PTY, capture every byte, feed a
script of timed inputs, stop on a marker or timeout. Evidence only.
usage: pty_probe.py <out-log> <marks-log> <cwd> <timeout-seconds>
       <json-inputs> <command> [args...]
inputs: list of [delay_seconds, text] sent from the first byte seen
"""
import json
import os
import pty
import select
import sys
import time

out_log, marks_log, cwd, timeout_s, inputs_json = sys.argv[1:6]
cmd = sys.argv[6:]
inputs = json.loads(inputs_json)
deadline = time.time() + float(timeout_s)

marks = open(marks_log, "a")
cap = open(out_log, "wb")


def mark(text):
    marks.write(f"{time.time():.3f} {text}\n")
    marks.flush()


env = dict(os.environ)
pid, fd = pty.fork()
if pid == 0:
    os.chdir(cwd)
    os.execvp(cmd[0], cmd)

START = time.time()
sent = 0
buf = b""
while time.time() < deadline:
    r, _, _ = select.select([fd], [], [], 0.3)
    if r:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        cap.write(chunk)
        cap.flush()
        if sent == 0:
            mark("first output")
            sent = 1
    now = time.time()
    while inputs and sent >= 1 and now - START >= float(inputs[0][0]):
        _, text = inputs.pop(0)
        os.write(fd, text.encode())
        mark(f"sent: {text!r}")

cap.close()
marks.close()
try:
    os.close(fd)
except OSError:
    pass
try:
    os.kill(pid, 15)
except ProcessLookupError:
    pass
print(f"{len(buf)} bytes captured")
