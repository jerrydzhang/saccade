"""T-36 phase-1 harness tests.

Run: python3 -m unittest discover -s client/harness/tests -t . -v
Each module's tests carry their own skip conditions where a test needs
a real binary (sac) or a WebKit browser; the pure tests always run.
"""
