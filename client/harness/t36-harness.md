T-36 PHASE-1 ACCEPTANCE HARNESS - what landed, how to run it
============================================================

Components R1-R6 of the harness design (t-36@r54, adapters approved
r55), built on this branch. Build only - no red run has been started.
Existing free-text lane output is untouched everywhere; everything
here is added surface.

Files (all in client/harness/)
------------------------------
t36_results.py      R1  stable check ids (S01..S20 plus lane), TAP13
                        emitter, run manifest, manifest validation.
t36_adapters.py     R1  adapter set A1-A7 as data carrying
                        ADAPTER_SET_REVISION; the named-only guard.
t36_driver.py       R2  two-run driver: constant plans, provenance
                        refusal, per-check diff. --dry-run is default.
t36_observation.py  R3  the app-side readout script (evaluate door),
                        the readout contract, the STOP rule.
t36_shim.js         R4  the prototype shim: one readout shape, two
                        sides; observation only.
t36_prototype.py    R4  materialize the compiled prototype from
                        t-22@r3 through real sac, vendor its CDN
                        scripts, serve loopback with URLs rewritten,
                        drive WebKit (playwright for mechanics,
                        WebKitWebDriver for the lane).
t36_windows.py      R5  host-side window surface: request shapes,
                        per-window readout, bounded window-count wait.
t36_seeds.py        R6  per-run seeded db: group-made boards through
                        real sac serve, commit list for the manifest.
webdriver_client.py R5  +window_handles/current_window/switch_to_window.
webdriver_session.py R3/R5 +window verbs on the payload command
                        surface (windows/open-window/focus-window/
                        observe-window/evaluate-window).
entry.py            R5  host verbs for the same; the new modules join
                        check-fast's pyflakes pass.

Tests (no app, no nix build, no gated lane)
-------------------------------------------
python3 -m unittest discover -s client/harness/tests -t . -v

  test_t36_results.py    inventory, results, TAP, manifest (pure).
  test_t36_adapters.py   adapter registry and guard (pure).
  test_t36_driver.py     provenance refusal, plan constancy, diff.
  test_t36_observation.py the readout script in real WebKit against a
                         harness-owned fixture page (the attribute
                         split absorbed), the STOP rule (pure part
                         always runs).
  test_t36_shim.py       the shim against the ACTUAL compiled
                         prototype, plus the input round trip
                         (space space opens the picker, read back
                         through the shim). Static and artifact tests
                         always run; the WebKit run skips cleanly if
                         playwright or sac is missing.
  test_t36_windows.py    payload window verbs through handle_request
                         with a fake WebDriver; host builders.
  test_t36_seeds.py      REAL seed through real sac serve on a
                         throwaway db (group boards crossing the real
                         wire); pure board-shape tests always run.

The four required proofs: TAP emitter and manifest validation
(test_t36_results), shim readout against the actual prototype
(test_t36_shim.PrototypeReadoutTest), driver dry-run refusal without
provenance (test_t36_driver plus the CLI exit-2 path), and the real
seed (test_t36_seeds.RealSeedTest).

Standalone smokes
-----------------
python3 client/harness/t36_prototype.py          # boot + shim + input
python3 client/harness/t36_driver.py --baseline-bin <store-bin> \
    --baseline-ref <sha> --candidate-bin <store-bin> \
    --candidate-ref <sha> --board <abs.yaml> --tracker-db <abs.db> \
    --serve-bin <store-sac> --serve-ref <sha> --checks S08,S18
    # dry-run plans; the run refuses without any ref
python3 client/harness/t36_seeds.py --root /tmp/t36-seed-smoke

Deviations announced (the problem was not silently modified)
------------------------------------------------------------
1.  The prototype artifact. The design says "load inst16.html and
    shell16.html". Those raw files are build INPUTS: shell16 carries
    /*DATA*//*INST*/ placeholders and inst16 /*TOKENS*//*COMPONENTS*/
    /*DIFF*/ - WebKit parses them and dies at the JSON step (verified:
    "JSON Parse error: Unrecognized token '/'"). The executable
    prototype is the compiled round-16 mockup, and its provenance is
    on the tracker: t-22@r3 (commit 77db4e9a), built by
    reference/build16.py from exactly those templates. The harness
    materializes that artifact through the real sac binary at run
    time, refuses anything carrying template markers, and static
    tests still pin the shim's expectations against the raw
    inst16.html/shell16.html text.
2.  Serving, not editing. The compiled page names three pinned CDN
    scripts (js-yaml 4.1.0, marked 12.0.2, mermaid 10.9.1). The
    red-run sandbox has no network, so the harness vendors those
    three files once (hashes recorded in the manifest) and the
    loopback server rewrites only those three URLs to same-origin
    paths. The page bytes on disk are never modified.
3.  Manifest provenance is caller-supplied. Nothing stamps the app
    binary (the design forbids stamp code), so app build refs arrive
    as --baseline-ref/--candidate-ref and ride the manifest
    unverified-but-recorded; the /nix/store executable path is
    validated. A --version-style pin would be an app change; not
    made.

Known stops to carry into the red run
-------------------------------------
- tauri-driver window-handle passthrough is unproven (the findings
  say no lane ever used it). The R5 verbs answer ok:false with the
  recorded inventory when enumeration fails; S18/S20 then stop, they
  do not fail silently. WebKitWebDriver's promise resolution for
  execute/sync (suite_windows invoke) rides on the same run.
- The app's trail/history readout does not exist in inspection();
  checks that need it stop via require_fields (S14's A5 comparison
  needs the prototype side plus whatever the app exposes; the red run
  records the exact gap).
- WebKitWebDriverPrototypeSession (the lane path for R4) needs a
  WebKit browser binary and the WebKitWebDriver binary under the lane
  sandbox; both are provenance-refused, never guessed, and the path
  is exercised only at the red run.

Manifest shape (t36-run-manifest/1)
-----------------------------------
schema, lane, harness_commit, adapter_set_revision,
checks_inventory (t36-inventory/2), app{bin,ref}, serve{bin,ref},
run_id, started, seeds{db_path, db_hash, repo, serve_bin,
commits[{task, revision, message, commit}]}, prototype{source,
sha256, serving, libs}. Validation refuses: non-store app bins, empty
refs, a serve binary without its declared build ref or on a
non-store path (t-36@r58 - a store path alone shape-validated a
stale serve binary; the ref is what catches serve skew), wrong
schema, bad hashes, and - with require_seeds - a seedless run.

TAP shape
---------
TAP version 13; "# manifest:" comments carry the manifest; one line
per result ordered by (id, lane, name); "ok N - S08/session <name>",
"not ok N - ... # SKIP stopped: <reason>" for the R3 stops, plain
"not ok" for failures; failures carry a yaml detail block and adapters
a comment. tap_exit_code: any non-skip "not ok" fails the run.

TAP always exists (t-36@r58)
----------------------------
The driver writes tap-<role>.tap on every path. The lane's own
free-text check lines are parsed into the pinned legacy gates
(t36_results.GATES, ids G01..; G00 is the synthetic lane-completion
entry for a lane that exited without any recorded check line). A
failed gate is a FAIL entry with its evidence; declared checks the
lane never reached STOP against it ("not reached: the lane stopped at
failed check ..."). The per-check diff lands beside the run roots
once both roles have TAP. Run 1's failure shape (the tracker-load
gate refusing t-1 on serve skew) is a test fixture: it produces a
failing TAP with SKIPs instead of a pre-TAP early exit.
