# Port staging: ratified interaction modules (t-36 slice 2)

Pure, dependency-free TypeScript extractions of the ratified
interaction modules, staged here before slice 2 lifts them into the
port tree. Law: t-36@r62's clean-room rule - prototype interaction
code is sanctioned where it exactly implements a ratified contract,
recorded as module -> contract -> covering checks. The ratified
contracts are `t-36@r46:reference/input-context-contract.ts` (the
implemented `input-context.ts`) and the scenario inventory
`t-36@r50:reference/acceptance-scenarios.txt`. The executed prototype
of record is the round-16 mockup (templates
`t-34@r6:reference/inst16.html` + `shell16.html`; the compiled
artifact t-22@r3 is the same code assembled), materialized by
`client/harness/t36_prototype.py`. Fidelity: port the logic as the
prototype behaves it, except where a ratified difference applies.

Gates: `node --experimental-strip-types --test *.test.ts` (60 tests,
green); `tsc --noEmit` strict over the module files (tsconfig.json;
the tests need @types/node, which slice 2's port tree provides).

## Modules

| module | contract | scenario clauses | prototype source | app source (sanctioned reuse) |
|---|---|---|---|---|
| `input-context.ts` | input-context-contract types: KeyInput, Action, StrokePattern/KeySequence + constructors, CommandBinding, InputContext + context constructors, CommandDescription + describeCommands, KeyResolution, ContextMatcher | 10 (help and execution read the same declarations; queries read current commands) | inst16.html:71-97 (one commands() table read fresh by runCmd and showHelp) | menu.ts@b8240b5:38-40 (space reads "spc") |
| `key-matcher.ts` | createMatcher/ContextMatcher; SEQUENCE_TIMEOUT_MS | 10 (unmatched application keys consumed in the command context; text entry matches declared commands first, other typing native, never the board keymap; sequences through one matcher; partial never fires early, never survives a handoff; timeout 1000 ms ratified t-36@r49, abandon pending at expiry, received keys then match their own bindings if declared, else consumed) | inst16.html:458-516 (keydown chain: branches consume; :499 gg via gPending) | adapter A1 (harness t36_adapters.py:45-59) carries the ratified timeout difference |
| `label-select.ts` | - (the selection interaction's narrowing; contract Action not needed) | 12 (selecting by label selects that target, never a silent neighbour); 19 line 439 (the label case-fold fix, green as S12); mandate's case-fold wording | inst16.html:100-101 (labelsFor), :106 (hintKey: pending prefix, exact-single, Escape clears) | select.ts@b8240b5 (case fold + two namespaces + text-target narrowing, originally dd53b50) |
| `find-session.ts` | contract findExample bindings (Escape/Enter/Shift-Enter) | 16 (bare / enters Find; typing updates matches; match movement is not trail navigation; exit changes neither revision nor history; reference activation ends Find; a click alone never closes; Back does not resurrect Find) | inst16.html:402-409 (findText: trim, case-insensitive over the shown board), :71-79/:501 (prototype entry: space then /) | App.svelte@4d59c35:1055-1061 (open/close), :2836-2841 (Escape), :3486-3491 (nextMatch wrap), :3521 (Enter/Shift-Enter), :3187-3190 (derived occurrences); lib/board/projection.ts:202-238 (searchUnits, collectMatches, normalizeQuery); adapter A2 (t36_adapters.py:61-67) |
| `menu-invocation.ts` | contract Action identity; KeyDisposition consume | 9 (menu key after selection, right-click-then-Enter, right-click-then-click, plain click: same action object; plain click means follow here; close-before-execute; Escape/outside click dismiss without acting; unrelated keys as the prototype - no universal close rule) | inst16.html:37-43 (KINDS; ref's first act Follow here), :55-59 (select; item click), :66 (act: close then run; invalid index closes without acting), :69 (primary = first act, no menu), :504 (Escape/Enter/act-key/unrelated-key), :517 (outside pointer-down dismisses), :531 (contextmenu opens at pointer) | menu.ts@b8240b5 (onClose on every close, so selection state follows) |

## Ratified differences carried (per the adapters and the record)

- Sequence timeout: 1000 ms (vim timeoutlen), pending abandoned at
  expiry, received keys then matching their own bindings or consumed -
  the prototype drops its pending g silently at 700 ms (adapter A1;
  ratified t-36@r49; scenario 10). The host schedules the timer;
  `expire(now?)` carries the semantics and `SEQUENCE_TIMEOUT_MS` the
  value. Changing the value is a mechanical task; changing what expiry
  means is a ratification.
- Find entry: bare `/` - the prototype enters with space then /
  through the leader (adapter A2; scenario 16 differential note).

## Staged decisions on draft-open points (for the human's pinning)

The contract declares these draft; the prototype executed some related
behavior. What the staged modules do, to be pinned or re-decided:

1. Stroke matching is exact field equality (key and all four
   modifiers). Normalization values stay draft-open, so the host
   normalizes KeyboardEvents before calling; nothing is invented here.
2. Unmatched modifier chords in a command context are consumed. The
   prototype lets Ctrl/Meta/Alt combos fall through to the browser
   (inst16.html:510) - the IME/native-editing boundary set is
   draft-open; the draft invariant "command contexts consume unmatched
   application keys" is staged as-is.
3. An unrelated key during a pending sequence resolves on its own and
   leaves the pending buffer untouched - the executed gg behavior
   (inst16.html:499: non-g keys fall through their own branches while
   gPending stays). Draft-open ("prefix mismatch behavior must be
   pinned"); alternatives (vim-style replay of the buffer, or
   abandonment) were not chosen.
4. A stroke that completes a sequence but could still extend waits
   (vim timeout semantics); at expiry the completed spelling fires
   ("received keys then match their own bindings").
5. Duplicate sequences: first declared wins (the draft's
   collision-as-validation-error invariant is NOT implemented - it is
   listed in the contract as not-implemented and scenario 10 keeps
   collision handling draft).
6. `expire()` extends the draft's two-method ContextMatcher surface:
   the ratified timeout semantics need a host-scheduled drain, and
   resolve() must stay single-resolution.
7. Expiry drain fallback per context kind: consume in a command
   context, native-editing in text entry (the per-kind unmatched rules
   from the draft invariants; scenario 10's sentence "else are
   consumed" is read as the command-context case).
8. Label pools are alphabetical in two simultaneous namespaces
   (lowercase on text targets, uppercase on the rest) - the app's
   accepted shape (select.ts@b8240b5). The prototype used one
   namespace per hint session in home-row order (LETTERS,
   inst16.html:100) and had no text-target narrowing; the two-
   namespace narrowing comes from the accepted app behavior. Not
   pinned by scenario text; flagged for review.
9. Digits narrow text (any single character does). The prototype's
   hintKey ignores non-letters (inst16.html:106). Accepted app
   behavior; flagged.
10. Find: the prototype's 2-character minimum query is not carried
    (baseline green behavior: any non-blank query searches). The
    active match index persists while typing and entry never clears
    the query - baseline behavior, including the arithmetic where the
    first Enter lands on index 1.
11. Find's navigate-away exit (a reference activation ends Find) is
    scenario-ratified but executed by neither source: the prototype's
    find is a prompt dialog and the baseline leaves the search bar
    open through a citation activation. The module follows the
    scenario; the differential carries Find behavior through adapter
    A2's entry-only transformation, so this clause is judged against
    the ratified text, not the prototype.
12. Key formatting for help ("Ctrl-d", "Shift-Enter", "g g", space as
    "spc") is a staging default; the draft names a shared formatter
    without pinning its strings.

## Out of scope here

The leader which-key menu (space then a key; inst16.html:84-97,
different close behavior), menu positioning/hit-testing (component
lane), scroll-dismissal of a menu (inst16.html:532, executed but not
part of scenario 9's ratified surface), and the prototype's
find-marks hint selection (a presentation of module 2's narrowing on
find hits).
