/* T-36 phase-1 prototype observation shim (R4).
 *
 * Injected into the compiled round-16 prototype page (shell16 with the
 * inst16 instruments assembled in, t-22@r3). It mirrors the app-side
 * readout contract (t36_observation.READOUT_JS): one readout shape, two
 * sides, so a differential check reads both the same way.
 *
 * OBSERVATION ONLY. The shim reads the prototype's own surfaces and
 * writes nothing but its single `window.__t36Shim` answer holder:
 *   - SUITE.states (ids, names, instruments, saved snapshots, focus,
 *     device, recent) and the attached `frames` map,
 *   - each frame's document, for the [data-kind]/[data-loc] rows the
 *     prototype renders (task rows, agent rows, refs, board parts,
 *     revs, slides, tabs), plus .selected/.menu and .picker.open state,
 *   - window.snapshot()/SUITE.states[id].snap for the viewer, deck and
 *     tasks state (trails, history and marginalia flags, deck slides),
 *   - SESSIONS for the words-like agent states,
 *   - the strip's .loc/.name and the deckbar text per window.
 * No prototype function is called, no listener is installed, no state
 * is mutated. The data-task vs data-tasks split does not exist on the
 * prototype side; the shared rows shape carries the prototype's
 * data-loc rows and the app's normalized tasks[] rows under one key.
 */
(function () {
  var out = {
    side: "prototype",
    booted: false,
    missing: [],
    rows: [],
    words: [],
    windows: [],
    trails: {},
    strip: {location: null, state: null, deckLine: null, deckState: null, link: null},
    deck: null,
    historyEntries: [],
    picker: {open: false, value: null},
    selection: "",
    suite: null,
  };
  if (typeof SUITE === "undefined" || !SUITE || !SUITE.states) {
    out.missing.push("SUITE");
    return out;
  }
  if (typeof SESSIONS === "undefined") {
    out.missing.push("SESSIONS");
  }
  out.booted = true;
  out.suite = {
    focus: SUITE.focus || null,
    device: SUITE.device || null,
    recent: (SUITE.recent || []).slice(0, 10),
  };

  var attached = [];
  if (typeof frames !== "undefined" && frames) {
    attached = Object.keys(frames);
  }

  function locOf(cur) {
    if (!cur) return null;
    if (cur.machine) return cur.machine + ":" + cur.path;
    if (cur.task != null) return "t-" + cur.task + (cur.rev ? "@r" + cur.rev : "");
    return null;
  }

  function deckOf(snap) {
    var deck = snap && snap.deck;
    if (!deck || !deck.on) return null;
    return {on: true, slides: (deck.slides || []).length, position: deck.pos, waiting: deck.waiting || 0};
  }

  function trailOf(snap) {
    var viewer = snap && snap.viewer;
    if (!viewer) return null;
    return {
      back: (viewer.back || []).length,
      forward: (viewer.fwd || []).length,
      history: !!viewer.hist,
      marginalia: !!viewer.marg,
      location: locOf(viewer.cur),
    };
  }

  function windowReadout(id, frameEl) {
    var state = SUITE.states[id] || {};
    var snap = state.snap || {};
    var readout = {
      id: id,
      name: state.name || null,
      inst: state.inst || null,
      attached: attached.indexOf(id) >= 0,
      focused: SUITE.focus === id,
      location: locOf(snap.viewer && snap.viewer.cur) || snap.loc || null,
      deck: deckOf(snap),
      trail: trailOf(snap),
      tasksScope: snap.tasks && snap.tasks.scope ? snap.tasks.scope : null,
      selection: null,
      picker: {open: false, value: null},
      strip: {name: null, location: null, deckLine: null, deckState: null, link: null},
      historyEntries: [],
    };
    var frameWindow = frameEl && frameEl.querySelector ? frameEl.querySelector("iframe") : null;
    var doc = frameWindow ? frameWindow.contentDocument : null;
    if (!doc) return readout;
    var nameEl = doc.querySelector(".strip .name");
    var locEl = doc.querySelector(".strip .loc");
    readout.strip.name = nameEl ? (nameEl.textContent || "").trim() : null;
    readout.strip.location = locEl ? (locEl.textContent || "").trim() : null;
    var chips = doc.querySelectorAll(".strip .chip");
    for (var i = 0; i < chips.length; i++) {
      var chipText = (chips[i].textContent || "").trim();
      if (chipText.indexOf("deck") === 0) readout.strip.deckLine = chipText;
    }
    var whatEl = doc.querySelector(".deckbar .what");
    readout.strip.deckState = whatEl ? (whatEl.textContent || "").trim() : null;
    var linkEl = doc.querySelector(".strip .lnk");
    readout.strip.link = linkEl ? (linkEl.textContent || "").trim() : null;
    var selected = doc.querySelector(".selected");
    if (selected) {
      readout.selection = {
        kind: selected.dataset ? selected.dataset.kind : selected.getAttribute("data-kind"),
        loc: selected.dataset ? selected.dataset.loc : selected.getAttribute("data-loc"),
        label: selected.dataset ? selected.dataset.label : selected.getAttribute("data-label"),
      };
    }
    var menuEl = doc.getElementById("menu");
    if (menuEl && menuEl.classList.contains("open")) {
      readout.selection = readout.selection || {};
      readout.selection.menuOpen = true;
    }
    var pickerEl = doc.querySelector(".picker.open input");
    if (pickerEl) readout.picker = {open: true, value: pickerEl.value || ""};
    var entries = doc.querySelectorAll(".clog .ent");
    for (var e = 0; e < entries.length && e < 50; e++) {
      readout.historyEntries.push((entries[e].textContent || "").trim().slice(0, 300));
    }
    return readout;
  }

  for (var w = 0; w < attached.length; w++) {
    var frameEl = frames[attached[w]];
    var readout = windowReadout(attached[w], frameEl);
    out.windows.push(readout);
    out.trails[attached[w]] = readout.trail;
    if (SUITE.focus === attached[w]) {
      out.strip = readout.strip;
      out.deck = readout.deck;
      out.historyEntries = readout.historyEntries;
      out.picker = readout.picker;
      var sel = readout.selection || {};
      out.selection = sel.loc || sel.menuOpen ? (sel.kind || "") + (sel.loc ? " " + sel.loc : "") + (sel.menuOpen ? " menu-open" : "") : "";
    }
  }
  // detached windows still carry saved state in the pool: report them,
  // without frame DOM, so the inventory matches SUITE.states
  var allIds = Object.keys(SUITE.states);
  for (var s = 0; s < allIds.length; s++) {
    var id = allIds[s];
    if (attached.indexOf(id) >= 0) continue;
    var detached = windowReadout(id, null);
    out.windows.push(detached);
    out.trails[id] = detached.trail;
  }

  function collectRows(doc, windowId) {
    var rows = doc.querySelectorAll("[data-kind]");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      var mark = el.querySelector(".st");
      var word = null;
      if (mark) {
        var classes = mark.className || "";
        var match = classes.match(/\b(error|blocked|working|idle|done)\b/);
        word = match ? match[1] : null;
      }
      out.rows.push({
        window: windowId,
        kind: el.getAttribute("data-kind"),
        loc: el.getAttribute("data-loc"),
        label: el.getAttribute("data-label"),
        state: el.getAttribute("data-state"),
        chg: el.getAttribute("data-chg"),
        path: el.getAttribute("data-path"),
        addr: el.getAttribute("data-addr"),
        claim: el.getAttribute("data-claim"),
        key: el.getAttribute("data-key"),
        word: word,
        text: (el.textContent || "").trim().slice(0, 120),
      });
    }
  }

  for (var f = 0; f < attached.length; f++) {
    var frame = frames[attached[f]];
    var iframe = frame && frame.querySelector ? frame.querySelector("iframe") : null;
    if (iframe && iframe.contentDocument) collectRows(iframe.contentDocument, attached[f]);
  }

  if (typeof SESSIONS !== "undefined" && SESSIONS) {
    for (var s2 = 0; s2 < SESSIONS.length; s2++) {
      var session = SESSIONS[s2];
      out.words.push({
        id: session.id,
        harness: session.harness || null,
        word: session.report || null,
        msg: session.msg || null,
        progress: session.progress != null ? session.progress : null,
        claims: (session.claims || []).map(function (t) { return "t-" + t; }),
      });
    }
  }
  return out;
})()
