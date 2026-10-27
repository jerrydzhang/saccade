import { EditorState, EditorSelection, Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { minimalSetup } from "codemirror";
import { lineNumbers } from "@codemirror/view";
import { headerFolding, sectionFold } from "./editor.mjs";
import { markdown } from "@codemirror/lang-markdown";
import { vim, Vim, getCM } from "@replit/codemirror-vim";
import { Fzf } from "fzf";
import { Workbench, commandFor, confirmsLanded, createTaskCommand, confirmsTaskCreated, launchRequest, launchOutcome, leaves } from "./workbench.mjs";
import { richDocuments } from "./rich.mjs";
import { sourceAppearance } from "./appearance.mjs";

const bench = new Workbench();
const editors = new Map();
const readers = new Map();
const panes = new Map();
const history = new Map();
const observedModes = new WeakSet();
let snapshot, identity, storageKey;
let syncCursor;           // the record cursor the read-side follow last saw
let picker = null;
let leader = null;         // " " global prefix pieces, or "," local
let pendingG = false;
let pendingZ = false;
let pendingWindow = false;
let pickerPreviewGeneration = 0;
let claimedName = null;    // the name a human typed when no cookie claimed one yet
let stagedTask = null;     // the task name a prompt confirmed, awaiting its create command
let stagedLaunch = null;   // the workspace choice a picker confirmed, awaiting its launch command
let submitting = false;
let nextDraft = 1;
let messageTimer = 0;
// first-pass home bounds, provisional by agreement: Show more reveals
// older entries in the same increments and no totals are shown
const ACTIVITY_STEP = 20;
const TASKS_STEP = 10;
const $ = (selector) => document.querySelector(selector);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const task = (id) => snapshot.tasks.find((item) => item.id === id);
const comment = (id) => snapshot.comments.find((item) => item.id === id);
const birth = (seq) => snapshot.comments.find((item) => item.seq === seq);
const current = () => bench.buffer(bench.view().bufferId);
const currentEditor = () => editors.get(bench.focus);
const time = (value) => new Date(value * 1000).toISOString();

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Object.assign(new Error(typeof data.error === "string" ? data.error : JSON.stringify(data.error)), { data });
  return data;
}

function status(text) {
  const node = $("#message");
  node.textContent = text;
  clearTimeout(messageTimer);
  if (text) messageTimer = setTimeout(() => { node.textContent = ""; }, 6000);
}

function referenceLabel(seq) {
  const source = birth(seq);
  return source ? `${source.id}  t-${source.home} ${task(source.home)?.name ?? ""}` : `record ${seq}`;
}

function kindName(kind) {
  return { home: "Home", task: "Task", note: "Note", offer: "Offer", accept: "Accept", reject: "Reject", use: "Use", draft: "Draft" }[kind] ?? kind;
}

// the bar shows the tracker kind: a draft names its eventual record
// kind, a comment its actual record kind even mid-revision, and
// home/task stay themselves
function statusKind(buffer) {
  if (buffer.kind === "draft") return buffer.eventual?.kind ?? "draft";
  return buffer.source?.kind ?? buffer.kind;
}

function commentText(source) {
  const lines = [
    `${source.id}  ${source.kind}${source.title ? `  ${source.title}` : ""}`,
    `home: t-${source.home}  ${task(source.home)?.name ?? ""}`,
    `by: ${source.actor}/${source.tier}`,
    `event_time: ${time(source.born_at)}`,
  ];
  if (source.revised) lines.push(`presentation: revised${source.reviser ? ` by ${source.reviser}` : ""}`);
  if (source.withdrawn) lines.push(`presentation: withdrawn\n${source.withdrawal_note}`);
  if (source.offer !== null) lines.push(`offer: c-${source.offer}`);
  if (source.source !== null) lines.push(`source: t-${source.source}`);
  if (source.binding) lines.push(`selected offer: c-${source.binding.offer}\ngrounds: ${source.binding.grounds}`);
  const body = source.withdrawn ? source.withdrawal_note ?? "" : source.body;
  const bodyFrom = lines.join("\n").length + 2;
  const richSpans = (source.rich_ranges ?? []).map((span) => ({ ...span, from: bodyFrom + span.from, to: bodyFrom + span.to }));
  lines.push("", body);
  if (source.references.length) {
    lines.push("", "References");
    source.references.forEach((seq, n) => {
      lines.push(`[^${n + 1}]: ${referenceLabel(seq)}`);
      const artifact = source.artifact_previews?.find((item) => item.index === n);
      if (artifact) {
        const at = lines.join("\n").length;
        richSpans.push({ from: at, to: at, kind: "artifact", html: artifact.html });
      }
    });
  }
  if (source.cited_by.length) {
    lines.push("", "Cited by");
    for (const num of source.cited_by) {
      const citing = comment(`c-${num}`);
      lines.push(`${citing.id}  t-${citing.home}  ${citing.actor}`, citing.body || citing.withdrawal_note || citing.kind, "");
    }
  }
  return { text: lines.join("\n"), richSpans, bodyFrom, bodyTo: bodyFrom + body.length };
}

function published(id) {
  let data;
  if (id === "home") {
    data = { id, kind: "home", title: "Home", text: "Saccade\n\nFind work, inspect contributions, and write with sources beside you.\n\n: opens commands\nSpace Space  find task, contribution, or local draft\nSpace sg  search recorded content\n, e  revise the focused contribution\n, a / , x  accept / reject the standing offer\n, t  title the focused draft\n, p  publish the focused draft\ngd  follow id or citation\ngr  show backlinks\nCtrl+O / Ctrl+I  jump backward / forward\nSpace nn  new note\nSpace nt  new task\nSpace nl  launch a session\n\nOpen only the buffers and panes you want." };
  } else if (id.startsWith("t-")) {
    const source = task(Number(id.slice(2)));
    if (!source) throw new Error("task is unavailable");
    const lines = [`${id}  ${source.name}`, `state: ${source.state}`, `owner: ${source.owner ?? "none recorded"}`];
    if (source.explanation) lines.push(`explanation: ${source.explanation}`);
    lines.push("", "Contributions");
    for (const item of snapshot.comments.filter((c) => c.home === source.id)) {
      lines.push("", `${item.id}  ${item.kind}${item.title ? `  ${item.title}` : ""}  ${item.actor}  ${time(item.born_at)}`,
        item.body || item.withdrawal_note || item.kind);
    }
    data = { id, kind: "task", title: `${id} ${source.name}`, home: source.id, text: lines.join("\n") };
  } else {
    const source = comment(id);
    if (!source) throw new Error("contribution is unavailable");
    data = { id, kind: "comment", title: `${id} | ${source.title ?? task(source.home)?.name ?? ""}`, home: source.home, ...commentText(source), source };
  }
  const existing = bench.buffers.get(id);
  if (existing) {
    const keepRevision = existing.revisionPending;
    const keep = keepRevision ? { text: existing.text, title: existing.title } : {};
    Object.assign(existing, data, keep);
    if (!keepRevision) existing.source = data.source;
  }
  return existing ?? bench.register(data);
}

function clampSelection(buffer, position) {
  const size = buffer.text.length;
  return EditorSelection.single(Math.min(position.anchor, size), Math.min(position.head, size));
}

function editable(buffer) {
  return buffer.kind === "draft" || !!buffer.revisionPending;
}

function editorState(buffer, position) {
  if (!buffer.master || buffer.master.doc.toString() !== buffer.text) {
    buffer.master = EditorState.create({
      doc: buffer.text,
      extensions: [vim(), minimalSetup, sourceAppearance, lineNumbers(), headerFolding, markdown({ completeHTMLTags: false, pasteURLAsLink: false }), richDocuments(buffer), EditorView.lineWrapping,
        EditorState.readOnly.of(!editable(buffer)),
        EditorState.transactionFilter.of((tr) => !editable(buffer) && tr.docChanged ? [] : tr),
        Prec.highest(keymap.of([{ key: "Escape", run: () => false }])),
        EditorView.theme({ "&": { height: "100%", fontSize: "15px" }, ".cm-scroller": { overflow: "auto", fontFamily: "var(--mono)" }, ".cm-content": { padding: "16px 12px 160px" }, ".cm-focused": { outline: "none" } }),
      ],
    });
  }
  return buffer.master.update({ selection: clampSelection(buffer, position) }).state;
}

function presentation(viewId = bench.focus) {
  const pane = bench.view(viewId);
  pane.presentations ??= {};
  return pane.presentations[pane.bufferId] ??= { mode: "view", scrollTop: 0, sourceSeen: false };
}

function isReading(viewId = bench.focus) {
  const pane = bench.views.get(viewId);
  return !!pane && ["home", "comment", "draft", "task"].includes(bench.buffer(pane.bufferId).kind) && presentation(viewId).mode === "view";
}

function threadFocused(viewId = bench.focus) {
  const pane = bench.views.get(viewId);
  return !!pane && bench.buffer(pane.bufferId).kind === "task" && isReading(viewId);
}

function threadSelection(viewId = bench.focus) {
  if (!threadFocused(viewId)) return null;
  return presentation(viewId).selected ?? null;
}

// the comment the contextual actions name: the focused contribution
// buffer, or the focused thread's selected card, only when that card
// lives on the focused thread
function actionTarget() {
  const buffer = current();
  if (buffer.kind === "comment") return buffer.source ? buffer : null;
  const id = threadSelection();
  if (!id || comment(id)?.home !== buffer.home) return null;
  return bench.buffers.get(id) ?? published(id);
}

function remember(viewId) {
  if (isReading(viewId)) return;
  const view = editors.get(viewId);
  if (!view || !bench.views.has(viewId)) return;
  const selection = view.state.selection.main;
  bench.remember({ anchor: selection.anchor, head: selection.head, scrollTop: view.scrollDOM.scrollTop }, viewId);
}

function mode() {
  if (isReading()) return "View";
  const cm = currentEditor() && getCM(currentEditor());
  const state = cm?.state.vim;
  return state?.insertMode ? "Insert" : state?.visualMode ? "Visual" : "Normal";
}

function observeMode(view) {
  const cm = getCM(view);
  if (!observedModes.has(cm)) {
    observedModes.add(cm);
    cm.on("vim-mode-change", () => queueMicrotask(refreshChrome));
  }
}

function savepoint(buffer) {
  if (buffer.revisionPending) {
    const changed = buffer.revisionPending.title !== buffer.revisionPending.savedTitle
      || buffer.text !== buffer.revisionPending.savedBody;
    return changed ? "[+]" : "";
  }
  if (buffer.kind !== "draft") return "[-]";
  return buffer.text !== (buffer.savedText ?? "") ? "[+]" : "";
}

function identifier(buffer) {
  return buffer.id;
}

function identifierInfo(buffer) {
  const store = buffer.revisionPending ? buffer.revisionPending.submission
    : buffer.kind === "draft" ? buffer.submission : buffer.decision;
  const tag = { landed: "recorded", unknown: "submission unconfirmed" }[store?.kind];
  const parts = [];
  if (buffer.revisionPending) {
    const titled = ["note", "offer"].includes(buffer.source.kind);
    parts.push(`revising${titled ? `  ${buffer.revisionPending.title || "no title"}` : ""}`);
  } else if (buffer.kind === "draft") {
    if (buffer.eventual?.kind === "reject") parts.push(`rejecting c-${buffer.eventual.offer}`);
    else parts.push(buffer.home === null ? "no home" : `t-${buffer.home}`, buffer.authoredTitle || "no title");
  }
  if (buffer.kind === "task") {
    const selected = threadSelection();
    const card = selected && comment(selected);
    if (card && card.home === buffer.home) parts.push(`selected ${selected}  ${card.kind}${card.title ? `  ${card.title}` : ""}`);
  }
  if (tag) parts.push(tag);
  return parts.join("  ");
}

function refreshChrome() {
  const buffer = current();
  $("#bar-mode").textContent = mode() === "View" ? "RENDERED" : mode().toUpperCase();
  $("#bar-kind").textContent = kindName(statusKind(buffer));
  $("#bar-ident").textContent = identifier(buffer);
  $("#bar-info").textContent = identifierInfo(buffer);
  $("#bar-flags").textContent = savepoint(buffer);
  for (const [id, node] of panes) {
    node.classList.toggle("active", id === bench.focus);
    const view = bench.views.get(id);
    if (!view) continue;
    const paneBuffer = bench.buffer(view.bufferId);
    node.querySelector(".buffer-title").textContent = paneBuffer.revisionPending
      ? `${paneBuffer.id} (revising)` : paneBuffer.id === "home" ? "Home" : paneBuffer.id.startsWith("t-")
        ? `t-${paneBuffer.home} ${task(paneBuffer.home)?.name ?? ""}` : paneBuffer.id;
  }
}

function focusPane(id) {
  bench.focus = id;
  refreshChrome();
  if (isReading(id)) readers.get(id)?.focus(); else editors.get(id)?.focus();
}

function dispatch(viewId, transactions, view) {
  view.update(transactions);
  const buffer = bench.buffer(bench.view(viewId).bufferId);
  const changed = transactions.some((tr) => tr.docChanged);
  if (changed) {
    const text = view.state.doc.toString();
    if (buffer.text !== text) bench.edit(buffer.id, text);
    let changes = transactions[0].changes;
    for (const tr of transactions.slice(1)) changes = changes.compose(tr.changes);
    buffer.master = view.state;
    for (const [otherId, other] of editors) {
      if (otherId === viewId || bench.view(otherId).bufferId !== buffer.id) continue;
      const selection = other.state.selection.map(changes);
      const scroll = other.scrollDOM.scrollTop;
      other.setState(buffer.master.update({ selection }).state);
      observeMode(other);
      other.scrollDOM.scrollTop = scroll;
      remember(otherId);
    }
  } else buffer.master = view.state;
  remember(viewId);
  for (const id of editors.keys()) {
    if (bench.view(id).bufferId === buffer.id && isReading(id)) showPresentation(id);
  }
  persist();
  refreshChrome();
}

function mountEditor(viewId, node) {
  const pane = bench.view(viewId);
  const buffer = bench.buffer(pane.bufferId);
  const previous = editors.get(viewId);
  if (previous && previous.bufferId === buffer.id && buffer.master) { showPresentation(viewId); return; }
  previous?.destroy();
  const host = node.querySelector(".document");
  host.replaceChildren();
  const position = bench.position(viewId);
  const view = new EditorView({ state: editorState(buffer, position), parent: host,
    dispatchTransactions: (transactions, editor) => dispatch(viewId, transactions, editor),
  });
  view.bufferId = buffer.id;
  editors.set(viewId, view);
  const reader = el("article", undefined, "reading rich");
  reader.tabIndex = 0;
  reader.setAttribute("aria-label", "Rendered document");
  host.append(reader); readers.set(viewId, reader);
  reader.addEventListener("focusin", () => { bench.focus = viewId; refreshChrome(); });
  reader.addEventListener("scroll", () => {
    if (readers.get(viewId) === reader && isReading(viewId) && !reader.pendingSignature) { presentation(viewId).scrollTop = reader.scrollTop; persist(); }
  });
  showPresentation(viewId);
  view.scrollDOM.addEventListener("scroll", () => { if (!isReading(viewId)) { remember(viewId); persist(); } });
  view.contentDOM.addEventListener("focus", () => { bench.focus = viewId; refreshChrome(); });
  observeMode(view);
  requestAnimationFrame(() => { view.scrollDOM.scrollTop = position.scrollTop; });
}

function showPresentation(viewId) {
  const reader = readers.get(viewId), editor = editors.get(viewId);
  if (!reader || !editor) return;
  const buffer = bench.buffer(bench.view(viewId).bufferId);
  const reading = isReading(viewId);
  editor.dom.hidden = reading;
  reader.hidden = !reading;
  if (!reading) {
    reader.previewToken = null; reader.pendingSignature = null;
    return;
  }
  if (buffer.kind === "task") {
    showThread(viewId, reader, buffer);
    return;
  }
  if (buffer.kind === "home") {
    showHome(viewId, reader, buffer);
    return;
  }
  if (buffer.kind === "draft" || buffer.revisionPending) {
    showDraft(viewId, reader, buffer);
    return;
  }
  const signature = JSON.stringify([buffer.text, buffer.source.html]);
  if (reader.signature !== signature) {
    const header = el("pre", buffer.text.slice(0, buffer.bodyFrom - 2), "reading-meta");
    const home = el("a", `Open home t-${buffer.home}`);
    home.href = `/t/${buffer.home}`; home.dataset.recordTarget = `t-${buffer.home}`;
    const body = el("div"); body.innerHTML = buffer.source.html;
    const footer = el("pre", buffer.text.slice(buffer.bodyTo), "reading-meta");
    // the scroll guard spans the swap and its restore, so the transient
    // clamped scroll a re-render can fire is never taken for reading
    reader.pendingSignature = signature;
    reader.replaceChildren(header, home, body, footer);
    reader.signature = signature;
  }
  requestAnimationFrame(() => {
    if (isReading(viewId)) reader.scrollTop = presentation(viewId).scrollTop;
    reader.pendingSignature = null;
  });
}

function draftSignature(buffer) {
  if (buffer.revisionPending) {
    return JSON.stringify([buffer.revision, buffer.text, buffer.source.references, buffer.revisionPending.title]);
  }
  return JSON.stringify([buffer.revision, buffer.text, buffer.references, buffer.home, buffer.authoredTitle]);
}

// the authoring facts the ratified design keeps visible: home task and
// authored title, named as missing while they are missing
function draftMetaLine(buffer, revising) {
  if (revising) {
    const home = `home: t-${buffer.source.home}  ${task(buffer.source.home)?.name ?? ""}`;
    const titled = ["note", "offer"].includes(buffer.source.kind);
    return `${home}${titled ? `\ntitle: ${buffer.revisionPending.title || "not authored"}` : ""}`;
  }
  if (buffer.eventual?.kind === "reject") {
    const home = buffer.eventual.offerSeq === undefined ? buffer.home : comment(buffer.eventual.offer)?.home;
    return `rejecting c-${buffer.eventual.offer}\nhome: ${home === undefined || home === null ? "unresolved" : `t-${home}  ${task(home)?.name ?? ""}`}`;
  }
  return `home: ${buffer.home === null ? "not selected" : `t-${buffer.home}  ${task(buffer.home)?.name ?? ""}`}\ntitle: ${buffer.authoredTitle || "not authored"}`;
}

// rendered panes show the authoring facts; keep them live when a prompt
// rewrites those facts outside the editor
function refreshRendered(bufferId) {
  for (const id of editors.keys()) {
    const view = bench.views.get(id);
    if (!view || view.bufferId !== bufferId || !isReading(id)) continue;
    showPresentation(id);
  }
}

function showDraft(viewId, reader, buffer) {
  const signature = draftSignature(buffer);
  if (reader.signature === signature || reader.pendingSignature === signature) return;
  const token = {};
  reader.previewToken = token; reader.pendingSignature = signature;
  const revising = !!buffer.revisionPending;
  const meta = el("p", `${revising ? "Proposed repair preview. Viewing does not publish." : "Local draft preview. Viewing does not publish."}\n${draftMetaLine(buffer, revising)}`, "reading-meta");
  let body = reader.querySelector(".draft-body");
  if (!body) body = el("div", "Rendering draft...", "draft-body");
  reader.replaceChildren(meta, body);
  if (revising && buffer.revisionPending.title) {
    const title = el("h2", buffer.revisionPending.title, "draft-title");
    reader.prepend(title);
  }
  reader.setAttribute("aria-busy", "true");
  const references = revising ? buffer.source.references : buffer.references;
  const valid = () => readers.get(viewId) === reader && reader.previewToken === token && isReading(viewId) && bench.view(viewId).bufferId === buffer.id && draftSignature(buffer) === signature;
  api("/api/v1/workbench/preview", { body: buffer.text, references: [...references] }).then((result) => {
    if (!valid()) return;
    body.innerHTML = result.html;
    reader.signature = signature; reader.pendingSignature = null;
    reader.setAttribute("aria-busy", "false");
    reader.scrollTop = presentation(viewId).scrollTop;
  }).catch((error) => {
    if (!valid()) return;
    reader.pendingSignature = null;
    reader.setAttribute("aria-busy", "false");
    if (!reader.signature) body.textContent = `Draft preview unavailable: ${error.message}. Source remains available; viewing does not publish.`;
    else status(`Draft preview update unavailable: ${error.message}. Previous rendering retained; Source has the current text.`);
  });
}

// a task thread renders one card per contribution with the same
// construction the standalone contribution view uses; the selection is
// applied here and toggled in place by the j/k commands
function showThread(viewId, reader, buffer) {
  const selected = presentation(viewId).selected ?? null;
  const signature = JSON.stringify([buffer.text, selected]);
  if (reader.signature === signature) return;
  const source = task(buffer.home);
  const meta = [`${buffer.id}  ${source.name}`, `state: ${source.state}`, `owner: ${source.owner ?? "none recorded"}`];
  if (source.explanation) meta.push(`explanation: ${source.explanation}`);
  const nodes = [el("pre", meta.join("\n"), "reading-meta thread-meta")];
  for (const item of snapshot.comments.filter((c) => c.home === buffer.home)) {
    const view = commentText(item);
    const card = el("article", undefined, "thread-card");
    card.dataset.comment = item.id;
    if (item.id === selected) card.classList.add("selected");
    const body = el("div", undefined, "rich");
    body.innerHTML = item.html;
    card.append(
      el("pre", view.text.slice(0, view.bodyFrom - 2), "reading-meta"),
      body,
      el("pre", view.text.slice(view.bodyTo), "reading-meta"),
    );
    nodes.push(card);
  }
  // the scroll guard spans the swap and its restore, so the transient
  // clamped scroll a re-render can fire is never taken for reading
  reader.pendingSignature = signature;
  reader.replaceChildren(...nodes);
  reader.signature = signature;
  requestAnimationFrame(() => {
    if (isReading(viewId)) reader.scrollTop = presentation(viewId).scrollTop;
    reader.pendingSignature = null;
  });
}

// one recent-activity entry: the record facts it carries, linked the
// way its kind anchors - contribution records land on their card in
// the home thread, the rest on the task itself
function activityNode(entry) {
  const node = el("a", undefined, "home-entry");
  node.href = entry.comment !== null ? `/t/${entry.home}#c-${entry.comment}` : `/t/${entry.home}`;
  const source = entry.comment !== null ? comment(`c-${entry.comment}`) : null;
  const parts = [];
  if (entry.event === "task_created") parts.push(`t-${entry.home} created`, task(entry.home)?.name ?? "");
  else if (entry.event === "task_claimed") parts.push(`t-${entry.home} claimed`);
  else if (entry.event === "task_cancelled") parts.push(`t-${entry.home} cancelled`);
  else if (entry.event === "task_reopened") parts.push(`t-${entry.home} reopened`);
  else if (entry.event === "commented") parts.push(`c-${entry.comment} ${source?.kind ?? ""}`, source?.title ?? "");
  else if (entry.event === "comment_revised") parts.push(`c-${entry.comment} revised`, source?.title ?? "");
  else if (entry.event === "comment_withdrawn") parts.push(`c-${entry.comment} withdrawn`);
  else if (entry.event === "use_bound") parts.push(`c-${entry.comment} use bound`, entry.offer !== null ? `-> c-${entry.offer}` : "");
  else if (entry.event === "artifact_added") parts.push("artifact", entry.artifact ?? "");
  node.append(
    el("span", parts.filter((part) => part !== "").join("  "), "home-what"),
    el("span", `${entry.actor}/${entry.tier}  ${time(entry.at)}`, "home-when"),
  );
  return node;
}

// one recently active task: its identity, name, and standing facts -
// ordered by its latest included activity, never by judged relevance
function recentTaskNode(entry) {
  const source = task(entry.task);
  const node = el("a", undefined, "home-entry");
  node.href = `/t/${entry.task}`;
  node.append(
    el("span", `t-${entry.task}  ${source?.name ?? ""}`, "home-what"),
    el("span", `${source?.state ?? ""}${source?.owner ? `  ${source.owner}` : ""}  ${time(entry.at)}`, "home-when"),
  );
  return node;
}

// the home surfaces: recent activity first; recently active tasks
// beside it when the pane itself is wide, below when narrow - the
// stylesheet's container query reads the pane, not the window. The
// reveal bounds are pane state, never totals.
function showHome(viewId, reader, buffer) {
  const pane = presentation(viewId);
  const activityShown = pane.activityShown ?? ACTIVITY_STEP;
  const tasksShown = pane.tasksShown ?? TASKS_STEP;
  const signature = JSON.stringify([buffer.text, snapshot.cursor, activityShown, tasksShown]);
  if (reader.signature === signature) return;
  const surfaces = el("div", undefined, "home-surfaces");
  const activity = el("section", undefined, "home-activity");
  activity.append(el("h2", "Recent activity"));
  const activityList = el("div", undefined, "home-list");
  for (const entry of snapshot.recent_activity.slice(0, activityShown)) activityList.append(activityNode(entry));
  if (snapshot.recent_activity.length > activityShown) {
    const more = button("Show more", () => { pane.activityShown = activityShown + ACTIVITY_STEP; showPresentation(viewId); persist(); }, "Reveal older entries");
    more.className = "home-more";
    activityList.append(more);
  }
  activity.append(activityList);
  const active = el("section", undefined, "home-tasks");
  active.append(el("h2", "Recently active tasks"));
  const taskList = el("div", undefined, "home-list");
  for (const entry of snapshot.recently_active_tasks.slice(0, tasksShown)) taskList.append(recentTaskNode(entry));
  if (snapshot.recently_active_tasks.length > tasksShown) {
    const more = button("Show more", () => { pane.tasksShown = tasksShown + TASKS_STEP; showPresentation(viewId); persist(); }, "Reveal older entries");
    more.className = "home-more";
    taskList.append(more);
  }
  active.append(taskList);
  surfaces.append(activity, active);
  // a Show more click focuses its button and the re-render removes it:
  // keep focus on the reader or the pane's keys go dead
  const hadFocus = reader.contains(document.activeElement);
  // the scroll guard spans the swap and its restore, so the transient
  // clamped scroll a re-render can fire is never taken for reading
  reader.pendingSignature = signature;
  reader.replaceChildren(surfaces);
  if (hadFocus) reader.focus({ preventScroll: true });
  reader.signature = signature;
  requestAnimationFrame(() => {
    if (isReading(viewId)) reader.scrollTop = presentation(viewId).scrollTop;
    reader.pendingSignature = null;
  });
}

function button(text, fn, title) {
  const node = el("button", text);
  node.type = "button";
  if (title) node.title = title;
  node.addEventListener("click", fn);
  return node;
}

function paneNode(viewId) {
  if (panes.has(viewId)) return panes.get(viewId);
  const node = el("section", undefined, "pane");
  node.dataset.pane = viewId;
  const header = el("header", undefined, "pane-header");
  const title = button("", () => { focusPane(viewId); openPicker("tasks"); }, "Find task, contribution, or local draft (Space Space)");
  title.className = "buffer-title";
  title.draggable = true;
  title.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/x-saccade-pane", String(viewId)));
  header.append(title);
  node.append(header, el("div", undefined, "document"));
  node.addEventListener("pointerdown", () => { bench.focus = viewId; refreshChrome(); });
  node.addEventListener("dragover", (event) => {
    if ([...event.dataTransfer.types].includes("text/x-saccade-pane")) event.preventDefault();
  });
  node.addEventListener("drop", (event) => {
    const origin = Number(event.dataTransfer.getData("text/x-saccade-pane"));
    if (bench.views.has(origin)) {
      event.preventDefault(); remember(origin); remember(viewId);
      bench.swapPanes(origin, viewId); renderLayout(); focusPane(origin);
    }
  });
  panes.set(viewId, node);
  return node;
}

function renderLayout() {
  for (const [id, view] of editors) {
    if (!bench.views.has(id)) { view.destroy(); editors.delete(id); readers.delete(id); panes.get(id)?.remove(); panes.delete(id); }
  }
  function build(tree) {
    if (tree.view) {
      const node = paneNode(tree.view);
      node.style.flex = "";
      mountEditor(tree.view, node);
      return node;
    }
    const node = el("div", undefined, `split ${tree.axis}`);
    const left = build(tree.children[0]), right = build(tree.children[1]);
    const divider = el("div", undefined, "divider");
    divider.setAttribute("role", "separator");
    divider.setAttribute("aria-label", "Resize panes");
    divider.setAttribute("aria-orientation", tree.axis === "row" ? "vertical" : "horizontal");
    divider.tabIndex = 0;
    function resize(ratio) {
      tree.ratio = Math.max(0.15, Math.min(0.85, ratio));
      left.style.flex = `${tree.ratio} 1 0`; right.style.flex = `${1 - tree.ratio} 1 0`;
    }
    resize(tree.ratio);
    divider.addEventListener("pointerdown", (event) => { divider.setPointerCapture(event.pointerId); event.preventDefault(); });
    divider.addEventListener("pointermove", (event) => {
      if (!divider.hasPointerCapture(event.pointerId)) return;
      const rect = node.getBoundingClientRect();
      resize(tree.axis === "row" ? (event.clientX - rect.left) / rect.width : (event.clientY - rect.top) / rect.height);
    });
    divider.addEventListener("pointerup", () => persist());
    divider.addEventListener("keydown", (event) => {
      if (["ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown"].includes(event.key)) {
        event.preventDefault(); resize(tree.ratio + (["ArrowLeft", "ArrowUp"].includes(event.key) ? -0.05 : 0.05)); persist();
      }
    });
    node.append(left, divider, right);
    return node;
  }
  // the rebuild re-parents the cached pane nodes, and a re-parented
  // element's scroll restarts at 0 (Chrome can also report the reset as
  // a scroll event, which would capture 0 over the captured position);
  // record each undisturbed pane's fresh DOM offset just before the move
  // and write it back in the same task, so neither a frame nor a capture
  // ever sees the reset. A pane whose reader or editor was replaced or
  // is mid-rebuild is skipped: its own render path owns the restore
  const moved = new Map();
  for (const [id] of panes) {
    const reader = readers.get(id), view = editors.get(id);
    if (reader && isReading(id)) {
      if (!reader.pendingSignature) moved.set(id, { reader, scroll: reader.scrollTop });
    } else if (view && !view.dom.hidden) {
      moved.set(id, { view, bufferId: view.bufferId, scroll: view.scrollDOM.scrollTop });
    }
  }
  $("#work").replaceChildren(build(bench.layout));
  for (const [id, move] of moved) {
    const reader = readers.get(id), view = editors.get(id);
    if (move.reader) {
      if (reader === move.reader && isReading(id) && !reader.pendingSignature && reader.scrollTop !== move.scroll) reader.scrollTop = move.scroll;
    } else if (view === move.view && view.bufferId === move.bufferId && !isReading(id) && view.scrollDOM.scrollTop !== move.scroll) {
      view.scrollDOM.scrollTop = move.scroll;
    }
  }
  refreshChrome();
  persist();
}

function jumpState(viewId) {
  if (!history.has(viewId)) history.set(viewId, { back: [], forward: [] });
  return history.get(viewId);
}

function jumpLocation(viewId) {
  return { bufferId: bench.view(viewId).bufferId, position: bench.position(viewId), presentation: ["home", "comment", "draft", "task"].includes(bench.buffer(bench.view(viewId).bufferId).kind) ? structuredClone(presentation(viewId)) : null };
}

function pushHistory(viewId) {
  const stack = jumpState(viewId);
  stack.back.push(jumpLocation(viewId));
  stack.forward.length = 0;
}

function openBuffer(id, viewId = bench.focus, preferred = "view", selected) {
  remember(viewId);
  if (!bench.buffers.has(id)) published(id);
  if (bench.view(viewId).bufferId !== id) pushHistory(viewId);
  bench.open(id, viewId);
  if (["comment", "draft", "task"].includes(bench.buffer(id).kind)) presentation(viewId).mode = preferred;
  // a passed selection assigns exactly - null clears, omitted remembers
  if (selected !== undefined) presentation(viewId).selected = selected;
  renderLayout(); focusPane(viewId);
  if (selected && bench.buffer(id).kind === "task") scrollCardIntoView(viewId, selected);
}

function scrollCardIntoView(viewId, id) {
  requestAnimationFrame(() => readers.get(viewId)?.querySelector(`[data-comment="${id}"]`)?.scrollIntoView({ block: "start" }));
}

function split(axis, id = bench.view().bufferId) {
  remember(bench.focus);
  const same = id === bench.view().bufferId && ["home", "comment", "draft", "task"].includes(current().kind) ? structuredClone(presentation()) : null;
  bench.split(axis, id);
  if (same) { presentation(); bench.view().presentations[id] = same; }
  renderLayout(); focusPane(bench.focus);
}

function closePane() {
  remember(bench.focus);
  if (!bench.closePane()) { status("The final pane remains open. Switch its buffer instead."); return; }
  renderLayout(); focusPane(bench.focus);
}

function onlyPane() {
  for (const id of bench.views.keys()) remember(id);
  bench.onlyPane(); renderLayout(); focusPane(bench.focus);
}

function neighbor(direction) {
  const currentRect = panes.get(bench.focus).getBoundingClientRect();
  const x = currentRect.x + currentRect.width / 2, y = currentRect.y + currentRect.height / 2;
  const options = [...panes].filter(([id]) => id !== bench.focus).map(([id, node]) => {
    const rect = node.getBoundingClientRect();
    return { id, dx: rect.x + rect.width / 2 - x, dy: rect.y + rect.height / 2 - y };
  }).filter((p) => ({ h: p.dx < 0, l: p.dx > 0, j: p.dy > 0, k: p.dy < 0 })[direction]);
  options.sort((a, b) => Math.hypot(a.dx, a.dy) - Math.hypot(b.dx, b.dy));
  if (options[0]) focusPane(options[0].id);
}

function jump(direction) {
  const stack = jumpState(bench.focus);
  const entry = stack[direction].pop();
  if (!entry) return status(`No ${direction === "back" ? "earlier" : "later"} location in this pane.`);
  remember(bench.focus);
  stack[direction === "back" ? "forward" : "back"].push(jumpLocation(bench.focus));
  bench.open(entry.bufferId); bench.remember(entry.position);
  if (entry.presentation) {
    presentation();
    bench.view().presentations[entry.bufferId] = structuredClone(entry.presentation);
  }
  renderLayout(); focusPane(bench.focus);
}

function back() { jump("back"); }
function forward() { jump("forward"); }

function backlinks() {
  const target = actionTarget();
  if (!target || target.kind !== "comment") return status("Open a contribution to inspect its explicit backlinks.");
  openPicker("backlinks", { sourceId: target.id });
}

function togglePresentation() {
  const id = bench.focus;
  const state = presentation(id);
  const reading = isReading(id);
  const editor = currentEditor(), reader = readers.get(id);
  const ratio = reader.scrollTop / Math.max(1, reader.scrollHeight - reader.clientHeight);
  if (reading) state.scrollTop = reader.scrollTop; else remember(id);
  if (!reading) {
    const selection = mode() === "Normal" ? editor.state.selection : null;
    Vim.handleKey(getCM(editor), "<Esc>");
    if (selection) editor.dispatch({ selection });
    state.sourceSeen = true;
  }
  state.mode = reading ? "source" : "view";
  showPresentation(id); focusPane(id);
  if (reading && !state.sourceSeen) {
    const initialSelection = editor.state.selection;
    requestAnimationFrame(() => {
      if (editors.get(id) !== editor || isReading(id) || editor.state.selection !== initialSelection) return;
      editor.scrollDOM.scrollTop = ratio * Math.max(0, editor.scrollDOM.scrollHeight - editor.scrollDOM.clientHeight);
      const at = editor.lineBlockAtHeight(editor.scrollDOM.scrollTop).from;
      editor.dispatch({ selection: { anchor: at } });
      remember(id); persist();
    });
  }
  if (reading) state.sourceSeen = true;
  persist();
  status(reading ? `Source: ${editable(current()) ? "editable" : "read-only"} Vim text. Space uv returns to View.`
    : threadFocused() ? "View: j/k selects the next or previous contribution; Tab or mouse follows links. Space uv opens Source."
    : "View: j/k scroll; Tab or mouse selects links. Space uv opens Source.");
}

// -- the surfaces that mutate -------------------------------------------------

function nextDraftId() { return `draft:${nextDraft++}`; }

function newNote() {
  const home = current().home ?? null;
  const id = nextDraftId();
  bench.register({ id, kind: "draft", title: `Note ${id.slice(6)}`, text: "", home, references: [], submission: null, authoredTitle: "", eventual: null, savedText: "" });
  openBuffer(id, bench.focus, "source");
}

function startRevision() {
  const buffer = actionTarget();
  if (!buffer || buffer.kind !== "comment" || !buffer.source) return;
  if (buffer.revisionPending) { status("This contribution already has a pending repair under review."); return; }
  const standing = buffer.source;
  buffer.revisionPending = {
    title: standing.title ?? null,
    savedTitle: standing.title ?? null,
    savedBody: standing.body,
    originalTitle: standing.title ?? null,
    originalBody: standing.body,
    submission: null,
  };
  buffer.text = standing.body;
  buffer.master = null;
  if (current().id !== buffer.id) openBuffer(buffer.id, bench.focus, "source");
  presentation().mode = "source";
  renderLayout(); focusPane(bench.focus);
  status(`Revising ${buffer.id}: edit Source, View previews the proposed repair, , p publishes it. , e again discards.`);
}

function discardRevision() {
  const buffer = current();
  if (buffer.kind !== "comment" || !buffer.revisionPending) return status("No pending repair on this buffer.");
  buffer.revisionPending = null;
  published(buffer.id);
  buffer.master = null;
  renderLayout(); focusPane(bench.focus);
  status(`Pending repair on ${buffer.id} discarded; the recorded contribution stands.`);
}

function standingOffer(buffer) {
  if (!buffer || buffer.kind !== "comment" || !buffer.source) return null;
  if (buffer.source.kind !== "offer" || buffer.source.withdrawn) return null;
  const home = task(buffer.source.home);
  if (!home || home.state !== "offered" || home.offer !== buffer.source.num) return null;
  return buffer.source;
}

function startAccept() {
  const buffer = actionTarget();
  const offer = standingOffer(buffer);
  if (!offer) return status("Accept applies to the standing offer of an offered task.");
  submitSurface(acceptSurface(buffer));
}

function startReject() {
  const buffer = actionTarget();
  const offer = standingOffer(buffer);
  if (!offer) return status("Reject applies to the standing offer of an offered task.");
  const existing = [...bench.buffers.values()].find((b) => b.kind === "draft" && b.eventual?.kind === "reject" && b.eventual.offer === offer.id);
  if (existing) { openBuffer(existing.id, bench.focus, "source"); status(`Grounds draft for ${offer.id} already open.`); return; }
  const id = nextDraftId();
  bench.register({ id, kind: "draft", title: `Grounds ${id.slice(6)}`, text: "", home: offer.home, references: [], submission: null, authoredTitle: "", eventual: { kind: "reject", offer: offer.id, offerSeq: offer.seq }, savedText: "" });
  openBuffer(id, bench.focus, "source");
  status(`Grounds for rejecting ${offer.id}: , p publishes the decision. Home and target are fixed.`);
}

// -- task creation and incarnation launch: separate explicit acts ---------

// the staged-choice law these two commands share with publication: a
// prompt or picker Enter stages the value and returns; the explicit
// command, run again, is the only thing that ever acts
function newTask() {
  if (stagedTask !== null) return createTask(stagedTask);
  if (!identity.actor && !claimedName) {
    openPicker("name", { after: () => status(`Name recorded: ${claimedName}. Run New task again to create.`) });
    return;
  }
  openPicker("task-name");
}

async function sendCreate(name, actor) {
  const request = { actor, command: createTaskCommand(name) };
  const response = await fetch("/api/v1/workbench/command", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
  });
  let data;
  try { data = await response.json(); } catch { throw new Error("The response could not be read."); }
  if (!response.ok) {
    throw Object.assign(new Error("refused"), { refused: true, detail: typeof data.error === "string" ? data.error : `${data.error?.code}: ${data.error?.detail ?? ""}` });
  }
  const records = data.records;
  if (!Array.isArray(records) || records.length !== 1 || !confirmsTaskCreated(records[0], name) || records[0].actor !== actor || records[0].tier !== "human") {
    throw new Error("The response did not confirm the exact expected record.");
  }
  return { record: records[0], id: typeof data.id === "string" ? data.id : null };
}

async function createTask(name) {
  if (submitting) return status("A submission is already in flight.");
  stagedTask = null;
  submitting = true;
  status("Creating once...");
  let finalMessage = "Creating once...";
  try {
    const { record, id } = await sendCreate(name, identity.actor ?? claimedName);
    finalMessage = `Created task (birth record ${record.seq}).`;
    if (id) {
      finalMessage = `Created ${id}: ${name}.`;
      try {
        await reload(false);
        if (snapshot.tasks.some((item) => `t-${item.id}` === id)) openBuffer(id, bench.focus, "view");
        else finalMessage = `Created task (birth record ${record.seq}). Refresh unavailable; the thread opens from the finder.`;
      } catch {
        finalMessage = `Created ${id}: ${name}. Refresh unavailable; the thread opens from the finder.`;
      }
    }
  } catch (error) {
    finalMessage = error.refused
      ? `Not created: ${error.detail} Nothing is staged; run New task to author again.`
      : `Creation outcome unknown: ${error.message} Inspect the board before creating again; nothing is staged.`;
  } finally {
    submitting = false;
    persist(); renderLayout(); focusPane(bench.focus);
    // the status is the last act in every submission: a release signal
    // that fires before the panes have been rebuilt would let the next
    // keystrokes land in detached DOM
    status(finalMessage);
  }
}

function startLaunch() {
  if (stagedLaunch) return submitLaunch(stagedLaunch);
  openPicker("workspace");
}

// a reply that does not name its own facts is an unknown outcome, not a
// confirmed submission; the recorded-material note belongs to the flow
// that actually changed nothing
function submittedMessage(data, recordedNote) {
  if (data?.submitted === true && typeof data.terminal === "string" && typeof data.workspace === "string") {
    return `Submitted to the host: terminal ${data.terminal} at ${data.workspace}. The session registers itself when it starts.${recordedNote ? " Nothing recorded changed." : ""}`;
  }
  return "Submission outcome unknown: the reply did not name the submitted facts. The host may have accepted the launch; inspect the host's terminals. Nothing is staged.";
}

function launchOutcomeMessage(error) {
  const outcome = launchOutcome(error);
  if (outcome.kind === "unknown") {
    return `Launch outcome unknown: ${outcome.detail} The host may have accepted the submission; inspect the host's terminals before another attempt. Nothing is staged.`;
  }
  if (outcome.code === "launch_host_no_handle") {
    return `The host created a terminal but named no handle: ${outcome.detail} The launch went through; inspect the host's terminals. Nothing is staged.`;
  }
  return `Launch not submitted: ${outcome.code}: ${outcome.detail} Nothing is staged; run Launch to choose again.`;
}

function provisionOutcomeMessage(error) {
  const outcome = launchOutcome(error);
  if (outcome.kind === "unknown") {
    return `Provisioning outcome unknown: ${outcome.detail} The server may have provisioned a workspace; inspect the board before provisioning again. The launch was not submitted. Nothing is staged.`;
  }
  if (outcome.code === "storage") {
    // the door's own partial fact: the effects ran and the detail says
    // exactly what stands
    return `Provisioning outcome unclear: ${outcome.detail} The launch was not submitted. Nothing is staged.`;
  }
  return `Not provisioned: ${outcome.code}: ${outcome.detail} The launch was not submitted. Nothing is staged; run Launch to choose again.`;
}

async function submitLaunch(staged) {
  if (submitting) return status("A submission is already in flight.");
  stagedLaunch = null;
  submitting = true;
  status("Submitting to the host once...");
  let finalMessage = "Submitting to the host once...";
  try {
    if (staged.new) {
      let created = null;
      try {
        const reply = await api("/api/v1/workspace/create", { label: staged.new.label, client: "workbench" });
        if (typeof reply?.workspace?.path !== "string" || typeof reply?.workspace?.seq !== "number") {
          throw new Error("The reply did not name the provisioned workspace.");
        }
        // created only ever holds a validated reply: a misshapen one
        // fails the dependent launch closed, it never launches undefined
        created = reply;
      } catch (error) {
        finalMessage = provisionOutcomeMessage(error);
      }
      if (created) {
        const provisioned = `Workspace provisioned - record ${created.workspace.seq}, branch ${created.workspace.branch} at ${created.workspace.path}.`;
        try { await reload(false); } catch {}
        try {
          const data = await api("/api/v1/workbench/launch", launchRequest(created.workspace.path));
          finalMessage = `${provisioned} ${submittedMessage(data, false)}`;
        } catch (error) {
          finalMessage = `${provisioned} ${launchOutcomeMessage(error)}`;
        }
      }
    } else {
      try {
        const data = await api("/api/v1/workbench/launch", launchRequest(staged.existing));
        finalMessage = submittedMessage(data, true);
      } catch (error) {
        finalMessage = launchOutcomeMessage(error);
      }
    }
  } finally {
    submitting = false;
    persist(); renderLayout(); focusPane(bench.focus);
    status(finalMessage);
  }
}

// the title target behind a prompt: note drafts author their own title,
// note and offer repairs revise the recorded one, everything else has none
function titleTarget(buffer) {
  if (buffer.revisionPending) {
    if (!["note", "offer"].includes(buffer.source.kind)) return null;
    return {
      bufferId: buffer.id,
      get: () => buffer.revisionPending.title ?? "",
      set: (value) => { buffer.revisionPending.title = value; },
    };
  }
  if (buffer.kind === "draft" && buffer.eventual?.kind !== "reject") {
    return {
      bufferId: buffer.id,
      get: () => buffer.authoredTitle ?? "",
      set: (value) => { buffer.authoredTitle = value; },
    };
  }
  return null;
}

function editTitle() {
  const target = titleTarget(current());
  if (!target) return status("Titles belong to note drafts and note or offer repairs.");
  openPicker("title", { target });
}

function draftSurface(buffer) {
  if (buffer.revisionPending) {
    const pending = buffer.revisionPending;
    return {
      kind: "revision", buffer,
      title: ["note", "offer"].includes(buffer.source.kind) ? pending.title : null,
      body: buffer.text,
      references: buffer.source.references,
      home: buffer.source.home,
      target: buffer.id,
      // the diff baseline rides the surface, so a pending discarded or
      // restarted mid-flight cannot redirect the confirmation
      originalTitle: pending.originalTitle,
      originalBody: pending.originalBody,
      get submission() { return buffer.revisionPending?.submission; },
      set submission(value) { if (buffer.revisionPending) buffer.revisionPending.submission = value; },
    };
  }
  if (buffer.kind !== "draft") return null;
  if (buffer.eventual?.kind === "reject") {
    return {
      kind: "reject", buffer,
      title: null, body: buffer.text,
      references: [], home: buffer.eventual.offerSeq === undefined ? buffer.home : comment(buffer.eventual.offer)?.home,
      target: buffer.eventual.offer,
      get submission() { return buffer.submission; },
      set submission(value) { buffer.submission = value; },
    };
  }
  return {
    kind: "note", buffer,
    title: buffer.authoredTitle, body: buffer.text,
    references: buffer.references, home: buffer.home,
    target: null,
    get submission() { return buffer.submission; },
    set submission(value) { buffer.submission = value; },
  };
}

function acceptSurface(buffer) {
  const offer = standingOffer(buffer);
  if (!offer) return null;
  buffer.decision ??= {};
  return {
    kind: "accept", buffer, title: null, body: null, references: null,
    home: offer.home, target: offer.id,
    get submission() { return buffer.decision; },
    set submission(value) { buffer.decision = value; },
  };
}

// -- submission: one explicit door, missing fields prompt and return ------

function publishDraft() {
  const surface = draftSurface(current());
  if (!surface) return status("Nothing publishable is focused: open a draft or repair.");
  submitSurface(surface);
}

// missing information prompts for itself and returns to the origin pane;
// the explicit command, run again, is the only thing that ever submits
function submitSurface(surface) {
  if (submitting) return status("A submission is already in flight.");
  const state = surface.submission?.kind;
  if (state === "landed" || state === "unknown") return status(surface.submission.message);
  if (!identity.actor && !claimedName) {
    openPicker("name", { after: () => status(`Name recorded: ${claimedName}. Run the command again to submit.`) });
    return;
  }
  if (surface.kind === "note" && surface.home === null) {
    openPicker("home", { after: () => status(`Home task set: t-${surface.buffer.home}. Publish again to submit.`) });
    return;
  }
  if (surface.kind === "note" && !surface.title) {
    openPicker("title", { target: titleTarget(surface.buffer), after: () => status("Title authored. Publish again to submit.") });
    return;
  }
  if (surface.kind === "reject" && !surface.body.trim()) {
    return status("Grounds are required: write them in the focused draft, then publish again.");
  }
  submitConfirmed(surface, identity.actor ?? claimedName);
}

async function send(surface, actor) {
  const request = { actor, command: commandFor(surface, actor) };
  const response = await fetch("/api/v1/workbench/command", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
  });
  let data;
  try { data = await response.json(); } catch { throw new Error("The response could not be read."); }
  if (!response.ok) {
    throw Object.assign(new Error("refused"), { refused: true, detail: typeof data.error === "string" ? data.error : `${data.error?.code}: ${data.error?.detail ?? ""}` });
  }
  const records = data.records;
  if (!Array.isArray(records) || records.length !== 1 || !confirmsLanded(surface, records[0]) || records[0].actor !== actor || records[0].tier !== "human") {
    throw new Error("The response did not confirm the exact expected record.");
  }
  return records[0];
}

async function submitConfirmed(surface, actor) {
  submitting = true;
  surface.submission = { kind: "unknown", message: "Submission was initiated but no final answer is confirmed. Inspect the task record before another attempt." };
  // the bytes that went on the wire and the pending they belonged to: the
  // author may keep working while the answer is outstanding
  const pending = surface.kind === "revision" ? surface.buffer.revisionPending : null;
  const sentBody = surface.body;
  const sentTitle = surface.title;
  let finalMessage = "Submitting once...";
  if (!persist()) {
    submitting = false;
    surface.submission = { kind: "blocked", message: "Submission not attempted: local submission state could not be saved." };
    status(surface.submission.message);
    refreshChrome();
    return;
  }
  status(finalMessage);
  try {
    const record = await send(surface, actor);
    let message = `Recorded at birth record ${record.seq}.`;
    let fresh = true;
    try {
      await reload(false);
      const born = surface.kind === "revision" ? surface.buffer.source : birth(record.seq);
      if (born) message = `Recorded as ${surface.kind === "revision" ? surface.buffer.id : born.id}.`;
    } catch { fresh = false; message += " Refresh unavailable."; }
    if (surface.kind !== "revision") {
      surface.submission = { kind: "landed", seq: record.seq, message };
      finalMessage = message;
    } else {
      const buffer = surface.buffer;
      if (buffer.revisionPending !== pending) {
        // discarded or restarted while outstanding: the record stands and
        // the buffer's current pending is not this surface's business
        finalMessage = `${message} The local repair changed while submitting; the record stands.`;
      } else if (!fresh) {
        // the landed bytes cannot be seen: nothing is discarded, no retry
        pending.submission = { kind: "landed", seq: record.seq, message };
        finalMessage = message;
      } else if (buffer.text === sentBody && (pending.title ?? null) === (sentTitle ?? null)) {
        // the sent bytes are still the whole repair: the repair is done
        pending.submission = { kind: "landed", seq: record.seq, message };
        finalMessage = message;
        buffer.revisionPending = null;
        published(buffer.id);
        buffer.master = null;
      } else {
        // local work arrived while sending: the repair continues, rebased
        // on the recorded bytes. A field the author never touched adopts
        // the record into both value and baseline - advancing a baseline
        // alone would manufacture an edit to an untouched field on the
        // next publish; a touched field keeps the author's value and
        // advances only its baseline
        const recorded = buffer.source;
        if ((pending.title ?? null) === (sentTitle ?? null)) pending.title = recorded.title ?? null;
        if (buffer.text === sentBody && recorded.body !== buffer.text) {
          buffer.text = recorded.body;
          buffer.master = null;
        }
        pending.originalTitle = recorded.title ?? null;
        pending.originalBody = recorded.body;
        pending.savedTitle = pending.originalTitle;
        pending.savedBody = pending.originalBody;
        pending.submission = null;
        finalMessage = `${message} Local work arrived while sending; the repair continues on the recorded bytes.`;
      }
    }
  } catch (error) {
    let outcome;
    if (error.refused) {
      outcome = { kind: "refused", message: `Not sent: ${error.detail}. The draft is retained for correction.` };
    } else {
      outcome = { kind: "unknown", message: `Submission outcome unknown: ${error.message} No retry will be sent; inspect the task record.` };
    }
    if (surface.kind === "revision" && surface.buffer.revisionPending !== pending) {
      // the pending changed in flight: the outcome names no live surface
      finalMessage = outcome.message;
    } else {
      surface.submission = outcome;
      finalMessage = outcome.message;
    }
  } finally {
    submitting = false;
    persist(); renderLayout(); focusPane(bench.focus);
    status(finalMessage);
  }
}

function persist() {
  if (!storageKey) return false;
  try {
    const buffers = [...bench.buffers.values()].map((buffer) => ({
      id: buffer.id, kind: buffer.kind,
      ...(buffer.kind === "draft" ? { text: buffer.text, references: buffer.references, home: buffer.home, submission: buffer.submission, authoredTitle: buffer.authoredTitle, eventual: buffer.eventual, savedText: buffer.savedText } : {}),
      ...(buffer.revisionPending ? { text: buffer.text, revisionPending: buffer.revisionPending } : {}),
      ...(buffer.decision ? { decision: buffer.decision } : {}),
      lastPosition: buffer.lastPosition, inline: buffer.inline,
    }));
    localStorage.setItem(storageKey, JSON.stringify({ buffers, layout: bench.layout, focus: bench.focus,
      nextView: bench.nextView, nextDraft, views: [...bench.views.values()].map((view) => ({ ...view, positions: [...view.positions] })) }));
    return true;
  } catch { status("Local save failed. Open drafts remain in memory; do not close this browser."); return false; }
}

function migrateDraftShape(buffer) {
  if (buffer.kind !== "draft") return;
  buffer.authoredTitle ??= "";
  buffer.savedText ??= "";
  buffer.eventual ??= null;
}

function restoreLegacy() {
  if (!identity?.legacy_instance) return false;
  const raw = localStorage.getItem(`saccade-web-vim:${identity.legacy_instance}`);
  if (!raw) return false;
  try {
    const data = JSON.parse(raw);
    for (const buffer of data.buffers ?? []) {
      if (buffer.kind === "draft") {
        // the prototype's generated display name stays display-only
        bench.register({ id: buffer.id, kind: "draft", title: buffer.title, text: buffer.text,
          references: buffer.references ?? [], home: buffer.home, submission: buffer.submission ?? null,
          authoredTitle: "", eventual: null, savedText: buffer.text });
      } else {
        published(buffer.id);
      }
      const restored = bench.buffers.get(buffer.id);
      if (!restored) continue;
      restored.lastPosition = buffer.lastPosition;
      restored.inline = buffer.inline;
    }
    bench.layout = data.layout ?? null;
    bench.focus = data.focus ?? null;
    bench.nextView = Math.max(1, data.nextView ?? 1);
    nextDraft = data.nextDraft ?? 1;
    for (const view of data.views ?? []) {
      bench.views.set(view.id, { ...view, positions: new Map(view.positions) });
    }
    return bench.views.size > 0;
  } catch { return false; }
}

function restore() {
  const raw = localStorage.getItem(storageKey);
  if (!raw) return false;
  try {
    const data = JSON.parse(raw);
    for (const buffer of data.buffers) {
      const restored = buffer.kind === "draft" ? bench.register(buffer) : published(buffer.id);
      migrateDraftShape(restored);
      restored.lastPosition = buffer.lastPosition;
      restored.inline = buffer.inline;
      if (buffer.revisionPending) {
        restored.text = buffer.text;
        restored.revisionPending = buffer.revisionPending;
      }
      if (buffer.decision) restored.decision = buffer.decision;
    }
    bench.layout = data.layout;
    bench.focus = data.focus;
    bench.nextView = data.nextView;
    nextDraft = data.nextDraft;
    bench.views = new Map(data.views.map((view) => [view.id, { ...view, positions: new Map(view.positions) }]));
    return bench.views.has(bench.focus);
  } catch { bench.views.clear(); bench.layout = null; bench.focus = null; return false; }
}

// reconcile one fetched snapshot into the open surfaces, synchronously:
// recorded buffers refresh from the fold, local drafts keep their text,
// pending repairs keep their text, title, and original baseline while
// the recorded source advances beneath them, and selection, cursor,
// scroll, focus, pane layout, open pickers, and unknown or in-flight
// submissions are all state this never touches. It never navigates.
function absorb(fresh) {
  snapshot = fresh;
  for (const buffer of bench.buffers.values()) {
    if (buffer.kind === "draft") continue;
    const pending = buffer.revisionPending;
    const pendingText = pending ? buffer.text : undefined;
    published(buffer.id);
    if (pending && !buffer.revisionPending) {
      buffer.revisionPending = pending;
      buffer.text = pendingText;
    }
    for (const [id, view] of editors) {
      if (bench.view(id).bufferId !== buffer.id) continue;
      if (buffer.revisionPending) continue;
      // a view whose own document is already the published text is left
      // alone: setState on an unchanged document resets its scroll, and
      // another pane's update advancing the shared master says nothing
      // about this view
      if (view.state.doc.toString() === buffer.text) continue;
      const scroll = view.scrollDOM.scrollTop;
      view.setState(editorState(buffer, bench.position(id)));
      observeMode(view);
      view.scrollDOM.scrollTop = scroll;
    }
  }
  for (const id of editors.keys()) {
    if (["comment", "home", "task"].includes(bench.buffer(bench.view(id).bufferId).kind) && isReading(id)) showPresentation(id);
  }
  refreshChrome();
}

// a delayed response carrying an older fold is superseded by what
// already stands on screen: absorbing it would roll back both the
// rendered record and the cursor and hide newer facts. The cursor
// orders positions on the one append-only log - an empty fold's null
// is the position before any record, older than every held position -
// so anything strictly older is dropped.
function absorbIfCurrent(fresh) {
  const standing = typeof syncCursor === "number" ? syncCursor : null;
  const arriving = typeof fresh.cursor === "number" ? fresh.cursor : null;
  if (standing !== null && (arriving === null || arriving < standing)) return false;
  syncCursor = fresh.cursor;
  absorb(fresh);
  return true;
}

async function reload(announce = true) {
  const fresh = await api("/api/v1/workbench");
  absorbIfCurrent(fresh);
  if (announce) status("Recorded material refreshed. Draft references were not retargeted.");
}

// the read-side follow: one parked read at a time, held at most wait_ms
// and woken early by the record's landing through the same door the
// mailbox read uses; the server re-checks its cursor on a bounded slice
// so records written by other processes surface too. No subscription is
// kept on the tracker. A bounded pause after a failed request is
// reconnect transport; the stale cursor then mismatches and the fresh
// fold returns whole, so a dropped connection catches up from the
// record, not from memory.
const followPause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function followRecords() {
  for (;;) {
    try {
      const query = syncCursor === null || syncCursor === undefined
        ? "?wait_ms=25000" : `?cursor=${syncCursor}&wait_ms=25000`;
      const fresh = await api(`/api/v1/workbench${query}`);
      absorbIfCurrent(fresh);
    } catch {
      await followPause(2000);
    }
  }
}

// -- the command registry ------------------------------------------------------

function revisableContext() {
  const target = actionTarget();
  return !!target?.source && !target.source.withdrawn
    && ["note", "offer", "reject", "use"].includes(target.source.kind);
}

const registry = [
  { id: "tasks", label: "Find task, contribution, or local draft", binding: " ", run: () => openPicker("tasks") },
  { id: "contents", label: "Search recorded content", binding: "sg", run: () => openPicker("contents") },
  { id: "new", label: "New note", binding: "nn", run: newNote },
  { id: "task-new", label: "New task", binding: "nt", run: newTask },
  { id: "launch", label: "Launch a session in a workspace", binding: "nl", run: startLaunch },
  { id: "cite", label: "Insert citation", binding: "ci", local: "i", when: () => current().kind === "draft" && current().eventual?.kind !== "reject" && !isReading(), run: () => openPicker("cite") },
  { id: "home-task", label: "Set draft home task", binding: "nh", local: "h", when: () => current().kind === "draft" && current().eventual?.kind !== "reject", run: () => openPicker("home") },
  { id: "publish", label: "Publish the focused draft or repair", local: "p", when: () => current().kind === "draft" || !!current().revisionPending, run: publishDraft },
  { id: "revise", label: "Revise this contribution", local: "e", when: () => revisableContext() && !actionTarget()?.revisionPending, run: startRevision },
  { id: "discard-revision", label: "Discard pending repair", local: "e", when: () => !!current().revisionPending, run: discardRevision },
  { id: "accept", label: "Accept the standing offer", local: "a", when: () => !!standingOffer(actionTarget()), run: startAccept },
  { id: "reject", label: "Reject the standing offer with grounds", local: "x", when: () => !!standingOffer(actionTarget()), run: startReject },
  { id: "title", label: "Title the focused draft or repair", local: "t", when: () => current().kind === "draft" || !!current().revisionPending, run: editTitle },
  { id: "write", aliases: ["w"], label: "Save draft locally", when: () => current().kind === "draft" || current().revisionPending, run: () => { const b = current(); const prior = { savedText: b.savedText, savedTitle: b.revisionPending?.savedTitle, savedBody: b.revisionPending?.savedBody }; if (b.kind === "draft") b.savedText = b.text; if (b.revisionPending) { b.revisionPending.savedTitle = b.revisionPending.title; b.revisionPending.savedBody = b.text; } if (persist()) { refreshChrome(); status("Saved locally in this browser. Nothing sent."); } else { if (b.kind === "draft") b.savedText = prior.savedText; if (b.revisionPending) { b.revisionPending.savedTitle = prior.savedTitle; b.revisionPending.savedBody = prior.savedBody; } refreshChrome(); } } },
  { id: "undo", label: "Undo edit", when: () => editable(current()) && !isReading(), run: () => Vim.handleKey(getCM(currentEditor()), "u") },
  { id: "redo", label: "Redo edit", when: () => editable(current()) && !isReading(), run: () => Vim.handleKey(getCM(currentEditor()), "<C-r>") },
  { id: "fold-toggle", label: "Toggle Markdown section fold (za)", when: sourceFoldable, run: () => foldHere("a") },
  { id: "fold-open", label: "Open Markdown section fold (zo)", when: sourceFoldable, run: () => foldHere("o") },
  { id: "fold-close", label: "Close Markdown section fold (zc)", when: sourceFoldable, run: () => foldHere("c") },
  { id: "fold-close-all", label: "Close all Markdown sections (zM)", when: sourceFoldable, run: () => foldHere("M") },
  { id: "fold-open-all", label: "Open all Markdown sections (zR)", when: sourceFoldable, run: () => foldHere("R") },
  { id: "reader-down", label: "Scroll rendered pane down", when: () => isReading(), run: () => scrollReader(56) },
  { id: "reader-up", label: "Scroll rendered pane up", when: () => isReading(), run: () => scrollReader(-56) },
  { id: "reader-top", label: "Scroll rendered pane to top", when: () => isReading(), run: () => scrollReader("top") },
  { id: "reader-bottom", label: "Scroll rendered pane to bottom", when: () => isReading(), run: () => scrollReader("bottom") },
  { id: "thread-next", label: "Select next contribution in the thread", when: threadFocused, run: () => selectCard(1) },
  { id: "thread-prev", label: "Select previous contribution in the thread", when: threadFocused, run: () => selectCard(-1) },
  { id: "focus-h", label: "Focus pane left", when: () => bench.views.size > 1, run: () => focusLeaf(-1) },
  { id: "focus-l", label: "Focus pane right", when: () => bench.views.size > 1, run: () => focusLeaf(1) },
  { id: "focus-k", label: "Focus pane above", when: () => bench.views.size > 1, run: () => focusLeaf(-1) },
  { id: "focus-j", label: "Focus pane below", when: () => bench.views.size > 1, run: () => focusLeaf(1) },
  { id: "split-row", aliases: ["vsplit", "vs"], label: "vsplit - split pane beside", binding: "wv", run: () => split("row") },
  { id: "split-column", aliases: ["split", "sp"], label: "split - split pane below", binding: "ws", run: () => split("column") },
  { id: "close", aliases: ["q"], label: "Close pane", binding: "wc", run: closePane },
  { id: "only", label: "Show only current pane", binding: "wo", run: onlyPane },
  ...["h", "j", "k", "l"].map((d) => ({ id: `focus-${d}`, binding: `w${d}`, label: `Focus pane ${{ h: "left", j: "below", k: "above", l: "right" }[d]}`, run: () => neighbor(d) })),
  { id: "follow", label: "Follow id or reference under cursor", binding: "gd", when: () => !isReading() && !!currentEditor(), run: follow },
  { id: "backlinks", label: "Show explicit backlinks", binding: "gr", when: () => actionTarget()?.kind === "comment", run: backlinks },
  { id: "rendering", label: "Toggle View / Source", binding: "uv", when: () => ["home", "comment", "draft", "task"].includes(current().kind), run: togglePresentation },
  { id: "back", label: "Jump backward", binding: "jb", run: back },
  { id: "forward", label: "Jump forward", run: forward },
  { id: "home", label: "Open home buffer", binding: "hh", run: () => openBuffer("home") },
  { id: "reload", label: "Refresh recorded material", run: reload },
];

function commands() {
  return registry.filter((item) => item.when === undefined || item.when());
}

function sourceFoldable() {
  return !isReading() && !!currentEditor();
}

function foldHere(action) {
  if (!sectionFold(currentEditor(), action)) status("No Markdown section here.");
}

function scrollReader(amount) {
  const reader = readers.get(bench.focus);
  if (!reader) return;
  reader.scrollTop = amount === "top" ? 0 : amount === "bottom" ? reader.scrollHeight
    : reader.scrollTop + amount;
  presentation().scrollTop = reader.scrollTop;
  persist();
}

// j and k step the thread selection, clamped at the ends; from no
// selection either key lands on the first card
function selectCard(step) {
  if (!threadFocused()) return;
  const cards = snapshot.comments.filter((c) => c.home === current().home);
  if (!cards.length) return status("This thread has no contributions to select.");
  const at = cards.findIndex((c) => c.id === presentation().selected);
  const index = at === -1 ? 0 : Math.max(0, Math.min(cards.length - 1, at + step));
  if (cards[index].id === presentation().selected) return;
  presentation().selected = cards[index].id;
  const reader = readers.get(bench.focus);
  reader.querySelectorAll(".thread-card.selected").forEach((node) => node.classList.remove("selected"));
  const card = reader.querySelector(`[data-comment="${cards[index].id}"]`);
  card.classList.add("selected");
  card.scrollIntoView({ block: "nearest" });
  persist(); refreshChrome();
}

function focusLeaf(step) {
  const order = leaves(bench.layout);
  const at = order.indexOf(bench.focus);
  const target = order[at + step];
  if (!target) return status("No pane that way.");
  focusPane(target);
}

function runCommand(id) {
  pendingWindow = false; pendingZ = false;
  const command = commands().find((item) => item.id === id);
  if (!command) return status("That command is not available in this buffer.");
  Promise.resolve(command.run()).catch((error) => status(error.message));
}

function showLeader(prefix) {
  const scope = leader === "," ? "," : "Space ";
  const options = commands().flatMap((command) => {
    const entries = [];
    if (leader !== "," && command.binding && command.binding.startsWith(prefix) && prefix !== "") entries.push([command.binding, command]);
    if (leader !== "," && command.binding === " " && prefix === "") entries.push(["Space", command]);
    if (leader === "," && command.local && prefix === "") entries.push([`, ${command.local}`, command]);
    return entries;
  });
  $("#leader").textContent = `${scope}${prefix}  |  ${options.map(([keys, command]) => `${keys}: ${command.label}`).join("  |  ")}`;
  $("#leader").hidden = false;
}

function cancelLeader() { leader = null; $("#leader").hidden = true; }

function candidateComments(query) {
  const exact = /^c-\d+$/.test(query.trim());
  return exact ? snapshot.comments.filter((item) => item.id === query.trim()) : snapshot.comments.filter((item) => !item.withdrawn);
}

function candidates() {
  const query = $("#query").value;
  let list, selector;
  if (picker.kind === "commands") {
    list = commands();
    const exact = list.find((item) => item.id === query || item.aliases?.includes(query));
    if (exact) return [exact];
    selector = (item) => `${item.label} ${item.id}`;
  } else if (picker.kind === "title" || picker.kind === "name" || picker.kind === "task-name" || picker.kind === "launch-label" || picker.kind === "workspace-path") {
    return [];
  } else if (picker.kind === "workspace") {
    list = [{ finderType: "workspace-new" }, { finderType: "workspace-path" }, ...snapshot.workspaces.map((item) => ({ ...item, finderType: "workspace" }))];
    selector = (item) => item.finderType === "workspace-new" ? "Provision a new workspace"
      : item.finderType === "workspace-path" ? "Name an existing directory"
      : `${item.path} ${item.branch}`;
  } else if (["tasks", "home"].includes(picker.kind)) {
    list = snapshot.tasks;
    if (/^t-\d+$/.test(query.trim())) return list.filter((item) => `t-${item.id}` === query.trim()).map((item) => ({ ...item, finderType: "task" }));
    list = list.map((item) => ({ ...item, finderType: "task" }));
    if (picker.kind === "tasks") {
      list.push(...[...bench.buffers.values()].filter((buffer) => buffer.kind === "draft").map((buffer) => ({ ...buffer, finderType: "draft" })));
      list.push(...snapshot.comments.filter((item) => ["note", "offer"].includes(item.kind) && !item.withdrawn).map((item) => ({ ...item, finderType: "comment" })));
    }
    selector = (item) => item.finderType === "draft" ? `${item.text} ${task(item.home)?.name ?? ""}`
      : item.finderType === "comment" ? `${item.title ?? ""} ${item.body}`
      : item.name;
  } else if (picker.kind === "backlinks") {
    list = (comment(picker.sourceId)?.cited_by ?? []).map((num) => comment(`c-${num}`));
    selector = (item) => item.body || item.withdrawal_note || item.kind;
  } else {
    list = candidateComments(query);
    if (/^c-\d+$/.test(query.trim())) return list;
    selector = (item) => `${item.title ?? ""} ${item.body}`;
  }
  return query ? new Fzf(list, { selector }).find(query).map((entry) => entry.item) : list;
}

function excerpt(text) {
  const line = text.split(/\r?\n/).find((part) => part.trim()) ?? "";
  if (line.length <= 160) return line;
  const end = line.lastIndexOf(" ", 160);
  return `${line.slice(0, end > 0 ? end : 160)}...`;
}

function candidateLabel(item) {
  if (picker.kind === "commands") return item.label;
  if (picker.kind === "workspace") {
    return item.finderType === "workspace-new"
      ? "Provision a new workspace\nThe server cuts a fresh worktree and branch, then the launch is submitted there"
      : item.finderType === "workspace-path"
        ? "Name an existing directory\nType an existing directory on the host; the launch validates it"
        : `Workspace record ${item.seq}  ${item.branch}\n${item.path}`;
  }
  if (["tasks", "home"].includes(picker.kind)) {
    if (item.finderType === "task") return `Task t-${item.id}  ${item.name}`;
    if (item.finderType === "comment") return `${kindName(item.kind)} ${item.id}  ${item.title}\nt-${item.home} ${task(item.home)?.name ?? ""}  ${item.actor}`;
    const label = item.submission?.kind === "landed" ? "Recorded local copy" : item.submission?.kind === "unknown" ? "Local draft | submission unconfirmed" : "Local draft";
    return `${label} | ${item.home === null ? "no home selected" : `t-${item.home} ${task(item.home)?.name ?? ""}`}\n${excerpt(item.text) || "(empty draft)"}`;
  }
  return `${item.id}  ${item.actor} | t-${item.home} ${task(item.home)?.name ?? ""}\n${excerpt(item.title ? `${item.title} - ${item.body || item.withdrawal_note}` : item.withdrawn ? item.withdrawal_note : item.body)}`;
}

function renderPicker() {
  picker.items = candidates();
  picker.selected = Math.max(0, Math.min(picker.selected, picker.items.length - 1));
  $("#results").replaceChildren();
  picker.items.forEach((item, index) => {
    const node = el("button", candidateLabel(item), index === picker.selected ? "selected" : "");
    node.type = "button";
    node.setAttribute("role", "option");
    node.setAttribute("aria-selected", String(index === picker.selected));
    node.addEventListener("click", () => { picker.selected = index; choose(false); });
    node.addEventListener("mouseenter", () => { picker.selected = index; highlightPicker(); });
    $("#results").append(node);
  });
  highlightPicker();
}

function highlightPicker() {
  const nodes = [...$("#results").children];
  nodes.forEach((node, i) => { node.classList.toggle("selected", i === picker.selected); node.setAttribute("aria-selected", String(i === picker.selected)); });
  nodes[picker.selected]?.scrollIntoView({ block: "nearest" });
  const item = picker.items[picker.selected];
  const generation = ++pickerPreviewGeneration;
  const activePicker = picker;
  const host = $("#picker-preview");
  host.replaceChildren();
  if (!item) { host.textContent = "No matching material."; return; }
  if (["cite", "contents", "backlinks"].includes(picker.kind)) {
    host.append(el("pre", `${item.id}  ${item.kind}\nhome: t-${item.home} ${task(item.home)?.name ?? ""}\nby: ${item.actor}/${item.tier}\nevent_time: ${time(item.born_at)}${item.revised ? "\npresentation: revised" : ""}${item.withdrawn ? "\npresentation: withdrawn" : ""}${item.title ? `\ntitle: ${item.title}` : ""}`));
    const body = el("div", undefined, "rich"); body.innerHTML = item.html;
    host.append(body);
  } else if (["tasks", "home"].includes(picker.kind) && item.finderType === "comment") {
    host.append(el("pre", `${item.id}  ${item.kind}\nt-${item.home} ${task(item.home)?.name ?? ""}  by ${item.actor}`));
    const body = el("div", undefined, "rich"); body.innerHTML = item.html; host.append(body);
  } else if (item.finderType === "draft") {
    host.append(el("pre", `Local draft\nhome: ${item.home === null ? "not selected" : `t-${item.home} ${task(item.home)?.name ?? ""}`}${item.authoredTitle ? `\ntitle: ${item.authoredTitle}` : ""}`));
    const body = el("div", item.text, "rich"); host.append(body);
    api("/api/v1/workbench/preview", { body: item.text, references: item.references }).then((result) => {
      if (picker === activePicker && pickerPreviewGeneration === generation) body.innerHTML = result.html;
    }).catch(() => {});
  } else if (["tasks", "home"].includes(picker.kind)) {
    host.append(el("pre", `t-${item.id}  ${item.name}\nstate: ${item.state}`));
    for (const source of snapshot.comments.filter((c) => c.home === item.id && !c.withdrawn)) {
      host.append(el("h3", `${source.id}  ${source.actor}`));
      const body = el("div", undefined, "rich"); body.innerHTML = source.html; host.append(body);
    }
  } else if (picker.kind === "workspace") {
    host.append(el("pre", item.finderType === "workspace-new"
      ? "Provision a new workspace\nThe server cuts a fresh worktree and branch (workspace-<token> without a label),\nthen the launch is submitted there. Nothing is claimed, owned, or bound."
      : item.finderType === "workspace-path"
        ? "Name an existing directory\nAn existing directory on the host, typed as an absolute path - the same\nselection the launch door has always taken. The server validates it;\nnothing is created, claimed, owned, or bound."
        : `record ${item.seq}\nbranch ${item.branch}\nbase ${item.base}\npath ${item.path}`));
  } else host.append(el("pre", item.text ?? item.label));
}

function closePicker() {
  const origin = picker?.viewId;
  picker = null;
  $("#picker").close();
  if (origin && bench.views.has(origin)) focusPane(origin);
}

function openPicker(kind, options = {}) {
  cancelLeader(); pendingZ = false; pendingWindow = false;
  const prompt = ["title", "name", "task-name", "launch-label", "workspace-path"].includes(kind);
  $("#picker").classList.toggle("commands", kind === "commands");
  $("#picker").classList.toggle("prompt", prompt);
  $("#picker-preview").hidden = kind === "commands" || prompt;
  $("#choose").textContent = kind === "commands" ? "Run command"
    : kind === "title" ? "Set title" : kind === "name" ? "Record name"
    : kind === "task-name" ? "Set name" : kind === "launch-label" ? "Set label"
    : kind === "workspace-path" ? "Set directory"
    : kind === "workspace" ? "Stage workspace" : "Open / select";
  $("#query-label-text").textContent = kind === "title" ? "Title" : kind === "name" ? "Your name"
    : kind === "task-name" ? "Task name" : kind === "launch-label" ? "Label (optional)"
    : kind === "workspace-path" ? "Existing directory" : "Search";
  $("#picker-hint").textContent =
    kind === "task-name" ? "Enter stages the name | Escape cancels; run New task again to create"
    : kind === "launch-label" ? "Enter stages the label | Escape cancels; run Launch again to provision and submit"
    : kind === "workspace-path" ? "Enter stages the directory | Escape cancels; run Launch again to submit"
    : kind === "workspace" ? "Up/Down choose | Enter stages | Escape cancels; run Launch again to submit"
    : prompt ? "Enter confirm | Escape cancel"
    : kind === "commands" ? "Up/Down choose | Enter run | Escape cancel"
    : "Up/Down preview | Enter select | Shift+Enter beside | Escape cancel";
  remember(bench.focus);
  const buffer = current();
  if (kind === "cite" && buffer.kind !== "draft") return status("Insert citations while drafting.");
  picker = { kind, viewId: bench.focus, sourceId: options.sourceId ?? buffer.id, selected: 0, items: [],
    after: options.after ?? null, titleTarget: options.target ?? null,
    ticket: kind === "cite" ? bench.beginCitation(buffer.id, currentEditor().state.selection.main.head) : null };
  $("#picker-title").textContent = ({ commands: ": Commands", tasks: "Find task, contribution, or local draft", contents: "Search recorded content", backlinks: `Backlinks to ${options.sourceId ?? buffer.id}`, cite: "Insert citation: search or enter c-N", home: "Choose draft home task", title: `Authored title for ${buffer.id}`, name: "Your claimed name", "task-name": "New task name", workspace: "Launch: choose the workspace", "launch-label": "New workspace label (optional)", "workspace-path": "Existing directory" })[kind];
  $("#query").value = picker.titleTarget ? picker.titleTarget.get() : "";
  $("#beside").hidden = ["commands", "cite", "home", "workspace"].includes(kind) || prompt;
  renderPicker();
  $("#picker").showModal();
  $("#query").focus();
}

function choose(beside) {
  const { kind, ticket, viewId, after, titleTarget } = picker;
  if (kind === "task-name" || kind === "launch-label" || kind === "workspace-path") {
    const value = $("#query").value.trim();
    closePicker();
    if (kind === "task-name") {
      if (!value) return status("No name given; nothing is staged.");
      stagedTask = value;
      status(`Task name staged: ${value}. Run New task again to create it.`);
    } else if (kind === "launch-label") {
      stagedLaunch = { new: { label: value || null } };
      status(`Workspace staged: new${value ? ` (${value})` : ""}. Run Launch again to provision and submit.`);
    } else {
      if (!value) return status("No directory named; nothing is staged.");
      stagedLaunch = { existing: value };
      status(`Workspace staged: ${value}. Run Launch again to submit.`);
    }
    if (after) after(value);
    return;
  }
  if (kind === "title" || kind === "name") {
    const value = $("#query").value.trim();
    closePicker();
    if (kind === "title") {
      titleTarget.set(value);
      persist(); refreshChrome();
      refreshRendered(titleTarget.bufferId);
      if (after) after(value);
      else status(value ? `Title set: ${value}` : "Title cleared; publication requires one.");
    } else {
      if (!value) return status("No name given; nothing recorded.");
      claimedName = value;
      if (after) after(value); else status(`Name recorded: ${value}`);
    }
    return;
  }
  const selection = picker.items[picker.selected];
  if (!selection) return;
  closePicker();
  if (kind === "commands") return runCommand(selection.id);
  if (kind === "workspace") {
    if (selection.finderType === "workspace-new") {
      openPicker("launch-label");
      return;
    }
    if (selection.finderType === "workspace-path") {
      openPicker("workspace-path");
      return;
    }
    stagedLaunch = { existing: selection.path };
    status(`Workspace staged: ${selection.path}. Run Launch again to submit.`);
    return;
  }
  if (kind === "home") {
    const buffer = bench.buffer(bench.view(viewId).bufferId);
    buffer.home = selection.id; persist(); refreshChrome();
    refreshRendered(buffer.id);
    if (after) after(selection);
    return;
  }
  if (kind === "cite") {
    try {
      const { marker, cursor } = bench.insertCitation(ticket, selection.seq);
      const view = editors.get(viewId);
      view.dispatch({ changes: { from: ticket.at, insert: marker }, selection: { anchor: cursor } });
      persist(); focusPane(viewId); status(`Inserted ${marker} for ${selection.id}.`);
    } catch (error) { status(error.message); }
    return;
  }
  const id = selection.finderType === "task" ? `t-${selection.id}` : selection.finderType === "comment" ? selection.id : selection.id;
  if (!bench.buffers.has(id)) published(id);
  if (beside) split("row", id); else openBuffer(id, viewId);
}

function follow() {
  if (isReading()) {
    const link = document.activeElement.closest("a,[data-record-target]");
    if (link && readers.get(bench.focus).contains(link)) link.click();
    else status("View: Tab to a link and press Enter or gd, or click it. Space uv opens Source for cursor-based following.");
    return;
  }
  const view = currentEditor();
  const at = view.state.selection.main.head;
  const line = view.state.doc.lineAt(at);
  const offset = at - line.from;
  const token = [...line.text.matchAll(/c-\d+|t-\d+|\[\^(\d+)\]/g)].find((match) => offset >= match.index && offset <= match.index + match[0].length);
  if (!token) return status("No id or reference under the cursor.");
  if (token[1]) {
    const buffer = current();
    const references = buffer.kind === "draft" ? buffer.references : buffer.source?.references;
    const source = birth(references?.[Number(token[1]) - 1]);
    if (!source) return status("That reference does not name a contribution in this view.");
    openBuffer(source.id);
  } else {
    try { openBuffer(token[0]); } catch (error) { status(error.message); }
  }
}

// -- wiring --------------------------------------------------------------------

$("#query").addEventListener("input", () => { picker.selected = 0; renderPicker(); });
$("#picker").addEventListener("keydown", (event) => {
  if (["ArrowDown", "ArrowUp"].includes(event.key)) {
    event.preventDefault(); picker.selected = Math.max(0, Math.min(picker.items.length - 1, picker.selected + (event.key === "ArrowDown" ? 1 : -1))); highlightPicker();
  } else if (event.key === "Enter" && event.target.tagName !== "BUTTON" && event.target.tagName !== "A") {
    event.preventDefault(); choose(event.shiftKey && !["commands", "cite", "home"].includes(picker.kind));
  }
});
$("#picker").addEventListener("cancel", (event) => { event.preventDefault(); closePicker(); });
$("#picker-close").onclick = closePicker;
$("#choose").onclick = () => choose(false);
$("#beside").onclick = () => choose(true);
$("#open-commands").onclick = () => { focusPane(bench.focus); openPicker("commands"); };

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-record-target]");
  if (target) {
    event.preventDefault();
    openBuffer(target.dataset.recordTarget);
    return;
  }
  const link = event.target.closest(".rich a");
  if (!link) return;
  const url = new URL(link.href);
  if (url.origin !== location.origin || !/^\/t\/\d+$/.test(url.pathname)) return;
  event.preventDefault();
  const thread = `t-${url.pathname.slice(3)}`;
  const named = /^#c-\d+$/.test(url.hash) ? url.hash.slice(1) : null;
  // the same membership contract as deepLink: the hash selects only a
  // contribution that lives in the named thread, and clears otherwise
  const selected = named && comment(named)?.home === Number(url.pathname.slice(3)) ? named : null;
  if (picker) closePicker();
  if (event.shiftKey) {
    split("row", thread);
    presentation().selected = selected;
    showPresentation(bench.focus);
    if (selected) scrollCardIntoView(bench.focus, selected);
  } else openBuffer(thread, bench.focus, "view", selected);
});

// Only the requested Normal-mode jump chords override browser navigation.
document.addEventListener("keydown", (event) => {
  if (!snapshot || $("#picker").open || event.isComposing) return;
  if (event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === "w" && event.target.closest(".cm-content,.reading") && ["Normal", "View"].includes(mode())) {
    event.preventDefault(); event.stopImmediatePropagation();
    pendingWindow = true; pendingZ = false; cancelLeader(); return;
  }
  if (event.ctrlKey && !event.metaKey && !event.altKey && ["i", "o"].includes(event.key.toLowerCase()) && event.target.closest(".cm-content,.reading") && ["Normal", "View"].includes(mode())) {
    event.preventDefault(); event.stopImmediatePropagation();
    pendingG = false; cancelLeader();
    event.key.toLowerCase() === "o" ? back() : forward();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) {
    if ((event.ctrlKey || event.metaKey) && "dflnoprtuw".includes(event.key.toLowerCase())) {
      event.stopImmediatePropagation();
    }
    return;
  }
  if (event.target.closest("input,textarea") || !event.target.closest(".cm-editor,.reading") || mode() === "Insert") return;
  if (mode() === "Visual" && ![" "].includes(event.key)) return;
  if (event.key === "Escape") { cancelLeader(); pendingG = false; pendingZ = false; pendingWindow = false; return; }
  if (pendingWindow) {
    event.preventDefault(); event.stopImmediatePropagation(); pendingWindow = false;
    const id = { v: "split-row", s: "split-column", c: "close", q: "close", o: "only", h: "focus-h", j: "focus-j", k: "focus-k", l: "focus-l" }[event.key];
    if (id) runCommand(id); else status("Ctrl+W: v/s split, c close, o only, h/j/k/l focus.");
    return;
  }
  if (pendingZ) {
    event.preventDefault(); event.stopImmediatePropagation(); pendingZ = false;
    const action = { A: "a", C: "c", O: "o" }[event.key] ?? event.key;
    const fold = { a: "fold-toggle", c: "fold-close", o: "fold-open", M: "fold-close-all", R: "fold-open-all" }[action];
    if (fold) { runCommand(fold); return; }
    if ("jk".includes(action)) {
      if (!sectionFold(currentEditor(), action)) status("No Markdown section here.");
      return;
    }
    if (["z", "t", "b"].includes(event.key)) {
      Vim.handleKey(getCM(currentEditor()), "z"); Vim.handleKey(getCM(currentEditor()), event.key);
    } else status("Section folds: za toggle, zc close, zo open, zM close all, zR open all.");
    return;
  }
  if (event.key === "z" && mode() === "Normal") {
    event.preventDefault(); event.stopImmediatePropagation();
    Vim.handleKey(getCM(currentEditor()), "<Esc>"); pendingG = false; pendingZ = true; return;
  }
  if (leader !== null) {
    event.preventDefault(); event.stopImmediatePropagation();
    if (leader === ",") {
      cancelLeader();
      const command = commands().find((item) => item.local === event.key);
      if (command) runCommand(command.id);
      else status("Comma commands: " + commands().filter((item) => item.local).map((item) => `${item.local} ${item.label.toLowerCase()}`).join(", "));
      return;
    }
    const next = leader + event.key;
    const bound = commands().find((item) => item.binding === next);
    if (bound) { cancelLeader(); runCommand(bound.id); }
    else if (next === " ") { cancelLeader(); runCommand("tasks"); }
    else if (commands().some((item) => item.binding?.startsWith(next))) { leader = next; showLeader(next); }
    else { cancelLeader(); status("Unknown leader sequence. Space shows available commands."); }
    return;
  }
  if (event.key === ":") {
    if (pendingG) Vim.handleKey(getCM(currentEditor()), "<Esc>");
    pendingG = false;
    event.preventDefault(); event.stopImmediatePropagation(); openPicker("commands"); return;
  }
  if (event.key === " " && ["Normal", "View"].includes(mode())) {
    if (pendingG) Vim.handleKey(getCM(currentEditor()), "<Esc>");
    pendingG = false;
    event.preventDefault(); event.stopImmediatePropagation(); leader = ""; showLeader(""); return;
  }
  if (event.key === "," && ["Normal", "View"].includes(mode())) {
    if (pendingG) Vim.handleKey(getCM(currentEditor()), "<Esc>");
    pendingG = false;
    event.preventDefault(); event.stopImmediatePropagation(); leader = ","; showLeader(""); return;
  }
  if (pendingG && ["d", "r"].includes(event.key)) {
    event.preventDefault(); event.stopImmediatePropagation(); pendingG = false;
    Vim.handleKey(getCM(currentEditor()), "<Esc>");
    runCommand(event.key === "d" ? "follow" : "backlinks"); return;
  }
  if (isReading()) {
    const reader = readers.get(bench.focus);
    if (["j", "k", "G"].includes(event.key) || (pendingG && event.key === "g")) {
      event.preventDefault(); event.stopImmediatePropagation();
      const threadKey = (event.key === "j" || event.key === "k") && threadFocused();
      runCommand(threadKey ? (event.key === "j" ? "thread-next" : "thread-prev")
        : event.key === "G" ? "reader-bottom" : (pendingG && event.key === "g") ? "reader-top"
        : event.key === "j" ? "reader-down" : "reader-up");
      pendingG = false; return;
    }
  }
  pendingG = event.key === "g" && !pendingG;
}, true);

// the URL names either a contribution or a thread, and a task path with
// a #c-M hash names the thread with that contribution selected
function deepLink() {
  const thread = location.pathname.match(/^\/t\/(\d+)$/);
  const card = location.hash.match(/^#c-\d+$/);
  if (thread) {
    const named = card ? location.hash.slice(1) : null;
    // the hash selects only a contribution that lives in the named thread
    const selected = named === null ? undefined
      : comment(named)?.home === Number(thread[1]) ? named : null;
    return { id: `t-${thread[1]}`, selected };
  }
  if (card) return { id: location.hash.slice(1) };
  if (/^\/c\/\d+$/.test(location.pathname)) return { id: location.pathname.slice(1).replace("/", "-") };
  return { id: "home" };
}

async function start() {
  const version = await api("/api/v1/version");
  storageKey = `saccade-workbench:${version.db}`;
  identity = await api("/api/v1/workbench/identity");
  snapshot = await api("/api/v1/workbench");
  syncCursor = snapshot.cursor;
  followRecords();
  if (!restore() && !restoreLegacy()) {
    published("home");
    bench.open("home");
  }
  renderLayout(); focusPane(bench.focus);
  status("Space Space: task/contribution/draft | Space sg: content | , e revise, , a accept, , x reject, , t title, , p publish | : commands");
  const linked = deepLink();
  if (linked.id !== "home") {
    try { openBuffer(linked.id, bench.focus, "view", linked.selected); }
    catch { status(`The linked ${linked.id} names nothing recorded; home stands.`); }
  }
  window.addEventListener("hashchange", () => {
    const linked = deepLink();
    if (linked.id === "home") return;
    try { openBuffer(linked.id, bench.focus, "view", linked.selected); }
    catch { status(`The linked ${linked.id} names nothing recorded.`); }
  });
  window.workbenchUI = { state: () => ({
    focus: bench.focus, layout: structuredClone(bench.layout),
    buffers: [...bench.buffers.values()].map(({ id, kind, text, references, home, submission, authoredTitle, eventual, revisionPending, decision }) => ({ id, kind, text, references, home, submission, authoredTitle, eventual, revisionPending, decision })),
    panes: [...bench.views.values()].map((pane) => ({ id: pane.id, bufferId: pane.bufferId, position: bench.position(pane.id) })),
    mode: mode(), cursor: snapshot.cursor, presentation: isReading() ? "view" : "source",
    actor: identity.actor,
    staged: { task: stagedTask, launch: stagedLaunch },
  }) };
}
start().catch((error) => { console.error(error); status(`Workbench unavailable: ${error.message}`); });
