const point = () => ({ anchor: 0, head: 0, scrollTop: 0 });

export class Workbench {
  constructor() {
    this.buffers = new Map();
    this.views = new Map();
    this.nextView = 1;
    this.layout = null;
    this.focus = null;
  }

  register(buffer) {
    if (this.buffers.has(buffer.id)) return this.buffers.get(buffer.id);
    if (!["home", "task", "comment", "draft"].includes(buffer.kind)) {
      throw new Error("unknown buffer kind");
    }
    const stored = {
      ...buffer,
      text: buffer.text ?? "",
      references: [...(buffer.references ?? [])],
      lastPosition: point(),
      revision: 0,
    };
    this.buffers.set(stored.id, stored);
    return stored;
  }

  buffer(id) {
    const buffer = this.buffers.get(id);
    if (!buffer) throw new Error("buffer is not open");
    return buffer;
  }

  view(id = this.focus) {
    const view = this.views.get(id);
    if (!view) throw new Error("pane is not open");
    return view;
  }

  open(bufferId, viewId = this.focus) {
    const buffer = this.buffer(bufferId);
    if (this.layout === null) {
      const id = this.nextView++;
      this.views.set(id, { id, bufferId, positions: new Map([[bufferId, { ...buffer.lastPosition }]]) });
      this.layout = { view: id };
      this.focus = id;
    } else {
      const view = this.view(viewId);
      view.bufferId = bufferId;
      if (!view.positions.has(bufferId)) {
        view.positions.set(bufferId, { ...buffer.lastPosition });
      }
      this.focus = view.id;
    }
    return this.view();
  }

  position(viewId = this.focus) {
    const view = this.view(viewId);
    return { ...(view.positions.get(view.bufferId) ?? point()) };
  }

  remember(position, viewId = this.focus) {
    const view = this.view(viewId);
    const length = this.buffer(view.bufferId).text.length;
    const next = { ...this.position(viewId), ...position };
    for (const key of ["anchor", "head"]) {
      if (!Number.isInteger(next[key]) || next[key] < 0 || next[key] > length) {
        throw new Error("position is outside the buffer");
      }
    }
    if (!Number.isFinite(next.scrollTop) || next.scrollTop < 0) {
      throw new Error("invalid scroll position");
    }
    view.positions.set(view.bufferId, next);
    this.buffer(view.bufferId).lastPosition = { ...next };
  }

  split(axis, bufferId = this.view().bufferId) {
    if (!["row", "column"].includes(axis)) throw new Error("invalid split axis");
    this.buffer(bufferId);
    const source = this.view();
    const id = this.nextView++;
    const position = bufferId === source.bufferId
      ? this.position() : { ...this.buffer(bufferId).lastPosition };
    const positions = new Map([[bufferId, position]]);
    this.views.set(id, { id, bufferId, positions });
    this.layout = replaceLeaf(this.layout, source.id, {
      axis, ratio: 0.5, children: [{ view: source.id }, { view: id }],
    });
    this.focus = id;
    return this.view();
  }

  swapPanes(first, second) {
    this.view(first); this.view(second);
    function swap(tree) {
      if (tree.view) return { view: tree.view === first ? second : tree.view === second ? first : tree.view };
      return { ...tree, children: tree.children.map(swap) };
    }
    this.layout = swap(this.layout);
    this.focus = first;
  }

  closePane(viewId = this.focus) {
    this.view(viewId);
    if (this.views.size === 1) return false;
    this.layout = removeLeaf(this.layout, viewId);
    this.views.delete(viewId);
    if (this.focus === viewId) this.focus = leaves(this.layout)[0];
    return true;
  }

  onlyPane(viewId = this.focus) {
    const selected = this.view(viewId);
    this.views = new Map([[viewId, selected]]);
    this.layout = { view: viewId };
    this.focus = viewId;
  }

  edit(bufferId, text) {
    const buffer = this.buffer(bufferId);
    if (buffer.kind !== "draft" && !buffer.revisionPending) {
      throw new Error("published buffers are read-only");
    }
    if (typeof text !== "string") throw new Error("draft text must be a string");
    buffer.text = text;
    buffer.revision++;
    return buffer;
  }

  beginCitation(bufferId, at) {
    const buffer = this.buffer(bufferId);
    if (buffer.kind !== "draft") throw new Error("citation insertion needs a draft");
    if (!Number.isInteger(at) || at < 0 || at > buffer.text.length) {
      throw new Error("citation position is outside the draft");
    }
    return { bufferId, at, revision: buffer.revision };
  }

  insertCitation(ticket, birthRecord) {
    const buffer = this.buffer(ticket.bufferId);
    if (buffer.kind !== "draft" || buffer.revision !== ticket.revision) {
      throw new Error("draft changed while choosing the citation; choose again");
    }
    if (!Number.isSafeInteger(birthRecord) || birthRecord < 0) {
      throw new Error("citation needs a birth-record reference");
    }
    const references = [...buffer.references];
    let index = references.indexOf(birthRecord);
    if (index === -1) {
      index = references.length;
      references.push(birthRecord);
    }
    const marker = `[^${index + 1}]`;
    const text = buffer.text.slice(0, ticket.at) + marker + buffer.text.slice(ticket.at);
    this.edit(buffer.id, text);
    buffer.references = references;
    return { marker, cursor: ticket.at + marker.length, buffer };
  }
}

function replaceLeaf(node, viewId, replacement) {
  if (node.view === viewId) return replacement;
  if (!node.children) return node;
  return { ...node, children: node.children.map((child) => replaceLeaf(child, viewId, replacement)) };
}

function removeLeaf(node, viewId) {
  if (node.view === viewId) return null;
  if (!node.children) return node;
  const children = node.children.map((child) => removeLeaf(child, viewId)).filter(Boolean);
  return children.length === 1 ? children[0] : { ...node, children };
}

function leaves(node) {
  return node.children ? node.children.flatMap(leaves) : [node.view];
}
