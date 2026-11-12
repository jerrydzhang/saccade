<script lang="ts">
  // The open-location picker (F1): the m-490 surface - the field with
  // the ranked list beneath it. An empty query lists EVERYTHING (the
  // list is the discovery); typing filters through the model's tiers;
  // ArrowDown/Ctrl-n and ArrowUp/Ctrl-p move the selection, Enter
  // opens it, Escape closes, click opens; typing resets the
  // selection to the first row. At most 60 rows; the selected row is
  // kept scrolled into view. Composition only - the ranking lives in
  // lib/picker/model.ts.
  import { pickList, isPathMode, type PickContext } from "../lib/picker/model.ts";

  let {
    context,
    submit,
    close,
  }: {
    context: PickContext;
    submit(spec: string): void;
    close(): void;
  } = $props();

  let field: HTMLInputElement | null = $state(null);
  let query = $state("");
  let selected = $state(0);
  let listElement: HTMLElement | null = $state(null);

  const entries = $derived(pickList(query, context).slice(0, 60));

  // opening clears the field and focuses it (m-490)
  $effect(() => {
    field?.focus();
  });

  $effect(() => {
    query;
    selected = 0;
  });

  $effect(() => {
    const element = listElement?.children[selected] ?? null;
    if (element instanceof HTMLElement) {
      element.scrollIntoView({ block: "nearest" });
    }
  });

  function open(index: number): void {
    const entry = entries[index];
    if (entry === undefined) {
      // no row to open: the typed text is the location (the legacy
      // path stays - typed locations are how a tracker is first
      // reached)
      if (query.trim().length > 0) submit(query.trim());
      return;
    }
    submit(entry.spec);
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.key === "ArrowDown" || (event.ctrlKey && event.key === "n")) {
      event.preventDefault();
      selected = Math.min(entries.length - 1, selected + 1);
    } else if (event.key === "ArrowUp" || (event.ctrlKey && event.key === "p")) {
      event.preventDefault();
      selected = Math.max(0, selected - 1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      open(selected);
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (event.key === "Tab" && isPathMode(query)) {
      // path mode's completion: the selected row's path fills the
      // field (m-490's Tab rule; the FS listing needs a door the
      // port records as absent, so completion covers the known list)
      event.preventDefault();
      const entry = entries[selected];
      if (entry !== undefined) {
        query = entry.id;
      }
    }
  }
</script>

<div class="port-picker" data-port-picker="">
  <input
    bind:this={field}
    bind:value={query}
    data-port-control="picker-input"
    placeholder="open: host:8811/tracker/t-N, a task name, /absolute/path"
    onkeydown={onKeydown}
  />
  {#if entries.length > 0}
    <ul class="port-picker-list" bind:this={listElement} data-port-picker-list="">
      {#each entries as entry, index (entry.spec)}
        <li>
          <button
            class="port-picker-row"
            class:port-picker-selected={index === selected}
            data-port-picker-row={entry.id}
            onclick={(event) => {
              event.preventDefault();
              open(index);
            }}
          >
            <span class="port-picker-kind">{entry.kind}</span>
            <span class="port-picker-id">{entry.id}</span>
            <span class="port-picker-name">{entry.name}</span>
            <span class="port-picker-here" aria-hidden="true">
              {context.openHere === entry.spec ? "open here" : ""}
            </span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>
