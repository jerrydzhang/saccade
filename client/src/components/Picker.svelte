<script lang="ts">
  // The picker (round 16): one input, every letter types, the results
  // move on arrows and Ctrl-n Ctrl-p (t-31). It focuses itself when it
  // opens; closing hands the keys back to the window (the caller's
  // onClose).
  import { pickerLabelKind, type PickerItem } from "../lib/picker.ts";

  type Props = {
    query: string;
    active: number;
    results: PickerItem[];
    placeholder: string;
    onKey: (event: KeyboardEvent) => void;
    onChoose: (item: PickerItem) => void;
  };

  let { query = $bindable(), active = $bindable(), results, placeholder, onKey, onChoose }: Props = $props();
  let input: HTMLInputElement | undefined = $state(undefined);

  $effect(() => {
    input?.focus();
  });
</script>

<div class="picker-overlay" role="dialog" aria-label="picker" data-control="picker:overlay">
  <div class="picker-panel">
    <input
      type="text"
      class="picker-input"
      {placeholder}
      aria-label="picker query"
      data-control="picker:input"
      bind:this={input}
      bind:value={query}
      onkeydown={onKey}
    />
    <div class="picker-list" data-control="picker:list">
      {#each results as item, index (item.location)}
        <button
          type="button"
          class="picker-row"
          class:picker-active={index === active}
          data-control="picker:result"
          onclick={() => onChoose(item)}
          onmouseenter={() => (active = index)}
        >
          <span class="picker-kind">{pickerLabelKind(item).kind}</span>
          <span class="picker-name">{pickerLabelKind(item).label}</span>
        </button>
      {:else}
        <p class="picker-empty">nothing matches</p>
      {/each}
    </div>
  </div>
</div>
