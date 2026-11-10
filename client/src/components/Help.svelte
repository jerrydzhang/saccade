<script lang="ts">
  // The help sheet (round 16's showHelp): this window's keys, then its
  // space commands from the same table the which-key menu reads, then
  // the look notice.
  import { keyLabel, type MenuAct } from "../lib/menu.ts";

  type Props = {
    kind: string;
    keys: Array<[string, string]>;
    commands: MenuAct[];
    lookNotice: string | null;
  };

  let { kind, keys, commands, lookNotice }: Props = $props();
</script>

<div class="help-overlay" role="dialog" aria-label="keys and commands" data-control="help:overlay">
  <div class="help-panel">
    <div class="help-section">
      <p class="help-title">keys - {kind}</p>
      {#each keys as [key, label] (key)}
        <p class="help-row"><span class="ctx-menu-key">{key}</span> {label}</p>
      {/each}
    </div>
    <div class="help-section">
      <p class="help-title">space ...</p>
      {#each commands as command (command.key)}
        <p class="help-row"><span class="ctx-menu-key">{keyLabel(command.key)}</span> {command.label}</p>
      {/each}
    </div>
    {#if lookNotice}
      <div class="help-section">
        <p class="help-title">look</p>
        <p class="help-row" data-control="help:look">{lookNotice}</p>
      </div>
    {/if}
  </div>
</div>
