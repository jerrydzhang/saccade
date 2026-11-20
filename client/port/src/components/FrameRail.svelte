<!-- The frame's rail, painted: the projects this device has
registered, and beneath each - drawn generically, no kind knowledge -
whatever targets the instruments publish there (the labels carry
their own words). The add field rides at the rail's foot; the foot
group holds the published targets no project claims. Any window can
show the rail; the frame (App.svelte) hosts this pane beside the
content and feeds it the frame rail's live view. The pane's entries
are ordinary targets of the shown buffer: select mode labels them
with the content's (the labels ride the same session, measured in the
pane), a plain click or Enter runs the primary act through the shown
lease, a right-click opens the act list. With no shown buffer (the
empty window), a project's plain click opens its claimed view
directly. The wheel scrolls the rail and keeps the shown project in
view; the reading keys never scroll it. -->
<script lang="ts">
  import type { AddView } from "../lib/frame/add.ts";
  import type { RailPublication } from "../contracts/buffer.ts";
  import type { MenuView } from "../lib/menu/session.ts";
  import LabelOverlay from "./LabelOverlay.svelte";
  import ActMenu from "./ActMenu.svelte";
  import { partMeasureIn } from "../lib/select/measure.ts";

  let {
    projects,
    beneath,
    foot,
    notes,
    view,
    shown,
    labels,
    typed = "",
    menu,
    menuSession,
    typeAdd,
    submitAdd,
    closeAdd,
    entryActivate,
    entryMenu,
    attachPane,
  }: {
    /** The registered projects: the frame rail's headers. */
    projects: FrameRailProject[];
    /** The published targets beneath each project (the claims seam's
     * answer), each with its instrument's row component when handed
     * over the registration. */
    beneath: readonly { project: FrameRailProject; targets: readonly RailPublication[] }[];
    /** The published targets no project claims: the frame's foot
     * group. */
    foot: readonly RailPublication[];
    /** The unreachable notes under the project headers, keyed by the
     * project's rail id. */
    notes: readonly { part: string; note: string }[];
    view: { visible: boolean; footnote: string | null; add: AddView; pane: HTMLElement | null };
    /** The project this window shows: the rail's mark. */
    shown: string | null;
    /** The select session's labels over the presented set (the
     * content's and the rail's): the pane paints its own. */
    labels: readonly { part: string; label: string; nested?: boolean }[];
    /** The select session's typed label, marked on its chips. */
    typed?: string;
    /** The act menu's view, when the menu's target is a rail entry
     * (the content's menu is the surface's own). */
    menu: MenuView | null;
    /** The menu's session (the shown buffer's one menu session; the
     * pane renders it for its own entries). */
    menuSession: unknown;
    typeAdd(value: string): void;
    submitAdd(): void;
    closeAdd(): void;
    /** A plain click or Enter on an entry: through the shown lease
     * when one is present, else a project's entry opens its claimed
     * view. */
    entryActivate(part: string, event: MouseEvent): void;
    /** A right-click on an entry: the act list at the pointer. */
    entryMenu(part: string, event: MouseEvent): void;
    attachPane(element: HTMLElement | null): void;
  } = $props();

  let pane: HTMLElement | null = $state(null);
  let list: HTMLElement | null = $state(null);
  let addInput: HTMLInputElement | null = $state(null);
  // the pointer's hover marking (the rows' highlight, the plane's
  // own): the entry under the pointer carries data-port-hover; any
  // scroll clears it
  let hovered = $state<string | null>(null);
  let lastPointer: { x: number; y: number } | null = null;
  function pointerMoved(event: MouseEvent): void {
    if (lastPointer !== null && lastPointer.x === event.clientX && lastPointer.y === event.clientY) return;
    lastPointer = { x: event.clientX, y: event.clientY };
    const element = (event.target as HTMLElement | null)?.closest?.(
      '[data-port-kind="project"], [data-port-kind="agent"]',
    );
    hovered = element instanceof HTMLElement ? element.getAttribute("data-port-part") : null;
  }
  $effect(() => {
    const clear = (): void => {
      hovered = null;
    };
    document.addEventListener("scroll", clear, { capture: true, passive: true });
    return () => document.removeEventListener("scroll", clear, { capture: true });
  });
  $effect(() => {
    attachPane(pane);
  });
  // the add field opens focused: typing lands in it natively (Enter
  // and Escape are its captured context's own commands)
  $effect(() => {
    if (view.add.open) addInput?.focus({ preventScroll: true });
  });

  const chipLabels = $derived(labels.filter((entry) => entry.part.startsWith("rail/")));
  const measure = $derived(partMeasureIn(pane));

  // The wheel scrolls the rail's own list; the shown project stays in
  // view (the rail's clamping rule), whichever project's row the wheel
  // would push out.
  function keepShownInView(): void {
    if (shown === null || list === null) return;
    const marked = list.querySelector(`[data-port-control=${JSON.stringify(`project-${shown}`)}]`);
    marked?.scrollIntoView({ block: "nearest" });
  }

  function partOf(event: Event): string | null {
    return (event.target as Element | null)?.closest("[data-port-part]")?.getAttribute("data-port-part") ?? null;
  }

  function railMenu(): MenuView | null {
    return menu !== null && menu.part.startsWith("rail/") ? menu : null;
  }

  function noteFor(part: string): string | null {
    return notes.find((entry) => entry.part === part)?.note ?? null;
  }
</script>

{#if view.visible}
  <nav
    class="port-rail"
    data-port-control="rail"
    data-port-frame-capture=""
    aria-label="projects"
    bind:this={pane}
  >
    <!-- a press on an entry never moves focus onto it: the entries are
    driven by their clicks and the select session, never by focus, and
    a button left focused would draw the keyboard ring when the next
    key (Escape) arrives -->
    <div
      class="port-rail-list"
      bind:this={list}
      role="presentation"
      onwheel={() => keepShownInView()}
      onmousedown={(event) => {
        if ((event.target as Element | null)?.closest("button") !== null) event.preventDefault();
      }}
    >
      {#each projects as project (project.part)}
        <section class="port-rail-project" data-port-shown={project.current ? "" : undefined}>
          <button
            class="port-rail-name"
            data-port-control={`project-${project.name}`}
            data-port-part={project.part}
            data-port-kind="project"
            data-port-title={project.name}
            data-port-hover={hovered === project.part ? "" : undefined}
            aria-current={project.current ? "page" : undefined}
            onclick={(event) => entryActivate(partOf(event) ?? project.part, event)}
            oncontextmenu={(event) => {
              event.preventDefault();
              entryMenu(partOf(event) ?? project.part, event);
            }}
          >{project.name}</button>
          {#if noteFor(project.part) !== null}
            <div class="port-rail-note" title={noteFor(project.part) ?? undefined}>unreachable</div>
          {/if}
          {#each beneath.find((group) => group.project.part === project.part)?.targets as publication (publication.target.id)}
            <button
              class="port-rail-entry"
              data-port-control={`rail-entry-${publication.target.id}`}
              data-port-part={publication.target.id}
              data-port-kind={publication.target.kind}
              data-port-title={publication.target.label}
              data-port-hover={hovered === publication.target.id ? "" : undefined}
              onclick={(event) => entryActivate(partOf(event) ?? publication.target.id, event)}
              oncontextmenu={(event) => {
                event.preventDefault();
                entryMenu(partOf(event) ?? publication.target.id, event);
              }}
            >
              {#if publication.row !== null}
                {@const Row = publication.row}
                <Row target={publication.target} />
              {:else}
                {publication.target.label}
              {/if}
            </button>
          {/each}
        </section>
      {/each}
      {#if view.add.open}
        <div class="port-rail-add">
          <!-- the field's keys are the frame capture's own (Enter
          submits, Escape leaves - the window routes them here) -->
          <input
            bind:this={addInput}
            data-port-control="project-input"
            data-port-captured-input=""
            type="text"
            placeholder="machine:/path/to/repository"
            aria-label="register a project"
            value={view.add.value}
            oninput={(event) => typeAdd(event.currentTarget.value)}
          />
          {#if view.add.problem !== null}
            <div class="port-rail-note" data-port-control="project-input-problem">{view.add.problem}</div>
          {/if}
        </div>
      {/if}
      {#if foot.length > 0}
        <section class="port-rail-project" data-port-control="unregistered">
          <div class="port-rail-name port-rail-group">unregistered</div>
          {#each foot as publication (publication.target.id)}
            <button
              class="port-rail-entry"
              data-port-control={`unregistered-${publication.target.id}`}
              data-port-part={publication.target.id}
              data-port-kind={publication.target.kind}
              data-port-title={publication.target.label}
              data-port-hover={hovered === publication.target.id ? "" : undefined}
              onclick={(event) => entryActivate(publication.target.id, event)}
              oncontextmenu={(event) => {
                event.preventDefault();
                entryMenu(publication.target.id, event);
              }}
            >
              {#if publication.row !== null}
                {@const Row = publication.row}
                <Row target={publication.target} />
              {:else}
                {publication.target.label}
              {/if}
            </button>
          {/each}
        </section>
      {/if}
    </div>
    {#if chipLabels.length > 0}
      <LabelOverlay chips={chipLabels} {measure} {typed} />
    {/if}
    {#if railMenu() !== null}
      <ActMenu menu={railMenu()!} host={pane} session={menuSession as never} />
    {/if}
    {#if view.footnote !== null}
      <div class="port-rail-note" data-port-control="herdr-note">{view.footnote}</div>
    {/if}
  </nav>
{/if}
