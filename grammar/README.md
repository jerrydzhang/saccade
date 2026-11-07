# Board grammar, version 1

This is the normative statement of board grammar version 1. The
`board-grammar` crate parses boards written in it: `parse` accepts exactly
what this file describes and refuses everything else, with an error that
says what to do, not only what failed.

## Versions

The grammar number is a breaking-change boundary and nothing more. The
number changes only when a kind is removed or a kind's meaning is
reinterpreted; new kinds join without a number change, each admitted by
ratification.

Within one supported number there are two postures:

- Authoring validation - the tracker's commit and check, and this
  crate's `parse` - is strict: it refuses unknown kinds and every other
  shape this file does not describe.
- Reading - the viewer and other reader components - is lenient inside
  the numbers it supports: a component naming a kind the build does not
  know renders as its raw authored source beside a diagnostic, and never
  refuses the board. A board whose grammar number sits outside the
  reader's supported range still refuses.

The viewer's bridge verdicts (`Ok` / `UnknownKind` / `Refused`) carry
exactly this split.

## The board

A board is a single YAML document: one mapping with exactly the fields
`grammar`, `task`, and `content`.

- `grammar` is the integer 1. The crate declares the versions it parses in
  `SUPPORTED_GRAMMAR_VERSIONS`; today that is exactly 1.
- `task` names the task the board accounts for, as the tracker addresses
  it: `t-N`, dense, no leading zeros. The committing task must match: the
  tracker refuses a board whose task field names another task.
- `content` is a list of components. The list may be empty.

No other fields are allowed at the top level. The file holds exactly one
YAML document. YAML anchors, aliases, and duplicate keys are refused: a
board states each piece of content once, written out in full. Mapping keys
are plain text names; numbers, booleans, and collections are not keys.

## Components

Each item of `content` is a mapping with exactly one key naming the
component kind. An item naming no kind, several kinds, or an unknown kind
is refused; a new kind is admitted only by ratification. The five kinds:

### text

`text` holds the component's markdown, dollar math included. Tables and
Mermaid fenced blocks are ordinary text content, not separate kinds.

### group

`group` is a mapping with a required `content` list of components and an
optional `title` (text). No other fields; layout hints do not exist here.

### figure

`figure` is a mapping with required `input` and `caption`.

- `input` is a relative path to a file inside the board tree. It may not
  be absolute, and no path segment may be `..`. Parse does not check the
  file exists or its size; the caller does, against the board tree.
- `caption` is text.

### incorporation

`incorporation` is a mapping with required `source` and `content`.

- `source` is a mapping with exactly `task`, `revision`, and `select`.
  `task` names the source task as `t-N`. `revision` is the full
  40-character lowercase hex commit hash of the pinned source revision.
  `select` is a non-empty list of select paths (below) naming whole
  components in that revision.
- `content` is a list of components and must include at least one `text`
  component with non-blank text, at any depth through groups: the receiving
  text that says what is taken from the source. Text inside a nested
  `incorporation` belongs to that incorporation's own source and does not
  count toward the outer one.

### annotation

`annotation` is the concern as markdown text, written as a scalar. It sits
in the content list near what it addresses; it takes no fields.

## Select paths

A select path is a positional structural path into the pinned source
revision and selects whole components. The written forms:

    /content/INDEX
    /content/INDEX/group/content/INDEX
    /content/INDEX/incorporation/content/INDEX

The `group`/`incorporation` step pairs repeat for deeper nesting. The path
starts at the source board's content list; an index picks one component;
a component that carries nested content is entered by naming its kind
(`group` or `incorporation`), then its `content` list, then an index.
Indices are plain decimal digits and start at 0. Paths must start with a
slash and carry no empty steps.

## What parse refuses

- malformed YAML, no YAML document at all, or more than one document
- YAML anchors and aliases
- duplicate keys in any mapping
- a missing `grammar`, `task`, or `content` field
- a `task` field (board top or incorporation source) that does not read
  as `t-N`
- unknown fields in the board mapping, a component, or an incorporation source
- unknown component kinds, or an item naming no kind or several kinds
- a grammar version outside `SUPPORTED_GRAMMAR_VERSIONS`, or a `grammar`
  value that is not an integer
- an incorporation whose content carries no non-blank `text` component
- a `revision` that is not the full 40-character lowercase hex commit hash
- a figure `input` that is absolute or contains a `..` segment
- an empty `select` list, or a select path that does not match the language above
- content nesting deeper than `MAX_NESTING_DEPTH` (100 levels)

Empty lists are legal (`content: []`, a group with no items), and empty
text or captions are legal; only the incorporation receiving-text rule
demands non-blank text.
