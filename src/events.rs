use crate::harness::Harness;
use crate::objects::comment::{CommentId, CommentKind};
use crate::objects::incarnation::IncarnationId;
use crate::objects::message::MessageId;
use crate::objects::task::TaskId;
use crate::types::artifact::Artifact;
use crate::types::pointers::{GitBranch, GitCommit, NativeSessionId, SessionLocator, WorktreePath};
use crate::types::prose::Prose;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    // Task events
    TaskCreated {
        name: Prose,
    },
    /// A transfer of whole-task responsibility: never a state
    /// transition, never a permission grant. The claim history
    /// accumulates on the context.
    TaskClaimed {
        id: TaskId,
        /// The record-derived incarnation claiming, when the claim
        /// rides one; absent means the named actor claims directly
        incarnation: Option<IncarnationId>,
    },
    /// The human's stop of an unfinished undertaking, with the
    /// explanation that survives in the record.
    TaskCancelled {
        id: TaskId,
        explanation: Prose,
    },
    /// The human's explicit resumption of a cancelled undertaking,
    /// with the explanation of why it is pursued again.
    TaskReopened {
        id: TaskId,
        explanation: Prose,
    },
    // Comment events
    /// The payload is its variant, title and body included on the
    /// deposits; Accept carries neither. Every comment names its one
    /// home task and carries its citation mapping: marker k names
    /// references[k-1].
    Commented {
        home: TaskId,
        kind: CommentKind,
        references: Vec<RecordRef>,
    },
    /// A presentation repair, not a transition: the fold swaps the
    /// variant's prose and/or its title - each independently, at
    /// least one required - and points at the latest revision record.
    /// The original bytes stay in their birth event, references never
    /// change, and no other variant takes a title.
    CommentRevised {
        id: CommentId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<Prose>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        body: Option<Prose>,
    },
    /// The standing-error repair: the deposit's implicit claim - this
    /// is reached, this earns residency - was false. The body leaves
    /// the fold's presentation, a tombstone stays; the log is
    /// untouched. Note deposits only.
    CommentWithdrawn {
        id: CommentId,
        note: Prose,
    },
    /// A use's explicit selection of a conclusion: the offer named
    /// with required grounds. Binding a prospective use and rebinding
    /// a bound one are the same record; the log keeps every earlier
    /// selection.
    UseBound {
        id: CommentId,
        offer: CommentId,
        grounds: Prose,
    },
    /// A thread holding an artifact: name and content identity, the
    /// bytes stay in the store. Authored like a comment, any tier.
    ArtifactAdded {
        root: TaskId,
        artifact: Artifact,
    },
    // Workspace events: the establishment is the machinery verb,
    // System-authored by role; the checkpoint is the exception any
    // tier may record
    /// The workspace-establishment birth: task reference, canonical
    /// base, branch, and the workspace's absolute path. The record's
    /// position is the shared workspace/incarnation identity.
    WorkspaceEstablished {
        task_id: TaskId,
        base: GitCommit,
        branch: GitBranch,
        worktree: WorktreePath,
    },
    TaskWorkspaceCheckpointed {
        task_id: TaskId,
        checkpoint: GitCommit,
    },
    /// The created workspace's birth: the canonical base commit,
    /// branch, and absolute path provisioning established. The
    /// record's position is the workspace's own identity; no task, no
    /// incarnation, no number rides the payload.
    WorkspaceCreated {
        base: GitCommit,
        branch: GitBranch,
        worktree: WorktreePath,
    },
    /// The confirmed change of where a native incarnation lives,
    /// published only after the host's liveness answer permitted it.
    /// It asserts the confirmed change - never a pending intent, never
    /// a running process.
    WorkspaceChanged {
        incarnation: IncarnationId,
        worktree: WorktreePath,
    },
    /// A completed workspace cleanup: host closure verified and the Git
    /// checkout confirmed absent before this lands. It references the
    /// existing workspace birth identity only - the birth carries the
    /// immutable branch/path/base, this carries nothing else.
    WorkspaceCleaned {
        workspace: RecordRef,
    },
    /// A completed workspace restoration: the matching Git checkout
    /// established or validated before this lands. It references the
    /// existing workspace birth identity only.
    WorkspaceRestored {
        workspace: RecordRef,
    },
    // Root destination events: the human's explicit association of
    // the primary checkout with a Herdr space/tab pair
    /// The chosen destination for launches at the primary checkout,
    /// recorded from the terminal standing in the manual root space.
    /// A chosen destination, never a host-space-existence claim; the
    /// association does not depend on any shell staying alive.
    RootSpaceBound {
        space: String,
        tab: String,
    },
    /// The removal of the root association: root launches go back to
    /// unplaced until a later bind.
    RootSpaceCleared {},
    // Incarnation and message events: registration and submission are
    // machinery verbs, System-authored by role; a message is authored
    // by its sending session's registered identity
    /// The native-session birth: the durable native id and the
    /// saved-session locator ordinary resume validates. No task, no
    /// workspace, no home; registering an already-known native
    /// session is resolved by the effect, never a second birth.
    IncarnationRegistered {
        native_id: NativeSessionId,
        saved_session: Option<SessionLocator>,
        address: Option<String>,
        /// The address kind: launched (a new host terminal) or adopted
        /// (the caller's own). Terminal liveness reads as session
        /// liveness only for the launched kind.
        #[serde(default)]
        adopted: bool,
        /// The harness the session runs under; native identity is
        /// interpreted within its harness, and registrations born
        /// before the field existed fold as pi.
        #[serde(default)]
        harness: Harness,
    },
    /// The attach completes a door-birthed registration: pi's own
    /// saved-session locator, known at attach before the first save.
    SavedSessionBound {
        incarnation: IncarnationId,
        saved_session: SessionLocator,
    },
    /// The association's host address moves with a relaunch: the same
    /// incarnation, another terminal, and the kind it now speaks.
    /// Opaque bytes, per the address law.
    HostAddressAssigned {
        incarnation: IncarnationId,
        address: Option<String>,
        #[serde(default)]
        adopted: bool,
    },
    /// A durable addressed message. The sender is the sending
    /// session's registered identity, never authored prose; the body
    /// is the only caller-authored field. Task context may ride
    /// without publication; no thread home exists.
    MessageSent {
        from: IncarnationId,
        to: IncarnationId,
        reply_to: Option<MessageId>,
        task: Option<TaskId>,
        body: Prose,
    },
    /// The recipient's integration handed the message to its native
    /// session: ordinary pending/submission bookkeeping, never a
    /// claim of reading, agreement, completion, or task acceptance.
    MessageSubmitted {
        id: MessageId,
        by: IncarnationId,
    },
}

/// A stable internal reference: the birth record of the thing cited.
pub type RecordRef = crate::store::RecordId;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    // Task commands
    CreateTask {
        name: Prose,
    },
    ClaimTask {
        id: TaskId,
        incarnation: Option<IncarnationId>,
    },
    CancelTask {
        id: TaskId,
        explanation: Prose,
    },
    ReopenTask {
        id: TaskId,
        explanation: Prose,
    },
    // Comment commands
    Comment {
        home: TaskId,
        kind: CommentKind,
        references: Vec<RecordRef>,
    },
    ReviseComment {
        id: CommentId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<Prose>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        body: Option<Prose>,
    },
    WithdrawComment {
        id: CommentId,
        note: Prose,
    },
    /// Select a conclusion for a use: binding a prospective use,
    /// rebinding a bound one.
    BindUse {
        id: CommentId,
        offer: CommentId,
        grounds: Prose,
    },
    Artifact {
        root: TaskId,
        artifact: Artifact,
    },
    // Machinery verbs: System authorship comes from the role, never input
    EstablishWorkspace {
        task_id: TaskId,
        base: GitCommit,
        branch: GitBranch,
        worktree: WorktreePath,
    },
    /// Retired from the public wire: the name no longer decodes at
    /// /api/v1/command, so no caller can land a checkpoint; the shape
    /// stays constructible for the trusted fixtures and decides to its
    /// historical event.
    #[serde(skip_deserializing)]
    CheckpointWorkspace {
        task_id: TaskId,
        checkpoint: GitCommit,
    },
    // Creation and change are machinery verbs: System authorship comes
    // from the role, never input
    CreateWorkspace {
        base: GitCommit,
        branch: GitBranch,
        worktree: WorktreePath,
    },
    ChangeWorkspace {
        incarnation: IncarnationId,
        worktree: WorktreePath,
    },
    // Cleanup and restoration are machinery verbs: System authorship
    // comes from the role, never input - a door requests the effect,
    // the serving controller records the completed fact
    CleanWorkspace {
        workspace: RecordRef,
    },
    RestoreWorkspace {
        workspace: RecordRef,
    },
    // The primary checkout's Herdr destination: the human binds the
    // chosen manual root space's exact pair from the terminal standing
    // in it; a rebind replaces the latest choice, clear removes it,
    // and the log keeps every earlier choice.
    BindRootSpace {
        space: String,
        tab: String,
    },
    ClearRootSpace {},
    // Registration and submission are machinery verbs: System
    // authorship comes from the role, never input. A message is
    // authored by its sending session's registered identity. The
    // door-birthed registration carries the opaque host address and
    // may precede the saved locator, which the attach binds; the
    // address moves when a registered incarnation launches again.
    RegisterIncarnation {
        native_id: NativeSessionId,
        saved_session: Option<SessionLocator>,
        address: Option<String>,
        #[serde(default)]
        adopted: bool,
        /// The harness the session runs under; absent folds as pi.
        #[serde(default)]
        harness: Harness,
    },
    BindSavedSession {
        incarnation: IncarnationId,
        saved_session: SessionLocator,
    },
    AssignHostAddress {
        incarnation: IncarnationId,
        address: Option<String>,
        #[serde(default)]
        adopted: bool,
    },
    /// The sending side: `from` is the registered session's own
    /// identity, matched against the caller's declared scope at the
    /// store, never free authorship
    SendMessage {
        from: IncarnationId,
        to: IncarnationId,
        reply_to: Option<MessageId>,
        task: Option<TaskId>,
        body: Prose,
    },
    SubmitMessage {
        id: MessageId,
        by: IncarnationId,
    },
}
