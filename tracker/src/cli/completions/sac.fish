# sac fish completions, emitted by `sac completions fish`.
# Dynamic candidates come from `sac complete <kind>`, a hidden read
# door over the fold; it answers empty outside a repo or without a
# reachable destination, so completion stays silent and records
# nothing. An older sac on PATH answers nothing and the script
# degrades to its static half.

complete -c sac -f

function __saccade_positions
    # the positional tokens before the cursor, after the program name:
    # global and local value flags consume the token after them, a
    # leading 'workspace' stays visible for verb matching
    set -l tokens (commandline -opc)
    set -e tokens[1]
    set -l out
    set -l prev_value 0
    for t in $tokens
        set was_value $prev_value
        set prev_value 0
        if test $was_value -eq 1
            continue
        end
        switch $t
            case --db --repo --server --actor --at --explanation --grounds --to --reply --task --workspace --resume --new --base --ask --until --after --out --bind --port --history --within
                set prev_value 1
                continue
            case '-*'
                continue
        end
        set out $out $t
    end
    if test (count $out) -gt 0
        printf '%s\n' $out
    end
end

function __saccade_npos
    # positional count with the container verb word dropped: 1 is the
    # first argument position of the (sub)verb
    set -l have (__saccade_positions)
    if test "$have[1]" = workspace
        set -e have[1]
    end
    count $have
end

function __saccade_verb
    # the command's leading verb words match exactly the names given
    set -l want $argv
    set -l have (__saccade_positions)
    if test (count $have) -lt (count $want)
        return 1
    end
    for i in (seq (count $want))
        if test "$have[$i]" != "$want[$i]"
            return 1
        end
    end
    return 0
end

function __saccade_no_verb
    set -l have (__saccade_positions)
    test (count $have) -eq 0
end

function __saccade_prev
    set -l tokens (commandline -opc)
    test "$tokens[-1]" = "$argv[1]"
end

function __saccade_prev_value_flag
    # the token before the cursor is a flag expecting a value: the
    # cursor is completing that value, no id belongs there
    set -l tokens (commandline -opc)
    switch $tokens[-1]
        case --db --repo --server --actor --at --explanation --grounds --to --reply --task --workspace --resume --new --base --ask --until --after --out --bind --port --history --within
            return 0
    end
    return 1
end

function __saccade_verbs
    printf 'create\tCreate a new task\n'
    printf 'claim\tTake whole-task responsibility; a transfer, never a state change\n'
    printf 'cancel\tHuman-only: stop an unfinished undertaking, explaining why\n'
    printf 'reopen\tHuman-only: resume a cancelled undertaking, explaining why\n'
    printf 'send\tSend a durable addressed message\n'
    printf 'messages\tShow the messages an incarnation participates in\n'
    printf 'list\tList objects (world projection)\n'
    printf 'log\tPrint raw event records\n'
    printf 'show\tEverything about tasks and records, many at once\n'
    printf 'search\tSearch the boards: current, history, references\n'
    printf 'offer\tOffer the task current board revision\n'
    printf 'accept\tAccept the offered board\n'
    printf 'reject\tReject the offered board with grounds\n'
    printf 'board\tThe task board controller\n'
    printf 'serve\tServe the console and api over HTTP\n'
    printf 'launch\tLaunch the composed native Pi in a workspace\n'
    printf 'workspace\tThe task-free workspace controller\n'
    printf 'identity\tReport this session declared identity and routing\n'
    printf 'clone\tClone the log prefix through a storage seq into a fresh tracker\n'
    printf 'skill\tDeploy this binary embedded skill, or verify the copy\n'
    printf 'wait\tBlock until a matching record lands on the target\n'
    printf 'completions\tEmit a shell completion script\n'
end

function __saccade_candidates
    # the hidden read door over the fold; nothing on any failure
    command -q sac; or return 1
    sac complete $argv 2>/dev/null
end

function __saccade_shells
    printf 'fish\tthe primary shell\n'
end

function __saccade_workspace_verbs
    printf 'new\tProvision a NEW workspace through Git effects\n'
    printf 'switch\tMove a stopped native session saved directory\n'
    printf 'clean\tClose an exact recorded workspace, work preserved\n'
    printf 'restore\tBring an exact recorded cleaned workspace back\n'
    printf 'reconcile\tVerify every recorded workspace; Cleaned facts ride --apply\n'
    printf 'root\tBind, show, or clear the primary checkout destination\n'
end

function __saccade_root_verbs
    printf 'bind\tBind the manual root space from this terminal\n'
    printf 'show\tShow the bound root destination\n'
    printf 'clear\tClear the bound root destination\n'
end

complete -c sac -n '__saccade_no_verb; and not __saccade_prev_value_flag' -a '(__saccade_verbs)'
complete -c sac -n '__saccade_verb workspace; and test (__saccade_npos) = 0; and not __saccade_prev_value_flag' -a '(__saccade_workspace_verbs)'
complete -c sac -n '__saccade_verb workspace root; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_root_verbs)'

# static flags
complete -c sac -l db -r -f -d 'select local operations on this database file'
complete -c sac -l repo -r -f -d 'the tracker repo when the working directory is not inside it'
complete -c sac -l actor -r -f -d 'actor name recorded on events'
complete -c sac -l json -d 'machine-readable output'
complete -c sac -l at -r -f -d 'event time override'
complete -c sac -l server -r -f -d 'select this running server for reads, writes, and waits'
complete -c sac -n '__saccade_verb create; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a 'task'
complete -c sac -n '__saccade_verb list; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a 'task'
complete -c sac -n '__saccade_verb skill; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a 'install'
complete -c sac -n '__saccade_verb skill; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a 'check'
complete -c sac -n '__saccade_verb completions; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_shells)'
complete -c sac -n '__saccade_verb cancel; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb reopen; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb claim; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb wait; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb show; and test (__saccade_npos) -ge 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb workspace clean; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates workspaces-clean)'
complete -c sac -n '__saccade_verb workspace restore; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates workspaces-restore)'

# per-verb flag names, exactly as the CLI defines them: fish offers
# option names only on dash-prefixed tokens, so these never leak into
# argument or value positions; global flags stay registered above,
# position-independent beside this set. A registered option's value
# completes only from candidates riding that option's registrations,
# so the value-bearing candidates hang off the same lines.
complete -c sac -n '__saccade_verb cancel' -l explanation -r -f -d 'the human explanation the record keeps'
complete -c sac -n '__saccade_verb reopen' -l explanation -r -f -d 'the human explanation the record keeps'
complete -c sac -n '__saccade_verb send' -l to -r -f -d 'the recipient (i-N)' -a '(__saccade_candidates incarnations)'
complete -c sac -n '__saccade_verb send' -l reply -r -f -d 'the message this one answers (m-N)'
complete -c sac -n '__saccade_verb send' -l task -r -f -d 'task context riding the exchange (t-N)' -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb send' -l stdin -d 'read the body from stdin instead of an argument'
complete -c sac -n '__saccade_verb show' -l stdin -d 'read one id per line from stdin'
complete -c sac -n '__saccade_verb search' -l history -d 'the revisions where terms appeared or disappeared'
complete -c sac -n '__saccade_verb search' -l within -r -f -d 'narrow to one task revisions (t-N or t-N@rA..rB)'
complete -c sac -n '__saccade_verb offer' -r -f -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb accept' -r -f -a '(__saccade_candidates tasks)'
complete -c sac -n '__saccade_verb reject' -l grounds -r -f -d 'the grounds that reopen the task'
complete -c sac -n '__saccade_verb board' -r -f -a 'edit check checkpoint commit update discard get diff log path'
complete -c sac -n '__saccade_verb board; and __fish_seen_subcommand_from get diff' -r -f -a 't-(commandline -ot)'
complete -c sac -n '__saccade_verb serve' -l bind -r -f -d 'the address to bind'
complete -c sac -n '__saccade_verb serve' -l port -r -f -d 'the port to serve on'
complete -c sac -n '__saccade_verb launch' -l workspace -r -f -d 'the workspace directory to work in' -a '(__saccade_candidates workspaces)' -k
complete -c sac -n '__saccade_verb launch' -l workspace -r -f -a '(__fish_complete_directories)'
complete -c sac -n '__saccade_verb launch' -l resume -r -f -d 'continue this saved incarnation (i-N)' -a '(__saccade_candidates incarnations)'
complete -c sac -n '__saccade_verb launch' -l new -d 'provision a NEW workspace and launch there'
complete -c sac -n '__saccade_verb launch' -l base -r -f -d 'the Git base for --new'
complete -c sac -n '__saccade_verb launch' -l ask -r -f -d 'the initial question for a fresh peer'
complete -c sac -n '__saccade_verb launch' -l harness -r -f -d 'the harness a fresh launch runs; a resume runs its recorded kind' -a 'pi claude-code'
complete -c sac -n '__saccade_verb launch' -l here -d 'launch through the host, then attach it in this terminal'
complete -c sac -n '__saccade_verb workspace new' -l base -r -f -d 'the Git base: a ref, defaulting to the primary HEAD'
complete -c sac -n '__saccade_verb workspace switch' -l resume -d 'resume the session in the new directory'
complete -c sac -n '__saccade_verb workspace switch' -l to -r -f -d 'the selected directory: it must already exist' -a '(__saccade_candidates workspaces)' -k
complete -c sac -n '__saccade_verb workspace switch' -l to -r -f -a '(__fish_complete_directories)'
complete -c sac -n '__saccade_verb workspace switch; and test (__saccade_npos) = 1; and not __saccade_prev_value_flag' -a '(__saccade_candidates incarnations)'
complete -c sac -n '__saccade_verb workspace reconcile' -l apply -d 'record the verified Cleaned facts'
complete -c sac -n '__saccade_verb migrate' -l out -r -f -d 'the fresh tracker to write; must not exist'
complete -c sac -n '__saccade_verb clone' -l at -r -f -d 'the storage seq to copy through, inclusive'
complete -c sac -n '__saccade_verb clone' -l out -r -f -d 'the fresh tracker to write; must not exist'
complete -c sac -n '__saccade_verb wait' -l after -r -f -d 'the global position matched strictly after'
complete -c sac -n '__saccade_verb wait' -l until -r -f -a 'activity offer'
