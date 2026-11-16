// ---- the guard's terminal arms: a real pty, piped prose ----

use crate::common::{run_with_terminals, scratch_db, scrubbed};

/// The guard's terminal arms: a terminal anywhere proceeds, nothing refuses.
#[test]
fn piped_human_prose_passes_and_unattended_refuses() {
    let bin = env!("CARGO_BIN_EXE_sac");
    let db = scratch_db("ws-guard");

    // the task exists; possession (an agent) seeds it offline
    let seed = scrubbed(std::process::Command::new(bin))
        .arg("--db")
        .arg(&db)
        .env("SACCADE_ACTOR", "pi")
        .args(["create", "task", "guarded writes"])
        .output()
        .expect("spawn sac");
    assert!(
        seed.status.success(),
        "seed: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    let args = |sub: &[&str]| {
        let mut full = vec!["--db".to_string(), db.display().to_string()];
        full.extend(sub.iter().map(|s| s.to_string()));
        full
    };

    // piped stdin, terminal stdout and stderr: the human writes
    let out = run_with_terminals(
        bin,
        &args(&[
            "cancel",
            "t-0",
            "--explanation",
            "stopping from a human shell",
        ]),
        true,
        true,
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // the record landed, human-authored
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert!(matches!(
        world.tasks[0].task.state,
        saccade::record::objects::task::TaskState::Cancelled(_)
    ));

    // no terminal anywhere and no SACCADE_ACTOR: refusal, taught
    let out = run_with_terminals(bin, &args(&["claim", "t-0"]), false, false);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no SACCADE_ACTOR and no terminal"), "{err}");
    assert!(err.contains("export SACCADE_ACTOR"), "{err}");

    // and nothing landed: the task stays cancelled
    let conn = saccade::db::open_read(&db).unwrap();
    let world = match saccade::db::load(&conn).unwrap().state {
        saccade::db::LoadState::Full(world) => world,
        _ => panic!("the log folds"),
    };
    assert!(matches!(
        world.tasks[0].task.state,
        saccade::record::objects::task::TaskState::Cancelled(_)
    ));

    std::fs::remove_dir_all(db.parent().unwrap()).unwrap();
}
