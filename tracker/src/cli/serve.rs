use super::*;

pub(super) fn serve_door(
    cli: &Cli,
    bind: &str,
    allow_origin: &[String],
    allow_host: &[String],
) -> Result<String, Fail> {
    if cli.server.is_some() {
        return Err(Fail::Usage(
            "serve starts a server; --server selects one for clients (--db names the file to serve)".into(),
        ));
    }
    // the serving repo is the one resolved here - effects and bytes
    // belong to the declared tracker, never the caller's cwd. It is
    // also the port's only source: the declaration file lives at its
    // root, so a serve without a repository has nothing to bind.
    let repo_root = resolve_repo(cli).map_err(|_| {
        Fail::Usage(
            "serve reads its port from workbench.toml at the repository's root; no repository resolved (--repo names one)"
                .into(),
        )
    })?;
    let port = saccade::workbench::tracker_port(&repo_root).map_err(Fail::Usage)?;
    let db_path = match &cli.db {
        Some(db) => db.clone(),
        None => saccade::paths::db_at(&repo_root),
    };
    let _ = tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(saccade::attempts::WarnsTee::beside(&db_path))
        .try_init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Fail::Usage(format!("runtime: {e}")))?;
    match runtime.block_on(saccade::serve::run(
        &db_path,
        Some(repo_root),
        bind,
        port,
        allow_origin.to_vec(),
        allow_host.to_vec(),
    )) {
        Err(e) => Err(Fail::Usage(e)),
        Ok(infallible) => match infallible {},
    }
}
