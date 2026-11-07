use super::*;

pub(super) fn serve_door(cli: &Cli, bind: &str, port: &u16) -> Result<String, Fail> {
    if cli.server.is_some() {
        return Err(Fail::Usage(
            "serve starts a server; --server selects one for clients (--db names the file to serve)".into(),
        ));
    }
    // the serving repo is the one resolved here - effects and bytes
    // belong to the declared tracker, never the caller's cwd
    let repo_root = resolve_repo(cli).ok();
    let db_path = match &cli.db {
        Some(db) => db.clone(),
        None => {
            let root = repo_root.as_ref().ok_or_else(|| {
                Fail::Usage(
                    "no repo resolved: --repo names the tracker, --db names the file to serve"
                        .into(),
                )
            })?;
            saccade::paths::db_at(root)
        }
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
    match runtime.block_on(saccade::serve::run(&db_path, repo_root, bind, *port)) {
        Err(e) => Err(Fail::Usage(e)),
        Ok(infallible) => match infallible {},
    }
}
