//! The /api/v1 surface over a real loopback socket: one envelope, records
//! back or a typed error, the tier law holding through HTTP exactly as it
//! holds in process.

mod cli_workspace;
mod command;
mod common;
mod destination;
mod lifecycle;
mod pages;
mod pty_guard;
mod tags;
mod workspace_ops;
