// MegaResearcher Phase 6b — interactive research TUI.
// The audit trail is the interface: a tree that grows, killed hypotheses
// stay on screen dimmed with a one-line kill reason, the red-team loop
// animates (smith -> critique -> revise ↻, round N/3).
//
// SPDX-License-Identifier: GPL-3.0

pub mod app;
pub mod bootstrap;
pub mod cost;
pub mod figures;
pub mod guard;
pub mod io;
pub mod surface;
pub mod theme;
pub mod widget;

use std::path::Path;

/// Entry point. Sets up the terminal, wraps it in a `TerminalGuard` (RAII
/// restore), drives the `App` state machine over the async event loop, and
/// returns. `TerminalGuard::drop` restores the terminal even on error.
pub async fn run(cwd: &Path) -> anyhow::Result<()> {
    let terminal = bootstrap::setup_terminal()?;
    let mut guard = guard::TerminalGuard::new(terminal);
    let mut app = app::App::new(cwd.to_path_buf());
    app.run(guard.inner_mut()).await
}
