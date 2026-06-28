//! RAII terminal restore. `Drop` calls `restore_terminal`, so even an early
//! return or an `?`-propagated error cleans up raw mode.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::io::Stdout;

use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::bootstrap::restore_terminal;

pub struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    pub fn new(terminal: Terminal<CrosstermBackend<Stdout>>) -> Self {
        Self { terminal }
    }

    pub fn inner_mut(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = restore_terminal(&mut self.terminal);
    }
}
