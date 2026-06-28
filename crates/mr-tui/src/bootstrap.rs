//! Terminal bootstrap — enable raw mode, enter the alternate screen, enable
//! mouse capture. Adapted from claurst-tui's `setup_terminal` (drop bracketed
//! paste + kitty keyboard flags — a research tree does not need them).
//!
//! SPDX-License-Identifier: GPL-3.0

use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::{self, Stdout};

fn restore_terminal_cleanup() -> io::Result<()> {
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}

pub fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    let main_thread_id = std::thread::current().id();
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Only the main thread restores the terminal — tokio worker panics
        // must not wreck the live TUI while the render loop is still running.
        if std::thread::current().id() == main_thread_id {
            let _ = disable_raw_mode();
            let _ = restore_terminal_cleanup();
            let _ = execute!(io::stdout(), crossterm::cursor::Show);
        }
        original_hook(panic_info);
    }));
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    let _ = execute!(terminal.backend_mut(), crossterm::terminal::SetTitle(""));
    restore_terminal_cleanup()?;
    terminal.show_cursor()?;
    Ok(())
}
