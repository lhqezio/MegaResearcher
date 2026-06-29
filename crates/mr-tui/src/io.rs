//! TuiUserIo — the Converge I/O seam. Implements `research::phases::UserIo`
//! by routing `print` into an mpsc channel the App drains into the conversation
//! buffer, and resolving `read_line` from an mpsc channel the App feeds from
//! the inline input field. Same `drive_session` engine as 6a, new view.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::io;

use async_trait::async_trait;
use megaresearcher_research::phases::UserIo;
use tokio::sync::mpsc;

/// The App-side handle: drains prints, feeds input lines.
pub struct TuiUserIoHandle {
    pub print_rx: mpsc::UnboundedReceiver<String>,
    pub input_tx: mpsc::UnboundedSender<String>,
}

/// The session-side I/O. Held by the spawned `drive_session` task.
///
/// `UserIo: Send + Sync`, but `mpsc::UnboundedReceiver` is `Send` and NOT
/// `Sync`. Wrapping the receiver in `tokio::sync::Mutex` makes `TuiUserIo`
/// `Send + Sync` — `read_line` takes the lock before `recv()`.
pub struct TuiUserIo {
    print_tx: mpsc::UnboundedSender<String>,
    input_rx: tokio::sync::Mutex<mpsc::UnboundedReceiver<String>>,
}

/// Construct the paired (session-side, app-side) channels.
pub fn tui_user_io() -> (TuiUserIo, TuiUserIoHandle) {
    let (print_tx, print_rx) = mpsc::unbounded_channel::<String>();
    let (input_tx, input_rx) = mpsc::unbounded_channel::<String>();
    (
        TuiUserIo {
            print_tx,
            input_rx: tokio::sync::Mutex::new(input_rx),
        },
        TuiUserIoHandle { print_rx, input_tx },
    )
}

#[async_trait]
impl UserIo for TuiUserIo {
    async fn print(&self, text: &str) -> io::Result<()> {
        self.print_tx
            .send(text.to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e))
    }

    async fn read_line(&self) -> io::Result<String> {
        self.input_rx
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "input channel closed"))
    }
}
