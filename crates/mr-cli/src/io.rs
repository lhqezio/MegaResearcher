//! Production `UserIo` over stdin/stdout.

use async_trait::async_trait;
use std::io::{self, BufRead, Write};

use megaresearcher_research::phases::UserIo;

pub struct StdinStdoutIo;

#[async_trait]
impl UserIo for StdinStdoutIo {
    async fn print(&self, text: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        out.write_all(text.as_bytes())?;
        out.flush()?;
        Ok(())
    }
    async fn read_line(&self) -> io::Result<String> {
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        Ok(line)
    }
}
