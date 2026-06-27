//! Run-id generation. The orchestrator skill's rule: `YYYY-MM-DD-HHMM-<6hex>`
//! (UTC date+time + 6 lowercase hex chars).

use std::io;

use chrono::Utc;

/// Assemble a run-id from a pre-formatted UTC stamp (`YYYY-MM-DD-HHMM`) and a
/// 6-char lowercase hex string. Pure; deterministic; unit-tested directly.
pub fn run_id_from_parts(stamp: &str, hex6: &str) -> String {
    format!("{}-{}", stamp, hex6)
}

/// Generate a run-id from the current UTC time and 3 random bytes (6 hex chars).
pub fn generate_run_id() -> io::Result<String> {
    let stamp = Utc::now().format("%Y-%m-%d-%H%M").to_string();
    let mut buf = [0u8; 3];
    getrandom::getrandom(&mut buf).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let hex6 = format!("{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2]);
    Ok(run_id_from_parts(&stamp, &hex6))
}
