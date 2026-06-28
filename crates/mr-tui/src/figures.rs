//! Box-drawing + signature glyphs for the research tree.
//! Authored fresh — only `↻` (U+21BB) exists in the claurst `figures.rs`;
//! the rest are inline literals scattered across the chat crate, so we
//! name them here as constants.
//!
//! SPDX-License-Identifier: GPL-3.0

pub const BRANCH: &str = "├"; // U+251C
pub const LAST: &str = "└"; // U+2514
pub const PIPE: &str = "│"; // U+2502
pub const DASH: &str = "─"; // U+2500
pub const PENCIL: &str = "✎"; // U+270E
pub const CHECK: &str = "✓"; // U+2713
pub const CROSS: &str = "✗"; // U+2717
pub const REVISE: &str = "↻"; // U+21BB — the red-team cycling signature
pub const ARROW: &str = "▸"; // U+25B8
pub const COLLAPSE: &str = "▾"; // U+25BE
pub const WARN: &str = "⚠"; // U+26A0
