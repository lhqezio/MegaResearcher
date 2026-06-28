//! The research palette — one theme (Rubin: not seven).
//! Adapted from claurst `theme_colors` shape, with net-new fields the spec
//! §1.4 calls for: alive / killed (dim) / running (accent) / escalation (warn).
//!
//! SPDX-License-Identifier: GPL-3.0

use ratatui::style::Color;

#[derive(Debug, Clone, Copy)]
pub struct ColorPalette {
    pub error: Color,
    pub success: Color,
    pub warning: Color,
    pub info: Color,
    pub action: Color,
    pub disabled: Color,
    pub accent: Color,
    pub secondary_accent: Color,
    pub text_light: Color,
    pub text_dark: Color,
    pub border: Color,
    // Research-specific fields (spec §1.4).
    pub alive: Color,
    pub killed: Color,
    pub running: Color,
    pub escalation: Color,
}

/// The single research (dark) palette. `for_theme("research")` returns this;
/// the settings screen ships exactly one theme option (`research (dark)`).
pub fn research() -> ColorPalette {
    ColorPalette {
        error: Color::Rgb(255, 87, 51),
        success: Color::Rgb(76, 175, 80),
        warning: Color::Rgb(255, 152, 0),
        info: Color::Cyan,
        action: Color::Cyan,
        disabled: Color::DarkGray,
        accent: Color::Cyan,
        secondary_accent: Color::Rgb(233, 30, 99),
        text_light: Color::White,
        text_dark: Color::Black,
        border: Color::DarkGray,
        alive: Color::White,
        killed: Color::DarkGray, // dim — the greyed-kill signature
        running: Color::Cyan,
        escalation: Color::Rgb(255, 152, 0),
    }
}

/// Resolve a palette by name. Only "research" exists; everything else falls
/// back to the research palette (one default, no options).
pub fn for_theme(name: &str) -> ColorPalette {
    let _ = name;
    research()
}
