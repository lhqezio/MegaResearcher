//! MrConfig — the mr-specific TOML store at ~/.config/mr/config.toml.
//! The file is the source of truth; the Settings screen edits it, headless
//! `mr` reads it (T13). One store, so a save takes effect on the next run.
//!
//! SPDX-License-Identifier: GPL-3.0

use std::io;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MrConfig {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_max_parallel")]
    pub max_parallel: u32,
    #[serde(default = "default_on_escalate")]
    pub on_escalate: String,
    #[serde(default = "default_mcp")]
    pub mcp: bool,
    #[serde(default = "default_cost_ceiling")]
    pub cost_ceiling_sandbox: f64,
    #[serde(default = "default_cost_ceiling")]
    pub cost_ceiling_api: f64,
    #[serde(default = "default_theme")]
    pub theme: String,
}

fn default_max_parallel() -> u32 {
    4
}
fn default_on_escalate() -> String {
    "pause".to_string()
}
fn default_mcp() -> bool {
    true
}
fn default_cost_ceiling() -> f64 {
    5.0
}
fn default_theme() -> String {
    "research".to_string()
}

impl Default for MrConfig {
    fn default() -> Self {
        Self {
            provider: None,
            api_key: None,
            model: None,
            max_parallel: default_max_parallel(),
            on_escalate: default_on_escalate(),
            mcp: default_mcp(),
            cost_ceiling_sandbox: default_cost_ceiling(),
            cost_ceiling_api: default_cost_ceiling(),
            theme: default_theme(),
        }
    }
}

impl MrConfig {
    /// `~/.config/mr/config.toml` (platform config dir + "mr/config.toml").
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mr")
            .join("config.toml")
    }

    /// Load from the config file, falling back to `Default` if missing/unreadable.
    pub fn load() -> Self {
        let path = Self::config_path();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Write to the config file, creating the parent dir.
    pub fn save(&self) -> io::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = toml::to_string(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, s)
    }

    /// True if the provider or API key is unset (first-run detection).
    pub fn needs_provider_key(&self) -> bool {
        self.provider
            .as_ref()
            .map(|s| s.trim().is_empty())
            .unwrap_or(true)
            || self
                .api_key
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
    }
}
