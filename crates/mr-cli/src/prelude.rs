//! Resolve an `Arc<dyn LlmProvider>` by mirroring claurst's CLI prelude:
//! `Settings::load_hierarchical` -> `effective_config` -> apply overrides ->
//! `selected_provider_id` -> `resolve_anthropic_auth_async` -> `ClientConfig` ->
//! `ProviderRegistry::from_config` -> `default_provider`.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use claurst_api::{client::ClientConfig, LlmProvider, ProviderRegistry};

/// Returns the provider + the resolved model string.
pub async fn resolve_provider(
    cwd: &Path,
    model: Option<String>,
    provider_id: Option<String>,
    api_key: Option<String>,
) -> anyhow::Result<(Arc<dyn LlmProvider>, String)> {
    // MrConfig (~/.config/mr/config.toml) is the first source — the TUI and
    // headless `mr` share it. Explicit args override the config.
    let mr_cfg = mr_tui::config::MrConfig::load();
    let model = model.or(mr_cfg.model);
    let provider_id = provider_id.or(mr_cfg.provider);
    let api_key = api_key.or(mr_cfg.api_key);
    let settings = claurst_core::config::Settings::load_hierarchical(cwd).await;
    let mut config = settings.effective_config();
    if let Some(m) = model {
        config.model = Some(m);
    }
    if let Some(p) = provider_id {
        config.provider = Some(p);
    }
    if let Some(k) = api_key {
        config.api_key = Some(k);
    }

    let active = config.selected_provider_id().to_string();
    let (key, use_bearer) = if active == "anthropic" {
        config
            .resolve_anthropic_auth_async()
            .await
            .unwrap_or((String::new(), false))
    } else {
        (
            config.resolve_provider_api_key(&active).unwrap_or_default(),
            false,
        )
    };
    let client_config = ClientConfig {
        api_key: key.clone(),
        api_base: config.resolve_anthropic_api_base(),
        use_bearer_auth: use_bearer,
        ..Default::default()
    };
    let registry = ProviderRegistry::from_config(&config, client_config);
    let provider = registry
        .default_provider()
        .cloned()
        .context("no provider registered — set an API key (e.g. ANTHROPIC_API_KEY) or run `claurst auth login`")?;
    let model = config
        .model
        .clone()
        .unwrap_or_else(|| "claude-sonnet-4-6".to_string());
    Ok((provider, model))
}
