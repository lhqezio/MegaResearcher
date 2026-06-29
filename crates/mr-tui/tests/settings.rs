use mr_tui::app::{App, Surface};
use mr_tui::config::MrConfig;
use mr_tui::surface::settings::SettingsState;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

#[test]
fn mrconfig_round_trips_to_toml() {
    let cfg = MrConfig {
        max_parallel: 8,
        on_escalate: "fail".into(),
        ..Default::default()
    };
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("config.toml");
    std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
    let loaded: MrConfig = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(loaded.max_parallel, 8);
    assert_eq!(loaded.on_escalate, "fail");
}

#[test]
fn mrconfig_needs_provider_key_when_absent() {
    let cfg = MrConfig::default();
    assert!(cfg.needs_provider_key());
    let cfg = MrConfig {
        provider: Some("anthropic".into()),
        api_key: Some("sk-ant-x".into()),
        ..Default::default()
    };
    assert!(!cfg.needs_provider_key());
}

#[test]
fn settings_surface_renders_masked_key() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Settings;
    let cfg = MrConfig {
        api_key: Some("sk-ant-1234567890".into()),
        ..Default::default()
    };
    app.settings = Some(SettingsState::from_config(cfg));
    let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let content: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    // The API key is masked: "sk-ant-" + bullets, no plaintext tail.
    assert!(
        !content.contains("1234567890"),
        "key must be masked: {content}"
    );
    assert!(content.contains('•'), "masked with bullets: {content}");
}
