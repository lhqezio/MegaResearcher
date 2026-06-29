use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use mr_tui::app::{App, AppEvent, Surface};

#[test]
fn start_submits_question_to_converge() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Start;
    app.question = "can SAEs surface causal circuits?".into();
    let ev = app.handle_key(KeyEvent::new_with_kind(
        KeyCode::Enter,
        KeyModifiers::NONE,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev, AppEvent::SubmitQuestion));
    // After submit, the app transitions to Converge (the loop acts on
    // SubmitQuestion by transitioning; here we assert the event).
}

#[test]
fn quit_on_q_or_ctrl_c() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    let ev_q = app.handle_key(KeyEvent::new_with_kind(
        KeyCode::Char('q'),
        KeyModifiers::NONE,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev_q, AppEvent::Quit));
    let ev_c = app.handle_key(KeyEvent::new_with_kind(
        KeyCode::Char('c'),
        KeyModifiers::CONTROL,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev_c, AppEvent::Quit));
}

#[test]
fn s_key_goes_to_settings() {
    let mut app = App::new(std::path::PathBuf::from("/tmp"), None);
    app.surface = Surface::Start;
    let ev = app.handle_key(KeyEvent::new_with_kind(
        KeyCode::Char('s'),
        KeyModifiers::NONE,
        KeyEventKind::Press,
    ));
    assert!(matches!(ev, AppEvent::ToSurface(Surface::Settings)));
}

#[test]
fn start_renders_ghosted_example() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    let app = App::new(std::path::PathBuf::from("/tmp"), None);
    let mut terminal = Terminal::new(TestBackend::new(80, 10)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let content: String = buf
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains("What do you want to know?"));
}
