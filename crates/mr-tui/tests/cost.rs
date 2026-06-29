mod common;

use std::sync::Arc;

use claurst_api::{LlmProvider, ProviderRequest, StreamEvent};
use claurst_core::cost::CostTracker;
use claurst_core::types::UsageInfo;
use mr_tui::cost::{render_cost_meter, CountingProvider};
use mr_tui::theme::research;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

use common::fake_provider::FakeProvider;

fn usage_turn(input: u64, output: u64) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart {
            id: "m".into(),
            model: "fake".into(),
            usage: UsageInfo {
                input_tokens: input,
                output_tokens: 0,
                ..Default::default()
            },
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: claurst_core::types::ContentBlock::Text {
                text: String::new(),
            },
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "hi".into(),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta {
            stop_reason: Some(claurst_api::StopReason::EndTurn),
            usage: Some(UsageInfo {
                input_tokens: 0,
                output_tokens: output,
                ..Default::default()
            }),
        },
        StreamEvent::MessageStop,
    ]
}

#[tokio::test]
async fn counting_provider_feeds_tracker_from_stream() {
    let fake = Arc::new(FakeProvider::new("fake", vec![usage_turn(1000, 500)]));
    let tracker = CostTracker::with_model("claude-sonnet-4-6");
    let counting = CountingProvider::new(fake.clone() as Arc<dyn LlmProvider>, tracker.clone());
    let req = ProviderRequest {
        model: "fake".into(),
        messages: vec![],
        system_prompt: None,
        tools: vec![],
        max_tokens: 1,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: serde_json::Value::Null,
    };
    let mut stream = counting.create_message_stream(req).await.unwrap();
    use futures::StreamExt;
    while let Some(_ev) = stream.next().await {
        // drain
    }
    // input 1000 (from MessageStart) + output 500 (from MessageDelta) = 1500 tokens.
    assert_eq!(tracker.total_tokens(), 1500);
}

#[test]
fn cost_meter_renders_dollars_and_tokens() {
    let tracker = CostTracker::with_model("claude-sonnet-4-6");
    tracker.add_usage(100_000, 50_000, 0, 0);
    let theme = research();
    let mut terminal = Terminal::new(TestBackend::new(40, 3)).unwrap();
    terminal
        .draw(|f| {
            render_cost_meter(f, f.area(), &tracker, &theme);
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let content: String = buf
        .content()
        .iter()
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(content.contains('$'), "dollar sign: {content}");
    assert!(content.contains('k'), "k suffix: {content}");
}
