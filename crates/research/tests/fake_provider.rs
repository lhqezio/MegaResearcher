//! Isolation test for the FakeProvider itself (independent of the worker).

mod common;

use std::pin::Pin;

use claurst_api::{LlmProvider, ProviderRequest, StopReason, StreamEvent};
use claurst_core::types::{ContentBlock, Message, UsageInfo};
use futures::Stream;
use futures::StreamExt;

use common::fake_provider::FakeProvider;

fn dummy_request() -> ProviderRequest {
    ProviderRequest {
        model: "fake-model".into(),
        messages: vec![Message::user("hi")],
        system_prompt: None,
        tools: vec![],
        max_tokens: 1024,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: serde_json::json!({}),
    }
}

#[tokio::test]
async fn test_fake_provider_emits_scripted_events() {
    let turn = vec![
        StreamEvent::MessageStart {
            id: "m1".into(),
            model: "fake-model".into(),
            usage: UsageInfo::default(),
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "hello".into(),
        },
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(UsageInfo::default()),
        },
        StreamEvent::MessageStop,
    ];
    let provider = FakeProvider::new("fake", vec![turn]);
    assert_eq!(provider.call_count(), 0);

    let stream: Pin<Box<dyn Stream<Item = _> + Send>> = provider
        .create_message_stream(dummy_request())
        .await
        .unwrap();
    futures::pin_mut!(stream);
    let mut collected = Vec::new();
    while let Some(item) = stream.next().await {
        collected.push(item.unwrap());
    }
    assert_eq!(collected.len(), 6);
    assert!(matches!(collected[0], StreamEvent::MessageStart { .. }));
    assert!(matches!(
        collected[2],
        StreamEvent::TextDelta { text: _, .. }
    ));
    assert!(matches!(collected[5], StreamEvent::MessageStop));
    assert_eq!(provider.call_count(), 1);
}

#[tokio::test]
async fn test_fake_provider_repeats_last_turn_when_exhausted() {
    let turn = vec![StreamEvent::MessageStop];
    let provider = FakeProvider::new("fake", vec![turn]);
    for _ in 0..3 {
        let mut stream = provider
            .create_message_stream(dummy_request())
            .await
            .unwrap();
        while stream.next().await.is_some() {}
    }
    assert_eq!(provider.call_count(), 3);
}
