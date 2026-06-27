//! A deterministic `LlmProvider` that emits canned `StreamEvent` turn sequences.
//!
//! Used by the worker-contract tests (Task 5) and the Phase 4 orchestrator
//! integration test. Lives in test infra, not the shipped library.

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::stream::iter;
use futures::Stream;

use claurst_api::{
    LlmProvider, ModelInfo, ProviderCapabilities, ProviderError, ProviderRequest, ProviderResponse,
    ProviderStatus, StreamEvent, SystemPromptStyle,
};
use claurst_core::provider_id::ProviderId;

/// A fake provider that returns scripted turn sequences.
///
/// Each call to `create_message_stream` pops the next turn (a `Vec<StreamEvent>`).
/// Once the scripted turns are exhausted, it re-emits the last turn — so a
/// worker that keeps issuing tool-use turns terminates on `max_turns`
/// deterministically rather than panicking on an out-of-range index.
pub struct FakeProvider {
    id: ProviderId,
    turns: Vec<Vec<StreamEvent>>,
    call_index: AtomicUsize,
}

impl FakeProvider {
    pub fn new(id: &str, turns: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            id: ProviderId::new(id),
            turns,
            call_index: AtomicUsize::new(0),
        }
    }

    /// Number of `create_message_stream` calls so far.
    pub fn call_count(&self) -> usize {
        self.call_index.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for FakeProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "fake"
    }

    async fn create_message(
        &self,
        _req: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        Err(ProviderError::Other {
            provider: self.id.clone(),
            message: "FakeProvider: use create_message_stream".into(),
            status: None,
            body: None,
        })
    }

    async fn create_message_stream(
        &self,
        _req: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let idx = self.call_index.fetch_add(1, Ordering::SeqCst);
        let events = self
            .turns
            .get(idx)
            .or_else(|| self.turns.last())
            .cloned()
            .unwrap_or_default();
        let stream = iter(events.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(vec![])
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: SystemPromptStyle::TopLevel,
        }
    }
}
