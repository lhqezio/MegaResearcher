//! A provider wrapper that feeds `CostTracker` from the stream, and the
//! single top-right cost meter (`$X.XX · Nk`). One number (Rubin: no chart).
//!
//! SPDX-License-Identifier: GPL-3.0

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use claurst_api::{
    LlmProvider, ModelInfo, ProviderCapabilities, ProviderError, ProviderRequest, ProviderResponse,
    ProviderStatus, StreamEvent,
};
use claurst_core::cost::CostTracker;
use claurst_core::provider_id::ProviderId;
use futures::stream::StreamExt as _;
use futures::Stream;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::theme::ColorPalette;

pub struct CountingProvider {
    inner: Arc<dyn LlmProvider>,
    tracker: Arc<CostTracker>,
}

impl CountingProvider {
    pub fn new(inner: Arc<dyn LlmProvider>, tracker: Arc<CostTracker>) -> Self {
        Self { inner, tracker }
    }

    pub fn tracker(&self) -> &Arc<CostTracker> {
        &self.tracker
    }
}

#[async_trait]
impl LlmProvider for CountingProvider {
    fn id(&self) -> &ProviderId {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.inner.create_message(request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let tracker = self.tracker.clone();
        let inner_stream = self.inner.create_message_stream(request).await?;
        // Wrap the stream: inspect each event, feed usage into the tracker,
        // and re-emit the event unchanged.
        let mapped = inner_stream.map(move |res| {
            if let Ok(ev) = &res {
                match ev {
                    StreamEvent::MessageStart { usage, .. } => {
                        tracker.add_usage(
                            usage.input_tokens,
                            usage.output_tokens,
                            usage.cache_creation_input_tokens,
                            usage.cache_read_input_tokens,
                        );
                    }
                    StreamEvent::MessageDelta { usage: Some(u), .. } => {
                        tracker.add_usage(
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_creation_input_tokens,
                            u.cache_read_input_tokens,
                        );
                    }
                    _ => {}
                }
            }
            res
        });
        Ok(Box::pin(mapped))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        self.inner.list_models().await
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        self.inner.health_check().await
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }
}

/// Render the single cost meter: `$X.XX · Nk` (dollars + thousands of tokens).
pub fn render_cost_meter(
    frame: &mut ratatui::Frame,
    area: Rect,
    tracker: &CostTracker,
    theme: &ColorPalette,
) {
    let text = format!(
        "${:.2} · {}k",
        tracker.total_cost_usd(),
        tracker.total_tokens() / 1000
    );
    let line = Line::from(vec![Span::styled(
        text,
        Style::default().fg(theme.text_light),
    )]);
    frame.render_widget(Paragraph::new(line).alignment(Alignment::Right), area);
}
