//! The in-process worker primitive: a fresh-context query loop against an `LlmProvider`.
//!
//! First-cut loop (spec §6), dropping autocompact / normalization / abort:
//! build a ProviderRequest -> stream -> accumulate blocks -> dispatch tool_use
//! -> append tool_result -> loop, bounded by max_turns. The result is the
//! final assistant text only.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use serde_json::Value;

use claurst_api::{
    LlmProvider, ProviderError, ProviderRequest, StopReason, StreamEvent, SystemPrompt,
};
use claurst_core::types::{ContentBlock, Message, ToolDefinition, ToolResultContent, UsageInfo};

use crate::worker_tools::{Tool, ToolResult};

/// Default turn ceiling for a worker. Tunable per worker via `WorkerConfig`.
pub const DEFAULT_MAX_TURNS: u32 = 50;

/// Per-worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub max_turns: u32,
    pub max_tokens: u32,
    pub model: String,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            max_turns: DEFAULT_MAX_TURNS,
            max_tokens: 4096,
            model: "claude-sonnet-4-6".to_string(),
        }
    }
}

/// Why the worker stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStop {
    /// The provider ended the turn with no pending tool calls.
    EndTurn,
    /// The turn ceiling was reached with tool calls still pending.
    MaxTurns,
}

/// Worker-level error.
#[derive(Debug, Clone)]
pub enum WorkerError {
    Provider(ProviderError),
    BadStream(String),
}

/// The outcome of a worker run.
#[derive(Debug, Clone)]
pub struct WorkerOutcome {
    /// The final assistant text (the worker's result, per spec §6).
    pub final_text: String,
    /// Number of provider turns executed.
    pub turns: u32,
    /// Why the loop stopped.
    pub stop: WorkerStop,
    /// Accumulated usage from the final turn.
    pub usage: UsageInfo,
}

/// A worker: a system prompt, a tool set, a provider, and a turn ceiling.
pub struct Worker {
    pub system_prompt: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub provider: Arc<dyn LlmProvider>,
    pub config: WorkerConfig,
    pub output_dir: PathBuf,
}

impl Worker {
    pub fn new(
        system_prompt: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
        provider: Arc<dyn LlmProvider>,
        config: WorkerConfig,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            tools,
            provider,
            config,
            output_dir: output_dir.into(),
        }
    }

    fn tool_defs(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Run the worker against a single user prompt. Returns the final assistant
    /// text plus stop metadata.
    pub async fn run(&self, prompt: &str) -> Result<WorkerOutcome, WorkerError> {
        let mut messages: Vec<Message> = vec![Message::user(prompt.to_string())];
        let tool_defs = self.tool_defs();
        let mut last_assistant_text = String::new();
        let mut usage = UsageInfo::default();

        for turn in 0..self.config.max_turns {
            let req = ProviderRequest {
                model: self.config.model.clone(),
                messages: messages.clone(),
                system_prompt: Some(SystemPrompt::Text(self.system_prompt.clone())),
                tools: tool_defs.clone(),
                max_tokens: self.config.max_tokens,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
                thinking: None,
                provider_options: Value::Object(serde_json::Map::new()),
            };
            let stream = self
                .provider
                .create_message_stream(req)
                .await
                .map_err(WorkerError::Provider)?;
            let (blocks, _stop_reason, turn_usage) = accumulate(stream).await?;
            if let Some(u) = turn_usage {
                usage = u;
            }

            // Pull text + tool_use out of the finalized blocks, preserving order.
            let mut tool_uses: Vec<(String, String, Value)> = Vec::new();
            for block in &blocks {
                match block {
                    ContentBlock::Text { text } => {
                        if !text.is_empty() {
                            last_assistant_text = text.clone();
                        }
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_uses.push((id.clone(), name.clone(), input.clone()));
                    }
                    _ => {}
                }
            }

            messages.push(Message::assistant_blocks(blocks));

            if tool_uses.is_empty() {
                return Ok(WorkerOutcome {
                    final_text: last_assistant_text,
                    turns: turn + 1,
                    stop: WorkerStop::EndTurn,
                    usage,
                });
            }

            // Dispatch each tool use and collect ToolResult blocks.
            let mut results: Vec<ContentBlock> = Vec::new();
            for (id, name, input) in tool_uses {
                let result = match self.tools.iter().find(|t| t.name() == name) {
                    Some(tool) => tool.call(input).await,
                    None => ToolResult::err(format!(
                        "<tool_use_error>unknown tool: {name}</tool_use_error>"
                    )),
                };
                results.push(ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: ToolResultContent::Text(result.content),
                    is_error: Some(result.is_error),
                });
            }
            messages.push(Message::user_blocks(results));
        }

        Ok(WorkerOutcome {
            final_text: last_assistant_text,
            turns: self.config.max_turns,
            stop: WorkerStop::MaxTurns,
            usage,
        })
    }
}

/// Accumulate a stream of `StreamEvent`s into finalized `ContentBlock`s in
/// ascending block-index order, plus the final stop reason and usage.
pub(crate) async fn accumulate(
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
) -> Result<(Vec<ContentBlock>, Option<StopReason>, Option<UsageInfo>), WorkerError> {
    let mut seeds: BTreeMap<usize, ContentBlock> = BTreeMap::new();
    let mut text_bufs: BTreeMap<usize, String> = BTreeMap::new();
    let mut json_bufs: BTreeMap<usize, String> = BTreeMap::new();
    let mut finished: BTreeMap<usize, ContentBlock> = BTreeMap::new();
    let mut stop_reason: Option<StopReason> = None;
    let mut usage: Option<UsageInfo> = None;

    let mut stream = stream;
    while let Some(event) = stream.next().await {
        let event = event.map_err(WorkerError::Provider)?;
        match event {
            StreamEvent::MessageStart { .. } => {}
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                seeds.insert(index, content_block);
            }
            StreamEvent::TextDelta { index, text } => {
                text_bufs.entry(index).or_default().push_str(&text);
            }
            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => {
                json_bufs.entry(index).or_default().push_str(&partial_json);
            }
            StreamEvent::ContentBlockStop { index } => {
                if let Some(mut block) = seeds.remove(&index) {
                    match &mut block {
                        ContentBlock::Text { text } => {
                            if let Some(buf) = text_bufs.remove(&index) {
                                *text = buf;
                            }
                        }
                        ContentBlock::ToolUse { input, .. } => {
                            if let Some(buf) = json_bufs.remove(&index) {
                                if let Ok(value) = serde_json::from_str::<Value>(&buf) {
                                    *input = value;
                                }
                            }
                        }
                        _ => {}
                    }
                    finished.insert(index, block);
                }
            }
            StreamEvent::MessageDelta {
                stop_reason: sr,
                usage: u,
            } => {
                if let Some(sr) = sr {
                    stop_reason = Some(sr);
                }
                if let Some(u) = u {
                    usage = Some(u);
                }
            }
            StreamEvent::MessageStop => break,
            StreamEvent::Error { message, .. } => {
                return Err(WorkerError::BadStream(message));
            }
            // ThinkingDelta / SignatureDelta / ReasoningDelta: ignored (no
            // thinking config in Phase 3).
            _ => {}
        }
    }

    let blocks: Vec<ContentBlock> = finished.into_values().collect();
    Ok((blocks, stop_reason, usage))
}
