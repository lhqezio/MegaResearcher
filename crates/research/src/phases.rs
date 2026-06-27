//! Front-half guided sessions (design §6/§8/§67). A `GuidedSession` is a
//! multi-turn LLM conversation on `LlmProvider` where the flow body is the first
//! user message; the model may call tools (e.g. `ScopedWrite` to draft the
//! spec/plan) between turns. `run_to_checkpoint` runs the model to a natural
//! pause (a turn with no pending tool calls) or the turn ceiling. `drive_session`
//! is the approval-gate driver: it prints assistant text, reads a user line,
//! and either advances through artifact gates (Rust-enforced: the artifact file
//! must exist before an approval is accepted) or injects the line as the next
//! user turn. Reuses `worker::accumulate` so existing `FakeProvider` test infra
//! works unchanged (FakeProvider scripts only `create_message_stream`).

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use claurst_api::{LlmProvider, ProviderError, ProviderRequest, SystemPrompt};
use claurst_core::types::{ContentBlock, Message, ToolDefinition, ToolResultContent, UsageInfo};

use crate::worker::accumulate;
use crate::worker_tools::{Tool, ToolResult};

/// A fixed system prompt for guided sessions: stay in your lane, drive the
/// flow body, do not invoke external skills, use only the provided tools.
const SESSION_SYSTEM_PROMPT: &str = "\
You are a MegaResearcher guided-session facilitator. Follow the flow body given \
as the first user message exactly. Work one step at a time. Use only the tools \
provided. Do not invoke, reference, or pretend to call any external skill or \
plugin. When you reach an approval gate, state plainly what you produced and \
where, then ask the user to approve.";

/// Why a session errored.
#[derive(Debug)]
pub enum SessionError {
    Provider(ProviderError),
    BadStream(String),
    Io(io::Error),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(e) => write!(f, "provider error: {e}"),
            Self::BadStream(s) => write!(f, "bad stream: {s}"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}
impl std::error::Error for SessionError {}
impl From<ProviderError> for SessionError {
    fn from(e: ProviderError) -> Self {
        Self::Provider(e)
    }
}
impl From<io::Error> for SessionError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<crate::worker::WorkerError> for SessionError {
    fn from(e: crate::worker::WorkerError) -> Self {
        match e {
            crate::worker::WorkerError::Provider(p) => Self::Provider(p),
            crate::worker::WorkerError::BadStream(s) => Self::BadStream(s),
        }
    }
}

/// A natural pause point in a guided session.
#[derive(Debug, Clone)]
pub enum Checkpoint {
    /// The model ended a turn with no pending tool calls.
    EndTurn {
        assistant_text: String,
        turns: u32,
        usage: UsageInfo,
    },
    /// The per-checkpoint turn ceiling was reached with tool calls pending.
    MaxTurns { assistant_text: String, turns: u32 },
}

/// Abstraction over user I/O so `drive_session` is testable with a fake.
#[async_trait]
pub trait UserIo: Send + Sync {
    async fn print(&self, text: &str) -> io::Result<()>;
    async fn read_line(&self) -> io::Result<String>;
}

/// One approval gate: an artifact that must exist on disk before the user's
/// approval is accepted, plus a human label.
#[derive(Debug, Clone)]
pub struct Gate {
    pub artifact: PathBuf,
    pub label: String,
}

/// The outcome of `drive_session`.
#[derive(Debug, Clone)]
pub enum DriveOutcome {
    /// All gates advanced (the run's artifacts exist and the user approved).
    Approved { gates_passed: usize },
    /// The session hit its turn ceiling before approval.
    MaxTurns,
}

/// A multi-turn guided LLM session.
pub struct GuidedSession {
    messages: Vec<Message>,
    tools: Vec<Arc<dyn Tool>>,
    provider: Arc<dyn LlmProvider>,
    model: String,
    max_tokens: u32,
    max_turns: u32,
}

impl GuidedSession {
    /// Seed the session with `flow_body` as the first user message.
    pub fn new(
        flow_body: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
        provider: Arc<dyn LlmProvider>,
        model: impl Into<String>,
        max_tokens: u32,
        max_turns: u32,
    ) -> Self {
        Self {
            messages: vec![Message::user(flow_body.into())],
            tools,
            provider,
            model: model.into(),
            max_tokens,
            max_turns,
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

    /// Append a user message (an answer, a revision request, or steering text).
    pub fn inject_user(&mut self, text: &str) {
        self.messages.push(Message::user(text.to_string()));
    }

    /// Run the model until a turn has no pending tool calls (a checkpoint) or
    /// the per-checkpoint turn ceiling is hit. Tool calls are dispatched
    /// internally, mirroring `worker::Worker::run`.
    pub async fn run_to_checkpoint(&mut self) -> Result<Checkpoint, SessionError> {
        let tool_defs = self.tool_defs();
        let mut last_assistant_text = String::new();
        let mut usage = UsageInfo::default();

        for turn in 0..self.max_turns {
            let req = ProviderRequest {
                model: self.model.clone(),
                messages: self.messages.clone(),
                system_prompt: Some(SystemPrompt::Text(SESSION_SYSTEM_PROMPT.to_string())),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
                thinking: None,
                provider_options: Value::Object(serde_json::Map::new()),
            };
            let stream = self.provider.create_message_stream(req).await?;
            let (blocks, _stop_reason, turn_usage) = accumulate(stream).await?;
            if let Some(u) = turn_usage {
                usage = u;
            }

            // Pull text + tool_use out of the finalized blocks, preserving order.
            let mut tool_uses: Vec<(String, String, Value)> = Vec::new();
            for block in &blocks {
                match block {
                    ContentBlock::Text { text } if !text.is_empty() => {
                        last_assistant_text = text.clone();
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_uses.push((id.clone(), name.clone(), input.clone()));
                    }
                    _ => {}
                }
            }
            self.messages.push(Message::assistant_blocks(blocks));

            if tool_uses.is_empty() {
                return Ok(Checkpoint::EndTurn {
                    assistant_text: last_assistant_text,
                    turns: turn + 1,
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
            self.messages.push(Message::user_blocks(results));
        }

        Ok(Checkpoint::MaxTurns {
            assistant_text: last_assistant_text,
            turns: self.max_turns,
        })
    }
}

/// Drive a guided session through its approval gates.
///
/// Loop: run to a checkpoint, print the assistant text, read a user line. If the
/// line is an approval word and the current gate's artifact exists, advance the
/// gate; when all gates are passed, return `Approved`. If the line is an
/// approval word but the artifact is missing, refuse and continue (the line is
/// not re-injected). Otherwise inject the line as the next user turn. Bounded by
/// `total_ceiling`; a `Checkpoint::MaxTurns` from `run_to_checkpoint` returns
/// `DriveOutcome::MaxTurns` immediately.
pub async fn drive_session(
    session: &mut GuidedSession,
    io: &dyn UserIo,
    gates: Vec<Gate>,
    approve_words: &[&str],
) -> Result<DriveOutcome, SessionError> {
    let total_ceiling = session
        .max_turns
        .saturating_mul(gates.len().max(1) as u32 + 1);
    let mut spent: u32 = 0;
    let mut gate_idx: usize = 0;

    loop {
        if spent >= total_ceiling {
            return Ok(DriveOutcome::MaxTurns);
        }
        let checkpoint = session.run_to_checkpoint().await?;
        let is_max_turns = matches!(checkpoint, Checkpoint::MaxTurns { .. });
        let (text, turns) = match checkpoint {
            Checkpoint::EndTurn {
                assistant_text,
                turns,
                ..
            } => (assistant_text, turns),
            Checkpoint::MaxTurns {
                assistant_text,
                turns,
            } => (assistant_text, turns),
        };
        spent = spent.saturating_add(turns);
        if !text.is_empty() {
            io.print(&text).await?;
            io.print("\n").await?;
        }
        if is_max_turns {
            return Ok(DriveOutcome::MaxTurns);
        }

        let line = io.read_line().await?;
        let trimmed = line.trim().to_lowercase();
        let is_approve = approve_words.iter().any(|w| trimmed == *w);

        if is_approve {
            if gates.is_empty() {
                return Ok(DriveOutcome::Approved { gates_passed: 0 });
            }
            let gate = &gates[gate_idx];
            if gate.artifact.exists() {
                gate_idx += 1;
                if gate_idx >= gates.len() {
                    return Ok(DriveOutcome::Approved {
                        gates_passed: gate_idx,
                    });
                }
                let next = &gates[gate_idx];
                io.print(&format!("Approved {}. Next: {}.\n", gate.label, next.label))
                    .await?;
            } else {
                io.print(&format!(
                    "Can't approve — {} not found. Did you write it? Respond with feedback or approve once it exists.\n",
                    gate.artifact.display()
                ))
                .await?;
            }
        } else {
            session.inject_user(&line);
        }
    }
}
