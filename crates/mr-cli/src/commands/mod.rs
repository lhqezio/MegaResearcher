use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use claurst_api::LlmProvider;
use claurst_api::{
    ModelInfo, ProviderCapabilities, ProviderError, ProviderRequest, ProviderResponse,
    ProviderStatus, StreamEvent, SystemPromptStyle,
};
use claurst_core::provider_id::ProviderId;
use futures::Stream;

use crate::Command;

pub async fn dispatch(
    cmd: Command,
    cwd: PathBuf,
    provider: Option<(Arc<dyn LlmProvider>, String)>,
) -> anyhow::Result<()> {
    match cmd {
        Command::List => crate::render::list_runs(&cwd).await,
        Command::Watch { run_dir } => crate::render::watch(&cwd, run_dir).await,
        Command::Init { question } => {
            let p = provider.ok_or_else(|| {
                anyhow::anyhow!(
                    "init requires a provider — set an API key (run `mr` to open settings)"
                )
            })?;
            init::run(&cwd, p, &question).await
        }
        Command::Brainstorm { topic } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("brainstorm requires a provider"))?;
            session::run_session(&cwd, p, "brainstorm", &topic).await
        }
        Command::Spec { topic } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("spec requires a provider"))?;
            session::run_session(&cwd, p, "spec", &topic).await
        }
        Command::Plan { topic } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("plan requires a provider"))?;
            session::run_session(&cwd, p, "plan", &topic).await
        }
        Command::Execute {
            plan,
            paper,
            headless,
            no_mcp,
            on_escalate,
        } => {
            let p = provider.ok_or_else(|| anyhow::anyhow!("execute requires a provider"))?;
            execute::run(&cwd, p, plan, paper, headless, no_mcp, on_escalate).await
        }
        Command::Verify { run_dir } => {
            // Verify does not call the LLM — pass a dummy; the function ignores it.
            let dummy: (Arc<dyn LlmProvider>, String) = (
                Arc::new(DummyProvider) as Arc<dyn LlmProvider>,
                String::new(),
            );
            verify::run(&cwd, dummy, run_dir).await
        }
    }
}

// A zero-cost provider for Verify (which ignores the provider). Avoids pulling
// in a concrete provider when none is configured (e.g. `mr verify` with no API
// key).
struct DummyProvider;

#[async_trait::async_trait]
impl LlmProvider for DummyProvider {
    fn id(&self) -> &ProviderId {
        unreachable!()
    }
    fn name(&self) -> &str {
        "dummy"
    }
    async fn create_message(
        &self,
        _req: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        unreachable!()
    }
    async fn create_message_stream(
        &self,
        _req: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        unreachable!()
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(vec![])
    }
    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        Ok(ProviderStatus::Healthy)
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: false,
            tool_calling: false,
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

pub mod execute;
pub mod init;
pub mod session;
pub mod verify;
