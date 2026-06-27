#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mr_cli::run_cli(std::env::args().collect()).await
}
