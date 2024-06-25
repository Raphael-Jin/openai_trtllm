use anyhow::{Context, Result};
use tonic::transport::Channel;
use axum::routing::{get, post};
use axum::Router;
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;

use crate::config::Config;
use crate::history::HistoryBuilder;
use crate::routes;
use crate::state::AppState;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use tokio::time::Duration;

async fn connect_with_retry(config: &Config, max_retries: u32) -> Result<GrpcInferenceServiceClient<Channel>> {
    let mut retries = 0;
    let retry_delay = Duration::from_secs(1); // 1 second delay between retries

    loop {
        match GrpcInferenceServiceClient::connect(config.triton_endpoint.clone()).await {
            Ok(client) => return Ok(client),
            Err(e) => {
                retries += 1;
                if retries >= max_retries {
                    return Err(e).context("failed to connect triton endpoint after max retries");
                }
                eprintln!("Connection attempt {} failed: {}. Retrying in {:?}...", retries, e, retry_delay);
                tokio::time::sleep(retry_delay).await;
            }
        }
    }
}

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    tracing::info!("Connecting to triton endpoint: {}", config.triton_endpoint);

    let max_int_value = i32::MAX as u32;

    let grpc_client = connect_with_retry(&config, max_int_value).await?;

    let history_builder =
        HistoryBuilder::new(&config.history_template, &config.history_template_file)?;
    let state = AppState {
        grpc_client,
        history_builder,
    };

    let app = Router::new()
        .route("/v1/completions", post(routes::compat_completions))
        .route(
            "/v1/chat/completions",
            post(routes::compat_chat_completions),
        )
        .with_state(state)
        .layer(OtelAxumLayer::default())
        .route("/health_check", get(routes::health_check));

    let address = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting server at {}", address);

    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");

    opentelemetry::global::shutdown_tracer_provider();
}
