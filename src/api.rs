use std::time::Duration;

use serde::Deserialize;
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, sleep_until};
use tracing::{debug, warn};

const POLL_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone)]
pub enum ModelsSnapshot {
    Loaded {
        version: Option<String>,
        models: Vec<LoadedModel>,
    },
    Unreachable {
        #[allow(dead_code)] // surfaced via debug logs; future TUI tooltip
        reason: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // model/size/digest deserialize-only — preserved for /api/ps fidelity
pub struct LoadedModel {
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: Option<ModelDetails>,
    #[serde(default, alias = "context_length")]
    pub context_length: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // format unused in current TUI columns; preserved for fidelity
pub struct ModelDetails {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PsResp {
    #[serde(default)]
    models: Vec<LoadedModel>,
}

#[derive(Debug, Deserialize)]
struct VersionResp {
    version: String,
}

pub async fn poll_models(
    base_url: String,
    tx: mpsc::Sender<ModelsSnapshot>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            warn!(error = %err, "failed to build polling http client");
            return;
        }
    };

    loop {
        let snap = fetch_snapshot(&client, &base_url).await;
        if tx.send(snap).await.is_err() {
            debug!("models channel closed; poller exiting");
            return;
        }

        let next = Instant::now() + POLL_INTERVAL;
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { return; }
            }
            _ = sleep_until(next) => {}
        }
    }
}

async fn fetch_snapshot(client: &reqwest::Client, base_url: &str) -> ModelsSnapshot {
    let version = match client.get(format!("{}/api/version", base_url)).send().await {
        Ok(r) if r.status().is_success() => r.json::<VersionResp>().await.ok().map(|v| v.version),
        Ok(r) => {
            return ModelsSnapshot::Unreachable {
                reason: format!("/api/version → HTTP {}", r.status()),
            };
        }
        Err(err) => {
            return ModelsSnapshot::Unreachable {
                reason: format!("{}", err),
            };
        }
    };

    let ps = match client.get(format!("{}/api/ps", base_url)).send().await {
        Ok(r) if r.status().is_success() => match r.json::<PsResp>().await {
            Ok(p) => p,
            Err(err) => {
                return ModelsSnapshot::Unreachable {
                    reason: format!("/api/ps decode error: {}", err),
                };
            }
        },
        Ok(r) => {
            return ModelsSnapshot::Unreachable {
                reason: format!("/api/ps → HTTP {}", r.status()),
            };
        }
        Err(err) => {
            return ModelsSnapshot::Unreachable {
                reason: format!("{}", err),
            };
        }
    };

    ModelsSnapshot::Loaded {
        version,
        models: ps.models,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_api_ps_fixture() {
        let json = r#"{"models":[{"name":"qwen3:14b","model":"qwen3:14b","size":9000000000,"size_vram":8500000000,"expires_at":"2026-05-05T11:00:00Z","digest":"sha256-abc","details":{"format":"gguf","family":"qwen3","parameter_size":"14B","quantization_level":"Q4_K_M"},"context_length":4096}]}"#;
        let p: PsResp = serde_json::from_str(json).unwrap();
        assert_eq!(p.models.len(), 1);
        assert_eq!(p.models[0].name, "qwen3:14b");
        assert_eq!(p.models[0].context_length, Some(4096));
        assert_eq!(
            p.models[0].details.as_ref().unwrap().family.as_deref(),
            Some("qwen3")
        );
    }

    #[test]
    fn parse_api_ps_empty() {
        let json = r#"{"models":[]}"#;
        let p: PsResp = serde_json::from_str(json).unwrap();
        assert!(p.models.is_empty());
    }
}
