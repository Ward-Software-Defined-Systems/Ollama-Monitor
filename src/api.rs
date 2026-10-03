use std::time::Duration;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, sleep_until};
use tracing::{debug, warn};

const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// One row of the models panel. Field names mirror LMS-Monitor's `api::ModelInfo` so the
/// widget code is a straight port; values are mapped from `/api/tags` + `/api/ps`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    /// "llm" / "vlm" / "embeddings", derived from `capabilities`.
    pub kind: Option<String>,
    /// Weight format ("gguf", "safetensors", …), or "cloud" for remote models.
    pub compatibility_type: Option<String>,
    pub quantization: Option<String>,
    pub max_context_length: Option<u64>,
    /// "loaded" (resident per `/api/ps`), "cloud" (runs on ollama.com), or "not-loaded".
    pub state: String,
}

#[derive(Debug, Clone)]
pub enum ModelsSnapshot {
    Loaded {
        version: Option<String>,
        models: Vec<ModelInfo>,
    },
    Unreachable {
        reason: String,
    },
}

/// `/api/tags` entry. Everything is optional so older Ollama builds (no `capabilities`,
/// `context_length` or `remote_host`) still parse.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TagModel {
    pub name: String,
    pub model: String,
    pub remote_host: Option<String>,
    pub details: Option<ModelDetails>,
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ModelDetails {
    pub format: Option<String>,
    pub quantization_level: Option<String>,
    /// Present in `/api/tags` details on Ollama 0.3x+.
    pub context_length: Option<u64>,
}

/// `/api/ps` entry: a model resident in memory.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct LoadedModel {
    pub name: String,
    pub model: String,
    pub details: Option<ModelDetails>,
    /// Allocated (not max) context; top-level here, unlike `/api/tags`.
    pub context_length: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TagsResp {
    models: Vec<TagModel>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct PsResp {
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
    let base_url = base_url.trim_end_matches('/').to_string();

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

/// One poll: `/api/version` decides reachability, then `/api/tags` (every installed
/// model) and `/api/ps` (which of them are resident) are fetched concurrently and merged.
pub(crate) async fn fetch_snapshot(client: &reqwest::Client, base_url: &str) -> ModelsSnapshot {
    let version = match client.get(format!("{base_url}/api/version")).send().await {
        Ok(r) if r.status().is_success() => r.json::<VersionResp>().await.ok().map(|v| v.version),
        Ok(r) => {
            return ModelsSnapshot::Unreachable {
                reason: format!("/api/version → HTTP {}", r.status()),
            };
        }
        Err(err) => {
            return ModelsSnapshot::Unreachable {
                reason: describe("/api/version", &err),
            };
        }
    };

    let (tags, ps) = tokio::join!(
        get_json::<TagsResp>(client, base_url, "/api/tags"),
        get_json::<PsResp>(client, base_url, "/api/ps"),
    );
    snapshot_from(version, tags.map(|t| t.models), ps.map(|p| p.models))
}

async fn get_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
) -> Result<T, String> {
    let resp = client
        .get(format!("{base_url}{path}"))
        .send()
        .await
        .map_err(|err| describe(path, &err))?;
    if !resp.status().is_success() {
        return Err(format!("{path} → HTTP {}", resp.status()));
    }
    resp.json::<T>()
        .await
        .map_err(|err| format!("{path} decode: {}", root_cause(&err)))
}

/// A partial answer beats none: if only one of tags / ps fails, show what arrived.
fn snapshot_from(
    version: Option<String>,
    tags: Result<Vec<TagModel>, String>,
    ps: Result<Vec<LoadedModel>, String>,
) -> ModelsSnapshot {
    match (tags, ps) {
        (Err(tags_err), Err(ps_err)) => ModelsSnapshot::Unreachable {
            reason: format!("{tags_err}; {ps_err}"),
        },
        (tags, ps) => {
            if let Err(err) = &tags {
                debug!(error = %err, "/api/tags failed; showing resident models only");
            }
            if let Err(err) = &ps {
                debug!(error = %err, "/api/ps failed; residency unknown");
            }
            ModelsSnapshot::Loaded {
                version,
                models: merge_models(tags.unwrap_or_default(), ps.unwrap_or_default()),
            }
        }
    }
}

/// Header-sized error text: the endpoint plus the innermost cause, e.g.
/// `/api/version: Connection refused (os error 61)`, not reqwest's whole chain.
fn describe(path: &str, err: &reqwest::Error) -> String {
    format!("{path}: {}", root_cause(err))
}

fn root_cause(err: &(dyn std::error::Error + 'static)) -> String {
    let mut cur = err;
    while let Some(next) = cur.source() {
        cur = next;
    }
    cur.to_string()
}

/// Merge installed models (`/api/tags`) with resident ones (`/api/ps`) into panel rows,
/// ordered loaded → cloud → not-loaded (stable within each group).
pub fn merge_models(tags: Vec<TagModel>, ps: Vec<LoadedModel>) -> Vec<ModelInfo> {
    let mut ps_matched = vec![false; ps.len()];
    let mut rows = Vec::with_capacity(tags.len() + ps.len());

    for tag in &tags {
        let Some(id) = non_empty(Some(&tag.name)).or_else(|| non_empty(Some(&tag.model))) else {
            continue;
        };
        let resident_idx = ps.iter().position(|p| {
            [p.name.as_str(), p.model.as_str()]
                .iter()
                .any(|n| !n.is_empty() && (*n == id || *n == tag.model))
        });
        if let Some(i) = resident_idx {
            ps_matched[i] = true;
        }
        let resident = resident_idx.map(|i| &ps[i]);
        let cloud = is_cloud(&id, tag.remote_host.as_deref());
        let details = tag.details.as_ref();
        let resident_details = resident.and_then(|p| p.details.as_ref());

        rows.push(ModelInfo {
            kind: kind_from_capabilities(tag.capabilities.as_deref()),
            compatibility_type: if cloud {
                Some("cloud".into())
            } else {
                non_empty(details.and_then(|d| d.format.as_ref()))
                    .or_else(|| non_empty(resident_details.and_then(|d| d.format.as_ref())))
            },
            quantization: non_empty(details.and_then(|d| d.quantization_level.as_ref()))
                .or_else(|| non_empty(resident_details.and_then(|d| d.quantization_level.as_ref()))),
            max_context_length: details
                .and_then(|d| d.context_length)
                .or_else(|| resident.and_then(|p| p.context_length)),
            state: if cloud {
                "cloud"
            } else if resident.is_some() {
                "loaded"
            } else {
                "not-loaded"
            }
            .into(),
            id,
        });
    }

    // Resident but missing from tags (e.g. /api/tags failed this poll).
    for (p, _) in ps.iter().zip(&ps_matched).filter(|(_, matched)| !**matched) {
        let Some(id) = non_empty(Some(&p.name)).or_else(|| non_empty(Some(&p.model))) else {
            continue;
        };
        let cloud = is_cloud(&id, None);
        let details = p.details.as_ref();
        rows.push(ModelInfo {
            kind: None,
            compatibility_type: if cloud {
                Some("cloud".into())
            } else {
                non_empty(details.and_then(|d| d.format.as_ref()))
            },
            quantization: non_empty(details.and_then(|d| d.quantization_level.as_ref())),
            max_context_length: p.context_length.or_else(|| details.and_then(|d| d.context_length)),
            state: if cloud { "cloud" } else { "loaded" }.into(),
            id,
        });
    }

    rows.sort_by_key(|m| match m.state.as_str() {
        "loaded" => 0,
        "cloud" => 1,
        _ => 2,
    });
    rows
}

fn non_empty(s: Option<&String>) -> Option<String> {
    s.map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn kind_from_capabilities(caps: Option<&[String]>) -> Option<String> {
    let caps = caps.filter(|c| !c.is_empty())?;
    let has = |want: &str| caps.iter().any(|c| c.eq_ignore_ascii_case(want));
    let kind = if has("embedding") {
        "embeddings"
    } else if has("vision") {
        "vlm"
    } else {
        "llm"
    };
    Some(kind.to_string())
}

fn is_cloud(name: &str, remote_host: Option<&str>) -> bool {
    if remote_host.is_some_and(|h| !h.trim().is_empty()) {
        return true;
    }
    split_tag(name).is_some_and(|(_, tag)| {
        let tag = tag.to_ascii_lowercase();
        tag == "cloud" || tag.ends_with("-cloud")
    })
}

/// Split `name:tag` at the last `:`. A remainder containing `/` means the colon belongs
/// to a registry `host:port/…` prefix, so there is no tag.
fn split_tag(name: &str) -> Option<(&str, &str)> {
    let (base, tag) = name.rsplit_once(':')?;
    if base.is_empty() || tag.contains('/') {
        return None;
    }
    Some((base, tag))
}

/// Normalise a model name for matching captured records to `/api/tags` rows:
/// `qwen3:latest` ≡ `qwen3`, and cloud tags (`deepseek-v4-pro:cloud`,
/// `gpt-oss:120b-cloud`) ≡ the bare names their OpenAI-compatible responses report.
pub fn canonical_model_name(name: &str) -> String {
    let name = name.trim().to_ascii_lowercase();
    if let Some((base, tag)) = split_tag(&name) {
        if matches!(tag, "" | "latest" | "cloud") {
            return base.to_string();
        }
        if let Some(stem) = tag.strip_suffix("-cloud") {
            return if stem.is_empty() {
                base.to_string()
            } else {
                format!("{base}:{stem}")
            };
        }
    }
    name
}

/// The panel row a captured record's `model_id` refers to: an exact (case-insensitive)
/// id match wins, otherwise the first canonical match in display order.
pub fn resolve_model_id<'a>(models: &'a [ModelInfo], record_model: Option<&str>) -> Option<&'a str> {
    let record_model = record_model?;
    if let Some(m) = models.iter().find(|m| m.id.eq_ignore_ascii_case(record_model)) {
        return Some(&m.id);
    }
    let want = canonical_model_name(record_model);
    models
        .iter()
        .find(|m| canonical_model_name(&m.id) == want)
        .map(|m| m.id.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAGS_CLOUD_FIXTURE: &str = include_str!("../fixtures/api-tags-cloud.json");
    const PS_FIXTURE: &str = include_str!("../fixtures/api-ps.json");

    fn tags_fixture() -> Vec<TagModel> {
        serde_json::from_str::<TagsResp>(TAGS_CLOUD_FIXTURE).unwrap().models
    }

    fn ps_fixture() -> Vec<LoadedModel> {
        serde_json::from_str::<PsResp>(PS_FIXTURE).unwrap().models
    }

    fn tag(name: &str, format: &str, quant: &str, ctx: Option<u64>, caps: &[&str]) -> TagModel {
        TagModel {
            name: name.into(),
            model: name.into(),
            remote_host: None,
            details: Some(ModelDetails {
                format: Some(format.into()),
                quantization_level: Some(quant.into()),
                context_length: ctx,
            }),
            capabilities: Some(caps.iter().map(|c| c.to_string()).collect()),
        }
    }

    fn info(id: &str, state: &str) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            kind: None,
            compatibility_type: None,
            quantization: None,
            max_context_length: None,
            state: state.into(),
        }
    }

    #[test]
    fn parse_tags_cloud_fixture() {
        let tags = tags_fixture();
        assert_eq!(tags.len(), 1);
        let m = &tags[0];
        assert_eq!(m.name, "deepseek-v4-pro:cloud");
        assert_eq!(m.remote_host.as_deref(), Some("https://ollama.com"));
        let d = m.details.as_ref().unwrap();
        assert_eq!(d.context_length, Some(1_048_576));
        assert_eq!(d.quantization_level.as_deref(), Some("FP8"));
        assert_eq!(d.format.as_deref(), Some(""));
        assert_eq!(m.capabilities.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn parse_tags_minimal_older_ollama() {
        let tags: TagsResp = serde_json::from_str(r#"{"models":[{"name":"x"}]}"#).unwrap();
        assert_eq!(tags.models.len(), 1);
        assert!(tags.models[0].details.is_none());
        assert!(tags.models[0].capabilities.is_none());
        let empty: TagsResp = serde_json::from_str("{}").unwrap();
        assert!(empty.models.is_empty());
    }

    #[test]
    fn parse_api_ps_fixture() {
        let json = r#"{"models":[{"name":"qwen3:14b","model":"qwen3:14b","size":9000000000,"size_vram":8500000000,"expires_at":"2026-05-05T11:00:00Z","digest":"sha256-abc","details":{"format":"gguf","family":"qwen3","parameter_size":"14B","quantization_level":"Q4_K_M"},"context_length":4096}]}"#;
        let p: PsResp = serde_json::from_str(json).unwrap();
        assert_eq!(p.models.len(), 1);
        assert_eq!(p.models[0].name, "qwen3:14b");
        assert_eq!(p.models[0].context_length, Some(4096));
        assert_eq!(
            p.models[0].details.as_ref().unwrap().quantization_level.as_deref(),
            Some("Q4_K_M")
        );
    }

    #[test]
    fn parse_api_ps_fixture_file() {
        let ps = ps_fixture();
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].name, "qwen3:14b");
        assert_eq!(ps[0].context_length, Some(4096));
        assert_eq!(ps[0].details.as_ref().unwrap().format.as_deref(), Some("gguf"));
    }

    #[test]
    fn parse_api_ps_empty() {
        let p: PsResp = serde_json::from_str(r#"{"models":[]}"#).unwrap();
        assert!(p.models.is_empty());
    }

    #[test]
    fn merge_cloud_only() {
        let rows = merge_models(tags_fixture(), vec![]);
        assert_eq!(
            rows,
            vec![ModelInfo {
                id: "deepseek-v4-pro:cloud".into(),
                kind: Some("llm".into()),
                compatibility_type: Some("cloud".into()),
                quantization: Some("FP8".into()),
                max_context_length: Some(1_048_576),
                state: "cloud".into(),
            }]
        );
    }

    #[test]
    fn merge_local_loaded_and_embedding() {
        let tags = vec![
            tag("nomic-embed-text:latest", "gguf", "F16", Some(2048), &["embedding"]),
            tag("qwen3:14b", "gguf", "Q4_K_M", None, &["completion", "tools"]),
            tag("llava:7b", "gguf", "Q4_0", Some(4096), &["completion", "vision"]),
        ];
        let rows = merge_models(tags, ps_fixture());
        let ids: Vec<&str> = rows.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["qwen3:14b", "nomic-embed-text:latest", "llava:7b"]);

        let qwen = &rows[0];
        assert_eq!(qwen.state, "loaded");
        assert_eq!(qwen.kind.as_deref(), Some("llm"));
        assert_eq!(qwen.compatibility_type.as_deref(), Some("gguf"));
        // No context_length in tags details: falls back to the allocated ctx from /api/ps.
        assert_eq!(qwen.max_context_length, Some(4096));

        assert_eq!(rows[1].state, "not-loaded");
        assert_eq!(rows[1].kind.as_deref(), Some("embeddings"));
        assert_eq!(rows[2].kind.as_deref(), Some("vlm"));
    }

    #[test]
    fn merge_sort_is_stable_within_groups() {
        let mut cloud = tag("b:cloud", "", "FP8", None, &["completion"]);
        cloud.remote_host = Some("https://ollama.com".into());
        let tags = vec![
            tag("a", "gguf", "Q4", None, &[]),
            cloud,
            tag("c", "gguf", "Q4", None, &[]),
            tag("qwen3:14b", "gguf", "Q4_K_M", None, &[]),
        ];
        let rows = merge_models(tags, ps_fixture());
        let ids: Vec<&str> = rows.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["qwen3:14b", "b:cloud", "a", "c"]);
        // Empty capabilities list → no kind rather than a guess.
        assert_eq!(rows[2].kind, None);
    }

    #[test]
    fn merge_ps_only_appended_as_loaded() {
        let rows = merge_models(vec![], ps_fixture());
        assert_eq!(
            rows,
            vec![ModelInfo {
                id: "qwen3:14b".into(),
                kind: None,
                compatibility_type: Some("gguf".into()),
                quantization: Some("Q4_K_M".into()),
                max_context_length: Some(4096),
                state: "loaded".into(),
            }]
        );
    }

    #[test]
    fn merge_cloud_in_ps_stays_cloud() {
        let ps = vec![LoadedModel {
            name: "deepseek-v4-pro:cloud".into(),
            model: "deepseek-v4-pro:cloud".into(),
            ..Default::default()
        }];
        let rows = merge_models(tags_fixture(), ps);
        assert_eq!(rows.len(), 1, "ps entry must merge into the tags row, not duplicate it");
        assert_eq!(rows[0].state, "cloud");
    }

    #[test]
    fn cloud_detected_from_tag_without_remote_host() {
        // Older Ollama: no remote_host field, but the tag still says cloud.
        let rows = merge_models(vec![tag("gpt-oss:120b-cloud", "", "", None, &[])], vec![]);
        assert_eq!(rows[0].state, "cloud");
        assert_eq!(rows[0].compatibility_type.as_deref(), Some("cloud"));
        assert_eq!(rows[0].quantization, None, "empty strings count as missing");
    }

    #[test]
    fn canonical_model_name_table() {
        for (input, want) in [
            ("deepseek-v4-pro:cloud", "deepseek-v4-pro"),
            ("deepseek-v4-pro", "deepseek-v4-pro"),
            ("gpt-oss:120b-cloud", "gpt-oss:120b"),
            ("qwen3:latest", "qwen3"),
            ("Qwen3:14B", "qwen3:14b"),
            ("  qwen3:14b ", "qwen3:14b"),
            ("qwen3:", "qwen3"),
            ("registry.local:5000/ns/model", "registry.local:5000/ns/model"),
            ("registry.local:5000/ns/model:latest", "registry.local:5000/ns/model"),
        ] {
            assert_eq!(canonical_model_name(input), want, "input {input:?}");
        }
    }

    #[test]
    fn resolve_prefers_exact_then_canonical() {
        let models = vec![info("gpt-oss:120b-cloud", "cloud"), info("gpt-oss:120b", "loaded")];
        assert_eq!(resolve_model_id(&models, Some("gpt-oss:120b")), Some("gpt-oss:120b"));
        assert_eq!(resolve_model_id(&models, Some("GPT-OSS:120B")), Some("gpt-oss:120b"));

        let models = vec![info("deepseek-v4-pro:cloud", "cloud"), info("qwen3:latest", "not-loaded")];
        assert_eq!(
            resolve_model_id(&models, Some("deepseek-v4-pro")),
            Some("deepseek-v4-pro:cloud")
        );
        assert_eq!(resolve_model_id(&models, Some("qwen3")), Some("qwen3:latest"));
        assert_eq!(resolve_model_id(&models, Some("llama3")), None);
        assert_eq!(resolve_model_id(&models, None), None);
    }

    #[test]
    fn snapshot_from_partial_failure() {
        match snapshot_from(Some("0.35.0".into()), Ok(tags_fixture()), Err("/api/ps: boom".into())) {
            ModelsSnapshot::Loaded { version, models } => {
                assert_eq!(version.as_deref(), Some("0.35.0"));
                assert_eq!(models.len(), 1);
                assert_eq!(models[0].state, "cloud");
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        match snapshot_from(None, Err("/api/tags: boom".into()), Ok(ps_fixture())) {
            ModelsSnapshot::Loaded { models, .. } => {
                assert_eq!(models.len(), 1);
                assert_eq!(models[0].state, "loaded");
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        match snapshot_from(None, Err("tags down".into()), Err("ps down".into())) {
            ModelsSnapshot::Unreachable { reason } => assert_eq!(reason, "tags down; ps down"),
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }
}
