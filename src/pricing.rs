use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Deserialize;
use tracing::warn;

use crate::config::UserConfig;

pub const FRONTIER_MODELS: &[&str] = &[
    "claude-fable-5",
    "claude-opus-4-8",
    "gemini-3-1-pro",
];

const BAKED_PRICING: &str = include_str!("../pricing.toml");

#[derive(Debug, Clone, Deserialize)]
struct PricingFile {
    providers: BTreeMap<String, Provider>,
}

#[derive(Debug, Clone, Deserialize)]
struct Provider {
    models: BTreeMap<String, ModelPricing>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ModelPricing {
    pub input_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
}

#[derive(Debug, Clone)]
pub struct PricingTable {
    by_model: BTreeMap<String, ModelPricing>,
}

impl PricingTable {
    pub fn get(&self, model_key: &str) -> Option<ModelPricing> {
        self.by_model.get(model_key).copied()
    }
}

pub fn load(user: Option<&UserConfig>) -> Arc<PricingTable> {
    let baked: PricingFile = toml::from_str(BAKED_PRICING).expect("baked-in pricing.toml is valid");
    let mut by_model: BTreeMap<String, ModelPricing> = BTreeMap::new();
    for provider in baked.providers.values() {
        for (name, price) in &provider.models {
            by_model.insert(name.clone(), *price);
        }
    }

    if let Some(overrides) = user.and_then(|u| u.pricing.as_ref()).and_then(|p| p.providers.as_ref()) {
        match overrides.clone().try_into::<BTreeMap<String, Provider>>() {
            Ok(provs) => {
                for provider in provs.values() {
                    for (name, price) in &provider.models {
                        by_model.insert(name.clone(), *price);
                    }
                }
            }
            Err(err) => warn!(error = %err, "ignoring invalid [pricing.providers] in user config"),
        }
    }

    for key in FRONTIER_MODELS {
        if !by_model.contains_key(*key) {
            warn!(model = %key, "no baked or user pricing for frontier model; cost will be n/a");
        }
    }

    Arc::new(PricingTable { by_model })
}
