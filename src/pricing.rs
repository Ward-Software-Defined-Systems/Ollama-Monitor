use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Deserialize;
use tracing::warn;

use crate::config::UserConfig;

pub const FRONTIER_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
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

    if let Some(overrides) = user
        .and_then(|u| u.pricing.as_ref())
        .and_then(|p| p.providers.as_ref())
    {
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

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HypotheticalCost {
    pub input_usd: f64,
    pub output_usd: f64,
    pub total_usd: f64,
}

pub fn hypothetical_cost(
    prompt_tokens: u64,
    gen_tokens: u64,
    price: &ModelPricing,
) -> HypotheticalCost {
    let input_usd = prompt_tokens as f64 / 1_000_000.0 * price.input_per_mtok_usd;
    let output_usd = gen_tokens as f64 / 1_000_000.0 * price.output_per_mtok_usd;
    HypotheticalCost {
        input_usd,
        output_usd,
        total_usd: input_usd + output_usd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_have_all_frontier_models() {
        let table = load(None);
        for key in FRONTIER_MODELS {
            assert!(
                table.get(key).is_some(),
                "frontier model {key} missing from defaults"
            );
        }
    }

    #[test]
    fn defaults_match_pricing_toml() {
        let table = load(None);
        for (key, input, output) in [
            ("claude-fable-5-1", 10.00, 50.00),
            ("claude-fable-5", 10.00, 50.00),
            ("claude-opus-5-5", 4.00, 20.00),
            ("claude-opus-5", 5.00, 25.00),
            ("claude-opus-4-8", 5.00, 25.00),
            ("gemini-3-1-pro", 2.00, 12.00),
        ] {
            let p = table.get(key).unwrap();
            assert_eq!(
                (p.input_per_mtok_usd, p.output_per_mtok_usd),
                (input, output),
                "{key}"
            );
        }
    }

    #[test]
    fn one_million_in_one_million_out_matches_rates() {
        let table = load(None);
        let opus = table.get("claude-opus-4-8").unwrap();
        let cost = hypothetical_cost(1_000_000, 1_000_000, &opus);
        assert_eq!(cost.input_usd, 5.00);
        assert_eq!(cost.output_usd, 25.00);
        assert_eq!(cost.total_usd, 30.00);
    }
}
