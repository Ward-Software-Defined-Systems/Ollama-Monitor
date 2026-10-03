use std::collections::BTreeMap;

use chrono::{DateTime, Duration, Utc};

use crate::db::InferenceRecord;

#[derive(Debug, Clone, Default)]
pub struct WindowStats {
    pub request_count: u64,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub mean_tps: f64,
    pub p50_tps: f64,
    pub p95_tps: f64,
    pub mean_ttft_sec: f64,
}

#[derive(Debug, Clone, Default)]
pub struct AggregateSnapshot {
    pub one_minute: WindowStats,
    pub five_minute: WindowStats,
    pub fifteen_minute: WindowStats,
    pub session: WindowStats,
    #[allow(dead_code)] // surfaced in tests + reserved for a future per-model panel
    pub per_model: BTreeMap<String, WindowStats>,
}

#[derive(Debug, Default)]
pub struct Aggregator {
    records: Vec<InferenceRecord>,
}

impl Aggregator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(&mut self, record: InferenceRecord) {
        self.records.push(record);
    }

    pub fn reset(&mut self) {
        self.records.clear();
    }

    pub fn snapshot(&self, now: DateTime<Utc>) -> AggregateSnapshot {
        let cutoff_1m = now - Duration::seconds(60);
        let cutoff_5m = now - Duration::seconds(5 * 60);
        let cutoff_15m = now - Duration::seconds(15 * 60);

        let one_minute = window(self.records.iter().filter(|r| r.completed_at >= cutoff_1m));
        let five_minute = window(self.records.iter().filter(|r| r.completed_at >= cutoff_5m));
        let fifteen_minute = window(self.records.iter().filter(|r| r.completed_at >= cutoff_15m));
        let session = window(self.records.iter());

        let mut per_model: BTreeMap<String, Vec<&InferenceRecord>> = BTreeMap::new();
        for r in &self.records {
            per_model.entry(r.model_id.clone()).or_default().push(r);
        }
        let per_model = per_model
            .into_iter()
            .map(|(k, v)| (k, window(v.into_iter())))
            .collect();

        AggregateSnapshot {
            one_minute,
            five_minute,
            fifteen_minute,
            session,
            per_model,
        }
    }
}

fn window<'a, I>(it: I) -> WindowStats
where
    I: Iterator<Item = &'a InferenceRecord>,
{
    let mut count = 0u64;
    let mut prompt_sum = 0u64;
    let mut gen_sum = 0u64;
    let mut tps_vals: Vec<f64> = Vec::new();
    let mut ttft_sum = 0f64;

    for r in it {
        count += 1;
        prompt_sum += r.prompt_tokens;
        gen_sum += r.gen_tokens;
        if r.tokens_per_sec.is_finite() && r.tokens_per_sec > 0.0 {
            tps_vals.push(r.tokens_per_sec);
        }
        ttft_sum += r.ttft_sec;
    }

    let mean_tps = if !tps_vals.is_empty() {
        tps_vals.iter().sum::<f64>() / tps_vals.len() as f64
    } else {
        0.0
    };

    let mut sorted = tps_vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);

    let mean_ttft_sec = if count > 0 { ttft_sum / count as f64 } else { 0.0 };

    WindowStats {
        request_count: count,
        prompt_tokens: prompt_sum,
        gen_tokens: gen_sum,
        mean_tps,
        p50_tps: p50,
        p95_tps: p95,
        mean_ttft_sec,
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(model: &str, tps: f64, prompt: u64, gen_n: u64, ago_secs: i64) -> InferenceRecord {
        InferenceRecord {
            session_id: 1,
            model_id: model.to_string(),
            prompt_tokens: prompt,
            gen_tokens: gen_n,
            tokens_per_sec: tps,
            ttft_sec: 0.5,
            total_time_sec: 1.0,
            stop_reason: "stop".into(),
            completed_at: Utc::now() - Duration::seconds(ago_secs),
            envelope: "ollama-stream".into(),
        }
    }

    #[test]
    fn rolling_window_filters_by_cutoff() {
        let mut agg = Aggregator::new();
        agg.ingest(rec("m1", 10.0, 1, 10, 30)); // in 1m
        agg.ingest(rec("m1", 20.0, 2, 20, 90)); // in 5m
        agg.ingest(rec("m1", 30.0, 3, 30, 600)); // in 15m
        agg.ingest(rec("m1", 40.0, 4, 40, 5_000)); // session only

        let s = agg.snapshot(Utc::now());
        assert_eq!(s.one_minute.request_count, 1);
        assert_eq!(s.five_minute.request_count, 2);
        assert_eq!(s.fifteen_minute.request_count, 3);
        assert_eq!(s.session.request_count, 4);
    }

    #[test]
    fn per_model_breakdown() {
        let mut agg = Aggregator::new();
        agg.ingest(rec("a", 10.0, 1, 1, 0));
        agg.ingest(rec("a", 30.0, 1, 1, 0));
        agg.ingest(rec("b", 50.0, 1, 1, 0));

        let s = agg.snapshot(Utc::now());
        assert_eq!(s.per_model.len(), 2);
        assert_eq!(s.per_model["a"].request_count, 2);
        assert_eq!(s.per_model["b"].request_count, 1);
        assert!((s.per_model["a"].mean_tps - 20.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_indexing() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        assert!((percentile(&v, 0.5) - 3.0).abs() < 1e-6);
        assert!((percentile(&v, 0.95) - 100.0).abs() < 1e-6);
        assert!((percentile(&[], 0.5) - 0.0).abs() < 1e-6);
    }
}
