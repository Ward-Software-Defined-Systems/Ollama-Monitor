use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::Cli;

#[derive(Debug, Clone)]
pub struct Paths {
    pub config_file: PathBuf,
    pub db_file: PathBuf,
    pub log_file: PathBuf,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UserConfig {
    pub pricing: Option<PricingOverrides>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PricingOverrides {
    pub providers: Option<toml::Value>,
}

pub fn resolve_paths(cli: &Cli) -> Result<Paths> {
    let app_dir = default_app_dir()?;
    let config_file = cli.config.clone().unwrap_or_else(|| app_dir.join("config.toml"));
    let db_file = cli.db.clone().unwrap_or_else(|| app_dir.join("usage.db"));
    let log_file = app_dir.join("ollama-monitor.log");
    Ok(Paths {
        config_file,
        db_file,
        log_file,
    })
}

pub fn load_user_config(path: &Path) -> Result<Option<UserConfig>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read config {}", path.display()))?;
    let parsed: UserConfig =
        toml::from_str(&raw).with_context(|| format!("parse config {}", path.display()))?;
    Ok(Some(parsed))
}

fn default_app_dir() -> Result<PathBuf> {
    // macOS: ~/Library/Application Support/ollama-monitor/
    let home = std::env::var_os("HOME")
        .ok_or_else(|| anyhow::anyhow!("HOME env var not set"))?;
    let dir = PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join("ollama-monitor");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create app dir {}", dir.display()))?;
    Ok(dir)
}
