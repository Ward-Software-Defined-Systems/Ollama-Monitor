use std::ffi::OsStr;
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

/// Where the app keeps its files by default. One directory on macOS, two on Linux.
#[derive(Debug, Clone, PartialEq)]
struct AppDirs {
    /// `config.toml`
    config_dir: PathBuf,
    /// `usage.db` and the log
    data_dir: PathBuf,
}

pub fn resolve_paths(cli: &Cli) -> Result<Paths> {
    let dirs = default_dirs()?;
    // The log always lives in the data dir, so create it up front and fail with a labelled
    // error (`init_tracing` and the DB open would create it too). The config dir isn't
    // created: the file is optional.
    std::fs::create_dir_all(&dirs.data_dir)
        .with_context(|| format!("create app dir {}", dirs.data_dir.display()))?;
    let config_file = cli
        .config
        .clone()
        .unwrap_or_else(|| dirs.config_dir.join("config.toml"));
    let db_file = cli
        .db
        .clone()
        .unwrap_or_else(|| dirs.data_dir.join("usage.db"));
    let log_file = dirs.data_dir.join("ollama-monitor.log");
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
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    let parsed: UserConfig =
        toml::from_str(&raw).with_context(|| format!("parse config {}", path.display()))?;
    Ok(Some(parsed))
}

fn home_dir() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME env var not set"))
}

/// macOS: everything under `~/Library/Application Support/ollama-monitor/`.
#[cfg(target_os = "macos")]
fn default_dirs() -> Result<AppDirs> {
    let dir = home_dir()?
        .join("Library")
        .join("Application Support")
        .join("ollama-monitor");
    Ok(AppDirs {
        config_dir: dir.clone(),
        data_dir: dir,
    })
}

/// Linux (and other Unix): XDG base directories, `~/.config/ollama-monitor/` for the
/// config and `~/.local/share/ollama-monitor/` for the database and log.
#[cfg(not(target_os = "macos"))]
fn default_dirs() -> Result<AppDirs> {
    Ok(xdg_dirs(
        &home_dir()?,
        std::env::var_os("XDG_CONFIG_HOME").as_deref(),
        std::env::var_os("XDG_DATA_HOME").as_deref(),
    ))
}

/// Pure XDG resolution, kept free of environment access so it can be tested. Per the
/// spec, an unset, empty or relative `XDG_*_HOME` is ignored in favour of the default.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn xdg_dirs(home: &Path, config_home: Option<&OsStr>, data_home: Option<&OsStr>) -> AppDirs {
    let pick = |value: Option<&OsStr>, default: PathBuf| -> PathBuf {
        match value.map(Path::new) {
            Some(p) if p.is_absolute() => p.to_path_buf(),
            _ => default,
        }
    };
    AppDirs {
        config_dir: pick(config_home, home.join(".config")).join("ollama-monitor"),
        data_dir: pick(data_home, home.join(".local").join("share")).join("ollama-monitor"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdg_dirs_defaults_to_home() {
        let dirs = xdg_dirs(Path::new("/home/u"), None, None);
        assert_eq!(
            dirs.config_dir,
            PathBuf::from("/home/u/.config/ollama-monitor")
        );
        assert_eq!(
            dirs.data_dir,
            PathBuf::from("/home/u/.local/share/ollama-monitor")
        );
    }

    #[test]
    fn xdg_dirs_honours_absolute_overrides() {
        let dirs = xdg_dirs(
            Path::new("/home/u"),
            Some(OsStr::new("/etc/xdg-config")),
            Some(OsStr::new("/srv/data")),
        );
        assert_eq!(
            dirs.config_dir,
            PathBuf::from("/etc/xdg-config/ollama-monitor")
        );
        assert_eq!(dirs.data_dir, PathBuf::from("/srv/data/ollama-monitor"));
    }

    #[test]
    fn xdg_dirs_ignores_empty_and_relative() {
        let dirs = xdg_dirs(
            Path::new("/home/u"),
            Some(OsStr::new("")),
            Some(OsStr::new("relative/dir")),
        );
        assert_eq!(
            dirs.config_dir,
            PathBuf::from("/home/u/.config/ollama-monitor")
        );
        assert_eq!(
            dirs.data_dir,
            PathBuf::from("/home/u/.local/share/ollama-monitor")
        );
    }
}
