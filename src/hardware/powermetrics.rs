//! macOS telemetry backend: `sudo -n powermetrics`, text-parsed for the GPU active
//! residency and ANE power lines.
//!
//! `prime` runs `sudo -v` so the password prompt happens against a normal cooked terminal;
//! `command` then launches powermetrics non-interactively (`-n`) with no stdin and its own
//! process group, so it never competes with crossterm for the TUI's terminal.

use std::process::Stdio;

use tokio::process::Command;
use tracing::{debug, warn};

use super::{SAMPLE_INTERVAL, Telemetry};

pub(super) const PROGRAM: &str = "powermetrics";

/// Runs `sudo -v` synchronously so the password prompt happens against a normal
/// cooked terminal — must be called before `crossterm::enable_raw_mode`.
/// Failure is logged but not fatal (powermetrics will then fail to start, and GPU/ANE
/// will show n/a).
pub(super) fn prime() {
    eprintln!("ollama-monitor needs sudo to run powermetrics for GPU/ANE telemetry.");
    eprintln!("(Skip with --no-tui to avoid the prompt.)");
    let status = std::process::Command::new("sudo").args(["-v"]).status();
    match status {
        Ok(s) if s.success() => debug!("sudo -v cached credentials"),
        Ok(s) => warn!(
            "sudo -v exited with status {} — GPU/ANE may be unavailable",
            s
        ),
        Err(err) => warn!(error = %err, "sudo -v failed — GPU/ANE may be unavailable"),
    }
}

/// `sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power -i 2000`. The
/// `cpu_power` sampler is included because on M1/M4 Macs the unified power summary that
/// holds the `ANE Power:` line only appears when it's requested.
pub(super) fn command() -> Command {
    let mut cmd = Command::new("sudo");
    cmd.args([
        "-n", // non-interactive: relies on prime() priming the timestamp
        PROGRAM,
        "--samplers",
        "cpu_power,gpu_power,ane_power",
        "-i",
    ]);
    cmd.arg(SAMPLE_INTERVAL.as_millis().to_string());
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::null());
    // Keep sudo off the TUI's terminal. sudo >= 1.9.14 defaults to use_pty, and a sudo in
    // the terminal's foreground process group may read terminal input to relay to its
    // command, competing with crossterm for keystrokes. With no stdin and its own process
    // group it never reads from or reconfigures our terminal. Safe only because of -n: a
    // background sudo that prompted would stop on SIGTTIN. Not setsid: sudo's cached
    // credentials are tied to the terminal session.
    cmd.process_group(0);
    cmd.kill_on_drop(true);
    cmd
}

/// The latest GPU and ANE readings.
#[derive(Debug, Default)]
pub(super) struct State {
    gpu_util_pct: Option<f32>,
    ane_power_mw: Option<f32>,
}

impl State {
    /// Returns whether `line` carried a reading.
    pub(super) fn apply_line(&mut self, line: &str) -> bool {
        if let Some(pct) = parse_gpu_active_residency(line) {
            self.gpu_util_pct = Some(pct);
            true
        } else if let Some(mw) = parse_ane_power(line) {
            self.ane_power_mw = Some(mw);
            true
        } else {
            false
        }
    }

    pub(super) fn publish(&self) -> Telemetry {
        Telemetry {
            gpu_util_pct: self.gpu_util_pct,
            gpu_mem_used_bytes: None,
            ane_power_mw: self.ane_power_mw,
        }
    }
}

/// Matches lines like:
///   `GPU HW active residency:   12.34% (...)`
///   `GPU active residency: 9.5%`
fn parse_gpu_active_residency(line: &str) -> Option<f32> {
    let l = line.trim();
    let needle = "GPU";
    if !l.starts_with(needle) {
        return None;
    }
    if !l.to_ascii_lowercase().contains("active residency") {
        return None;
    }
    let after_colon = l.split_once(':').map(|(_, r)| r)?;
    let pct_str = after_colon.trim().split('%').next()?.trim();
    pct_str.parse::<f32>().ok()
}

/// Matches lines like `ANE Power: 12 mW` or `ANE Power: 0 mW`.
fn parse_ane_power(line: &str) -> Option<f32> {
    let l = line.trim();
    if !l.to_ascii_lowercase().starts_with("ane power") {
        return None;
    }
    let after_colon = l.split_once(':').map(|(_, r)| r)?;
    let value = after_colon.split_whitespace().next()?;
    value.parse::<f32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gpu_residency_variants() {
        assert!(
            (parse_gpu_active_residency("GPU HW active residency:   12.34% (...)").unwrap()
                - 12.34)
                .abs()
                < 1e-3
        );
        assert!(
            (parse_gpu_active_residency("GPU active residency: 9.5%").unwrap() - 9.5).abs() < 1e-3
        );
        assert!(parse_gpu_active_residency("System Average frequency: 1234 MHz").is_none());
    }

    #[test]
    fn parse_ane_variants() {
        assert!((parse_ane_power("ANE Power: 25 mW").unwrap() - 25.0).abs() < 1e-3);
        assert!((parse_ane_power("ANE Power:0 mW").unwrap() - 0.0).abs() < 1e-3);
        assert!(parse_ane_power("CPU Power: 1500 mW").is_none());
    }

    #[test]
    fn state_applies_gpu_and_ane_lines() {
        let mut state = State::default();
        assert!(state.apply_line("GPU HW active residency:   12.34% (...)"));
        assert!(state.apply_line("ANE Power: 25 mW"));
        assert!(!state.apply_line("System Average frequency: 1234 MHz"));
        let t = state.publish();
        assert!((t.gpu_util_pct.unwrap() - 12.34).abs() < 1e-3);
        assert!((t.ane_power_mw.unwrap() - 25.0).abs() < 1e-3);
        assert_eq!(t.gpu_mem_used_bytes, None);
    }
}
