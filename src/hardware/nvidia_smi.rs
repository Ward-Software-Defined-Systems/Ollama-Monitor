//! Linux telemetry backend: `nvidia-smi` in loop mode, CSV-parsed for GPU utilisation and
//! VRAM used. It needs no privilege, so `prime` does nothing. Machines without the NVIDIA
//! driver (or with Intel / AMD GPUs only) fail to spawn it or get no device lines, and the
//! GPU figures stay n/a.
//!
//! Unlike the sudo child on macOS, nvidia-smi never touches the terminal, so it stays in the
//! TUI's process group: a closing terminal's SIGHUP reaches it too.

use std::collections::BTreeMap;
use std::process::Stdio;

use tokio::process::Command;

use super::{SAMPLE_INTERVAL, Telemetry};

pub(super) const PROGRAM: &str = "nvidia-smi";

/// nvidia-smi reports memory in MiB.
const MIB: u64 = 1024 * 1024;

/// Nothing to do: nvidia-smi needs no privilege.
pub(super) fn prime() {}

/// `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
/// -lms 2000`: one line per GPU per interval, e.g. `0, 26, 55` (percent and MiB).
pub(super) fn command() -> Command {
    let mut cmd = Command::new(PROGRAM);
    cmd.args([
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-lms",
    ]);
    cmd.arg(SAMPLE_INTERVAL.as_millis().to_string());
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::null());
    cmd.kill_on_drop(true);
    cmd
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GpuReading {
    util_pct: Option<f32>,
    mem_used_bytes: Option<u64>,
}

/// The latest reading per GPU index.
#[derive(Debug, Default)]
pub(super) struct State {
    gpus: BTreeMap<u32, GpuReading>,
}

impl State {
    /// Returns whether `line` was a device line (its reading replaces that GPU's).
    pub(super) fn apply_line(&mut self, line: &str) -> bool {
        match parse_line(line) {
            Some((index, reading)) => {
                self.gpus.insert(index, reading);
                true
            }
            None => false,
        }
    }

    /// Utilisation is the busiest GPU's; memory is summed. A field nvidia-smi reports as
    /// `[N/A]` on every GPU stays `None`.
    pub(super) fn publish(&self) -> Telemetry {
        Telemetry {
            gpu_util_pct: self
                .gpus
                .values()
                .filter_map(|g| g.util_pct)
                .reduce(f32::max),
            gpu_mem_used_bytes: self
                .gpus
                .values()
                .filter_map(|g| g.mem_used_bytes)
                .reduce(|a, b| a.saturating_add(b)),
            ane_power_mw: None,
        }
    }
}

/// Parses one `index, utilization.gpu, memory.used` line. Anything else (blank lines,
/// `No devices were found`, driver error text) is `None`. A field given as `[N/A]` or
/// `[Not Supported]` becomes `None` on its own, without dropping the line.
fn parse_line(line: &str) -> Option<(u32, GpuReading)> {
    let mut fields = line.split(',').map(str::trim);
    let index = fields.next()?.parse::<u32>().ok()?;
    let util = fields.next()?;
    let mem = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    Some((
        index,
        GpuReading {
            util_pct: util.parse::<f32>().ok(),
            mem_used_bytes: mem.parse::<u64>().ok().map(|mib| mib.saturating_mul(MIB)),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_line_variants() {
        assert_eq!(
            parse_line("0, 26, 55"),
            Some((
                0,
                GpuReading {
                    util_pct: Some(26.0),
                    mem_used_bytes: Some(55 * MIB),
                }
            ))
        );
        assert_eq!(
            parse_line("1, [N/A], 1024"),
            Some((
                1,
                GpuReading {
                    util_pct: None,
                    mem_used_bytes: Some(1024 * MIB),
                }
            ))
        );
        assert_eq!(parse_line("No devices were found"), None);
        assert_eq!(parse_line(""), None);
        assert_eq!(parse_line("0, 26"), None);
        assert_eq!(parse_line("0, 26, 55, 16376"), None);
    }

    #[test]
    fn state_aggregates_max_util_and_summed_memory() {
        let mut state = State::default();
        assert!(state.apply_line("0, 10, 1000"));
        assert!(state.apply_line("1, 40, 2000"));
        // A later line for the same GPU replaces its reading; it doesn't accumulate.
        assert!(state.apply_line("0, 30, 1500"));
        assert!(!state.apply_line("NVIDIA-SMI has failed"));
        let t = state.publish();
        assert_eq!(t.gpu_util_pct, Some(40.0));
        assert_eq!(t.gpu_mem_used_bytes, Some(3500 * MIB));
        assert_eq!(t.ane_power_mw, None);
    }

    #[test]
    fn state_with_no_gpus_publishes_none() {
        assert_eq!(State::default().publish(), Telemetry::default());
        let mut state = State::default();
        assert!(state.apply_line("0, [N/A], [N/A]"));
        assert_eq!(state.publish(), Telemetry::default());
    }
}
