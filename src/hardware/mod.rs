//! Hardware sampler.
//!
//! Two paths combined into one snapshot per tick:
//! 1. `sysinfo` for system CPU/MEM and the Ollama processes' CPU%/RSS — no privilege.
//! 2. A platform telemetry child whose stdout is parsed line by line for the GPU figures:
//!    - macOS: `sudo -n powermetrics` for GPU active residency and ANE power
//!      (`powermetrics.rs`). The sudo prompt is taken by `prime` BEFORE TUI raw mode so
//!      the user can type their password into a normal cooked terminal.
//!    - Linux: `nvidia-smi` in loop mode for GPU utilisation and VRAM used
//!      (`nvidia_smi.rs`), no privilege. Other GPUs show n/a.
//!
//! Both backends compile on every platform so their parsers are tested everywhere; only
//! the `telemetry` alias is platform-specific. The child gets no stdin and never touches
//! the TUI's terminal. On shutdown it receives SIGTERM via libc::kill (sudo relays it to
//! powermetrics, so nothing orphans as root). If it exits on its own, the GPU figures drop
//! back to n/a.

use std::sync::Arc;
use std::time::Duration;

use sysinfo::{ProcessRefreshKind, RefreshKind, System, UpdateKind};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::sync::{Mutex, mpsc, watch};
use tracing::{debug, trace, warn};

#[cfg_attr(target_os = "macos", allow(dead_code))]
mod nvidia_smi;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod powermetrics;

#[cfg(not(target_os = "macos"))]
use nvidia_smi as telemetry;
#[cfg(target_os = "macos")]
use powermetrics as telemetry;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(2000);

/// Substring matches against the executable path or `argv[0]` that identify a process as
/// part of the Ollama tree. Directories only: a bare `/usr/local/bin/ollama` would also
/// match an `ollama-monitor` installed there.
const OLLAMA_PATH_HINTS: &[&str] = &[
    // macOS
    "/Applications/Ollama.app/",
    "/opt/homebrew/opt/ollama/",
    "/opt/homebrew/Cellar/ollama/",
    "/usr/local/opt/ollama/",
    "/usr/local/Cellar/ollama/",
    // Linux: the install script's library dir (older releases kept their
    // `ollama_llama_server` runners there), distro packages, snap
    "/usr/local/lib/ollama/",
    "/usr/lib/ollama/",
    "/snap/ollama/",
    // both
    "/.ollama/",
];

/// Name-based fallback, matched exactly against `Process::name()`: the executable name,
/// `ollama` for the server and its runners alike. Deliberately not a prefix match, which
/// would count `ollama-monitor` itself.
///
/// On Linux this is the match that does the work: the name (from `/proc/<pid>/stat`) and
/// `argv[0]` (from `/proc/<pid>/cmdline`) are readable for every user's processes, but
/// `/proc/<pid>/exe` is not, so `Process::exe()` is empty for a systemd `ollama` service
/// running as its own user.
const OLLAMA_NAME_HINTS: &[&str] = &["ollama"];

#[derive(Debug, Clone, Default)]
pub struct HardwareSnapshot {
    pub system_cpu_percent: f32,
    pub system_mem_used_bytes: u64,
    pub system_mem_total_bytes: u64,
    pub system_mem_available_bytes: u64,
    pub ollama_process_count: usize,
    pub ollama_cpu_percent: f32,
    pub ollama_rss_bytes: u64,
    /// macOS: powermetrics GPU HW active residency. Linux: nvidia-smi `utilization.gpu`,
    /// the busiest NVIDIA GPU's.
    pub gpu_util_pct: Option<f32>,
    /// Linux only: nvidia-smi `memory.used`, summed across NVIDIA GPUs.
    pub gpu_mem_used_bytes: Option<u64>,
    /// macOS only: powermetrics `ANE Power`.
    pub ane_power_mw: Option<f32>,
}

/// The telemetry child's most recent readings. Reset to default when it exits.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Telemetry {
    pub gpu_util_pct: Option<f32>,
    pub gpu_mem_used_bytes: Option<u64>,
    pub ane_power_mw: Option<f32>,
}

/// Whatever the telemetry backend needs done before the TUI enters raw mode: on macOS the
/// `sudo -v` password prompt, which needs a cooked terminal; on Linux nothing.
pub fn prime() {
    telemetry::prime();
}

pub async fn sample(tx: mpsc::Sender<HardwareSnapshot>, mut shutdown_rx: watch::Receiver<bool>) {
    let last = Arc::new(Mutex::new(Telemetry::default()));

    // Spawn the telemetry child in the background; its reader task updates `last`.
    let child = spawn_telemetry(last.clone(), shutdown_rx.clone());

    let mut sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_processes(process_refresh())
            .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage())
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    // Prime sysinfo CPU%; first sample is always 0.0.
    sys.refresh_cpu_usage();
    sys.refresh_processes_specifics(sysinfo::ProcessesToUpdate::All, true, process_refresh());
    tokio::time::sleep(Duration::from_millis(200)).await;

    loop {
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        sys.refresh_processes_specifics(sysinfo::ProcessesToUpdate::All, true, process_refresh());

        let mut snap = collect_sysinfo(&sys);

        let telemetry = *last.lock().await;
        snap.gpu_util_pct = telemetry.gpu_util_pct;
        snap.gpu_mem_used_bytes = telemetry.gpu_mem_used_bytes;
        snap.ane_power_mw = telemetry.ane_power_mw;

        if tx.send(snap).await.is_err() {
            debug!("hardware channel closed; sampler exiting");
            break;
        }

        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { break; }
            }
            _ = tokio::time::sleep(SAMPLE_INTERVAL) => {}
        }
    }

    // Best-effort SIGTERM to the telemetry child so it doesn't outlive us. On macOS sudo
    // relays it to powermetrics, so nothing orphans as root.
    if let Some(mut child) = child {
        if let Some(pid) = child.id() {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
        let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
    }
}

/// What to read per process. `without_tasks` matters on Linux, where sysinfo otherwise
/// lists every thread as a process of its own (each Ollama thread is named `ollama`, so the
/// count, RSS and CPU would all be inflated). `cmd` feeds the `argv[0]` hint match;
/// `OnlyIfNotSet` reads it once per process.
fn process_refresh() -> ProcessRefreshKind {
    ProcessRefreshKind::nothing()
        .with_cpu()
        .with_memory()
        .with_exe(UpdateKind::OnlyIfNotSet)
        .with_cmd(UpdateKind::OnlyIfNotSet)
        .without_tasks()
}

fn collect_sysinfo(sys: &System) -> HardwareSnapshot {
    let cpu_count = sys.cpus().len().max(1) as f32;
    let system_cpu = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / cpu_count;

    let mem_total = sys.total_memory();
    let mem_used = sys.used_memory();
    let mem_available = sys.available_memory();

    let mut ollama_count = 0usize;
    let mut ollama_cpu = 0f32;
    let mut ollama_rss = 0u64;
    for proc_ in sys.processes().values() {
        // Belt and braces for Linux: a thread listed as a process (only when the refresh
        // kind asks for tasks) must not count again. Always `None` elsewhere.
        if proc_.thread_kind().is_some() || !is_ollama_process(proc_) {
            continue;
        }
        ollama_count += 1;
        ollama_cpu += proc_.cpu_usage();
        ollama_rss += proc_.memory();
    }

    HardwareSnapshot {
        system_cpu_percent: system_cpu,
        system_mem_used_bytes: mem_used,
        system_mem_total_bytes: mem_total,
        system_mem_available_bytes: mem_available,
        ollama_process_count: ollama_count,
        ollama_cpu_percent: ollama_cpu,
        ollama_rss_bytes: ollama_rss,
        gpu_util_pct: None,
        gpu_mem_used_bytes: None,
        ane_power_mw: None,
    }
}

fn is_ollama_process(p: &sysinfo::Process) -> bool {
    if let Some(path) = p.exe() {
        let s = path.to_string_lossy();
        if OLLAMA_PATH_HINTS.iter().any(|h| s.contains(h)) {
            return true;
        }
    }
    let name = p.name().to_string_lossy().to_lowercase();
    if OLLAMA_NAME_HINTS.iter().any(|h| name == *h) {
        return true;
    }
    if let Some(cmd) = p.cmd().first() {
        let s = cmd.to_string_lossy();
        if OLLAMA_PATH_HINTS.iter().any(|h| s.contains(h)) {
            return true;
        }
    }
    false
}

/// Starts the platform's telemetry child plus a task that parses its stdout into `last`.
/// Returns `None`, with a warning, when it can't be started; the GPU figures then stay n/a.
fn spawn_telemetry(
    last: Arc<Mutex<Telemetry>>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Option<Child> {
    let mut child = match telemetry::command().spawn() {
        Ok(c) => c,
        Err(err) => {
            warn!(
                error = %err,
                "spawn {} failed; GPU telemetry will be n/a",
                telemetry::PROGRAM
            );
            return None;
        }
    };
    let stdout = child.stdout.take()?;

    tokio::spawn(async move {
        let mut state = telemetry::State::default();
        let mut reader = BufReader::new(stdout).lines();
        loop {
            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() { return; }
                }
                line_res = reader.next_line() => {
                    match line_res {
                        Ok(Some(line)) => {
                            trace!(line = %line, "{}", telemetry::PROGRAM);
                            if state.apply_line(&line) {
                                *last.lock().await = state.publish();
                            }
                        }
                        Ok(None) => {
                            // A closing terminal can take the child down a beat before
                            // the shutdown flag reaches us; that's not worth a warning.
                            if !*shutdown_rx.borrow() {
                                warn!(
                                    "{} exited; GPU telemetry will show n/a",
                                    telemetry::PROGRAM
                                );
                            }
                            *last.lock().await = Telemetry::default();
                            return;
                        }
                        Err(err) => {
                            warn!(
                                error = %err,
                                "{} read failed; GPU telemetry will show n/a",
                                telemetry::PROGRAM
                            );
                            *last.lock().await = Telemetry::default();
                            return;
                        }
                    }
                }
            }
        }
    });

    Some(child)
}

/// Decimal units, matching LMS-Monitor (and Activity Monitor's "GB").
pub fn format_bytes(b: u64) -> String {
    let f = b as f64;
    if f >= 1.0e9 {
        format!("{:.1} GB", f / 1.0e9)
    } else if f >= 1.0e6 {
        format!("{:.0} MB", f / 1.0e6)
    } else if f >= 1.0e3 {
        format!("{:.0} KB", f / 1.0e3)
    } else {
        format!("{b} B")
    }
}

/// Format `used` and `total` with a single shared unit chosen from `total`,
/// e.g. `38.2/137.4 GB` — compact enough for a one-line panel.
pub fn format_bytes_ratio(used: u64, total: u64) -> String {
    let (div, unit, prec) = if total >= 1_000_000_000 {
        (1.0e9, "GB", 1)
    } else if total >= 1_000_000 {
        (1.0e6, "MB", 0)
    } else if total >= 1_000 {
        (1.0e3, "KB", 0)
    } else {
        (1.0, "B", 0)
    };
    format!(
        "{:.prec$}/{:.prec$} {unit}",
        used as f64 / div,
        total as f64 / div,
        prec = prec
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_picks_unit() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(999), "999 B");
        assert_eq!(format_bytes(1_500), "2 KB");
        assert_eq!(format_bytes(1_500_000), "2 MB");
        assert_eq!(format_bytes(38_200_000_000), "38.2 GB");
    }

    #[test]
    fn format_bytes_ratio_shares_one_unit() {
        assert_eq!(
            format_bytes_ratio(38_200_000_000, 137_400_000_000),
            "38.2/137.4 GB"
        );
        assert_eq!(
            format_bytes_ratio(512_000_000, 137_400_000_000),
            "0.5/137.4 GB"
        );
        assert_eq!(format_bytes_ratio(1_500_000, 8_000_000), "2/8 MB");
        assert_eq!(format_bytes_ratio(0, 0), "0/0 B");
    }

    #[test]
    fn collect_sysinfo_reports_available_memory() {
        let mut sys = System::new();
        sys.refresh_memory();
        let snap = collect_sysinfo(&sys);
        assert!(snap.system_mem_total_bytes > 0);
        assert!(
            snap.system_mem_available_bytes > 0,
            "available memory should be sampled"
        );
        assert!(snap.system_mem_available_bytes <= snap.system_mem_total_bytes);
    }

    /// Both backends expose the same interface to `spawn_telemetry`, whichever one the
    /// platform selects; an empty state must publish an empty reading.
    #[test]
    fn both_backends_publish_defaults() {
        assert_eq!(
            powermetrics::State::default().publish(),
            Telemetry::default()
        );
        assert_eq!(nvidia_smi::State::default().publish(), Telemetry::default());
    }

    /// Real sysinfo sample without the telemetry child:
    /// `cargo test live_sysinfo -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn live_sysinfo_snapshot() {
        let mut sys = System::new_all();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        sys.refresh_cpu_usage();
        sys.refresh_processes_specifics(sysinfo::ProcessesToUpdate::All, true, process_refresh());
        let snap = collect_sysinfo(&sys);
        println!(
            "cpu={:.1}% mem={} ({} free) ollama procs={} cpu={:.1}% rss={}",
            snap.system_cpu_percent,
            format_bytes_ratio(snap.system_mem_used_bytes, snap.system_mem_total_bytes),
            format_bytes(snap.system_mem_available_bytes),
            snap.ollama_process_count,
            snap.ollama_cpu_percent,
            format_bytes(snap.ollama_rss_bytes),
        );
    }
}
