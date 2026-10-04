//! Hardware sampler.
//!
//! Two paths combined into one snapshot per tick:
//! 1. `sysinfo` for system CPU/MEM and the Ollama-process-tree CPU%/RSS — no privilege.
//! 2. `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 2000` for GPU
//!    active residency and ANE power. The sudo prompt is taken BEFORE TUI raw mode so
//!    the user can type their password into a normal cooked terminal.
//!
//! The sudo child gets no stdin and its own process group, so it never touches the TUI's
//! terminal (see `spawn_powermetrics`). On shutdown it receives SIGTERM via libc::kill,
//! which sudo relays to powermetrics, so nothing orphans as root. If powermetrics exits
//! on its own, GPU/ANE drop back to n/a.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use sysinfo::{ProcessRefreshKind, RefreshKind, System};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, watch};
use tracing::{debug, trace, warn};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(2000);

/// Substring matches that identify a process as part of the Ollama tree.
const OLLAMA_PATH_HINTS: &[&str] = &[
    "/Applications/Ollama.app/",
    "/opt/homebrew/opt/ollama/",
    "/opt/homebrew/Cellar/ollama/",
    "/usr/local/opt/ollama/",
    "/usr/local/Cellar/ollama/",
    "/.ollama/",
];

/// Name-based fallback, matched exactly against `Process::name()`: the executable name,
/// `ollama` for the server and its runners alike. Deliberately not a prefix match, which
/// would count `ollama-monitor` itself.
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
    pub gpu_active_residency_pct: Option<f32>,
    pub ane_power_mw: Option<f32>,
}

/// Runs `sudo -v` synchronously so the password prompt happens against a normal
/// cooked terminal — must be called before `crossterm::enable_raw_mode`.
/// Failure is logged but not fatal (powermetrics will then fail in `sample`,
/// and GPU/ANE will show n/a).
pub fn prime_sudo() {
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

pub async fn sample(tx: mpsc::Sender<HardwareSnapshot>, mut shutdown_rx: watch::Receiver<bool>) {
    let last = Arc::new(Mutex::new(HardwareSnapshot::default()));

    // Spawn the powermetrics tail in the background; updates `last`.
    let pm_handle = spawn_powermetrics(last.clone(), shutdown_rx.clone()).await;

    let mut sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_processes(ProcessRefreshKind::nothing().with_cpu().with_memory())
            .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage())
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    // Prime sysinfo CPU%; first sample is always 0.0.
    sys.refresh_cpu_usage();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    tokio::time::sleep(Duration::from_millis(200)).await;

    loop {
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

        let mut snap = collect_sysinfo(&sys);

        let pm = last.lock().await;
        snap.gpu_active_residency_pct = pm.gpu_active_residency_pct;
        snap.ane_power_mw = pm.ane_power_mw;
        drop(pm);

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

    // Best-effort SIGTERM to the powermetrics child so it doesn't orphan as root.
    if let Some(mut child) = pm_handle {
        if let Some(pid) = child.id() {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
        let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
    }
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
        if !is_ollama_process(proc_) {
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
        gpu_active_residency_pct: None,
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

async fn spawn_powermetrics(
    last: Arc<Mutex<HardwareSnapshot>>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Option<Child> {
    let mut cmd = Command::new("sudo");
    cmd.args([
        "-n", // non-interactive: relies on prime_sudo() priming the timestamp
        "powermetrics",
        "--samplers",
        "cpu_power,gpu_power,ane_power",
        "-i",
        "2000",
    ]);
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

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(err) => {
            warn!(error = %err, "spawn powermetrics failed; GPU/ANE will be n/a");
            return None;
        }
    };
    let stdout = child.stdout.take()?;

    tokio::spawn(async move {
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
                            trace!(line = %line, "powermetrics");
                            apply_pm_line(&line, &last).await;
                        }
                        Ok(None) => {
                            warn!("powermetrics exited; GPU/ANE will show n/a");
                            clear_pm(&last).await;
                            return;
                        }
                        Err(err) => {
                            warn!(error = %err, "powermetrics read failed; GPU/ANE will show n/a");
                            clear_pm(&last).await;
                            return;
                        }
                    }
                }
            }
        }
    });

    Some(child)
}

/// Forget the last GPU/ANE readings once powermetrics is gone, so the panel shows n/a
/// instead of numbers frozen at their final values.
async fn clear_pm(last: &Mutex<HardwareSnapshot>) {
    let mut g = last.lock().await;
    g.gpu_active_residency_pct = None;
    g.ane_power_mw = None;
}

async fn apply_pm_line(line: &str, last: &Mutex<HardwareSnapshot>) {
    if let Some(pct) = parse_gpu_active_residency(line) {
        let mut g = last.lock().await;
        g.gpu_active_residency_pct = Some(pct);
        return;
    }
    if let Some(mw) = parse_ane_power(line) {
        let mut g = last.lock().await;
        g.ane_power_mw = Some(mw);
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

    /// Real sysinfo sample without powermetrics/sudo:
    /// `cargo test live_sysinfo -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn live_sysinfo_snapshot() {
        let mut sys = System::new_all();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        sys.refresh_cpu_usage();
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
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
}
