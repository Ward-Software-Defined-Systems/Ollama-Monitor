use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{info, warn};

/// Schema v1, frozen. Binaries from before v2 run exactly this on every open, then insert
/// version 1 if `schema_version` is empty; `migrates_v1_and_old_binary_keeps_working`
/// replays that. Later changes go in their own batch.
const V1_SQL: &str = "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
         CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT
         );
         CREATE TABLE IF NOT EXISTS inference_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            completed_at TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            gen_tokens INTEGER NOT NULL,
            tokens_per_sec REAL NOT NULL,
            ttft_sec REAL NOT NULL,
            total_time_sec REAL NOT NULL,
            stop_reason TEXT NOT NULL,
            envelope TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
         );
         CREATE INDEX IF NOT EXISTS idx_records_session ON inference_records(session_id);
         CREATE INDEX IF NOT EXISTS idx_records_model ON inference_records(model_id);";

/// Schema v2 adds `failed_requests`. It's all `IF NOT EXISTS` / `OR IGNORE`, so there's no
/// migration step, and a binary from before v2 keeps working on a v2 file: its `V1_SQL` is
/// all no-ops there, its version check finds a row, and it never touches the new table.
/// `schema_version` holds one row per version applied.
const V2_SQL: &str = "CREATE TABLE IF NOT EXISTS failed_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            failed_at TEXT NOT NULL,
            model_id TEXT,
            path TEXT NOT NULL,
            status INTEGER NOT NULL,
            source TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
         );
         CREATE INDEX IF NOT EXISTS idx_failures_session ON failed_requests(session_id);
         INSERT OR IGNORE INTO schema_version (version) VALUES (1), (2);";

#[derive(Debug, Clone)]
pub struct InferenceRecord {
    pub session_id: i64,
    pub model_id: String,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub tokens_per_sec: f64,
    pub ttft_sec: f64,
    pub total_time_sec: f64,
    pub stop_reason: String,
    pub completed_at: DateTime<Utc>,
    /// `parser::Envelope::as_str()`: "ollama-stream", "ollama-single", "openai-sse",
    /// "openai-single", or "openai-sse-approx" for estimated counts.
    pub envelope: String,
}

/// A tracked inference request whose response was a 4xx or 5xx. Metadata only: the error
/// body is a response body, so nothing reads or keeps it.
#[derive(Debug, Clone)]
pub struct FailedRequest {
    pub session_id: i64,
    pub failed_at: DateTime<Utc>,
    /// The request's `model`, from the body the proxy buffers on the OpenAI-compatible
    /// paths. `None` on the native paths, whose bodies stream through unread, and when the
    /// body couldn't be read.
    pub model_id: Option<String>,
    pub path: String,
    pub status: u16,
    pub source: FailureSource,
}

impl FailedRequest {
    /// `failed (HTTP 429)` for a status from Ollama, `proxy (HTTP 502)` for one the proxy
    /// produced itself: the feed's stop cell and the headless line.
    pub fn summary(&self) -> String {
        let label = match self.source {
            FailureSource::Ollama => "failed",
            FailureSource::Proxy => "proxy",
        };
        format!("{label} (HTTP {})", self.status)
    }
}

/// Who answered a failed request with its status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureSource {
    /// Ollama, or ollama.com behind a `:cloud` model.
    Ollama,
    /// The proxy itself: Ollama unreachable (502), a buffered body over the cap (413), an
    /// unreadable request body (400).
    Proxy,
}

impl FailureSource {
    /// Stored in `failed_requests.source`.
    pub fn as_str(self) -> &'static str {
        match self {
            FailureSource::Ollama => "ollama",
            FailureSource::Proxy => "proxy",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LifetimeTotals {
    pub session_count: u64,
    pub total_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_gen_tokens: u64,
}

enum DbCmd {
    Persist(InferenceRecord),
    PersistFailure(FailedRequest),
    StartSession(oneshot::Sender<Result<i64>>),
    EndSession(i64, oneshot::Sender<Result<()>>),
}

#[derive(Clone)]
pub struct DbHandle {
    tx: mpsc::Sender<DbCmd>,
    db_path: PathBuf,
}

impl DbHandle {
    pub async fn persist(&self, record: InferenceRecord) -> Result<()> {
        self.tx
            .send(DbCmd::Persist(record))
            .await
            .map_err(|_| anyhow::anyhow!("db writer channel closed"))?;
        Ok(())
    }

    pub async fn persist_failure(&self, failure: FailedRequest) -> Result<()> {
        self.tx
            .send(DbCmd::PersistFailure(failure))
            .await
            .map_err(|_| anyhow::anyhow!("db writer channel closed"))?;
        Ok(())
    }

    pub async fn start_session(&self) -> Result<i64> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(DbCmd::StartSession(tx))
            .await
            .map_err(|_| anyhow::anyhow!("db writer channel closed"))?;
        rx.await?
    }

    pub async fn end_session(&self, session_id: i64) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(DbCmd::EndSession(session_id, tx))
            .await
            .map_err(|_| anyhow::anyhow!("db writer channel closed"))?;
        rx.await?
    }

    /// Open a second connection for the lifetime-totals poller. It's an ordinary
    /// read-write connection that only ever reads; the writer task stays the only writer.
    pub fn open_reader(&self) -> Result<Connection> {
        let conn = Connection::open(&self.db_path)
            .with_context(|| format!("open reader for {}", self.db_path.display()))?;
        conn.busy_timeout(std::time::Duration::from_millis(500))?;
        Ok(conn)
    }
}

pub async fn open_and_spawn_writer(
    db_path: &Path,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<DbHandle> {
    let owned_path = db_path.to_path_buf();
    let conn = tokio::task::spawn_blocking({
        let path = owned_path.clone();
        move || open_and_migrate(&path)
    })
    .await??;

    let (tx, mut rx) = mpsc::channel::<DbCmd>(256);
    let writer_path = owned_path.clone();
    tokio::task::spawn_blocking(move || {
        let conn = conn;
        info!(db = %writer_path.display(), "db writer task started");
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                DbCmd::Persist(rec) => {
                    if let Err(err) = insert_record(&conn, &rec) {
                        warn!(error = %err, "failed to insert inference record");
                    }
                }
                DbCmd::PersistFailure(failure) => {
                    if let Err(err) = insert_failure(&conn, &failure) {
                        warn!(error = %err, "failed to insert failed request");
                    }
                }
                DbCmd::StartSession(reply) => {
                    let _ = reply.send(insert_session(&conn));
                }
                DbCmd::EndSession(id, reply) => {
                    let _ = reply.send(close_session(&conn, id));
                }
            }
        }
        info!("db writer task exiting");
    });

    let handle = DbHandle {
        tx,
        db_path: owned_path,
    };

    // Drain the shutdown signal in the background so it isn't dropped before main
    // gets a chance to gracefully wind down. The writer task itself exits when the
    // mpsc sender is dropped.
    tokio::spawn(async move {
        let _ = shutdown_rx.changed().await;
    });

    Ok(handle)
}

pub fn lifetime_totals(reader: &Connection) -> Result<LifetimeTotals> {
    let mut stmt = reader.prepare(
        "SELECT (SELECT COUNT(*) FROM sessions), \
                COALESCE(COUNT(*), 0), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(gen_tokens), 0) \
         FROM inference_records",
    )?;
    let totals = stmt.query_row([], |row| {
        Ok(LifetimeTotals {
            session_count: row.get::<_, i64>(0)? as u64,
            total_requests: row.get::<_, i64>(1)? as u64,
            total_prompt_tokens: row.get::<_, i64>(2)? as u64,
            total_gen_tokens: row.get::<_, i64>(3)? as u64,
        })
    })?;
    Ok(totals)
}

fn open_and_migrate(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path).with_context(|| format!("open db {}", path.display()))?;
    conn.execute_batch(V1_SQL)?;
    conn.execute_batch(V2_SQL)?;
    Ok(conn)
}

fn insert_session(conn: &Connection) -> Result<i64> {
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO sessions (started_at) VALUES (?1)",
        params![now],
    )?;
    Ok(conn.last_insert_rowid())
}

fn close_session(conn: &Connection, id: i64) -> Result<()> {
    let now = Utc::now().to_rfc3339();
    let updated = conn.execute(
        "UPDATE sessions SET ended_at = ?1 WHERE id = ?2 AND ended_at IS NULL",
        params![now, id],
    )?;
    if updated == 0 {
        warn!(session_id = id, "no open session row to close");
    }
    Ok(())
}

fn insert_record(conn: &Connection, rec: &InferenceRecord) -> Result<()> {
    conn.execute(
        "INSERT INTO inference_records (
            session_id, completed_at, model_id, prompt_tokens, gen_tokens,
            tokens_per_sec, ttft_sec, total_time_sec, stop_reason, envelope
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            rec.session_id,
            rec.completed_at.to_rfc3339(),
            rec.model_id,
            rec.prompt_tokens as i64,
            rec.gen_tokens as i64,
            rec.tokens_per_sec,
            rec.ttft_sec,
            rec.total_time_sec,
            rec.stop_reason,
            rec.envelope,
        ],
    )?;
    Ok(())
}

fn insert_failure(conn: &Connection, failure: &FailedRequest) -> Result<()> {
    conn.execute(
        "INSERT INTO failed_requests (
            session_id, failed_at, model_id, path, status, source
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            failure.session_id,
            failure.failed_at.to_rfc3339(),
            failure.model_id,
            failure.path,
            failure.status,
            failure.source.as_str(),
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db_path() -> PathBuf {
        // A counter, not the clock: macOS SystemTime only resolves microseconds, so
        // tests running in parallel could otherwise share (and clobber) one file.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "ollama-monitor-test-{}-{}.db",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn versions(conn: &Connection) -> Vec<i64> {
        let mut stmt = conn
            .prepare("SELECT version FROM schema_version ORDER BY version")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    fn record(session_id: i64) -> InferenceRecord {
        InferenceRecord {
            session_id,
            model_id: "qwen3:14b".into(),
            prompt_tokens: 12,
            gen_tokens: 88,
            tokens_per_sec: 18.4,
            ttft_sec: 0.3,
            total_time_sec: 5.2,
            stop_reason: "stop".into(),
            completed_at: Utc::now(),
            envelope: "ollama-stream".into(),
        }
    }

    fn failure(session_id: i64, model_id: Option<&str>, status: u16) -> FailedRequest {
        FailedRequest {
            session_id,
            failed_at: Utc::now(),
            model_id: model_id.map(str::to_string),
            path: "/v1/chat/completions".into(),
            status,
            source: FailureSource::Ollama,
        }
    }

    /// What a binary from before v2 does on every open.
    fn open_as_v1_binary(path: &Path) -> Connection {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(V1_SQL).unwrap();
        let current: Option<i32> = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .ok();
        if current.is_none() {
            conn.execute("INSERT INTO schema_version (version) VALUES (1)", [])
                .unwrap();
        }
        conn
    }

    #[test]
    fn schema_applies_idempotently() {
        let path = temp_db_path();
        let _ = open_and_migrate(&path).unwrap();
        let _ = open_and_migrate(&path).unwrap();
        let conn = Connection::open(&path).unwrap();
        assert_eq!(versions(&conn), [1, 2]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fresh_db_is_v2() {
        let path = temp_db_path();
        let conn = open_and_migrate(&path).unwrap();
        assert_eq!(versions(&conn), [1, 2]);
        let failures: i64 = conn
            .query_row("SELECT COUNT(*) FROM failed_requests", [], |r| r.get(0))
            .unwrap();
        assert_eq!(failures, 0);
        let _ = std::fs::remove_file(&path);
    }

    /// A monitor built before v2 can be running against the same file while a newer one
    /// adds `failed_requests`, and it has to keep working.
    #[test]
    fn migrates_v1_and_old_binary_keeps_working() {
        let path = temp_db_path();
        let old = open_as_v1_binary(&path);
        let session_id = insert_session(&old).unwrap();
        insert_record(&old, &record(session_id)).unwrap();
        assert_eq!(versions(&old), [1]);

        let new = open_and_migrate(&path).unwrap();
        assert_eq!(versions(&new), [1, 2]);

        // The old connection stays open across the upgrade and keeps inserting, and its
        // setup is a no-op when it starts again.
        insert_record(&old, &record(session_id)).unwrap();
        let reopened = open_as_v1_binary(&path);
        assert_eq!(versions(&reopened), [1, 2]);
        assert_eq!(lifetime_totals(&new).unwrap().total_requests, 2);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn writer_persists_failures() {
        let path = temp_db_path();
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let db = open_and_spawn_writer(&path, shutdown_rx).await.unwrap();
        let session_id = db.start_session().await.unwrap();
        db.persist_failure(failure(session_id, Some("deepseek-v4-pro:cloud"), 429))
            .await
            .unwrap();
        db.persist_failure(FailedRequest {
            source: FailureSource::Proxy,
            ..failure(session_id, None, 502)
        })
        .await
        .unwrap();
        // The writer takes commands in order, so this reply means both inserts have run.
        db.end_session(session_id).await.unwrap();

        let conn = Connection::open(&path).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT session_id, model_id, path, status, source
                 FROM failed_requests ORDER BY id",
            )
            .unwrap();
        let rows: Vec<(i64, Option<String>, String, i64, String)> = stmt
            .query_map([], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let v1 = "/v1/chat/completions".to_string();
        assert_eq!(
            rows,
            [
                (
                    session_id,
                    Some("deepseek-v4-pro:cloud".to_string()),
                    v1.clone(),
                    429,
                    "ollama".to_string()
                ),
                (session_id, None, v1, 502, "proxy".to_string()),
            ]
        );
        // Nothing ran, so a failed request isn't counted as one.
        assert_eq!(lifetime_totals(&conn).unwrap().total_requests, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failure_summary_names_who_answered() {
        let from_ollama = failure(1, None, 429);
        assert_eq!(from_ollama.summary(), "failed (HTTP 429)");
        let from_proxy = FailedRequest {
            source: FailureSource::Proxy,
            ..failure(1, None, 502)
        };
        assert_eq!(from_proxy.summary(), "proxy (HTTP 502)");
    }

    #[test]
    fn insert_and_aggregate() {
        let path = temp_db_path();
        let conn = open_and_migrate(&path).unwrap();
        let session_id = insert_session(&conn).unwrap();
        let rec = record(session_id);
        insert_record(&conn, &rec).unwrap();
        insert_record(&conn, &rec).unwrap();
        let totals = lifetime_totals(&conn).unwrap();
        assert_eq!(totals.session_count, 1);
        assert_eq!(totals.total_requests, 2);
        assert_eq!(totals.total_prompt_tokens, 24);
        assert_eq!(totals.total_gen_tokens, 176);
        close_session(&conn, session_id).unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lifetime_counts_sessions_without_records() {
        let path = temp_db_path();
        let conn = open_and_migrate(&path).unwrap();
        for _ in 0..3 {
            insert_session(&conn).unwrap();
        }
        let totals = lifetime_totals(&conn).unwrap();
        assert_eq!(totals.session_count, 3);
        assert_eq!(totals.total_requests, 0);
        assert_eq!(totals.total_prompt_tokens, 0);
        let _ = std::fs::remove_file(&path);
    }
}
