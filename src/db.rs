use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{info, warn};

const SCHEMA_VERSION: i32 = 1;

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
    pub envelope: String, // "ollama-stream" / "ollama-single" / "openai-sse" / "openai-single"
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LifetimeTotals {
    pub total_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_gen_tokens: u64,
}

enum DbCmd {
    Persist(InferenceRecord),
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

    /// Open a fresh read-only connection. Used by the lifetime-totals poller.
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
        "SELECT COALESCE(COUNT(*), 0), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(gen_tokens), 0) \
         FROM inference_records",
    )?;
    let totals = stmt.query_row([], |row| {
        Ok(LifetimeTotals {
            total_requests: row.get::<_, i64>(0)? as u64,
            total_prompt_tokens: row.get::<_, i64>(1)? as u64,
            total_gen_tokens: row.get::<_, i64>(2)? as u64,
        })
    })?;
    Ok(totals)
}

fn open_and_migrate(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path).with_context(|| format!("open db {}", path.display()))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
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
         CREATE INDEX IF NOT EXISTS idx_records_model ON inference_records(model_id);",
    )?;

    let current: Option<i32> = conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
        .ok();
    if current.is_none() {
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?1)",
            params![SCHEMA_VERSION],
        )?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db_path() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "ollama-monitor-test-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn schema_applies_idempotently() {
        let path = temp_db_path();
        let _ = open_and_migrate(&path).unwrap();
        let _ = open_and_migrate(&path).unwrap();
        let conn = Connection::open(&path).unwrap();
        let v: i32 = conn
            .query_row("SELECT version FROM schema_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn insert_and_aggregate() {
        let path = temp_db_path();
        let conn = open_and_migrate(&path).unwrap();
        let session_id = insert_session(&conn).unwrap();
        let rec = InferenceRecord {
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
        };
        insert_record(&conn, &rec).unwrap();
        insert_record(&conn, &rec).unwrap();
        let totals = lifetime_totals(&conn).unwrap();
        assert_eq!(totals.total_requests, 2);
        assert_eq!(totals.total_prompt_tokens, 24);
        assert_eq!(totals.total_gen_tokens, 176);
        close_session(&conn, session_id).unwrap();
        let _ = std::fs::remove_file(&path);
    }
}
