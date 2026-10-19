//! Perf DB command: schema, ingest, winners, report.
//!
//! Originally a line-for-line port of the (deleted) Python perfdb tool; kernel
//! ids, row selection and report text are unchanged so existing DBs stay valid.
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde_json::Value;
use sha1::{Digest, Sha1};

use crate::pyjson::dumps_sorted;

/// The perf DB schema. Source of truth; v1 columns are frozen (add, never rename).
pub const SCHEMA: &str = include_str!("schema.sql");

/// The schema version this binary reads and writes (table `schema_version`).
/// 1: every DB before the table existed (microbench.superseded);
/// 2: microbench status / clocks, runs idle reading, paired_results;
/// 3: disturbed (the idle guard during the run) on runs, measurements, paired,
///    microbench and paired_results;
/// 4: lt_algo (the cuBLASLt algorithm cublaslt_best picked, every config
///    attribute, JSON) on measurements and winners.
/// 5: profile ("measured" / "unmeasured": did the device match its arch profile,
///    src/hw/profile.h) on measurements, paired, microbench and paired_results.
/// 6: telemetry (opt-in, src/hw/telemetry.h) on measurements: max_temp_c,
///    throttle_reasons (NVML clock-event reasons that slow the clocks, "a|b"; "" none),
///    clock_event_reasons (every reason bit seen), sm_clock_min_mhz / sm_clock_max_mhz.
/// 7: build identity: runs.build_sha (40 hex) / build_dirty / allow_dirty / kind;
///    paired_results.build_sha_a/_b, build_dirty_a/_b, allow_dirty, device_id,
///    sm_arch, dtype; winners.lt_ratio_median (paired) and dispatch.
/// Bump it with every migration in migrate(). connect() backs the DB up
/// (VACUUM INTO <db>.bak-v<old>-<date>) before migrating an older one.
pub const SCHEMA_VERSION: i64 = 7;

/// The DB's schema version: None when it has no schema_version table (a v1
/// DB, or not yet created).
pub fn schema_version(con: &Connection) -> Result<Option<i64>> {
    let has_table: bool =
        con.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'")?.exists([])?;
    if !has_table {
        return Ok(None);
    }
    Ok(con.query_row("SELECT max(version) FROM schema_version", [], |r| r.get::<_, Option<i64>>(0))?)
}

fn version_mismatch(db: &Path, found: i64) -> anyhow::Error {
    let fix = if found < SCHEMA_VERSION {
        "migrate it with this binary: aegis-perf --db <DB> init"
    } else {
        "rebuild aegis-perf from a newer checkout"
    };
    anyhow::anyhow!("{}: DB schema v{found}, binary expects v{SCHEMA_VERSION} ({fix})", db.display())
}

/// Open (creating if needed) and migrate the perf DB to SCHEMA_VERSION.
/// Refuses a DB from a newer aegis-perf instead of touching it.
pub fn connect(db: &Path) -> Result<Connection> {
    if let Some(parent) = db.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let con = Connection::open(db).with_context(|| format!("open {}", db.display()))?;
    if let Some(v) = schema_version(&con).with_context(|| format!("{} is not an AEGIS perf DB", db.display()))? {
        if v > SCHEMA_VERSION {
            return Err(version_mismatch(db, v));
        }
        if v < SCHEMA_VERSION {
            let bak = backup_path(db, v);
            con.execute("VACUUM INTO ?1", [bak.to_string_lossy()])
                .with_context(|| format!("back up {} to {} before migrating", db.display(), bak.display()))?;
            eprintln!("perf DB: v{v} -> v{SCHEMA_VERSION}; backup {}", bak.display());
        }
    }
    con.execute_batch(SCHEMA)?;
    migrate(&con)?;
    con.execute("INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?1)", [SCHEMA_VERSION])?;
    Ok(con)
}

/// Where connect() backs up a DB before migrating it from schema `v`
/// (a free name: never overwrites an earlier backup).
fn backup_path(db: &Path, v: i64) -> std::path::PathBuf {
    let day = chrono::Local::now().format("%Y-%m-%d");
    let base = format!("{}.bak-v{v}-{day}", db.display());
    let mut p = std::path::PathBuf::from(&base);
    let mut i = 1;
    while p.exists() {
        p = format!("{base}.{i}").into();
        i += 1;
    }
    p
}

/// Open an existing perf DB read-only (SQLite URI `mode=ro`), for viewers: no
/// schema creation, no migration, never a write, so a running tune or sweep
/// is never blocked. Refuses a DB whose schema_version is not SCHEMA_VERSION
/// ("DB schema vN, binary expects vM"), and one missing a column this crate
/// reads.
pub fn connect_ro(db: &Path) -> Result<Connection> {
    let path = db.to_str().with_context(|| format!("non-UTF-8 path {}", db.display()))?;
    let uri = format!("file:{}?mode=ro", path.replace('%', "%25").replace('?', "%3f").replace('#', "%23"));
    let con = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open {} read-only", db.display()))?;
    let found = schema_version(&con).with_context(|| format!("{} is not an AEGIS perf DB", db.display()))?;
    if found != Some(SCHEMA_VERSION) {
        return Err(version_mismatch(db, found.unwrap_or(1)));
    }
    for (table, column, _) in ADDED_COLUMNS {
        let exists: bool = con
            .prepare(&format!("SELECT 1 FROM pragma_table_info('{table}') WHERE name = ?"))?
            .exists([column])
            .with_context(|| format!("{} is not an AEGIS perf DB", db.display()))?;
        if !exists {
            bail!("{}: {table}.{column} missing; the DB predates this aegis-perf (migrate it with aegis-perf init)", db.display());
        }
    }
    Ok(con)
}

/// Columns added after v1. CREATE TABLE IF NOT EXISTS leaves existing tables
/// alone, so add any missing column here (additive only; never rename).
const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    ("winners", "tied", "INTEGER NOT NULL DEFAULT 0"),
    ("winners", "tie_set", "TEXT"),
    ("winners", "pick_reason", "TEXT"),
    // Microbench row status (replaced the superseded flag; see migrate()).
    ("microbench", "status", "TEXT NOT NULL DEFAULT 'valid' CHECK (status IN ('valid', 'replaced', 'invalidated'))"),
    ("microbench", "status_reason", "TEXT CHECK (status <> 'invalidated' OR length(trim(coalesce(status_reason, ''))) > 0)"),
    ("microbench", "replaced_by", "TEXT"),
    // Latency rows: the SM clock at measurement and whether it was locked.
    ("microbench", "sm_clock_mhz", "REAL"),
    ("microbench", "clocks_locked", "INTEGER"),
    ("microbench", "clock_lock", "TEXT"),
    // paired_results is created by the schema; listed so connect_ro refuses a
    // DB that predates it.
    ("paired_results", "finished_at", "TEXT NOT NULL DEFAULT ''"),
    // Timing protocol v2: rows from before it get protocol_version = 1.
    ("measurements", "protocol_version", "INTEGER NOT NULL DEFAULT 1"),
    ("measurements", "batch_n", "INTEGER"),
    ("measurements", "timer_res_ns", "REAL"),
    ("measurements", "cache_mode", "TEXT"),
    ("measurements", "buffer_sets", "INTEGER"),
    ("measurements", "window_ms", "REAL"),
    ("measurements", "allow_busy", "INTEGER NOT NULL DEFAULT 0"),
    ("measurements", "idle_util_pct", "REAL"),
    ("winners", "protocol_version", "INTEGER NOT NULL DEFAULT 1"),
    ("winners", "warm_median_us", "REAL"),
    ("winners", "pct_of_cublaslt_best", "REAL"),
    ("winners", "tie_method", "TEXT"),
    ("winners", "ratio_ci_lo", "REAL"),
    ("winners", "ratio_ci_hi", "REAL"),
    ("winners", "lt_ratio_ci_lo", "REAL"),
    ("winners", "lt_ratio_ci_hi", "REAL"),
    // The idle guard's reading, per run (every run since the blocking guard;
    // older runs backfilled from their measurement rows where those carry it).
    ("runs", "idle_util_pct", "REAL"),
    ("runs", "idle_source", "TEXT"),
    ("runs", "allow_busy", "INTEGER NOT NULL DEFAULT 0"),
    // The idle guard during the run (bench_gemm --idle-watch): 1 = other
    // processes used the GPU above the limit mid-run, 0 = watched and clean,
    // NULL = not watched. Every row of a disturbed run is marked.
    ("runs", "disturbed", "INTEGER"),
    ("runs", "disturbed_detail", "TEXT"),
    ("measurements", "disturbed", "INTEGER"),
    ("paired", "disturbed", "INTEGER"),
    ("microbench", "disturbed", "INTEGER"),
    ("paired_results", "disturbed", "INTEGER"),
    ("paired_results", "disturbed_detail", "TEXT"),
    // v4: the cuBLASLt algorithm cublaslt_best picked (JSON: algo_id and every
    // config attribute; src/lt_algo.h), so aegis::matmul can rebuild it.
    ("measurements", "lt_algo", "TEXT"),
    ("winners", "lt_algo", "TEXT"),
    // v5: the device profile the row was measured under (NULL: before v5).
    ("measurements", "profile", "TEXT"),
    ("paired", "profile", "TEXT"),
    ("microbench", "profile", "TEXT"),
    ("paired_results", "profile", "TEXT"),
    // v6: telemetry over the measurement's timed windows (NULL: not sampled).
    ("measurements", "max_temp_c", "REAL"),
    ("measurements", "throttle_reasons", "TEXT"),
    ("measurements", "clock_event_reasons", "TEXT"),
    ("measurements", "sm_clock_min_mhz", "INTEGER"),
    ("measurements", "sm_clock_max_mhz", "INTEGER"),
    // v7: build identity. build_sha: the full commit the build was made from (its
    // rows' "build_sha", src/bench/build_info.h); build_dirty: 1 its tree had
    // uncommitted tracked changes, 0 clean, NULL unknown (runs before v7).
    // allow_dirty: written with --allow-dirty (a dirty or unknown build): marked
    // and excluded by default (Selection, view, regress), like allow_busy.
    ("runs", "build_sha", "TEXT"),
    ("runs", "build_dirty", "INTEGER"),
    ("runs", "allow_dirty", "INTEGER NOT NULL DEFAULT 0"),
    // What the run is (RunKind): tune, sweep, nightly, microbench, ingest.
    ("runs", "kind", "TEXT"),
    ("paired_results", "build_sha_a", "TEXT"),
    ("paired_results", "build_sha_b", "TEXT"),
    ("paired_results", "build_dirty_a", "INTEGER"),
    ("paired_results", "build_dirty_b", "INTEGER"),
    ("paired_results", "allow_dirty", "INTEGER NOT NULL DEFAULT 0"),
    ("paired_results", "device_id", "TEXT"),
    ("paired_results", "sm_arch", "INTEGER"),
    ("paired_results", "dtype", "TEXT"),
    // v7: the pick's paired median ratio cublaslt_best / pick (the number of record;
    // pct_of_cublaslt_best is the unpaired median), and where aegis::matmul sends
    // the shape ("aegis" | "cublaslt": tune::dispatch on the paired CI).
    ("winners", "lt_ratio_median", "REAL"),
    ("winners", "dispatch", "TEXT"),
];

/// Runs whose rows are excluded by default for their build: written with
/// --allow-dirty, or from a build known to be dirty.
pub const DIRTY_RUNS: &str = "SELECT run_id FROM runs WHERE allow_dirty = 1 OR coalesce(build_dirty, 0) = 1";

/// What a run is (runs.kind, schema v7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunKind {
    /// aegis-perf tune (aegis_tune: every config, paired tie tests).
    Tune,
    /// aegis-perf sweep run by hand.
    Sweep,
    /// The nightly regression sweep (tools/nightly.sh: aegis-perf sweep --kind nightly).
    Nightly,
    /// microbench/run_all.sh, run_latency_locked.sh (ingest-microbench).
    Microbench,
    /// aegis-perf ingest of a results file.
    Ingest,
}

impl RunKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RunKind::Tune => "tune",
            RunKind::Sweep => "sweep",
            RunKind::Nightly => "nightly",
            RunKind::Microbench => "microbench",
            RunKind::Ingest => "ingest",
        }
    }

    pub fn parse(s: &str) -> Option<RunKind> {
        [RunKind::Tune, RunKind::Sweep, RunKind::Nightly, RunKind::Microbench, RunKind::Ingest]
            .into_iter()
            .find(|k| k.as_str() == s)
    }

    /// A sweep-like run (a subset of configs, no paired tie tests): sweep or nightly.
    pub fn is_sweep(self) -> bool {
        matches!(self, RunKind::Sweep | RunKind::Nightly)
    }
}

/// A build's identity: the commit it was made from and whether its tree had
/// uncommitted tracked changes when it was built (src/bench/build_info.h).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildId {
    /// 40 hex digits.
    pub sha: String,
    pub dirty: bool,
}

impl BuildId {
    pub fn new(sha: &str, dirty: bool) -> Result<BuildId> {
        if sha.len() != 40 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("build sha '{sha}' is not a full 40-hex-digit commit");
        }
        Ok(BuildId { sha: sha.to_ascii_lowercase(), dirty })
    }

    /// The pre-v7 short form: 12 hex digits, "-dirty" if dirty (paired_results.sha_a/sha_b).
    pub fn label(&self) -> String {
        format!("{}{}", &self.sha[..12], if self.dirty { "-dirty" } else { "" })
    }

    /// runs.git_sha (7 hex digits, no dirty marker; kept for pre-v7 readers).
    pub fn short(&self) -> &str {
        &self.sha[..7]
    }
}

/// The build a result row names ("build_sha", "build_dirty"); None when the row
/// carries none (a binary from before schema v7, or a build outside a checkout).
pub fn build_of_row(r: &Value) -> Result<Option<BuildId>> {
    match r.get("build_sha").and_then(Value::as_str) {
        None => Ok(None),
        Some(sha) => Ok(Some(BuildId::new(sha, r.get("build_dirty").and_then(Value::as_bool).unwrap_or(true))?)),
    }
}

/// The build of a bench_gemm binary (`bench_gemm --build-info`); None for a
/// binary that predates the flag or was built outside a checkout.
pub fn build_of(bench: &Path) -> Result<Option<BuildId>> {
    let out = Command::new(bench)
        .arg("--build-info")
        .stderr(Stdio::null())
        .output()
        .with_context(|| format!("run {} --build-info", bench.display()))?;
    if !out.status.success() {
        return Ok(None);
    }
    let v: Value = serde_json::from_slice(&out.stdout).with_context(|| format!("{} --build-info", bench.display()))?;
    build_of_row(&v)
}

/// Refuse to write results from a dirty or unknown build unless `allow_dirty`:
/// a result counts only if a commit rebuilds what was measured.
pub fn check_build(what: &str, build: Option<&BuildId>, allow_dirty: bool) -> Result<()> {
    let problem = match build {
        Some(b) if !b.dirty => return Ok(()),
        Some(b) => format!("build {} has uncommitted changes", b.label()),
        None => "the build's commit is unknown (a binary from before schema v7, or built outside a checkout)".to_string(),
    };
    if !allow_dirty {
        bail!(
            "{what}: refusing to store results: {problem}. Commit, rebuild and rerun, or pass --allow-dirty \
             (rows are marked allow_dirty and excluded by default)"
        );
    }
    eprintln!("{what}: --allow-dirty: {problem}; every row is marked allow_dirty and excluded by default");
    Ok(())
}

/// Reference implementations: always eligible (they are what --check compares
/// against), never winners.
const BASELINES: &str = "'cublas', 'cublaslt_best'";

/// aegis::matmul itself (bench_gemm --impl aegis_matmul): the dispatch as a
/// library caller sees it. Measured (nightly sweep, paired-ab), never a winner.
pub const DISPATCH_IMPL: &str = "aegis_matmul";

pub fn is_baseline(impl_: &str) -> bool {
    impl_ == "cublas" || impl_ == "cublaslt_best"
}

/// Which measurements a winners / report query reads.
#[derive(Clone, Debug)]
pub struct Selection {
    pub protocol: i64,
    /// v2 cache mode ("cold" / "warm"); ignored for protocol 1.
    pub cache: String,
    /// Include rows measured on a busy GPU (--allow-busy) or disturbed mid-run.
    pub include_busy: bool,
    /// Include rows of runs from a dirty build or written with --allow-dirty
    /// (DIRTY_RUNS). Runs from before schema v7 (build unknown) are included.
    pub include_dirty: bool,
}

impl Selection {
    /// What winners are picked from: protocol v2, cold, idle GPU, clean build.
    pub fn ranking() -> Self {
        Selection { protocol: 2, cache: "cold".to_string(), include_busy: false, include_dirty: false }
    }

    pub(crate) fn sql(&self) -> String {
        let mut s = format!(" AND m.protocol_version = {}", self.protocol);
        if self.protocol >= 2 {
            s += &format!(" AND m.cache_mode = '{}'", self.cache.replace('\'', "''"));
        }
        if !self.include_busy {
            s += " AND m.allow_busy = 0 AND coalesce(m.disturbed, 0) = 0";
        }
        if !self.include_dirty {
            s += &format!(" AND m.run_id NOT IN ({DIRTY_RUNS})");
        }
        s
    }
}

fn migrate(con: &Connection) -> Result<()> {
    for (table, column, decl) in ADDED_COLUMNS {
        let exists: bool = con
            .prepare(&format!("SELECT 1 FROM pragma_table_info('{table}') WHERE name = ?"))?
            .exists([column])?;
        if !exists {
            con.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))?;
            if (*table, *column) == ("runs", "allow_busy") {
                // Backfill once: a run's reading is what its measurement rows recorded.
                con.execute_batch(
                    "UPDATE runs SET \
                       idle_util_pct = (SELECT max(m.idle_util_pct) FROM measurements m WHERE m.run_id = runs.run_id), \
                       allow_busy = coalesce((SELECT max(m.allow_busy) FROM measurements m WHERE m.run_id = runs.run_id), 0)",
                )?;
            }
            if (*table, *column) == ("runs", "kind") {
                // v7 backfill, from what each run stored (every row, disturbed or
                // not): paired tie tests = a tune, microbench rows = microbench,
                // else a sweep (a nightly and a hand-run sweep look alike before v7).
                con.execute_batch(
                    "UPDATE runs SET kind = CASE \
                       WHEN EXISTS (SELECT 1 FROM paired p WHERE p.run_id = runs.run_id) THEN 'tune' \
                       WHEN EXISTS (SELECT 1 FROM microbench b WHERE b.run_id = runs.run_id) THEN 'microbench' \
                       WHEN EXISTS (SELECT 1 FROM measurements m WHERE m.run_id = runs.run_id) THEN 'sweep' END",
                )?;
            }
            if (*table, *column) == ("paired_results", "build_dirty_b") {
                // v7 backfill: before v7 a dirty build was the "-dirty" suffix on
                // sha_a / sha_b (12 hex digits; NULL = unknown). The full SHA is not
                // recoverable, so build_sha_a / _b stay NULL.
                con.execute_batch(
                    "UPDATE paired_results SET \
                       build_dirty_a = CASE WHEN sha_a IS NULL THEN NULL WHEN sha_a LIKE '%-dirty' THEN 1 ELSE 0 END, \
                       build_dirty_b = CASE WHEN sha_b IS NULL THEN NULL WHEN sha_b LIKE '%-dirty' THEN 1 ELSE 0 END",
                )?;
            }
        }
    }
    // superseded (0/1) -> status: superseded rows were re-measured by a later
    // run, which is "replaced" (still valid history), never "invalidated".
    let has_superseded: bool =
        con.prepare("SELECT 1 FROM pragma_table_info('microbench') WHERE name = 'superseded'")?.exists([])?;
    if has_superseded {
        con.execute_batch(
            "SAVEPOINT superseded_to_status;
             UPDATE microbench SET status = 'replaced', replaced_by = superseded_by WHERE superseded = 1;
             ALTER TABLE microbench DROP COLUMN superseded_by;
             ALTER TABLE microbench DROP COLUMN superseded;
             RELEASE superseded_to_status;",
        )?;
    }
    Ok(())
}

pub fn shape_bucket(m: i64, n: i64, k: i64) -> String {
    format!("{m}x{n}x{k}")
}

pub fn kernel_id(op: &str, impl_: &str, config: &Value) -> String {
    let key = format!("{op}|{impl_}|{}", dumps_sorted(config));
    let digest = Sha1::digest(key.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect::<String>()[..16].to_string()
}

fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Jetson: nvpmodel mode. Elsewhere: None.
fn power_mode() -> Option<String> {
    let out = command_stdout("nvpmodel", &["-q"])?;
    out.trim().lines().next().map(str::to_string)
}

fn gpu_temp_c() -> Option<f64> {
    if let Some(out) = command_stdout(
        "nvidia-smi",
        &["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
    ) {
        if let Some(t) = out.trim().lines().next().and_then(|l| l.trim().parse().ok()) {
            return Some(t);
        }
    }
    // Jetson thermal zones (millidegrees); take the GPU zone if named, else max.
    let mut temps = Vec::new();
    for entry in std::fs::read_dir("/sys/class/thermal").ok()?.flatten() {
        if !entry.file_name().to_string_lossy().starts_with("thermal_zone") {
            continue;
        }
        let p = entry.path();
        let (Ok(t), Ok(name)) = (std::fs::read_to_string(p.join("temp")), std::fs::read_to_string(p.join("type")))
        else {
            continue;
        };
        let Ok(t) = t.trim().parse::<f64>() else { continue };
        let t = t / 1000.0;
        if name.trim().to_lowercase().contains("gpu") {
            return Some(t);
        }
        temps.push(t);
    }
    temps.into_iter().reduce(f64::max)
}

pub fn hostname() -> String {
    gethostname::gethostname().to_string_lossy().into_owned()
}

/// runs.device_id's convention: the device_id of results/manifest-<host>.json, else the host name.
pub fn device_id(results: &Path) -> String {
    let host = hostname();
    std::fs::read_to_string(results.join(format!("manifest-{host}.json")))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get("device_id").and_then(Value::as_str).map(str::to_string))
        .unwrap_or(host)
}

/// JSON -> SQLite value with the types Python's sqlite3 would bind.
fn sql(v: Option<&Value>) -> SqlValue {
    match v {
        None | Some(Value::Null) => SqlValue::Null,
        Some(Value::Bool(b)) => SqlValue::Integer(*b as i64),
        Some(Value::Number(n)) => match n.as_i64() {
            Some(i) => SqlValue::Integer(i),
            None => SqlValue::Real(n.as_f64().unwrap_or(f64::NAN)),
        },
        Some(Value::String(s)) => SqlValue::Text(s.clone()),
        Some(other) => SqlValue::Text(other.to_string()),
    }
}

fn get<'a>(r: &'a Value, key: &str) -> Result<&'a Value> {
    r.get(key).with_context(|| format!("results row missing \"{key}\""))
}

const INSERT_RUN: &str = "INSERT INTO runs (run_id, device_id, sm_arch, gpu_name, cuda_ver, driver_ver, git_sha, \
     power_mode, clocks_locked, temp_start_c, measured_bw_gbps, started_at, idle_util_pct, idle_source, allow_busy, \
     build_sha, build_dirty, allow_dirty, kind) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

/// How a run is stored: what it is, and whether a dirty or unknown build may
/// write it (--allow-dirty; its rows are then marked and excluded by default).
#[derive(Clone, Copy, Debug)]
pub struct IngestOpts {
    pub kind: RunKind,
    pub allow_dirty: bool,
}

impl IngestOpts {
    pub fn new(kind: RunKind) -> Self {
        IngestOpts { kind, allow_dirty: false }
    }
}

/// The build every row of a results file names; an error if rows disagree.
fn build_of_rows(rows: &[&Value]) -> Result<Option<BuildId>> {
    let mut builds = Vec::new();
    for r in rows {
        let b = build_of_row(r)?;
        if !builds.contains(&b) {
            builds.push(b);
        }
    }
    match builds.len() {
        0 => Ok(None),
        1 => Ok(builds.pop().flatten()),
        _ => bail!("rows of one results file name different builds: {builds:?}"),
    }
}

/// A run's idle-guard reading (Benchmark policy: stored on every run row).
#[derive(Clone, Debug, Default)]
pub struct RunIdle {
    pub pct: Option<f64>,
    pub source: Option<String>,
    pub allow_busy: bool,
}

/// Store what the in-run idle watch saw on a run and on every row of it
/// (measurements, paired tie tests, microbench): disturbed 1 / 0, or NULL when
/// the watch had no source.
pub fn mark_disturbed(con: &Connection, run_id: &str, w: &crate::watch::WatchReport) -> Result<()> {
    let v = w.disturbed.map(i64::from);
    let n = con.execute(
        "UPDATE runs SET disturbed = ?2, disturbed_detail = ?3 WHERE run_id = ?1",
        params![run_id, v, w.detail()],
    )?;
    if n == 0 {
        bail!("no run {run_id}");
    }
    for table in ["measurements", "paired", "microbench"] {
        con.execute(&format!("UPDATE {table} SET disturbed = ?2 WHERE run_id = ?1"), params![run_id, v])?;
    }
    if w.disturbed == Some(true) {
        eprintln!("run {run_id}: every row marked disturbed=1 ({})", w.detail());
        // A disturbed run is not evidence, so it replaces nothing: the microbench rows its
        // ingest marked replaced are valid again (2026-10-18: a disturbed throttle_profile
        // run had retired the clean one).
        let restored = con.execute(
            "UPDATE microbench SET status = 'valid', replaced_by = NULL WHERE replaced_by = ?1 AND status = 'replaced'",
            params![run_id],
        )?;
        if restored > 0 {
            eprintln!("run {run_id}: {restored} older microbench rows it had replaced are valid again");
        }
    }
    Ok(())
}

pub fn cmd_init(db: &Path) -> Result<()> {
    connect(db)?;
    println!("initialized {}", db.display());
    Ok(())
}

pub fn cmd_ingest(
    db: &Path,
    results: &Path,
    manifest: Option<&Path>,
    run_id: Option<String>,
    temp_start: Option<f64>,
    opts: IngestOpts,
) -> Result<String> {
    let text = std::fs::read_to_string(results).with_context(|| format!("read {}", results.display()))?;
    let all: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    // aegis_tune interleaves paired tie-test lines ("kind":"paired") with measurement rows.
    let (paired, rows): (Vec<Value>, Vec<Value>) = all.into_iter().partition(|r| r["kind"] == "paired");
    if rows.is_empty() {
        bail!("no rows in results file");
    }
    // The build that measured the rows (every row names it, schema v7): refused
    // when dirty or unknown unless --allow-dirty.
    let build = build_of_rows(&rows.iter().chain(&paired).collect::<Vec<_>>())?;
    check_build(&format!("ingest {}", results.display()), build.as_ref(), opts.allow_dirty)?;
    let manifest: Value = match manifest {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => Value::Object(Default::default()),
    };
    let first = &rows[0];
    let mut con = connect(db)?;
    let tx = con.transaction()?;
    let run_id = run_id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()[..12].to_string());
    let device_id = manifest
        .get("device_id")
        .cloned()
        .unwrap_or_else(|| Value::String(hostname()));
    // Python: datetime.now(timezone.utc).isoformat()
    let started_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string();
    // The run's idle-guard reading: aegis_tune puts it on every row, sweep
    // (bench_gemm --idle-util-pct) too; allow_busy if any row ran busy.
    let idle = RunIdle {
        pct: rows.iter().find_map(|r| r.get("idle_util_pct").and_then(Value::as_f64)),
        source: rows.iter().find_map(|r| r.get("idle_source").and_then(Value::as_str).map(str::to_string)),
        allow_busy: rows.iter().any(|r| r.get("allow_busy").and_then(Value::as_bool) == Some(true)),
    };
    tx.execute(
        INSERT_RUN,
        params![
            run_id,
            sql(Some(&device_id)),
            sql(Some(get(first, "sm_arch")?)),
            sql(first.get("device")),
            sql(first.get("cuda_runtime")),
            sql(first.get("cuda_driver")),
            build.as_ref().map(BuildId::short),
            power_mode(),
            SqlValue::Null,
            temp_start.or_else(gpu_temp_c),
            sql(manifest.get("measured_d2d_bw_gbps")),
            started_at,
            idle.pct,
            idle.source,
            idle.allow_busy as i64,
            build.as_ref().map(|b| b.sha.as_str()),
            build.as_ref().map(|b| b.dirty as i64),
            (opts.allow_dirty && build.as_ref().is_none_or(|b| b.dirty)) as i64,
            opts.kind.as_str(),
        ],
    )?;
    for r in &rows {
        let op = get(r, "op")?.as_str().unwrap_or_default();
        let impl_ = get(r, "impl")?.as_str().unwrap_or_default();
        let config = get(r, "config")?;
        let kid = kernel_id(op, impl_, config);
        tx.execute(
            "INSERT OR IGNORE INTO kernels(kernel_id, op, impl, config_json, regs, smem_bytes, local_bytes) \
             VALUES (?,?,?,?,?,?,?)",
            params![kid, op, impl_, dumps_sorted(config), sql(r.get("regs")), sql(r.get("smem_bytes")),
                    sql(r.get("local_bytes"))],
        )?;
        let s = get(r, "shape")?;
        // Rows written before timing protocol v2 carry no protocol_version: v1.
        let protocol = r.get("protocol_version").and_then(Value::as_i64).unwrap_or(1);
        tx.execute(
            "INSERT INTO measurements (run_id, kernel_id, dtype, m, n, k, median_us, p10_us, p90_us, \
             tflops, gbps, reps, checked, pass, max_abs_err, rel_err, power_w, protocol_version, batch_n, \
             timer_res_ns, cache_mode, buffer_sets, window_ms, allow_busy, idle_util_pct, lt_algo, profile, \
             max_temp_c, throttle_reasons, clock_event_reasons, sm_clock_min_mhz, sm_clock_max_mhz) \
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            params![
                run_id, kid, sql(Some(get(r, "dtype")?)),
                sql(Some(get(s, "M")?)), sql(Some(get(s, "N")?)), sql(Some(get(s, "K")?)),
                sql(Some(get(r, "median_us")?)), sql(Some(get(r, "p10_us")?)), sql(Some(get(r, "p90_us")?)),
                sql(Some(get(r, "tflops")?)), sql(Some(get(r, "gbps")?)), sql(Some(get(r, "reps")?)),
                get(r, "checked")?.as_bool().unwrap_or(false) as i64,
                get(r, "pass")?.as_bool().unwrap_or(false) as i64,
                sql(Some(get(r, "max_abs_err")?)), sql(Some(get(r, "rel_err")?)),
                sql(r.get("power_w")),
                protocol,
                sql(r.get("batch_n")),
                sql(r.get("timer_res_ns")),
                sql(r.get("cache_mode")),
                sql(r.get("buffer_sets")),
                sql(r.get("window_ms")),
                r.get("allow_busy").and_then(Value::as_bool).unwrap_or(false) as i64,
                sql(r.get("idle_util_pct")),
                r.get("lt").filter(|v| v.is_object()).map(dumps_sorted),
                r.get("profile").and_then(Value::as_str),
                sql(r.get("max_temp_c")),
                r.get("throttle_reasons").and_then(Value::as_str),
                r.get("clock_event_reasons").and_then(Value::as_str),
                sql(r.get("sm_clock_min_mhz")),
                sql(r.get("sm_clock_max_mhz")),
            ],
        )?;
    }
    for r in &paired {
        let s = get(r, "shape")?;
        tx.execute(
            "INSERT INTO paired (run_id, op, dtype, m, n, k, a_impl, b_impl, role, pairs, ratio_median, \
             ratio_ci_lo, ratio_ci_hi, tied, bootstrap_resamples, bootstrap_seed, protocol_version, cache_mode, \
             allow_busy, profile) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            params![
                run_id, sql(Some(get(r, "op")?)), sql(Some(get(r, "dtype")?)),
                sql(Some(get(s, "M")?)), sql(Some(get(s, "N")?)), sql(Some(get(s, "K")?)),
                sql(Some(get(r, "a")?)), sql(Some(get(r, "b")?)), sql(r.get("role")), sql(Some(get(r, "pairs")?)),
                sql(Some(get(r, "ratio_median")?)), sql(Some(get(r, "ratio_ci_lo")?)),
                sql(Some(get(r, "ratio_ci_hi")?)), get(r, "tied")?.as_bool().unwrap_or(false) as i64,
                sql(r.get("bootstrap_resamples")), sql(r.get("bootstrap_seed")), sql(r.get("protocol_version")),
                sql(r.get("cache_mode")), r.get("allow_busy").and_then(Value::as_bool).unwrap_or(false) as i64,
                r.get("profile").and_then(Value::as_str),
            ],
        )?;
    }
    tx.commit()?;
    println!("run {run_id}: ingested {} measurements, {} paired tie tests", rows.len(), paired.len());
    Ok(run_id)
}

// arch, op, dtype, m, n, k, impl. Keyed by impl, not kernel_id: a code change that
// alters the config serialization (e.g. "smem":"pad" -> "xor") gives the same
// config a new kernel_id, and its older rows must not compete with the new ones.
type KernelShapeKey = (i64, String, String, i64, i64, i64, String);
type ShapeKey = (i64, String, String, i64, i64, i64); // arch, op, dtype, m, n, k

/// Microbench probe lines (microbench/*): one run row with the device metadata
/// of the first line, then (run_id, probe, param_json, value, unit) per line.
/// `build`: the build the probes came from (probe lines carry none; the CLI
/// takes it from the same build's `bench_gemm --build-info`).
pub fn cmd_ingest_microbench(
    db: &Path,
    results: &Path,
    manifest: Option<&Path>,
    run_id: Option<String>,
    idle: &RunIdle,
    build: Option<&BuildId>,
    allow_dirty: bool,
) -> Result<String> {
    check_build(&format!("ingest-microbench {}", results.display()), build, allow_dirty)?;
    let text = std::fs::read_to_string(results).with_context(|| format!("read {}", results.display()))?;
    let rows: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if rows.is_empty() {
        bail!("no rows in microbench file");
    }
    let manifest: Value = match manifest {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => Value::Object(Default::default()),
    };
    let first = &rows[0];
    // runs.clocks_locked: 1 if every row that records a lock state was locked,
    // 0 if any was not, NULL if no row records one (non-latency probes).
    let lock_states: Vec<bool> = rows.iter().filter_map(|r| r.get("clocks_locked").and_then(Value::as_bool)).collect();
    let clocks_locked: Option<i64> = (!lock_states.is_empty()).then(|| i64::from(lock_states.iter().all(|&l| l)));
    let mut con = connect(db)?;
    let tx = con.transaction()?;
    let run_id = run_id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()[..12].to_string());
    let device_id = manifest.get("device_id").cloned().unwrap_or_else(|| Value::String(hostname()));
    let started_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string();
    tx.execute(
        INSERT_RUN,
        params![
            run_id, sql(Some(&device_id)), sql(Some(get(first, "sm_arch")?)), sql(first.get("device")),
            sql(first.get("cuda_runtime")), sql(first.get("cuda_driver")), build.map(BuildId::short), power_mode(),
            clocks_locked, gpu_temp_c(), sql(manifest.get("measured_d2d_bw_gbps")), started_at,
            idle.pct, idle.source, idle.allow_busy as i64,
            build.map(|b| b.sha.as_str()), build.map(|b| b.dirty as i64),
            (allow_dirty && build.is_none_or(|b| b.dirty)) as i64, RunKind::Microbench.as_str(),
        ],
    )?;
    let mut probes: Vec<String> = Vec::new();
    for r in &rows {
        let probe = get(r, "probe")?.as_str().unwrap_or_default().to_string();
        tx.execute(
            "INSERT INTO microbench (run_id, probe, param_json, value, unit, sm_clock_mhz, clocks_locked, clock_lock, \
             profile) VALUES (?,?,?,?,?,?,?,?,?)",
            params![
                run_id,
                probe,
                dumps_sorted(get(r, "param")?),
                sql(Some(get(r, "value")?)),
                sql(Some(get(r, "unit")?)),
                r.get("sm_clock_mhz").and_then(Value::as_f64),
                r.get("clocks_locked").and_then(Value::as_bool).map(i64::from),
                r.get("clock_lock").and_then(Value::as_str),
                r.get("profile").and_then(Value::as_str),
            ],
        )?;
        if !probes.contains(&probe) {
            probes.push(probe);
        }
    }
    // Older valid rows of a re-measured probe on the same arch are kept as
    // history, marked replaced (never invalidated: nothing says they were wrong).
    let mut replaced = 0;
    for probe in &probes {
        replaced += tx.execute(
            "UPDATE microbench SET status = 'replaced', replaced_by = ?1 \
             WHERE probe = ?2 AND run_id <> ?1 AND status = 'valid' \
               AND run_id IN (SELECT run_id FROM runs WHERE sm_arch = ?3)",
            params![run_id, probe, sql(Some(get(first, "sm_arch")?))],
        )?;
    }
    tx.commit()?;
    println!("run {run_id}: ingested {} microbench rows; marked {replaced} older rows replaced", rows.len());
    Ok(run_id)
}

/// The run the hardware model uses for a probe on an arch: the newest run with
/// valid rows; if there is none, the newest replaced one (still valid history).
/// Invalidated rows never count.
pub fn current_run(con: &Connection, arch: i64, probe: &str) -> Result<Option<String>> {
    for status in ["valid", "replaced"] {
        let run: Option<String> = con
            .query_row(
                "SELECT m.run_id FROM microbench m JOIN runs r USING(run_id) \
                 WHERE m.probe = ?1 AND r.sm_arch = ?2 AND m.status = ?3 AND coalesce(m.disturbed, 0) = 0 ORDER BY r.started_at DESC LIMIT 1",
                params![probe, arch, status],
                |r| r.get(0),
            )
            .optional()?;
        if run.is_some() {
            return Ok(run);
        }
    }
    Ok(None)
}

/// Mark a run's microbench rows (all probes, or one) invalidated: known bad,
/// hidden everywhere. The reason is required. If those rows were the current
/// (valid) ones, the rows they had replaced become valid again, so the probe
/// still has a current run; rows of a run that was itself already replaced
/// leave the newer run current.
pub fn invalidate_microbench(con: &mut Connection, run_id: &str, probe: Option<&str>, reason: &str) -> Result<usize> {
    if reason.trim().is_empty() {
        bail!("invalidating needs a reason");
    }
    let tx = con.transaction()?;
    let probes: Vec<(String, String)> = tx
        .prepare(
            "SELECT DISTINCT probe, status FROM microbench WHERE run_id = ?1 AND (?2 IS NULL OR probe = ?2) \
             AND status <> 'invalidated'",
        )?
        .query_map(params![run_id, probe], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    if probes.is_empty() {
        bail!("run {run_id} has no microbench rows to invalidate{}", probe.map(|p| format!(" for probe {p}")).unwrap_or_default());
    }
    let mut n = 0;
    for (p, status) in &probes {
        n += tx.execute(
            "UPDATE microbench SET status = 'invalidated', status_reason = ?3, replaced_by = NULL \
             WHERE run_id = ?1 AND probe = ?2 AND status <> 'invalidated'",
            params![run_id, p, reason],
        )?;
        if status == "valid" {
            tx.execute(
                "UPDATE microbench SET status = 'valid', replaced_by = NULL \
                 WHERE probe = ?2 AND replaced_by = ?1 AND status = 'replaced'",
                params![run_id, p],
            )?;
        }
    }
    tx.commit()?;
    Ok(n)
}

pub fn cmd_invalidate(db: &Path, run_id: &str, probe: Option<&str>, reason: &str) -> Result<()> {
    let mut con = connect(db)?;
    let n = invalidate_microbench(&mut con, run_id, probe, reason)?;
    println!("run {run_id}: invalidated {n} microbench rows ({reason})");
    Ok(())
}

/// Latest passing measurement per (arch, impl, shape).
#[derive(Clone, Debug)]
pub struct Latest {
    pub sm_arch: i64,
    pub op: String,
    pub dtype: String,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub impl_: String,
    pub kernel_id: String,
    pub run_id: String,
    pub median_us: f64,
    pub tflops: f64,
    pub p10_us: f64,
    pub p90_us: f64,
    pub regs: Option<i64>,
    pub smem_bytes: Option<i64>,
}

/// Most recent measurement per (arch, impl, shape); only checked+passing rows
/// are eligible, except cuBLAS which is the reference itself. Keeps Python's
/// dict semantics: first-seen key order, later runs overwrite values.
pub fn latest_per_kernel_shape(con: &Connection, arch: Option<i64>, sel: &Selection) -> Result<Vec<Latest>> {
    latest_rows(con, arch, sel, "")
}

/// Winners rank on tune runs only: a run with paired tie tests timed every
/// config of its shapes (aegis_tune). A sweep (the nightly one included) times
/// a subset and has no paired tests; were its rows newer, they would decide
/// the shapes on the band rule and drop the paired CIs.
const TUNE_RUNS: &str = " AND m.run_id IN (SELECT DISTINCT run_id FROM paired)";

fn latest_rows(con: &Connection, arch: Option<i64>, sel: &Selection, extra: &str) -> Result<Vec<Latest>> {
    let arch = arch.filter(|&a| a != 0); // Python: `if arch`
    let q = format!(
        "SELECT r.sm_arch, k.op, m.dtype, m.m, m.n, m.k, k.impl, k.kernel_id, m.run_id, \
                m.median_us, m.tflops, m.p10_us, m.p90_us, k.regs, k.smem_bytes \
         FROM measurements m JOIN runs r USING(run_id) JOIN kernels k USING(kernel_id) \
         WHERE (k.impl IN ({BASELINES}) OR (m.checked = 1 AND m.pass = 1)) {}{extra}{} ORDER BY r.started_at",
        sel.sql(),
        if arch.is_some() { " AND r.sm_arch = ?" } else { "" }
    );
    let mut stmt = con.prepare(&q)?;
    let map = |row: &rusqlite::Row| -> rusqlite::Result<Latest> {
        Ok(Latest {
            sm_arch: row.get(0)?,
            op: row.get(1)?,
            dtype: row.get(2)?,
            m: row.get(3)?,
            n: row.get(4)?,
            k: row.get(5)?,
            impl_: row.get(6)?,
            kernel_id: row.get(7)?,
            run_id: row.get(8)?,
            median_us: row.get(9)?,
            tflops: row.get(10)?,
            p10_us: row.get(11)?,
            p90_us: row.get(12)?,
            regs: row.get(13)?,
            smem_bytes: row.get(14)?,
        })
    };
    let rows: Vec<Latest> = match arch {
        Some(a) => stmt.query_map([a], map)?.collect::<rusqlite::Result<_>>()?,
        None => stmt.query_map([], map)?.collect::<rusqlite::Result<_>>()?,
    };
    let mut order: Vec<Latest> = Vec::new();
    let mut index: HashMap<KernelShapeKey, usize> = HashMap::new();
    for r in rows {
        let key = (r.sm_arch, r.op.clone(), r.dtype.clone(), r.m, r.n, r.k, r.impl_.clone());
        match index.get(&key) {
            Some(&i) => order[i] = r,
            None => {
                index.insert(key, order.len());
                order.push(r);
            }
        }
    }
    Ok(order)
}

/// The fastest config, plus the runner-up when the runner-up's median lies
/// inside the fastest one's p10-p90 band (a tie). Only the runner-up joins:
/// widening to every config inside the band let one noisy p90 (3.5x the
/// median at 512x4864x896) pull in configs up to 2x slower.
///
/// When aegis_tune ran the paired tie test for exactly these two (same run),
/// that decides instead: tied when the 95% bootstrap CI of the median ratio
/// runner-up / fastest contains 1.0 (tie_method "paired"). Otherwise the band
/// rule above applies (tie_method "band": v1 rows, sweep rows, --no-paired).
struct TieCall<'a> {
    set: Vec<&'a Latest>,
    ci: Option<(f64, f64)>,
    method: &'static str,
}

fn tie_set<'a>(ours: &[&'a Latest], paired: &PairedMap) -> TieCall<'a> {
    let mut sorted: Vec<&Latest> = ours.to_vec();
    sorted.sort_by(|a, b| a.median_us.total_cmp(&b.median_us).then_with(|| a.impl_.cmp(&b.impl_)));
    match sorted[..] {
        [] => TieCall { set: Vec::new(), ci: None, method: "none" },
        [w] => TieCall { set: vec![w], ci: None, method: "none" },
        [w, r, ..] => match paired_ci(paired, w, &r.impl_, &r.run_id) {
            // CI entirely below 1.0: in paired windows the runner-up is
            // faster, whatever the unpaired medians said; it wins outright.
            Some((lo, hi)) if hi < 1.0 => {
                TieCall { set: vec![r], ci: Some((1.0 / hi, 1.0 / lo)), method: "paired_swap" }
            }
            Some((lo, hi)) => {
                let set = if lo <= 1.0 && hi >= 1.0 { vec![w, r] } else { vec![w] };
                TieCall { set, ci: Some((lo, hi)), method: "paired" }
            }
            None => {
                let (lo, hi) = (w.p10_us.min(w.median_us), w.p90_us.max(w.median_us));
                let set = if r.median_us >= lo && r.median_us <= hi { vec![w, r] } else { vec![w] };
                TieCall { set, ci: None, method: "band" }
            }
        },
    }
}

/// Paired tie tests: (run_id, m, n, k, a_impl, b_impl) -> (95% CI lo, hi, median) of b / a.
type PairedMap = HashMap<(String, i64, i64, i64, String, String), (f64, f64, f64)>;

fn load_paired(con: &Connection) -> Result<PairedMap> {
    let mut stmt = con.prepare(&format!(
        "SELECT run_id, m, n, k, a_impl, b_impl, ratio_ci_lo, ratio_ci_hi, ratio_median FROM paired \
         WHERE protocol_version = 2 AND cache_mode = 'cold' AND allow_busy = 0 AND coalesce(disturbed, 0) = 0 \
           AND run_id NOT IN ({DIRTY_RUNS})"
    ))?;
    let rows = stmt.query_map([], |r| {
        Ok(((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?), (r.get(6)?, r.get(7)?, r.get(8)?)))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// (CI lo, CI hi, median) of `other` / `w` from a paired test in the run both rows came from.
fn paired_test(paired: &PairedMap, w: &Latest, other: &str, other_run: &str) -> Option<(f64, f64, f64)> {
    if w.run_id != other_run {
        return None;
    }
    let key = |a: &str, b: &str| (w.run_id.clone(), w.m, w.n, w.k, a.to_string(), b.to_string());
    if let Some(&t) = paired.get(&key(&w.impl_, other)) {
        return Some(t);
    }
    paired.get(&key(other, &w.impl_)).map(|&(lo, hi, med)| (1.0 / hi, 1.0 / lo, 1.0 / med))
}

/// CI of `other` / `w` from a paired test in the run both rows came from.
fn paired_ci(paired: &PairedMap, w: &Latest, other: &str, other_run: &str) -> Option<(f64, f64)> {
    paired_test(paired, w, other, other_run).map(|(lo, hi, _)| (lo, hi))
}

/// Neighboring shapes for the tie-break: same N and K with a different M (the
/// decode / prefill family), or every dimension within a factor of 2 (e.g.
/// adjacent square sizes).
fn neighbors(a: &ShapeKey, b: &ShapeKey) -> bool {
    if (a.0, &a.1, &a.2) != (b.0, &b.1, &b.2) || (a.3, a.4, a.5) == (b.3, b.4, b.5) {
        return false;
    }
    let within2 = |x: i64, y: i64| x.max(y) <= 2 * x.min(y);
    (a.4 == b.4 && a.5 == b.5) || (within2(a.3, b.3) && within2(a.4, b.4) && within2(a.5, b.5))
}

/// Impls a winner may be: the registry of the build that will run them
/// (`bench_gemm --list`). A config removed from the registry (e.g. the fixed-S
/// decode configs, retired for SM-count sizing on 2026-10-18) keeps its rows,
/// but they never decide a winner the build could not launch.
pub type Registry = std::collections::HashSet<String>;

pub fn registry_of(bench: &Path) -> Result<Registry> {
    Ok(crate::sweep::default_impls(bench)?.into_iter().collect())
}

/// Winner per (arch, op, dtype, shape). Clear win: the fastest median. Tie
/// (paired tie test CI of runner-up / fastest contains 1.0; without one,
/// the runner-up median inside the fastest one's p10-p90): of the two, the one
/// that wins or ties on the most neighboring shapes, then fewer registers,
/// then less smem, then lower median, then name.
pub fn cmd_winners(db: &Path, arch: Option<i64>, registry: Option<&Registry>) -> Result<()> {
    let con = connect(db)?;
    // Ranked on protocol v2, cold, idle-GPU rows only (CLAUDE.md Benchmark policy).
    let sel = Selection::ranking();
    let rows: Vec<Latest> = latest_rows(&con, arch, &sel, TUNE_RUNS)?
        .into_iter()
        .filter(|r| registry.is_none_or(|reg| is_baseline(&r.impl_) || r.impl_ == DISPATCH_IMPL || reg.contains(&r.impl_)))
        .collect();
    let paired = load_paired(&con)?;
    let mut groups: Vec<(ShapeKey, Vec<&Latest>)> = Vec::new();
    let mut index = HashMap::new();
    for r in &rows {
        let key = (r.sm_arch, r.op.clone(), r.dtype.clone(), r.m, r.n, r.k);
        let i = *index.entry(key.clone()).or_insert_with(|| {
            groups.push((key, Vec::new()));
            groups.len() - 1
        });
        groups[i].1.push(r);
    }
    // Win-or-tie set per shape, by impl name.
    let sets: Vec<(ShapeKey, Vec<String>)> = groups
        .iter()
        .map(|(key, rs)| {
            let ours: Vec<&Latest> = rs.iter().copied().filter(|r| !is_baseline(&r.impl_) && r.impl_ != DISPATCH_IMPL).collect();
            (key.clone(), tie_set(&ours, &paired).set.iter().map(|r| r.impl_.clone()).collect())
        })
        .collect();

    // Winners are derived data: drop the old ones of every arch recomputed here,
    // so shapes measured only under protocol v1 do not keep a stale winner.
    let mut archs: Vec<i64> = groups.iter().map(|(k, _)| k.0).collect();
    archs.dedup();
    for a in &archs {
        con.execute("DELETE FROM winners WHERE sm_arch = ?", [a])?;
    }
    let warm_of = |kernel_id: &str, (arch, dtype, m, n, k): (i64, &str, i64, i64, i64)| -> Result<Option<f64>> {
        let warm = Selection { cache: "warm".to_string(), ..sel.clone() };
        let q = format!(
            "SELECT m.median_us FROM measurements m JOIN runs r USING(run_id) \
             WHERE m.kernel_id = ? AND r.sm_arch = ? AND m.dtype = ? AND m.m = ? AND m.n = ? AND m.k = ? {}{TUNE_RUNS} \
             ORDER BY r.started_at DESC LIMIT 1",
            warm.sql()
        );
        Ok(con.query_row(&q, params![kernel_id, arch, dtype, m, n, k], |r| r.get(0)).ok())
    };

    let (mut n, mut tied_n) = (0, 0);
    for (key, rs) in &groups {
        let (arch, op, dtype, m, nn, k) = key;
        let cublas = rs.iter().find(|r| r.impl_ == "cublas");
        let lt_best = rs.iter().find(|r| r.impl_ == "cublaslt_best");
        let ours: Vec<&Latest> = rs.iter().copied().filter(|r| !is_baseline(&r.impl_) && r.impl_ != DISPATCH_IMPL).collect();
        let call = tie_set(&ours, &paired);
        let tie = &call.set;
        let Some(&fastest) = tie.first() else { continue };
        let (pick, reason) = if tie.len() == 1 {
            let why = if call.method == "paired_swap" {
                "fastest in the paired test (lower unpaired median: the runner-up)"
            } else {
                "fastest"
            };
            (fastest, why.to_string())
        } else {
            tied_n += 1;
            let support = |r: &Latest| {
                sets.iter().filter(|(other, set)| neighbors(key, other) && set.contains(&r.impl_)).count()
            };
            let mut ranked: Vec<(&Latest, usize)> = tie.iter().map(|&r| (r, support(r))).collect();
            ranked.sort_by(|(a, sa), (b, sb)| {
                sb.cmp(sa)
                    .then_with(|| a.regs.unwrap_or(i64::MAX).cmp(&b.regs.unwrap_or(i64::MAX)))
                    .then_with(|| a.smem_bytes.unwrap_or(i64::MAX).cmp(&b.smem_bytes.unwrap_or(i64::MAX)))
                    .then_with(|| a.median_us.total_cmp(&b.median_us))
                    .then_with(|| a.impl_.cmp(&b.impl_))
            });
            let (pick, s) = ranked[0];
            let (_, s2) = ranked[1];
            let by = if s != s2 {
                format!("neighbors ({s} of {} neighboring shapes)", sets.iter().filter(|(o, _)| neighbors(key, o)).count())
            } else if pick.regs != ranked[1].0.regs {
                format!("fewer registers ({}), neighbors tied at {s}", pick.regs.unwrap_or(-1))
            } else if pick.smem_bytes != ranked[1].0.smem_bytes {
                format!("less smem ({} B), neighbors and registers tied", pick.smem_bytes.unwrap_or(-1))
            } else {
                "lower median, all other criteria tied".to_string()
            };
            (pick, format!("tie-break: {by}"))
        };
        // The cublaslt_best CI of the dispatched config: from a paired test of
        // the pick itself, never of the other config in its tie (aegis_tune
        // pairs the runner-up too whenever it can become the pick). None when
        // no such test exists (rows tuned before that).
        let lt_test = lt_best.and_then(|lt| paired_test(&paired, pick, &lt.impl_, &lt.run_id));
        let lt_ci = lt_test.map(|(lo, hi, _)| (lo, hi));
        let lt_median = lt_test.map(|(_, _, med)| med);
        // The cuBLASLt algorithm cublaslt_best picked at this shape in the pick's run.
        let lt_algo: Option<String> = lt_best
            .map(|lt| {
                con.query_row(
                    "SELECT m.lt_algo FROM measurements m WHERE m.run_id = ?1 AND m.kernel_id = ?2 \
                     AND m.m = ?3 AND m.n = ?4 AND m.k = ?5 AND m.cache_mode = 'cold' AND m.lt_algo IS NOT NULL",
                    params![lt.run_id, lt.kernel_id, m, nn, k],
                    |r| r.get(0),
                )
                .optional()
            })
            .transpose()?
            .flatten();
        let pct = cublas.map(|c| c.median_us / pick.median_us * 100.0);
        let pct_lt = lt_best.map(|c| c.median_us / pick.median_us * 100.0);
        // Where aegis::matmul sends the shape (the dispatch floor on the paired CI), stored so
        // readers need not restate the rule or parse the header (schema v7).
        let dispatch = crate::tune::dispatch(lt_ci, pct_lt);
        let set_json = serde_json::to_string(&tie.iter().map(|r| r.impl_.as_str()).collect::<Vec<_>>())?;
        con.execute(
            "INSERT OR REPLACE INTO winners (sm_arch, op, dtype, shape_bucket, kernel_id, run_id, median_us, \
             pct_of_cublas, tied, tie_set, pick_reason, protocol_version, warm_median_us, pct_of_cublaslt_best, \
             tie_method, ratio_ci_lo, ratio_ci_hi, lt_ratio_ci_lo, lt_ratio_ci_hi, lt_algo, lt_ratio_median, dispatch) \
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            params![arch, op, dtype, shape_bucket(*m, *nn, *k), pick.kernel_id, pick.run_id, pick.median_us, pct,
                    (tie.len() > 1) as i64, set_json, reason, sel.protocol,
                    warm_of(&pick.kernel_id, (*arch, dtype.as_str(), *m, *nn, *k))?, pct_lt, call.method,
                    call.ci.map(|c| c.0), call.ci.map(|c| c.1), lt_ci.map(|c| c.0), lt_ci.map(|c| c.1),
                    lt_algo, lt_median, dispatch.as_str()],
        )?;
        n += 1;
    }
    println!(
        "wrote {n} winners ({tied_n} tied, resolved by the tie-break) from protocol v{} {} rows",
        sel.protocol, sel.cache
    );
    Ok(())
}

/// The report text (format unchanged since the Python perfdb).
/// The baseline a report's % cells divide by: cublaslt_best for protocol v2
/// rows (v1 runs never measured it), cuBLAS for v1.
pub fn default_baseline(sel: &Selection) -> &'static str {
    if sel.protocol >= 2 { "cublaslt_best" } else { "cublas" }
}

pub fn report_text(con: &Connection, arch: Option<i64>, sel: &Selection, baseline: &str) -> Result<String> {
    let rows = latest_per_kernel_shape(con, arch, sel)?;
    // table[(arch, m, n, k)][impl] = row; later rows overwrite (Python dict).
    let mut table: HashMap<(i64, i64, i64, i64), HashMap<String, &Latest>> = HashMap::new();
    for r in &rows {
        table.entry((r.sm_arch, r.m, r.n, r.k)).or_default().insert(r.impl_.clone(), r);
    }
    let mut impls: Vec<String> = rows.iter().filter(|r| r.impl_ != baseline).map(|r| r.impl_.clone()).collect();
    impls.sort();
    impls.dedup();
    let widths: Vec<usize> = impls.iter().map(|i| i.len().max(11)).collect();

    let mut out = String::new();
    let base_label = if baseline == "cublas" { "cuBLAS us" } else { "base us" };
    let mut hdr = format!("{:>5} {:>6} {:>6} {:>6} {:>10} {:>7} ", "arch", "M", "N", "K", base_label, "TFLOPS");
    hdr += &impls.iter().zip(&widths).map(|(i, w)| format!("{i:>w$}")).collect::<Vec<_>>().join(" ");
    writeln!(out, "{hdr}")?;
    writeln!(out, "{}", "-".repeat(hdr.chars().count()))?;
    let mut keys: Vec<_> = table.keys().copied().collect();
    keys.sort();
    for key in keys {
        let byimpl = &table[&key];
        let cb = byimpl.get(baseline);
        let cells: Vec<String> = impls
            .iter()
            .zip(&widths)
            .map(|(i, &w)| match (byimpl.get(i), cb) {
                (None, _) => format!("{:>w$}", "-"),
                (Some(r), Some(cb)) => format!("{:>w$.1}%", cb.median_us / r.median_us * 100.0, w = w - 1),
                (Some(r), None) => format!("{:>w$.1}us", r.median_us, w = w - 2),
            })
            .collect();
        let cbs = match cb {
            Some(cb) => format!("{:>10.1} {:>7.2}", cb.median_us, cb.tflops),
            None => format!("{:>10} {:>7}", "-", "-"),
        };
        let (a, m, n, k) = key;
        writeln!(out, "{a:>5} {m:>6} {n:>6} {k:>6} {cbs} {}", cells.join(" "))?;
    }
    if baseline == "cublas" {
        writeln!(out, "\nCells are % of cuBLAS speed (100% = parity, >100% = faster than cuBLAS).")?;
    } else {
        writeln!(out, "\nCells are % of {baseline} speed (100% = parity, >100% = faster); base us is its median.")?;
    }
    let cache = if sel.protocol >= 2 { format!(", cache_mode={}", sel.cache) } else { String::new() };
    let busy = if sel.include_busy { "including" } else { "excluding" };
    let dirty = if sel.include_dirty { "including" } else { "excluding" };
    writeln!(
        out,
        "Rows: timing protocol v{}{cache}, {busy} --allow-busy and disturbed rows, {dirty} dirty-build (--allow-dirty) runs.",
        sel.protocol
    )?;
    Ok(out)
}

/// Roof metrics per shape for the winner and the baselines, computed now from
/// include/aegis/arch/sm_<arch>.json (never stored): AI, TFLOPS and % of the
/// mma peak, GB/s and % of DRAM bandwidth, bound, % of the roofline.
pub fn roof_text(con: &Connection, arch: Option<i64>, sel: &Selection) -> Result<String> {
    let rows = latest_per_kernel_shape(con, arch, sel)?;
    let mut archs: Vec<i64> = rows.iter().map(|r| r.sm_arch).collect();
    archs.sort();
    archs.dedup();
    let mut out = String::new();
    for a in archs {
        let path = crate::params::arch_file(a);
        let p = match crate::params::load(&path) {
            Ok(p) => p,
            Err(e) => {
                writeln!(out, "\nsm_{a}: no roof metrics ({e:#}); run aegis-perf emit-arch --arch {a}")?;
                continue;
            }
        };
        writeln!(
            out,
            "\nRoofline, sm_{a}: read {:.1} GB/s (roof for M <= {}), copy {:.1} GB/s (roof otherwise), {:.1} TFLOPS; \
             ridge {:.0} / {:.0} FLOP/B (from {}, computed now)",
            p.read_bw_gbps,
            crate::params::DECODE_MAX_M,
            p.copy_bw_gbps,
            p.peak_tflops,
            p.ridge(crate::params::BwKind::Read),
            p.ridge(crate::params::BwKind::Copy),
            p.path.display()
        )?;
        let hdr = format!(
            "{:>6} {:>6} {:>6}  {:<30} {:>10} {:>7} {:>8} {:>6} {:>8} {:>5} {:>6} {:>6} {:>6}",
            "M", "N", "K", "impl", "median us", "AI", "TFLOPS", "%peak", "GB/s", "roof", "%bw", "bound", "%roof"
        );
        writeln!(out, "{hdr}\n{}", "-".repeat(hdr.len()))?;
        let winners: HashMap<String, String> = con
            .prepare(
                "SELECT w.shape_bucket, k.impl FROM winners w JOIN kernels k USING(kernel_id) \
                 WHERE w.sm_arch = ? AND w.op = 'gemm'",
            )?
            .query_map([a], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let mut shapes: Vec<(i64, i64, i64)> = rows.iter().filter(|r| r.sm_arch == a).map(|r| (r.m, r.n, r.k)).collect();
        shapes.sort();
        shapes.dedup();
        for (m, n, k) in shapes {
            let winner = winners.get(&shape_bucket(m, n, k)).cloned();
            for impl_ in winner.iter().map(String::as_str).chain(["cublaslt_best", "cublas"]) {
                let Some(r) = rows.iter().find(|r| r.sm_arch == a && (r.m, r.n, r.k) == (m, n, k) && r.impl_ == impl_)
                else {
                    continue;
                };
                let x = p.roof(m as f64, n as f64, k as f64, r.median_us);
                writeln!(
                    out,
                    "{m:>6} {n:>6} {k:>6}  {impl_:<30} {:>10.2} {:>7.1} {:>8.2} {:>5.1}% {:>8.1} {:>5} {:>5.1}% {:>6} {:>5.1}%",
                    r.median_us,
                    x.ai,
                    x.tflops,
                    x.pct_peak,
                    x.gbps,
                    x.bw.name(),
                    x.pct_bw,
                    if x.memory_bound { "mem" } else { "comp" },
                    x.pct_roof
                )?;
            }
        }
    }
    writeln!(
        out,
        "\nCompulsory traffic (A, B read once, C written once). First row per shape: the winner.\n\
         roof: the memory roof used, read-only stream bandwidth for GEMV/decode (M <= 16), copy bandwidth otherwise."
    )?;
    Ok(out)
}

/// The winners, with the number of record first (bars.rs): the kernel's paired
/// % of cublaslt_best and its 95% CI, the dispatch, then the unpaired median,
/// labelled, and flagged where it disagrees with the CI.
pub fn winners_text(con: &Connection, arch: Option<i64>) -> Result<String> {
    let mut out = String::new();
    let mut ws = crate::view::winners(con, arch)?;
    if ws.is_empty() {
        return Ok(out);
    }
    writeln!(out, "Winners: kernel % of cublaslt_best of record = paired median (95% CI); unpaired medians are secondary.")?;
    let hdr = format!(
        "{:>5} {:>6} {:>6} {:>6}  {:<32} {:<26} {:<9} {}",
        "arch", "M", "N", "K", "winner", "kernel (paired)", "dispatch", "unpaired median"
    );
    writeln!(out, "{hdr}\n{}", "-".repeat(hdr.len()))?;
    ws.sort_by_key(|w| (w.sm_arch, w.m, w.n, w.k));
    for w in &ws {
        let paired = w.kernel_pct();
        let dispatch = w.dispatch.unwrap_or_else(|| crate::tune::dispatch(w.lt_ratio_ci, w.pct_of_cublaslt_best));
        let unpaired = match (w.pct_of_cublaslt_best, &paired) {
            (Some(u), Some(p)) if crate::bars::median_disagrees(u, p) => {
                format!("{u:.1}% (DISAGREES with the paired CI by > {} points)", crate::bars::MEDIAN_CI_DISAGREE_PTS)
            }
            (Some(u), _) => format!("{u:.1}%"),
            (None, _) => "-".to_string(),
        };
        writeln!(
            out,
            "{:>5} {:>6} {:>6} {:>6}  {:<32} {:<26} {:<9} {unpaired}",
            w.sm_arch,
            w.m,
            w.n,
            w.k,
            w.kernel.impl_,
            paired.map_or("no paired test".to_string(), |p| p.text()),
            dispatch.as_str()
        )?;
    }
    writeln!(out)?;
    Ok(out)
}

pub fn cmd_report(db: &Path, arch: Option<i64>, sel: &Selection, baseline: Option<&str>) -> Result<()> {
    let con = connect(db)?;
    print!("{}", winners_text(&con, arch)?);
    println!("Every impl, unpaired medians (secondary; the paired figures above are of record):");
    print!("{}", report_text(&con, arch, sel, baseline.unwrap_or(default_baseline(sel)))?);
    print!("{}", roof_text(&con, arch, sel)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema creates every table and is idempotent.
    #[test]
    fn schema_creates_tables() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(SCHEMA).unwrap();
        con.execute_batch(SCHEMA).unwrap();
        for t in ["runs", "kernels", "measurements", "winners", "microbench"] {
            let n: i64 = con
                .query_row("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", [t], |r| r.get(0))
                .unwrap();
            assert_eq!(n, 1, "missing table {t}");
        }
    }

    /// A pre-v2 measurements table gains the protocol columns; its rows read as
    /// protocol v1, from an idle GPU.
    #[test]
    fn migrate_marks_old_rows_v1() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(
            "CREATE TABLE measurements (run_id TEXT NOT NULL, kernel_id TEXT NOT NULL, dtype TEXT NOT NULL, \
             m INTEGER, n INTEGER, k INTEGER, median_us REAL, p10_us REAL, p90_us REAL, tflops REAL, gbps REAL, \
             reps INTEGER, checked INTEGER, pass INTEGER, max_abs_err REAL, rel_err REAL, power_w REAL);
             INSERT INTO measurements (run_id, kernel_id, dtype) VALUES ('r', 'k', 'f16f32f16');",
        )
        .unwrap();
        con.execute_batch(SCHEMA).unwrap();
        migrate(&con).unwrap();
        let (p, busy, cache): (i64, i64, Option<String>) = con
            .query_row("SELECT protocol_version, allow_busy, cache_mode FROM measurements", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!((p, busy, cache), (1, 0, None));
    }

    /// A clean build (40 hex digits).
    const CLEAN: &str = "0123456789abcdef0123456789abcdef01234567";

    fn clean() -> BuildId {
        BuildId::new(CLEAN, false).unwrap()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("aegis-perf-{name}-{}", std::process::id()))
    }

    fn row(idle: f64, busy: bool) -> String {
        format!(
            r#"{{"op":"gemm","impl":"x","config":{{}},"shape":{{"M":8,"N":8,"K":8}},"dtype":"f16f32f16","median_us":1.0,"p10_us":1.0,"p90_us":1.0,"tflops":1.0,"gbps":1.0,"reps":1,"checked":true,"pass":true,"max_abs_err":0.0,"rel_err":0.0,"sm_arch":121,"protocol_version":2,"cache_mode":"cold","idle_util_pct":{idle},"idle_source":"nvml","allow_busy":{busy},"build_sha":"{CLEAN}","build_dirty":false}}"#
        )
    }

    /// v5: a row's device profile label is stored on measurements, paired and
    /// microbench rows; rows without one (older builds) store NULL.
    #[test]
    fn ingest_stores_the_profile_label() {
        let db = tmp("profile.db");
        let _ = std::fs::remove_file(&db);
        let f = tmp("profile.jsonl");
        let labelled = row(0.1, false).replacen("\"op\":\"gemm\"", "\"op\":\"gemm\",\"profile\":\"unmeasured\"", 1);
        let paired = r#"{"kind":"paired","op":"gemm","dtype":"f16f32f16","shape":{"M":8,"N":8,"K":8},"a":"x","b":"y","role":"runner_up","pairs":30,"ratio_median":1.0,"ratio_ci_lo":0.99,"ratio_ci_hi":1.01,"tied":true,"protocol_version":2,"cache_mode":"cold","profile":"measured","build_sha":"0123456789abcdef0123456789abcdef01234567","build_dirty":false}"#;
        std::fs::write(&f, format!("{labelled}\n{}\n{paired}\n", row(0.1, false))).unwrap();
        cmd_ingest(&db, &f, None, Some("p".into()), None, IngestOpts::new(RunKind::Sweep)).unwrap();
        std::fs::remove_file(&f).ok();
        let mb = tmp("profile-mb.jsonl");
        std::fs::write(&mb, r#"{"probe":"p","param":{},"value":1.0,"unit":"ns","sm_arch":121,"profile":"unmeasured"}"#.to_string() + "\n").unwrap();
        let idle = RunIdle { pct: Some(0.1), source: Some("nvml".into()), allow_busy: false };
        cmd_ingest_microbench(&db, &mb, None, Some("mb".into()), &idle, Some(&clean()), false).unwrap();
        std::fs::remove_file(&mb).ok();
        let con = connect(&db).unwrap();
        let labels: Vec<Option<String>> = con
            .prepare("SELECT profile FROM measurements WHERE run_id = 'p' ORDER BY rowid")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(labels, vec![Some("unmeasured".to_string()), None]);
        let p: Option<String> = con.query_row("SELECT profile FROM paired WHERE run_id = 'p'", [], |r| r.get(0)).unwrap();
        assert_eq!(p.as_deref(), Some("measured"));
        let m: Option<String> = con.query_row("SELECT profile FROM microbench WHERE run_id = 'mb'", [], |r| r.get(0)).unwrap();
        assert_eq!(m.as_deref(), Some("unmeasured"));
        std::fs::remove_file(&db).ok();
    }

    /// A disturbed microbench run replaces nothing: the clean run it would have
    /// superseded stays valid.
    #[test]
    fn disturbed_microbench_run_replaces_nothing() {
        let db = tmp("disturbed-mb.db");
        let _ = std::fs::remove_file(&db);
        let idle = RunIdle { pct: Some(0.1), source: Some("nvml".into()), allow_busy: false };
        let line = r#"{"probe":"p","param":{"stat":"x"},"value":1.0,"unit":"s","sm_arch":121}"#.to_string() + "\n";
        for id in ["clean", "noisy"] {
            let f = tmp(&format!("disturbed-mb-{id}.jsonl"));
            std::fs::write(&f, &line).unwrap();
            cmd_ingest_microbench(&db, &f, None, Some(id.into()), &idle, Some(&clean()), false).unwrap();
            std::fs::remove_file(&f).ok();
        }
        let con = connect(&db).unwrap();
        let status = |id: &str| -> String {
            con.query_row("SELECT status FROM microbench WHERE run_id = ?1", [id], |r| r.get(0)).unwrap()
        };
        assert_eq!(status("clean"), "replaced", "ingest replaces first");
        let report = crate::watch::WatchReport { disturbed: Some(true), ..Default::default() };
        mark_disturbed(&con, "noisy", &report).unwrap();
        assert_eq!(status("clean"), "valid", "the disturbed run gives the clean one back");
        assert_eq!(current_run(&con, 121, "p").unwrap().as_deref(), Some("clean"));
        std::fs::remove_file(&db).ok();
    }

    /// v6: telemetry fields on measurement rows (opt-in; NULL without).
    #[test]
    fn ingest_stores_telemetry() {
        let db = tmp("telemetry.db");
        let _ = std::fs::remove_file(&db);
        let f = tmp("telemetry.jsonl");
        let tel = row(0.1, false).replacen(
            "\"op\":\"gemm\"",
            "\"op\":\"gemm\",\"max_temp_c\":61.0,\"throttle_reasons\":\"sw_power_cap\",\"clock_event_reasons\":\"sw_power_cap|applications_clocks_setting\",\"sm_clock_min_mhz\":2400,\"sm_clock_max_mhz\":2520,\"power_w\":71.5",
            1,
        );
        std::fs::write(&f, format!("{tel}\n{}\n", row(0.1, false))).unwrap();
        cmd_ingest(&db, &f, None, Some("t".into()), None, IngestOpts::new(RunKind::Sweep)).unwrap();
        std::fs::remove_file(&f).ok();
        let con = connect(&db).unwrap();
        type Row = (Option<f64>, Option<String>, Option<String>, Option<i64>, Option<i64>, Option<f64>);
        let rows: Vec<Row> = con
            .prepare("SELECT max_temp_c, throttle_reasons, clock_event_reasons, sm_clock_min_mhz, sm_clock_max_mhz, power_w \
                      FROM measurements WHERE run_id = 't' ORDER BY rowid")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(rows[0], (Some(61.0), Some("sw_power_cap".into()),
                             Some("sw_power_cap|applications_clocks_setting".into()), Some(2400), Some(2520), Some(71.5)));
        assert_eq!(rows[1], (None, None, None, None, None, None), "no telemetry: NULL");
        assert_eq!(schema_version(&con).unwrap(), Some(SCHEMA_VERSION));
        std::fs::remove_file(&db).ok();
    }

    /// Every run row stores the idle guard's reading: an idle run, and a busy
    /// run that went ahead with --allow-busy (marked).
    #[test]
    fn ingest_stores_the_idle_reading_on_the_run() {
        let db = tmp("idle.db");
        let _ = std::fs::remove_file(&db);
        for (id, idle, busy) in [("idle_run", 0.4, false), ("busy_run", 2.9, true)] {
            let f = tmp(&format!("{id}.jsonl"));
            std::fs::write(&f, row(idle, busy) + "\n").unwrap();
            cmd_ingest(&db, &f, None, Some(id.into()), Some(40.0), IngestOpts::new(RunKind::Sweep)).unwrap();
            std::fs::remove_file(&f).ok();
        }
        let mb = tmp("mb.jsonl");
        std::fs::write(&mb, r#"{"probe":"p","param":{},"value":1.0,"unit":"ns","sm_arch":121}"#.to_string() + "\n").unwrap();
        let idle = RunIdle { pct: Some(0.7), source: Some("nvml".into()), allow_busy: false };
        cmd_ingest_microbench(&db, &mb, None, Some("mb_run".into()), &idle, Some(&clean()), false).unwrap();
        std::fs::remove_file(&mb).ok();
        let con = connect(&db).unwrap();
        let get = |id: &str| -> (Option<f64>, Option<String>, i64) {
            con.query_row("SELECT idle_util_pct, idle_source, allow_busy FROM runs WHERE run_id = ?", [id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap()
        };
        assert_eq!(get("idle_run"), (Some(0.4), Some("nvml".into()), 0));
        assert_eq!(get("busy_run"), (Some(2.9), Some("nvml".into()), 1));
        assert_eq!(get("mb_run"), (Some(0.7), Some("nvml".into()), 0));
        drop(con);
        std::fs::remove_file(&db).ok();
    }

    /// A disturbed run: the run and every row of it are marked, its rows drop
    /// out of ranking (Selection), and a clean run's rows get 0.
    #[test]
    fn mark_disturbed_marks_every_row_and_hides_it_from_ranking() {
        let db = tmp("disturbed.db");
        let _ = std::fs::remove_file(&db);
        let f = tmp("disturbed.jsonl");
        std::fs::write(&f, row(0.0, false) + "\n").unwrap();
        let bad = cmd_ingest(&db, &f, None, Some("bad".into()), None, IngestOpts::new(RunKind::Sweep)).unwrap();
        let good = cmd_ingest(&db, &f, None, Some("good".into()), None, IngestOpts::new(RunKind::Sweep)).unwrap();
        std::fs::remove_file(&f).ok();
        let con = connect(&db).unwrap();
        let w = |d| crate::watch::WatchReport {
            disturbed: Some(d),
            windows: 5,
            windows_over: if d { 2 } else { 0 },
            max_other_pct: if d { 30.0 } else { 0.5 },
            limit_pct: 2.0,
            worst: "firefox (pid 7) 30.0% at T".into(),
            error: String::new(),
            ..Default::default()
        };
        mark_disturbed(&con, &bad, &w(true)).unwrap();
        mark_disturbed(&con, &good, &w(false)).unwrap();
        let rows: Vec<(String, Option<i64>)> = con
            .prepare("SELECT run_id, disturbed FROM measurements ORDER BY run_id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(rows, vec![("bad".into(), Some(1)), ("good".into(), Some(0))]);
        let detail: String =
            con.query_row("SELECT disturbed_detail FROM runs WHERE run_id = 'bad'", [], |r| r.get(0)).unwrap();
        assert!(detail.contains("2 of 5") && detail.contains("firefox"), "{detail}");
        let ranked: Vec<String> = con
            .prepare(&format!("SELECT m.run_id FROM measurements m WHERE 1 = 1{}", Selection::ranking().sql()))
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(ranked, vec!["good".to_string()]);
        assert!(mark_disturbed(&con, "nope", &w(true)).is_err());
        drop(con);
        std::fs::remove_file(&db).ok();
    }

    /// The pre-status DB: superseded = 1 becomes replaced (with replaced_by),
    /// superseded = 0 valid; the old columns are gone.
    #[test]
    fn migrate_superseded_to_status() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, sm_arch INTEGER NOT NULL,
               gpu_name TEXT, cuda_ver INTEGER, driver_ver INTEGER, git_sha TEXT, power_mode TEXT, clocks_locked INTEGER,
               temp_start_c REAL, measured_bw_gbps REAL, started_at TEXT NOT NULL);
             CREATE TABLE microbench (run_id TEXT NOT NULL REFERENCES runs(run_id), probe TEXT NOT NULL,
               param_json TEXT NOT NULL, value REAL, unit TEXT NOT NULL,
               superseded INTEGER NOT NULL DEFAULT 0, superseded_by TEXT);
             INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES ('old', 'd', 121, '1'), ('new', 'd', 121, '2');
             INSERT INTO microbench VALUES ('old', 'p', '{}', 1.0, 'ns', 1, 'new'), ('new', 'p', '{}', 2.0, 'ns', 0, NULL);",
        )
        .unwrap();
        con.execute_batch(SCHEMA).unwrap();
        migrate(&con).unwrap();
        let rows: Vec<(String, String, Option<String>)> = con
            .prepare("SELECT run_id, status, replaced_by FROM microbench ORDER BY run_id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            rows,
            vec![("new".into(), "valid".into(), None), ("old".into(), "replaced".into(), Some("new".into()))]
        );
        let has_old: bool = con
            .prepare("SELECT 1 FROM pragma_table_info('microbench') WHERE name LIKE 'superseded%'")
            .unwrap()
            .exists([])
            .unwrap();
        assert!(!has_old, "superseded columns dropped");
        migrate(&con).unwrap(); // idempotent
    }

    /// A DB without schema_version is v1: readers refuse it with the version
    /// message, connect migrates it to SCHEMA_VERSION; a newer DB is refused
    /// by readers and writers alike and left untouched.
    #[test]
    fn schema_version_checked_by_every_reader() {
        let p = tmp("schema-version");
        let _ = std::fs::remove_file(&p);
        Connection::open(&p)
            .unwrap()
            .execute_batch(
                "CREATE TABLE runs (run_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, sm_arch INTEGER NOT NULL,
                   started_at TEXT NOT NULL);
                 CREATE TABLE microbench (run_id TEXT NOT NULL, probe TEXT NOT NULL, param_json TEXT NOT NULL,
                   value REAL, unit TEXT NOT NULL, superseded INTEGER NOT NULL DEFAULT 0, superseded_by TEXT);",
            )
            .unwrap();
        let err = connect_ro(&p).unwrap_err().to_string();
        assert!(err.contains(&format!("DB schema v1, binary expects v{SCHEMA_VERSION}")), "{err}");

        drop(connect(&p).unwrap());
        let con = connect_ro(&p).unwrap();
        assert_eq!(schema_version(&con).unwrap(), Some(SCHEMA_VERSION));
        drop(con);

        let newer = SCHEMA_VERSION + 1;
        Connection::open(&p).unwrap().execute("UPDATE schema_version SET version = ?", [newer]).unwrap();
        for err in [connect_ro(&p).unwrap_err().to_string(), connect(&p).unwrap_err().to_string()] {
            assert!(err.contains(&format!("DB schema v{newer}, binary expects v{SCHEMA_VERSION}")), "{err}");
        }
        let con = Connection::open(&p).unwrap();
        assert_eq!(schema_version(&con).unwrap(), Some(newer), "a newer DB is not rewritten");
        std::fs::remove_file(&p).unwrap();
    }

    /// invalidated needs a reason; status takes only the three values.
    #[test]
    fn status_constraints() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(SCHEMA).unwrap();
        migrate(&con).unwrap();
        con.execute_batch("INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES ('r', 'd', 121, '1');").unwrap();
        let ins = |status: &str, reason: Option<&str>| {
            con.execute(
                "INSERT INTO microbench (run_id, probe, param_json, value, unit, status, status_reason) \
                 VALUES ('r', 'p', '{}', 1.0, 'ns', ?1, ?2)",
                params![status, reason],
            )
        };
        assert!(ins("invalidated", None).is_err());
        assert!(ins("invalidated", Some("  ")).is_err());
        assert!(ins("invalidated", Some("clock throttled")).is_ok());
        assert!(ins("superseded", None).is_err());
        assert!(ins("replaced", None).is_ok());
    }

    /// Re-measuring replaces; invalidating the current run brings back the run
    /// it had replaced; the hardware model's run follows (current_run).
    #[test]
    fn replace_invalidate_and_current_run() {
        let db = tmp("status.db");
        let _ = std::fs::remove_file(&db);
        let idle = RunIdle::default();
        for id in ["a", "b"] {
            let f = tmp(&format!("{id}-mb.jsonl"));
            std::fs::write(&f, r#"{"probe":"p","param":{},"value":1.0,"unit":"ns","sm_arch":121}"#.to_string() + "\n")
                .unwrap();
            cmd_ingest_microbench(&db, &f, None, Some(id.into()), &idle, Some(&clean()), false).unwrap();
            std::fs::remove_file(&f).ok();
            std::thread::sleep(std::time::Duration::from_millis(5)); // distinct started_at
        }
        let mut con = connect(&db).unwrap();
        let status = |con: &Connection, id: &str| -> (String, Option<String>) {
            con.query_row("SELECT status, replaced_by FROM microbench WHERE run_id = ?", [id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap()
        };
        assert_eq!(status(&con, "a"), ("replaced".into(), Some("b".into())));
        assert_eq!(current_run(&con, 121, "p").unwrap().as_deref(), Some("b"));

        assert!(invalidate_microbench(&mut con, "b", None, " ").is_err(), "reason required");
        assert_eq!(invalidate_microbench(&mut con, "b", Some("p"), "throttled run").unwrap(), 1);
        assert_eq!(status(&con, "b").0, "invalidated");
        assert_eq!(status(&con, "a"), ("valid".into(), None));
        assert_eq!(current_run(&con, 121, "p").unwrap().as_deref(), Some("a"));

        // Only replaced rows left: the newest replaced run is the fallback.
        con.execute("UPDATE microbench SET status = 'replaced', replaced_by = 'b' WHERE run_id = 'a'", []).unwrap();
        assert_eq!(current_run(&con, 121, "p").unwrap().as_deref(), Some("a"));
        drop(con);
        std::fs::remove_file(&db).ok();
    }

    /// Latency rows keep cycles plus the clock they were measured at and the
    /// lock state; the run records whether every latency row was locked; the
    /// viewer derives ns at read time.
    #[test]
    fn ingest_microbench_stores_clock_and_lock() {
        let db = tmp("clk.db");
        let _ = std::fs::remove_file(&db);
        let f = tmp("clk.jsonl");
        std::fs::write(
            &f,
            concat!(
                r#"{"probe":"ptr_chase_latency","param":{"stat":"median"},"value":335.0,"unit":"cycles","sm_arch":121,"sm_clock_mhz":2407.0,"clocks_locked":false,"clock_lock":"lock requested at 2418 MHz via NVML: Insufficient Permissions"}"#,
                "\n",
                r#"{"probe":"dram_bandwidth","param":{},"value":236.0,"unit":"GB/s","sm_arch":121}"#,
                "\n"
            ),
        )
        .unwrap();
        cmd_ingest_microbench(&db, &f, None, Some("c".into()), &RunIdle::default(), Some(&clean()), false).unwrap();
        std::fs::remove_file(&f).ok();
        let con = connect_ro(&db).unwrap();
        let rows = crate::view::microbench(&con, 121).unwrap();
        let lat = rows.iter().find(|r| r.probe == "ptr_chase_latency").unwrap();
        assert_eq!((lat.sm_clock_mhz, lat.clocks_locked), (Some(2407.0), Some(false)));
        assert!(lat.clock_lock.as_deref().unwrap().contains("Insufficient Permissions"));
        assert!((lat.ns().unwrap() - 335.0 / 2407.0 * 1e3).abs() < 1e-9);
        let bw = rows.iter().find(|r| r.probe == "dram_bandwidth").unwrap();
        assert_eq!((bw.sm_clock_mhz, bw.clocks_locked, bw.ns()), (None, None, None));
        let run_locked: Option<i64> =
            con.query_row("SELECT clocks_locked FROM runs WHERE run_id = 'c'", [], |r| r.get(0)).unwrap();
        assert_eq!(run_locked, Some(0));
        drop(con);
        std::fs::remove_file(&db).ok();
    }

    /// Runs from before the run-level columns get them from their rows.
    #[test]
    fn migrate_backfills_run_idle_from_measurements() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(SCHEMA).unwrap();
        // An old DB: runs without the idle columns.
        con.execute_batch(
            "DROP TABLE runs; CREATE TABLE runs (run_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, sm_arch INTEGER NOT NULL,
               gpu_name TEXT, cuda_ver INTEGER, driver_ver INTEGER, git_sha TEXT, power_mode TEXT, clocks_locked INTEGER,
               temp_start_c REAL, measured_bw_gbps REAL, started_at TEXT NOT NULL);
             INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES ('a', 'd', 121, 't'), ('b', 'd', 121, 't');",
        )
        .unwrap();
        con.execute_batch(
            "INSERT INTO kernels (kernel_id, op, impl, config_json) VALUES ('k', 'gemm', 'x', '{}');
             INSERT INTO measurements (run_id, kernel_id, dtype, idle_util_pct, allow_busy) VALUES ('a', 'k', 'f', 3.5, 1);",
        )
        .unwrap();
        migrate(&con).unwrap();
        let a: (Option<f64>, i64) = con.query_row("SELECT idle_util_pct, allow_busy FROM runs WHERE run_id = 'a'", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        let b: (Option<f64>, i64) = con.query_row("SELECT idle_util_pct, allow_busy FROM runs WHERE run_id = 'b'", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(a, (Some(3.5), 1));
        assert_eq!(b, (None, 0), "no reading recorded: NULL, not an invented idle");
    }

    /// Winners rank on tune runs (paired tie tests) only: a newer sweep that
    /// timed one config, faster there, changes neither the pick nor its CI.
    #[test]
    fn winners_ignore_sweep_runs() {
        let p = std::env::temp_dir().join(format!("aegis-perf-tuneonly-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let con = connect(&p).unwrap();
        con.execute_batch(
            "INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES
               ('tune', 'dev', 121, '2026-01-01'), ('sweep', 'dev', 121, '2026-01-02');
             INSERT INTO kernels (kernel_id, op, impl, config_json, regs, smem_bytes) VALUES
               ('ka', 'gemm', 'a', '{}', 64, 0), ('kb', 'gemm', 'b', '{}', 64, 0),
               ('kl', 'gemm', 'cublaslt_best', '{}', -1, 0);
             INSERT INTO measurements (run_id, kernel_id, dtype, m, n, k, median_us, p10_us, p90_us, tflops,
                                       checked, pass, protocol_version, cache_mode, allow_busy) VALUES
               ('tune', 'ka', 'f16f32f16', 64, 64, 64, 10.0, 9.9, 10.1, 1, 1, 1, 2, 'cold', 0),
               ('tune', 'kb', 'f16f32f16', 64, 64, 64, 12.0, 11.9, 12.1, 1, 1, 1, 2, 'cold', 0),
               ('tune', 'kl', 'f16f32f16', 64, 64, 64, 9.0, 8.9, 9.1, 1, 0, 0, 2, 'cold', 0),
               ('sweep', 'kb', 'f16f32f16', 64, 64, 64, 8.0, 7.9, 8.1, 1, 1, 1, 2, 'cold', 0),
               ('sweep', 'kl', 'f16f32f16', 64, 64, 64, 9.0, 8.9, 9.1, 1, 0, 0, 2, 'cold', 0);
             INSERT INTO paired (run_id, op, dtype, m, n, k, a_impl, b_impl, role, pairs, ratio_median,
                                 ratio_ci_lo, ratio_ci_hi, tied, protocol_version, cache_mode) VALUES
               ('tune', 'gemm', 'f16f32f16', 64, 64, 64, 'a', 'b', 'runner_up', 30, 1.2, 1.19, 1.21, 0, 2, 'cold'),
               ('tune', 'gemm', 'f16f32f16', 64, 64, 64, 'a', 'cublaslt_best', 'cublaslt_best', 30, 0.9, 0.89, 0.91, 0, 2, 'cold');",
        )
        .unwrap();
        drop(con);
        cmd_winners(&p, Some(121), None).unwrap();
        let con = connect(&p).unwrap();
        let w: (String, String, Option<f64>) = con
            .query_row(
                "SELECT k.impl, w.tie_method, w.lt_ratio_ci_hi FROM winners w JOIN kernels k USING(kernel_id)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(w, ("a".to_string(), "paired".to_string(), Some(0.91)));
        // v7: the paired median of record and the dispatch are stored with the winner.
        let (med, disp): (Option<f64>, Option<String>) = con
            .query_row("SELECT lt_ratio_median, dispatch FROM winners", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!((med, disp.as_deref()), (Some(0.9), Some("aegis")));
        let v = &crate::view::winners(&con, Some(121)).unwrap()[0];
        assert_eq!(v.kernel_pct(), Some(crate::bars::PairedPct { median: 0.9, ci_lo: 0.89, ci_hi: 0.91 }));
        assert_eq!(v.dispatch, Some(crate::tune::Dispatch::Aegis));
        drop(con);
        std::fs::remove_file(&p).unwrap();
    }

    /// A config retired from the registry never wins, however fast its old rows:
    /// with registry {b}, b wins although a's (older run) median is lower.
    #[test]
    fn winners_only_from_the_registry() {
        let p = std::env::temp_dir().join(format!("aegis-perf-registry-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let con = connect(&p).unwrap();
        con.execute_batch(
            "INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES
               ('old', 'dev', 121, '2026-01-01'), ('new', 'dev', 121, '2026-01-02');
             INSERT INTO kernels (kernel_id, op, impl, config_json, regs, smem_bytes) VALUES
               ('ka', 'gemm', 'a', '{}', 64, 0), ('kb', 'gemm', 'b', '{}', 64, 0);
             INSERT INTO measurements (run_id, kernel_id, dtype, m, n, k, median_us, p10_us, p90_us, tflops,
                                       checked, pass, protocol_version, cache_mode, allow_busy) VALUES
               ('old', 'ka', 'f16f32f16', 64, 64, 64, 8.0, 7.9, 8.1, 1, 1, 1, 2, 'cold', 0),
               ('new', 'kb', 'f16f32f16', 64, 64, 64, 10.0, 9.9, 10.1, 1, 1, 1, 2, 'cold', 0);
             INSERT INTO paired (run_id, op, dtype, m, n, k, a_impl, b_impl, role, pairs, ratio_median,
                                 ratio_ci_lo, ratio_ci_hi, tied, protocol_version, cache_mode) VALUES
               ('old', 'gemm', 'f16f32f16', 64, 64, 64, 'a', 'cublaslt_best', 'cublaslt_best', 30, 1.0, 0.99, 1.01, 1, 2, 'cold'),
               ('new', 'gemm', 'f16f32f16', 64, 64, 64, 'b', 'cublaslt_best', 'cublaslt_best', 30, 1.0, 0.99, 1.01, 1, 2, 'cold');",
        )
        .unwrap();
        drop(con);
        let pick = |reg: Option<&Registry>| {
            cmd_winners(&p, Some(121), reg).unwrap();
            connect(&p)
                .unwrap()
                .query_row("SELECT k.impl FROM winners w JOIN kernels k USING(kernel_id)", [], |r| r.get::<_, String>(0))
                .unwrap()
        };
        assert_eq!(pick(None), "a", "without a registry every impl in the DB competes");
        let reg: Registry = ["b".to_string()].into_iter().collect();
        assert_eq!(pick(Some(&reg)), "b", "a is not registered: it cannot win");
        std::fs::remove_file(&p).unwrap();
    }

    /// A tie where the tie-break dispatches the runner-up (fewer registers):
    /// the winner's cublaslt_best CI is the runner-up's own paired test, and
    /// absent (never the fastest config's) when only the fastest was paired.
    #[test]
    fn winner_lt_ci_is_of_the_dispatched_config() {
        let p = std::env::temp_dir().join(format!("aegis-perf-ltci-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let seed = |with_pick_pairing: bool| {
            let con = connect(&p).unwrap();
            con.execute_batch(
                "DELETE FROM winners; DELETE FROM measurements; DELETE FROM paired; DELETE FROM runs; DELETE FROM kernels;
                 INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES ('r', 'dev', 121, '2026-01-01');
                 INSERT INTO kernels (kernel_id, op, impl, config_json, regs, smem_bytes) VALUES
                   ('kf', 'gemm', 'fast', '{}', 128, 0), ('kp', 'gemm', 'pick', '{}', 64, 0),
                   ('kl', 'gemm', 'cublaslt_best', '{}', -1, 0);
                 INSERT INTO measurements (run_id, kernel_id, dtype, m, n, k, median_us, p10_us, p90_us, tflops,
                                           checked, pass, protocol_version, cache_mode, allow_busy) VALUES
                   ('r', 'kf', 'f16f32f16', 64, 64, 64, 10.0, 9.9, 10.1, 1, 1, 1, 2, 'cold', 0),
                   ('r', 'kp', 'f16f32f16', 64, 64, 64, 10.05, 9.9, 10.2, 1, 1, 1, 2, 'cold', 0),
                   ('r', 'kl', 'f16f32f16', 64, 64, 64, 9.0, 8.9, 9.1, 1, 0, 0, 2, 'cold', 0);
                 INSERT INTO paired (run_id, op, dtype, m, n, k, a_impl, b_impl, role, pairs, ratio_median,
                                     ratio_ci_lo, ratio_ci_hi, tied, protocol_version, cache_mode) VALUES
                   ('r', 'gemm', 'f16f32f16', 64, 64, 64, 'fast', 'pick', 'runner_up', 30, 1.0, 0.99, 1.01, 1, 2, 'cold'),
                   ('r', 'gemm', 'f16f32f16', 64, 64, 64, 'fast', 'cublaslt_best', 'cublaslt_best', 30, 0.9, 0.89, 0.91, 0, 2, 'cold');",
            )
            .unwrap();
            if with_pick_pairing {
                con.execute_batch(
                    "INSERT INTO paired (run_id, op, dtype, m, n, k, a_impl, b_impl, role, pairs, ratio_median,
                                         ratio_ci_lo, ratio_ci_hi, tied, protocol_version, cache_mode) VALUES
                       ('r', 'gemm', 'f16f32f16', 64, 64, 64, 'pick', 'cublaslt_best', 'cublaslt_best', 30, 0.88,
                        0.87, 0.89, 0, 2, 'cold');",
                )
                .unwrap();
            }
        };
        let winner = || -> (String, Option<f64>, Option<f64>) {
            let con = connect(&p).unwrap();
            con.query_row(
                "SELECT k.impl, w.lt_ratio_ci_lo, w.lt_ratio_ci_hi FROM winners w JOIN kernels k USING(kernel_id)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap()
        };
        seed(false);
        cmd_winners(&p, Some(121), None).unwrap();
        assert_eq!(winner(), ("pick".to_string(), None, None), "fastest config's CI must not be borrowed");
        seed(true);
        cmd_winners(&p, Some(121), None).unwrap();
        assert_eq!(winner(), ("pick".to_string(), Some(0.87), Some(0.89)));
        std::fs::remove_file(&p).unwrap();
    }

    /// runs (run_id, git_sha, build_sha, build_dirty, allow_dirty, kind).
    type RunBuildRow = (String, Option<String>, Option<String>, Option<i64>, i64, String);

    /// A row from `build`: Some((sha, dirty)) names it, None is a binary from before v7.
    fn row_from(build: Option<(&str, bool)>) -> String {
        let r = row(0.1, false);
        let cut = r.find(",\"build_sha\"").unwrap();
        match build {
            Some((sha, dirty)) => format!("{},\"build_sha\":\"{sha}\",\"build_dirty\":{dirty}}}", &r[..cut]),
            None => format!("{}}}", &r[..cut]),
        }
    }

    fn ingest_rows(db: &Path, id: &str, lines: &[String], opts: IngestOpts) -> Result<String> {
        let f = tmp(&format!("{id}.jsonl"));
        std::fs::write(&f, lines.join("\n") + "\n").unwrap();
        let r = cmd_ingest(db, &f, None, Some(id.into()), None, opts);
        std::fs::remove_file(&f).ok();
        r
    }

    /// v7: a clean build is stored with its full SHA (git_sha keeps 7 digits);
    /// a dirty or unknown build is refused, and with --allow-dirty stored,
    /// marked and left out of every default selection.
    #[test]
    fn ingest_refuses_dirty_builds_unless_allowed() {
        let db = tmp("dirty-build.db");
        let _ = std::fs::remove_file(&db);
        let sweep = IngestOpts::new(RunKind::Sweep);
        let allow = IngestOpts { allow_dirty: true, ..sweep };
        ingest_rows(&db, "clean", &[row_from(Some((CLEAN, false)))], sweep).unwrap();
        for (id, b) in [("dirty", Some((CLEAN, true))), ("unknown", None)] {
            let err = ingest_rows(&db, id, &[row_from(b)], sweep).unwrap_err().to_string();
            assert!(err.contains("refusing to store results") && err.contains("--allow-dirty"), "{err}");
            ingest_rows(&db, id, &[row_from(b)], allow).unwrap();
        }
        let err = ingest_rows(&db, "mixed", &[row_from(Some((CLEAN, false))), row_from(Some((CLEAN, true)))], allow)
            .unwrap_err()
            .to_string();
        assert!(err.contains("different builds"), "{err}");
        assert!(ingest_rows(&db, "short", &[row_from(Some(("0123456", false)))], allow).is_err(), "7 digits: not a full SHA");

        let con = connect_ro(&db).unwrap();
        let runs: Vec<RunBuildRow> = con
            .prepare("SELECT run_id, git_sha, build_sha, build_dirty, allow_dirty, kind FROM runs ORDER BY run_id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let s = |x: &str| Some(x.to_string());
        assert_eq!(
            runs,
            vec![
                ("clean".into(), s("0123456"), s(CLEAN), Some(0), 0, "sweep".into()),
                ("dirty".into(), s("0123456"), s(CLEAN), Some(1), 1, "sweep".into()),
                ("unknown".into(), None, None, None, 1, "sweep".into()),
            ]
        );
        let ranked = |sel: &Selection| -> Vec<String> {
            con.prepare(&format!("SELECT m.run_id FROM measurements m WHERE 1 = 1{} ORDER BY m.run_id", sel.sql()))
                .unwrap()
                .query_map([], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        assert_eq!(ranked(&Selection::ranking()), vec!["clean".to_string()]);
        assert_eq!(ranked(&Selection { include_dirty: true, ..Selection::ranking() }).len(), 3);
        let views = crate::view::runs(&con).unwrap();
        let dirty: Vec<(&str, bool)> = views.iter().map(|r| (r.run_id.as_str(), r.dirty())).collect();
        assert_eq!(dirty, vec![("clean", false), ("dirty", true), ("unknown", true)]);
        assert_eq!(views[0].kind, Some(RunKind::Sweep));
        drop(con);
        std::fs::remove_file(&db).ok();
    }

    /// A microbench run takes its build from the caller (bench_gemm --build-info of the same build).
    #[test]
    fn microbench_ingest_checks_the_build() {
        let db = tmp("mb-build.db");
        let _ = std::fs::remove_file(&db);
        let f = tmp("mb-build.jsonl");
        std::fs::write(&f, r#"{"probe":"p","param":{},"value":1.0,"unit":"ns","sm_arch":121}"#.to_string() + "\n").unwrap();
        let idle = RunIdle::default();
        let dirty = BuildId::new(CLEAN, true).unwrap();
        assert!(cmd_ingest_microbench(&db, &f, None, Some("d".into()), &idle, Some(&dirty), false).is_err());
        assert!(cmd_ingest_microbench(&db, &f, None, Some("u".into()), &idle, None, false).is_err());
        cmd_ingest_microbench(&db, &f, None, Some("ok".into()), &idle, Some(&clean()), false).unwrap();
        std::fs::remove_file(&f).ok();
        let con = connect_ro(&db).unwrap();
        let r = crate::view::run(&con, "ok").unwrap().unwrap();
        assert_eq!((r.build_sha.as_deref(), r.build_dirty, r.kind), (Some(CLEAN), Some(false), Some(RunKind::Microbench)));
        drop(con);
        std::fs::remove_file(&db).ok();
    }

    /// Migrating to v7: runs get a kind from what they stored (paired tie tests in any state
    /// make a tune), and paired_results their dirty flags from the "-dirty" suffix.
    #[test]
    fn migrate_v7_backfills_kind_and_dirty_flags() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(SCHEMA).unwrap();
        con.execute_batch(
            "DROP TABLE runs; CREATE TABLE runs (run_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, sm_arch INTEGER NOT NULL,
               gpu_name TEXT, cuda_ver INTEGER, driver_ver INTEGER, git_sha TEXT, power_mode TEXT, clocks_locked INTEGER,
               temp_start_c REAL, measured_bw_gbps REAL, started_at TEXT NOT NULL);
             INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES
               ('tune', 'd', 121, '1'), ('sweep', 'd', 121, '2'), ('mb', 'd', 121, '3'), ('empty', 'd', 121, '4');
             INSERT INTO kernels (kernel_id, op, impl, config_json) VALUES ('k', 'gemm', 'x', '{}');
             INSERT INTO measurements (run_id, kernel_id, dtype) VALUES ('tune', 'k', 'f'), ('sweep', 'k', 'f');
             INSERT INTO paired (run_id, op, dtype, a_impl, b_impl) VALUES ('tune', 'gemm', 'f', 'x', 'y');
             INSERT INTO microbench (run_id, probe, param_json, unit) VALUES ('mb', 'p', '{}', 'ns');
             DROP TABLE paired_results;
             CREATE TABLE paired_results (id INTEGER PRIMARY KEY, sha_a TEXT, sha_b TEXT, shape TEXT NOT NULL,
               impl_a TEXT NOT NULL, impl_b TEXT NOT NULL, ratio REAL NOT NULL, ci_lo REAL NOT NULL, ci_hi REAL NOT NULL,
               n INTEGER NOT NULL, protocol_version INTEGER NOT NULL, cache_mode TEXT NOT NULL, idle_reading REAL,
               clocks_locked INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL, finished_at TEXT NOT NULL);
             INSERT INTO paired_results (id, sha_a, sha_b, shape, impl_a, impl_b, ratio, ci_lo, ci_hi, n, protocol_version,
                                         cache_mode, started_at, finished_at) VALUES
               (1, '0bd0f22a35a6-dirty', '0bd0f22a35a6-dirty', '1x1x1', 'a', 'a', 1, 1, 1, 30, 2, 'cold', 't', 't'),
               (2, '6b9a3cc228d2', '6b9a3cc228d2-dirty', '1x1x1', 'a', 'a', 1, 1, 1, 30, 2, 'cold', 't', 't'),
               (3, NULL, 'b0bedd58fbf9', '1x1x1', 'a', 'a', 1, 1, 1, 30, 2, 'cold', 't', 't');",
        )
        .unwrap();
        migrate(&con).unwrap();
        let kinds: Vec<(String, Option<String>)> = con
            .prepare("SELECT run_id, kind FROM runs ORDER BY started_at")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let s = |x: &str| Some(x.to_string());
        assert_eq!(
            kinds,
            vec![("tune".into(), s("tune")), ("sweep".into(), s("sweep")), ("mb".into(), s("microbench")), ("empty".into(), None)]
        );
        let flags: Vec<(Option<i64>, Option<i64>, Option<String>)> = con
            .prepare("SELECT build_dirty_a, build_dirty_b, build_sha_a FROM paired_results ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(flags, vec![(Some(1), Some(1), None), (Some(0), Some(1), None), (None, Some(0), None)]);
        let runs_dirty: Option<i64> = con.query_row("SELECT build_dirty FROM runs WHERE run_id = 'tune'", [], |r| r.get(0)).unwrap();
        assert_eq!(runs_dirty, None, "a pre-v7 run's build state is unknown, not guessed");
    }

    #[test]
    fn kernel_id_is_stable() {
        // The id every existing DB row uses for tiled (computed by the former Python tool).
        let cfg = serde_json::json!({"tile": 32, "block": [32, 32]});
        assert_eq!(kernel_id("gemm", "tiled", &cfg), PY_TILED_KID);
    }

    const PY_TILED_KID: &str = "35d3e137499cbc5c";
}
