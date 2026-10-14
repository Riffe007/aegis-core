//! Perf DB commands: schema, ingest, winners, report.
//!
//! Originally a line-for-line port of the (deleted) Python perfdb tool; kernel
//! ids, row selection and report text are unchanged so existing DBs stay valid.
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, Connection};
use serde_json::Value;
use sha1::{Digest, Sha1};

use crate::pyjson::dumps_sorted;

/// The perf DB schema. Source of truth; v1 columns are frozen (add, never rename).
pub const SCHEMA: &str = include_str!("schema.sql");

pub fn connect(db: &Path) -> Result<Connection> {
    if let Some(parent) = db.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let con = Connection::open(db).with_context(|| format!("open {}", db.display()))?;
    con.execute_batch(SCHEMA)?;
    migrate(&con)?;
    Ok(con)
}

/// Columns added after v1. CREATE TABLE IF NOT EXISTS leaves existing tables
/// alone, so add any missing column here (additive only; never rename).
const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    ("winners", "tied", "INTEGER NOT NULL DEFAULT 0"),
    ("winners", "tie_set", "TEXT"),
    ("winners", "pick_reason", "TEXT"),
    ("microbench", "superseded", "INTEGER NOT NULL DEFAULT 0"),
    ("microbench", "superseded_by", "TEXT"),
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
];

/// Reference implementations: always eligible (they are what --check compares
/// against), never winners.
const BASELINES: &str = "'cublas', 'cublaslt_best'";

pub fn is_baseline(impl_: &str) -> bool {
    impl_ == "cublas" || impl_ == "cublaslt_best"
}

/// Which measurements a winners / report query reads.
#[derive(Clone, Debug)]
pub struct Selection {
    pub protocol: i64,
    /// v2 cache mode ("cold" / "warm"); ignored for protocol 1.
    pub cache: String,
    /// Include rows measured on a busy GPU (--allow-busy).
    pub include_busy: bool,
}

impl Selection {
    /// What winners are picked from: protocol v2, cold, idle GPU.
    pub fn ranking() -> Self {
        Selection { protocol: 2, cache: "cold".to_string(), include_busy: false }
    }

    fn sql(&self) -> String {
        let mut s = format!(" AND m.protocol_version = {}", self.protocol);
        if self.protocol >= 2 {
            s += &format!(" AND m.cache_mode = '{}'", self.cache.replace('\'', "''"));
        }
        if !self.include_busy {
            s += " AND m.allow_busy = 0";
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
        }
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

fn git_sha() -> Option<String> {
    command_stdout("git", &["rev-parse", "--short", "HEAD"]).map(|s| s.trim().to_string())
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
) -> Result<()> {
    let text = std::fs::read_to_string(results).with_context(|| format!("read {}", results.display()))?;
    let rows: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if rows.is_empty() {
        bail!("no rows in results file");
    }
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
    tx.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        params![
            run_id,
            sql(Some(&device_id)),
            sql(Some(get(first, "sm_arch")?)),
            sql(first.get("device")),
            sql(first.get("cuda_runtime")),
            sql(first.get("cuda_driver")),
            git_sha(),
            power_mode(),
            SqlValue::Null,
            temp_start.or_else(gpu_temp_c),
            sql(manifest.get("measured_d2d_bw_gbps")),
            started_at,
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
             timer_res_ns, cache_mode, buffer_sets, window_ms, allow_busy, idle_util_pct) \
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
            ],
        )?;
    }
    tx.commit()?;
    println!("run {run_id}: ingested {} measurements", rows.len());
    Ok(())
}

type KernelShapeKey = (i64, String, String, i64, i64, i64, String); // arch, op, dtype, m, n, k, kernel_id
type ShapeKey = (i64, String, String, i64, i64, i64); // arch, op, dtype, m, n, k

/// Microbench probe lines (microbench/*): one run row with the device metadata
/// of the first line, then (run_id, probe, param_json, value, unit) per line.
pub fn cmd_ingest_microbench(db: &Path, results: &Path, manifest: Option<&Path>, run_id: Option<String>) -> Result<()> {
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
    let mut con = connect(db)?;
    let tx = con.transaction()?;
    let run_id = run_id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()[..12].to_string());
    let device_id = manifest.get("device_id").cloned().unwrap_or_else(|| Value::String(hostname()));
    let started_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string();
    tx.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        params![
            run_id, sql(Some(&device_id)), sql(Some(get(first, "sm_arch")?)), sql(first.get("device")),
            sql(first.get("cuda_runtime")), sql(first.get("cuda_driver")), git_sha(), power_mode(),
            SqlValue::Null, gpu_temp_c(), sql(manifest.get("measured_d2d_bw_gbps")), started_at,
        ],
    )?;
    let mut probes: Vec<String> = Vec::new();
    for r in &rows {
        let probe = get(r, "probe")?.as_str().unwrap_or_default().to_string();
        tx.execute(
            "INSERT INTO microbench (run_id, probe, param_json, value, unit) VALUES (?,?,?,?,?)",
            params![run_id, probe, dumps_sorted(get(r, "param")?), sql(Some(get(r, "value")?)), sql(Some(get(r, "unit")?))],
        )?;
        if !probes.contains(&probe) {
            probes.push(probe);
        }
    }
    // Older rows of a re-measured probe on the same arch are kept, marked superseded.
    let mut superseded = 0;
    for probe in &probes {
        superseded += tx.execute(
            "UPDATE microbench SET superseded = 1, superseded_by = ?1 \
             WHERE probe = ?2 AND run_id <> ?1 AND superseded = 0 \
               AND run_id IN (SELECT run_id FROM runs WHERE sm_arch = ?3)",
            params![run_id, probe, sql(Some(get(first, "sm_arch")?))],
        )?;
    }
    tx.commit()?;
    println!("run {run_id}: ingested {} microbench rows; marked {superseded} older rows superseded", rows.len());
    Ok(())
}

/// Latest passing measurement per (arch, kernel, shape).
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

/// Most recent measurement per (arch, kernel, shape); only checked+passing rows
/// are eligible, except cuBLAS which is the reference itself. Keeps Python's
/// dict semantics: first-seen key order, later runs overwrite values.
fn latest_per_kernel_shape(con: &Connection, arch: Option<i64>, sel: &Selection) -> Result<Vec<Latest>> {
    let arch = arch.filter(|&a| a != 0); // Python: `if arch`
    let q = format!(
        "SELECT r.sm_arch, k.op, m.dtype, m.m, m.n, m.k, k.impl, k.kernel_id, m.run_id, \
                m.median_us, m.tflops, m.p10_us, m.p90_us, k.regs, k.smem_bytes \
         FROM measurements m JOIN runs r USING(run_id) JOIN kernels k USING(kernel_id) \
         WHERE (k.impl IN ({BASELINES}) OR (m.checked = 1 AND m.pass = 1)) {}{} ORDER BY r.started_at",
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
        let key = (r.sm_arch, r.op.clone(), r.dtype.clone(), r.m, r.n, r.k, r.kernel_id.clone());
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
fn tie_set<'a>(ours: &[&'a Latest]) -> Vec<&'a Latest> {
    let mut sorted: Vec<&Latest> = ours.to_vec();
    sorted.sort_by(|a, b| a.median_us.total_cmp(&b.median_us).then_with(|| a.impl_.cmp(&b.impl_)));
    match sorted[..] {
        [] => Vec::new(),
        [w] => vec![w],
        [w, r, ..] => {
            let (lo, hi) = (w.p10_us.min(w.median_us), w.p90_us.max(w.median_us));
            if r.median_us >= lo && r.median_us <= hi { vec![w, r] } else { vec![w] }
        }
    }
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

/// Winner per (arch, op, dtype, shape). Clear win: the fastest median. Tie
/// (runner-up median inside the fastest one's p10-p90): of the two, the one
/// that wins or ties on the most neighboring shapes, then fewer registers,
/// then less smem, then lower median, then name.
pub fn cmd_winners(db: &Path, arch: Option<i64>) -> Result<()> {
    let con = connect(db)?;
    // Ranked on protocol v2, cold, idle-GPU rows only (CLAUDE.md Benchmark policy).
    let sel = Selection::ranking();
    let rows = latest_per_kernel_shape(&con, arch, &sel)?;
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
            let ours: Vec<&Latest> = rs.iter().copied().filter(|r| !is_baseline(&r.impl_)).collect();
            (key.clone(), tie_set(&ours).iter().map(|r| r.impl_.clone()).collect())
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
             WHERE m.kernel_id = ? AND r.sm_arch = ? AND m.dtype = ? AND m.m = ? AND m.n = ? AND m.k = ? {} \
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
        let ours: Vec<&Latest> = rs.iter().copied().filter(|r| !is_baseline(&r.impl_)).collect();
        let tie = tie_set(&ours);
        let Some(&fastest) = tie.first() else { continue };
        let (pick, reason) = if tie.len() == 1 {
            (fastest, "fastest".to_string())
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
        let pct = cublas.map(|c| c.median_us / pick.median_us * 100.0);
        let pct_lt = lt_best.map(|c| c.median_us / pick.median_us * 100.0);
        let set_json = serde_json::to_string(&tie.iter().map(|r| r.impl_.as_str()).collect::<Vec<_>>())?;
        con.execute(
            "INSERT OR REPLACE INTO winners (sm_arch, op, dtype, shape_bucket, kernel_id, run_id, median_us, \
             pct_of_cublas, tied, tie_set, pick_reason, protocol_version, warm_median_us, pct_of_cublaslt_best) \
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            params![arch, op, dtype, shape_bucket(*m, *nn, *k), pick.kernel_id, pick.run_id, pick.median_us, pct,
                    (tie.len() > 1) as i64, set_json, reason, sel.protocol,
                    warm_of(&pick.kernel_id, (*arch, dtype.as_str(), *m, *nn, *k))?, pct_lt],
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
    writeln!(out, "Rows: timing protocol v{}{cache}, {busy} --allow-busy rows.", sel.protocol)?;
    Ok(out)
}

pub fn cmd_report(db: &Path, arch: Option<i64>, sel: &Selection, baseline: Option<&str>) -> Result<()> {
    let con = connect(db)?;
    print!("{}", report_text(&con, arch, sel, baseline.unwrap_or(default_baseline(sel)))?);
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

    #[test]
    fn kernel_id_is_stable() {
        // The id every existing DB row uses for tiled (computed by the former Python tool).
        let cfg = serde_json::json!({"tile": 32, "block": [32, 32]});
        assert_eq!(kernel_id("gemm", "tiled", &cfg), PY_TILED_KID);
    }

    const PY_TILED_KID: &str = "35d3e137499cbc5c";
}
