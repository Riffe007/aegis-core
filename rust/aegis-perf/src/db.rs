//! Perf DB commands, a line-for-line port of tools/perfdb.py.
//!
//! Output text, SQL, kernel ids and row selection match the Python tool so
//! both can write the same DB and print identical reports.
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

/// Copied verbatim from `SCHEMA` in tools/perfdb.py (a test checks it).
pub const SCHEMA: &str = include_str!("schema.sql");

pub fn connect(db: &Path) -> Result<Connection> {
    if let Some(parent) = db.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let con = Connection::open(db).with_context(|| format!("open {}", db.display()))?;
    con.execute_batch(SCHEMA)?;
    Ok(con)
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
        tx.execute(
            "INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            params![
                run_id, kid, sql(Some(get(r, "dtype")?)),
                sql(Some(get(s, "M")?)), sql(Some(get(s, "N")?)), sql(Some(get(s, "K")?)),
                sql(Some(get(r, "median_us")?)), sql(Some(get(r, "p10_us")?)), sql(Some(get(r, "p90_us")?)),
                sql(Some(get(r, "tflops")?)), sql(Some(get(r, "gbps")?)), sql(Some(get(r, "reps")?)),
                get(r, "checked")?.as_bool().unwrap_or(false) as i64,
                get(r, "pass")?.as_bool().unwrap_or(false) as i64,
                sql(Some(get(r, "max_abs_err")?)), sql(Some(get(r, "rel_err")?)),
                sql(r.get("power_w")),
            ],
        )?;
    }
    tx.commit()?;
    println!("run {run_id}: ingested {} measurements", rows.len());
    Ok(())
}

type KernelShapeKey = (i64, String, String, i64, i64, i64, String); // arch, op, dtype, m, n, k, kernel_id
type ShapeKey = (i64, String, String, i64, i64, i64); // arch, op, dtype, m, n, k

/// One row of `_latest_per_kernel_shape` in perfdb.py.
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
}

/// Most recent measurement per (arch, kernel, shape); only checked+passing rows
/// are eligible, except cuBLAS which is the reference itself. Keeps Python's
/// dict semantics: first-seen key order, later runs overwrite values.
fn latest_per_kernel_shape(con: &Connection, arch: Option<i64>) -> Result<Vec<Latest>> {
    let arch = arch.filter(|&a| a != 0); // Python: `if arch`
    let q = format!(
        "SELECT r.sm_arch, k.op, m.dtype, m.m, m.n, m.k, k.impl, k.kernel_id, m.run_id, \
                m.median_us, m.tflops, k.regs, k.local_bytes, m.checked, m.pass \
         FROM measurements m JOIN runs r USING(run_id) JOIN kernels k USING(kernel_id) \
         WHERE (k.impl = 'cublas' OR (m.checked = 1 AND m.pass = 1)) {} ORDER BY r.started_at",
        if arch.is_some() { "AND r.sm_arch = ?" } else { "" }
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

pub fn cmd_winners(db: &Path, arch: Option<i64>) -> Result<()> {
    let con = connect(db)?;
    let rows = latest_per_kernel_shape(&con, arch)?;
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
    let mut n = 0;
    for ((arch, op, dtype, m, nn, k), rs) in &groups {
        let cublas: Vec<&&Latest> = rs.iter().filter(|r| r.impl_ == "cublas").collect();
        let ours: Vec<&&Latest> = rs.iter().filter(|r| r.impl_ != "cublas").collect();
        // Python min(): first of equal minima wins.
        let Some(best) = ours.iter().copied().reduce(|a, b| if b.median_us < a.median_us { b } else { a }) else {
            continue;
        };
        let pct = cublas.first().map(|c| c.median_us / best.median_us * 100.0);
        con.execute(
            "INSERT OR REPLACE INTO winners VALUES (?,?,?,?,?,?,?,?)",
            params![arch, op, dtype, shape_bucket(*m, *nn, *k), best.kernel_id, best.run_id, best.median_us, pct],
        )?;
        n += 1;
    }
    println!("wrote {n} winners");
    Ok(())
}

/// The report text, exactly as `perfdb.py report` prints it.
pub fn report_text(con: &Connection, arch: Option<i64>) -> Result<String> {
    let rows = latest_per_kernel_shape(con, arch)?;
    // table[(arch, m, n, k)][impl] = row; later rows overwrite (Python dict).
    let mut table: HashMap<(i64, i64, i64, i64), HashMap<String, &Latest>> = HashMap::new();
    for r in &rows {
        table.entry((r.sm_arch, r.m, r.n, r.k)).or_default().insert(r.impl_.clone(), r);
    }
    let mut impls: Vec<String> = rows.iter().filter(|r| r.impl_ != "cublas").map(|r| r.impl_.clone()).collect();
    impls.sort();
    impls.dedup();
    let widths: Vec<usize> = impls.iter().map(|i| i.len().max(11)).collect();

    let mut out = String::new();
    let mut hdr = format!("{:>5} {:>6} {:>6} {:>6} {:>10} {:>7} ", "arch", "M", "N", "K", "cuBLAS us", "TFLOPS");
    hdr += &impls.iter().zip(&widths).map(|(i, w)| format!("{i:>w$}")).collect::<Vec<_>>().join(" ");
    writeln!(out, "{hdr}")?;
    writeln!(out, "{}", "-".repeat(hdr.chars().count()))?;
    let mut keys: Vec<_> = table.keys().copied().collect();
    keys.sort();
    for key in keys {
        let byimpl = &table[&key];
        let cb = byimpl.get("cublas");
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
    writeln!(out, "\nCells are % of cuBLAS speed (100% = parity, >100% = faster than cuBLAS).")?;
    Ok(out)
}

pub fn cmd_report(db: &Path, arch: Option<i64>) -> Result<()> {
    let con = connect(db)?;
    print!("{}", report_text(&con, arch)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SCHEMA must stay a verbatim copy of tools/perfdb.py's.
    #[test]
    fn schema_matches_perfdb_py() {
        let py = include_str!("../../../tools/perfdb.py");
        let start = py.find("SCHEMA = \"\"\"").expect("SCHEMA in perfdb.py") + "SCHEMA = \"\"\"".len();
        let end = start + py[start..].find("\"\"\"").unwrap();
        assert_eq!(&py[start..end], SCHEMA, "schema.sql drifted from tools/perfdb.py SCHEMA");
    }

    #[test]
    fn kernel_id_matches_python() {
        // python3 -c 'import hashlib,json; print(hashlib.sha1(("gemm|tiled|"+json.dumps({"tile":32,"block":[32,32]},sort_keys=True)).encode()).hexdigest()[:16])'
        let cfg = serde_json::json!({"tile": 32, "block": [32, 32]});
        assert_eq!(kernel_id("gemm", "tiled", &cfg), PY_TILED_KID);
    }

    const PY_TILED_KID: &str = "35d3e137499cbc5c";
}
