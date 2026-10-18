//! Read-only queries for viewers (AEGIS Custos). Every query a UI needs lives
//! here, next to the schema, so a schema change breaks viewers at compile time
//! rather than at run time. Nothing here writes; open the DB with
//! [`crate::db::connect_ro`].
//!
//! Rows are returned as measured. Roof-derived metrics (% of peak, % of
//! bandwidth, % of roof) are not stored and not returned: viewers compute them
//! at view time from the arch file with [`crate::params::ArchParams::roof`].
use anyhow::{Context, Result};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection, Row};

use crate::db::Selection;

/// One run (`runs` table).
#[derive(Clone, Debug)]
pub struct Run {
    pub run_id: String,
    pub device_id: String,
    pub sm_arch: i64,
    pub gpu_name: Option<String>,
    pub cuda_ver: Option<i64>,
    pub driver_ver: Option<i64>,
    pub git_sha: Option<String>,
    pub power_mode: Option<String>,
    pub clocks_locked: Option<i64>,
    pub temp_start_c: Option<f64>,
    pub measured_bw_gbps: Option<f64>,
    pub started_at: String,
    /// The idle guard's mean GPU utilization (%) before the run; None if not
    /// recorded (older runs without it on their rows).
    pub idle_util_pct: Option<f64>,
    pub idle_source: Option<String>,
    /// The guard found the GPU busy and the run went ahead (--allow-busy).
    pub allow_busy: bool,
    /// The idle guard during the run: Some(true) other processes used the GPU
    /// above the limit mid-run (every row of the run is marked); Some(false)
    /// watched and clean; None not watched.
    pub disturbed: Option<bool>,
    pub disturbed_detail: Option<String>,
}

const RUN_COLS: &str = "run_id, device_id, sm_arch, gpu_name, cuda_ver, driver_ver, git_sha, power_mode, \
                        clocks_locked, temp_start_c, measured_bw_gbps, started_at, idle_util_pct, idle_source, \
                        allow_busy, disturbed, disturbed_detail";

fn run_row(r: &Row) -> rusqlite::Result<Run> {
    Ok(Run {
        run_id: r.get(0)?,
        device_id: r.get(1)?,
        sm_arch: r.get(2)?,
        gpu_name: r.get(3)?,
        cuda_ver: r.get(4)?,
        driver_ver: r.get(5)?,
        git_sha: r.get(6)?,
        power_mode: r.get(7)?,
        clocks_locked: r.get(8)?,
        temp_start_c: r.get(9)?,
        measured_bw_gbps: r.get(10)?,
        started_at: r.get(11)?,
        idle_util_pct: r.get(12)?,
        idle_source: r.get(13)?,
        allow_busy: r.get::<_, i64>(14)? != 0,
        disturbed: r.get::<_, Option<i64>>(15)?.map(|d| d != 0),
        disturbed_detail: r.get(16)?,
    })
}

/// Every run, oldest first.
pub fn runs(con: &Connection) -> Result<Vec<Run>> {
    let mut stmt = con.prepare(&format!("SELECT {RUN_COLS} FROM runs ORDER BY started_at"))?;
    let rows = stmt.query_map([], run_row)?.collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

pub fn run(con: &Connection, run_id: &str) -> Result<Option<Run>> {
    let mut stmt = con.prepare(&format!("SELECT {RUN_COLS} FROM runs WHERE run_id = ?"))?;
    let mut rows = stmt.query_map([run_id], run_row)?;
    Ok(rows.next().transpose()?)
}

/// One kernel (`kernels` table): an impl and its config.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub kernel_id: String,
    pub op: String,
    pub impl_: String,
    pub config_json: String,
    pub regs: Option<i64>,
    pub smem_bytes: Option<i64>,
    pub local_bytes: Option<i64>,
    pub ptx_hash: Option<String>,
    pub sass_path: Option<String>,
}

pub fn kernel(con: &Connection, kernel_id: &str) -> Result<Option<Kernel>> {
    let mut stmt = con.prepare(
        "SELECT kernel_id, op, impl, config_json, regs, smem_bytes, local_bytes, ptx_hash, sass_path \
         FROM kernels WHERE kernel_id = ?",
    )?;
    let mut rows = stmt.query_map([kernel_id], |r| {
        Ok(Kernel {
            kernel_id: r.get(0)?,
            op: r.get(1)?,
            impl_: r.get(2)?,
            config_json: r.get(3)?,
            regs: r.get(4)?,
            smem_bytes: r.get(5)?,
            local_bytes: r.get(6)?,
            ptx_hash: r.get(7)?,
            sass_path: r.get(8)?,
        })
    })?;
    Ok(rows.next().transpose()?)
}

/// One measurement row with its kernel and run context.
#[derive(Clone, Debug)]
pub struct Measurement {
    pub run_id: String,
    pub kernel_id: String,
    pub op: String,
    pub impl_: String,
    pub dtype: String,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub median_us: f64,
    pub p10_us: Option<f64>,
    pub p90_us: Option<f64>,
    pub reps: Option<i64>,
    pub checked: bool,
    pub pass: bool,
    pub max_abs_err: Option<f64>,
    pub rel_err: Option<f64>,
    pub power_w: Option<f64>,
    pub protocol_version: i64,
    pub batch_n: Option<i64>,
    pub timer_res_ns: Option<f64>,
    pub cache_mode: Option<String>,
    pub buffer_sets: Option<i64>,
    pub window_ms: Option<f64>,
    pub allow_busy: bool,
    pub idle_util_pct: Option<f64>,
    pub sm_arch: i64,
    pub device_id: String,
    pub git_sha: Option<String>,
    pub started_at: String,
}

/// Which measurements to read. The default is what every view shows unless
/// asked otherwise: protocol v2, cold cache, idle GPU, every arch.
#[derive(Clone, Debug)]
pub struct MeasurementFilter {
    pub sel: Selection,
    pub arch: Option<i64>,
    pub device_id: Option<String>,
    pub kernel_id: Option<String>,
    pub impl_: Option<String>,
    pub shape: Option<(i64, i64, i64)>,
    pub git_sha: Option<String>,
    pub run_id: Option<String>,
}

impl Default for MeasurementFilter {
    fn default() -> Self {
        MeasurementFilter {
            sel: Selection::ranking(),
            arch: None,
            device_id: None,
            kernel_id: None,
            impl_: None,
            shape: None,
            git_sha: None,
            run_id: None,
        }
    }
}

/// Measurements matching `f`, oldest run first.
pub fn measurements(con: &Connection, f: &MeasurementFilter) -> Result<Vec<Measurement>> {
    let mut q = String::from(
        "SELECT m.run_id, m.kernel_id, k.op, k.impl, m.dtype, m.m, m.n, m.k, m.median_us, m.p10_us, m.p90_us, \
                m.reps, m.checked, m.pass, m.max_abs_err, m.rel_err, m.power_w, m.protocol_version, m.batch_n, \
                m.timer_res_ns, m.cache_mode, m.buffer_sets, m.window_ms, m.allow_busy, m.idle_util_pct, \
                r.sm_arch, r.device_id, r.git_sha, r.started_at \
         FROM measurements m JOIN runs r USING(run_id) JOIN kernels k USING(kernel_id) WHERE 1 = 1",
    );
    q += &f.sel.sql();
    let mut args: Vec<SqlValue> = Vec::new();
    let mut eq = |col: &str, v: SqlValue| {
        q += &format!(" AND {col} = ?");
        args.push(v);
    };
    if let Some(a) = f.arch {
        eq("r.sm_arch", a.into());
    }
    if let Some(d) = &f.device_id {
        eq("r.device_id", d.clone().into());
    }
    if let Some(kid) = &f.kernel_id {
        eq("m.kernel_id", kid.clone().into());
    }
    if let Some(i) = &f.impl_ {
        eq("k.impl", i.clone().into());
    }
    if let Some((m, n, k)) = f.shape {
        eq("m.m", m.into());
        eq("m.n", n.into());
        eq("m.k", k.into());
    }
    if let Some(s) = &f.git_sha {
        eq("r.git_sha", s.clone().into());
    }
    if let Some(r) = &f.run_id {
        eq("m.run_id", r.clone().into());
    }
    q += " ORDER BY r.started_at, k.impl, m.m, m.n, m.k";
    let mut stmt = con.prepare(&q)?;
    let rows = stmt
        .query_map(params_from_iter(args), |r| {
            Ok(Measurement {
                run_id: r.get(0)?,
                kernel_id: r.get(1)?,
                op: r.get(2)?,
                impl_: r.get(3)?,
                dtype: r.get(4)?,
                m: r.get(5)?,
                n: r.get(6)?,
                k: r.get(7)?,
                median_us: r.get(8)?,
                p10_us: r.get(9)?,
                p90_us: r.get(10)?,
                reps: r.get(11)?,
                checked: r.get::<_, Option<i64>>(12)?.unwrap_or(0) != 0,
                pass: r.get::<_, Option<i64>>(13)?.unwrap_or(0) != 0,
                max_abs_err: r.get(14)?,
                rel_err: r.get(15)?,
                power_w: r.get(16)?,
                protocol_version: r.get(17)?,
                batch_n: r.get(18)?,
                timer_res_ns: r.get(19)?,
                cache_mode: r.get(20)?,
                buffer_sets: r.get(21)?,
                window_ms: r.get(22)?,
                allow_busy: r.get::<_, i64>(23)? != 0,
                idle_util_pct: r.get(24)?,
                sm_arch: r.get(25)?,
                device_id: r.get(26)?,
                git_sha: r.get(27)?,
                started_at: r.get(28)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// A paired tie test (`paired` table): ratio b / a, 95% bootstrap CI.
#[derive(Clone, Debug)]
pub struct Paired {
    pub run_id: String,
    pub a_impl: String,
    pub b_impl: String,
    pub role: Option<String>,
    pub pairs: Option<i64>,
    pub ratio_median: f64,
    pub ratio_ci_lo: f64,
    pub ratio_ci_hi: f64,
    pub tied: bool,
}

/// Paired tests of one shape in one run (protocol v2, cold, idle GPU).
pub fn paired(con: &Connection, run_id: &str, (m, n, k): (i64, i64, i64)) -> Result<Vec<Paired>> {
    let mut stmt = con.prepare(
        "SELECT run_id, a_impl, b_impl, role, pairs, ratio_median, ratio_ci_lo, ratio_ci_hi, tied FROM paired \
         WHERE run_id = ? AND m = ? AND n = ? AND k = ? \
           AND protocol_version = 2 AND cache_mode = 'cold' AND allow_busy = 0 AND coalesce(disturbed, 0) = 0",
    )?;
    let rows = stmt
        .query_map(params![run_id, m, n, k], |r| {
            Ok(Paired {
                run_id: r.get(0)?,
                a_impl: r.get(1)?,
                b_impl: r.get(2)?,
                role: r.get(3)?,
                pairs: r.get(4)?,
                ratio_median: r.get(5)?,
                ratio_ci_lo: r.get(6)?,
                ratio_ci_hi: r.get(7)?,
                tied: r.get::<_, Option<i64>>(8)?.unwrap_or(0) != 0,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// One winner (`winners` table) with its kernel, run and paired tests.
#[derive(Clone, Debug)]
pub struct Winner {
    pub sm_arch: i64,
    pub op: String,
    pub dtype: String,
    pub shape_bucket: String,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub kernel: Kernel,
    pub run_id: String,
    pub git_sha: Option<String>,
    pub median_us: f64,
    pub warm_median_us: Option<f64>,
    pub pct_of_cublas: Option<f64>,
    pub pct_of_cublaslt_best: Option<f64>,
    pub tied: bool,
    pub tie_set: Vec<String>,
    pub pick_reason: Option<String>,
    pub protocol_version: i64,
    /// 'paired', 'paired_swap' or 'band' (see db::tie_set).
    pub tie_method: Option<String>,
    /// runner-up / pick, 95% CI (paired tie test).
    pub ratio_ci: Option<(f64, f64)>,
    /// cublaslt_best / pick, 95% CI (paired test).
    pub lt_ratio_ci: Option<(f64, f64)>,
    /// The paired tests of this shape in the winner's run (runner-up, cublaslt_best).
    pub paired: Vec<Paired>,
}

fn ci(lo: Option<f64>, hi: Option<f64>) -> Option<(f64, f64)> {
    Some((lo?, hi?))
}

/// Winners of an arch (every arch: None), in table order.
pub fn winners(con: &Connection, arch: Option<i64>) -> Result<Vec<Winner>> {
    let mut stmt = con.prepare(
        "SELECT w.sm_arch, w.op, w.dtype, w.shape_bucket, w.kernel_id, w.run_id, r.git_sha, w.median_us, \
                w.warm_median_us, w.pct_of_cublas, w.pct_of_cublaslt_best, w.tied, w.tie_set, w.pick_reason, \
                w.protocol_version, w.tie_method, w.ratio_ci_lo, w.ratio_ci_hi, w.lt_ratio_ci_lo, w.lt_ratio_ci_hi \
         FROM winners w LEFT JOIN runs r USING(run_id) WHERE (?1 IS NULL OR w.sm_arch = ?1) ORDER BY w.rowid",
    )?;
    let raw = stmt
        .query_map([arch], |r| {
            Ok((
                (r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?),
                (r.get::<_, String>(4)?, r.get::<_, String>(5)?, r.get::<_, Option<String>>(6)?),
                (r.get::<_, f64>(7)?, r.get::<_, Option<f64>>(8)?, r.get::<_, Option<f64>>(9)?, r.get::<_, Option<f64>>(10)?),
                (r.get::<_, i64>(11)? != 0, r.get::<_, Option<String>>(12)?, r.get::<_, Option<String>>(13)?),
                (r.get::<_, i64>(14)?, r.get::<_, Option<String>>(15)?),
                (r.get::<_, Option<f64>>(16)?, r.get::<_, Option<f64>>(17)?, r.get::<_, Option<f64>>(18)?, r.get::<_, Option<f64>>(19)?),
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut out = Vec::with_capacity(raw.len());
    for ((sm_arch, op, dtype, shape_bucket), (kernel_id, run_id, git_sha), (median_us, warm, pct_cb, pct_lt), (tied, set, reason), (protocol_version, tie_method), (rlo, rhi, llo, lhi)) in raw {
        let d: Vec<i64> = shape_bucket.split('x').map(str::parse).collect::<Result<_, _>>().with_context(|| format!("shape_bucket {shape_bucket}"))?;
        let [m, n, k] = d[..] else { anyhow::bail!("shape_bucket {shape_bucket}: expected MxNxK") };
        let kernel = kernel(con, &kernel_id)?.with_context(|| format!("winner kernel {kernel_id} not in kernels"))?;
        let tie_set: Vec<String> = match set {
            Some(s) => serde_json::from_str(&s).with_context(|| format!("tie_set {s}"))?,
            None => Vec::new(),
        };
        let paired = paired(con, &run_id, (m, n, k))?;
        out.push(Winner {
            sm_arch,
            op,
            dtype,
            shape_bucket,
            m,
            n,
            k,
            kernel,
            run_id,
            git_sha,
            median_us,
            warm_median_us: warm,
            pct_of_cublas: pct_cb,
            pct_of_cublaslt_best: pct_lt,
            tied,
            tie_set,
            pick_reason: reason,
            protocol_version,
            tie_method,
            ratio_ci: ci(rlo, rhi),
            lt_ratio_ci: ci(llo, lhi),
            paired,
        });
    }
    Ok(out)
}

/// One microbench row (`microbench` table).
#[derive(Clone, Debug)]
pub struct Microbench {
    pub run_id: String,
    pub probe: String,
    pub param_json: String,
    pub value: Option<f64>,
    pub unit: String,
    pub status: MicrobenchStatus,
    /// Replaced: the run_id of the newer run of the same probe.
    pub replaced_by: Option<String>,
    /// Latency rows: SM clock at measurement, whether clocks were locked, and
    /// what the probe requested; None on older rows and non-latency probes.
    pub sm_clock_mhz: Option<f64>,
    pub clocks_locked: Option<bool>,
    pub clock_lock: Option<String>,
}

impl Microbench {
    /// A cycles row in ns at the clock it was measured at (derived, never stored).
    pub fn ns(&self) -> Option<f64> {
        match (self.unit.as_str(), self.value, self.sm_clock_mhz) {
            ("cycles", Some(c), Some(mhz)) if mhz > 0.0 => Some(c / mhz * 1e3),
            _ => None,
        }
    }
}

/// A microbench row's status. Invalidated rows (known bad) are never returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicrobenchStatus {
    /// The probe's current measurement.
    Valid,
    /// A newer run of the same probe exists; still valid history.
    Replaced,
}

/// Microbench rows of an arch, oldest run first: valid and replaced rows, each
/// labelled. Invalidated rows (known bad, with a reason) are hidden everywhere.
/// The hardware model uses the newest valid run per probe (db::current_run).
pub fn microbench(con: &Connection, arch: i64) -> Result<Vec<Microbench>> {
    let mut stmt = con.prepare(
        "SELECT m.run_id, m.probe, m.param_json, m.value, m.unit, m.status, m.replaced_by, \
                m.sm_clock_mhz, m.clocks_locked, m.clock_lock \
         FROM microbench m JOIN runs r USING(run_id) \
         WHERE r.sm_arch = ?1 AND m.status <> 'invalidated' ORDER BY r.started_at, m.rowid",
    )?;
    let rows = stmt
        .query_map(params![arch], |r| {
            let status: String = r.get(5)?;
            Ok(Microbench {
                run_id: r.get(0)?,
                probe: r.get(1)?,
                param_json: r.get(2)?,
                value: r.get(3)?,
                unit: r.get(4)?,
                status: if status == "replaced" { MicrobenchStatus::Replaced } else { MicrobenchStatus::Valid },
                replaced_by: r.get(6)?,
                sm_clock_mhz: r.get(7)?,
                clocks_locked: r.get::<_, Option<i64>>(8)?.map(|l| l != 0),
                clock_lock: r.get(9)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// One stored paired-ab result (`paired_results` table): build B vs build A on
/// one shape. ratio = median per-pair B/A time; tied when the CI contains 1.0.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedResult {
    pub id: i64,
    /// Each build's commit (`-dirty`: uncommitted tracked changes); None: unknown.
    pub sha_a: Option<String>,
    pub sha_b: Option<String>,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub impl_a: String,
    pub impl_b: String,
    pub ratio: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    /// Pairs (alternating process pairs).
    pub pairs: i64,
    pub protocol_version: i64,
    pub cache_mode: String,
    /// The idle guard's mean GPU utilization (%) before the run; None: unreadable.
    pub idle_reading: Option<f64>,
    pub clocks_locked: bool,
    pub started_at: String,
    pub finished_at: String,
    /// The idle guard during the run (see [`Run::disturbed`]).
    pub disturbed: Option<bool>,
    pub disturbed_detail: Option<String>,
}

impl PairedResult {
    pub fn tied(&self) -> bool {
        self.ci_lo <= 1.0 && self.ci_hi >= 1.0
    }
    /// Above the idle guard's limit (or unreadable): not evidence for a performance claim.
    pub fn busy(&self) -> bool {
        self.idle_reading.is_none_or(|u| u > crate::sweep::IDLE_MAX_PCT)
    }
}

/// Which paired-ab results to read; every field narrows.
#[derive(Clone, Debug, Default)]
pub struct PairedResultFilter {
    /// impl_a or impl_b.
    pub impl_: Option<String>,
    pub shape: Option<(i64, i64, i64)>,
    /// sha_a or sha_b starts with this.
    pub sha: Option<String>,
}

/// Stored paired-ab results, oldest first.
pub fn paired_results(con: &Connection, f: &PairedResultFilter) -> Result<Vec<PairedResult>> {
    let shape = f.shape.map(|(m, n, k)| format!("{m}x{n}x{k}"));
    let sha = f.sha.as_ref().map(|s| format!("{}%", s.replace('%', "")));
    let mut stmt = con.prepare(
        "SELECT id, sha_a, sha_b, shape, impl_a, impl_b, ratio, ci_lo, ci_hi, n, protocol_version, cache_mode, \
                idle_reading, clocks_locked, started_at, finished_at, disturbed, disturbed_detail FROM paired_results \
         WHERE (?1 IS NULL OR impl_a = ?1 OR impl_b = ?1) AND (?2 IS NULL OR shape = ?2) \
           AND (?3 IS NULL OR sha_a LIKE ?3 OR sha_b LIKE ?3) ORDER BY started_at, id",
    )?;
    let rows = stmt
        .query_map(params![f.impl_, shape, sha], |r| {
            let shape: String = r.get(3)?;
            let dims: Vec<i64> = shape.split('x').filter_map(|d| d.parse().ok()).collect();
            let dim = |i: usize| dims.get(i).copied().unwrap_or(0);
            Ok(PairedResult {
                id: r.get(0)?,
                sha_a: r.get(1)?,
                sha_b: r.get(2)?,
                m: dim(0),
                n: dim(1),
                k: dim(2),
                impl_a: r.get(4)?,
                impl_b: r.get(5)?,
                ratio: r.get(6)?,
                ci_lo: r.get(7)?,
                ci_hi: r.get(8)?,
                pairs: r.get(9)?,
                protocol_version: r.get(10)?,
                cache_mode: r.get(11)?,
                idle_reading: r.get(12)?,
                clocks_locked: r.get::<_, i64>(13)? != 0,
                started_at: r.get(14)?,
                finished_at: r.get(15)?,
                disturbed: r.get::<_, Option<i64>>(16)?.map(|d| d != 0),
                disturbed_detail: r.get(17)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn temp_db(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("aegis-perf-view-{}-{name}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    /// A small DB through the writing path, read back through the read-only one.
    fn seeded(name: &str) -> std::path::PathBuf {
        let p = temp_db(name);
        let con = db::connect(&p).unwrap();
        con.execute_batch(
            "INSERT INTO runs (run_id, device_id, sm_arch, git_sha, started_at) VALUES
               ('r1', 'dev', 121, 'aaa', '2026-01-01T00:00:00'), ('r2', 'dev', 121, 'bbb', '2026-01-02T00:00:00');
             INSERT INTO kernels (kernel_id, op, impl, config_json, regs) VALUES
               ('k1', 'gemm', 'mma_x', '{}', 96), ('kb', 'gemm', 'cublaslt_best', '{}', NULL);
             INSERT INTO measurements (run_id, kernel_id, dtype, m, n, k, median_us, checked, pass,
                                       protocol_version, cache_mode, allow_busy) VALUES
               ('r1', 'k1', 'f16f32f16', 64, 64, 64, 10.0, 1, 1, 2, 'cold', 0),
               ('r1', 'k1', 'f16f32f16', 64, 64, 64, 9.0, 1, 1, 2, 'warm', 0),
               ('r1', 'k1', 'f16f32f16', 64, 64, 64, 11.0, 1, 1, 1, NULL, 0),
               ('r2', 'k1', 'f16f32f16', 64, 64, 64, 12.0, 1, 1, 2, 'cold', 1),
               ('r2', 'kb', 'f16f32f16', 64, 64, 64, 8.0, 0, 0, 2, 'cold', 0);
             INSERT INTO winners (sm_arch, op, dtype, shape_bucket, kernel_id, run_id, median_us, tied, tie_set,
                                  protocol_version, tie_method, ratio_ci_lo, ratio_ci_hi) VALUES
               (121, 'gemm', 'f16f32f16', '64x64x64', 'k1', 'r1', 10.0, 1, '[\"mma_x\",\"mma_y\"]', 2, 'paired', 0.99, 1.01);
             INSERT INTO paired (run_id, op, dtype, m, n, k, a_impl, b_impl, role, ratio_median, ratio_ci_lo,
                                 ratio_ci_hi, tied, protocol_version, cache_mode) VALUES
               ('r1', 'gemm', 'f16f32f16', 64, 64, 64, 'mma_x', 'mma_y', 'runner_up', 1.0, 0.99, 1.01, 1, 2, 'cold');
             INSERT INTO microbench (run_id, probe, param_json, value, unit, status, status_reason, replaced_by) VALUES
               ('r1', 'dram_bandwidth', '{}', 200.0, 'GB/s', 'replaced', NULL, 'r2'), ('r2', 'dram_bandwidth', '{}', 230.0, 'GB/s', 'valid', NULL, NULL),
               ('r2', 'ptr_chase_latency', '{}', 1.0, 'ns', 'invalidated', 'clock not recorded', NULL);",
        )
        .unwrap();
        p
    }

    #[test]
    fn connect_ro_refuses_writes() {
        let p = seeded("ro");
        let con = db::connect_ro(&p).unwrap();
        assert!(con.execute("DELETE FROM runs", []).is_err());
        assert_eq!(runs(&con).unwrap().len(), 2);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn connect_ro_never_creates() {
        let p = temp_db("missing");
        assert!(db::connect_ro(&p).is_err());
        assert!(!p.exists());
    }

    #[test]
    fn measurements_default_is_v2_cold_idle() {
        let p = seeded("meas");
        let con = db::connect_ro(&p).unwrap();
        let rows = measurements(&con, &MeasurementFilter::default()).unwrap();
        let got: Vec<(String, f64)> = rows.iter().map(|r| (r.run_id.clone(), r.median_us)).collect();
        assert_eq!(got, vec![("r1".to_string(), 10.0), ("r2".to_string(), 8.0)]);
        let v1 = MeasurementFilter { sel: Selection { protocol: 1, cache: "cold".into(), include_busy: false }, ..Default::default() };
        assert_eq!(measurements(&con, &v1).unwrap().len(), 1);
        let sha = MeasurementFilter { git_sha: Some("bbb".into()), ..Default::default() };
        assert_eq!(measurements(&con, &sha).unwrap()[0].impl_, "cublaslt_best");
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn winners_carry_ci_and_paired() {
        let p = seeded("win");
        let con = db::connect_ro(&p).unwrap();
        let w = &winners(&con, Some(121)).unwrap()[0];
        assert_eq!((w.m, w.n, w.k), (64, 64, 64));
        assert_eq!(w.kernel.impl_, "mma_x");
        assert_eq!(w.tie_set, vec!["mma_x", "mma_y"]);
        assert_eq!(w.ratio_ci, Some((0.99, 1.01)));
        assert_eq!(w.lt_ratio_ci, None);
        assert_eq!(w.paired.len(), 1);
        assert!(winners(&con, Some(87)).unwrap().is_empty());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn microbench_hides_only_invalidated() {
        let p = seeded("mb");
        let con = db::connect_ro(&p).unwrap();
        let rows = microbench(&con, 121).unwrap();
        let got: Vec<(&str, &str, MicrobenchStatus)> =
            rows.iter().map(|r| (r.run_id.as_str(), r.probe.as_str(), r.status)).collect();
        assert_eq!(
            got,
            vec![
                ("r1", "dram_bandwidth", MicrobenchStatus::Replaced),
                ("r2", "dram_bandwidth", MicrobenchStatus::Valid),
            ]
        );
        assert_eq!(rows[0].replaced_by.as_deref(), Some("r2"));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn paired_results_round_trip_and_filters() {
        let p = std::env::temp_dir().join(format!("aegis-view-pr-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        {
            let con = db::connect(&p).unwrap();
            let r = |sha_b: &str, impl_: &str, shape: (i64, i64, i64), idle: Option<f64>, lo: f64, hi: f64| {
                crate::ab::NewPairedResult {
                    sha_a: Some("2c6aabb9dd18".into()),
                    sha_b: Some(sha_b.into()),
                    shape,
                    impl_a: impl_.into(),
                    impl_b: impl_.into(),
                    ratio: (lo + hi) / 2.0,
                    ci_lo: lo,
                    ci_hi: hi,
                    n: 30,
                    protocol_version: 2,
                    cache_mode: "cold".into(),
                    idle_reading: idle,
                    clocks_locked: false,
                    started_at: "2026-10-15T00:00:00+00:00".into(),
                    finished_at: "2026-10-15T00:05:00+00:00".into(),
                    disturbed: None,
                    disturbed_detail: None,
                    profile: None,
                }
            };
            crate::ab::store(&con, &r("abc123-dirty", "mma_a", (4096, 4096, 4096), Some(2.9), 0.983, 1.016)).unwrap();
            crate::ab::store(&con, &r("def456", "mma_b", (2048, 2048, 2048), Some(0.4), 1.002, 1.014)).unwrap();
        }
        let con = db::connect_ro(&p).unwrap();
        let all = paired_results(&con, &PairedResultFilter::default()).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!((all[0].m, all[0].n, all[0].k, all[0].pairs), (4096, 4096, 4096, 30));
        assert!(all[0].tied() && all[0].busy(), "2.9% > 2%: busy");
        assert!(!all[1].tied() && !all[1].busy());
        let by_impl = PairedResultFilter { impl_: Some("mma_b".into()), ..Default::default() };
        assert_eq!(paired_results(&con, &by_impl).unwrap().len(), 1);
        let by_shape = PairedResultFilter { shape: Some((4096, 4096, 4096)), ..Default::default() };
        assert_eq!(paired_results(&con, &by_shape).unwrap()[0].sha_b.as_deref(), Some("abc123-dirty"));
        let by_sha = PairedResultFilter { sha: Some("2c6aabb".into()), ..Default::default() };
        assert_eq!(paired_results(&con, &by_sha).unwrap().len(), 2);
        std::fs::remove_file(&p).unwrap();
    }
}
