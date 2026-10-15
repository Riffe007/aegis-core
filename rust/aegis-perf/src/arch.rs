//! `emit-arch`: the measured machine parameters for one arch, as one JSON file
//! (include/aegis/arch/sm_<arch>.json) for the cost model. Every number comes
//! from the perf DB (CLAUDE.md rule 8); each carries its source runs.
//!
//! - L2 / DRAM latency: ptr_chase_latency medians over the L2-resident working
//!   sets (256 KiB..8 MiB) and the DRAM ones (>= 32 MiB), from non-invalidated
//!   rows of the selected (run, layout) pairs. **Stored in cycles only**, with
//!   each source run's SM clock at measurement and whether clocks were locked
//!   (row-level since the clock-locking probes; older runs: the run's sm_clock
//!   row, lock not recorded). ns are derived at read time
//!   (params::Latency::ns), never written here. Runs at different clocks are
//!   combined in cycles, which are clock-invariant; ns are never averaged.
//! - DRAM bandwidth: the latest dram_bandwidth medians, copy (memcpy D2D) and
//!   read (read-only 128-bit stream).
//! - MMA peak: the highest current mma_throughput row (TFLOPS, HMMA/clk/SM).
use std::path::Path;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::db;

const KIB: i64 = 1024;
const L2_SETS: (i64, i64) = (256 * KIB, 8 * 1024 * KIB);
const DRAM_MIN_SET: i64 = 32 * 1024 * KIB;
/// Source-run SM clocks within this (max / min - 1) count as one clock.
pub const CLOCK_TOLERANCE: f64 = 0.02;

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

/// (run_id, Some(layout)) pairs; a bare run id takes both layouts.
fn parse_selection(sel: &[String]) -> Vec<(String, Option<String>)> {
    sel.iter()
        .map(|s| match s.split_once(':') {
            Some((r, v)) => (r.to_string(), Some(v.to_string())),
            None => (s.clone(), None),
        })
        .collect()
}

/// One source run's pointer-chase points inside one working-set range.
#[derive(Clone, Debug)]
pub struct RunPoints {
    pub run_id: String,
    pub layout: Option<String>,
    /// SM clock at measurement (MHz).
    pub sm_clock_mhz: f64,
    /// Whether clocks were locked; None: not recorded (runs before the
    /// clock-locking probes).
    pub clocks_locked: Option<bool>,
    /// What the probe requested and got (clock_lock.cuh), if recorded.
    pub clock_lock: Option<String>,
    pub cycles: Vec<f64>,
}

const CHASE_WHERE: &str = "m.run_id = ?1 AND r.sm_arch = ?2 AND m.probe = 'ptr_chase_latency' AND m.unit = 'cycles' \
     AND m.status <> 'invalidated' AND json_extract(m.param_json, '$.stat') = 'median' \
     AND (?3 IS NULL OR json_extract(m.param_json, '$.variant') = ?3) \
     AND json_extract(m.param_json, '$.working_set_bytes') BETWEEN ?4 AND ?5";

fn run_points(con: &Connection, arch: i64, sel: &[(String, Option<String>)], lo: i64, hi: i64) -> Result<Vec<RunPoints>> {
    sel.iter()
        .map(|(run, variant)| {
            let cycles: Vec<f64> = con
                .prepare(&format!("SELECT m.value FROM microbench m JOIN runs r USING(run_id) WHERE {CHASE_WHERE}"))?
                .query_map(params![run, arch, variant, lo, hi], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            if cycles.is_empty() {
                bail!(
                    "no ptr_chase_latency cycles rows (invalidated excluded) for run {run} ({}) on sm_{arch}",
                    variant.as_deref().unwrap_or("both layouts")
                );
            }
            // Row-level clock (clock-locking probes), else the run's sm_clock row.
            let (row_mhz, locked, detail): (Option<f64>, Option<i64>, Option<String>) = con.query_row(
                &format!(
                    "SELECT avg(m.sm_clock_mhz), min(m.clocks_locked), max(m.clock_lock) \
                     FROM microbench m JOIN runs r USING(run_id) WHERE {CHASE_WHERE}"
                ),
                params![run, arch, variant, lo, hi],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?;
            let mhz = match row_mhz {
                Some(m) => m,
                None => con
                    .query_row(
                        "SELECT value FROM microbench WHERE run_id = ? AND probe = 'sm_clock' AND status <> 'invalidated'",
                        [run],
                        |x| x.get(0),
                    )
                    .optional()?
                    .with_context(|| format!("run {run} has no SM clock: its latencies cannot be tied to a clock"))?,
            };
            Ok(RunPoints {
                run_id: run.clone(),
                layout: variant.clone(),
                sm_clock_mhz: mhz,
                clocks_locked: locked.map(|l| l != 0),
                clock_lock: detail,
                cycles,
            })
        })
        .collect()
}

/// The latency entry from per-run points, in cycles: the median of the cycle
/// points across runs (cycles are clock-invariant, so runs at different clocks
/// combine). Every source run's clock and lock state is recorded.
/// ref_clock_mhz is the clock readers state derived ns at: the locked clock
/// when every source run was locked at one clock (median of their clocks),
/// else the highest source-run clock (the least-throttled run).
pub fn latency_value(runs: &[RunPoints], sets: &str) -> Value {
    let cycles: Vec<f64> = runs.iter().flat_map(|r| r.cycles.iter().copied()).collect();
    let (cmin, cmax) = cycles.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    let (fmin, fmax) = runs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| {
        (a.min(r.sm_clock_mhz), b.max(r.sm_clock_mhz))
    });
    let spread = if fmin > 0.0 { fmax / fmin - 1.0 } else { f64::INFINITY };
    let all_locked = !runs.is_empty() && runs.iter().all(|r| r.clocks_locked == Some(true));
    let one_clock = spread <= CLOCK_TOLERANCE;
    let (ref_mhz, ref_rule) = if all_locked && one_clock {
        (median(runs.iter().map(|r| r.sm_clock_mhz).collect()).unwrap_or(fmax), "the locked clock of the source runs")
    } else {
        (fmax, "the highest source-run SM clock (clocks not locked in every source run)")
    };
    json!({
        "cycles": median(cycles.clone()),
        "cycles_min": cmin,
        "cycles_max": cmax,
        "clocks_locked": all_locked,
        "ref_clock_mhz": ref_mhz,
        "ref_clock_rule": ref_rule,
        "clock_spread_pct": spread * 100.0,
        "combine": if one_clock {
            format!("source-run clocks within {:.0}%", CLOCK_TOLERANCE * 100.0)
        } else {
            format!("source-run clocks differ by {:.1}% (> {:.0}%): combined in cycles only", spread * 100.0,
                    CLOCK_TOLERANCE * 100.0)
        },
        "points": cycles.len(),
        "working_sets": sets,
        "method": "dependent-load pointer chase, one thread, 128 B node stride, median of 5 repeats per point; \
                   cycles = median across points and runs; ns are derived by readers (cycles / clock)",
        "runs": runs.iter().map(|r| json!({
            "run_id": r.run_id,
            "layout": r.layout.as_deref().unwrap_or("both"),
            "sm_clock_mhz": r.sm_clock_mhz,
            "clocks_locked": r.clocks_locked,
            "clock_lock": r.clock_lock,
            "cycles": median(r.cycles.clone()),
            "points": r.cycles.len(),
        })).collect::<Vec<_>>(),
    })
}

pub fn cmd_emit_arch(db_path: &Path, arch: i64, ptr_chase: &[String], results: &Path, out: Option<&Path>) -> Result<()> {
    let con = db::connect(db_path)?;

    // Pointer-chase runs: given (replaced runs allowed; invalidated rows are
    // never read), or the newest valid run (db::current_run).
    let sel = if ptr_chase.is_empty() {
        db::current_run(&con, arch, "ptr_chase_latency")?.map(|r| (r, None)).into_iter().collect()
    } else {
        parse_selection(ptr_chase)
    };
    if sel.is_empty() {
        bail!("no ptr_chase_latency runs for sm_{arch}");
    }
    let l2 = latency_value(&run_points(&con, arch, &sel, L2_SETS.0, L2_SETS.1)?, "256 KiB - 8 MiB");
    let dram = latency_value(&run_points(&con, arch, &sel, DRAM_MIN_SET, i64::MAX)?, ">= 32 MiB");

    // DRAM bandwidth, per stream: the median and reps of the newest valid run.
    let bw_run = db::current_run(&con, arch, "dram_bandwidth")?.context("no dram_bandwidth run; run microbench")?;
    let bandwidth = |method: &str, desc: &str| -> Result<Value> {
        let (value, run): (f64, String) = con
            .query_row(
                "SELECT m.value, m.run_id FROM microbench m \
                 WHERE m.probe = 'dram_bandwidth' AND m.status <> 'invalidated' AND m.run_id = ?1 \
                   AND json_extract(m.param_json, '$.method') = ?2 \
                   AND json_extract(m.param_json, '$.stat') = 'median'",
                params![bw_run, method],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .with_context(|| format!("no dram_bandwidth {method} row; run build/microbench/mb_dram_bw and ingest"))?;
        let reps: Vec<f64> = con
            .prepare(
                "SELECT value FROM microbench WHERE run_id = ?1 AND probe = 'dram_bandwidth' \
                 AND json_extract(param_json, '$.method') = ?2 AND json_extract(param_json, '$.rep') IS NOT NULL",
            )?
            .query_map(params![run, method], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(json!({"value": value, "reps": reps, "method": desc,
                  "source": {"probe": "dram_bandwidth", "stream": method, "run_id": run}}))
    };
    let copy_bw = bandwidth(
        "memcpy_d2d",
        "cudaMemcpyAsync D2D, 512 MiB, (read + write bytes) / time; median of 20 copies per rep, median of reps",
    )?;
    let read_bw = bandwidth(
        "read_v4",
        "read-only stream, 4 independent 128-bit loads per thread per iteration (SASS-proven), 512 MiB, \
         bytes read / time; median of 20 runs per rep, median of reps",
    )?;

    // MMA peak: the highest point of the newest valid mma_throughput run.
    let mma_cur = db::current_run(&con, arch, "mma_throughput")?.context("no mma_throughput run; run microbench")?;
    let (tflops, mma_param, mma_run): (f64, String, String) = con
        .query_row(
            "SELECT m.value, m.param_json, m.run_id FROM microbench m \
             WHERE m.probe = 'mma_throughput' AND m.unit = 'TFLOPS' AND m.status <> 'invalidated' \
               AND m.run_id = ? ORDER BY m.value DESC LIMIT 1",
            [&mma_cur],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .context("no mma_throughput rows")?;
    let hmma: Option<f64> = con
        .query_row(
            "SELECT value FROM microbench WHERE run_id = ? AND probe = 'mma_throughput' AND unit = 'HMMA/clk/SM' \
             AND param_json = ?",
            params![mma_run, mma_param],
            |r| r.get(0),
        )
        .ok();

    // Device facts from the manifest (aegis_manifest; rule 8 allows it).
    let manifest_path = results.join(format!("manifest-{}.json", db::hostname()));
    let manifest: Value = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null);

    let doc = json!({
        "schema": "aegis-arch-params/3",
        "sm_arch": arch,
        "device": manifest.get("gpu_name"),
        "sm_count": manifest.get("sm_count"),
        "l2_bytes": manifest.get("l2_bytes"),
        "generated_by": format!("aegis-perf emit-arch on {}, {}", db::hostname(),
                                chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")),
        "note": "Measured on this device (CLAUDE.md rule 8). Regenerate with aegis-perf emit-arch after re-running \
                 the probes; do not edit by hand.",
        "l2_latency": with_probe(l2),
        "dram_latency": with_probe(dram),
        "copy_bw_gbps": copy_bw,
        "read_bw_gbps": read_bw,
        "roof_rule": "memory roof: read_bw_gbps for GEMV and decode shapes (M <= 16), copy_bw_gbps elsewhere",
        "mma_peak": {
            "tflops": tflops,
            "hmma_per_clk_per_sm": hmma,
            "instruction": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            "at": serde_json::from_str::<Value>(&mma_param).unwrap_or(Value::Null),
            "source": {"probe": "mma_throughput", "run_id": mma_run},
        },
    });
    let path = out.map(Path::to_path_buf).unwrap_or_else(|| format!("include/aegis/arch/sm_{arch}.json").into());
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(&doc)? + "\n")?;
    println!("wrote {}", path.display());
    Ok(())
}

fn with_probe(mut v: Value) -> Value {
    v["probe"] = json!("ptr_chase_latency");
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: &str, mhz: f64, locked: Option<bool>, cycles: &[f64]) -> RunPoints {
        RunPoints {
            run_id: id.into(),
            layout: None,
            sm_clock_mhz: mhz,
            clocks_locked: locked,
            clock_lock: None,
            cycles: cycles.to_vec(),
        }
    }

    #[test]
    fn stores_cycles_and_clocks_never_ns() {
        let v = latency_value(&[run("a", 2557.0, None, &[338.0, 340.0]), run("b", 2540.0, None, &[336.0])], "x");
        assert_eq!(v["cycles"].as_f64(), Some(338.0));
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert!(keys.iter().all(|k| *k != "ns" && !k.starts_with("ns_") && !k.ends_with("_ns")), "no ns field: {keys:?}");
        assert!(v["runs"][0].get("ns").is_none(), "no per-run ns");
        assert_eq!(v["runs"][0]["sm_clock_mhz"].as_f64(), Some(2557.0));
        assert!(v["runs"][0]["clocks_locked"].is_null(), "lock not recorded on old runs");
        assert_eq!(v["clocks_locked"].as_bool(), Some(false));
        assert_eq!(v["ref_clock_mhz"].as_f64(), Some(2557.0), "unlocked: highest source-run clock");
    }

    #[test]
    fn runs_at_different_clocks_combine_in_cycles() {
        // 2557 vs 2407 MHz: 6.2% apart. Same cycles; each run keeps its clock.
        let v = latency_value(&[run("fast", 2557.0, None, &[338.0, 338.0]), run("slow", 2407.0, None, &[335.0])], "x");
        assert_eq!(v["cycles"].as_f64(), Some(338.0));
        assert!(v["clock_spread_pct"].as_f64().unwrap() > 6.0);
        assert!(v["combine"].as_str().unwrap().contains("cycles only"));
        assert_eq!(v["runs"][1]["sm_clock_mhz"].as_f64(), Some(2407.0));
    }

    #[test]
    fn locked_runs_state_the_locked_clock() {
        let v = latency_value(&[run("a", 2418.0, Some(true), &[335.0]), run("b", 2420.0, Some(true), &[336.0])], "x");
        assert_eq!(v["clocks_locked"].as_bool(), Some(true));
        assert_eq!(v["ref_clock_mhz"].as_f64(), Some(2419.0));
        let mixed = latency_value(&[run("a", 2418.0, Some(true), &[335.0]), run("b", 2557.0, Some(false), &[336.0])], "x");
        assert_eq!(mixed["clocks_locked"].as_bool(), Some(false));
        assert_eq!(mixed["ref_clock_mhz"].as_f64(), Some(2557.0));
    }
}
