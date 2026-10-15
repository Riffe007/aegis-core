//! `emit-arch`: the measured machine parameters for one arch, as one JSON file
//! (include/aegis/arch/sm_<arch>.json) for the cost model. Every number comes
//! from the perf DB (CLAUDE.md rule 8); each carries its source runs.
//!
//! - L2 / DRAM latency: ptr_chase_latency medians over the L2-resident working
//!   sets (256 KiB..8 MiB) and the DRAM ones (>= 32 MiB), from non-invalidated
//!   rows of the selected (run, layout) pairs. **Cycles are the primary value**
//!   (clock-invariant); each source run's SM clock (sm_clock probe) is recorded,
//!   and ns appears only derived, at a stated reference clock. When the runs'
//!   clocks agree within CLOCK_TOLERANCE the measured ns are summarized too;
//!   when they differ by more, measured ns are not averaged across runs.
//! - DRAM bandwidth: the latest dram_bandwidth medians, copy (memcpy D2D) and
//!   read (read-only 128-bit stream).
//! - MMA peak: the highest current mma_throughput row (TFLOPS, HMMA/clk/SM).
use std::path::Path;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::db;

const KIB: i64 = 1024;
const L2_SETS: (i64, i64) = (256 * KIB, 8 * 1024 * KIB);
const DRAM_MIN_SET: i64 = 32 * 1024 * KIB;
/// Source runs whose SM clocks differ by more than this (max / min - 1) are
/// combined in cycles only.
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
    pub sm_clock_mhz: f64,
    pub cycles: Vec<f64>,
    pub ns: Vec<f64>,
}

/// Non-invalidated pointer-chase medians of one (run, layout) in `unit`, with
/// the working set inside [lo, hi].
fn chase(con: &Connection, arch: i64, run: &str, variant: Option<&str>, unit: &str, lo: i64, hi: i64) -> Result<Vec<f64>> {
    let mut stmt = con.prepare(
        "SELECT m.value FROM microbench m JOIN runs r USING(run_id) \
         WHERE m.run_id = ?1 AND r.sm_arch = ?2 AND m.probe = 'ptr_chase_latency' AND m.unit = ?3 \
           AND m.status <> 'invalidated' \
           AND json_extract(m.param_json, '$.stat') = 'median' \
           AND (?4 IS NULL OR json_extract(m.param_json, '$.variant') = ?4) \
           AND json_extract(m.param_json, '$.working_set_bytes') BETWEEN ?5 AND ?6",
    )?;
    let vals: Vec<f64> = stmt
        .query_map(params![run, arch, unit, variant, lo, hi], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    if vals.is_empty() {
        bail!(
            "no ptr_chase_latency rows (invalidated excluded) for run {run} ({}) on sm_{arch}",
            variant.unwrap_or("both layouts")
        );
    }
    Ok(vals)
}

fn run_points(con: &Connection, arch: i64, sel: &[(String, Option<String>)], lo: i64, hi: i64) -> Result<Vec<RunPoints>> {
    sel.iter()
        .map(|(run, variant)| {
            let mhz: f64 = con
                .query_row("SELECT value FROM microbench WHERE run_id = ? AND probe = 'sm_clock'", [run], |x| x.get(0))
                .with_context(|| format!("run {run} has no sm_clock row: latencies cannot be tied to a clock"))?;
            Ok(RunPoints {
                run_id: run.clone(),
                layout: variant.clone(),
                sm_clock_mhz: mhz,
                cycles: chase(con, arch, run, variant.as_deref(), "cycles", lo, hi)?,
                ns: chase(con, arch, run, variant.as_deref(), "ns", lo, hi)?,
            })
        })
        .collect()
}

/// The latency entry from per-run points. Primary value: the median of the
/// cycle points across runs. Reference clock: the highest source-run clock
/// (the least-throttled run); ns_at_ref = cycles / ref MHz. Clocks within
/// CLOCK_TOLERANCE: measured ns are summarized as well ("ns_measured");
/// otherwise they are not combined (each run keeps its own median ns).
pub fn latency_value(runs: &[RunPoints], sets: &str) -> Value {
    let cycles: Vec<f64> = runs.iter().flat_map(|r| r.cycles.iter().copied()).collect();
    let (cmin, cmax) = cycles.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    let cyc = median(cycles.clone());
    let (fmin, fmax) = runs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| {
        (a.min(r.sm_clock_mhz), b.max(r.sm_clock_mhz))
    });
    let spread = if fmin > 0.0 { fmax / fmin - 1.0 } else { f64::INFINITY };
    let consistent = spread <= CLOCK_TOLERANCE;
    let ns_measured = if consistent {
        let ns: Vec<f64> = runs.iter().flat_map(|r| r.ns.iter().copied()).collect();
        let (lo, hi) = ns.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        json!({"median": median(ns), "min": lo, "max": hi})
    } else {
        Value::Null
    };
    json!({
        "cycles": cyc,
        "cycles_min": cmin,
        "cycles_max": cmax,
        "ref_clock_mhz": fmax,
        "ns_at_ref": cyc.map(|c| c / fmax * 1e3),
        "ns_measured": ns_measured,
        "clock_spread_pct": spread * 100.0,
        "combine": if consistent {
            format!("clocks within {:.0}%: cycles and measured ns both combined", CLOCK_TOLERANCE * 100.0)
        } else {
            format!("clocks differ by {:.1}% (> {:.0}%): combined in cycles only; measured ns per run",
                    spread * 100.0, CLOCK_TOLERANCE * 100.0)
        },
        "points": cycles.len(),
        "working_sets": sets,
        "method": "dependent-load pointer chase, one thread, 128 B node stride, median of 5 repeats per point; \
                   cycles = median across points and runs; ns_at_ref = cycles / ref_clock_mhz, ref = the highest \
                   source-run SM clock",
        "runs": runs.iter().map(|r| json!({
            "run_id": r.run_id,
            "layout": r.layout.as_deref().unwrap_or("both"),
            "sm_clock_mhz": r.sm_clock_mhz,
            "cycles": median(r.cycles.clone()),
            "ns": median(r.ns.clone()),
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
        "schema": "aegis-arch-params/2",
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

    fn run(id: &str, mhz: f64, cycles: &[f64]) -> RunPoints {
        RunPoints {
            run_id: id.into(),
            layout: None,
            sm_clock_mhz: mhz,
            cycles: cycles.to_vec(),
            ns: cycles.iter().map(|c| c / mhz * 1e3).collect(),
        }
    }

    #[test]
    fn clocks_within_tolerance_combine_ns_too() {
        let v = latency_value(&[run("a", 2557.0, &[338.0, 340.0]), run("b", 2540.0, &[336.0])], "x");
        assert_eq!(v["cycles"].as_f64(), Some(338.0));
        assert_eq!(v["ref_clock_mhz"].as_f64(), Some(2557.0));
        assert!((v["ns_at_ref"].as_f64().unwrap() - 338.0 / 2557.0 * 1e3).abs() < 1e-9);
        assert!(v["ns_measured"]["median"].as_f64().is_some());
        assert_eq!(v["runs"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn clocks_beyond_tolerance_use_cycles_only() {
        // 2557 vs 2407 MHz: 6.2% apart. Same cycles, different ns.
        let v = latency_value(&[run("fast", 2557.0, &[338.0, 338.0]), run("slow", 2407.0, &[335.0])], "x");
        assert!(v["ns_measured"].is_null(), "ns must not be averaged across clocks");
        assert_eq!(v["cycles"].as_f64(), Some(338.0));
        assert_eq!(v["ref_clock_mhz"].as_f64(), Some(2557.0));
        assert!(v["clock_spread_pct"].as_f64().unwrap() > 6.0);
        let runs = v["runs"].as_array().unwrap();
        assert_eq!(runs[1]["sm_clock_mhz"].as_f64(), Some(2407.0));
        assert!(runs[1]["ns"].as_f64().unwrap() > runs[0]["ns"].as_f64().unwrap());
    }
}
