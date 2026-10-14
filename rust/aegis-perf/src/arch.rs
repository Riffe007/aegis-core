//! `emit-arch`: the measured machine parameters for one arch, as one JSON file
//! (include/aegis/arch/sm_<arch>.json) for the cost model. Every number comes
//! from the perf DB (CLAUDE.md rule 8); each carries its source runs.
//!
//! - L2 / DRAM latency: ptr_chase_latency medians (ns and cycles) over the
//!   L2-resident working sets (256 KiB..8 MiB) and the DRAM ones (>= 32 MiB),
//!   from the selected (run, layout) pairs; the median across them.
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

/// Pointer-chase medians for the selected runs, in `unit`, with the working
/// set inside [lo, hi].
fn chase(con: &Connection, arch: i64, sel: &[(String, Option<String>)], unit: &str, lo: i64, hi: i64) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for (run, variant) in sel {
        let mut stmt = con.prepare(
            "SELECT m.value FROM microbench m JOIN runs r USING(run_id) \
             WHERE m.run_id = ?1 AND r.sm_arch = ?2 AND m.probe = 'ptr_chase_latency' AND m.unit = ?3 \
               AND json_extract(m.param_json, '$.stat') = 'median' \
               AND (?4 IS NULL OR json_extract(m.param_json, '$.variant') = ?4) \
               AND json_extract(m.param_json, '$.working_set_bytes') BETWEEN ?5 AND ?6",
        )?;
        let vals: Vec<f64> = stmt
            .query_map(params![run, arch, unit, variant, lo, hi], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        if vals.is_empty() {
            bail!("no ptr_chase_latency rows for run {run} ({}) on sm_{arch}", variant.as_deref().unwrap_or("both layouts"));
        }
        out.extend(vals);
    }
    Ok(out)
}

fn latency(con: &Connection, arch: i64, sel: &[(String, Option<String>)], lo: i64, hi: i64, sets: &str) -> Result<Value> {
    let ns = chase(con, arch, sel, "ns", lo, hi)?;
    let cycles = chase(con, arch, sel, "cycles", lo, hi)?;
    let (min, max) = ns.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    Ok(json!({
        "ns": median(ns.clone()),
        "cycles": median(cycles),
        "ns_min": min,
        "ns_max": max,
        "points": ns.len(),
        "working_sets": sets,
        "method": "dependent-load pointer chase, one thread, 128 B node stride, median of 5 repeats per point; \
                   the value is the median across points and runs",
    }))
}

pub fn cmd_emit_arch(db_path: &Path, arch: i64, ptr_chase: &[String], results: &Path, out: Option<&Path>) -> Result<()> {
    let con = db::connect(db_path)?;

    // Pointer-chase runs: given, or every current (non-superseded) one.
    let sel = if ptr_chase.is_empty() {
        let mut stmt = con.prepare(
            "SELECT DISTINCT m.run_id FROM microbench m JOIN runs r USING(run_id) \
             WHERE m.probe = 'ptr_chase_latency' AND m.superseded = 0 AND r.sm_arch = ?",
        )?;
        let runs: Vec<String> = stmt.query_map([arch], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        runs.into_iter().map(|r| (r, None)).collect()
    } else {
        parse_selection(ptr_chase)
    };
    if sel.is_empty() {
        bail!("no ptr_chase_latency runs for sm_{arch}");
    }
    let l2 = latency(&con, arch, &sel, L2_SETS.0, L2_SETS.1, "256 KiB - 8 MiB")?;
    let dram = latency(&con, arch, &sel, DRAM_MIN_SET, i64::MAX, ">= 32 MiB")?;
    let clocks: Vec<Value> = sel
        .iter()
        .map(|(r, _)| {
            let mhz: Option<f64> = con
                .query_row("SELECT value FROM microbench WHERE run_id = ? AND probe = 'sm_clock'", [r], |x| x.get(0))
                .ok();
            json!({"run_id": r, "sm_clock_mhz": mhz})
        })
        .collect();

    // DRAM bandwidth, per stream: the latest current median and its reps.
    let bandwidth = |method: &str, desc: &str| -> Result<Value> {
        let (value, run): (f64, String) = con
            .query_row(
                "SELECT m.value, m.run_id FROM microbench m JOIN runs r USING(run_id) \
                 WHERE m.probe = 'dram_bandwidth' AND m.superseded = 0 AND r.sm_arch = ?1 \
                   AND json_extract(m.param_json, '$.method') = ?2 \
                   AND json_extract(m.param_json, '$.stat') = 'median' ORDER BY r.started_at DESC LIMIT 1",
                params![arch, method],
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

    // MMA peak: the highest current mma_throughput point.
    let (tflops, mma_param, mma_run): (f64, String, String) = con
        .query_row(
            "SELECT m.value, m.param_json, m.run_id FROM microbench m JOIN runs r USING(run_id) \
             WHERE m.probe = 'mma_throughput' AND m.unit = 'TFLOPS' AND m.superseded = 0 AND r.sm_arch = ? \
             ORDER BY m.value DESC LIMIT 1",
            [arch],
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
        "schema": "aegis-arch-params/1",
        "sm_arch": arch,
        "device": manifest.get("gpu_name"),
        "sm_count": manifest.get("sm_count"),
        "l2_bytes": manifest.get("l2_bytes"),
        "generated_by": format!("aegis-perf emit-arch on {}, {}", db::hostname(),
                                chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")),
        "note": "Measured on this device (CLAUDE.md rule 8). Regenerate with aegis-perf emit-arch after re-running \
                 the probes; do not edit by hand.",
        "l2_latency": { "value": l2, "source": {"probe": "ptr_chase_latency", "runs": sel_json(&sel), "clocks": clocks} },
        "dram_latency": { "value": dram, "source": {"probe": "ptr_chase_latency", "runs": sel_json(&sel)} },
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

fn sel_json(sel: &[(String, Option<String>)]) -> Value {
    Value::Array(
        sel.iter()
            .map(|(r, v)| json!({"run_id": r, "layout": v.as_deref().unwrap_or("both")}))
            .collect(),
    )
}
