//! Measured machine parameters (include/aegis/arch/sm_<arch>.json, written by
//! `emit-arch`) and the roof metrics derived from them at report time.
//!
//! Nothing derived from a roof is stored in the perf DB: % of bandwidth, % of
//! mma peak and roofline position are computed here, from the arch file, every
//! time a report runs, so a re-measured roof updates every number at once.
//!
//! Memory roof per shape: the read-only stream bandwidth for GEMV and decode
//! shapes (M <= DECODE_MAX_M: the traffic is almost all reads of B), the copy
//! bandwidth elsewhere.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

/// Largest M treated as GEMV / decode for the memory roof.
pub const DECODE_MAX_M: f64 = 16.0;

#[derive(Clone, Debug)]
pub struct ArchParams {
    pub path: PathBuf,
    pub copy_bw_gbps: f64,
    pub read_bw_gbps: f64,
    pub peak_tflops: f64,
    /// Pointer-chase latencies (schema aegis-arch-params/2); None in older files.
    pub l2_latency: Option<Latency>,
    pub dram_latency: Option<Latency>,
}

/// A latency in SM cycles (clock-invariant, the primary value), with the
/// reference clock its derived ns were stated at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Latency {
    pub cycles: f64,
    pub ref_clock_mhz: f64,
    pub ns_at_ref: f64,
}

impl Latency {
    /// The latency in ns at a given SM clock.
    pub fn ns_at(&self, mhz: f64) -> f64 {
        self.cycles / mhz * 1e3
    }

    fn parse(v: &Value) -> Option<Latency> {
        Some(Latency {
            cycles: v["cycles"].as_f64()?,
            ref_clock_mhz: v["ref_clock_mhz"].as_f64()?,
            ns_at_ref: v["ns_at_ref"].as_f64()?,
        })
    }
}

pub fn arch_file(arch: i64) -> PathBuf {
    PathBuf::from(format!("include/aegis/arch/sm_{arch}.json"))
}

pub fn load(path: &Path) -> Result<ArchParams> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(ArchParams {
        path: path.to_path_buf(),
        copy_bw_gbps: v["copy_bw_gbps"]["value"].as_f64().context("copy_bw_gbps.value")?,
        read_bw_gbps: v["read_bw_gbps"]["value"].as_f64().context("read_bw_gbps.value")?,
        peak_tflops: v["mma_peak"]["tflops"].as_f64().context("mma_peak.tflops")?,
        l2_latency: Latency::parse(&v["l2_latency"]),
        dram_latency: Latency::parse(&v["dram_latency"]),
    })
}

/// Which memory roof a shape is held to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BwKind {
    Read,
    Copy,
}

impl BwKind {
    pub fn name(self) -> &'static str {
        match self {
            BwKind::Read => "read",
            BwKind::Copy => "copy",
        }
    }
}

/// Roof metrics of one GEMM measurement (compulsory traffic: A, B read once,
/// C written once, FP16).
#[derive(Clone, Copy, Debug)]
pub struct Roof {
    pub ai: f64, // FLOP / byte
    pub tflops: f64,
    pub gbps: f64,
    pub bw: BwKind,
    pub pct_peak: f64, // of the mma.sync peak
    pub pct_bw: f64,   // of the shape's memory roof
    pub pct_roof: f64, // of min(bw x AI, peak)
    pub memory_bound: bool,
}

impl ArchParams {
    pub fn bw_kind(&self, m: f64) -> BwKind {
        if m <= DECODE_MAX_M { BwKind::Read } else { BwKind::Copy }
    }

    pub fn bw(&self, kind: BwKind) -> f64 {
        match kind {
            BwKind::Read => self.read_bw_gbps,
            BwKind::Copy => self.copy_bw_gbps,
        }
    }

    /// FLOP/B where the given memory roof meets the compute roof.
    pub fn ridge(&self, kind: BwKind) -> f64 {
        self.peak_tflops * 1e3 / self.bw(kind)
    }

    pub fn roof(&self, m: f64, n: f64, k: f64, median_us: f64) -> Roof {
        let flops = 2.0 * m * n * k;
        let bytes = 2.0 * (m * k + k * n + m * n);
        let ai = flops / bytes;
        let tflops = flops / (median_us * 1e-6) / 1e12;
        let gbps = bytes / (median_us * 1e-6) / 1e9;
        let kind = self.bw_kind(m);
        let attainable = (self.bw(kind) * ai / 1e3).min(self.peak_tflops);
        Roof {
            ai,
            tflops,
            gbps,
            bw: kind,
            pct_peak: tflops / self.peak_tflops * 100.0,
            pct_bw: gbps / self.bw(kind) * 100.0,
            pct_roof: tflops / attainable * 100.0,
            memory_bound: ai < self.ridge(kind),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{latency_value, RunPoints};

    #[test]
    fn loads_cycle_latencies_and_converts_at_any_clock() {
        let runs = [RunPoints {
            run_id: "r".into(),
            layout: None,
            sm_clock_mhz: 2407.0,
            cycles: vec![335.0],
            ns: vec![139.2],
        }];
        let mut l2 = latency_value(&runs, "x");
        l2["probe"] = serde_json::json!("ptr_chase_latency");
        let doc = serde_json::json!({
            "schema": "aegis-arch-params/2",
            "copy_bw_gbps": {"value": 236.7}, "read_bw_gbps": {"value": 257.8},
            "mma_peak": {"tflops": 125.6}, "l2_latency": l2,
        });
        let dir = std::env::temp_dir().join(format!("aegis-params-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sm_0.json");
        std::fs::write(&path, doc.to_string()).unwrap();
        let p = load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let l = p.l2_latency.expect("l2 latency");
        assert_eq!(l.cycles, 335.0);
        assert_eq!(l.ref_clock_mhz, 2407.0);
        assert!((l.ns_at_ref - 335.0 / 2407.0 * 1e3).abs() < 1e-9);
        assert!((l.ns_at(2557.0) - 131.01).abs() < 0.01);
        assert!(p.dram_latency.is_none(), "missing entry -> None");
    }
}
