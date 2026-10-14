//! Measured machine parameters (include/aegis/arch/sm_<arch>.json, written by
//! `emit-arch`) and the roof metrics derived from them at report time.
//!
//! Nothing derived from a roof is stored in the perf DB: % of bandwidth, % of
//! mma peak and roofline position are computed here, from the arch file, every
//! time a report runs, so a re-measured roof updates every number at once.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ArchParams {
    pub path: PathBuf,
    pub bw_gbps: f64,
    pub peak_tflops: f64,
}

pub fn arch_file(arch: i64) -> PathBuf {
    PathBuf::from(format!("include/aegis/arch/sm_{arch}.json"))
}

pub fn load(path: &Path) -> Result<ArchParams> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))?;
    let bw = v["dram_bandwidth_gbps"]["value"].as_f64().context("dram_bandwidth_gbps.value")?;
    let peak = v["mma_peak"]["tflops"].as_f64().context("mma_peak.tflops")?;
    Ok(ArchParams { path: path.to_path_buf(), bw_gbps: bw, peak_tflops: peak })
}

/// Roof metrics of one GEMM measurement (compulsory traffic: A, B read once,
/// C written once, FP16).
#[derive(Clone, Copy, Debug)]
pub struct Roof {
    pub ai: f64,          // FLOP / byte
    pub tflops: f64,
    pub gbps: f64,
    pub pct_peak: f64,    // of the mma.sync peak
    pub pct_bw: f64,      // of DRAM bandwidth
    pub pct_roof: f64,    // of min(bw x AI, peak)
    pub memory_bound: bool,
}

impl ArchParams {
    /// FLOP/B where the bandwidth and compute roofs meet.
    pub fn ridge(&self) -> f64 {
        self.peak_tflops * 1e3 / self.bw_gbps
    }

    pub fn roof(&self, m: f64, n: f64, k: f64, median_us: f64) -> Roof {
        let flops = 2.0 * m * n * k;
        let bytes = 2.0 * (m * k + k * n + m * n);
        let ai = flops / bytes;
        let tflops = flops / (median_us * 1e-6) / 1e12;
        let gbps = bytes / (median_us * 1e-6) / 1e9;
        let attainable = (self.bw_gbps * ai / 1e3).min(self.peak_tflops);
        Roof {
            ai,
            tflops,
            gbps,
            pct_peak: tflops / self.peak_tflops * 100.0,
            pct_bw: gbps / self.bw_gbps * 100.0,
            pct_roof: tflops / attainable * 100.0,
            memory_bound: ai < self.ridge(),
        }
    }
}
