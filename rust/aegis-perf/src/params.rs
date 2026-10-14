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
