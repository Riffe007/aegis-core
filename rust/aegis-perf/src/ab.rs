//! `paired-ab`: compare the same impl built from two trees (e.g. before and
//! after a kernel change) with a paired test across processes.
//!
//! Runs `A/bench_gemm` and `B/bench_gemm` alternately (A, B, A, B, ...) on one
//! impl and shape, `pairs` times; each run is one protocol v2 measurement (cold
//! by default, `--reps` windows). ratio = B median / A median per pair; the 95%
//! CI of the median ratio is a percentile bootstrap (10,000 resamples, fixed
//! seed). In-process pairing (aegis_tune's paired test) is impossible across
//! builds; alternating processes keeps slow drifts (clocks, thermals,
//! background load) out of the ratio.
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

pub struct AbArgs<'a> {
    pub a: &'a Path,
    pub b: &'a Path,
    pub impl_a: &'a str,
    pub impl_b: &'a str,
    pub shape: (i64, i64, i64),
    pub pairs: usize,
    pub reps: u32,
    pub cache: &'a str,
    /// Run even if the idle guard finds the GPU busy (the result is marked).
    pub allow_busy: bool,
}

fn run_once(bench: &Path, impl_: &str, (m, n, k): (i64, i64, i64), reps: u32, cache: &str) -> Result<f64> {
    let out = crate::gpujob::command(bench)?
        .args(["--impl", impl_, "--m", &m.to_string(), "--n", &n.to_string(), "--k", &k.to_string()])
        .args(["--reps", &reps.to_string(), "--warmup", "3", "--cache", cache])
        .output()
        .with_context(|| format!("run {}", bench.display()))?;
    if !out.status.success() {
        bail!("{} failed: {}", bench.display(), String::from_utf8_lossy(&out.stderr));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let v: Value = serde_json::from_str(line.trim().lines().last().unwrap_or(""))?;
    v["median_us"].as_f64().context("bench_gemm row without median_us")
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// splitmix64: a small deterministic generator for the bootstrap.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Median ratio and its 95% percentile-bootstrap CI.
pub fn bootstrap(ratios: &[f64], resamples: usize, seed: u64) -> (f64, f64, f64) {
    let mut r = ratios.to_vec();
    let med = median(&mut r);
    let mut rng = Rng(seed);
    let mut meds = Vec::with_capacity(resamples);
    let mut s = vec![0.0; ratios.len()];
    for _ in 0..resamples {
        for x in s.iter_mut() {
            *x = ratios[(rng.next() % ratios.len() as u64) as usize];
        }
        meds.push(median(&mut s));
    }
    meds.sort_by(f64::total_cmp);
    let at = |p: f64| meds[((p * (meds.len() - 1) as f64).round()) as usize];
    (med, at(0.025), at(0.975))
}

pub fn cmd_paired_ab(a: &AbArgs) -> Result<()> {
    let _gpu = crate::gpujob::lock(&format!("aegis-perf paired-ab {} vs {}", a.impl_a, a.impl_b))?;
    // Blocking idle guard (Benchmark policy), with build B's bench_gemm.
    let idle = crate::sweep::idle_guard(a.b, a.allow_busy)?;
    let (mut ta, mut tb, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..a.pairs {
        // Alternate which side runs first so neither always follows the other.
        let (x, y) = if i % 2 == 0 {
            let x = run_once(a.a, a.impl_a, a.shape, a.reps, a.cache)?;
            (x, run_once(a.b, a.impl_b, a.shape, a.reps, a.cache)?)
        } else {
            let y = run_once(a.b, a.impl_b, a.shape, a.reps, a.cache)?;
            (run_once(a.a, a.impl_a, a.shape, a.reps, a.cache)?, y)
        };
        ta.push(x);
        tb.push(y);
        ratios.push(y / x);
    }
    let (med, lo, hi) = bootstrap(&ratios, 10_000, 0x5eed);
    let (m, n, k) = a.shape;
    println!(
        "paired-ab {m}x{n}x{k} {} ({}): A {} median {:.2} us, B {} median {:.2} us, B/A {:.4} [{:.4}, {:.4}] over {} pairs{}",
        a.impl_a,
        a.cache,
        a.a.display(),
        median(&mut ta),
        a.b.display(),
        median(&mut tb),
        med,
        lo,
        hi,
        a.pairs,
        if lo <= 1.0 && hi >= 1.0 { " (tie)" } else { "" }
    );
    println!(
        "  idle guard: {} ({}, limit {}%){}",
        idle.mean_pct.map_or("unreadable".to_string(), |u| format!("{u:.1}%")),
        idle.source,
        idle.limit_pct,
        if idle.busy { "; GPU BUSY, --allow-busy: not evidence for a performance claim" } else { "" }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_brackets_the_median() {
        let r: Vec<f64> = (0..30).map(|i| 1.0 + (i as f64 - 15.0) * 0.001).collect();
        let (m, lo, hi) = bootstrap(&r, 2000, 1);
        assert!(lo <= m && m <= hi && hi - lo < 0.02);
    }
}
