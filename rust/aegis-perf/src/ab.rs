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
//!
//! Every result is stored in the perf DB (`paired_results`, see schema.sql)
//! and printed. Each build's commit: `--sha-a` / `--sha-b`, else the HEAD of
//! the git tree that contains the binary (`-dirty` if it has uncommitted
//! tracked changes; NULL outside a tree). A binary built before the tree moved
//! on would be mislabelled: pass the SHA explicitly for saved binaries.
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use rusqlite::params;
use serde_json::Value;

pub struct AbArgs<'a> {
    pub db: &'a Path,
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
    pub sha_a: Option<&'a str>,
    pub sha_b: Option<&'a str>,
}

/// One bench_gemm run: (median_us, protocol_version).
fn run_once(bench: &Path, impl_: &str, (m, n, k): (i64, i64, i64), reps: u32, cache: &str) -> Result<(f64, i64, Option<String>)> {
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
    Ok((
        v["median_us"].as_f64().context("bench_gemm row without median_us")?,
        v["protocol_version"].as_i64().unwrap_or(1),
        v["profile"].as_str().map(str::to_string),
    ))
}

/// The commit of the git tree containing `bench` (`-dirty` with uncommitted
/// tracked changes), or None outside a tree.
pub fn build_sha(bench: &Path) -> Option<String> {
    let path = std::fs::canonicalize(bench).ok()?;
    let dir = path.ancestors().skip(1).find(|d| d.join(".git").exists())?;
    let git = |args: &[&str]| -> Option<String> {
        let o = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
        o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let sha = git(&["rev-parse", "--short=12", "HEAD"])?;
    let dirty = !git(&["status", "--porcelain", "--untracked-files=no"])?.is_empty();
    Some(if dirty { format!("{sha}-dirty") } else { sha })
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

/// One paired-ab result, as stored in `paired_results`.
#[derive(Clone, Debug, PartialEq)]
pub struct NewPairedResult {
    pub sha_a: Option<String>,
    pub sha_b: Option<String>,
    pub shape: (i64, i64, i64),
    pub impl_a: String,
    pub impl_b: String,
    pub ratio: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub n: usize,
    pub protocol_version: i64,
    pub cache_mode: String,
    pub idle_reading: Option<f64>,
    pub clocks_locked: bool,
    pub started_at: String,
    pub finished_at: String,
    /// The idle guard during the run: Some(true) other processes used the GPU
    /// above the limit mid-run; None not watched.
    pub disturbed: Option<bool>,
    pub disturbed_detail: Option<String>,
    /// The device profile both builds ran under ("measured" / "unmeasured"); "mixed" if
    /// their rows disagree; None if no row carried one (builds before schema v5).
    pub profile: Option<String>,
}

/// Insert one result; returns its id.
pub fn store(con: &rusqlite::Connection, r: &NewPairedResult) -> Result<i64> {
    let (m, n, k) = r.shape;
    con.execute(
        "INSERT INTO paired_results (sha_a, sha_b, shape, impl_a, impl_b, ratio, ci_lo, ci_hi, n, protocol_version, \
         cache_mode, idle_reading, clocks_locked, started_at, finished_at, disturbed, disturbed_detail, profile) \
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        params![
            r.sha_a, r.sha_b, format!("{m}x{n}x{k}"), r.impl_a, r.impl_b, r.ratio, r.ci_lo, r.ci_hi, r.n as i64,
            r.protocol_version, r.cache_mode, r.idle_reading, r.clocks_locked as i64, r.started_at, r.finished_at,
            r.disturbed.map(i64::from), r.disturbed_detail, r.profile,
        ],
    )?;
    Ok(con.last_insert_rowid())
}

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
}

pub fn cmd_paired_ab(a: &AbArgs) -> Result<()> {
    let _gpu = crate::gpujob::lock(&format!("aegis-perf paired-ab {} vs {}", a.impl_a, a.impl_b))?;
    // Blocking idle guard (Benchmark policy), with build B's bench_gemm.
    let idle = crate::sweep::idle_guard(a.b, a.allow_busy)?;
    let sha_a = a.sha_a.map(str::to_string).or_else(|| build_sha(a.a));
    let sha_b = a.sha_b.map(str::to_string).or_else(|| build_sha(a.b));
    let started_at = now();
    let watch = crate::watch::Watch::start(a.b)?;
    let (mut ta, mut tb, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    let mut protocols = Vec::new();
    let mut profiles: Vec<Option<String>> = Vec::new();
    for i in 0..a.pairs {
        // Alternate which side runs first so neither always follows the other.
        let (x, y) = if i % 2 == 0 {
            let x = run_once(a.a, a.impl_a, a.shape, a.reps, a.cache)?;
            (x, run_once(a.b, a.impl_b, a.shape, a.reps, a.cache)?)
        } else {
            let y = run_once(a.b, a.impl_b, a.shape, a.reps, a.cache)?;
            (run_once(a.a, a.impl_a, a.shape, a.reps, a.cache)?, y)
        };
        ta.push(x.0);
        tb.push(y.0);
        ratios.push(y.0 / x.0);
        protocols.extend([x.1, y.1]);
        profiles.extend([x.2, y.2]);
    }
    let finished_at = now();
    let report = watch.finish()?;
    if protocols.iter().any(|&p| p != protocols[0]) {
        bail!("builds A and B ran different timing protocols ({protocols:?}); not comparable");
    }
    let (med, lo, hi) = bootstrap(&ratios, 10_000, 0x5eed);
    let (m, n, k) = a.shape;
    let result = NewPairedResult {
        sha_a: sha_a.clone(),
        sha_b: sha_b.clone(),
        shape: a.shape,
        impl_a: a.impl_a.to_string(),
        impl_b: a.impl_b.to_string(),
        ratio: med,
        ci_lo: lo,
        ci_hi: hi,
        n: a.pairs,
        protocol_version: protocols[0],
        cache_mode: a.cache.to_string(),
        idle_reading: idle.mean_pct,
        clocks_locked: false, // paired-ab does not lock clocks
        started_at,
        finished_at,
        disturbed: report.disturbed,
        disturbed_detail: Some(report.detail()),
        profile: match profiles.iter().flatten().collect::<std::collections::BTreeSet<_>>() {
            s if s.is_empty() => None,
            s if s.len() == 1 => s.into_iter().next().cloned(),
            _ => Some("mixed".to_string()),
        },
    };
    let id = store(&crate::db::connect(a.db)?, &result)?;
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
        "  sha A {} / B {}; idle guard: {} ({}, limit {}%){}; stored as paired_results id {id}",
        sha_a.as_deref().unwrap_or("unknown"),
        sha_b.as_deref().unwrap_or("unknown"),
        idle.mean_pct.map_or("unreadable".to_string(), |u| format!("{u:.1}%")),
        idle.source,
        idle.limit_pct,
        if idle.busy { "; GPU BUSY, --allow-busy: not evidence for a performance claim" } else { "" }
    );
    match result.disturbed {
        Some(true) => println!("  DISTURBED mid-run (not evidence for a performance claim): {}", report.detail()),
        Some(false) => println!("  idle watch: clean ({})", report.detail()),
        None => println!("  idle watch: {}", report.detail()),
    }
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

    #[test]
    fn build_sha_outside_a_tree_is_none() {
        let p = std::env::temp_dir().join(format!("aegis-ab-sha-{}", std::process::id()));
        std::fs::write(&p, b"x").unwrap();
        assert_eq!(build_sha(&p), None);
        std::fs::remove_file(&p).ok();
        // Inside this repo: a 12-hex-digit commit, maybe -dirty.
        let sha = build_sha(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml").as_path()).unwrap();
        assert!(sha.len() >= 12 && sha[..12].chars().all(|c| c.is_ascii_hexdigit()), "{sha}");
    }
}
