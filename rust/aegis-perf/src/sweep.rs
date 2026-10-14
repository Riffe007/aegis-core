//! `sweep`: every registered impl (bench_gemm --list) over the shape set, one
//! bench_gemm process per (impl, shape), then ingest and recompute winners.
use std::io::Write as _;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::db;

/// Square shapes for the roofline, plus Qwen2.5-0.5B projection shapes
/// (hidden 896, MLP 4864) at decode (M=1, 16) and prefill (M=512).
pub fn square_shapes() -> Vec<(i64, i64, i64)> {
    [256, 512, 1024, 2048, 4096].iter().map(|&s| (s, s, s)).collect()
}

pub fn qwen_shapes() -> Vec<(i64, i64, i64)> {
    let mut v = Vec::new();
    for m in [1, 16, 512] {
        for (n, k) in [(896, 896), (4864, 896), (896, 4864)] {
            v.push((m, n, k));
        }
    }
    v
}

pub fn quick_shapes() -> Vec<(i64, i64, i64)> {
    vec![(512, 512, 512), (1, 896, 896), (1024, 1024, 1024)]
}

/// Reps (timed windows under protocol v2): cuBLAS 50; naive 5 and skipped above 2*2048^3
/// FLOPs; every other registered impl 20 reps.
pub struct Spec {
    pub reps: u32,
    pub max_flops: Option<i64>,
}

pub fn spec(impl_: &str) -> Spec {
    match impl_ {
        "cublas" | "cublaslt_best" => Spec { reps: 50, max_flops: None },
        "naive" => Spec { reps: 5, max_flops: Some(2 * 2048i64.pow(3)) },
        _ => Spec { reps: 20, max_flops: None },
    }
}

/// Default impl list: the baselines (cuBLAS, cublaslt_best), then the registry in order.
pub fn default_impls(bench: &Path) -> Result<Vec<String>> {
    let out = Command::new(bench).arg("--list").output().context("run bench_gemm --list")?;
    let mut impls = vec!["cublas".to_string(), "cublaslt_best".to_string()];
    for line in String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        impls.push(v["impl"].as_str().context("--list row without impl")?.to_string());
    }
    Ok(impls)
}

/// The idle guard (bench_gemm --idle-check): Ok(None) when idle, Ok(Some(mean %))
/// when busy and allowed, Err when busy and not allowed.
pub fn idle_guard(bench: &Path, allow_busy: bool) -> Result<Option<f64>> {
    let mut cmd = Command::new(bench);
    cmd.arg("--idle-check");
    if allow_busy {
        cmd.arg("--allow-busy");
    }
    let out = cmd.stderr(std::process::Stdio::inherit()).output().context("run bench_gemm --idle-check")?;
    if !out.status.success() {
        bail!("idle guard refused to start (GPU busy); stop other GPU work or pass --allow-busy");
    }
    let v: Value = serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())?;
    let mean = v["mean_pct"].as_f64().unwrap_or(f64::NAN);
    let busy = v["ok"].as_bool() != Some(true) || mean > IDLE_MAX_PCT;
    Ok(busy.then_some(mean))
}

/// Mean GPU utilization (%) above which the idle guard refuses (src/bench/harness.h kIdleMaxPct).
pub const IDLE_MAX_PCT: f64 = 2.0;

fn run_one(
    bench: &Path,
    impl_: &str,
    (m, n, k): (i64, i64, i64),
    reps: u32,
    busy: Option<f64>,
) -> Result<Option<(Value, String)>> {
    let mut cmd = Command::new(bench);
    cmd.args(["--impl", impl_, "--m", &m.to_string(), "--n", &n.to_string(), "--k", &k.to_string(),
              "--reps", &reps.to_string(), "--warmup", "3", "--check"]);
    if let Some(u) = busy {
        cmd.args(["--allow-busy", "--idle-util-pct", &format!("{u:.1}")]);
    }
    let p = cmd.output().context("run bench_gemm")?;
    let stdout = String::from_utf8_lossy(&p.stdout);
    let Some(line) = stdout.trim().lines().last().filter(|l| !l.is_empty()) else {
        eprintln!("  ! {impl_} {m}x{n}x{k}: no output\n{}", String::from_utf8_lossy(&p.stderr));
        return Ok(None);
    };
    let row: Value = serde_json::from_str(line)?;
    let status = if row["pass"].as_bool() == Some(true) { "ok" } else { "FAIL" };
    println!(
        "  {impl_:>8} {m:>5}x{n:>5}x{k:>5}  {:>10.1} us  {:>7.2} TFLOPS  {status}",
        row["median_us"].as_f64().unwrap_or(f64::NAN),
        row["tflops"].as_f64().unwrap_or(f64::NAN)
    );
    Ok(Some((row, line.to_string())))
}

pub fn cmd_sweep(
    db: &Path,
    build: &Path,
    results: &Path,
    impls: Option<Vec<String>>,
    quick: bool,
    allow_busy: bool,
) -> Result<()> {
    let bench = build.join("bench_gemm");
    if !bench.exists() {
        bail!("{} not found; run cmake --build build first", bench.display());
    }
    std::fs::create_dir_all(results)?;
    let host = db::hostname();
    let manifest = results.join(format!("manifest-{host}.json"));
    if !manifest.exists() {
        let out = Command::new(build.join("aegis_manifest")).output().context("run aegis_manifest")?;
        std::fs::write(&manifest, out.stdout)?;
    }
    let registered = default_impls(&bench)?;
    let impls = match impls {
        Some(list) => {
            for i in &list {
                if !registered.contains(i) {
                    bail!("unknown impl '{i}' (not in bench_gemm --list)");
                }
            }
            list
        }
        None => registered,
    };

    let busy = idle_guard(&bench, allow_busy)?;
    let shapes = if quick { quick_shapes() } else { [square_shapes(), qwen_shapes()].concat() };
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let out = results.join(format!("{host}-{stamp}.jsonl"));
    let mut f = std::fs::File::create(&out)?;
    let mut failures = 0;
    for &(m, n, k) in &shapes {
        for impl_ in &impls {
            let s = spec(impl_);
            if s.max_flops.is_some_and(|cap| 2 * m * n * k > cap) {
                continue;
            }
            if let Some((row, line)) = run_one(&bench, impl_, (m, n, k), s.reps, busy)? {
                writeln!(f, "{line}")?;
                if row["pass"].as_bool() != Some(true) {
                    failures += 1;
                }
            }
        }
    }
    drop(f);

    db::cmd_ingest(db, &out, Some(&manifest), None, None)?;
    db::cmd_winners(db, None)?;
    println!("\nresults: {}", out.display());
    if failures > 0 {
        bail!("{failures} correctness failures; failing rows are excluded from winners");
    }
    Ok(())
}
