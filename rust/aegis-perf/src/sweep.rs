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

/// One idle-guard reading (bench_gemm --idle-check), recorded on every run.
#[derive(Clone, Debug, PartialEq)]
pub struct IdleReading {
    /// Mean GPU utilization over the 2 s sample; None if no source answered.
    pub mean_pct: Option<f64>,
    /// "nvml", "sysfs:<path>", "tegrastats", or "none".
    pub source: String,
    /// The guard's limit (kIdleMaxPct, or lower via AEGIS_IDLE_MAX_PCT).
    pub limit_pct: f64,
    /// Above the limit (or unreadable): the run went ahead only with --allow-busy.
    pub busy: bool,
}

/// Parse bench_gemm --idle-check output.
pub fn parse_idle(stdout: &str) -> Result<IdleReading> {
    let v: Value = serde_json::from_str(stdout.trim()).context("parse bench_gemm --idle-check output")?;
    let ok = v["ok"].as_bool() == Some(true);
    let mean = v["mean_pct"].as_f64().filter(|_| ok);
    let limit = v["limit_pct"].as_f64().unwrap_or(IDLE_MAX_PCT);
    Ok(IdleReading {
        mean_pct: mean,
        source: v["source"].as_str().unwrap_or("none").to_string(),
        limit_pct: limit,
        busy: v["busy"].as_bool().unwrap_or_else(|| !ok || mean.is_none_or(|m| m > limit)),
    })
}

/// The idle guard (bench_gemm --idle-check), blocking: Err when the GPU is busy
/// (above the limit, or unreadable) and allow_busy is not set; otherwise the
/// reading, which the caller stores on every run row.
pub fn idle_guard(bench: &Path, allow_busy: bool) -> Result<IdleReading> {
    let mut cmd = Command::new(bench);
    cmd.arg("--idle-check");
    if allow_busy {
        cmd.arg("--allow-busy");
    }
    let out = cmd.stderr(std::process::Stdio::inherit()).output().context("run bench_gemm --idle-check")?;
    if !out.status.success() {
        bail!("idle guard refused to start (GPU busy); stop other GPU work or pass --allow-busy");
    }
    let r = parse_idle(&String::from_utf8_lossy(&out.stdout))?;
    check_idle(&r, allow_busy)?;
    Ok(r)
}

/// The blocking rule: a busy reading aborts the run unless allow_busy.
pub fn check_idle(r: &IdleReading, allow_busy: bool) -> Result<()> {
    if r.busy && !allow_busy {
        bail!("idle guard: GPU busy ({:?}% > {}%); stop other GPU work or pass --allow-busy", r.mean_pct, r.limit_pct);
    }
    Ok(())
}

/// Mean GPU utilization (%) above which the idle guard refuses (src/bench/harness.h kIdleMaxPct).
pub const IDLE_MAX_PCT: f64 = 2.0;

fn run_one(
    bench: &Path,
    impl_: &str,
    (m, n, k): (i64, i64, i64),
    reps: u32,
    idle: &IdleReading,
) -> Result<Option<(Value, String)>> {
    let mut cmd = crate::gpujob::command(bench)?;
    cmd.args(["--impl", impl_, "--m", &m.to_string(), "--n", &n.to_string(), "--k", &k.to_string(),
              "--reps", &reps.to_string(), "--warmup", "3", "--check"]);
    if let Some(u) = idle.mean_pct {
        cmd.args(["--idle-util-pct", &format!("{u:.1}")]);
    }
    cmd.args(["--idle-source", &idle.source]);
    if idle.busy {
        cmd.arg("--allow-busy");
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

pub struct SweepArgs<'a> {
    pub db: &'a Path,
    pub build: &'a Path,
    pub results: &'a Path,
    pub impls: Option<Vec<String>>,
    pub quick: bool,
    /// Shape file (M N K per line, # comments); overrides quick / the default set.
    pub shapes: Option<&'a Path>,
    pub allow_busy: bool,
    /// Do not recompute the winners table (nightly regression sweeps).
    pub no_winners: bool,
}

/// Read a shape file: one "M N K" per line, blank lines and # comments skipped.
pub fn read_shapes(path: &Path) -> Result<Vec<(i64, i64, i64)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut v = Vec::new();
    for line in text.lines().map(|l| l.split('#').next().unwrap_or("").trim()).filter(|l| !l.is_empty()) {
        let d: Vec<i64> = line.split_whitespace().map(str::parse).collect::<Result<_, _>>()
            .with_context(|| format!("{}: bad shape line '{line}'", path.display()))?;
        if d.len() != 3 {
            bail!("{}: bad shape line '{line}' (want M N K)", path.display());
        }
        v.push((d[0], d[1], d[2]));
    }
    Ok(v)
}

/// Returns the run id of the ingested rows.
pub fn cmd_sweep(a: &SweepArgs) -> Result<String> {
    let (db, build, results, allow_busy) = (a.db, a.build, a.results, a.allow_busy);
    let bench = build.join("bench_gemm");
    if !bench.exists() {
        bail!("{} not found; run cmake --build build first", bench.display());
    }
    std::fs::create_dir_all(results)?;
    let _gpu = crate::gpujob::lock("aegis-perf sweep")?;
    let host = db::hostname();
    let manifest = results.join(format!("manifest-{host}.json"));
    if !manifest.exists() {
        let out = Command::new(build.join("aegis_manifest")).output().context("run aegis_manifest")?;
        std::fs::write(&manifest, out.stdout)?;
    }
    let registered = default_impls(&bench)?;
    let impls = match a.impls.clone() {
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

    let idle = idle_guard(&bench, allow_busy)?;
    let shapes = match a.shapes {
        Some(p) => read_shapes(p)?,
        None if a.quick => quick_shapes(),
        None => [square_shapes(), qwen_shapes()].concat(),
    };
    let watch = crate::watch::Watch::start(&bench)?;
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
            if let Some((row, line)) = run_one(&bench, impl_, (m, n, k), s.reps, &idle)? {
                writeln!(f, "{line}")?;
                if row["pass"].as_bool() != Some(true) {
                    failures += 1;
                }
            }
        }
    }
    drop(f);
    let report = watch.finish()?;

    let run_id = db::cmd_ingest(db, &out, Some(&manifest), None, None)?;
    db::mark_disturbed(&db::connect(db)?, &run_id, &report)?;
    if !a.no_winners {
        db::cmd_winners(db, None, Some(&db::registry_of(&a.build.join("bench_gemm"))?))?;
    }
    println!("\nresults: {} (run {run_id})", out.display());
    if failures > 0 {
        bail!("{failures} correctness failures; failing rows are excluded from winners");
    }
    Ok(run_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_reading_passes_and_busy_reading_aborts_unless_allowed() {
        let idle = parse_idle(
            r#"{"ok":true,"mean_pct":0.40,"max_pct":1,"samples":20,"source":"nvml","limit_pct":2.00,"busy":false}"#,
        )
        .unwrap();
        assert_eq!(idle.mean_pct, Some(0.4));
        assert!(!idle.busy);
        assert!(check_idle(&idle, false).is_ok());

        let busy = parse_idle(
            r#"{"ok":true,"mean_pct":2.90,"max_pct":9,"samples":20,"source":"nvml","limit_pct":2.00,"busy":true}"#,
        )
        .unwrap();
        assert!(busy.busy);
        let err = check_idle(&busy, false).unwrap_err().to_string();
        assert!(err.contains("busy"), "{err}");
        assert!(check_idle(&busy, true).is_ok(), "--allow-busy runs (rows marked)");

        // No source answered: busy, never silently idle.
        let unreadable = parse_idle(r#"{"ok":false,"mean_pct":0.00,"max_pct":0,"samples":0}"#).unwrap();
        assert!(unreadable.busy && unreadable.mean_pct.is_none());
        assert!(check_idle(&unreadable, false).is_err());
    }
}
