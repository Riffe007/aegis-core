//! `aegis-perf calibrate`: measure the current device end to end and write its
//! hardware-model files.
//!
//!   1. preflight: hardware discovery (build/aegis_hw) names the device and arch;
//!      the GPU lock is taken for the whole run (rule 15) and passed to every step.
//!   2. clock lock: `sudo -n nvidia-smi -lgc F,F` at the default applications clock.
//!      sudo needs no password only where root is configured for it; otherwise the
//!      latency probes run unlocked, and that is recorded (rows: clocks_locked).
//!   3. microbench: microbench/run_all.sh (every probe, SASS proofs, its own idle
//!      guard and in-run watch, ingested into the perf DB).
//!   4. locked latency: microbench/run_latency_locked.sh F, then `nvidia-smi -rgc`
//!      (only when step 2 locked the clocks).
//!   5. tune: the tracked shape set (`aegis-perf tune`, idle guard, paired tie tests).
//!   6. emit-arch and emit-header into the output directory: arch/sm_<arch>.json,
//!      winners/sm_<arch>.h. Default include/aegis (the committed files); a scratch
//!      --out-dir leaves them alone.
//!   7. report: each step's time, appended to results/calibrate-history.jsonl (the
//!      dry run's estimates), DIR/calibrate.json, and, when DIR is not include/aegis,
//!      the new arch file's cycle counts against the committed one's.
//!
//! Every device allocation in the steps goes through the GPU budget (their binaries).
//! `--dry-run` lists the steps with an estimated duration and runs nothing.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::{arch, db, tune};

pub struct CalibrateArgs<'a> {
    pub db: &'a Path,
    pub build: &'a Path,
    pub results: &'a Path,
    pub shapes: &'a Path,
    pub out_dir: &'a Path,
    pub dry_run: bool,
    pub allow_busy: bool,
    /// Run a dirty or unknown build anyway: every step is told (--allow-dirty, ALLOW_DIRTY=1)
    /// and its rows are marked allow_dirty, excluded by default.
    pub allow_dirty: bool,
}

/// The device as discovery reports it.
#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub arch: i64,
    pub sm_count: i64,
}

pub fn discover(build: &Path) -> Result<Device> {
    let exe = build.join("aegis_hw");
    let out = Command::new(&exe).output().with_context(|| format!("run {}", exe.display()))?;
    if !out.status.success() {
        bail!("{} failed: {}", exe.display(), String::from_utf8_lossy(&out.stderr));
    }
    let v: Value = serde_json::from_slice(&out.stdout).context("aegis_hw JSON")?;
    let g = &v["gpu"];
    if g["present"].as_bool() != Some(true) {
        bail!("no GPU found by aegis_hw");
    }
    Ok(Device {
        name: g["name"].as_str().unwrap_or("?").to_string(),
        arch: g["arch"].as_i64().context("gpu.arch")?,
        sm_count: g["sm_count"].as_i64().unwrap_or(0),
    })
}

/// One planned step and how long it should take.
#[derive(Debug)]
pub struct Planned {
    pub step: &'static str,
    pub what: String,
    pub seconds: f64,
    pub basis: String,
}

/// Seconds per step from the newest history entry for this device, else None.
pub fn history_estimate(history: &str, device: &str, step: &str) -> Option<f64> {
    history
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["device"].as_str() == Some(device) && v["step"].as_str() == Some(step))
        .filter_map(|v| v["seconds"].as_f64())
        .next_back()
}

/// Rough models when there is no history (stated as such in the plan).
const MODEL_MICROBENCH_S: f64 = 600.0;
const MODEL_LATENCY_S: f64 = 180.0;
const MODEL_TUNE_PER_PAIR_S: f64 = 0.5;

pub fn plan(dev: &Device, shapes: usize, configs: usize, history: &str, out_dir: &Path, clocks_lockable: bool) -> Vec<Planned> {
    let est = |step: &'static str, model: f64, model_basis: String| -> (f64, String) {
        match history_estimate(history, &dev.name, step) {
            Some(s) => (s, "the last calibrate on this device".to_string()),
            None => (model, format!("rough model, no calibrate history: {model_basis}")),
        }
    };
    let mut v = Vec::new();
    let mut push = |step: &'static str, what: String, (seconds, basis): (f64, String)| {
        v.push(Planned { step, what, seconds, basis })
    };
    push("preflight", format!("{} (sm_{}, {} SMs); take the GPU lock", dev.name, dev.arch, dev.sm_count), (1.0, "fixed".into()));
    push(
        "clock_lock",
        if clocks_lockable {
            "sudo -n nvidia-smi -lgc at the default applications clock".into()
        } else {
            "sudo -n refused: latency probes run unlocked (recorded on every row)".into()
        },
        (1.0, "fixed".into()),
    );
    push("microbench", "microbench/run_all.sh: every probe, SASS proofs, idle guard, ingest".into(),
         est("microbench", MODEL_MICROBENCH_S, format!("{MODEL_MICROBENCH_S:.0} s")));
    if clocks_lockable {
        push("latency_locked", "microbench/run_latency_locked.sh, then nvidia-smi -rgc".into(),
             est("latency_locked", MODEL_LATENCY_S, format!("{MODEL_LATENCY_S:.0} s")));
    }
    let pairs = shapes * configs;
    push("tune", format!("aegis-perf tune: {shapes} shapes x up to {configs} configs"),
         est("tune", pairs as f64 * MODEL_TUNE_PER_PAIR_S, format!("{pairs} config-shape pairs x {MODEL_TUNE_PER_PAIR_S} s")));
    push("emit", format!("{}/arch/sm_{a}.json and {}/winners/sm_{a}.h", out_dir.display(), out_dir.display(), a = dev.arch),
         (2.0, "fixed".into()));
    v
}

pub fn fmt_duration(s: f64) -> String {
    let s = s.round() as i64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// `sudo -n` works (no password prompt) and nvidia-smi is there.
fn clocks_lockable() -> bool {
    Command::new("sudo").args(["-n", "true"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn default_app_clock() -> Result<u32> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.default_applications.graphics", "--format=csv,noheader,nounits"])
        .output()
        .context("nvidia-smi")?;
    String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("").trim().parse().context("default applications clock")
}

/// Every "cycles" value of the committed and the new arch file, keyed by its section.
pub fn compare_cycles(old: &Value, new: &Value) -> Vec<(String, f64, f64, f64)> {
    let mut rows = Vec::new();
    if let (Some(o), Some(n)) = (old.as_object(), new.as_object()) {
        for (k, ov) in o {
            if let (Some(oc), Some(nc)) = (ov["cycles"].as_f64(), n.get(k).and_then(|nv| nv["cycles"].as_f64())) {
                rows.push((k.clone(), oc, nc, (nc / oc - 1.0) * 100.0));
            }
        }
    }
    rows
}

/// The newest run in the perf DB (the step that just ingested): id, disturbed flag, detail.
fn newest_run(db_path: &Path) -> Result<(String, Option<i64>, Option<String>)> {
    let con = db::connect(db_path)?;
    Ok(con.query_row(
        "SELECT run_id, disturbed, disturbed_detail FROM runs ORDER BY started_at DESC LIMIT 1",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?)
}

fn run_step(name: &str, cmd: &mut Command) -> Result<()> {
    eprintln!("== calibrate: {name}: {cmd:?}");
    let st = cmd.status().with_context(|| format!("run {name}"))?;
    if !st.success() {
        bail!("calibrate step {name} failed ({st})");
    }
    Ok(())
}

pub fn cmd_calibrate(a: &CalibrateArgs) -> Result<()> {
    let dev = discover(a.build)?;
    let configs = Command::new(a.build.join("bench_gemm")).arg("--list").output().map(|o| {
        String::from_utf8_lossy(&o.stdout).lines().filter(|l| l.starts_with('{')).count()
    })?;
    let shapes = std::fs::read_to_string(a.shapes)
        .with_context(|| format!("read {}", a.shapes.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .count();
    let history_path = a.results.join("calibrate-history.jsonl");
    let history = std::fs::read_to_string(&history_path).unwrap_or_default();
    let lockable = clocks_lockable();
    let steps = plan(&dev, shapes, configs, &history, a.out_dir, lockable);
    let total: f64 = steps.iter().map(|s| s.seconds).sum();
    println!("calibrate on {} (sm_{}, {} SMs), output {}:", dev.name, dev.arch, dev.sm_count, a.out_dir.display());
    for (i, s) in steps.iter().enumerate() {
        println!("  {}. {:<15} ~{:>7}  {}  [{}]", i + 1, s.step, fmt_duration(s.seconds), s.what, s.basis);
    }
    println!("  estimated total ~{}", fmt_duration(total));
    if a.dry_run {
        return Ok(());
    }
    // A dirty or unknown build is refused before any GPU work (schema v7, build identity).
    db::check_build("calibrate", db::build_of(&a.build.join("bench_gemm"))?.as_ref(), a.allow_dirty)?;

    // 1. The GPU lock for every step; children inherit it (AEGIS_GPU_LOCK_HELD), and
    // in-process steps (tune) see it through the same variable.
    let _gpu = crate::gpujob::lock("aegis-perf calibrate")?;
    std::env::set_var(crate::gpujob::HELD_ENV, std::process::id().to_string());
    std::env::set_var("AEGIS_PERFDB", a.db);
    let mut done: Vec<Value> = Vec::new();
    // Disturbed steps: their rows are excluded from winners and arch files, so the
    // outputs rest on older runs. Reported, and calibrate exits with an error.
    let mut warnings: Vec<String> = Vec::new();
    let host = db::hostname();
    let mut record = |step: &str, t0: Instant, detail: Value| -> Result<()> {
        let seconds = t0.elapsed().as_secs_f64();
        let row = json!({"device": dev.name, "host": host, "sm_count": dev.sm_count, "step": step,
                         "seconds": seconds, "at": chrono::Utc::now().to_rfc3339(), "detail": detail});
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&history_path)?;
        writeln!(f, "{row}")?;
        eprintln!("== calibrate: {step} done in {}", fmt_duration(seconds));
        done.push(row);
        Ok(())
    };

    // 2. Clock lock (only without a password prompt).
    let t0 = Instant::now();
    let mut locked_mhz = None;
    let clock_detail = if lockable {
        let f = default_app_clock()?;
        let ok = Command::new("sudo").args(["-n", "nvidia-smi", "-lgc", &format!("{f},{f}")]).status()?.success();
        if ok {
            locked_mhz = Some(f);
            json!({"locked": true, "mhz": f})
        } else {
            json!({"locked": false, "reason": "sudo -n nvidia-smi -lgc failed"})
        }
    } else {
        json!({"locked": false, "reason": "sudo -n needs a password; latency probes ran unlocked"})
    };
    record("clock_lock", t0, clock_detail.clone())?;

    // 3-4. Probes. Clocks are restored whatever happens.
    let probes = (|| -> Result<()> {
        let mut env_busy = Command::new("microbench/run_all.sh");
        if a.allow_busy {
            env_busy.env("ALLOW_BUSY", "1");
        }
        if a.allow_dirty {
            env_busy.env("ALLOW_DIRTY", "1");
        }
        let t0 = Instant::now();
        run_step("microbench", &mut env_busy)?;
        let (run, disturbed, detail) = newest_run(a.db)?;
        if disturbed == Some(1) {
            warnings.push(format!("microbench run {run} disturbed: {}", detail.clone().unwrap_or_default()));
        }
        record("microbench", t0, json!({"run_id": run, "disturbed": disturbed, "disturbed_detail": detail}))?;
        if let Some(f) = locked_mhz {
            let t0 = Instant::now();
            let mut locked = Command::new("microbench/run_latency_locked.sh");
            locked.arg(f.to_string());
            if a.allow_dirty {
                locked.env("ALLOW_DIRTY", "1");
            }
            run_step("latency_locked", &mut locked)?;
            record("latency_locked", t0, json!({"mhz": f}))?;
        }
        Ok(())
    })();
    if locked_mhz.is_some() {
        let _ = Command::new("sudo").args(["-n", "nvidia-smi", "-rgc"]).status();
    }
    probes?;

    // 5. Tune.
    let t0 = Instant::now();
    tune::cmd_tune(&tune::TuneArgs {
        db: a.db,
        build: a.build,
        results: a.results,
        shapes: a.shapes,
        seed: None,
        prefix: None,
        allow_busy: a.allow_busy,
        pair_with: None,
        allow_dirty: a.allow_dirty,
    })?;
    let (run, disturbed, detail) = newest_run(a.db)?;
    if disturbed == Some(1) {
        warnings.push(format!("tune run {run} disturbed: {}", detail.clone().unwrap_or_default()));
    }
    record("tune", t0, json!({"shapes": a.shapes, "run_id": run, "disturbed": disturbed, "disturbed_detail": detail}))?;

    // 6. The hardware-model files.
    let t0 = Instant::now();
    let arch_out = a.out_dir.join(format!("arch/sm_{}.json", dev.arch));
    let header_out = a.out_dir.join(format!("winners/sm_{}.h", dev.arch));
    arch::cmd_emit_arch(a.db, dev.arch, &[], a.results, Some(&arch_out))?;
    tune::cmd_emit_header(a.db, Some(dev.arch), a.results, Some(&header_out))?;
    record("emit", t0, json!({"arch": arch_out, "header": header_out}))?;

    // 7. Report.
    let committed = PathBuf::from(format!("include/aegis/arch/sm_{}.json", dev.arch));
    let mut comparison = Vec::new();
    if arch_out != committed && committed.exists() {
        let old: Value = serde_json::from_str(&std::fs::read_to_string(&committed)?)?;
        let new: Value = serde_json::from_str(&std::fs::read_to_string(&arch_out)?)?;
        println!("cycles vs the committed {}:", committed.display());
        for (k, o, n, d) in compare_cycles(&old, &new) {
            println!("  {k:<14} {o:>10.2} -> {n:>10.2}  {d:+.2}%{}", if d.abs() > 3.0 { "  (> 3%)" } else { "" });
            comparison.push(json!({"key": k, "committed": o, "new": n, "pct": d}));
        }
    }
    let summary = json!({"device": dev.name, "sm_arch": dev.arch, "sm_count": dev.sm_count, "host": db::hostname(),
                         "clocks": clock_detail, "steps": done, "arch_file": arch_out, "winners_header": header_out,
                         "vs_committed": comparison, "disturbed": warnings});
    let sp = a.out_dir.join("calibrate.json");
    std::fs::write(&sp, serde_json::to_string_pretty(&summary)? + "\n")?;
    println!("wrote {}", sp.display());
    if !warnings.is_empty() {
        for w in &warnings {
            eprintln!("calibrate: WARNING: {w}");
        }
        bail!("calibrate: {} step(s) disturbed; their rows do not count (the outputs use older clean runs). Rerun on an idle GPU.", warnings.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device { name: "NVIDIA GB10".into(), arch: 121, sm_count: 48 }
    }

    #[test]
    fn plan_uses_history_then_the_model() {
        let p = plan(&dev(), 14, 50, "", Path::new("/tmp/x"), false);
        let tune = p.iter().find(|s| s.step == "tune").unwrap();
        assert_eq!(tune.seconds, 14.0 * 50.0 * MODEL_TUNE_PER_PAIR_S);
        assert!(tune.basis.starts_with("rough model"));
        assert!(p.iter().all(|s| s.step != "latency_locked"), "no locked-latency step without sudo");
        let h = concat!(
            r#"{"device":"NVIDIA GB10","step":"tune","seconds":100.0}"#, "\n",
            r#"{"device":"Orin","step":"tune","seconds":5.0}"#, "\n",
            r#"{"device":"NVIDIA GB10","step":"tune","seconds":4321.0}"#, "\n"
        );
        let p = plan(&dev(), 14, 50, h, Path::new("/tmp/x"), true);
        let tune = p.iter().find(|s| s.step == "tune").unwrap();
        assert_eq!(tune.seconds, 4321.0, "newest entry for this device");
        assert!(p.iter().any(|s| s.step == "latency_locked"));
    }

    #[test]
    fn cycles_compare_by_section() {
        let old = json!({"l2_latency": {"cycles": 335.437}, "dram_latency": {"cycles": 956.63}, "l2_bytes": 1});
        let new = json!({"l2_latency": {"cycles": 340.0}, "dram_latency": {"cycles": 900.0}});
        let mut rows = compare_cycles(&old, &new);
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(rows.len(), 2);
        assert!((rows[1].3 - 1.36).abs() < 0.01);
        assert!(rows[0].3 < -5.0);
    }

    #[test]
    fn durations_read_well() {
        assert_eq!(fmt_duration(42.4), "42s");
        assert_eq!(fmt_duration(125.0), "2m05s");
        assert_eq!(fmt_duration(7380.0), "2h03m");
    }
}
