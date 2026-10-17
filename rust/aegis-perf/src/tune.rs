//! `tune` and `emit-header`: the autotuner front end.
//!
//! `tune` runs build/aegis_tune (all registered configs x a shapes file, in one
//! process), saves its rows to results/tune-<host>-<ts>.jsonl, ingests them as
//! one run, recomputes winners and prints winner / % of cuBLAS / runner-up gap
//! per shape. `emit-header` writes include/aegis/winners/sm_<arch>.h from the
//! winners table for aegis::matmul.
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::db;

pub struct TuneArgs<'a> {
    pub db: &'a Path,
    pub build: &'a Path,
    pub results: &'a Path,
    pub shapes: &'a Path,
    pub seed: Option<u64>,
    pub prefix: Option<String>,
    pub allow_busy: bool,
    pub pair_with: Option<String>,
}

pub fn cmd_tune(a: &TuneArgs) -> Result<()> {
    let exe = a.build.join("aegis_tune");
    if !exe.exists() {
        bail!("{} not found; run cmake --build build first", exe.display());
    }
    let _gpu = crate::gpujob::lock(&format!("aegis-perf tune --shapes {}", a.shapes.display()))?;
    std::fs::create_dir_all(a.results)?;
    let host = db::hostname();
    let manifest = a.results.join(format!("manifest-{host}.json"));
    if !manifest.exists() {
        let out = Command::new(a.build.join("aegis_manifest")).output().context("run aegis_manifest")?;
        std::fs::write(&manifest, out.stdout)?;
    }

    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let out_path = a.results.join(format!("tune-{host}-{stamp}.jsonl"));
    let mut cmd = crate::gpujob::command(&exe)?;
    cmd.arg("--shapes").arg(a.shapes);
    if let Some(s) = a.seed {
        cmd.args(["--seed", &s.to_string()]);
    }
    if let Some(p) = &a.prefix {
        cmd.args(["--prefix", p]);
    }
    if let Some(p) = &a.pair_with {
        cmd.args(["--pair-with", p]);
    }
    if a.allow_busy {
        cmd.arg("--allow-busy");
    }
    // aegis_tune runs its own idle guard first; the watch covers the whole tune.
    let watch = crate::watch::Watch::start(&a.build.join("bench_gemm"))?;
    let t0 = Instant::now();
    let out = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // progress, seed and skipped configs
        .output()
        .context("run aegis_tune")?;
    let wall = t0.elapsed().as_secs_f64();
    let report = watch.finish()?;
    if !out.status.success() {
        bail!("aegis_tune failed ({})", out.status);
    }
    std::fs::write(&out_path, &out.stdout)?;

    let run_id = db::cmd_ingest(a.db, &out_path, Some(&manifest), None, None)?;
    db::mark_disturbed(&db::connect(a.db)?, &run_id, &report)?;
    db::cmd_winners(a.db, None)?;
    println!("\nresults: {}", out_path.display());
    println!("tune wall time (aegis_tune, measured by aegis-perf): {wall:.1} s\n");
    print!("{}", run_table(&String::from_utf8_lossy(&out.stdout))?);
    Ok(())
}

/// Winner per shape from one run's rows, with the gap to the runner-up.
/// Same layout as the original Python tuner's table.
pub fn run_table(jsonl: &str) -> Result<String> {
    let mut order: Vec<(i64, i64, i64)> = Vec::new();
    let mut rows: HashMap<(i64, i64, i64), Vec<Value>> = HashMap::new();
    let mut paired: Vec<Value> = Vec::new();
    for line in jsonl.lines().filter(|l| !l.trim().is_empty()) {
        let r: Value = serde_json::from_str(line)?;
        if r["kind"] == "paired" {
            paired.push(r);
            continue;
        }
        let s = &r["shape"];
        let key = (s["M"].as_i64().unwrap_or(0), s["N"].as_i64().unwrap_or(0), s["K"].as_i64().unwrap_or(0));
        if !rows.contains_key(&key) {
            order.push(key);
        }
        rows.entry(key).or_default().push(r);
    }
    let med = |r: &Value| r["median_us"].as_f64().unwrap_or(f64::INFINITY);
    let mut out = String::new();
    let hdr = format!(
        "{:>5} {:>5} {:>5}  {:<30} {:>9} {:>9}  {:<30} {:>6} {:>15}  {:>9}  {:>15}  {:<9}",
        "M", "N", "K", "winner", "% cuBLAS", "kernel %Lt", "runner-up", "gap", "ru/win CI", "warm %", "lt/win CI", "dispatch"
    );
    writeln!(out, "{hdr}")?;
    writeln!(out, "{}", "-".repeat(hdr.len()))?;
    for key in order {
        // Rank on cold rows (v1 rows have no warm pass); warm rows only annotate.
        let is_warm = |r: &&Value| r["cache_mode"] == "warm";
        let (warm, rs): (Vec<&Value>, Vec<&Value>) = rows[&key].iter().partition(is_warm);
        let cb = rs.iter().copied().find(|r| r["impl"] == "cublas");
        let lt = rs.iter().copied().find(|r| r["impl"] == "cublaslt_best");
        let is_base = |r: &Value| r["impl"] == "cublas" || r["impl"] == "cublaslt_best";
        let mut ok: Vec<&Value> = rs.iter().copied().filter(|r| !is_base(r) && r["pass"] == true).collect();
        ok.sort_by(|a, b| med(a).total_cmp(&med(b)));
        let (m, n, k) = key;
        let Some(w) = ok.first() else {
            writeln!(out, "{m:>5} {n:>5} {k:>5}  {:<30}", "(no passing config)")?;
            continue;
        };
        let pct_lt = match lt {
            Some(c) => format!("{:8.1}%", med(c) / med(w) * 100.0),
            None => format!("{:>9}", "-"),
        };
        let pct = match cb {
            Some(c) => format!("{:8.1}%", med(c) / med(w) * 100.0),
            None => format!("{:>9}", "-"),
        };
        let p90 = |r: &Value| r["p90_us"].as_f64().unwrap_or(f64::INFINITY);
        let pair_ci = |a: &Value, b: &Value| {
            paired
                .iter()
                .find(|p| p["shape"] == rows[&key][0]["shape"] && p["a"] == *a && p["b"] == *b)
                .map(|p| (p["ratio_ci_lo"].as_f64().unwrap_or(0.0), p["ratio_ci_hi"].as_f64().unwrap_or(0.0)))
        };
        let ci = |o: Option<&&Value>| match o.and_then(|r| pair_ci(&w["impl"], &r["impl"])) {
            Some((lo, hi)) => format!("[{lo:.3},{hi:.3}]"),
            None => "-".to_string(),
        };
        let ci_ru = ci(ok.get(1));
        let ci_lt = ci(lt.as_ref());
        // Where the floor rule would send the shape (on this run's CI; emit-header decides from the DB).
        let disp = match lt.and_then(|r| pair_ci(&w["impl"], &r["impl"])) {
            Some(c) => if dispatch(Some(c), None) == Dispatch::Cublaslt { "cublaslt" } else { "aegis" },
            None => "-",
        };
        let (ru, gap) = match ok.get(1) {
            // The paired tie test decides when it ran for this pair (same pair as winners).
            Some(r) if pair_ci(&w["impl"], &r["impl"]).is_some() => {
                let (lo, hi) = pair_ci(&w["impl"], &r["impl"]).unwrap_or_default();
                let g = if lo <= 1.0 && hi >= 1.0 { "  tied".to_string() } else { format!("{:5.1}%", (med(r) / med(w) - 1.0) * 100.0) };
                (r["impl"].as_str().unwrap_or("?").to_string(), g)
            }
            Some(r) if med(r) <= p90(w) => (r["impl"].as_str().unwrap_or("?").to_string(), "  tied".to_string()),
            Some(r) => (r["impl"].as_str().unwrap_or("?").to_string(), format!("{:5.1}%", (med(r) / med(w) - 1.0) * 100.0)),
            None => ("-".to_string(), format!("{:>6}", "-")),
        };
        let warm_of = |i: &Value| warm.iter().find(|r| r["impl"] == *i).map(|r| med(r));
        let warm_pct = match (warm_of(&Value::from("cublaslt_best")), warm_of(&w["impl"])) {
            (Some(c), Some(o)) => format!("{:8.1}%", c / o * 100.0),
            _ => format!("{:>9}", "-"),
        };
        writeln!(
            out,
            "{m:>5} {n:>5} {k:>5}  {:<30} {pct} {pct_lt}  {ru:<30} {gap} {ci_ru:>15}  {warm_pct}  {ci_lt:>15}  {disp:<9}",
            w["impl"].as_str().unwrap_or("?")
        )?;
    }
    writeln!(
        out,
        "\n% cuBLAS: cuBLAS median / winner median (100% = parity). gap: how much slower the runner-up's median is than the winner's;\n\
         \"tied\": the runner-up's median is inside the winner's p10-p90 (this run only; the DB winner is picked by the tie-break).\n\
         kernel %Lt: the winner kernel against cublaslt_best (not the dispatch). Ranked on cache_mode=cold rows.\n\
         dispatch: where the 80% floor sends the shape by this run's lt/win CI (aegis::matmul's own % is measured\n\
         separately: bench_gemm --impl aegis_matmul, nightly sweep, paired-ab).\n\
         warm %: cublaslt_best warm median / winner warm median (protocol v2).\n\
         ru/win CI, lt/win CI: 95% bootstrap CI of the paired median ratio runner-up / winner and cublaslt_best / winner\n\
         (30 alternating windows); when present it decides \"tied\" (the CI contains 1.0)."
    )?;
    Ok(out)
}

/// Production bar 3: aegis::matmul never runs a kernel below 80% of
/// cublaslt_best (aegis/matmul.h kDispatchFloor).
pub const DISPATCH_FLOOR: f64 = 0.80;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dispatch {
    Aegis,
    Cublaslt,
}

impl Dispatch {
    fn cpp(self) -> &'static str {
        match self {
            Dispatch::Aegis => "Aegis",
            Dispatch::Cublaslt => "Cublaslt",
        }
    }
}

/// Where aegis::matmul sends a winner's shapes: cuBLASLt when the paired CI of
/// cublaslt_best / winner time lies entirely below the floor (upper bound
/// below it); without a paired CI, when the unpaired median ratio is below it.
/// Neither known: the kernel.
pub fn dispatch(lt_ci: Option<(f64, f64)>, pct_lt: Option<f64>) -> Dispatch {
    let below = match (lt_ci, pct_lt) {
        (Some((_, hi)), _) => hi < DISPATCH_FLOOR,
        (None, Some(p)) => p / 100.0 < DISPATCH_FLOOR,
        (None, None) => false,
    };
    if below { Dispatch::Cublaslt } else { Dispatch::Aegis }
}

/// winners.lt_algo (JSON, src/lt_algo.h config_json) as an
/// aegis::detail::LtAlgoConfig initializer `{id, tile, stages, splitk,
/// reduction, swizzle, custom, inner_shape, cluster_shape}`; None without an
/// algo_id.
pub fn lt_config(json: &str) -> Result<Option<String>> {
    let v: Value = serde_json::from_str(json).context("winners.lt_algo")?;
    let Some(id) = v["algo_id"].as_i64() else { return Ok(None) };
    let f = |k: &str, d: i64| v[k].as_i64().unwrap_or(d);
    Ok(Some(format!(
        "{{{id}, {}, {}, {}, {}, {}, {}, {}, {}}}",
        f("tile_id", 0),
        f("stages_id", 0),
        f("splitk", 1),
        f("reduction", 0),
        f("swizzle", 0),
        f("custom", 0),
        f("inner_shape", 0),
        f("cluster_shape", 0)
    )))
}

struct Entry {
    m: i64,
    n: i64,
    k: i64,
    impl_: String,
    pct: Option<f64>,
    tie_note: String,
    pct_lt: Option<f64>,
    lt_ci: Option<(f64, f64)>,
    dispatch: Dispatch,
    /// cublaslt_best's algorithm at this shape: the LtAlgoConfig initializer.
    lt: Option<String>,
}

/// The generated header text (format unchanged from the original Python tuner
/// apart from line 1: generator name, host, time).
pub fn header_text(con: &rusqlite::Connection, arch: i64, generated: &str) -> Result<(String, usize)> {
    let mut stmt = con.prepare(
        "SELECT w.shape_bucket, k.impl, w.pct_of_cublas, w.tied, w.tie_set, w.pick_reason, w.pct_of_cublaslt_best, \
                w.tie_method, w.ratio_ci_lo, w.ratio_ci_hi, w.lt_ratio_ci_lo, w.lt_ratio_ci_hi, w.lt_algo \
         FROM winners w \
         JOIN kernels k USING(kernel_id) WHERE w.sm_arch = ? AND w.op = 'gemm' \
         AND w.dtype = 'f16f32f16'",
    )?;
    let mut entries: Vec<Entry> = Vec::new();
    for row in stmt.query_map([arch], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<f64>>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, Option<String>>(4)?,
            r.get::<_, Option<String>>(5)?,
            r.get::<_, Option<f64>>(6)?,
            r.get::<_, Option<String>>(7)?,
            r.get::<_, Option<f64>>(8)?,
            r.get::<_, Option<f64>>(9)?,
            (r.get::<_, Option<f64>>(10)?, r.get::<_, Option<f64>>(11)?),
            r.get::<_, Option<String>>(12)?,
        ))
    })? {
        let (bucket, impl_, pct, tied, tie_set, reason, pct_lt, method, ci_lo, ci_hi, lt_ci, lt_algo) = row?;
        // Paired tie test: runner-up / winner, 95% CI of the median ratio.
        let ci_note = match (method.as_deref(), ci_lo, ci_hi) {
            (Some("paired" | "paired_swap"), Some(lo), Some(hi)) => format!(" [paired CI {lo:.3}-{hi:.3}]"),
            _ => String::new(),
        };
        // Tie status: which configs the measurements could not separate, and why this one.
        let lt_note = pct_lt.map(|p| format!(", {p:.1}% of cublaslt_best")).unwrap_or_default();
        let tie_note = if tied != 0 {
            let others: Vec<String> = serde_json::from_str::<Vec<String>>(tie_set.as_deref().unwrap_or("[]"))?
                .into_iter()
                .filter(|i| *i != impl_)
                .collect();
            format!("{lt_note}; TIED with {}{ci_note} ({})", others.join(", "), reason.unwrap_or_default())
        } else {
            format!("{lt_note}{ci_note}")
        };
        let dims: Vec<i64> = bucket.split('x').map(str::parse).collect::<Result<_, _>>().context("shape_bucket")?;
        let [m, n, k] = dims[..] else { bail!("bad shape_bucket {bucket}") };
        let lt_ci = match lt_ci {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        };
        let lt = lt_algo.as_deref().map(lt_config).transpose()?.flatten();
        entries.push(Entry { m, n, k, impl_, pct, tie_note, pct_lt, lt_ci, dispatch: dispatch(lt_ci, pct_lt), lt });
    }
    entries.sort_by_key(|e| (e.n, e.k, e.m)); // stable, like Python's sort

    let mut lines = vec![
        format!("// Generated by {generated}."),
        "// Do not edit. Regenerate on the device after tuning.".to_string(),
        format!("// Tuned GEMM winners for sm_{arch}, read by aegis::matmul (src/matmul.cu)."),
        "// Row: M, N, K, tuned impl, dispatch (Aegis: the kernel; Cublaslt: the dispatch floor),".to_string(),
        "// % of cublaslt_best (unpaired medians), paired CI of cublaslt_best / winner time (0, 0: none).".to_string(),
        "// \"[paired CI]\" in a comment: runner-up / winner (the tie test).".to_string(),
        "#pragma once".to_string(),
        "#include \"aegis/matmul.h\"".to_string(),
        String::new(),
    ];
    if entries.is_empty() {
        lines.push(format!("// No winners in the perf DB for sm_{arch}; aegis::matmul uses cuBLAS."));
    } else {
        lines.push(format!("#define AEGIS_WINNERS_SM{arch} 1"));
        lines.push(format!("inline constexpr aegis::detail::Winner kWinners_sm{arch}[] = {{"));
        for e in &entries {
            let (m, n, k, impl_, tie_note) = (e.m, e.n, e.k, &e.impl_, &e.tie_note);
            let floor = match e.dispatch {
                Dispatch::Cublaslt => format!("; dispatch cublaslt: below the {:.0}% floor", DISPATCH_FLOOR * 100.0),
                Dispatch::Aegis => String::new(),
            };
            let note = match e.pct {
                Some(p) => format!("  // {p:.1}% of cuBLAS{tie_note}{floor}"),
                None if !tie_note.is_empty() => format!("  // {}{floor}", tie_note.trim_start_matches("; ")),
                None => String::new(),
            };
            let (lo, hi) = e.lt_ci.unwrap_or((0.0, 0.0));
            lines.push(format!(
                "    {{{m}, {n}, {k}, \"{impl_}\", aegis::detail::Dispatch::{}, {:.1}, {lo:.4}, {hi:.4}, {}}},{note}",
                e.dispatch.cpp(),
                e.pct_lt.unwrap_or(0.0),
                e.lt.as_deref().unwrap_or("{}")
            ));
        }
        lines.push("};".to_string());
    }
    Ok((lines.join("\n") + "\n", entries.len()))
}

pub fn cmd_emit_header(db: &Path, arch: Option<i64>, results: &Path, out: Option<&Path>) -> Result<()> {
    let host = db::hostname();
    let arch = match arch {
        Some(a) => a,
        None => {
            let m = results.join(format!("manifest-{host}.json"));
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&m).with_context(|| format!("read {}", m.display()))?)?;
            v["sm_arch"].as_i64().context("manifest without sm_arch")?
        }
    };
    let con = db::connect(db)?;
    let when = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC");
    let (text, n) = header_text(&con, arch, &format!("aegis-perf emit-header on {host}, {when}"))?;
    let default = format!("include/aegis/winners/sm_{arch}.h");
    let path = out.map(Path::to_path_buf).unwrap_or_else(|| default.clone().into());
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, text)?;
    println!("wrote {n} winners to {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_floor_uses_the_ci_upper_bound() {
        // 2026-10-17 rows: 4096^3 [0.7015, 0.7083] -> cuBLASLt; 512x4864x896 [0.7924, 0.8002] stays.
        assert_eq!(dispatch(Some((0.7015, 0.7083)), Some(70.2)), Dispatch::Cublaslt);
        assert_eq!(dispatch(Some((0.7924, 0.8002)), Some(79.3)), Dispatch::Aegis);
        // The CI decides over the unpaired median ratio.
        assert_eq!(dispatch(Some((0.81, 0.83)), Some(75.0)), Dispatch::Aegis);
        // No paired CI: the unpaired ratio; neither: the kernel.
        assert_eq!(dispatch(None, Some(79.9)), Dispatch::Cublaslt);
        assert_eq!(dispatch(None, Some(80.0)), Dispatch::Aegis);
        assert_eq!(dispatch(None, None), Dispatch::Aegis);
    }

    #[test]
    fn lt_algo_becomes_an_initializer() {
        let j = r#"{"algo_id":21,"tile_id":15,"stages_id":12,"splitk":3,"reduction":2,"swizzle":0,"custom":0,"inner_shape":0,"cluster_shape":0,"workspace":172032}"#;
        assert_eq!(lt_config(j).unwrap().as_deref(), Some("{21, 15, 12, 3, 2, 0, 0, 0, 0}"));
        assert_eq!(lt_config("{}").unwrap(), None);
    }
}
