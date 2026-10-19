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
    /// Tune a dirty or unknown build anyway (rows marked allow_dirty, excluded by default).
    pub allow_dirty: bool,
}

pub fn cmd_tune(a: &TuneArgs) -> Result<()> {
    let exe = a.build.join("aegis_tune");
    if !exe.exists() {
        bail!("{} not found; run cmake --build build first", exe.display());
    }
    // Refuse a dirty build before an hour of GPU time, not at ingest.
    db::check_build("tune", db::build_of(&a.build.join("bench_gemm"))?.as_ref(), a.allow_dirty)?;
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

    let run_id = db::cmd_ingest(a.db, &out_path, Some(&manifest), None, None, db::IngestOpts { kind: db::RunKind::Tune, allow_dirty: a.allow_dirty })?;
    db::mark_disturbed(&db::connect(a.db)?, &run_id, &report)?;
    db::cmd_winners(a.db, None, Some(&db::registry_of(&a.build.join("bench_gemm"))?))?;
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
        "{:>5} {:>5} {:>5}  {:<30} {:<26} {:<9} {:>11} {:>12}  {:<30} {:>6} {:>15}  {:>9}",
        "M", "N", "K", "winner", "kernel %Lt (paired)", "dispatch", "%Lt unpaired", "%cuBLAS unp.", "runner-up", "gap", "ru/win CI", "warm %"
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
            Some(c) => format!("{:10.1}%", med(c) / med(w) * 100.0),
            None => format!("{:>11}", "-"),
        };
        let pct = match cb {
            Some(c) => format!("{:11.1}%", med(c) / med(w) * 100.0),
            None => format!("{:>12}", "-"),
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
        // The number of record (bars.rs): the paired median cublaslt_best / winner and its CI.
        let lt_paired = lt
            .and_then(|r| {
                paired.iter().find(|p| p["shape"] == rows[&key][0]["shape"] && p["a"] == w["impl"] && p["b"] == r["impl"])
            })
            .and_then(|p| {
                Some(crate::bars::PairedPct {
                    median: p["ratio_median"].as_f64()?,
                    ci_lo: p["ratio_ci_lo"].as_f64()?,
                    ci_hi: p["ratio_ci_hi"].as_f64()?,
                })
            });
        let kernel = lt_paired.map_or("no paired test".to_string(), |p| p.text());
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
            "{m:>5} {n:>5} {k:>5}  {:<30} {kernel:<26} {disp:<9} {pct_lt} {pct}  {ru:<30} {gap} {ci_ru:>15}  {warm_pct}",
            w["impl"].as_str().unwrap_or("?")
        )?;
    }
    writeln!(
        out,
        "\nkernel %Lt (paired): the number of record (aegis_perf::bars): 100 x the paired median ratio cublaslt_best / winner\n\
         time and its 95% bootstrap CI (30 alternating windows); \"tie\" when the CI contains 100%. The winner kernel, not the dispatch.\n\
         %Lt unpaired, %cuBLAS unp.: cublaslt_best / cuBLAS median over the winner median (secondary; 100% = parity).\n\
         gap: how much slower the runner-up's median is than the winner's;\n\
         \"tied\": the runner-up's median is inside the winner's p10-p90 (this run only; the DB winner is picked by the tie-break).\n\
         Ranked on cache_mode=cold rows.\n\
         dispatch: where the 80% floor sends the shape by this run's paired CI (aegis::matmul's own % is measured\n\
         separately: bench_gemm --impl aegis_matmul, nightly sweep, paired-ab).\n\
         warm %: cublaslt_best warm median / winner warm median (protocol v2).\n\
         ru/win CI: 95% bootstrap CI of the paired median ratio runner-up / winner; when present it decides \"tied\"\n\
         (the CI contains 1.0)."
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
    /// winners.dispatch: "aegis" | "cublaslt".
    pub fn as_str(self) -> &'static str {
        match self {
            Dispatch::Aegis => "aegis",
            Dispatch::Cublaslt => "cublaslt",
        }
    }

    pub fn parse(s: &str) -> Option<Dispatch> {
        match s {
            "aegis" => Some(Dispatch::Aegis),
            "cublaslt" => Some(Dispatch::Cublaslt),
            _ => None,
        }
    }

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
/// algo_id. When cublaslt_best's algorithm keeps FP16 split-K partials, the
/// harness adds `dispatch`: the fastest passing algorithm with an FP32
/// reduction (src/lt_algo.h reduction_ok), which is the one emitted.
pub fn lt_config(json: &str) -> Result<Option<String>> {
    let top: Value = serde_json::from_str(json).context("winners.lt_algo")?;
    let v = if top["dispatch"].is_object() { &top["dispatch"] } else { &top };
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
    lt_ci: Option<(f64, f64)>,
    /// Paired median ratio cublaslt_best / winner time (the number of record).
    lt_median: Option<f64>,
    dispatch: Dispatch,
    /// cublaslt_best's algorithm at this shape: the LtAlgoConfig initializer.
    lt: Option<String>,
}

/// The generated header text (format unchanged from the original Python tuner
/// apart from line 1: generator name, host, time).
pub fn header_text(con: &rusqlite::Connection, arch: i64, generated: &str) -> Result<(String, usize)> {
    let mut stmt = con.prepare(
        "SELECT w.shape_bucket, k.impl, w.pct_of_cublas, w.tied, w.tie_set, w.pick_reason, w.pct_of_cublaslt_best, \
                w.tie_method, w.ratio_ci_lo, w.ratio_ci_hi, w.lt_ratio_ci_lo, w.lt_ratio_ci_hi, w.lt_algo, \
                w.lt_ratio_median, w.dispatch \
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
            (r.get::<_, Option<f64>>(13)?, r.get::<_, Option<String>>(14)?),
        ))
    })? {
        let (bucket, impl_, pct, tied, tie_set, reason, pct_lt, method, ci_lo, ci_hi, lt_ci, lt_algo, (lt_median, stored)) = row?;
        // Paired tie test: runner-up / winner, 95% CI of the median ratio.
        let ci_note = match (method.as_deref(), ci_lo, ci_hi) {
            (Some("paired" | "paired_swap"), Some(lo), Some(hi)) => format!(" [tie test runner-up / winner {lo:.3}-{hi:.3}]"),
            _ => String::new(),
        };
        let lt_ci = match lt_ci {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        };
        // The number of record leads (bars.rs): the paired median and CI; the unpaired
        // median follows, labelled, and is flagged when it lies outside the CI by more
        // than bars::MEDIAN_CI_DISAGREE_PTS.
        let paired = match (lt_median, lt_ci) {
            (Some(median), Some((ci_lo, ci_hi))) => Some(crate::bars::PairedPct { median, ci_lo, ci_hi }),
            _ => None,
        };
        let lt_note = match (&paired, pct_lt) {
            (Some(p), Some(u)) => format!(
                "kernel {} of cublaslt_best (paired); unpaired median {u:.1}%{}",
                p.text(),
                if crate::bars::median_disagrees(u, p) { " (DISAGREES with the paired CI)" } else { "" }
            ),
            (Some(p), None) => format!("kernel {} of cublaslt_best (paired)", p.text()),
            (None, Some(u)) => format!("no paired test; unpaired median {u:.1}% of cublaslt_best"),
            (None, None) => String::new(),
        };
        // Tie status: which configs the measurements could not separate, and why this one.
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
        let lt = lt_algo.as_deref().map(lt_config).transpose()?.flatten();
        // The stored dispatch (schema v7); rows from before it: the same rule, now.
        let dispatch = stored.as_deref().and_then(Dispatch::parse).unwrap_or_else(|| dispatch(lt_ci, pct_lt));
        entries.push(Entry { m, n, k, impl_, pct, tie_note, lt_ci, lt_median, dispatch, lt });
    }
    entries.sort_by_key(|e| (e.n, e.k, e.m)); // stable, like Python's sort

    let mut lines = vec![
        format!("// Generated by {generated}."),
        "// Do not edit. Regenerate on the device after tuning.".to_string(),
        format!("// Tuned GEMM winners for sm_{arch}, read by aegis::matmul (src/matmul.cu)."),
        "// Row: M, N, K, tuned impl, dispatch (Aegis: the kernel; Cublaslt: the dispatch floor),".to_string(),
        "// kernel % of cublaslt_best (paired median; 0: no paired test), paired CI of cublaslt_best / winner time".to_string(),
        "// (0, 0: none). Comments: the paired figure first; unpaired medians are labelled (aegis_perf::bars).".to_string(),
        "// \"[tie test ...]\" in a comment: the paired CI of runner-up / winner time.".to_string(),
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
            let cublas = e.pct.map(|p| format!("; {p:.1}% of cuBLAS (unpaired)")).unwrap_or_default();
            let body = format!("{tie_note}{cublas}{floor}");
            let note = if body.is_empty() { String::new() } else { format!("  // {}", body.trim_start_matches("; ")) };
            let (lo, hi) = e.lt_ci.unwrap_or((0.0, 0.0));
            lines.push(format!(
                "    {{{m}, {n}, {k}, \"{impl_}\", aegis::detail::Dispatch::{}, {:.1}, {lo:.4}, {hi:.4}, {}}},{note}",
                e.dispatch.cpp(),
                e.lt_median.map_or(0.0, |r| r * 100.0),
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
        // cublaslt_best with FP16 partials (OUTPUT_TYPE): the compliant `dispatch` algorithm is emitted.
        let j = r#"{"algo_id":21,"splitk":4,"reduction":4,"dispatch":{"algo_id":67,"tile_id":412,"stages_id":35,"splitk":1,"reduction":0,"median_us":9.5}}"#;
        assert_eq!(lt_config(j).unwrap().as_deref(), Some("{67, 412, 35, 1, 0, 0, 0, 0, 0}"));
    }

    /// The header's number of record is the paired one: field 5 is the paired
    /// median %, the comment leads with it, the unpaired median follows,
    /// labelled, and a disagreement over bars::MEDIAN_CI_DISAGREE_PTS is flagged.
    #[test]
    fn header_leads_with_the_paired_ratio() {
        let p = std::env::temp_dir().join(format!("aegis-tune-hdr-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let con = crate::db::connect(&p).unwrap();
        con.execute_batch(
            "INSERT INTO runs (run_id, device_id, sm_arch, started_at) VALUES ('r', 'd', 121, '1');
             INSERT INTO kernels (kernel_id, op, impl, config_json) VALUES ('ka', 'gemm', 'mma_a', '{}'), ('kb', 'gemm', 'mma_b', '{}'), ('kc', 'gemm', 'mma_c', '{}');
             INSERT INTO winners (sm_arch, op, dtype, shape_bucket, kernel_id, run_id, median_us, pct_of_cublas, tied,
                                  protocol_version, pct_of_cublaslt_best, tie_method, lt_ratio_ci_lo, lt_ratio_ci_hi,
                                  lt_ratio_median, dispatch) VALUES
               (121, 'gemm', 'f16f32f16', '512x4864x896', 'ka', 'r', 1.0, 85.1, 0, 2, 82.2, 'paired', 0.7851, 0.7903, 0.788, 'cublaslt'),
               (121, 'gemm', 'f16f32f16', '1x896x896', 'kb', 'r', 1.0, 105.3, 0, 2, 105.3, 'paired', 1.0535, 1.0569, 1.054, 'aegis'),
               (121, 'gemm', 'f16f32f16', '64x64x64', 'kc', 'r', 1.0, NULL, 0, 2, 79.0, 'band', NULL, NULL, NULL, NULL);",
        )
        .unwrap();
        let (text, n) = header_text(&con, 121, "test").unwrap();
        assert_eq!(n, 3);
        let row = |shape: &str| text.lines().find(|l| l.trim_start().starts_with(&format!("{{{shape},"))).unwrap().to_string();
        let r = row("512, 4864, 896");
        assert!(r.contains("Dispatch::Cublaslt, 78.8, 0.7851, 0.7903"), "{r}");
        let c = r.split("// ").nth(1).unwrap();
        assert!(c.starts_with("kernel 78.8% (78.5-79.0%) of cublaslt_best (paired); unpaired median 82.2% (DISAGREES"), "{c}");
        assert!(c.contains("85.1% of cuBLAS (unpaired)") && c.ends_with("below the 80% floor"), "{c}");
        let r = row("1, 896, 896");
        assert!(r.contains("Dispatch::Aegis, 105.4,") && r.contains("// kernel 105.4% (105.4-105.7%) of cublaslt_best (paired); unpaired median 105.3%;"), "{r}");
        assert!(!r.contains("DISAGREES"));
        // No paired test: field 5 is 0, and the comment says so. Its dispatch comes
        // from the rule (no stored one): the unpaired 79.0% is below the floor.
        let r = row("64, 64, 64");
        assert!(r.contains("Dispatch::Cublaslt, 0.0, 0.0000, 0.0000"), "{r}");
        assert!(r.contains("// no paired test; unpaired median 79.0% of cublaslt_best"), "{r}");
        drop(con);
        std::fs::remove_file(&p).ok();
    }
}
