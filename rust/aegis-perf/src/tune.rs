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
}

pub fn cmd_tune(a: &TuneArgs) -> Result<()> {
    let exe = a.build.join("aegis_tune");
    if !exe.exists() {
        bail!("{} not found; run cmake --build build first", exe.display());
    }
    std::fs::create_dir_all(a.results)?;
    let host = db::hostname();
    let manifest = a.results.join(format!("manifest-{host}.json"));
    if !manifest.exists() {
        let out = Command::new(a.build.join("aegis_manifest")).output().context("run aegis_manifest")?;
        std::fs::write(&manifest, out.stdout)?;
    }

    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let out_path = a.results.join(format!("tune-{host}-{stamp}.jsonl"));
    let mut cmd = Command::new(&exe);
    cmd.arg("--shapes").arg(a.shapes);
    if let Some(s) = a.seed {
        cmd.args(["--seed", &s.to_string()]);
    }
    if let Some(p) = &a.prefix {
        cmd.args(["--prefix", p]);
    }
    if a.allow_busy {
        cmd.arg("--allow-busy");
    }
    let t0 = Instant::now();
    let out = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // progress, seed and skipped configs
        .output()
        .context("run aegis_tune")?;
    let wall = t0.elapsed().as_secs_f64();
    if !out.status.success() {
        bail!("aegis_tune failed ({})", out.status);
    }
    std::fs::write(&out_path, &out.stdout)?;

    db::cmd_ingest(a.db, &out_path, Some(&manifest), None, None)?;
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
    for line in jsonl.lines().filter(|l| !l.trim().is_empty()) {
        let r: Value = serde_json::from_str(line)?;
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
        "{:>5} {:>5} {:>5}  {:<30} {:>9} {:>9}  {:<30} {:>6}  {:>9}",
        "M", "N", "K", "winner", "% cuBLAS", "% LtBest", "runner-up", "gap", "warm %"
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
        let (ru, gap) = match ok.get(1) {
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
            "{m:>5} {n:>5} {k:>5}  {:<30} {pct} {pct_lt}  {ru:<30} {gap}  {warm_pct}",
            w["impl"].as_str().unwrap_or("?")
        )?;
    }
    writeln!(
        out,
        "\n% cuBLAS: cuBLAS median / winner median (100% = parity). gap: how much slower the runner-up's median is than the winner's;\n\
         \"tied\": the runner-up's median is inside the winner's p10-p90 (this run only; the DB winner is picked by the tie-break).\n\
         % LtBest: the same against cublaslt_best. Ranked on cache_mode=cold rows.\n\
         warm %: cublaslt_best warm median / winner warm median (protocol v2)."
    )?;
    Ok(out)
}

/// The generated header text (format unchanged from the original Python tuner
/// apart from line 1: generator name, host, time).
pub fn header_text(con: &rusqlite::Connection, arch: i64, generated: &str) -> Result<(String, usize)> {
    let mut stmt = con.prepare(
        "SELECT w.shape_bucket, k.impl, w.pct_of_cublas, w.tied, w.tie_set, w.pick_reason, w.pct_of_cublaslt_best \
         FROM winners w \
         JOIN kernels k USING(kernel_id) WHERE w.sm_arch = ? AND w.op = 'gemm' \
         AND w.dtype = 'f16f32f16'",
    )?;
    let mut entries: Vec<(i64, i64, i64, String, Option<f64>, String)> = Vec::new();
    for row in stmt.query_map([arch], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<f64>>(2)?,
            r.get::<_, i64>(3)?,
            r.get::<_, Option<String>>(4)?,
            r.get::<_, Option<String>>(5)?,
            r.get::<_, Option<f64>>(6)?,
        ))
    })? {
        let (bucket, impl_, pct, tied, tie_set, reason, pct_lt) = row?;
        // Tie status: which configs the measurements could not separate, and why this one.
        let lt_note = pct_lt.map(|p| format!(", {p:.1}% of cublaslt_best")).unwrap_or_default();
        let tie_note = if tied != 0 {
            let others: Vec<String> = serde_json::from_str::<Vec<String>>(tie_set.as_deref().unwrap_or("[]"))?
                .into_iter()
                .filter(|i| *i != impl_)
                .collect();
            format!("{lt_note}; TIED with {} ({})", others.join(", "), reason.unwrap_or_default())
        } else {
            lt_note
        };
        let dims: Vec<i64> = bucket.split('x').map(str::parse).collect::<Result<_, _>>().context("shape_bucket")?;
        let [m, n, k] = dims[..] else { bail!("bad shape_bucket {bucket}") };
        entries.push((m, n, k, impl_, pct, tie_note));
    }
    entries.sort_by_key(|e| (e.1, e.2, e.0)); // stable, like Python's sort

    let mut lines = vec![
        format!("// Generated by {generated}."),
        "// Do not edit. Regenerate on the device after tuning.".to_string(),
        format!("// Tuned GEMM winners for sm_{arch}, read by aegis::matmul (src/matmul.cu)."),
        "#pragma once".to_string(),
        "#include \"aegis/matmul.h\"".to_string(),
        String::new(),
    ];
    if entries.is_empty() {
        lines.push(format!("// No winners in the perf DB for sm_{arch}; aegis::matmul uses cuBLAS."));
    } else {
        lines.push(format!("#define AEGIS_WINNERS_SM{arch} 1"));
        lines.push(format!("inline constexpr aegis::detail::Winner kWinners_sm{arch}[] = {{"));
        for (m, n, k, impl_, pct, tie_note) in &entries {
            let note = match pct {
                Some(p) => format!("  // {p:.1}% of cuBLAS{tie_note}"),
                None if !tie_note.is_empty() => format!("  // {}", tie_note.trim_start_matches("; ")),
                None => String::new(),
            };
            lines.push(format!("    {{{m}, {n}, {k}, \"{impl_}\"}},{note}"));
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
