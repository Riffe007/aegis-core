//! Fuzz results in the perf DB (schema v7; core-request #10): `aegis-perf
//! ingest-fuzz` parses one aegis_fuzz output (its build line, its summary line
//! and the FAIL / note case lines with their indented details) into
//! `fuzz_runs` / `fuzz_cases`; `aegis-perf fuzz-verdict` records an
//! adjudication of a case (e.g. the FP64 ulp comparison of case 24632, LOG
//! 2026-10-15); [`fuzz_runs`] reads them back.
//!
//! The parser is the one definition of aegis_fuzz's text format, next to the
//! tool that writes it, so readers never scrape it themselves.
use std::path::Path;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

use crate::db::{self, BuildId};

/// One aegis_fuzz run, parsed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    pub build: Option<BuildId>,
    pub cases: i64,
    pub failures: i64,
    pub fp64_reference: Option<i64>,
    pub cublaslt_reference: Option<i64>,
    pub nan_inf: Option<i64>,
    pub misaligned: Option<i64>,
    pub skipped: Option<i64>,
    pub overruled: Option<i64>,
    pub seconds: Option<f64>,
    pub seed: Option<i64>,
    pub reported: Vec<ParsedCase>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParsedCase {
    pub case_no: i64,
    /// "FAIL" or "note".
    pub kind: String,
    pub summary: String,
    pub detail: Option<String>,
}

/// The integer before `word` in `s` ("37333 CPU-FP64 reference" -> 37333).
fn count_before(s: &str, word: &str) -> Option<i64> {
    let i = s.find(word)?;
    s[..i].trim_end().rsplit([' ', '(', ',', ';']).next()?.parse().ok()
}

/// Parse aegis_fuzz's stdout. The summary line is required (it is the last
/// `aegis_fuzz:` line); the build line is absent for builds from before schema v7.
pub fn parse(text: &str) -> Result<Parsed> {
    let mut p = Parsed::default();
    let mut summary = None;
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if let Some(j) = l.strip_prefix("aegis_fuzz build: ") {
            let v: Value = serde_json::from_str(j).context("aegis_fuzz build line")?;
            p.build = db::build_of_row(&v)?;
        } else if l.starts_with("aegis_fuzz: ") {
            summary = Some(l);
        } else if let Some((kind, rest)) = l.strip_prefix("FAIL case ").map(|r| ("FAIL", r)).or_else(|| l.strip_prefix("note case ").map(|r| ("note", r))) {
            let case_no: i64 = rest.split_whitespace().next().unwrap_or("").parse().with_context(|| format!("case line: {l}"))?;
            let summary = rest.split_once("): ").map_or(rest, |(_, s)| s).trim().to_string();
            let mut detail = Vec::new();
            while i + 1 < lines.len() && lines[i + 1].starts_with("  ") {
                i += 1;
                detail.push(lines[i].trim());
            }
            p.reported.push(ParsedCase {
                case_no,
                kind: kind.to_string(),
                summary,
                detail: (!detail.is_empty()).then(|| detail.join("\n")),
            });
        }
        i += 1;
    }
    let s = summary.context("no 'aegis_fuzz: N cases, F failures ...' summary line (the run did not finish)")?;
    let body = s.trim_start_matches("aegis_fuzz: ");
    p.cases = count_before(body, " cases").context("summary: cases")?;
    p.failures = count_before(body, " failures")
        .or_else(|| count_before(body, " corruptions NOT caught"))
        .context("summary: failures")?;
    p.fp64_reference = count_before(body, " CPU-FP64 reference");
    p.cublaslt_reference = count_before(body, " cuBLASLt;");
    p.nan_inf = count_before(body, " with NaN/Inf");
    p.misaligned = count_before(body, " misaligned");
    p.skipped = count_before(body, " aegis::matmul cases skipped");
    p.overruled = count_before(body, " cuBLASLt-reference failures overruled");
    p.seconds = body.rsplit_once(" s, seed").and_then(|(a, _)| a.rsplit(", ").next()?.parse().ok());
    p.seed = body.rsplit_once("seed ").and_then(|(_, s)| s.trim().parse().ok());
    Ok(p)
}

pub struct IngestArgs<'a> {
    pub db: &'a Path,
    pub file: &'a Path,
    /// "nightly" or "manual".
    pub kind: &'a str,
    pub started_at: Option<String>,
    pub device_id: String,
    pub sm_arch: Option<i64>,
    pub allow_dirty: bool,
}

/// Store one aegis_fuzz output; returns its fuzz_id.
pub fn ingest(a: &IngestArgs) -> Result<String> {
    if !["nightly", "manual"].contains(&a.kind) {
        bail!("fuzz run kind '{}': want nightly or manual", a.kind);
    }
    let text = std::fs::read_to_string(a.file).with_context(|| format!("read {}", a.file.display()))?;
    let p = parse(&text).with_context(|| a.file.display().to_string())?;
    db::check_build(&format!("ingest-fuzz {}", a.file.display()), p.build.as_ref(), a.allow_dirty)?;
    let mut con = db::connect(a.db)?;
    let tx = con.transaction()?;
    let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    let started = a
        .started_at
        .clone()
        .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string());
    tx.execute(
        "INSERT INTO fuzz_runs (fuzz_id, kind, device_id, sm_arch, build_sha, build_dirty, allow_dirty, seed, cases, \
         failures, fp64_reference, cublaslt_reference, nan_inf, misaligned, skipped, overruled, seconds, started_at, source) \
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        params![
            id, a.kind, a.device_id, a.sm_arch, p.build.as_ref().map(|b| b.sha.as_str()), p.build.as_ref().map(|b| b.dirty as i64),
            (a.allow_dirty && p.build.as_ref().is_none_or(|b| b.dirty)) as i64, p.seed, p.cases, p.failures,
            p.fp64_reference, p.cublaslt_reference, p.nan_inf, p.misaligned, p.skipped, p.overruled, p.seconds, started,
            a.file.display().to_string(),
        ],
    )?;
    for c in &p.reported {
        tx.execute(
            "INSERT INTO fuzz_cases (fuzz_id, case_no, kind, summary, detail) VALUES (?,?,?,?,?)",
            params![id, c.case_no, c.kind, c.summary, c.detail],
        )?;
    }
    tx.commit()?;
    println!(
        "fuzz run {id}: {} cases, {} failures, {} reported cases (seed {})",
        p.cases,
        p.failures,
        p.reported.len(),
        p.seed.map_or("?".to_string(), |s| s.to_string())
    );
    Ok(id)
}

/// An adjudication of one reported case.
pub struct Verdict<'a> {
    pub fuzz_id: &'a str,
    pub case_no: i64,
    /// What the case is, e.g. "a reference bug" or "ours".
    pub verdict: &'a str,
    /// ulp error at the failing element against FP64: ours and the reference's.
    pub ours_ulp: Option<f64>,
    pub ref_ulp: Option<f64>,
    /// Where the adjudication is written up (e.g. "docs/LOG.md 2026-10-15 — Fuzz case 24632 ...").
    pub source: &'a str,
}

pub fn record_verdict(db: &Path, v: &Verdict) -> Result<()> {
    if v.verdict.trim().is_empty() || v.source.trim().is_empty() {
        bail!("a verdict needs its text and its source");
    }
    let con = db::connect(db)?;
    let n = con.execute(
        "UPDATE fuzz_cases SET verdict = ?3, ours_ulp = ?4, ref_ulp = ?5, verdict_source = ?6 WHERE fuzz_id = ?1 AND case_no = ?2",
        params![v.fuzz_id, v.case_no, v.verdict, v.ours_ulp, v.ref_ulp, v.source],
    )?;
    if n == 0 {
        bail!("fuzz run {} has no reported case {}", v.fuzz_id, v.case_no);
    }
    println!("fuzz run {} case {}: {}", v.fuzz_id, v.case_no, v.verdict);
    Ok(())
}

/// One stored fuzz run with its reported cases.
#[derive(Clone, Debug, PartialEq)]
pub struct FuzzRun {
    pub fuzz_id: String,
    pub kind: String,
    pub device_id: Option<String>,
    pub sm_arch: Option<i64>,
    pub build_sha: Option<String>,
    pub build_dirty: Option<bool>,
    pub allow_dirty: bool,
    pub seed: Option<i64>,
    pub cases: i64,
    pub failures: i64,
    pub fp64_reference: Option<i64>,
    pub cublaslt_reference: Option<i64>,
    pub nan_inf: Option<i64>,
    pub misaligned: Option<i64>,
    pub skipped: Option<i64>,
    pub overruled: Option<i64>,
    pub seconds: Option<f64>,
    pub started_at: String,
    pub source: Option<String>,
    pub reported: Vec<FuzzCase>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuzzCase {
    pub case_no: i64,
    pub kind: String,
    pub summary: String,
    pub detail: Option<String>,
    pub verdict: Option<String>,
    pub ours_ulp: Option<f64>,
    pub ref_ulp: Option<f64>,
    pub verdict_source: Option<String>,
}

/// Fuzz runs, oldest first. Runs from a dirty build or stored with
/// --allow-dirty only with `include_dirty`.
pub fn fuzz_runs(con: &Connection, include_dirty: bool) -> Result<Vec<FuzzRun>> {
    let mut stmt = con.prepare(
        "SELECT fuzz_id, kind, device_id, sm_arch, build_sha, build_dirty, allow_dirty, seed, cases, failures, \
                fp64_reference, cublaslt_reference, nan_inf, misaligned, skipped, overruled, seconds, started_at, source \
         FROM fuzz_runs WHERE ?1 OR (allow_dirty = 0 AND coalesce(build_dirty, 0) = 0) ORDER BY started_at, fuzz_id",
    )?;
    let mut runs: Vec<FuzzRun> = stmt
        .query_map([include_dirty], |r| {
            Ok(FuzzRun {
                fuzz_id: r.get(0)?,
                kind: r.get(1)?,
                device_id: r.get(2)?,
                sm_arch: r.get(3)?,
                build_sha: r.get(4)?,
                build_dirty: r.get::<_, Option<i64>>(5)?.map(|d| d != 0),
                allow_dirty: r.get::<_, i64>(6)? != 0,
                seed: r.get(7)?,
                cases: r.get(8)?,
                failures: r.get(9)?,
                fp64_reference: r.get(10)?,
                cublaslt_reference: r.get(11)?,
                nan_inf: r.get(12)?,
                misaligned: r.get(13)?,
                skipped: r.get(14)?,
                overruled: r.get(15)?,
                seconds: r.get(16)?,
                started_at: r.get(17)?,
                source: r.get(18)?,
                reported: Vec::new(),
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut cases = con.prepare(
        "SELECT case_no, kind, summary, detail, verdict, ours_ulp, ref_ulp, verdict_source FROM fuzz_cases \
         WHERE fuzz_id = ? ORDER BY case_no",
    )?;
    for run in &mut runs {
        run.reported = cases
            .query_map([&run.fuzz_id], |r| {
                Ok(FuzzCase {
                    case_no: r.get(0)?,
                    kind: r.get(1)?,
                    summary: r.get(2)?,
                    detail: r.get(3)?,
                    verdict: r.get(4)?,
                    ours_ulp: r.get(5)?,
                    ref_ulp: r.get(6)?,
                    verdict_source: r.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
    }
    Ok(runs)
}

/// The sm_arch of results/manifest-<host>.json, if there is one.
pub fn manifest_arch(results: &Path) -> Option<i64> {
    let m = results.join(format!("manifest-{}.json", db::hostname()));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(m).ok()?).ok()?;
    v["sm_arch"].as_i64()
}

/// A stored run by id (any build state).
pub fn fuzz_run(con: &Connection, fuzz_id: &str) -> Result<Option<FuzzRun>> {
    let exists: Option<i64> = con.query_row("SELECT 1 FROM fuzz_runs WHERE fuzz_id = ?", [fuzz_id], |r| r.get(0)).optional()?;
    if exists.is_none() {
        return Ok(None);
    }
    Ok(fuzz_runs(con, true)?.into_iter().find(|r| r.fuzz_id == fuzz_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2026-10-14 nightly's output (results/nightly/fuzz-20261014-201534.txt,
    /// abridged), plus the build line aegis_fuzz prints since schema v7.
    const NIGHT: &str = "\
ok   case 1: gemv_tc_16x64x64_1x4_s4_k1 1x896x896 RR  max_ratio 0.120
FAIL case 24632 (--seed 20261014 --case 24632): mma_pipe_128x128x32_2x4_s3 1951x624x3159 CR lda=1956 ldb=625 ldc=635 alpha=-0.0126576 beta=-1 batch=1 strides=0/0/0 offsets=0/2/0
  vs cuBLASLt: max_abs_err 0.00293 > tol 0.00244 at (1373, 488): got 1.00684, ref 1.00977
  cuBLASLt plan: heuristic, algo 21 splitk 3 reduction 2 (hint: none)
note case 31000 (--seed 20261014 --case 31000): matmul 17x900x13 RC
  vs cuBLASLt: fail; vs FP64: pass (max 0.80 x tol): cuBLASLt-reference failure overruled
aegis_fuzz build: {\"build_sha\":\"0123456789abcdef0123456789abcdef01234567\",\"build_dirty\":false}
aegis_fuzz: 50000 cases, 1 failures (37333 CPU-FP64 reference, 12667 cuBLASLt; 2557 with NaN/Inf, 24532 misaligned, 1 cuBLASLt-reference failures overruled by FP64), 1003.0 s, seed 20261014
";

    #[test]
    fn parses_the_summary_build_and_cases() {
        let p = parse(NIGHT).unwrap();
        assert_eq!((p.cases, p.failures, p.seed, p.seconds), (50000, 1, Some(20261014), Some(1003.0)));
        assert_eq!((p.fp64_reference, p.cublaslt_reference, p.nan_inf, p.misaligned), (Some(37333), Some(12667), Some(2557), Some(24532)));
        assert_eq!((p.skipped, p.overruled), (None, Some(1)));
        assert_eq!(p.build, Some(BuildId::new("0123456789abcdef0123456789abcdef01234567", false).unwrap()));
        assert_eq!(p.reported.len(), 2);
        let f = &p.reported[0];
        assert_eq!((f.case_no, f.kind.as_str()), (24632, "FAIL"));
        assert!(f.summary.starts_with("mma_pipe_128x128x32_2x4_s3 1951x624x3159 CR"), "{}", f.summary);
        assert!(f.detail.as_deref().unwrap().contains("got 1.00684, ref 1.00977") && f.detail.as_deref().unwrap().contains("algo 21"));
        assert_eq!((p.reported[1].case_no, p.reported[1].kind.as_str()), (31000, "note"));
        // A pre-v7 output: no build line. A run cut short: no summary.
        assert_eq!(parse(&NIGHT.replace("aegis_fuzz build: ", "x")).unwrap().build, None);
        assert!(parse("FAIL case 1 (--seed 1 --case 1): x\n").is_err());
    }

    #[test]
    fn ingest_refuses_unknown_builds_records_verdicts_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("aegis-fuzz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("p.db");
        let (clean, old) = (dir.join("clean.txt"), dir.join("old.txt"));
        std::fs::write(&clean, NIGHT).unwrap();
        std::fs::write(&old, NIGHT.replace("aegis_fuzz build: ", "x")).unwrap();
        fn args<'a>(db: &'a Path, file: &'a Path, allow_dirty: bool) -> IngestArgs<'a> {
            IngestArgs {
                db,
                file,
                kind: "nightly",
                started_at: Some("2026-10-14T20:15:34+00:00".into()),
                device_id: "spark".into(),
                sm_arch: Some(121),
                allow_dirty,
            }
        }
        let err = ingest(&args(&db, &old, false)).unwrap_err().to_string();
        assert!(err.contains("--allow-dirty"), "{err}");
        let id = ingest(&args(&db, &clean, false)).unwrap();
        ingest(&args(&db, &old, true)).unwrap();
        assert!(record_verdict(&db, &Verdict { fuzz_id: &id, case_no: 7, verdict: "x", ours_ulp: None, ref_ulp: None, source: "s" }).is_err());
        record_verdict(
            &db,
            &Verdict {
                fuzz_id: &id,
                case_no: 24632,
                verdict: "a reference bug",
                ours_ulp: Some(0.43),
                ref_ulp: Some(2.57),
                source: "docs/LOG.md 2026-10-15 — Fuzz case 24632: a cuBLASLt reference error, not ours",
            },
        )
        .unwrap();
        let con = db::connect_ro(&db).unwrap();
        let runs = fuzz_runs(&con, false).unwrap();
        assert_eq!(runs.len(), 1, "the unknown-build run is excluded by default");
        let r = &runs[0];
        assert_eq!((r.kind.as_str(), r.cases, r.failures, r.sm_arch, r.build_dirty), ("nightly", 50000, 1, Some(121), Some(false)));
        let c = r.reported.iter().find(|c| c.case_no == 24632).unwrap();
        assert_eq!((c.verdict.as_deref(), c.ours_ulp, c.ref_ulp), (Some("a reference bug"), Some(0.43), Some(2.57)));
        let all = fuzz_runs(&con, true).unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|r| r.allow_dirty && r.build_sha.is_none()));
        assert_eq!(fuzz_run(&con, &id).unwrap().unwrap().fuzz_id, id);
        drop(con);
        std::fs::remove_dir_all(&dir).ok();
    }
}
