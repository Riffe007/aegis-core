//! `regress`: the nightly regression check. For every shape in a winners header
//! (what `aegis::matmul` dispatches), the dispatched impl's % of cublaslt_best
//! in one run (a regression sweep), against the newest earlier clean
//! measurement of the same pair. A drop of more than 3% (relative) is a
//! "watch": one unpaired measurement each side cannot tell a slowdown from
//! run-to-run noise (2026-10-16/17: 16x4864x896 99.4 -> 96.1 -> 99.3%). It is a
//! regression (Production bar: the regression gate's 3%) only when confirmed:
//! - two consecutive nightlies: this run and the previous one, itself a sweep
//!   (a run without paired tests, like the nightly's), are both more than 3%
//!   below the run before the previous one, or
//! - a paired-ab (`paired_results`, same impl on both sides, clean) started
//!   after the previous run, with the B build at least 3% slower by median and
//!   its CI entirely above 1.0.
//!
//! Reads only; never touches winners headers, arch files or the winners table.
use std::path::Path;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

/// Relative drop in % of cublaslt_best that counts as a regression.
pub const REGRESSION: f64 = 0.03;

/// (M, N, K, impl) per row of a winners header (`{M, N, K, "impl", dispatch, ...},`; the
/// tuned impl, also on a row dispatched to cuBLASLt by the dispatch floor).
pub fn header_winners(text: &str) -> Vec<(i64, i64, i64, String)> {
    let mut v = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let body = line.trim_start_matches('{').split('}').next().unwrap_or("");
        let parts: Vec<&str> = body.split(',').map(str::trim).collect();
        if parts.len() < 4 {
            continue;
        }
        let (Ok(m), Ok(n), Ok(k)) = (parts[0].parse(), parts[1].parse(), parts[2].parse()) else { continue };
        v.push((m, n, k, parts[3].trim_matches('"').to_string()));
    }
    v
}

/// Clean v2 cold median of `impl_` at a shape in one run (passing rows only).
fn median_in(con: &Connection, run: &str, impl_: &str, (m, n, k): (i64, i64, i64)) -> Result<Option<f64>> {
    Ok(con
        .query_row(
            "SELECT min(x.median_us) FROM measurements x JOIN kernels kc USING(kernel_id) \
             WHERE x.run_id = ?1 AND kc.impl = ?2 AND x.m = ?3 AND x.n = ?4 AND x.k = ?5 AND x.pass = 1 \
               AND x.protocol_version = 2 AND x.cache_mode = 'cold' AND x.allow_busy = 0 \
               AND coalesce(x.disturbed, 0) = 0",
            params![run, impl_, m, n, k],
            |r| r.get::<_, Option<f64>>(0),
        )
        .optional()?
        .flatten())
}

/// % of cublaslt_best of `impl_` in `run`, if both were measured cleanly there.
fn pct_in(con: &Connection, run: &str, impl_: &str, shape: (i64, i64, i64)) -> Result<Option<f64>> {
    Ok(match (median_in(con, run, impl_, shape)?, median_in(con, run, "cublaslt_best", shape)?) {
        (Some(ours), Some(lt)) if ours > 0.0 => Some(100.0 * lt / ours),
        _ => None,
    })
}

/// One shape's comparison.
#[derive(Debug, Default)]
pub struct ShapeCheck {
    pub shape: (i64, i64, i64),
    pub impl_: String,
    pub pct: Option<f64>,
    /// (earlier run, its %): the newest earlier clean measurement of the pair.
    pub before: Option<(String, f64)>,
    /// The one before that.
    pub before2: Option<(String, f64)>,
    /// `before` is a sweep (no paired tests), i.e. a nightly, not a tune.
    pub before_is_sweep: bool,
    /// A paired-ab confirming a slowdown: (paired_results id, ratio B/A, CI).
    pub paired: Option<(i64, f64, f64, f64)>,
}

fn below(p: f64, reference: f64) -> bool {
    p < reference * (1.0 - REGRESSION)
}

impl ShapeCheck {
    /// More than 3% below the previous run: a watch at least.
    pub fn dropped(&self) -> bool {
        matches!((self.pct, &self.before), (Some(p), Some((_, b))) if below(p, *b))
    }

    /// This run and the previous one (a sweep) both more than 3% below the run before them.
    pub fn two_consecutive(&self) -> bool {
        self.before_is_sweep
            && matches!((self.pct, &self.before, &self.before2),
                 (Some(p), Some((_, b)), Some((_, b2))) if below(p, *b2) && below(*b, *b2))
    }

    /// A confirmed regression: two consecutive runs, or a paired-ab.
    pub fn regressed(&self) -> bool {
        (self.dropped() || self.two_consecutive()) && (self.two_consecutive() || self.paired.is_some())
    }

    /// A single unpaired drop, not (yet) confirmed.
    pub fn watch(&self) -> bool {
        self.dropped() && !self.regressed()
    }

    fn why(&self) -> &'static str {
        if self.two_consecutive() { "2 consecutive nightlies" } else { "confirmed by paired-ab" }
    }
}

/// The newest clean paired-ab of `impl_` against itself (two builds) at `shape`,
/// started after run `prev`, that shows B at least 3% slower with a CI entirely above 1.0.
fn paired_slowdown(con: &Connection, prev: &str, impl_: &str, (m, n, k): (i64, i64, i64)) -> Result<Option<(i64, f64, f64, f64)>> {
    Ok(con
        .query_row(
            "SELECT p.id, p.ratio, p.ci_lo, p.ci_hi FROM paired_results p \
             WHERE p.shape = ?1 AND p.impl_a = ?2 AND p.impl_b = ?2 AND coalesce(p.disturbed, 0) = 0 \
               AND p.started_at > (SELECT started_at FROM runs WHERE run_id = ?3) \
             ORDER BY p.started_at DESC LIMIT 1",
            params![format!("{m}x{n}x{k}"), impl_, prev],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?)),
        )
        .optional()?
        .filter(|&(_, ratio, lo, _)| ratio >= 1.0 + REGRESSION && lo > 1.0))
}

pub fn check(con: &Connection, run: &str, winners: &[(i64, i64, i64, String)]) -> Result<Vec<ShapeCheck>> {
    let (arch, started): (i64, String) = con
        .query_row("SELECT sm_arch, started_at FROM runs WHERE run_id = ?", [run], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?
        .with_context(|| format!("no run {run}"))?;
    let mut out = Vec::new();
    for (m, n, k, impl_) in winners {
        let shape = (*m, *n, *k);
        let pct = pct_in(con, run, impl_, shape)?;
        // Newest earlier run (same arch) with a clean measurement of the pair.
        let mut stmt = con.prepare(
            "SELECT DISTINCT r.run_id FROM runs r JOIN measurements x USING(run_id) JOIN kernels kc USING(kernel_id) \
             WHERE r.sm_arch = ?1 AND r.started_at < ?2 AND kc.impl = ?3 AND x.m = ?4 AND x.n = ?5 AND x.k = ?6 \
             ORDER BY r.started_at DESC",
        )?;
        let earlier: Vec<String> =
            stmt.query_map(params![arch, started, impl_, m, n, k], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        let mut found: Vec<(String, f64)> = Vec::new();
        for e in earlier {
            if let Some(p) = pct_in(con, &e, impl_, shape)? {
                found.push((e, p));
                if found.len() == 2 {
                    break;
                }
            }
        }
        let mut found = found.into_iter();
        let (before, before2) = (found.next(), found.next());
        let paired = match &before {
            Some((prev, _)) => paired_slowdown(con, prev, impl_, shape)?,
            None => None,
        };
        let before_is_sweep = match &before {
            Some((prev, _)) => !con.prepare("SELECT 1 FROM paired WHERE run_id = ?")?.exists([prev])?,
            None => false,
        };
        out.push(ShapeCheck { shape, impl_: impl_.clone(), pct, before, before2, before_is_sweep, paired });
    }
    Ok(out)
}

/// The per-shape table plus one summary line (the last line).
pub fn text(run: &str, disturbed: Option<bool>, checks: &[ShapeCheck]) -> String {
    let mut s = String::new();
    for c in checks {
        let (m, n, k) = c.shape;
        let now = c.pct.map_or("-".to_string(), |p| format!("{p:.1}%"));
        let was = c.before.as_ref().map_or("no earlier run".to_string(), |(r, p)| format!("{p:.1}% (run {r})"));
        s += &format!(
            "  {m}x{n}x{k} {}: {now} of cublaslt_best, before {was}{}\n",
            c.impl_,
            if c.regressed() {
                format!("  REGRESSION ({})", c.why())
            } else if c.watch() {
                "  watch (one unpaired drop; a regression if the next run or a paired-ab confirms it)".to_string()
            } else {
                String::new()
            }
        );
    }
    let measured: Vec<&ShapeCheck> = checks.iter().filter(|c| c.pct.is_some()).collect();
    let min = measured.iter().min_by(|a, b| a.pct.unwrap().total_cmp(&b.pct.unwrap()));
    let regressed: Vec<String> = checks
        .iter()
        .filter(|c| c.regressed())
        .map(|c| {
            let (m, n, k) = c.shape;
            let from = if c.two_consecutive() { c.before2.as_ref() } else { c.before.as_ref() };
            format!("{m}x{n}x{k} {:.1}->{:.1}% ({})", from.unwrap().1, c.pct.unwrap(), c.why())
        })
        .collect();
    let watched: Vec<String> = checks
        .iter()
        .filter(|c| c.watch())
        .map(|c| {
            let (m, n, k) = c.shape;
            format!("{m}x{n}x{k} {:.1}->{:.1}%", c.before.as_ref().unwrap().1, c.pct.unwrap())
        })
        .collect();
    s += &format!(
        "run {run}{}: {}/{} shapes measured, min {} of cublaslt_best; {} regression(s) >{:.0}%{}; {} watch{}",
        match disturbed {
            Some(true) => " DISTURBED",
            Some(false) => "",
            None => " (not watched)",
        },
        measured.len(),
        checks.len(),
        min.map_or("-".to_string(), |c| {
            let (m, n, k) = c.shape;
            format!("{:.1}% ({m}x{n}x{k})", c.pct.unwrap())
        }),
        regressed.len(),
        REGRESSION * 100.0,
        if regressed.is_empty() { String::new() } else { format!(": {}", regressed.join(", ")) },
        watched.len(),
        if watched.is_empty() { String::new() } else { format!(": {}", watched.join(", ")) },
    );
    s
}

pub fn cmd_regress(db: &Path, run: &str, header: &Path) -> Result<()> {
    let winners = header_winners(&std::fs::read_to_string(header).with_context(|| format!("read {}", header.display()))?);
    if winners.is_empty() {
        bail!("{}: no winner rows", header.display());
    }
    let con = crate::db::connect_ro(db)?;
    let disturbed: Option<bool> = con
        .query_row("SELECT disturbed FROM runs WHERE run_id = ?", [run], |r| r.get::<_, Option<i64>>(0))
        .optional()?
        .flatten()
        .map(|d| d != 0);
    println!("{}", text(run, disturbed, &check(&con, run, &winners)?));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_rows() {
        let h = "inline constexpr aegis::detail::Winner kWinners_sm121[] = {\n    {256, 256, 256, \"mma_pipe_64x64x32_2x2_s4\"},  // 98.2%\n    {1, 896, 4864, \"x\", aegis::detail::Dispatch::Cublaslt, 74.9, 0.7489, 0.7504},\n};";
        assert_eq!(
            header_winners(h),
            vec![(256, 256, 256, "mma_pipe_64x64x32_2x2_s4".to_string()), (1, 896, 4864, "x".to_string())]
        );
    }

    #[test]
    fn regression_needs_confirmation() {
        let run = |p: f64| Some(("r".to_string(), p));
        let c = |p: f64, b: f64, b2: Option<f64>| ShapeCheck {
            shape: (1, 1, 1),
            impl_: "x".into(),
            pct: Some(p),
            before: run(b),
            before2: b2.and_then(run),
            before_is_sweep: true,
            paired: None,
        };
        // A single unpaired drop over 3% is a watch, not a regression.
        assert!(c(96.9, 100.0, Some(100.0)).watch() && !c(96.9, 100.0, Some(100.0)).regressed());
        assert!(!c(97.1, 100.0, None).watch());
        // 2026-10-16/17 at 16x4864x896: 99.4 -> 96.1 (watch) -> 99.3 (back: nothing).
        assert!(c(96.1, 99.4, None).watch());
        assert!(!c(99.3, 96.1, Some(99.4)).watch() && !c(99.3, 96.1, Some(99.4)).regressed());
        // Two consecutive runs below the run before them: a regression.
        assert!(c(95.0, 96.0, Some(100.0)).regressed() && c(95.0, 96.0, Some(100.0)).two_consecutive());
        // Down twice, but the second run alone is not below the first: still a regression.
        assert!(c(96.5, 96.0, Some(100.0)).regressed());
        // The previous run was a tune, not a nightly: not "2 consecutive nightlies".
        let mut t = c(95.0, 96.0, Some(100.0));
        t.before_is_sweep = false;
        assert!(!t.regressed() && !t.two_consecutive());
        // Confirmed by a paired-ab.
        let mut p = c(96.0, 100.0, None);
        p.paired = Some((9, 1.05, 1.03, 1.07));
        assert!(p.regressed() && !p.watch());
        // No pct: nothing.
        assert!(!ShapeCheck { pct: None, before: run(99.0), ..Default::default() }.regressed());
    }
}
