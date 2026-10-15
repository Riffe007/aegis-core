//! `regress`: the nightly regression check. For every shape in a winners header
//! (what `aegis::matmul` dispatches), the dispatched impl's % of cublaslt_best
//! in one run (a regression sweep), against the newest earlier clean
//! measurement of the same pair. A drop of more than 3% (relative) is a
//! regression (Production bar: the regression gate's 3%). Reads only; never
//! touches winners headers, arch files or the winners table.
use std::path::Path;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

/// Relative drop in % of cublaslt_best that counts as a regression.
pub const REGRESSION: f64 = 0.03;

/// (M, N, K, impl) per row of a winners header (`{M, N, K, "impl"},`).
pub fn header_winners(text: &str) -> Vec<(i64, i64, i64, String)> {
    let mut v = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let body = line.trim_start_matches('{').split('}').next().unwrap_or("");
        let parts: Vec<&str> = body.split(',').map(str::trim).collect();
        if parts.len() != 4 {
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
#[derive(Debug)]
pub struct ShapeCheck {
    pub shape: (i64, i64, i64),
    pub impl_: String,
    pub pct: Option<f64>,
    /// (earlier run, its %).
    pub before: Option<(String, f64)>,
}

impl ShapeCheck {
    pub fn regressed(&self) -> bool {
        matches!((self.pct, &self.before), (Some(p), Some((_, b))) if p < b * (1.0 - REGRESSION))
    }
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
        let mut before = None;
        for e in earlier {
            if let Some(p) = pct_in(con, &e, impl_, shape)? {
                before = Some((e, p));
                break;
            }
        }
        out.push(ShapeCheck { shape, impl_: impl_.clone(), pct, before });
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
            if c.regressed() { "  REGRESSION" } else { "" }
        );
    }
    let measured: Vec<&ShapeCheck> = checks.iter().filter(|c| c.pct.is_some()).collect();
    let min = measured.iter().min_by(|a, b| a.pct.unwrap().total_cmp(&b.pct.unwrap()));
    let regressed: Vec<String> = checks
        .iter()
        .filter(|c| c.regressed())
        .map(|c| {
            let (m, n, k) = c.shape;
            format!("{m}x{n}x{k} {:.1}->{:.1}%", c.before.as_ref().unwrap().1, c.pct.unwrap())
        })
        .collect();
    s += &format!(
        "run {run}{}: {}/{} shapes measured, min {} of cublaslt_best; {} regression(s) >{:.0}%{}",
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
        let h = "inline constexpr aegis::detail::Winner kWinners_sm121[] = {\n    {256, 256, 256, \"mma_pipe_64x64x32_2x2_s4\"},  // 98.2%\n    {1, 896, 4864, \"x\"},\n};";
        assert_eq!(
            header_winners(h),
            vec![(256, 256, 256, "mma_pipe_64x64x32_2x2_s4".to_string()), (1, 896, 4864, "x".to_string())]
        );
    }

    #[test]
    fn regression_is_a_relative_drop_over_3pct() {
        let c = |p: f64, b: f64| ShapeCheck { shape: (1, 1, 1), impl_: "x".into(), pct: Some(p), before: Some(("r".into(), b)) };
        assert!(!c(97.1, 100.0).regressed());
        assert!(c(96.9, 100.0).regressed());
        assert!(!ShapeCheck { shape: (1, 1, 1), impl_: "x".into(), pct: None, before: Some(("r".into(), 99.0)) }.regressed());
    }
}
