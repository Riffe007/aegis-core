//! The production bars (CLAUDE.md "Production bar", 3. Performance) and the
//! number of record, as data, so every reader (reports, the winners header,
//! AEGIS Custos) applies one definition.
//!
//! **Number of record (2026-10-18).** Wherever a paired test exists, a kernel's
//! % of cublaslt_best is the paired one: 100 x the median of the paired ratio
//! cublaslt_best time / kernel time, with its 95% bootstrap CI. The unpaired
//! median (cublaslt_best median / kernel median, from two independent sets of
//! windows) is secondary and always labelled "unpaired". Ties, the dispatch
//! floor and the bars are decided on the paired CI.
//!
//! **Median vs CI.** The two disagree when the unpaired median % lies more than
//! [`MEDIAN_CI_DISAGREE_PTS`] percentage points outside the paired CI. A
//! disagreement is shown next to the number (report, header comment); the
//! paired CI still decides. Seen on 512x4864x896 (unpaired 82.2%, paired CI
//! 78.5-79.0%: the header's 82% read as above the 80% floor while dispatch
//! correctly used the CI) and 256^3 in an earlier tune (98.2% vs 89.9-90.2%).

/// Shapes with every dimension >= this are "large" (bar [`LARGE`]).
pub const LARGE_MIN_DIM: i64 = 1024;
/// Large shapes: >= 90% of cublaslt_best.
pub const LARGE: f64 = 0.90;
/// Decode (M <= params::DECODE_MAX_M): >= 100% of cublaslt_best, a tie or a win
/// (a tie: the paired CI contains 1.0).
pub const DECODE: f64 = 1.00;
/// No shape below this: aegis::matmul dispatches to cuBLASLt instead (tune::DISPATCH_FLOOR).
pub const FLOOR: f64 = crate::tune::DISPATCH_FLOOR;
/// Percentage points by which the unpaired median may lie outside the paired CI.
pub const MEDIAN_CI_DISAGREE_PTS: f64 = 2.0;

/// A kernel's standing against cublaslt_best: the paired test (ratio
/// cublaslt_best time / kernel time: > 1 means the kernel is faster).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedPct {
    pub median: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
}

impl PairedPct {
    /// The CI contains 1.0.
    pub fn tie(&self) -> bool {
        self.ci_lo <= 1.0 && self.ci_hi >= 1.0
    }
    /// CI entirely above 1.0.
    pub fn win(&self) -> bool {
        self.ci_lo > 1.0
    }
    /// "105.3% (105.2-105.7%)", or with " tie" when the CI contains 100%.
    pub fn text(&self) -> String {
        format!(
            "{:.1}% ({:.1}-{:.1}%{})",
            self.median * 100.0,
            self.ci_lo * 100.0,
            self.ci_hi * 100.0,
            if self.tie() { ", tie" } else { "" }
        )
    }
}

/// The unpaired median % (cublaslt_best median / kernel median x 100) lies more
/// than MEDIAN_CI_DISAGREE_PTS points outside the paired CI.
pub fn median_disagrees(unpaired_pct: f64, p: &PairedPct) -> bool {
    unpaired_pct < p.ci_lo * 100.0 - MEDIAN_CI_DISAGREE_PTS || unpaired_pct > p.ci_hi * 100.0 + MEDIAN_CI_DISAGREE_PTS
}

/// Which performance bar a shape is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bar {
    /// M <= params::DECODE_MAX_M: [`DECODE`] (a tie or a win).
    Decode,
    /// Every dimension >= [`LARGE_MIN_DIM`]: [`LARGE`].
    Large,
    /// Everything else: only the floor ([`FLOOR`]).
    Floor,
}

impl Bar {
    pub fn of(m: i64, n: i64, k: i64) -> Bar {
        if m as f64 <= crate::params::DECODE_MAX_M {
            Bar::Decode
        } else if m.min(n).min(k) >= LARGE_MIN_DIM {
            Bar::Large
        } else {
            Bar::Floor
        }
    }

    pub fn threshold(self) -> f64 {
        match self {
            Bar::Decode => DECODE,
            Bar::Large => LARGE,
            Bar::Floor => FLOOR,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Bar::Decode => "decode (>= 100% of cublaslt_best, a tie or a win)",
            Bar::Large => "large (>= 90% of cublaslt_best)",
            Bar::Floor => "floor (>= 80% of cublaslt_best, else dispatch to cuBLASLt)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The paired CI reaches the bar (its upper bound >= the threshold: not
    /// statistically below it; for decode, exactly "a tie or a win").
    Meets,
    /// The paired CI lies entirely below the bar.
    Below,
    /// No paired test: never judged on an unpaired median.
    NotMeasured,
}

/// A kernel against its shape's bar, on its paired CI (vs cublaslt_best).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarCheck {
    pub bar: Bar,
    pub verdict: Verdict,
    /// The CI lies entirely below the floor: aegis::matmul dispatches to cuBLASLt
    /// (tune::dispatch; the kernel still fails its bar).
    pub below_floor: bool,
}

/// Core's pass / fail for one shape, so every reader gives the same verdict.
pub fn check(m: i64, n: i64, k: i64, paired: Option<&PairedPct>) -> BarCheck {
    let bar = Bar::of(m, n, k);
    match paired {
        None => BarCheck { bar, verdict: Verdict::NotMeasured, below_floor: false },
        Some(p) => BarCheck {
            bar,
            verdict: if p.ci_hi >= bar.threshold() { Verdict::Meets } else { Verdict::Below },
            below_floor: p.ci_hi < FLOOR,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_vs_ci() {
        // 512x4864x896 (v0.4.0 fixture): unpaired 82.21%, paired CI [0.7851, 0.7903].
        let p = PairedPct { median: 0.788, ci_lo: 0.7851, ci_hi: 0.7903 };
        assert!(median_disagrees(82.21, &p));
        assert!(!median_disagrees(80.9, &p), "1.9 points above the CI: within the rule");
        assert!(!p.tie() && !p.win());
        // 1x4864x896 (Phase C step 4 paired-ab): a tie.
        let t = PairedPct { median: 1.002, ci_lo: 0.977, ci_hi: 1.012 };
        assert!(t.tie());
        assert_eq!(t.text(), "100.2% (97.7-101.2%, tie)");
    }

    /// Cases from the LOG's tunes (in-process paired CIs) against their bars.
    #[test]
    fn bars_on_the_paired_ci() {
        let p = |lo: f64, hi: f64| PairedPct { median: (lo + hi) / 2.0, ci_lo: lo, ci_hi: hi };
        // Decode: a tie meets the bar, a CI below 1.0 does not.
        assert_eq!(check(1, 4864, 896, Some(&p(0.9987, 1.0012))).verdict, Verdict::Meets);
        assert_eq!(check(16, 896, 4864, Some(&p(0.983, 0.994))).verdict, Verdict::Below);
        // Large: 2048^3 at 85.7-86.2% is below 90%; 1024^3 at 98.2-98.7% meets it.
        assert_eq!(check(2048, 2048, 2048, Some(&p(0.8566, 0.8621))), BarCheck { bar: Bar::Large, verdict: Verdict::Below, below_floor: false });
        assert_eq!(check(1024, 1024, 1024, Some(&p(0.9819, 0.9869))).verdict, Verdict::Meets);
        // 4096^3: below its bar and below the floor (dispatched to cuBLASLt).
        assert!(check(4096, 4096, 4096, Some(&p(0.7337, 0.7428))).below_floor);
        // 512x4864x896: only the floor applies, and it is below it.
        let c = check(512, 4864, 896, Some(&p(0.7851, 0.7903)));
        assert_eq!((c.bar, c.verdict, c.below_floor), (Bar::Floor, Verdict::Below, true));
        assert_eq!(check(512, 896, 896, None).verdict, Verdict::NotMeasured);
        assert_eq!(Bar::of(1024, 1024, 512), Bar::Floor);
    }
}
