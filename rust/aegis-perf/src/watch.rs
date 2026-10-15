//! The idle guard during a run (Benchmark policy): `bench_gemm --idle-watch
//! --root <this pid>` samples GPU use by processes outside this process's tree
//! every 2 s until its stdin closes. A window above the limit makes the run
//! disturbed: every row of the run is marked (`db::mark_disturbed`), and the
//! watcher logs each such window on stderr as it happens.
use std::io::Read as _;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result};
use serde_json::Value;

/// What the watch saw over the whole run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WatchReport {
    /// Some(true): other processes went above the limit at least once;
    /// Some(false): watched and clean; None: no per-process source.
    pub disturbed: Option<bool>,
    pub windows: i64,
    pub windows_over: i64,
    pub max_other_pct: f64,
    pub limit_pct: f64,
    /// The busiest other process in the busiest window.
    pub worst: String,
    pub error: String,
}

impl WatchReport {
    /// One line for runs.disturbed_detail and logs.
    pub fn detail(&self) -> String {
        match self.disturbed {
            None => format!("not watched: {}", self.error),
            Some(d) => format!(
                "{} of {} 2 s windows over {}%; busiest window {:.1}% from other processes{}",
                self.windows_over,
                self.windows,
                self.limit_pct,
                self.max_other_pct,
                if self.worst.is_empty() || !d { String::new() } else { format!(", top: {}", self.worst) },
            ),
        }
    }
}

/// Parse bench_gemm --idle-watch output.
pub fn parse(stdout: &str) -> Result<WatchReport> {
    let v: Value = serde_json::from_str(stdout.trim()).context("parse bench_gemm --idle-watch output")?;
    let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
    Ok(WatchReport {
        disturbed: if v["ok"].as_bool() == Some(true) { v["disturbed"].as_bool() } else { None },
        windows: v["windows"].as_i64().unwrap_or(0),
        windows_over: v["windows_over"].as_i64().unwrap_or(0),
        max_other_pct: v["max_other_pct"].as_f64().unwrap_or(0.0),
        limit_pct: v["limit_pct"].as_f64().unwrap_or(crate::sweep::IDLE_MAX_PCT),
        worst: s("worst"),
        error: s("error"),
    })
}

/// A running watch; [`Watch::finish`] stops it and returns the report.
pub struct Watch {
    child: Child,
}

impl Watch {
    /// Start watching (after the lock and the idle guard, before the first
    /// measurement). `bench` is any bench_gemm build.
    pub fn start(bench: &Path) -> Result<Watch> {
        let child = Command::new(bench)
            .args(["--idle-watch", "--root", &std::process::id().to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("start bench_gemm --idle-watch")?;
        Ok(Watch { child })
    }

    /// Close the watcher's stdin, collect its report.
    pub fn finish(mut self) -> Result<WatchReport> {
        drop(self.child.stdin.take());
        let mut out = String::new();
        self.child.stdout.take().context("watcher stdout")?.read_to_string(&mut out)?;
        let status = self.child.wait()?;
        if !status.success() {
            anyhow::bail!("bench_gemm --idle-watch failed ({status})");
        }
        let r = parse(&out)?;
        match r.disturbed {
            Some(true) => eprintln!("idle watch: run DISTURBED: {}", r.detail()),
            Some(false) => eprintln!("idle watch: clean: {}", r.detail()),
            None => eprintln!("idle watch: {}", r.detail()),
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_disturbed_clean_and_unwatched() {
        let d = parse(
            r#"{"ok":true,"source":"nvml-process","limit_pct":2.00,"windows":4,"windows_over":2,"max_other_pct":3.00,"disturbed":true,"worst":"firefox (pid 7) 2.0% at T","error":""}"#,
        )
        .unwrap();
        assert_eq!(d.disturbed, Some(true));
        assert!(d.detail().contains("2 of 4") && d.detail().contains("firefox"), "{}", d.detail());
        let c = parse(r#"{"ok":true,"limit_pct":2.00,"windows":3,"windows_over":0,"max_other_pct":0.50,"disturbed":false,"worst":"","error":""}"#).unwrap();
        assert_eq!(c.disturbed, Some(false));
        let n = parse(r#"{"ok":false,"windows":0,"disturbed":null,"error":"no NVML"}"#).unwrap();
        assert_eq!(n.disturbed, None);
        assert!(n.detail().contains("no NVML"));
    }
}
