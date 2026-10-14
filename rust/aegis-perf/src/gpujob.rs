//! Long-running GPU jobs go through `tools/gpu_job.sh` (CLAUDE.md Benchmark
//! policy): oom_score_adj 1000 so a runaway is the first OOM victim, plus a
//! MemoryMax host-memory cap (AEGIS_MEM_MAX, default 24G; inherited from the
//! environment). GB10 memory is unified, so GPU allocations come out of host
//! RAM; an unguarded job once drove the machine into a global OOM.
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Result};

pub const WRAPPER: &str = "tools/gpu_job.sh";

/// A Command running `exe` under the wrapper. Fails (never runs unguarded) if
/// the wrapper is missing, e.g. when not run from the repo root.
pub fn command(exe: &Path) -> Result<Command> {
    let wrapper = Path::new(WRAPPER);
    if !wrapper.exists() {
        bail!("{WRAPPER} not found: run aegis-perf from the repo root (GPU jobs never run unguarded)");
    }
    let mut c = Command::new(wrapper);
    c.arg(exe);
    Ok(c)
}
