//! Long-running GPU jobs go through `tools/gpu_job.sh` (CLAUDE.md Benchmark
//! policy): oom_score_adj 1000 so a runaway is the first OOM victim, plus a
//! MemoryMax host-memory cap (AEGIS_MEM_MAX, default 24G; inherited from the
//! environment). GB10 memory is unified, so GPU allocations come out of host
//! RAM; an unguarded job once drove the machine into a global OOM.
//!
//! And every GPU job holds the exclusive GPU lock (CLAUDE.md rule 15): an
//! flock on `results/gpu.lock` (AEGIS_GPU_LOCK overrides), the same lock
//! `tools/gpu_lock.sh` takes. aegis-perf takes it once per command
//! ([`lock`]) and passes `AEGIS_GPU_LOCK_HELD` to its children, so the
//! wrapper does not lock again.
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};

pub const WRAPPER: &str = "tools/gpu_job.sh";
pub const HELD_ENV: &str = "AEGIS_GPU_LOCK_HELD";

static HELD: AtomicBool = AtomicBool::new(false);

/// The GPU lock, held until dropped. `None` inside: an ancestor holds it.
pub struct GpuLock {
    _file: Option<File>,
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        if let Some(f) = &self._file {
            let _ = f.set_len(0);
        }
    }
}

pub fn lock_path() -> PathBuf {
    std::env::var_os("AEGIS_GPU_LOCK").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("results/gpu.lock"))
}

/// Take the GPU lock for this process (non-blocking). Inherited when an
/// ancestor (tools/gpu_lock.sh) already holds it.
pub fn lock(what: &str) -> Result<GpuLock> {
    if std::env::var_os(HELD_ENV).is_some() {
        HELD.store(true, Ordering::SeqCst);
        return Ok(GpuLock { _file: None });
    }
    let l = lock_at(&lock_path(), what)?;
    HELD.store(true, Ordering::SeqCst);
    Ok(l)
}

pub fn lock_at(path: &Path, what: &str) -> Result<GpuLock> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).read(true).open(path)
        .with_context(|| format!("open GPU lock {}", path.display()))?;
    match f.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            let mut holder = String::new();
            let _ = f.read_to_string(&mut holder);
            bail!("the GPU lock ({}) is held by: {}", path.display(), holder.trim());
        }
        Err(TryLockError::Error(e)) => return Err(e).context("flock GPU lock"),
    }
    f.set_len(0)?;
    writeln!(f, "{} pid {} ({}): {what}", chrono::Utc::now().to_rfc3339(), std::process::id(), crate::db::hostname())?;
    Ok(GpuLock { _file: Some(f) })
}

/// A Command running `exe` under the wrapper. Fails (never runs unguarded) if
/// the wrapper is missing, e.g. when not run from the repo root.
pub fn command(exe: &Path) -> Result<Command> {
    let wrapper = Path::new(WRAPPER);
    if !wrapper.exists() {
        bail!("{WRAPPER} not found: run aegis-perf from the repo root (GPU jobs never run unguarded)");
    }
    let mut c = Command::new(wrapper);
    c.arg(exe);
    if HELD.load(Ordering::SeqCst) {
        c.env(HELD_ENV, std::process::id().to_string());
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_holder_is_refused_until_release() {
        let path = std::env::temp_dir().join(format!("aegis-gpu-lock-test-{}", std::process::id()));
        let first = lock_at(&path, "first").unwrap();
        let err = lock_at(&path, "second").err().expect("lock is exclusive").to_string();
        assert!(err.contains("held by") && err.contains("first"), "{err}");
        drop(first);
        // Another test spawning a process can fork while the lock fd is open; the
        // child holds the flock until its exec closes the fd (O_CLOEXEC), a few
        // microseconds. Retry briefly instead of failing on that window.
        let mut again = lock_at(&path, "third");
        for _ in 0..100 {
            if again.is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            again = lock_at(&path, "third");
        }
        let again = again.expect("released on drop");
        drop(again);
        std::fs::remove_file(&path).ok();
    }
}
