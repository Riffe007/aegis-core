//! aegis-perf: the AEGIS perf DB from Rust. Same commands, schema and output
//! as tools/perfdb.py and tools/sweep.py. Run from the repo root.
//!
//!     aegis-perf init
//!     aegis-perf ingest RESULTS.jsonl --manifest M
//!     aegis-perf ingest-microbench MB.jsonl --manifest M
//!     aegis-perf winners [--arch 121]
//!     aegis-perf report [--arch 121]
//!     aegis-perf sweep [--impls tiled cublas] [--quick]
mod db;
mod pyjson;
mod sweep;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "AEGIS perf DB: the shared spine every AEGIS project reads or writes")]
struct Cli {
    /// Perf DB path.
    #[arg(long, env = "AEGIS_PERFDB", default_value = "results/aegis_perf.db", global = true)]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create the schema.
    Init,
    /// Add one run's measurements.
    Ingest {
        results: PathBuf,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        temp_start: Option<f64>,
    },
    /// Add microbench probe results (microbench/run_all.sh output).
    IngestMicrobench {
        results: PathBuf,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        run_id: Option<String>,
    },
    /// Recompute the best kernel per (arch, shape).
    Winners {
        #[arg(long)]
        arch: Option<i64>,
    },
    /// % of cuBLAS per shape.
    Report {
        #[arg(long)]
        arch: Option<i64>,
    },
    /// Run every registered impl over the shape set, then ingest and recompute winners.
    Sweep {
        #[arg(long, num_args = 1..)]
        impls: Option<Vec<String>>,
        /// 3 shapes, for iterating.
        #[arg(long)]
        quick: bool,
        #[arg(long, default_value = "build")]
        build_dir: PathBuf,
        #[arg(long, default_value = "results")]
        results_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init => db::cmd_init(&cli.db),
        Cmd::Ingest { results, manifest, run_id, temp_start } => {
            db::cmd_ingest(&cli.db, &results, manifest.as_deref(), run_id, temp_start)
        }
        Cmd::IngestMicrobench { results, manifest, run_id } => {
            db::cmd_ingest_microbench(&cli.db, &results, manifest.as_deref(), run_id)
        }
        Cmd::Winners { arch } => db::cmd_winners(&cli.db, arch),
        Cmd::Report { arch } => db::cmd_report(&cli.db, arch),
        Cmd::Sweep { impls, quick, build_dir, results_dir } => {
            sweep::cmd_sweep(&cli.db, &build_dir, &results_dir, impls, quick)
        }
    }
}
