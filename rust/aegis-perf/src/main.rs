//! aegis-perf: the AEGIS perf DB from Rust. Same commands, schema and output
//! as tools/perfdb.py and tools/sweep.py. Run from the repo root.
//!
//!     aegis-perf init
//!     aegis-perf ingest RESULTS.jsonl --manifest M
//!     aegis-perf ingest-microbench MB.jsonl --manifest M
//!     aegis-perf winners [--arch 121]
//!     aegis-perf report [--arch 121]
//!     aegis-perf sweep [--impls tiled cublas] [--quick]
//!     aegis-perf tune [--shapes shapes/default.txt] [--seed S]
//!     aegis-perf emit-header [--arch 121]
mod db;
mod pyjson;
mod sweep;
mod tune;

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
    /// Autotune: build/aegis_tune over a shapes file, ingest, recompute winners, print the table.
    Tune {
        #[arg(long, default_value = "shapes/default.txt")]
        shapes: PathBuf,
        /// Config-order shuffle seed (default: random, logged by aegis_tune).
        #[arg(long)]
        seed: Option<u64>,
        /// Comma-separated impl name prefixes (default: aegis_tune's).
        #[arg(long)]
        prefix: Option<String>,
        #[arg(long, default_value = "build")]
        build_dir: PathBuf,
        #[arg(long, default_value = "results")]
        results_dir: PathBuf,
    },
    /// Winners for an arch -> include/aegis/winners/sm_<arch>.h (commit it).
    EmitHeader {
        /// Default: sm_arch from results/manifest-<host>.json.
        #[arg(long)]
        arch: Option<i64>,
        /// Output path (default include/aegis/winners/sm_<arch>.h).
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = "results")]
        results_dir: PathBuf,
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
        Cmd::Tune { shapes, seed, prefix, build_dir, results_dir } => tune::cmd_tune(&tune::TuneArgs {
            db: &cli.db,
            build: &build_dir,
            results: &results_dir,
            shapes: &shapes,
            seed,
            prefix,
        }),
        Cmd::EmitHeader { arch, out, results_dir } => {
            tune::cmd_emit_header(&cli.db, arch, &results_dir, out.as_deref())
        }
        Cmd::Sweep { impls, quick, build_dir, results_dir } => {
            sweep::cmd_sweep(&cli.db, &build_dir, &results_dir, impls, quick)
        }
    }
}
