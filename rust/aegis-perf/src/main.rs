//! aegis-perf: the AEGIS perf DB and tooling CLI. Run from the repo root.
//!
//!     aegis-perf init
//!     aegis-perf ingest RESULTS.jsonl --manifest M
//!     aegis-perf ingest-microbench MB.jsonl --manifest M
//!     aegis-perf winners [--arch 121]
//!     aegis-perf report [--arch 121] [--protocol 2] [--cache cold|warm] [--include-busy] [--baseline B]
//!     aegis-perf sweep [--impls tiled cublas] [--quick] [--allow-busy]
//!     aegis-perf tune [--shapes shapes/default.txt] [--seed S] [--allow-busy]
//!     aegis-perf emit-header [--arch 121]
//!     aegis-perf roofline --arch 121 [--out results/roofline-sm121.svg]
//!     aegis-perf emit-arch --arch 121 [--ptr-chase RUN[:LAYOUT]]...
//!     aegis-perf paired-ab --a OLD/build/bench_gemm --b build/bench_gemm --impl X --shape M N K
use std::path::PathBuf;

use aegis_perf::{ab, arch, db, roofline, sweep, tune};
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
        /// The idle guard's reading for this run (run_all.sh passes it).
        #[arg(long)]
        idle_util_pct: Option<f64>,
        #[arg(long)]
        idle_source: Option<String>,
        /// The guard found the GPU busy and the run went ahead anyway.
        #[arg(long)]
        allow_busy: bool,
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
        /// Timing protocol of the rows to show (1 = rows from before protocol v2).
        #[arg(long, default_value_t = 2)]
        protocol: i64,
        /// v2 cache mode: cold or warm.
        #[arg(long, default_value = "cold")]
        cache: String,
        /// Also show rows measured on a busy GPU (--allow-busy).
        #[arg(long)]
        include_busy: bool,
        /// Baseline for the % cells: cublas or cublaslt_best (default: cublaslt_best
        /// for protocol 2, cublas for protocol 1).
        #[arg(long)]
        baseline: Option<String>,
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
        /// Run even if the idle guard finds the GPU busy (> 2% utilization); rows are marked.
        #[arg(long)]
        allow_busy: bool,
        /// Also run the paired tie test of the fastest config against this impl
        /// (e.g. the previous winner), when it is not already the runner-up.
        #[arg(long)]
        pair_with: Option<String>,
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
    /// Same impl from two builds, alternating processes: paired ratio B/A with a bootstrap CI.
    PairedAb {
        /// bench_gemm of build A (e.g. the tree before a change).
        #[arg(long)]
        a: PathBuf,
        /// bench_gemm of build B.
        #[arg(long)]
        b: PathBuf,
        #[arg(long = "impl")]
        impl_: String,
        /// Impl name in build B if it differs from A's.
        #[arg(long)]
        impl_b: Option<String>,
        #[arg(long, num_args = 3, value_names = ["M", "N", "K"])]
        shape: Vec<i64>,
        #[arg(long, default_value_t = 30)]
        pairs: usize,
        /// Timed windows per process.
        #[arg(long, default_value_t = 10)]
        reps: u32,
        #[arg(long, default_value = "cold")]
        cache: String,
        /// Run even if the idle guard finds the GPU busy (> 2% utilization); the result is marked.
        #[arg(long)]
        allow_busy: bool,
    },
    /// Measured machine parameters (L2/DRAM latency, DRAM bandwidth, mma peak)
    /// -> include/aegis/arch/sm_<arch>.json, for the cost model.
    EmitArch {
        #[arg(long)]
        arch: i64,
        /// Pointer-chase runs to use, RUN or RUN:LAYOUT (cross_page / page_local_64k);
        /// repeatable. Default: every current (non-superseded) run.
        #[arg(long = "ptr-chase")]
        ptr_chase: Vec<String>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = "results")]
        results_dir: PathBuf,
    },
    /// SVG roofline from the measured roofs in include/aegis/arch/sm_<arch>.json and winners.
    Roofline {
        #[arg(long)]
        arch: i64,
        /// Default: results/roofline-sm<arch>.svg
        #[arg(long)]
        out: Option<PathBuf>,
        /// Measured parameters (default include/aegis/arch/sm_<arch>.json).
        #[arg(long)]
        arch_file: Option<PathBuf>,
    },
    /// Run every registered impl over the shape set, then ingest and recompute winners.
    Sweep {
        #[arg(long, num_args = 1..)]
        impls: Option<Vec<String>>,
        /// 3 shapes, for iterating.
        #[arg(long)]
        quick: bool,
        /// Run even if the idle guard finds the GPU busy (> 2% utilization); rows are marked.
        #[arg(long)]
        allow_busy: bool,
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
        Cmd::IngestMicrobench { results, manifest, run_id, idle_util_pct, idle_source, allow_busy } => {
            let idle = db::RunIdle { pct: idle_util_pct, source: idle_source, allow_busy };
            db::cmd_ingest_microbench(&cli.db, &results, manifest.as_deref(), run_id, &idle)
        }
        Cmd::Winners { arch } => db::cmd_winners(&cli.db, arch),
        Cmd::Report { arch, protocol, cache, include_busy, baseline } => {
            db::cmd_report(&cli.db, arch, &db::Selection { protocol, cache, include_busy }, baseline.as_deref())
        }
        Cmd::Tune { shapes, seed, prefix, allow_busy, pair_with, build_dir, results_dir } => tune::cmd_tune(&tune::TuneArgs {
            db: &cli.db,
            build: &build_dir,
            results: &results_dir,
            shapes: &shapes,
            seed,
            prefix,
            allow_busy,
            pair_with,
        }),
        Cmd::EmitHeader { arch, out, results_dir } => {
            tune::cmd_emit_header(&cli.db, arch, &results_dir, out.as_deref())
        }
        Cmd::PairedAb { a, b, impl_, impl_b, shape, pairs, reps, cache, allow_busy } => ab::cmd_paired_ab(&ab::AbArgs {
            a: &a,
            b: &b,
            impl_a: &impl_,
            impl_b: impl_b.as_deref().unwrap_or(&impl_),
            shape: (shape[0], shape[1], shape[2]),
            pairs,
            reps,
            cache: &cache,
            allow_busy,
        }),
        Cmd::EmitArch { arch, ptr_chase, out, results_dir } => {
            arch::cmd_emit_arch(&cli.db, arch, &ptr_chase, &results_dir, out.as_deref())
        }
        Cmd::Roofline { arch, out, arch_file } => {
            let out = out.unwrap_or_else(|| PathBuf::from(format!("results/roofline-sm{arch}.svg")));
            roofline::cmd_roofline(&cli.db, arch, &out, arch_file.as_deref())
        }
        Cmd::Sweep { impls, quick, allow_busy, build_dir, results_dir } => {
            sweep::cmd_sweep(&cli.db, &build_dir, &results_dir, impls, quick, allow_busy)
        }
    }
}
