# AEGIS Core — instructions for Claude Code

AEGIS Core is a hardware-aware GPU kernel library and autotuner. It targets two
NVIDIA architectures:

| Device | GPU | Arch | Notes |
|---|---|---|---|
| DGX Spark | GB10 (Blackwell) | sm_121 | 128 GB unified LPDDR5x, CUDA 13 |
| Jetson Orin Nano (x3) | Ampere | sm_87 | 8 GB unified LPDDR5, JetPack CUDA 12.x |

The v1 goal is narrow: an FP16 GEMM ladder (naive → tiled → register-blocked →
tensor-core `mma.sync` with `cp.async` pipelining), a brute-force autotuner that
caches winners per (arch, shape bucket), and a roofline report on both archs.
Everything else is v2. See "Out of scope" below.

## Language map

| Language | Where | Role |
|---|---|---|
| CUDA C++ (C++20) | `src/gemm/`, `src/bench/`, `bench/`, `tools/aegis_tune.cu`, `src/matmul.cu` | kernels, timing harness, tuner, `aegis::matmul` dispatch |
| PTX (inline asm) | `src/gemm/mma_common.cuh`, `microbench/` | `ldmatrix`, `mma.sync`, `cp.async`; probe loops. Cite the PTX ISA target notes for every instruction, valid on both sm_87 and sm_121 |
| C (C99 ABI, C11 tests) | `include/aegis/aegis.h`, `src/capi.cpp`, `tests/capi_test.c` | the only stable interface for other languages: opaque handles, int status codes |
| Rust | `rust/aegis-sys`, `rust/aegis-perf` | bindings + safe wrapper over the C ABI; all tooling: perf DB, sweep, tune, emit-header, roofline |
| AArch64 asm / intrinsics | planned | CPU kernels on the Grace / Orin cores; not started |

Other languages bind to `aegis.h`, never to C++ symbols. The shared library
exports only `aegis_*`; changing a signature there means bumping
`AEGIS_ABI_VERSION_MAJOR`.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAEGIS_SM87_GATE=ON  # arch: local GPU ("native"); + sm_87 compile gate
cmake --build build -j                            # kernels, bench, C ABI (.a/.so), microbench
tools/gpu_lock.sh ctest --test-dir build --output-on-failure   # GPU lock (rule 15); every impl on odd shapes, matmul dispatch, C ABI, sm87_gate
./build/aegis_manifest > results/manifest-$(hostname).json

# Rust (needs the CMake build first: aegis-sys links build/libaegis_capi.so)
cargo build --release --manifest-path rust/Cargo.toml
tools/gpu_lock.sh cargo test --release --manifest-path rust/Cargo.toml   # bindings vs CPU reference (GPU), schema parity
cargo clippy --release --all-targets --manifest-path rust/Cargo.toml
# AEGIS_LIB_DIR=/path/to/build overrides where aegis-sys finds the library.

rust/target/release/aegis-perf sweep              # shape sweep -> results/*.jsonl -> perf DB
rust/target/release/aegis-perf report             # % of cuBLAS per shape
rust/target/release/aegis-perf tune               # aegis_tune: every config x shapes/default.txt -> winners
rust/target/release/aegis-perf emit-header        # winners -> include/aegis/winners/sm_<arch>.h (commit it)
rust/target/release/aegis-perf init               # create / migrate the perf DB (schema v5; backs up first: <db>.bak-v<old>-<date>)
rust/target/release/aegis-perf roofline --arch 121  # results/roofline-sm121.svg from measured roofs
rust/target/release/aegis-perf emit-arch --arch 121  # include/aegis/arch/sm_121.json: measured L2/DRAM latency, DRAM bw, mma peak
tools/sass_dump.sh                                # SASS + register usage per kernel
microbench/run_all.sh                             # hardware probes -> perf DB microbench table (takes the GPU lock)
./build/aegis_fuzz --cases 2000 --seed 1          # randomized correctness vs FP64 / cuBLASLt (ctest fuzz_2000)
./build/aegis_fuzz --decode --poison --impls gemv_,matmul  # M <= 16 (some 17..40), K up to 16384 (ctest fuzz_decode)
./build/aegis_fuzz --large-k --poison --impls matmul_lt     # K 14000..20000 vs FP64; matmul_lt / matmul_cublas force the fallback paths (ctest fuzz_large_k)
cmake --build build --target fuzz_nightly         # 50,000 fuzz cases
tools/sanitize.sh                                 # ctest suite under compute-sanitizer memcheck/racecheck/initcheck/synccheck
tools/ncu_gemm.sh IMPL M N K; tools/ncu_summary.sh artifacts/ncu/*.ncu-rep   # Nsight profile + key metrics
rust/target/release/aegis-perf paired-ab --a OLD/bench_gemm --b build/bench_gemm --impl X --shape M N K  # A/B across builds
tools/nightly.sh                                  # nightly (user timer 03:00; --install): 50k fuzz, regression sweep, DB backup -> results/nightly.log
                                                  # summary lines: status=OK / FAIL / SKIPPED reason="..."
tools/kernel_log_guard.sh --since TIME            # new Xid / NVRM lines? (profiler lines inside logged sessions ignored)
rust/target/release/aegis-perf regress --run R   # winners-header shapes: % of cublaslt_best in run R vs the newest earlier clean run;
                                                  # >3% drop = watch, regression if 2 consecutive nightlies or a paired-ab confirm
rust/target/release/aegis-perf hw                 # hardware discovery JSON (GPU, CPU clusters/ISA, memory, thermal, NVDEC/NVENC)
rust/target/release/aegis-perf calibrate [--dry-run] [--out-dir DIR]  # this device: microbench (clocks locked if sudo -n can), tune, then
                                                  # DIR/arch/sm_<arch>.json + DIR/winners/sm_<arch>.h (default include/aegis); step times -> results/calibrate-history.jsonl
tools/record_hw_fixture.sh tests/fixtures/hw/NAME # record a device's sysfs for the discovery tests
tools/screen_off.sh CMD                          # run CMD with the local display blanked (in-run idle watch; see Benchmark policy)
./build/bench_gemm --impl aegis_matmul ...       # time aegis::matmul itself (dispatch), e.g. paired-ab --impl cublaslt_best --impl-b aegis_matmul
```

Once per clone: `git config core.hooksPath .githooks` (pre-commit builds and runs
ctest; a failing test refuses the commit). Run every `aegis-perf` and `tools/` command from the repo root (paths such as
`results/` and `build/` are relative). Always build natively on the device you
are measuring. Do not cross-compile.

## Layout

```
include/aegis/      public headers: gemm.h (registry, config tables), matmul.h, aegis.h (C ABI),
                    manifest.h, winners/sm_<arch>.h (generated by aegis-perf emit-header, committed)
                    arch/sm_<arch>.json (measured machine parameters for the cost model;
                    generated by aegis-perf emit-arch, committed)
src/gemm/           one .cu per implementation tier, registry.cu, mma_common.cuh, mma_pipe_impl.cuh (tier 4, shared with tier 5's fallback)
                    splitk.cuh (deterministic split-K: partition, ordered-ticket reduction, epilogue),
                    workspace.{h,cu} (per-stream workspace cache: split-K counters/partials, cuBLASLt)
src/hw/             device.{h,cu}: per-device limits (SM count, smem, regs, L2), per-device once, simulated profiles;
                    discover.cpp: hardware discovery (include/aegis/hw.h; C ABI 1.3 aegis_hw_*)
tests/fixtures/hw/  sysfs/procfs trees for discovery tests: spark-gb10 (recorded, tools/record_hw_fixture.sh),
                    orin-nano-synthetic (hand-written L4T layout until a Nano is connected)
src/                matmul.cu, capi.cpp, manifest.cpp; src/bench/ harness (timing protocol)
shapes/             shape sets for tune and sweep
bench/              bench_gemm.cu: timing, L2 flush, cuBLAS reference check, JSON output
tests/              test_matmul.cu, capi_test.c (C11)
microbench/         one binary per hardware probe, run_all.sh
rust/               cargo workspace: aegis-sys, aegis-perf
tools/              aegis_manifest.cu, aegis_tune.cu, sass_dump.sh
results/            JSONL runs, device manifests, perf DB (gitignored)
artifacts/          SASS dumps per arch, microbench SASS proofs (gitignored)
docs/               LOG.md (dated run log), analysis/ (consolidated analyses, e.g. gemm-large.md)
```

## Rules (non-negotiable)

1. **No performance claim without a perf DB row.** Never write a speedup number
   in a README, commit message, or comment unless it came from a sweep
   (`aegis-perf sweep`, `aegis-perf tune`) on real hardware. If you have not run
   it, say "unmeasured".
2. **Every kernel passes `--check` against cuBLAS before it is benchmarked.**
   Tolerance: max_abs_err / max|ref| <= 4e-3 (about 4 FP16 ulps). Odd shapes
   (non-multiples of the tile) must pass too; add them to `tests` in CMakeLists.
   Correctness proper (layouts, ld, alpha/beta, batch, NaN/Inf) is the
   Production-bar bound c * sqrt(K) * eps_fp16 * max|A| * max|B| (* |alpha|, + eps *
   |beta| * max|C0|) with **c = 2** (`src/bench/reference.{h,cu}`, measured with
   `aegis_fuzz --calibrate`), checked by `aegis_fuzz` (ctest fuzz_2000; target
   fuzz_nightly: 50,000 cases).
3. **The baseline is cuBLAS, not naive.** Report results as % of
   `cublaslt_best` (protocol v2; % of plain `cublas` alongside) on the same
   shape, device and cache mode.
4. **One impl per file, registered in one place.** Every launchable impl is an
   entry in `aegis::gemm::registry()` (`src/gemm/registry.cu`); `bench_gemm`
   dispatches through it and `bench_gemm --list` prints it. A templated tier
   declares a constexpr config table in `gemm.h` (one row per instantiation,
   impl name `<tier>_<params>`) and an `<tier>_entries()` function that the
   registry concatenates. Adding a kernel = new file in `src/gemm/`, its table
   or declarations in `gemm.h`, its entries in `registry.cu`, the names in
   and `AEGIS_IMPLS` in `CMakeLists.txt` (sweep and tune read the registry).
   Use `/new-kernel`.
5. **Look at the SASS before claiming an optimization worked.** If you add
   inline PTX, confirm with `tools/sass_dump.sh` that the expected instructions
   (HMMA, LDSM, LDGSTS) appear and that register count did not spill.
6. **Benchmark hygiene.** Pause background training jobs before a sweep. The
   harness controls cache state and reports median/p10/p90 (see Benchmark
   policy). Do not reduce reps to make a sweep faster without saying so.
7. **Do not change public signatures** in `gemm.h` without updating every impl
   and the bench dispatch in the same commit. `aegis.h` is a versioned ABI:
   additions bump the minor version, any incompatible change the major.
8. **Microbench numbers, not datasheets, feed the cost model.** Any roofline,
   peak, latency or bandwidth used in analysis, the tuner, or a future cost
   model comes from the perf DB (`microbench` table or `aegis_manifest`), on
   the same device. Datasheet figures may appear only as labeled context. A
   probe's number counts only if its SASS proof passed (run_all.sh exits 0);
   if a probe changes, rerun it before reusing its numbers. Latencies are
   stored in cycles with the SM clock they were measured at and whether
   clocks were locked (latency probes request locked clocks,
   `microbench/clock_lock.cuh`); `arch/*.json` stores cycles and clocks, and
   aegis-perf derives ns at read time. Locked-clock pointer-chase run:
   `sudo nvidia-smi -lgc F,F; microbench/run_latency_locked.sh F; sudo nvidia-smi -rgc`.
9. **No Python in AEGIS Core.** Kernels and runtime: C++20/CUDA/PTX. ABI: C11.
   CPU kernels: AArch64 asm/intrinsics. Tooling: Rust. The pre-commit hook
   refuses any `*.py` file. Kernel ids keep their canonical config
   serialization (`rust/aegis-perf/src/pyjson.rs`) so existing perf DBs stay
   valid.
10. **The library never exits the process.** Kernel host code (`launch_*`,
   `attrs_*`) and `aegis::matmul` return `cudaError_t`; the C ABI returns
   `AEGIS_ERR_*` and sets `aegis_last_error_string()`. Only bench, test and
   tool binaries may abort (`AEGIS_CUDA_CHECK`).
11. **Winners are tie-aware.** A shape is *tied* when the paired tie test
   (aegis_tune: 30 alternating windows of the fastest config and the runner-up,
   95% bootstrap CI of the median ratio) has a CI containing 1.0; rows without
   a paired test (v1, sweep) fall back to "runner-up median inside the fastest
   config's p10-p90". Of the two, pick the one that wins
   or ties on the most neighboring shapes (same N and K, or every dimension
   within 2x), then fewer registers, then less smem. The `winners` table
   records `tied`, `tie_set`, `pick_reason`, `tie_method` and the CIs
   (`ratio_ci_lo/hi`, `lt_ratio_ci_lo/hi` vs cublaslt_best); `emit-header` writes
   them as comments. Never hand-edit a winner. Winners rank on tune runs only
   (runs with paired tie tests); sweep rows, the nightly's included, time a
   subset of configs and never decide a winner.
12. **Never pass `--num-cuda-barriers` to compute-sanitizer.** GB10 memory is
   unified: the sanitizer's tracking allocations come out of host RAM. On
   2026-10-14 synccheck runs with a raised barrier limit drove the machine into
   a global OOM and the kernel killed the interactive session (claude,
   gnome-terminal scope). The barrier-table overflow came from cuBLAS kernels
   (`cuda::barrier`), not ours, so synccheck never runs cuBLAS instead (see
   Benchmark policy).
13. **Every device allocation in bench, tool, test and microbench code goes
   through the in-process GPU budget** (`src/bench/gpu_budget.h`:
   `aegis::budget::gpu_malloc` / `gpu_free`), never raw `cudaMalloc`. It
   refuses, with `cudaErrorMemoryAllocation` and the reason on stderr, any
   allocation that would take the process past `AEGIS_GPU_BUDGET` (default
   min(16 GiB, total / 2)) or leave less than `AEGIS_GPU_MIN_FREE` (default
   min(16 GiB, total / 4); total = the device's `cudaMemGetInfo` total, so both
   are 16 GiB on the Spark and fit an 8 GB Orin Nano) free system-wide per `cudaMemGetInfo` (which on GB10 reports host RAM: 111.7 GiB
   free when `free` said 112). This, not the cgroup cap, is the real protection
   on unified memory: the driver does not charge device allocations to the
   job's cgroup. Tests: ctest `gpu_budget_*` (deliberate oversize requests).
14. **One session owns main at a time. Nobody force-pushes main.** Every
   other session works on its own branch (in a separate `git worktree add`
   when the main working tree is in use) and stops for review; the owner of
   main merges. No session pushes or rewrites another's branch.
15. **All GPU work holds the exclusive GPU lock**: an flock on
   `results/gpu.lock` (`tools/gpu_lock.sh CMD`, or in-process in aegis-perf).
   Tunes, sweeps, paired-ab, microbench (`run_all.sh`), sanitizers, the
   nightly fuzz (`tools/gpu_job.sh` locks too) and GPU ctests. Every ctest
   but the compile-only `sm87_gate` is labelled `gpu` and fails its
   `gpu_lock_guard` fixture unless the run holds the lock. The holder is
   written into the lock file. A busy lock is refused (exit 75), never waited
   for silently; the pre-commit hook then runs `ctest -LE gpu` only and says
   that it skipped the GPU tests.

## Production bar (minimum for "done")

This bar is the minimum for a kernel to count as done. Peak-silicon tuning is stage 3
and starts only after all Core components meet it.

1. Correctness
   - Any M, N, K >= 1, including primes and odd K.
   - Row- and column-major A and B (NN, NT, TN, TT) and arbitrary leading dimensions.
   - Misaligned pointers or leading dimensions fall back to a narrower vector width or
     a scalar path; never fault.
   - alpha and beta: C = alpha*A*B + beta*C.
   - Batched and strided-batched GEMM.
   - M, N or K equal to 0 is a no-op that returns success.
   - NaN and Inf propagate; nothing is silently flushed.
   - Tolerance scales with K: c * sqrt(K) * eps_fp16 * max|A| * max|B|, with c
     documented.
   - Clean under compute-sanitizer memcheck, racecheck, initcheck and synccheck.
   - Randomized fuzzer against a reference: 2,000 cases in ctest, 50,000 nightly.

2. Robustness
   - Deterministic output on repeated runs by default (no atomics in split-K unless
     explicitly requested).
   - Stream-safe.
   - No allocation and no host synchronization on the call path.
   - Every failure returns an error code; the library never exits the process.

3. Performance (paired CI, cold cache, v2 protocol)
   - Shapes >= 1024 in every dimension: >= 90% of cublaslt_best.
   - Decode (M <= 16): >= 100% of cublaslt_best (a tie or a win).
   - No shape in the sweep below 80%; aegis::matmul dispatches to cuBLASLt instead of
     running a kernel that falls below 80%. Implemented as the dispatch floor: a winner whose
     paired CI vs cublaslt_best has its upper bound below 0.80 is emitted with dispatch
     `Cublaslt` (winners header column) and runs a cached cuBLASLt plan
     (`aegis/matmul.h`) with the algorithm cublaslt_best picked in the tune
     (perf DB `lt_algo`, header column; rebuilt by `src/lt_algo.h`). Reports show
     the AEGIS kernel % and the dispatch % (measured: `aegis_matmul`) as separate
     columns, in LOG.md too.

4. Maintainability
   - One templated kernel family with CUTLASS-style structure (threadblock / warp /
     MMA / epilogue); no copy-pasted variants.
   - Fused epilogues: bias, SiLU, GELU, residual add.
   - Documented dispatch heuristics for shapes not in the winners table.
   - A pre-commit regression gate that fails on a slowdown over 3% on the tracked
     shape set.

## Benchmark policy

The timing protocol lives in `src/bench/harness.{h,cu}` (one implementation,
shared by `bench_gemm` and `aegis_tune`). Protocol v2 is the default; v1 exists
only to compare with old rows (`--protocol 1`).

- **What v2 measures: steady-state throughput.** N back-to-back launches are
  replayed from one CUDA Graph per timed window; N is calibrated so a window
  lasts >= max(1 ms, 500 timer ticks). Per-launch time = window / N;
  median/p10/p90 across >= 20 windows. Every row records `protocol_version`,
  `batch_n`, `timer_res_ns`, `cache_mode`, `buffer_sets`.
- **What v2 does not measure: single-launch latency.** Launch overhead, the
  CPU submission path, first-launch and cold-start costs belong to the
  launch-overhead probe (`microbench/launch_overhead.cu`), never to this harness.
  Do not compare a v2 per-launch time with an end-to-end latency.
- **Cache state is explicit.** `cold` (ranks configs, the default):
  launches rotate through K = max(2, ceil(3 x L2 / footprint)) buffer sets
  inside the window, L2 is flushed before each window and nothing but the
  launches runs inside it. `warm`: one buffer set, no flush. Winners are picked
  on cold; warm is recorded for them. Never mix cold and warm rows in one
  comparison.
- **Timer.** The event timer's resolution is measured at startup (32 ns on
  GB10). v1 rows are quantized to 2.048 us steps by the memset in front of each
  start event; do not rank shapes under ~50 us on v1 rows.
- **Latency probes run with clocks locked at 2418 MHz** (GB10's default
  applications clock; `nvidia-smi -lgc` works on GB10 as root):
  `sudo nvidia-smi -lgc 2418,2418; microbench/run_latency_locked.sh 2418;
  sudo nvidia-smi -rgc`. Keep the desktop idle for the whole run, not just
  the idle check: an app that starts on the GPU mid-run spreads the L2 plateau
  (LOG 2026-10-15, run `e3d94274b31d`).
- **Benchmark over SSH, remote desktop off.** A remote-desktop session
  (gnome-remote-desktop) encodes video on the GPU and shares L2 and DRAM with
  the benchmark; disconnect it before a sweep, tune or microbench run.
  At the local desktop, run measurements under `tools/screen_off.sh`: with
  the screen on, the compositor's per-process GPU samples rise with our
  bursty work and the in-run watch marks the run disturbed (LOG 2026-10-17).
- **Idle guard (blocking).** `aegis-perf sweep`, `tune` and `paired-ab` (and
  `aegis_tune`, `microbench/run_all.sh`) sample GPU utilization for 2 s first
  and abort above 2% mean (or when no source can tell) unless `--allow-busy`
  (`ALLOW_BUSY=1` for run_all.sh). `AEGIS_IDLE_MAX_PCT` can only lower the
  limit (tests use -1 to force the busy path). The reading is stored on every
  run row (`runs.idle_util_pct`, `idle_source`, `allow_busy`). Source: NVML
  (loaded with dlopen) where it exists; on Tegra (no NVML) the sysfs GPU load
  file (`.../gpu.0/load` or `.../17000000.gpu/load`, per mille), else
  `tegrastats` GR3D_FREQ. The source is logged.
  `--allow-busy` overrides and marks every row (`allow_busy`, `idle_util_pct`);
  marked rows are not evidence for a performance claim.
- **Idle guard during the run.** The same commands (and `run_latency_locked.sh`)
  keep watching while they run: `bench_gemm --idle-watch --root <pid>` ("ours"
  = that process tree) polls NVML every 250 ms. It first takes a 2 s baseline
  before the run's GPU work: device utilization D0 and the per-process sum of
  foreign processes F0, which give the scale k = D0 / F0 (at most 1; 0.25,
  GB10's measured idle-desktop ratio, when the baseline sees no foreign
  process). Then, per 2 s window, the foreign share is min(device - ours, k *
  foreign per-process), all window means. Polls from the one before a drop of
  our share by 10 points or more (a run process exited) to 2 s after it are
  left out of the means; a pid once seen in our tree stays ours. A window is over when that exceeds 2% or
  when any foreign process holds a compute context. Each such window is
  logged, and any one of them marks the run and every row of it
  `disturbed = 1` (`runs.disturbed`, `disturbed_detail`; also on
  `measurements`, `paired`, `microbench`, `paired_results`). 0 = watched and
  clean, NULL = not watched. Disturbed rows are excluded with busy ones
  (winners, reports, `emit-arch`) and are not evidence.
  - Why the scale: on GB10, per-process samples are on another scale than the
    device total (idle desktop: ~1% device, ~5% summed per process), and they
    lag (a steady full load reads 59 -> 82% per process over 3 s while the
    device reads 96%).
  - Why the settling: after our process exits, the device total still shows
    its work for about a second while ours reads 0, and foreign per-process
    samples (stale averages) jump just then (gnome-shell 1 -> 8% with the
    device at 1%; LOG 2026-10-17). Foreign compute contexts count in every
    poll, settling or not.
  - NVML ignores `lastSeenTimeStamp` on GB10, and its sample timestamps are
    not a clock, so none are used.
- **GPU memory.** First line of defence: the in-process GPU budget (rule 13),
  which refuses oversize device allocations before they happen. Second line:
  the guarded-job wrapper (next item).
- **Guarded GPU jobs.** Every long-running GPU job (sanitizers, `aegis-perf
  tune` / `sweep` / `paired-ab`, the nightly fuzz) runs through
  `tools/gpu_job.sh`: `oom_score_adj` 1000, so in any OOM the job dies first and
  its GPU memory is freed, plus a systemd `--user` scope with
  `MemoryMax=${AEGIS_MEM_MAX:-24G}` and no swap. The cap covers the job's host
  memory only: GB10's driver does not charge `cudaMalloc` to the job's cgroup
  (a 3 GiB cudaMalloc succeeds under MemoryMax=1G); the budget (rule 13) and
  the OOM priority are what stand between a runaway allocation and the session.
- **Before any run:** `nvidia-smi` answers, no orphaned `bench_gemm`,
  `aegis_fuzz`, `aegis_tune` or `compute-sanitizer` processes, and no new
  `Xid` / `NVRM` errors in `dmesg` / `journalctl -k`. If anything looks wrong,
  stop. Exception, and only this one: Nsight Compute makes the driver log
  `NVRM: refcntRequestReference_IMPL: Failed to enter state 1 (...0x00000056)`.
  `tools/kernel_log_guard.sh` (the nightly's check) ignores such a line only
  inside a profiler session window logged in `results/profiler_sessions.log`
  (`tools/profiler_session.sh`; gpu_lock.sh and the ncu wrappers use it). So run
  ncu through `tools/gpu_lock.sh` or a wrapper, never bare.
- **Sanitizer scope** (`tools/sanitize.sh`): one tool per invocation, never
  nested (the `memcheck_*` ctests are skipped by the script, as are `sm87_gate`
  and `gpu_budget_*`, which launch no kernels); JOBS 1 for
  racecheck, initcheck and synccheck, at most 2 for memcheck; fuzz tests at 100
  cases with dims <= 512. synccheck runs our kernels only and no cuBLAS /
  cuBLASLt kernel at all: `bench_gemm --check none`, `aegis_fuzz
  --cpu-ref-only`, and the cuBLAS-using tests (`capi_*`, `matmul_dispatch`)
  excluded. memcheck, racecheck and initcheck cover the whole suite.
- **Baselines.** `cublas` (cublasGemmEx, default algorithm) and
  `cublaslt_best`: up to 8 cuBLASLt heuristic algorithms, each checked against
  cuBLAS and timed under the same protocol; the fastest passing one is the row
  (which algorithm: the row's `lt` field). v2 reports and winners quote %
  of `cublaslt_best`; `pct_of_cublas` stays alongside.

## Decisions

- **No driver-API launcher (2026-10-14).** The launch-overhead probe
  (`microbench/launch_overhead.cu`, run `0de7ff68bb2f`) put runtime,
  driver (`cuLaunchKernel`), CUDA Graph replay and `aegis::matmul` within
  0.4 µs of each other at the median for the M = 1 winner (1x896x896:
  13.89–14.38 µs host-to-completion, ~4.4–4.8 µs of it fixed overhead). A
  driver-API launch path is rejected: it would save ~0.2 µs. Reducing the
  fixed cost has to remove launches, not make them cheaper: multi-kernel CUDA
  Graphs (a whole decode step per graph) or a persistent decode kernel. That
  is a separate experiment, not started.

- **Spark first (2026-10-14).** All Core work happens on the DGX Spark
  (sm_121) until v1 is complete; the Jetson (sm_87) port comes after. Until
  then sm_87 is a compile-only gate (no runs, no sm_87 numbers: CMake option
  `AEGIS_SM87_GATE`, ctest `sm87_gate`, on in the pre-commit hook), and the
  Jetson items in the milestone list wait.

- **Generated hardware-model files belong to Core (2026-10-15).** Hardware-model
  files (arch/*.json, winners headers) are generated only by Core sessions.
  Other projects may change the aegis-perf library API, via a separate git
  worktree, never the main working tree.

## Architecture notes you must respect

- GB10 (sm_121) is consumer-lineage Blackwell. It does **not** have the sm_100
  datacenter path (`tcgen05`, TMEM, UMMA, 2-CTA MMA). Tensor cores are driven
  with warp-level `mma.sync`, same family as Ampere. Verify any new instruction
  against the PTX ISA for sm_120/sm_121 before using it.
- **TMA is available on sm_121** (2026-10-15). cuBLASLt's sm_121 GEMM kernels
  (`nvjet_sm121_*_tmaAB_*`) issue `UTMALDG`. PTX ISA: `cp.async.bulk.tensor`
  (.tile, .mbarrier::complete_tx::bytes) is "PTX ISA 8.0, sm_90" (Data Movement
  instructions, cp.async.bulk.tensor; as encoded in CUDA 13.0's generated
  `cccl/cuda/__ptx/instructions/generated/cp_async_bulk_tensor.h`); only the
  .cta_group and gather4/scatter4 variants need sm_100a/sm_101a. Checked with
  ptxas: the 2d tile form assembles to `UTMALDG.2D` for sm_120 and sm_121, and is
  rejected for sm_87 ("requires .target sm_90 or higher"). So a TMA load path is
  sm_121-only; the shared kernel keeps `cp.async` for sm_87.
- **Tier 5, `mma_tma` (2026-10-17, `src/gemm/mma_tma.cu`)**: tier 3's warp
  tiles fed by TMA (`cp.async.bulk.tensor.3d.shared::cta`, PTX ISA 8.6; the
  `.shared::cluster` form compiles to a syscall fallback with a stack frame),
  an mbarrier ring, the 128B/64B hardware swizzle and OOB zero fill. Built only
  when every CMake arch is >= 90 (`AEGIS_TMA`; sm_87 never compiles it).
  Tensor maps are encoded on the host and cached per (pointer, shape, layout).
  Misaligned inputs (base not 16 B, ld or batch stride not a multiple of 8
  elements) run tier 4 on the same tile (`mma_tma_path()`, ctest `tma_paths`).
  Its dynamic smem is `__align__(1024)` with no slack: 1 KiB of slack cost
  128x128x32 s3 its second block per SM.
- **Decode GEMV family (2026-10-17, `src/gemm/gemv.cu`, `gemv_tc.cu`)**: M <= 16,
  B = weights, memory-bound on B; the roof is `read_bw_gbps` (arch json), the
  floor bytes(B) / read_bw. CUDA-core path (`gemv_*`) and tensor-core path
  (`gemv_tc_*`, M padded to 16), both with **deterministic split-K**
  (`splitk.cuh`): partials in FP32, the tile's last CTA (ordered ticket on a
  counter, never atomic accumulation) sums them in split order 0..S-1, so
  output is bitwise identical across runs (ctest `splitk_determinism`).
  Partials and counters come from the per-stream workspace cache
  (`workspace.h`): first call per stream outside capture, nothing allocates
  after (strict global-mode capture test). `KernelEntry::tune_max_m = 16`
  keeps aegis_tune from timing them on larger M; any M runs.
- **Launch sizing comes from the device, never from constants (aegis-hw step 1,
  2026-10-18, `src/hw/device.h`).** Limits are queried per device and cached
  (`hw::current()`: SM count, smem per SM / block / opt-in, regs, L2, memory).
  - Every tuned decode config is `_ksm`: split-K S is chosen at launch from the
    SM count (`splitk::sm_load_splits`, the S minimizing the busiest SM's K
    rows). Winners are per arch, but one arch spans SM counts (sm_87: Orin Nano,
    AGX Orin), so no table fixes S.
  - Smem attributes and TMA eligibility are set per device (`hw::PerDevice`),
    not once per process.
  - Tests size against simulated profiles with `hw::ScopedSimulation`
    (ctest `hw_sizing`: 24 / 48 / 96 SMs).
- **Capability-aware dispatch (aegis-hw step 4).**
  - Every registry entry declares a `KernelReq`: min arch, TMA, the most smem per
    block over all of its launches, and threads (constexpr in each family). Registers
    come from the binary.
  - `aegis::matmul` keeps only the winner rows whose kernel the device supports
    (`gemm::supported` against `gemm::current_caps()`), then takes the nearest M.
    Every row filtered out: cuBLASLt, with the row's tuned algorithm as the hint.
    No row: cuBLAS.
  - `detail::choose(table, M, N, K, caps)` is the testable core (ctest `hw_dispatch`).
- **Library fallbacks keep FP32 partials (2026-10-18).**
  - cuBLAS: `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`.
  - cuBLASLt: splitk 1 or `COMPUTE_TYPE` only, never INPLACE or OUTPUT_TYPE
    (`lt::reduction_ok`, heuristic mask; ctest `lt_reduction`, `fuzz_large_k`).
  - A stored cuBLASLt algorithm runs only on operands aligned to at least 16 B that meet its
    alignment caps (`cublasLtMatmulAlgoCheck` ignores pointers).
- **Device profiles (aegis-hw step 3, `src/hw/profile.h`).**
  - `include/aegis/arch/sm_<arch>.json` is compiled into the library (CMake) and used
    only when its sm_arch, sm_count, device name and l2_bytes all match the detected
    device. `$AEGIS_PROFILE_DIR/sm_<arch>.json` takes precedence;
    `AEGIS_PROFILE_EMBEDDED=0` turns the compiled-in files off.
  - No match: an unmeasured profile. Identity comes from the device query, and every
    measured roof is -1 (never a datasheet value). One stderr warning per process and
    device: "unmeasured device ...: run aegis-perf calibrate".
  - Every measurement row carries `"profile": "measured" | "unmeasured"` (perf DB v5:
    measurements, paired, microbench, paired_results).
  - Winners still dispatch on an unmeasured device of the same arch. S and grids follow
    its SM count, and step 4 filters kernels by capability.
- So one templated `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` kernel
  with `ldmatrix` and `cp.async` should run on both archs. The autotuner picks
  different tile configs per arch. That divergence is the project's story.
- Both devices have unified memory with very different bandwidth (Orin Nano
  ~68 GB/s nominal, GB10 ~273 GB/s nominal). Use the bandwidth that
  `aegis_manifest` *measures*, not the nominal number, for rooflines.
- Layout convention: row-major A[M,K], B[K,N], C[M,N]. FP16 inputs, FP32
  accumulate, FP16 output. cuBLAS is called with the transposed-operand trick
  (see `bench/bench_gemm.cu`).

## Current milestone (weekend 1–2)

- [x] Build system, harness, cuBLAS reference, perf DB, manifest, SASS dump
- [x] Tier 0: naive
- [x] Tier 1: shared-memory tiled (32x32)
- [x] Tier 2: register-blocked (each thread computes a TMxTN micro-tile), templated on BM/BN/BK/TM/TN
- [x] Tier 3: tensor-core `mma.sync` m16n8k16 + `ldmatrix`, templated on BM/BN/BK/warps
- [x] Tier 4: tier 3 + `cp.async` multi-stage pipeline (STAGES template param)
- [x] Tier 5 (sm_121 only): tier 3 + TMA loads, mbarrier ring, hardware swizzle (`mma_tma`)
- [x] Autotuner: enumerate template instantiations, time per shape, write `winners`
- [x] `aegis::matmul()` dispatch reading winners for the local arch
- [ ] Decode GEMV (M=1..16), then INT8 weight-only variant
- [ ] Fused RMSNorm and SwiGLU
- [ ] Roofline plots and README results table on both archs
- [x] C ABI, Rust bindings + perf CLI, microbench probes (Spark run in perf DB)
- [ ] Same on Jetson: build, ctest, `aegis-perf tune`, `microbench/run_all.sh`, commit `winners/sm_87.h`
- [x] Zero Python: `aegis_tune` + `aegis-perf tune/emit-header/roofline`

## Out of scope for v1 (do not start these)

HIP/AMD, SystemVerilog, learned performance predictors, INT4/FP4, convolution,
flash attention, adaptive runtime. If a task seems to need one, stop and ask.
