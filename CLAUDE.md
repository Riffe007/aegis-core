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

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # arch defaults to the local GPU ("native")
cmake --build build -j
ctest --test-dir build --output-on-failure        # correctness on odd shapes, every impl
./build/aegis_manifest > results/manifest-$(hostname).json
python3 tools/sweep.py                            # shape sweep -> results/*.jsonl -> perf DB
python3 tools/perfdb.py report                    # % of cuBLAS per shape
tools/sass_dump.sh                                # SASS + register usage per kernel
```

Always build natively on the device you are measuring. Do not cross-compile.

## Layout

```
include/aegis/      public headers (common.cuh, gemm.h)
src/gemm/           one .cu per implementation tier
bench/              bench_gemm.cu: timing, L2 flush, cuBLAS reference check, JSON output
tools/              aegis_manifest.cu, perfdb.py, sweep.py, sass_dump.sh
results/            JSONL runs, device manifests, perf DB (gitignored)
artifacts/sass/     SASS dumps per arch (gitignored)
```

## Rules (non-negotiable)

1. **No performance claim without a perf DB row.** Never write a speedup number
   in a README, commit message, or comment unless it came from `tools/sweep.py`
   output on real hardware. If you have not run it, say "unmeasured".
2. **Every kernel passes `--check` against cuBLAS before it is benchmarked.**
   Tolerance: max_abs_err / max|ref| <= 4e-3 (about 4 FP16 ulps). Odd shapes
   (non-multiples of the tile) must pass too; add them to `tests` in CMakeLists.
3. **The baseline is cuBLAS, not naive.** Report results as % of cuBLAS on the
   same shape and device.
4. **One impl per file, registered in one place.** Every launchable impl is an
   entry in `aegis::gemm::registry()` (`src/gemm/registry.cu`); `bench_gemm`
   dispatches through it and `bench_gemm --list` prints it. A templated tier
   declares a constexpr config table in `gemm.h` (one row per instantiation,
   impl name `<tier>_<params>`) and an `<tier>_entries()` function that the
   registry concatenates. Adding a kernel = new file in `src/gemm/`, its table
   or declarations in `gemm.h`, its entries in `registry.cu`, the names in
   `tools/sweep.py` `IMPLS` and `AEGIS_IMPLS` in `CMakeLists.txt`.
   Use `/new-kernel`.
5. **Look at the SASS before claiming an optimization worked.** If you add
   inline PTX, confirm with `tools/sass_dump.sh` that the expected instructions
   (HMMA, LDSM, LDGSTS) appear and that register count did not spill.
6. **Benchmark hygiene.** Pause background training jobs before a sweep. The
   harness flushes L2 between reps and reports median/p10/p90. Do not reduce
   reps to make a sweep faster without saying so.
7. **Do not change public signatures** in `gemm.h` without updating every impl
   and the bench dispatch in the same commit.

## Architecture notes you must respect

- GB10 (sm_121) is consumer-lineage Blackwell. It does **not** have the sm_100
  datacenter path (`tcgen05`, TMEM, UMMA, 2-CTA MMA). Tensor cores are driven
  with warp-level `mma.sync`, same family as Ampere. Verify any new instruction
  against the PTX ISA for sm_120/sm_121 before using it.
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
- [ ] Tier 4: tier 3 + `cp.async` multi-stage pipeline (STAGES template param)
- [ ] Autotuner: enumerate template instantiations, time per shape, write `winners`
- [ ] `aegis::matmul()` dispatch reading winners for the local arch
- [ ] Decode GEMV (M=1..16), then INT8 weight-only variant
- [ ] Fused RMSNorm and SwiGLU
- [ ] Roofline plots and README results table on both archs

## Out of scope for v1 (do not start these)

HIP/AMD, SystemVerilog, learned performance predictors, INT4/FP4, convolution,
flash attention, adaptive runtime. If a task seems to need one, stop and ask.
