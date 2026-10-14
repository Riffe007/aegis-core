# AEGIS Core

A hardware-aware GPU kernel library and autotuner for NVIDIA Ampere (Jetson
Orin Nano, sm_87) and Blackwell (DGX Spark GB10, sm_121).

Every kernel is checked against cuBLAS, timed with an L2 flush between reps,
recorded in a SQLite perf DB alongside its SASS, and reported as a percentage
of cuBLAS on the same device and shape.

## Status

| Tier | Impl | Technique | Status |
|---|---|---|---|
| 0 | `naive` | one thread per output, global memory only | implemented |
| 1 | `tiled` | 32x32 shared-memory tiles | implemented |
| 2 | `regblock_*` | register-blocked micro-tiles, 3 configs | implemented |
| 3 | `mma_*` | tensor cores via `mma.sync` m16n8k16 + `ldmatrix`, 4 configs | implemented |
| 4 | `mma_pipe_*` | tier 3 + multi-stage `cp.async` pipeline, STAGES 2/3/4, 12 configs | implemented |
| — | autotuner | `aegis_tune` + `aegis-perf tune` + `aegis::matmul()` dispatch from per-arch winners | implemented |

Results: none published yet. Numbers appear here only after a sweep
(`aegis-perf sweep`) has produced them on real hardware.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
cargo build --release --manifest-path rust/Cargo.toml
rust/target/release/aegis-perf sweep --quick
rust/target/release/aegis-perf report
tools/sass_dump.sh
```

Requirements: CUDA 12.x (Jetson, JetPack 6) or 13.x (DGX Spark), CMake 3.24+,
Rust 1.80+ (and libclang for bindgen). No Python. Build on the device you
measure.

Tuning: `rust/target/release/aegis-perf tune` (runs `build/aegis_tune` over
`shapes/default.txt`), then `aegis-perf emit-header` and commit
`include/aegis/winners/sm_<arch>.h`. `aegis-perf roofline --arch <N>` writes an
SVG roofline from measured roofs.

## Layout

- `include/aegis/`, `src/gemm/`: kernels, one file per tier
- `bench/bench_gemm.cu`: timing harness and cuBLAS correctness check
- `tools/aegis_manifest.cu`: device manifest with measured bandwidth
- `include/aegis/aegis.h`, `src/capi.cpp`: C ABI (`libaegis_capi.a` / `.so`)
- `rust/aegis-sys`: bindgen bindings + safe wrapper over the C ABI
- `rust/aegis-perf`: perf DB and tooling CLI (`init`, `ingest`, `ingest-microbench`, `winners`, `report`, `sweep`, `tune`, `emit-header`, `roofline`)
- `microbench/`: one binary per hardware probe (PTX), SASS-proven timed loops; `run_all.sh`
- `src/bench/harness.{h,cu}`: the timing protocol, shared by `bench_gemm` and `aegis_tune`
- `tools/aegis_tune.cu`: in-process autotuner over the kernel registry; `shapes/`: shape sets
- `tools/sass_dump.sh`: SASS, register usage, and instruction mix per kernel
