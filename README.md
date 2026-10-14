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
| 3 | `mma` | tensor cores via `mma.sync` m16n8k16 + `ldmatrix` | planned |
| 4 | `mma_pipe` | tier 3 + multi-stage `cp.async` pipeline | planned |
| — | autotuner | per-arch winner selection over template configs | planned |

Results: none published yet. Numbers appear here only after `tools/sweep.py`
has produced them on real hardware.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 tools/sweep.py --quick
python3 tools/perfdb.py report
tools/sass_dump.sh
```

Requirements: CUDA 12.x (Jetson, JetPack 6) or 13.x (DGX Spark), CMake 3.24+,
Python 3.9+. Build on the device you measure.

## Layout

- `include/aegis/`, `src/gemm/`: kernels, one file per tier
- `bench/bench_gemm.cu`: timing harness and cuBLAS correctness check
- `tools/aegis_manifest.cu`: device manifest with measured bandwidth
- `tools/perfdb.py`, `tools/sweep.py`: perf DB and sweep driver
- `tools/sass_dump.sh`: SASS, register usage, and instruction mix per kernel
