---
description: Add a new GEMM implementation tier, wired end to end
argument-hint: <impl-name> <one-line description>
---

Add a new GEMM implementation named `$1`: $2

Follow CLAUDE.md. Do all of these in one change:

1. Create `src/gemm/$1.cu`. Kernel lives in `aegis::gemm::(anonymous)`, named
   `gemm_$1_kernel`. Header comment says what this tier changes versus the
   previous one and what bottleneck it targets.
2. In `include/aegis/gemm.h`: for a fixed kernel declare `launch_$1`,
   `attrs_$1`, `CONFIG_$1`; for a templated one add a constexpr config table
   (one row per instantiation) and `$1_entries()`. Each entry's config JSON must
   state every template parameter.
3. Add the file to `aegis_kernels` in `CMakeLists.txt` and every impl name to
   `AEGIS_IMPLS` (the ctest list).
4. Add its entries to `registry()` in `src/gemm/registry.cu`. Check with
   `./build/bench_gemm --list` (sweep and tune read the registry from there).
5. Kernel host code returns `cudaError_t` and never exits the process.
6. Build, then run `ctest --test-dir build --output-on-failure`. All shapes,
   including the odd ones, must pass. Fix before going further.
7. Run `tools/sass_dump.sh` and report: registers, local (spill) bytes, and the
   instruction-mix row for the new kernel. If you used inline PTX, confirm the
   intended instructions appear.
8. Run `rust/target/release/aegis-perf sweep --impls <new names> tiled cublas --quick`
   and show the `aegis-perf report` output. Do not summarize performance
   beyond what that report shows.

Stop and ask before changing the public signature of any existing impl.
