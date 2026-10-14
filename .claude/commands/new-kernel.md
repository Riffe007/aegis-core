---
description: Add a new GEMM implementation tier, wired end to end
argument-hint: <impl-name> <one-line description>
---

Add a new GEMM implementation named `$1`: $2

Follow CLAUDE.md. Do all of these in one change:

1. Create `src/gemm/$1.cu`. Kernel lives in `aegis::gemm::(anonymous)`, named
   `gemm_$1_kernel`. Header comment says what this tier changes versus the
   previous one and what bottleneck it targets.
2. Declare `launch_$1`, `attrs_$1`, `CONFIG_$1` in `include/aegis/gemm.h`.
   If the kernel is templated, CONFIG must state every template parameter.
3. Add the file to `aegis_kernels` in `CMakeLists.txt` and add `$1` to the
   ctest `foreach(impl ...)` list.
4. Register it in `dispatch()` and `info()` in `bench/bench_gemm.cu`.
5. Add it to `IMPLS` in `tools/sweep.py`.
6. Build, then run `ctest --test-dir build --output-on-failure`. All shapes,
   including the odd ones, must pass. Fix before going further.
7. Run `tools/sass_dump.sh` and report: registers, local (spill) bytes, and the
   instruction-mix row for the new kernel. If you used inline PTX, confirm the
   intended instructions appear.
8. Run `python3 tools/sweep.py --impls $1 tiled cublas --quick` and show the
   `perfdb.py report` output. Do not summarize performance beyond what that
   report shows.

Stop and ask before changing the public signature of any existing impl.
