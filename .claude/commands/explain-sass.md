---
description: Compare the SASS of two kernels and explain the performance gap
argument-hint: <impl-a> <impl-b> [M N K]
---

Explain why `$1` and `$2` perform differently, grounded in their machine code.

1. Run `tools/sass_dump.sh`.
2. From `artifacts/sass/*/instruction-mix.txt` and `resource-usage.txt`, compare
   registers, spill bytes, shared memory, and counts of HMMA, LDSM, LDGSTS,
   LDG, STS for both kernels.
3. Open both `.sass` files and find each kernel's main K loop. Report its
   instruction count, the ratio of math instructions to memory instructions,
   and any barriers (BAR.SYNC) inside the loop.
4. If a shape was given (M=$3 N=$4 K=$5), pull both kernels' measurements for
   it from the perf DB and compute arithmetic intensity and % of the measured
   bandwidth roof from `results/manifest-*.json`.
5. Conclude with the single most likely limiter for the slower kernel and one
   concrete change to test. Label anything not supported by the data above as a
   hypothesis.
