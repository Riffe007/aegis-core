---
description: Full benchmark sweep on this device and a % of cuBLAS report
---

1. Check nothing else is using the GPU (`nvidia-smi` or `tegrastats` for a few
   seconds). If a training job is running, stop and tell me instead of sweeping.
2. Rebuild in Release and run `ctest --test-dir build --output-on-failure`.
3. Run `python3 tools/sweep.py`.
4. Run `python3 tools/perfdb.py report` and show it verbatim.
5. In at most five bullets, point out: the best impl per shape family
   (square / decode M<=16 / prefill M=512), any correctness failures, any
   kernel with nonzero local (spill) bytes, and the largest p90/median spread
   (noise warning if p90 > 1.15x median).
