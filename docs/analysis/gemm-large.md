# Large-GEMM analysis (sm_121, 2048³ and 4096³)

Consolidated from docs/LOG.md (2026-10-14 Nsight diagnosis; 2026-10-15
warp-tile experiment). Raw reports: `artifacts/ncu/*.ncu-rep` (gitignored),
made with `tools/ncu_gemm.sh` and tabulated with `tools/ncu_summary.sh`.
Method for every profile: Nsight Compute full section set, clocks locked to
base, caches flushed before each replay (cold); cublaslt_best pinned to the
tune's algorithm (`--lt-rank 1`, algo 67, `nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN`).
Every number below is from those reports or from perf DB runs named in place.

## Summary

| Question | Answer (evidence below) |
|---|---|
| What limits cublaslt_best? | The tensor pipe: 61–80% active, math_pipe_throttle the dominant stall. |
| What limited our 128x128 winner? | Too many instructions per MMA (5.5 vs 1.9) and 1.7–1.8x the L2 traffic; tensor pipe 42–50% active. |
| Did 64x64 warp tiles help? | At 4096³: +12.1% (paired), L2 sectors −24%, memory stalls gone; instructions per MMA unchanged. At 2048³: no gain (wave quantization, 2.67 waves at 1 block/SM). |
| Top limiter now | Instruction overhead around the HMMAs: a NOP after every HMMA (0.875/HMMA), ~2 per-thread integer address instructions per HMMA; `wait` the top stall. |
| What cublaslt_best does differently | Same `HMMA.16816.F32`, issued back to back without NOPs; address math on the uniform datapath; A/B tiles loaded with TMA (`UTMALDG`). |

## Part 1 — original diagnosis (2026-10-14)

### Profiles: winner 128x128x32_2x4_s2, best s3, best s4, cublaslt_best

`tools/ncu_gemm.sh` (full section set; ncu defaults: clocks locked to base,
caches flushed before each replay, so cold) on the winner
`mma_pipe_128x128x32_2x4_s2`, the best s3 and s4 configs from tune
`7ffc181804d7`, and cublaslt_best; `tools/ncu_summary.sh` tabulates
(`artifacts/ncu/`, gitignored). Gotcha found on the way: cublaslt_best picks
its algorithm by timing, and under ncu (serialized replays) it picked a
different, slower CUTLASS kernel (705 µs at 2048³). `bench_gemm --lt-rank R` /
`LT_RANK=R` now pins one heuristic rank; the tune's pick was rank 1
(algo 67, `nvjet_sm121_hsh_*`) at both shapes, and that is what is profiled.

| 2048³ | s2 (winner) | s3 128x128 | s4 128x128 | cublaslt_best |
|---|---|---|---|---|
| duration (µs, base clock) | 393 | 461 | 443 | **278** |
| grid / block / regs | 256 / 256 / 122 | 256 / 256 / 122 | 256 / 256 / 122 | 192 / 256 / 255 |
| dyn smem / blocks per SM (limit) | 37.9 KB / 2 (regs, smem) | 56.8 KB / 1 (smem) | 75.8 KB / 1 (smem) | 78.8 KB / 1 (regs, smem) |
| waves per SM | 2.67 | 5.33 | 5.33 | 4.00 |
| warps active, % of max | 30.5 | 17.9 | 17.0 | 16.5 |
| tensor pipe active, % elapsed | 42.5 | 35.6 | 37.2 | **61.2** |
| instructions executed | 23.4 M | 23.2 M | 22.8 M | **9.1 M** |
| issue active, % | 14.8 | 12.3 | 12.6 | 8.1 |
| L2 sectors / hit rate | 13.1 M / 95.3% | 13.1 M / 95.1% | 13.1 M / 95.1% | 7.9 M / 92.2% |
| smem ld bank conflicts | 6,574 | 0 | 0 | 0 |
| stalls per issue: long_scoreboard | **9.04** | 3.09 | 3.12 | 1.19 |
| barrier | 2.90 | 1.68 | 1.13 | 0.04 |
| wait | 3.49 | 3.51 | 3.51 | 0.78 |
| math_pipe_throttle | 3.45 | 2.37 | 2.37 | **11.42** |

| 4096³ | s2 (winner) | s3 128x64 | s4 128x128 | cublaslt_best |
|---|---|---|---|---|
| duration (ms, base clock) | 2.64 | 5.25 | 3.60 | **1.66** |
| grid / block / regs | 1024 / 256 / 122 | 2048 / 128 / 120 | 1024 / 256 / 122 | 640 / 256 / 255 |
| dyn smem / blocks per SM | 37.9 KB / 2 | 44.5 KB / 2 | 75.8 KB / 1 | 87.0 KB / 1 |
| waves per SM | 10.67 | 21.33 | 21.33 | 13.33 |
| warps active, % | 31.8 | 17.0 | 16.5 | 13.6 |
| tensor pipe active, % | 49.6 | 29.3 | 36.2 | **80.3** |
| instructions executed | 184 M | 245 M | 178 M | **64.5 M** |
| L2 sectors / hit rate | 102.8 M / 87.5% | 153.0 M / 84.2% | 102.7 M / 78.3% | 56.7 M / 89.0% |
| stalls: long_scoreboard / barrier / wait / math_pipe_throttle | 9.15 / 2.68 / 3.51 / 3.42 | 4.61 / 1.91 / 3.37 / 0.39 | 3.89 / 2.54 / 3.53 / 2.42 | 0.69 / 0.02 / 0.60 / **13.06** |

What the counters say (no explanation beyond them):
- cublaslt_best is **tensor-pipe bound**: tensor pipe 61% (2048³) and 80%
  (4096³) active, the dominant stall is math_pipe_throttle (11–13 per issue),
  almost no barrier or memory stalls, at 1 block of 8 warps per SM and 255
  registers.
- Ours keeps the tensor pipe 36–50% active and executes **2.6x (2048³) to
  2.85x (4096³) more instructions** for the same FLOPs; its stalls are spread
  over long_scoreboard, wait, barrier and math throttle instead of
  concentrated on the tensor pipe.
- Ours moves **1.7–1.8x more L2 sectors** than cublaslt_best (13.1 M vs 7.9 M;
  102.8 M vs 56.7 M) for the same problem.
- s3/s4 at 128x128 lose the second block per SM to shared memory (1 block, ~17%
  warps active) and are slower than s2 although their long_scoreboard stalls
  drop from ~9 to ~3; the 128x64 s3 at 4096³ executes the most instructions
  (245 M) and moves the most L2 sectors (153 M).
- 2048³ s2 runs 2.67 waves (the last wave 2/3 full); cublaslt_best runs
  exactly 4.

**Hypothesis (not tested):** the winner is issue/latency-limited by too little
tensor work per instruction and per byte moved: its 64x32 warp tile does
16 HMMA per k16 step against 6 ldmatrix.x4 plus address and cp.async work,
where cublaslt_best's single 8-warp block with 255 registers implies a much
larger per-warp tile. **One concrete change to test:** a config with a larger
warp tile, e.g. BM=128, BN=256, 2x4 warps (64x64 per warp), accepting 1 block
per SM, and compare instructions executed, tensor-pipe active % and L2 sectors
against the table above.

## Part 2 — 64x64 warp-tile experiment (2026-10-15)

### 1–2. New configs, correctness, registers

Added `mma_pipe_128x256x32_2x4_s{2,3}` and `mma_pipe_256x128x32_4x2_s{2,3}`:
8 warps with a 64x64 warp tile (MI = 4 x NI = 8: 32 MMAs per k16 step against
4 + 4 ldmatrix.x4; the winner's 64x32 tile: 16 against 4 + 2). Padding kept
(pad8). **s4 does not fit on sm_121**: a pad8 stage is 27,136 B / 29,184 B, so
s4 needs 108,544 / 116,736 B > the 101,376 B per-block opt-in (without padding
a stage is 24,576 B and s4 would need 98,304 B: one argument for the swizzle
experiment).

| Config | regs sm_121 | regs sm_87 | dyn smem | spills / local | ctest (5 odd shapes) |
|---|---|---|---|---|---|
| 128x256x32_2x4_s2 | 166 | 211–214 | 54,272 B | 0 / 0 | pass |
| 128x256x32_2x4_s3 | 166 | 211–214 | 81,408 B | 0 / 0 | pass |
| 256x128x32_4x2_s2 | 166 | 211–214 | 58,368 B | 0 / 0 | pass |
| 256x128x32_4x2_s3 | 166 | 211–214 | 87,552 B | 0 / 0 | pass |

(sm_87 values from the gate: four kernels at 211, 212, 213, 214; the gate does
not map them to names.) 129/129 ctest, sm87_gate 49 kernels pass.

`__launch_bounds__`: not needed. 166 registers x 256 threads allows one block
per SM already (and shared memory allows one or two), and nothing spills.
Making the 1-block bound explicit (`__launch_bounds__(THREADS, 1)`) for the
tier was tried and reverted: it changed register allocation for **every**
existing config, i.e. it
would have silently changed kernels that have tune data. (On sm_121: winner
122 → 132; `64x64x32_2x2_s4` 64 → 100.)

### 3. Tune at 2048³ and 4096³ (run `2b10fd68f576`)

`aegis-perf tune --shapes shapes/large.txt --pair-with mma_pipe_128x128x32_2x4_s2`
(new `--pair-with`: an extra paired test of the fastest config against a named
impl, here the previous winner). Idle guard 1.8% mean (9% max), no
`--allow-busy`. Cold medians, µs (p10–p90):

| Config | 2048³ | 4096³ |
|---|---|---|
| cublaslt_best | 196.0 (193.9–251.8) | 1422.7 (1410.8–1820.9) |
| **128x256x32_2x4_s3** (new) | 284.2 (282.8–337.0) | **2236.7** (2207.6–2655.6) |
| 128x256x32_2x4_s2 (new) | 296.0 | 2373.8 |
| 256x128x32_4x2_s3 (new) | 285.7 | 2377.2 |
| 256x128x32_4x2_s2 (new) | 295.4 | 2663.4 |
| **128x128x32_2x4_s2** (old winner) | **275.8** (274.4–373.0) | 2487.5 (2433.6–3086.4) |

Paired tests (ratio = B / A per pair, 95% CI, 30 pairs):

| Shape | A (fastest) | B | Ratio [CI] |
|---|---|---|---|
| 2048³ | 128x128_s2 (old winner) | 128x256_s3 (best new) | 1.022 [1.018, 1.026] |
| 2048³ | 128x128_s2 | cublaslt_best | 0.689 [0.686, 0.691] |
| 4096³ | 128x256_s3 (new winner) | 128x256_s2 | 1.061 [1.055, 1.067] |
| 4096³ | 128x256_s3 | 128x128_s2 (old winner) | **1.121 [1.109, 1.129]** |
| 4096³ | 128x256_s3 | cublaslt_best | 0.662 [0.657, 0.665] |

**4096³: the 64x64-warp-tile config wins, 10.8% faster than the old winner**
(old/new 1.121 per paired windows); cublaslt_best/ours goes from 0.580–0.590
(previous runs) to 0.662: 66% of cublaslt_best by paired ratio (63.6% by
medians). **2048³: no gain; the old winner stays**, the best new config is
2.2% slower. `winners/sm_121.h` regenerated (4096³ → `mma_pipe_128x256x32_2x4_s3`).

### 4. Nsight: new winner vs old winner vs cublaslt_best

Same `tools/ncu_gemm.sh` sections (base clocks, cold), new column for
`mma_pipe_128x256x32_2x4_s3`; the same config at 2048³ is added because it
lost there. (Idle guard read 8.2% just before this profile; counts below are
deterministic, durations may carry it.)

| 4096³ | old winner 128x128_s2 | **new winner 128x256_s3** | cublaslt_best |
|---|---|---|---|
| duration (ms, base clock) | 2.643 | **2.468** (−6.6%) | 1.659 |
| grid / regs / dyn smem / blocks per SM | 1024 / 122 / 37.9 KB / 2 | 512 / 166 / 81.4 KB / 1 | 640 / 255 / 87.0 KB / 1 |
| warps active, % | 31.8 | 14.7 | 13.6 |
| **instructions executed** | 183.6 M | **184.8 M** | 64.5 M |
| instructions per HMMA | 5.47 | 5.51 | 1.89 |
| **tensor pipe active, %** | 49.6 | **53.0** | 80.3 |
| issue active, % | 17.0 | 18.3 | 9.6 |
| **L2 sectors** / hit rate | 102.8 M / 87.5% | **77.6 M** / 83.4% | 56.7 M / 89.0% |
| **stalls per issue**: wait | 3.51 | **3.85** | 0.60 |
| math_pipe_throttle | 3.42 | 2.53 | **13.06** |
| long_scoreboard | **9.15** | 0.95 | 0.69 |
| barrier | 2.68 | 0.68 | 0.02 |
| mio_throttle | 1.57 | 0.37 | 0.77 |

| 2048³ | old winner 128x128_s2 | 128x256_s3 | cublaslt_best |
|---|---|---|---|
| duration (µs) | 393 | 408 | 278 |
| grid / waves per SM | 256 / 2.67 (2 blocks/SM) | 128 / 2.67 (1 block/SM) | 192 / 4.00 |
| instructions | 23.4 M | 23.5 M | 9.1 M |
| tensor pipe active, % | 42.5 | 40.4 | 61.2 |
| L2 sectors | 13.1 M | 10.0 M | 7.9 M |
| stalls: wait / math throttle / long_scoreboard / barrier | 3.49 / 3.45 / 9.04 / 2.90 | 3.82 / 2.48 / 1.86 / 1.20 | 0.78 / 11.42 / 1.19 / 0.04 |

Executed instructions by opcode at 4096³ (ncu `sass__inst_executed_per_opcode`):

| Opcode | old 128x128_s2 | new 128x256_s3 | cublaslt_best |
|---|---|---|---|
| HMMA | 33.55 M | 33.55 M | 34.08 M |
| NOP | 29.36 M | 29.36 M | — (not in top 14) |
| LDSM | 12.58 M | 8.39 M | 11.14 M |
| IMAD / LEA / ISETP / IADD3 / MOV | 19.1 / 8.5 / 16.9 / 6.5 / 13.8 M | 22.6 / 10.1 / 9.7 / 8.5 / 9.7 M | UIMAD 3.65, UIADD3 2.98, UISETP 2.36, MOV 1.36, IMAD 0.72 M |
| LDS | 6.34 M | 9.44 M | — |
| global → smem | LDGSTS 4.19 M | (not in top 14) | UTMALDG 0.66 M (TMA) |

The cublaslt_best kernel is `nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN`
(algo 67): CTA tile 128x208x64, TMA loads of A and B. Its SASS issues the same
`HMMA.16816.F32` back to back with no NOPs between them; ours has a `NOP`
after every HMMA (e.g. `HMMA.16816.F32 R124, R148, R156, R124 ; NOP ;
HMMA.16816.F32 R120, R140, R156, R120 ; NOP ; ...`), 0.875 NOPs per HMMA
executed in both configs.

### 5. Verdict on the warp-tile hypothesis

The hypothesis: the winner is limited by too little tensor work per
instruction and per byte; a larger warp tile (64x64) would cut instructions,
raise tensor-pipe busy % and cut L2 traffic.

**Partly held, at 4096³ only.**
- **Speed:** 4096³ +12.1% throughput (old/new 1.121 [1.109, 1.129], paired);
  cublaslt_best/ours 0.58–0.59 → 0.662. **2048³: no gain** (the best 64x64
  config is 2.2% slower, 1.022 [1.018, 1.026]); with 1 block per SM it runs
  128 tiles on 48 SMs = 2.67 waves, and tensor-pipe busy drops 42.5% → 40.4%.
- **Per byte — held:** L2 sectors −24% at both sizes (102.8 M → 77.6 M;
  13.1 M → 10.0 M); long_scoreboard stalls 9.15 → 0.95, barrier 2.68 → 0.68.
- **Per instruction — did not hold:** LDSM per HMMA fell exactly as predicted
  (12.58 M → 8.39 M), but total instructions did not move (183.6 M → 184.8 M;
  5.5 per HMMA vs cublaslt_best's 1.9). The saved ldmatrix were replaced by
  more integer address work and shared loads (IMAD +3.5 M, LEA +1.6 M,
  IADD3 +2.0 M, LDS +3.1 M). Tensor-pipe busy rose only 49.6% → 53.0%
  (cublaslt_best 80%).

**Top limiter now: instruction overhead around the HMMAs, i.e. issue, not
memory.** Per HMMA the kernel executes 0.875 NOP and ~2 per-thread integer
instructions (IMAD, LEA, ISETP, IADD3, SHF, MOV); the dominant stall is `wait`
(3.85 per issue, fixed-latency dependency waits), then math_pipe_throttle
(2.53); memory stalls are now small. cublaslt_best runs the same HMMA.16816
back to back without NOPs, does its address arithmetic on the uniform
datapath (UIMAD/UIADD3/UISETP) and loads tiles with TMA (UTMALDG), so its
tensor pipe, not issue, is the limit (math_pipe_throttle 13 per issue).

Hypotheses for the next experiments (none tested):
1. The NOP after every HMMA is a scheduling artifact of the operand order in
   `compute_tile` (consecutive HMMAs share B and vary A in our SASS; nvjet's
   share A and vary B). Test: reorder the MI/NI loops and compare NOP count.
2. Per-thread address math: precompute smem/global offsets per stage so the
   loop body is mostly LDSM + HMMA. Test: instructions per HMMA.
3. TMA: nvjet uses `UTMALDG` on sm_121, so the tensor-copy path exists on this
   chip (CLAUDE.md only rules out tcgen05/TMEM/UMMA). Verify
   `cp.async.bulk.tensor` against the PTX ISA for sm_120/sm_121 before use.
4. 2048³: wave quantization with 1 block per SM (2.67 waves); a split-K or
   persistent schedule, or a 192-tile grid like cublaslt_best's, would test it.

## Open experiments (order agreed 2026-10-15)

1. Phase A correctness hardening, with the predication designed not to add
   instructions to the main loop (masks per CTA tile, unpredicated interior
   fast path, offsets precomputed before the loop).
2. Phase B a: remove the NOP after every HMMA (accumulator / operand order).
3. Phase B b: XOR swizzle instead of padding (lets s4 fit at 128x256).
4. Later: TMA loads (sm_121 only; sm_87 has no `cp.async.bulk.tensor`), and
   stream-K or persistent scheduling for the 2048³ wave quantization.
