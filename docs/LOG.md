# AEGIS Core log

Newest entry last. Every number here comes from the perf DB on the named
device (CLAUDE.md rule 1); run ids are given so rows can be found.

## 2026-10-13 — DGX Spark (GB10, sm_121) bring-up through language footprint

Device: `spark-5822`, NVIDIA GB10 (sm_121, 48 SMs), driver 580.142, CUDA 13.0.88,
CMake 3.28.3, Rust 1.91. Measured D2D bandwidth 214.7 GB/s (manifest; 79% of
the 273 GB/s nominal). Measured SM clock under load 2553 MHz.

### Commits

| Commit | What |
|---|---|
| `71cd348` | Bring-up: CMake built for sm_75 instead of native (arch default must precede `project()`); `aegis_manifest` missing include dir. ctest 8/8. |
| `b16dd90` | Tier 2 `regblock_*` (3 configs) + kernel registry, `bench_gemm --list`. |
| `6bcddbe` | Tier 3 `mma_*` (4 configs): inline PTX `ldmatrix` / `.trans` / `mma.sync m16n8k16`, padded smem; PTX ISA 9.4 citations; compiles for sm_87 and sm_121. |
| `3bd2264` | Tier 4 `mma_pipe_*` (12 configs, STAGES 2/3/4): `cp.async.cg` ring; LDGSTS present, 0 spills at all STAGES. |
| `5155c5c` | `tools/tune.py` autotuner, `aegis::matmul()` dispatch (exact shape → nearest M → cuBLAS), `include/aegis/winners/sm_121.h`. |
| `90a8bbc` | C ABI `aegis.h`, `libaegis_capi.{a,so}` (exports only `aegis_*`), C11 test. |
| `5509991` | Rust workspace: `aegis-sys` (bindgen + safe wrapper), `aegis-perf` (init/ingest/winners/report/sweep). Python perf tools deprecated. |
| `397ff59` | Microbench: 6 probe binaries with SASS-proven timed loops; perf DB `microbench` table. |
| `ea33bd0` | CLAUDE.md: language map, cargo commands, rule 8 (microbench not datasheets), rule 9 (Rust/Python parity). |
| `6146771` | Pre-commit hook (build + ctest), `.claude/settings.json` permissions. |

ctest: 108/108 (5 shapes x 21 impls, `matmul_dispatch`, `capi_static`, `capi_shared`).

### Current winners (sm_121, tune run `831e0287bf69`, % of cuBLAS on the same shape)

| Shape (MxNxK) | Winner | % cuBLAS | Runner-up gap |
|---|---|---|---|
| 256x256x256 | mma_pipe_64x128x32_2x4_s3 | 80.0 | 0.0% |
| 512x512x512 | mma_pipe_64x128x32_2x4_s4 | 87.0 | 2.1% |
| 1024x1024x1024 | mma_pipe_128x128x32_2x4_s2 | 78.6 | 2.5% |
| 2048x2048x2048 | mma_pipe_128x128x32_2x4_s2 | 74.0 | 13.5% |
| 4096x4096x4096 | mma_pipe_128x128x32_2x4_s2 | 55.1 | 9.8% |
| 1x896x896 | mma_pipe_64x64x32_2x2_s4 | 109.1 | 8.2% |
| 1x4864x896 | mma_64x64x32_2x2 | 82.9 | 2.4% |
| 1x896x4864 | mma_pipe_64x64x32_2x2_s4 | 77.7 | 4.5% |
| 16x896x896 | mma_pipe_64x64x32_2x2_s4 | 109.6 | 8.9% |
| 16x4864x896 | mma_64x64x32_2x2 | 102.4 | 1.2% |
| 16x896x4864 | mma_pipe_64x64x32_2x2_s4 | 90.2 | 4.3% |
| 512x896x896 | mma_pipe_128x64x32_2x2_s3 | 90.7 | 0.0% |
| 512x4864x896 | mma_64x128x32_2x4 | 73.5 | 0.4% |
| 512x896x4864 | mma_pipe_128x128x32_2x4_s4 | 85.8 | 0.4% |

Gaps at or below 2.5% are inside run-to-run noise: those winners are not decisive.

### Rust vs Python perf tools (acceptance for `5509991`)

- Same JSONL ingested by both: reports byte-identical (each tool on each DB);
  kernels, measurements, winners identical except random run ids.
- Live full sweeps, same build (307 rows each, 0 correctness failures):
  identical rows and columns; median Rust/Python ratio 1.000 (Rust slower on
  122 rows, faster on 135). 11 of 307 cells have |median diff| > max(p90-p10)
  of the two runs: 3 at 4096³ (device variance, below), the rest small shapes
  where the difference is almost exactly 2.0 or 4.0 µs (discrete timing
  states, not a tool effect; both tools launch the identical bench_gemm).

### Microbench (run `55945fd9cbef`, all 11 SASS proofs pass)

- mma.sync m16n8k16 f32.f16.f16.f32: latency 29.0 cycles; peak 125.5 TFLOPS
  (8 warps/SM, 4–8 chains). cuBLAS 4096³ (91.6 TFLOPS) is 73% of it.
- ldmatrix.x4: peak 128 B/clk/SM at 8 warps/SM.
- Global→smem copy: cp.async 191 GB/s vs ld+st 183 GB/s (DRAM); 2203 vs 2170 GB/s (L2).
- Pointer chase: L1 plateau 14.5 ns up to 64 KiB; DRAM ~370–440 ns.
- The SASS proof caught ptxas hoisting all 8 `ld.shared` out of the
  bank-conflict loop; fixed with `ld.volatile.shared`.

### Open questions

1. **4096³ timing instability.** cuBLAS medians of 1496, 1973–2131 and
   2837 µs across runs; p90/median up to 1.3. Cause unmeasured (power,
   thermal or contention). The 4096³ winner rests on one measurement window.
2. **Pointer chase is not monotonic.** 8 MiB (133 ns) < 4 MiB (172 ns); jump
   at 16 MiB though L2 is 24 MiB; 128 MiB (373 ns) < 64 MiB (435 ns). One pass,
   no repeats. Not usable by a cost model yet.
3. **Bank-conflict baseline is 2 cycles/warp load, not 1.** 2-way stride
   looks free; higher ways read ~1.15x predicted. Loop likely issue- or
   latency-bound (hypothesis).
4. **ldmatrix drops 128 → 99 B/clk/SM from 8 to 16 warps/SM.** Unexplained.
5. Small-shape timings show discrete 2 µs steps between runs.
6. C ABI: kernel launchers still abort on a launch failure.
7. No git remote is configured, so nothing has been pushed.

### Next: Jetson bring-up

Build natively on the Orin Nano (sm_87), ctest, `aegis_manifest`, tune, emit
`include/aegis/winners/sm_87.h`, run `microbench/run_all.sh`, compare winners
with sm_121.

## 2026-10-13 (later) — zero Python, in-process tuner

Commits: `6146771` pre-commit hook + permissions, `98f4f26` first LOG entry,
`9aac036` harness extraction, `7365a7b` aegis_tune, `00445da` aegis-perf
tune/emit-header, `bb09d12` roofline, `4ea8a2a` Python deleted.

- **Harness**: the timing protocol lives in `src/bench/harness.{h,cu}`.
  bench_gemm output before vs after on 6 cases plus `--list`: byte-identical
  except median/p10/p90/tflops/gbps.
- **Tuning wall time, full shape set (14 shapes x 19 configs + cuBLAS, 280
  timed rows)**: `tools/tune.py` (one bench_gemm process per row) **204 s**;
  `aegis-perf tune` **9.1 s** total, of which `aegis_tune` itself **8.7 s**.
- **emit-header**: Rust output matches the tune.py-generated `sm_121.h` byte
  for byte from line 2 on (line 1 names generator and time).
- **Fresh aegis_tune (run `dea3667f1ba1`, seed 20261013) vs logged tune.py
  (`831e0287bf69`)**: same winner on 11/14 shapes. 256³ and 512x896x896 were
  logged ties. 16x896x896 is the one strict miss: its logged 8.9% gap was
  22.512 vs 24.512 µs.
- **Finding: short timings are quantized to 2.048 µs on GB10.** Medians in
  the aegis_tune run sit on multiples: 10.240, 12.288, 14.336, 22.528,
  24.576, 26.624, 28.672, 30.720 µs. Every shape under ~50 µs (all decode
  shapes, 256³, 512³) is ranked by timer ticks, not kernel speed. Fixing it
  needs a protocol change (e.g. time N back-to-back launches per event pair);
  not done, since the protocol is fixed unless agreed.
- Roofline (`results/roofline-sm121.svg`): ridge 585 FLOP/B (214.7 GB/s,
  125.5 TFLOPS mma.sync, both measured). Winners at 1024³–4096³ reach
  30–44% of roof; cuBLAS 47–67%.
- Python files tracked: 0. The hook refuses any `*.py` (tested).

Open: timer quantization (above); 4096³ variance; the three microbench
artifacts; no git remote.

## 2026-10-13 (late) — error propagation, tie-break, microbench artifacts

Commits: `7cf2cd5` library errors propagate (C ABI 1.1), `b199fdf` tie-aware
winners, then this entry's commit (microbench v2 + superseded rows).

### Error propagation
Kernel host code and `aegis::matmul` return `cudaError_t`. The C ABI adds
`AEGIS_ERR_LAUNCH_FAILED` and `aegis_last_error_string()`. Forced failure
(M = 10^7, N = K = 896, tuned kernel, grid.y > 65535): status -6, detail
"launch mma_pipe_128x128x32_2x4_s4: invalid argument" (CUDA 13 reports grid
overflow as invalid argument). The process survives, the context stays usable,
and later GEMMs are correct.

### Tie-break (rule 11)
11 of 14 sm_121 shapes are tied (runner-up median inside the winner's
p10-p90), consistent with the 2.048 µs timer ticks and noisy p90 tails. An
"every config inside the band" variant was rejected: a p90 of 3.5x median at
512x4864x896 admitted configs 2x slower, and at 1024³ it picked one 14% slower.

### Microbench artifacts (run `fb77bc085d64`; all 15 SASS proofs pass; the 136 rows of `55945fd9cbef` marked superseded, none deleted)

**1. Pointer chase** (5 repeats per size, fresh permutation per repeat; ns median [p10-p90]):

| size | cross_page | page_local_64k |
|---|---|---|
| 4–64 KiB | 14.6 | 14.6 |
| 128 KiB | 145.9 [135–152] | 146.4 [140–148] |
| 512 KiB | 262.0 [233–273] | 269.5 [258–275] |
| 1 MiB | 360.9 [329–372] | 364.2 [349–393] |
| 2 MiB | 427.7 [420–445] | 432.5 [426–441] |
| 8 MiB | 426.7 [420–430] | 430.3 [428–436] |
| 16 MiB | 431.6 [426–433] | 433.9 [429–436] |
| 512 MiB | 434.8 [430–437] | 430.3 [427–433] |

- The v1 anomalies were single-pass artifacts: 8 MiB is no longer faster than
  4 MiB, there is no 16 MiB step, and 128 MiB is not faster than 64 MiB.
- The two layouts agree within noise at every size, so TLB reach does not show up to 512 MiB.
- **The 16 MiB step did not persist, so the conditional L2-partition test was
  not run.**
- **New open question:** no L2 plateau. Latency climbs from 146 ns
  (128 KiB) to ~430 ns at 2 MiB and stays flat to 512 MiB, although L2 is
  24 MiB. Not a clock-state effect: rerunning right after a heavy memory load
  gives the same curve. **Confounder:** gnome-remote-desktop holds the GPU at
  5–8% utilization (video clock 2099 MHz) while AEGIS is idle; it shares the
  memory system and L2. Next: rerun with remote desktop disconnected, then a
  per-SM pinned chase (near/far L2 test).

**2. Bank conflicts** (ILP 4/8 independent loads, empty-loop subtracted, LDS.128 added):
- Conflict-free 32-bit warp load: **1.28 cycles** (ILP 4, 8 warps/SM; v1 was
  2.0); 1.6–2.1 at ILP 8 or 32 warps.
- Refit, 32-bit: cycles = 0.03 + 1.11 x ways (ILP 8, 8 warps; r² 0.991);
  slope 1.09–1.12 across variants.
- LDS.128: conflict-free 3.4–4.3 cycles (4 phases); slope 4.5–4.6 cycles/way; r² ≥ 0.999.
- Unexplained: 2-way (stride 2) costs the same as conflict-free in every
  variant. From 4-way on, the curve is linear.
- Caveat: subtracting the whole empty loop over-subtracts a little (loop
  overhead overlaps the loads), hence negative intercepts in some fits.

**3. ldmatrix x4, 8 vs 16 warps/SM.** ncu is blocked:
`ERR_NVGPUCTRPERM` (`RmProfilingAdminOnly: 1`). Script ready:
`sudo microbench/profile_ldmatrix.sh` (bank conflicts, wavefronts, LSU
utilization, occupancy, stall reasons; CSV in `artifacts/microbench/`).
Without counters: 8 warps gives **exactly 128.0 B/clk/SM on every run**
(the ceiling); 16 warps gives 99.1, 128.0, 119.1, 128.0, 109.2 across 5 runs.
So the "drop" is intermittent variance above 8 warps, not a fixed
property. **Cause: unexplained** until the counters are collected.

## 2026-10-14 — timing protocol v2 (steps 1–6; re-tune pending)

Commits 9e4e414 .. f681eda. Protocol v2 is the default in `src/bench/harness`;
`--protocol 1` keeps the old loop for comparison.

**Timer finding (step 1).** The cudaEvent timer resolves **32 ns** on GB10
(smallest step between distinct deltas over 1000 event pairs around spin
kernels). The 2.048 µs steps logged earlier are **not** the timer: they appear
only when the start event is recorded right behind the L2-flush memset, as v1
does every rep. Same spin kernels, 400 samples each: deltas on an exact
2048 ns multiple — memset then event: 158/400; no memset: 0/400; memset +
stream sync: 11/400. The spec's "smallest nonzero delta" measures event
spacing instead (736 ns with nothing between the events, ~2.2 µs around a
kernel), so the harness records the step (`timer_res_ns`) and logs the spacing.
The earlier explanation of the 16x896x896 disagreement ("one 2.048 µs tick")
still stands as a property of v1 rows, with this cause.

**Steps 2–3.** N launches are captured in one CUDA Graph and N is calibrated
until a window is >= max(1 ms, 500 ticks) = 1 ms. Cold: K = max(2, ceil(3 x L2
/ footprint)) buffer sets rotated inside the window (16x896x896: K = 46;
4096³: K = 2), L2 flushed before each window, outside it. Warm: one set.
Development spot checks (not policy-compliant runs, not claims): cuBLAS
16x896x896 is 24.6 µs per launch under v1, 11.0 µs v2 cold, 3.9 µs v2 warm; v1
pays the flush, the memset-quantized start event and launch spacing on every
rep. At 4096³ cold and warm agree (footprint 96 MiB > L2).

**Step 5.** Idle guard (NVML, 2 s, 2% mean): idle GPU 0.0% passes; a
loop of GEMM processes reads 5.2% and is refused (exit 3); a sustained GEMM
reads 96%.

**Step 6.** Migration on a copy of the live DB: 723 rows become protocol 1;
`report --protocol 1` equals the pre-v2 report apart from its footer. Reports
and winners now read v2 cold idle-GPU rows only.

**Step 6b (added).** `cublaslt_best`: the fastest passing of up to 8
cuBLASLt heuristic algorithms, timed under the same protocol. At 16x896x896 the
heuristic's first pick is cuBLAS's own algorithm (rel_err 0 between them).

**Step 7 not run.** This session is the local X desktop, not SSH, and
gnome-remote-desktop holds a GPU context (Firefox is open too). The idle guard
passes (0.0%), but the Benchmark policy asks for SSH with remote desktop off,
and step 7 regenerates the committed winners header. A functional `aegis-perf
tune --shapes shapes/quick.txt` against a scratch DB ran end to end in 7.5 s
(aegis_tune wall); its numbers are not recorded.

## 2026-10-14 (evening) — step 7: sm_121 re-tuned under protocol v2

**Conditions.** Session in the local gnome-terminal (not SSH); remote desktop
stopped (`systemctl --user stop gnome-remote-desktop`); Xorg, gnome-shell and
Firefox still on the GPU (graphics only). `RmProfilingAdminOnly: 0`. The idle
guard refused one attempt (6.8% mean) and passed the next (1.1% mean, max 2%,
no `--allow-busy`). Repeated 2 s samples with the desktop up read 1.0–1.5% with
occasional bursts (one read 10% mean / 19% max), so a burst during the run is
possible and undetected.

**Run** `46372b6f92ba` (aegis_tune seed 3452099652): 14 shapes x 21 candidates
(19 configs + cuBLAS + cublaslt_best), 350 rows, 0 failed `--check`, aegis_tune
26.3 s wall. Timer 32 ns. Winners and `winners/sm_121.h` regenerated from v2
cold rows.

| Shape | Winner | % of cublaslt_best, cold | % of cublaslt_best, warm | Tied |
|---|---|---|---|---|
| 1x896x896 | `mma_pipe_64x64x32_2x2_s4` | 84.9 | 45.8 | no |
| 16x896x896 | `mma_pipe_64x64x32_2x2_s4` | 96.0 | 84.4 | tied |
| 1x4864x896 | `mma_pipe_64x64x32_2x2_s4` | 89.4 | 158.6 ¹ | no |
| 1x896x4864 | `mma_pipe_64x64x32_2x2_s4` | 76.7 | 79.9 ¹ | tied |
| 16x4864x896 | `mma_pipe_64x128x32_2x4_s4` | 87.9 | 63.2 | no |
| 16x896x4864 | `mma_pipe_64x64x32_2x2_s4` | 76.6 | 19.1 | no |
| 256x256x256 | `mma_pipe_64x64x32_2x2_s4` | 92.7 | 80.0 | no |
| 512x512x512 | `mma_pipe_64x64x32_2x2_s4` | 80.6 | 64.0 | no |
| 512x896x896 | `mma_pipe_64x128x32_2x4_s3` | 74.1 | 66.1 | tied |
| 512x4864x896 | `mma_pipe_128x128x32_2x4_s2` | 65.4 | 67.6 | no |
| 512x896x4864 | `mma_pipe_64x128x32_2x4_s3` | 64.0 | 50.5 | tied |
| 1024x1024x1024 | `mma_pipe_128x128x32_2x4_s2` | 77.6 | 66.8 | tied |
| 2048x2048x2048 | `mma_pipe_128x128x32_2x4_s2` | 68.1 | 71.4 | tied |
| 4096x4096x4096 | `mma_pipe_128x128x32_2x4_s2` | 59.0 | 55.9 | tied |

Warm % = cublaslt_best warm median / winner warm median. ¹ Baseline warm rows
are bimodal there: cuBLAS warm at 1x4864x896 has p10 8.76 µs vs median 19.70;
cublaslt_best warm at 1x896x4864 has p10 9.71 vs median 37.74. Do not use
those two warm cells. Also note: cublaslt_best's warm row re-times the
algorithm picked cold, and at 16x896x896 plain cuBLAS warm (3.99 µs) is 2x
faster than it (7.95 µs), so the warm column understates cuBLAS on small
shapes.

**Verdicts on earlier claims (all v1 rows, logged 2026-10-13):**
- **Decode shapes above 100% of cuBLAS — died.** v1 had 1x896x896 109.1%,
  16x896x896 109.6%, 16x4864x896 102.4% (tune.py table) and 100.0–104.7% in
  the v1 winners table (plus 512x896x896 100.1%). Under v2 cold, vs
  cublaslt_best: 84.9%, 96.0%, 87.9% (512x896x896: 74.1%). The closest call,
  16x4864x896, is 101.2% of plain cuBLAS but 87.9% of cublaslt_best: cuBLASLt
  finds a 37.73 µs algorithm where cublasGemmEx's default takes 43.46 µs. No
  shape beats cublaslt_best cold.
- **The 16x896x896 "miss" — died as a disagreement.** v2 picks
  `mma_pipe_64x64x32_2x2_s4`, the tune.py winner, at 10.91 µs. The v1
  aegis_tune pick, `mma_pipe_64x128x32_2x4_s4`, is 13.39 µs (23% slower) under
  v2. The v1 "8.9% gap = one tick" was quantization noise around a real gap.
  The shape is still marked tied, against `_s3` (12.73 µs, just inside the
  winner's p90 of 12.74).
- **11/14 tied — partly survived: 7/14 under v2.** All four decode-family
  shapes with N=896, K=896 and 4864 resolved except 1x896x4864 and
  16x896x896; the ties that remain are 512x896x{896,4864}, 1024³, 2048³, 4096³
  and those two, i.e. mostly large shapes where the winner's p90 tail is wide.
- **Overall level:** v1's 55–110% of cuBLAS becomes 59–96% of cublaslt_best.

### ldmatrix x4, 8 vs 16 warps/SM — counters (step 2)

`microbench/profile_ldmatrix.sh` (now 5 runs per point, `--clock-control
none`), `RmProfilingAdminOnly: 0`, remote desktop stopped. Counters per SM,
identical across the 5 runs at each point (ranges shown where they moved):

| Counter | 8 warps | 16 warps |
|---|---|---|
| shared-load bank conflicts | 0 | 0 |
| LSU shared-load wavefronts / LDSM.x4 | 4.0 | 4.0 |
| LSU writeback active, % of active cycles | 99.86–99.96 | 99.98 |
| LSU pipe inst issued, % of peak | 49.94–49.99 | 49.99 |
| implied B/clk/SM (wavefronts x 128 B / active cycles) | 127.9 | 127.97 |
| warps active, % of max (48) | 16.7 | 33.4 |
| stall mio_throttle per issue | 1.43–1.58 | 11.34 |
| stall short_scoreboard per issue | 5.35–5.38 | 4.40–4.42 |
| stall not_selected per issue | 0.08 | 0.66–0.68 |
| issue active, % | 21.1 | 21.1 |
| sm cycles elapsed, max − min across SMs | 15–40 | 12–45 |
| sm cycles elapsed − active (idle inside the kernel) | 9.0k–21.1k of 4.2M | 11.0k–22.9k of 8.4M; one run 212.2k |

The probe's own number (block 0's clock64 window), separately, 15 runs each
without ncu: 8 warps 128.0 every run; 16 warps 128.0 in 14 runs, 120.0 once.
Under ncu it printed 115.4 once at 16 warps, but ncu replays the kernel over
several passes and that line may come from a different pass than the counters.
With remote desktop running (previous entry) 16 warps read 99–128.

**Cause: unexplained.** No counter shows a throughput loss at 16 warps: no
bank conflicts, ideal wavefronts, the LSU writeback path busy 99.98% of active
cycles, active cycles proportional to the work in every profiled pass, all
SMs finishing within 45 cycles of each other. One 16-warp pass had every SM
idle for 212k cycles (2.5%) inside the kernel, against 9–23k in the other
nine: a gap where no SM was working, not a slower ldmatrix pipe. The
extra warps only queue: mio_throttle stalls go from ~1.5 to 11.3 per issue at
identical throughput, so 8 warps already saturate ldmatrix at 128 B/clk/SM.
The intermittent low readings are seen only by the probe's clock64 window,
and are rarer with remote desktop off (1/15) than on (3/5). Hypothesis, not
tested: another GPU context (the desktop) occasionally preempts or shares the
SM during block 0's window, which clock64 counts. A test would read
%globaltimer and per-SM active cycles alongside clock64 in the probe.
