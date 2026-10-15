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
- **11/14 tied — partly survived: 7/14 under v2.** Four of the six decode
  shapes (M = 1, 16) now have a clear winner; 1x896x4864 and 16x896x896 stay
  tied, as do 512x896x896, 512x896x4864, 1024³, 2048³ and 4096³, mostly large
  shapes where the winner's per-window p90 tail is wide (see the 4096³ check).
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

### Pointer chase rerun (step 3): the L2 plateau is there

Run `6141123bbcc4` (same probe binary, SASS proof ok, 5 repeats per point;
145 rows ingested, the 145 rows of `fb77bc085d64` marked superseded). Remote
desktop stopped, after a reboot; desktop still up. Median ns [p10–p90], both
layouts agree within noise, cross_page shown:

| Working set | v2 (earlier, remote desktop on) | now |
|---|---|---|
| 4–64 KiB | — | 14 [14–14] |
| 128 KiB | 151.5 | 108 [106–115] |
| 256 KiB – 8 MiB | 262 → 434 | 132 [132–132 … 206] |
| 16 MiB | 439.6 | 138 [138–267] |
| 32 MiB | 438.5 | 373 [373–382] |
| 64–512 MiB | ~435 | 374–380 |

**Verdict: the missing plateau was not real.** L2 hits now take ~132 ns
(338 cycles at 2557 MHz) flat from 256 KiB to 16 MiB, and latency steps to
DRAM (~375 ns, 954 cycles) between 16 and 32 MiB, consistent with the 24 MiB
L2. In the earlier run the SM clock was the same (2532 MHz) but L2-sized sets
cost ~1100 cycles, the same as DRAM then, so L2 was effectively missing, and
DRAM was ~15% slower too. Two things changed between the runs: remote desktop
stopped and a reboot. Hypothesis, not tested: remote desktop's GPU context
(compute + graphics) time-slicing with the probe costs it its L2 contents.
Test: rerun with `systemctl --user start gnome-remote-desktop` and a client
connected. Until then, microbench and latency numbers taken with remote desktop
on are suspect (the ptr_chase rows of `fb77bc085d64` are superseded).

### 4096³ run-to-run variance under v2 (step 4)

Six back-to-back `bench_gemm --check` processes per impl, cold, 20 windows
each, idle guard 0.8%; rows ingested (`results/var4096-20261014-115315.jsonl`).

| Impl | Medians (µs) | Run-to-run spread (max−min)/mean | Within-run p90−p10, % of median |
|---|---|---|---|
| cublas | 1465.2–1476.1 | 0.7% | 12.6–28.1 |
| cublaslt_best | 1416.2–1431.0 | 1.0% | 10.1–31.8 |
| mma_pipe_128x128x32_2x4_s2 | 2470.1–2566.4 | 3.9% (one run 2566; the other five 2470–2485, 0.6%) | 17.1–22.0 |

**Run-to-run spread is under 5% for all three** (v1: cuBLAS medians from
1496 to 2837 µs across runs). The window-to-window spread is still wide: at
this size N = 1 (one 1.4–2.6 ms launch per window), so each window is a single
launch and its p10–p90 band is 10–32% of the median. That band is why 4096³
(and 2048³) stay "tied" in the winners table even though the medians repeat
within 1–4%. Not investigated tonight: whether the per-launch variation is
clock (power/thermal) or memory-system behaviour.

## 2026-10-14 (night) — steps 1–4 repeated with Firefox closed

**Conditions.** Local desktop session (gnome-terminal, not SSH); Xorg and
gnome-shell present on the GPU; remote desktop stopped. Firefox's window was
closed but its process (PID 8152, snap) was still running and still held a GPU
graphics context. `RmProfilingAdminOnly: 0`. Idle guard before the run: 0.35%
mean / 1% max; aegis_tune's own guard: 1.5% / 2%; no `--allow-busy`.

### Step 1 — re-tune, run `992182b466e4`

aegis_tune seed 2394767465, 350 rows, 0 failed `--check`, 26.0 s wall.
`winners/sm_121.h` regenerated.

| Shape | Winner | % of cublaslt_best, cold | % of cublaslt_best, warm | Tied |
|---|---|---|---|---|
| 1x896x896 | `mma_pipe_64x64x32_2x2_s4` | 81.6 | 39.2 | no |
| 16x896x896 | `mma_pipe_64x64x32_2x2_s4` | 94.9 | 83.7 | no |
| 1x4864x896 | `mma_pipe_64x128x32_2x4_s4` | 86.7 | 68.9 | tied |
| 1x896x4864 | `mma_pipe_64x64x32_2x2_s4` | 74.3 | 40.7 ¹ | tied |
| 16x4864x896 | `mma_pipe_64x128x32_2x4_s4` | 87.5 | 63.8 | tied |
| 16x896x4864 | `mma_pipe_64x64x32_2x2_s4` | 71.8 | 19.0 | no |
| 256x256x256 | `mma_pipe_64x64x32_2x2_s4` | 90.8 | 79.3 | no |
| 512x512x512 | `mma_pipe_64x64x32_2x2_s4` | 81.3 | 63.7 | tied |
| 512x896x896 | `mma_pipe_64x128x32_2x4_s3` | 74.0 | 64.8 | tied |
| 512x4864x896 | `mma_64x128x32_2x4` | 63.0 | 67.1 | tied |
| 512x896x4864 | `mma_pipe_64x128x32_2x4_s3` | 61.7 | 50.6 | tied |
| 1024x1024x1024 | `mma_pipe_128x128x32_2x4_s2` | 76.0 | 67.5 | tied |
| 2048x2048x2048 | `mma_pipe_128x128x32_2x4_s2` | 70.8 | 71.8 | tied |
| 4096x4096x4096 | `mma_pipe_128x128x32_2x4_s2` | 58.7 | 61.2 | tied |

¹ a baseline warm row is bimodal (median > 1.5 x p10); do not use the cell.
Warm % has the caveats of the previous entry.

Against the previous v2 run (`46372b6f92ba`, Firefox open): 12 of 14 winners
identical; 1x4864x896 and 512x4864x896 changed, both tied shapes. Cold % of
cublaslt_best (fastest config per shape) moved by at most 4.8 points
(16x896x4864: 76.6 → 71.8); 9 of 14 shapes moved by 2 points or less.

**Verdicts (unchanged by this rerun):**
- **Decode >100% of cuBLAS — died.** Best decode shape 16x896x896 at 94.9% of
  cublaslt_best; 16x4864x896 is 102.2% of plain cuBLAS but 87.5% of
  cublaslt_best. No shape reaches parity with cublaslt_best.
- **16x896x896 miss — died.** Winner `mma_pipe_64x64x32_2x2_s4` again (the
  tune.py winner), 10.96 µs, now untied: `_s3` 12.65 µs is outside its p90
  (12.11). The v1 aegis_tune pick (`mma_pipe_64x128x32_2x4_s4`) is 13.46 µs.
- **11/14 tied — not a stable number.** v2 gives 7/14 and 10/14 on two runs
  an hour apart. The large shapes (512x*, 1024³–4096³) are tied in both; the
  decode shapes flip. The tie rule (runner-up inside the winner's p10–p90)
  is sensitive to single slow windows in p90.

### Step 2 rerun — ldmatrix x4 counters, 8 vs 16 warps/SM

Same conditions; idle guard 0.5%. `profile_ldmatrix.sh 5` (`--clock-control
none`), 5 profiled runs per point:

| Counter (per SM unless noted) | 8 warps | 16 warps |
|---|---|---|
| shared-load bank conflicts | 0 | 0 |
| wavefronts per LDSM.x4 | 4.00 | 4.00 |
| LSU writeback active, % of active cycles | 99.94–99.96 | 99.98 |
| LSU pipe inst issued, % of peak | 49.97–49.99 | 49.99 |
| implied B/clk/SM (wavefronts x 128 B / active cycles) | 127.92–127.95 | 127.97 |
| elapsed − active cycles (idle inside the kernel) | 9.6k–20.2k | 11.2k–16.1k |
| elapsed cycles, max − min across SMs | 15–46 | 34–62 |
| warps active, % of 48 | 16.7 | 33.3 |
| stall mio_throttle / short_scoreboard / not_selected per issue | 1.54 / 5.38 / 0.08 | 11.34 / 4.42 / 0.68 |
| probe's own reading in the profiled run | 127.99 (5/5) | 128.00 (5/5) |

Without ncu, 15 runs each: 8 warps 128.0 in 15/15; 16 warps 128.0 in 13/15,
120.0 and 106.4 once each.

**Cause: unexplained.** Same picture as the first pass, now with no anomaly
inside any profiled run: the LSU path is saturated at 128 B/clk/SM at both
points, 16 warps only add mio_throttle queueing, and the idle-inside-kernel
gap seen once before (212k cycles) did not recur in 10 profiled runs. The low
readings appear only unprofiled and only at 16 warps so far (3 of 30 runs at
16 warps across both sessions, 0 of 30 at 8 warps; with this few events the
difference is not significant). The 16-warp kernel runs twice as long
(8.4M vs 4.2M cycles), so a fixed rate of outside interruptions would hit it
twice as often; that remains a hypothesis.

### Step 3 rerun — pointer chase

Three runs, same conditions:

| Median ns | 64 KiB | 128 KiB | 256 KiB–8 MiB | 16 MiB | 32 MiB | 512 MiB |
|---|---|---|---|---|---|---|
| `f1925d6a0fa4` cross_page (guard 1.0%) | 14 | 108 | 132–135 | 137 | 376 | 375 |
| `f1925d6a0fa4` page_local_64k, measured second | 14 | 117 | 132 → **177 at 4 MiB, 396 at 8 MiB** | 375 | 387 | 377 |
| repeat 1, `a825f4e557a3` (guard 0.0%), both layouts | 15 | 111 | 139 | 141–159 | 396–399 | 396–400 |
| repeat 2, not ingested (guard read 6.8% mean / 68% max just before) | — | — | 139 | 141–143 | 373–399 | — |

Repeat 1 ran at 2407 MHz against 2558 MHz for `f1925d6a0fa4`; in cycles the
plateau is the same (~335 cycles). `a825f4e557a3` is the current row set
(it superseded `f1925d6a0fa4`, which is kept).

**Verdict: the L2 plateau is real.** In five of six layout-runs since remote
desktop was stopped, L2-resident sets (256 KiB–16 MiB) cost 132–141 ns and
DRAM (>= 32 MiB) 373–400 ns, the step between 16 and 32 MiB matching the
24 MiB L2. **The earlier missing plateau is an intermittent effect, not a
property of the hardware:** in the second half of `f1925d6a0fa4` (the
page_local layout) L2-sized sets again cost DRAM latency from 8 MiB, with
remote desktop stopped, i.e. the same symptom as the first run. The desktop
still produces bursts (6.8% mean / 68% max over 2 s, minutes later). Cause not
established; the working hypothesis is still another GPU context evicting or
time-slicing L2, now Xorg/gnome-shell/Firefox rather than remote desktop.
Test: the same probe from a text console with `multi-user.target`.

### Step 4 rerun — 4096³ run-to-run variance

Same method (6 processes per impl, cold, 20 windows each), guard 0.0%, run
`cff022fb7ae3` (`results/var4096-20261014-120300.jsonl`):

| Impl | Medians (µs) | Run-to-run spread | Within-run p90−p10, % of median |
|---|---|---|---|
| cublas | 1486.9–1508.4 | 1.4% | 7.7–18.7 |
| cublaslt_best | 1432.3–1449.7 | 1.2% | 10.5–18.4 |
| mma_pipe_128x128x32_2x4_s2 | 2440.7–2541.7 | 4.0% | 17.7–21.9 |

**Under 5% again for all three** (previous session: 0.7 / 1.0 / 3.9%). The
winner's spread is now two clusters (2441–2445 and 2516–2542 µs) rather than
one outlier, so its 4% is real run-to-run variation, not one bad run. The
per-window band (single launch per window at this size) stays 8–22%, which
keeps 4096³ tied. Mean of the six medians vs the previous session: cublas
+1.8%, cublaslt_best +0.9%, winner +0.3%.

## 2026-10-14 (night, later) — paired tie test, arch file, launch overhead

Conditions as in the previous entry: local desktop session, Xorg, gnome-shell
and Firefox (PID 8152) on the GPU, remote desktop stopped; idle guard
0.7–1.0% mean before each run, no `--allow-busy`.

### 1. Paired tie test

`aegis_tune` now runs, per shape, 30 alternating cold windows (A, B, A, B…)
of the fastest config vs the runner-up and vs cublaslt_best; ratio = B / A per
pair; 10,000-resample percentile bootstrap of the median; tied = the 95% CI
contains 1.0 (`winners.tie_method = 'paired'`, `ratio_ci_lo/hi`,
`lt_ratio_ci_lo/hi`). Paired CIs are 0.5–2% wide where p10–p90 bands were
10–30%.

New case the rule had to cover: at 16x4864x896 in the full run the
runner-up/fastest CI was [0.986, 0.994], entirely below 1.0 — in paired
windows the config with the higher unpaired median was faster. Such a pair
is now resolved for the runner-up (`tie_method = 'paired_swap'`).

**Full re-tune `7ffc181804d7`** (350 rows, 28 paired tests): **0 of 14 shapes
tied** (v2 band rule: 7/14 and 10/14 on the two earlier runs).

**Decode shapes twice** (`8d7ae60c856b`, `6e1000a59892`), runner-up / winner CI:

| Shape | Run 1 | Run 2 | Stable? |
|---|---|---|---|
| 1x896x896 | s4 vs s3 [1.176, 1.188] no | s4 vs s3 [1.176, 1.185] no | yes |
| 1x896x4864 | s4 vs s3 [1.158, 1.166] no | s4 vs s3 [1.172, 1.185] no | yes |
| 16x896x896 | s4 vs s3 [1.158, 1.170] no | s4 vs s3 [1.146, 1.155] no | yes |
| 16x896x4864 | s4 vs s3 [1.156, 1.171] no | s4 vs s3 [1.153, 1.165] no | yes |
| 1x4864x896 | 64x64_s4 vs 64x128_s4 [0.995, 1.006] **tied** | 64x128_s4 vs 64x64_s4 [0.995, 1.002] **tied** | yes (tied both) |
| 16x4864x896 | 64x128_s4 vs 64x64_s4 [1.003, 1.012] no | 64x128_s4 vs 64x64_s4 [0.997, 1.006] **tied** | **no** |

(s4/s3 = `mma_pipe_64x64x32_2x2_s4/_s3`; 64x128_s4 = `mma_pipe_64x128x32_2x4_s4`.)

**Verdict: tie calls are stable on 5 of 6 decode shapes across the two
runs** (on the earlier band rule, decode tie calls flipped on 3 of 6 between
two runs). The one flip, 16x4864x896, is a real sub-1% difference sitting at
the CI's resolution: the same two configs are compared both times and their
ratio moves from 1.003–1.012 to 0.997–1.006. The full run also differed on
1x4864x896 because its unpaired runner-up was a different config (`_s3`,
[1.006, 1.016]); the paired test only compares the first two by unpaired
median. Final DB state (latest rows: decode run 2 for the decode shapes, full
run for the rest) regenerated `winners/sm_121.h`; 16x4864x896 and 1x4864x896
are tied and resolved by the tie-break (less smem → `mma_pipe_64x64x32_2x2_s4`).

### 2. ldmatrix x4, 8 vs 16 warps/SM — CLOSED

**8 warps saturate ldmatrix.x4 at 128 B/clk/SM** (two ncu sessions, 20
profiled runs: 0 bank conflicts, 4 wavefronts per LDSM.x4, LSU writeback busy
99.9–99.98% of active cycles at both 8 and 16 warps; 16 warps only add
mio_throttle queueing, ~1.5 → 11.3 stalls per issue). **The 16-warp dips
(99–128 B/clk with remote desktop on; 3 of 30 unprofiled runs since) remain
an untested interference hypothesis**: another GPU context interrupting the
probe's clock64 window, more often for the 16-warp kernel because it runs
twice as long. Not pursued further; the cost model uses 128 B/clk/SM with 8+
warps.

### 3. Measured parameters: `include/aegis/arch/sm_121.json`

Generated by the new `aegis-perf emit-arch --arch 121` from perf DB rows only
(rule 8); every value names its source runs.

| Parameter | Value | Source |
|---|---|---|
| L2 latency | **132.3 ns** (338 cycles); 30 points, 132.1–139.4 ns | ptr_chase 256 KiB–8 MiB, runs `6141123bbcc4` (both layouts), `f1925d6a0fa4` (cross_page only: its page_local half had the L2-loss episode), `a825f4e557a3` (both) |
| DRAM latency | **381.3 ns** (961 cycles); 25 points, 370–401 ns | same runs, working sets >= 32 MiB |
| DRAM bandwidth | **236.8 GB/s** (5 reps, 235.9–238.2) | new probe `mb_dram_bw` (the aegis_manifest D2D-copy method, repeated), run `d601cda25a28` |
| mma.sync peak | **125.6 TFLOPS** (0.25 m16n8k16 HMMA/clk/SM, 32 warps x 8 chains) | `mma_throughput`, re-run `eb3e197afc8e` (all SASS proofs pass; 125.55 in the remote-desktop-era run) |

Bandwidth moved: the bring-up manifest said 214.7 GB/s (remote desktop on);
three aegis_manifest runs tonight read 227.0, 233.9, 236.3 GB/s and the new
probe 235.9–238.2. Runs ingested before tonight carry 214.7 in
`runs.measured_bw_gbps`.

### 4. Launch-overhead probe (`microbench/launch_overhead.cu`)

Host-to-completion latency of one launch of the M = 1 winner
`mma_pipe_64x64x32_2x2_s4` (all three M = 1 shapes): steady_clock before the
call to after `cudaStreamSynchronize`, 2,000 samples after 200 warm-ups,
methods interleaved round-robin, operands warm in L2. Run `0de7ff68bb2f`,
idle guard 1.1%. Not CPU-pinned; CPU frequency not controlled.

| Shape | Method | Median µs | p99 µs |
|---|---|---|---|
| 1x896x896 | runtime (`<<<>>>` via registry) | 14.26 | 20.46 |
| | driver (`cuLaunchKernel`) | 14.03 | 20.02 |
| | CUDA Graph replay | **13.89** | **17.55** |
| | `aegis::matmul` end to end | 14.38 | 19.42 |
| 1x4864x896 | runtime | 17.07 | 25.07 |
| | driver | 16.80 | 26.14 |
| | graph | 16.67 | 23.33 |
| | matmul | 17.15 | 25.65 |
| 1x896x4864 | runtime | 51.52 | 54.40 |
| | driver | 51.31 | 55.09 |
| | graph | 51.17 | 54.40 |
| | matmul | 51.68 | 55.22 |
| any | empty kernel `<<<1,32>>>` (runtime) | 4.99–5.01 | 8.32–8.90 |
| any | `cudaStreamSynchronize` on an idle stream | 0.18 | 0.19–0.27 |

Against the winner's warm v2 per-launch time from decode run `6e1000a59892`
(9.47, 12.23, 47.15 µs), the single-launch cost is kernel + **~4.4–4.8 µs**
(runtime path), matching the 5.0 µs empty-kernel floor. Method differences are
small: graph replay is 0.3–0.4 µs faster than the runtime launch at the median
and has the lowest p99 on the two short shapes; the driver API saves ~0.2 µs;
`aegis::matmul`'s winner lookup adds ~0.1 µs over a direct registry launch.
At 1x896x896 the fixed cost is ~1/3 of the end-to-end time: larger than the
winner's cold kernel gap to cublaslt_best (10.76 vs 8.83 µs, 1.9 µs) and
smaller than its warm gap (9.47 vs 3.21 µs, 6.3 µs).

## 2026-10-14 (late night) — derived metrics, launch decision, Jetson

### 1. Roof metrics computed at report time

Nothing derived from a roof is stored. `aegis-perf report` now ends with a
roofline table per arch (AI, TFLOPS and % of mma peak, GB/s and % of DRAM
bandwidth, memory/compute bound, % of roof) computed when it runs from
`include/aegis/arch/sm_<arch>.json`; `aegis-perf roofline` takes its roofs from
the same file (it used the latest `runs.measured_bw_gbps`, still 214.7 GB/s on
most runs; that column is now only a record of what the manifest said).
Baseline points in the roofline are cublaslt_best (v2 cold), no longer cuBLAS.

Old percentages that moved (old: bring-up LOG, v1 rows, 214.7 GB/s and
125.5 TFLOPS roofs, ridge 585 FLOP/B; new: v2 cold rows, 236.8 GB/s and
125.6 TFLOPS, ridge 530 FLOP/B):

| Shape | Old winner % of roof | New winner % of roof | Old cuBLAS % of roof | New cublaslt_best (cublas) % of roof | Bound |
|---|---|---|---|---|---|
| 1024³ | 30–44 (range over 1024³–4096³) | 56.4 | 47–67 (range) | 75.0 (75.1) | memory (AI 341) |
| 2048³ | ″ | 49.2 | ″ | 71.4 (66.9) | compute (AI 683) |
| 4096³ | ″ | 44.5 | ″ | 76.8 (74.7) | compute (AI 1365) |

The bandwidth change (214.7 → 236.8 GB/s, +10%) moves only memory-bound
points; 2048³ and 4096³ sit above the ridge, so their movement is all from
the v1 → v2 measurements. Observation, not explained: at 1x4864x896 cuBLAS and
cublaslt_best reach 100.9% and 101.3% of the memcpy-measured bandwidth (and
16x4864x896 cublaslt_best 100.0%), so the D2D-copy figure is not an upper
bound for every access pattern.

## 2026-10-14 (late night, Spark-only) — read bandwidth, sm_87 gate, Nsight

Decision recorded in CLAUDE.md: all Core work stays on the Spark until v1 is
complete; the Jetson port comes after. No Jetson work was started.

### 1. Read-only bandwidth

`mb_dram_bw` gained a read-only stream: 4 independent `ld.global.v4.u32` per
thread per iteration over 512 MiB, folded into an XOR reduction, 2048
threads/SM. SASS proof: the timed loop holds exactly 4 `LDG.E.128` (ok). Run
`941ff9215564`, idle guard 1.5%:

| Stream | Median of 5 reps | Reps |
|---|---|---|
| copy (memcpy D2D, read + write bytes) | **236.7 GB/s** | 230.2–237.9 |
| read-only (bytes read) | **257.8 GB/s** | 256.1–258.8 |

`arch/sm_121.json` now has `copy_bw_gbps` and `read_bw_gbps` (the old
`dram_bandwidth_gbps` key is gone) and a `roof_rule`. Rooflines and the report
hold GEMV/decode shapes (M <= 16) to the read roof (ridge 487 FLOP/B) and every
other shape to the copy roof (ridge 531 FLOP/B); the report prints which on
each row. Effect: the decode points that exceeded the copy roof are now under
the read roof — 1x4864x896 cublaslt_best 101.3% → 93.1%, cublas 100.9% →
92.7%; 16x4864x896 cublaslt_best 100.0% → 91.8%. The M = 1/16 winner points
drop by ~8% of roof (e.g. 1x896x896 63.1% → 58.0%).

### 2. sm_87 compile gate

CMake option `AEGIS_SM87_GATE` compiles every source with kernels (the GEMM
tiers, registry, matmul, harness, all microbench probes, aegis_manifest; 17
objects) for `sm_87-real` as an object library that is never linked or run.
ctest `sm87_gate` (`tools/sm87_gate.sh`, cuobjdump -res-usage) fails on any
sm_87 kernel with STACK or LOCAL > 0 (spill / local memory), static shared
memory > 48 KB, or an object without sm_87 code; a source that does not
compile for sm_87 fails the build. The pre-commit hook enables the option.
Nothing runs on sm_87.

Result now: **45 sm_87 kernels, all pass**: no spills, no local memory, all
shared memory dynamic or ≤ 48 KB static (largest static: 32768 B), most
registers per thread REG:128. Negative checks:
a kernel built with `-maxrregcount=24` → `FAIL spill/local (STACK:9208)`; an
sm_121-only object → `FAIL no sm_87 code`; 52 KB of static smem → ptxas
compile error ("uses too much shared data"), i.e. a failed build. No CUDA
12.x issue to fix: CUDA 13.0's nvcc compiles everything for sm_87 as is (the
JetPack 12.x toolchain itself is untested until the port).

### 3. Nsight Compute diagnosis, 2048³ and 4096³

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

## 2026-10-15 — warp-tile hypothesis (64x64 warp tiles)

The hypothesis is in the previous entry (Nsight diagnosis, step 3); there is
no `docs/analysis/gemm-large.md` in the repo.

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

## 2026-10-15 — Phase A: correctness hardening

Commits: `2a60073` (API, ABI 1.2, predicated kernels), `8ac64d4` (reference,
tolerance c = 2), `3fa89b3` (fuzzer, BLAS alpha == 0), and the sanitizer step
below. Baseline for before/after: worktree `aegis-baseline-e8d7e6b` (the commit
before Phase A), built separately.

**Fuzzer results.** `fuzz_2000` (seed 1) 0 failures; nightly 50,000 cases
(seed 20261015) 0 failures (2,501 with NaN/Inf, 24,499 misaligned). Found
along the way: alpha == 0 with NaN/Inf in A or B. Our kernels computed
0 * NaN = NaN; cuBLASLt and cuBLAS are inconsistent there (some algorithms give
beta*C, others NaN). Semantics are now BLAS everywhere (A*B does not contribute
when alpha == 0); `aegis::matmul` handles alpha == 0 with a C = beta*C kernel.

**Boundary loads never read past the logical matrix.** The first version used
cp.async with a src-size operand (zero-fill), which PTX defines as reading only
src-size bytes, but initcheck reported those as 16-byte reads of memory past the
matrix, and empty chunks still passed a global address. Boundary chunks now:
whole -> cp.async; empty -> a 16-byte zero store to smem, no global access;
partial -> only the valid halves are loaded, the rest stored as exact zeros.
Interior CTAs are unchanged (unpredicated cp.async).

**NaN-poison fuzzer mode** (`--poison`, ctest `fuzz_poison`, and the nightly
target): every element outside the logical matrices (ld padding, batch gaps,
the tail) is NaN, plus 64-element NaN guard regions around each allocation; C
is all NaN when beta == 0. 2,000 cases (seed 21): 0 failures. Sensitivity
check: with a deliberate over-read reintroduced (partial chunks reading all 8
halves), poison mode fails 5 of the 2,000 cases while the normal mode passes
all 2,000 — poison mode catches exactly this class of bug.

**Instructions at 4096³** (ncu, `mma_pipe_128x256x32_2x4_s3`, row-major NN):
184.8 M before Phase A, 106.0 M after (5.51 -> 3.16 per HMMA). HMMA count is
unchanged (33.55 M); the removed work is per-load predicates and address math
on the interior path: IMAD 22.6 -> 8.0 M, LEA 10.1 -> 2.7 M, ISETP 9.7 -> 2.1 M,
LDS 9.4 -> 1.6 M, MOV 9.7 M -> below the top 14, BSSY/BSYNC gone. NOPs remain
(29.4 -> 29.9 M).

### OOM incident (2026-10-14 15:08 / 15:11) and the guards

The interrupted tool calls were the kernel OOM killer: `journalctl -k` shows
NVRM `NV_ERR_NO_MEMORY` allocation failures, a global OOM, and `claude`
(oom_score_adj 200) killed in the gnome-terminal scope, plus two
`bench_gemm`s. Both times the running job was `compute-sanitizer --tool
synccheck --num-cuda-barriers N` around `bench_gemm --check`; raising the
barrier limit makes the sanitizer allocate tracking state, which on GB10's
unified memory comes out of host RAM. The barrier-table overflow it was meant
to fix came from cuBLAS kernels (cuda::barrier), not ours. Fixes:
- CLAUDE.md rule 12: never `--num-cuda-barriers`.
- synccheck never runs cuBLAS: `bench_gemm --check none`, `aegis_fuzz
  --cpu-ref-only`, cuBLAS-using tests excluded (Benchmark policy, Sanitizer scope).
- `tools/gpu_job.sh` wraps every long GPU job (sanitizers, tune, sweep,
  paired-ab, nightly fuzz, the memcheck ctests): oom_score_adj 1000 + systemd
  scope MemoryMax (AEGIS_MEM_MAX, 24G). Measured limit of the cap: GB10's driver
  does not charge cudaMalloc to the job's cgroup (3 GiB cudaMalloc + memset
  succeeds under MemoryMax=1G), so MemoryMax caps host memory only; the
  OOM-priority setting is what protects the session from GPU allocations.
- sanitize.sh: one tool per invocation, never nested, JOBS 1 (racecheck,
  initcheck, synccheck) / 2 (memcheck).
- Pre-run health check (nvidia-smi, orphans, new Xid/NVRM) in the Benchmark policy.

### NOPs after HMMA (input to Phase B a)

Our 4096³ kernel issues a NOP after almost every HMMA (29.9 M NOPs / 33.55 M
HMMA). Compile-only experiments (scratch kernels with the same fragments and
MMA loop): consecutive HMMAs already target different accumulators, and loop
order (mi-outer, ni-outer, serpentine) does not change the count; the same
source has ~0 NOPs for sm_80 / sm_87 / sm_90 / sm_100 / sm_110 but ~0.83 per
HMMA for sm_120 / sm_121 (incl. the a variants); with independent
instructions between the HMMAs, ptxas uses them and the NOPs mostly vanish.
cuBLASLt's nvjet kernel issues HMMA.16816.F32 back to back on sm_121, so this
is **a property of our code shape under ptxas for sm_12x** (nothing
independent to schedule between consecutive HMMAs), not an absolute ptxas rule.
Candidate: fragment double buffering (load k-step s+1's ldmatrix fragments
while step s's HMMAs issue): in the scratch kernel NOPs 365 -> 155 per 448 HMMA
(-58%) at 168 -> 206 registers. Not yet measured in the real kernel.

### 5. compute-sanitizer: all four tools clean on our kernels

`tools/sanitize.sh`, one tool per invocation, under `tools/gpu_job.sh`, with a
health check (nvidia-smi, orphans, Xid/NVRM) before each:

| Tool | Tests | JOBS | Reports | Scope notes |
|---|---|---|---|---|
| memcheck | 131 | 2 | 0 | whole suite (GEMM tests incl. cuBLAS --check, fuzz 100 cases <= 512, C ABI, dispatch) |
| racecheck | 131 | 1 | 0 hazards | same |
| initcheck | 131 | 1 | 0 | same; ran after the boundary-load fix (the first version's report is gone) |
| synccheck | 128 | 1 | 0 | our kernels only: `bench_gemm --check none`, `aegis_fuzz --cpu-ref-only`, capi/dispatch tests excluded; no `--num-cuda-barriers` |

Skipped by every tool: `sm87_gate` and `gpu_budget_*` (no kernels; the budget
tests need their ctest environment, and the default-mode test makes no CUDA
call, which compute-sanitizer reports as "terminated before first
instrumented API call") and the `memcheck_*` ctests (they already run
memcheck; nesting is not supported). racecheck ran on the build before the GPU
budget wrapper; the kernel SASS of that build and the final one is identical
(cuobjdump diff without paths: 0 lines). Also: the launchers check grid limits
before launching (an oversized grid returns cudaErrorInvalidValue without a
failed CUDA call, which memcheck reported), and ctest gains `memcheck_fuzz`,
`memcheck_capi`, `memcheck_bench_odd`, `memcheck_bench_scalar`.

**In-process GPU budget** (rule 13, `src/bench/gpu_budget.h`): every device
allocation in the harness, aegis_tune, the fuzzer, the references, test_matmul
and the microbenchmarks goes through `aegis::budget::gpu_malloc`, which refuses
past `AEGIS_GPU_BUDGET` (16 GiB) or below `AEGIS_GPU_MIN_FREE` (16 GiB) free per
`cudaMemGetInfo` (111.7 GiB reported vs `free` 112 GiB: it measures host RAM).
ctest `gpu_budget_*`: 17 GiB refused by default; 64 MiB budget: 32 MiB ok,
+48 MiB refused, ok after the free, 1 TiB refused; min-free floor above RAM:
1 MiB refused.

### 6. 4096³ before vs after Phase A (paired, across builds)

`aegis-perf paired-ab` (new: alternating processes of two builds, ratio B/A per
pair, 10,000-resample bootstrap of the median), `mma_pipe_128x256x32_2x4_s3`,
4096³ row-major NN, cold, 30 pairs x 10 windows; A = baseline worktree
(`e8d7e6b`, before Phase A), B = after (`c5b8f7d`, now `b6d2b81` after the 2026-10-15 history rewrite). Idle guard 1.4–2.0%
(desktop up, remote desktop off).

| | Instructions (ncu) | Instructions / HMMA | Median (µs, cold v2) |
|---|---|---|---|
| before Phase A | 184.8 M | 5.51 | 2317.4 |
| after Phase A | 106.0 M | 3.16 | 2292.3 |
| B / A | 0.57 | | **0.995 [0.973, 1.014]**, tie |

**No regression** (CI upper bound +1.4% < 2%), and **no speedup** either: 43%
fewer instructions did not move the time, so the removed integer/predicate
instructions were not the binding limit at 4096³ (issue slots were not
saturated: issue active ~18% before). The per-shape re-tune vs. the previous
winners follows with Phase B, on the final kernels.

## 2026-10-15 — Phase B a: fragment double buffering (negative, reverted)

On the hardened kernel, `compute_tile` loaded k-step s+1's ldmatrix fragments
before issuing step s's MMAs (tier 4 only; on tier 3 it spilled
`mma_128x64x32_2x2` with column-major A on sm_121 and sm_87).

- **SASS, `mma_pipe_128x256x32_2x4_s3` NN:** NOPs 171 per 192 HMMA before and
  after (static count, unchanged); registers 188 -> 206; no spills. ptxas
  does interleave the next step's 8 LDSMs into the current step's 32 HMMAs,
  but 8 independent instructions cannot fill ~31 gaps, and fragments of the
  next BK tile cannot be loaded early (a barrier separates the tiles). The
  scratch kernel's -58% came from cross-iteration work it had to interleave;
  the real kernel has none within a tile.
- **Paired vs. the Phase A build** (alternating processes, 30 pairs, cold;
  idle guard read 2.9% before the run): 4096³ B/A 0.999 [0.983, 1.016] (tie);
  2048³ (`mma_pipe_128x128x32_2x4_s2`) B/A 1.007 [1.002, 1.014] (0.7% slower).
  ~~SUSPECT~~ (marked 2026-10-15, cleared the same day): the idle reading,
  2.9%, was above the 2% threshold. The re-run on an idle GPU (below,
  `paired_results` ids 1 and 2) reproduces both results.
- **Verdict:** no NOP reduction, more registers, no speedup -> reverted (kernel
  SASS identical to Phase A again). The NOPs between back-to-back HMMAs need
  either more independent work per k-step (larger k-steps, or moving global
  -> smem copy issue into the MMA stream, as TMA would) or nothing at all: with
  the tensor pipe ~50% busy and issue active ~18%, they may not be the binding
  limit. Not pursued further in Phase B.

## 2026-10-15 — Phase B b: XOR swizzle replaces padding; s4 at 128x256 / 256x128

`SmemOperand<ROWS, COLS>::at(row, col)` (`mma_common.cuh`) swizzles 16-byte
chunks instead of padding rows by 8 halves: rows of >= 8 chunks put chunk c at
c ^ (row & 7) within each group of 8; rows of 4 chunks (k-contiguous BK = 32)
pack two rows per 128-byte line and XOR the position with the line index. The
term repeats every 8 (16) rows, so `OperandPlan` and `compute_tile` keep
compile-time offsets. A 128x256x32 stage shrinks from 27,136 to 24,576 B, which
fits 4 stages in sm_121's 101,376 B: new configs `mma_pipe_128x256x32_2x4_s4`
and `mma_pipe_256x128x32_4x2_s4` (98,304 B). Config JSON now carries
`"smem":"xor"`, so every tier-3/4 kernel_id changed.

- **Correctness:** ctest 149/149 (fuzz_2000, fuzz_poison, sm87_gate).
  Registers: 128x256 s3 188 -> 204, s4 206, 256x128 s4 206, 128x128 s2 124;
  no spills.
- **Bank conflicts (ncu):** ldmatrix (LDSM) conflicts 0 in every profiled
  config; the standard ld/st bank-conflict counters are 0/0 for the three 4096³
  configs (128x128 s3 @ 2048³: 3,009 ld, against 18.9 M instructions). The
  remaining excess shared wavefronts are all on LDGSTS (cp.async smem writes)
  of the A tile: 128x256 s3 @ 4096³ 4.19 M excess of 16.78 M (padding: 25.17 M
  of 37.75 M). That is exactly 4 extra per A-tile LDGSTS (8 instead of 4);
  B-tile LDGSTS are ideal. Each A instruction writes 8 rows of 64 B and the
  swizzle is a permutation inside each 128-byte line, so the addresses do not
  conflict. Hypothesis (unverified): LDGSTS writes shared memory per
  global row segment (one wavefront per 64 B row), which only a BK = 64 tile
  (128 B rows) would avoid.
- **Tune** (`shapes/large.txt`, run `92ea023af8b5`; `shapes/default.txt`, run
  `41cc8ffb03fd`; idle guard 0.0%). Instructions from ncu; HMMA = MNK / 2048.

| Shape | Winner (Phase A -> B b) | Instr / HMMA | Tensor busy | lt / ours (paired) | % cublaslt_best |
|---|---|---|---|---|---|
| 2048³ | 128x128 s2 -> **128x128 s3** | 4.50 | 57.2% | 0.833 [0.830, 0.836] | 71.1% -> 83.6% |
| 4096³ | 128x256 s3 -> **256x128 s4** | 3.72 (Phase A 3.16) | 64.0% (Phase A 53.0%) | 0.711 [0.705, 0.715] | 63.6% -> 71.0% |

  At 4096³ the default-set run's unpaired medians put 128x128 s3 first, but its
  paired test gave runner-up / fastest [0.994, 0.998], so `paired_swap` picked
  256x128 s4, matching run `92ea023af8b5` (256x128 s4 over 128x128 s3, [1.009,
  1.033]). 128x256 s4 loses to s3 (ncu 2.536 vs 2.145 ms); not investigated.
- **Swizzle alone, same impl across builds** (`paired-ab`, A = Phase A build,
  30 pairs, cold): 128x256 s3 @ 4096³ B/A 0.944 [0.916, 0.959]; 128x128 s3 @
  2048³ 0.740 [0.738, 0.743]; 128x128 s2 @ 2048³ 0.892 [0.889, 0.898];
  64x64 s4 @ 1024³ 0.687 [0.684, 0.689], @ 1x896x896 0.968 [0.965, 0.969],
  @ 1x896x4864 0.972 [0.969, 0.974], @ 16x896x4864 0.986 [0.983, 0.988]. Faster
  everywhere measured. Lower decode % of cublaslt_best vs. the previous header
  (1x896x4864 77.8 -> 74.9) is the cuBLASLt side across runs, not ours.
- **Per-shape winners vs. previous header:** 1024³ 128x128 s2 -> 128x64 s2
  (75.2 -> 98.8%); 512x896x896 and 512x896x4864 128x64 s3 -> 64x64 s4 (74.8 ->
  90.1%, 63.8 -> 82.6%); 512x4864x896 128x128 s2 -> 128x64 s2 (67.1 -> 79.3%);
  256³ 91.8 -> 98.2%; 512³ 79.7 -> 87.5%; M = 1/16 winners unchanged
  (64x64 s4).
- **Tool fix:** `latest_per_kernel_shape` keyed rows by kernel_id, so the
  pre-swizzle rows of the same impl names (old kernel_ids) still competed: the
  first emit-header showed stale numbers and "TIED with" an empty partner (an
  impl tied with its own old row). Now keyed by impl: the newest measurement of
  a config supersedes older code versions.

## 2026-10-15 — History rewrite: build-dev/ stripped

`e799a4f` (Phase A 5) had committed `build-dev/` (415 files, ~184 MB of build
output); `c5b8f7d` untracked it. Backup first: `git clone --mirror --no-local`
to `~/backups/aegis-core-2026-10-14.git` (52.6 MiB pack, fsck clean). Then
`git filter-repo --invert-paths --path build-dev/ --refs e799a4f^..main`
(v2.47.0) and a force-push (lease on `6dacc55`). The partial range is
deliberate: a whole-history run also strips the SSH commit signatures and so
re-hashes all 72 commits, which would orphan every `runs.git_sha` in the perf
DB and every hash cited here. With the range, everything before `e799a4f`
keeps its hash and signature; the 7 rewritten commits (old -> new):

| old | new | commit |
|---|---|---|
| e799a4f | 796e02a | Phase A 5 |
| c5b8f7d | b6d2b81 | untrack build-dev/ |
| ad8d0f4 | 2c6aabb | LOG: 4096³ before/after Phase A |
| 56d0fe4 | b0bedd5 | aegis-perf library API |
| 7fe7a84 | 4220a8c | LOG: Phase B a |
| 6dacc55 | 16d33ec | Phase B b |
| 0edacf2 | c776ae4 | pre-commit 5 MB check |

Machine-readable full SHAs: `results/sha-remap.txt` (gitignored, next to the
DB; the DB's two runs at `7fe7a84` are the only DB rows affected). Pack 52.6
MiB -> 330 KiB. A fresh clone from GitHub (397 KiB) builds and passes ctest
149/149. Guards: pre-commit rejects any staged file over 5 MB; `.gitignore`
covers `build*/`, `cmake-build-*/`, `out/` and object, library and profile files.

## 2026-10-15 — Paused here (Core)

Core work stops here. State on sm_121 (winners header from runs `92ea023af8b5`
and `41cc8ffb03fd`; paired, cold, v2), against the Production bar:

| Shape | Winner | % cublaslt_best | Bar | Met |
|---|---|---|---|---|
| 1024³ | mma_pipe_128x64x32_2x2_s2 | 98.8% | >= 90% | yes |
| 2048³ | mma_pipe_128x128x32_2x4_s3 | 83.6% | >= 90% | no |
| 4096³ | mma_pipe_256x128x32_4x2_s4 | 71.0% | >= 90% | no |
| 1x896x896 | mma_pipe_64x64x32_2x2_s4 | 84.0% | >= 100% (decode) | no |
| 1x4864x896 | mma_pipe_64x64x32_2x2_s4 | 99.0% | >= 100% | no |
| 1x896x4864 | mma_pipe_64x64x32_2x2_s4 | 74.9% | >= 100% | no |
| 16x896x896 | mma_pipe_64x64x32_2x2_s4 | 96.2% | >= 100% | no |
| 16x4864x896 | mma_pipe_64x64x32_2x2_s4 | 99.4% | >= 100% | no |
| 16x896x4864 | mma_pipe_64x64x32_2x2_s4 | 72.7% | >= 100% | no |
| 256³ | mma_pipe_64x64x32_2x2_s4 | 98.2% | >= 80% | yes |
| 512³ | mma_pipe_64x64x32_2x2_s4 | 87.5% | >= 80% | yes |
| 512x896x896 | mma_pipe_64x64x32_2x2_s4 | 90.1% | >= 80% | yes |
| 512x896x4864 | mma_pipe_64x64x32_2x2_s4 | 82.6% | >= 80% | yes |
| 512x4864x896 | mma_pipe_128x64x32_2x2_s2 | 79.3% | >= 80% | no |

Below the 80% floor: 4096³, 1x896x4864, 16x896x4864, 512x4864x896. The
"dispatch to cuBLASLt below 80%" rule is not implemented: `aegis::matmul`
runs any tuned winner and falls back to plain cuBLAS only for untuned shapes.

**Done.**
- Phase A (hardening) complete:
  - GemmParams API and ABI 1.2: layouts, ld, alpha/beta, batched.
  - Exact predicated boundary loads with an interior fast path (4096³ tie,
    no regression).
  - K-scaled tolerance, c = 2.
  - `aegis_fuzz`: 2,000 cases in ctest and 50,000 nightly, 0 failures;
    NaN-poison over-read test.
  - Clean under memcheck, racecheck, initcheck and synccheck.
  - GPU budget and OOM guards.
- Phase B:
  - (a) Fragment double buffering: **rejected**. NOPs unchanged, +18
    registers, 0.7% slower at 2048³.
  - (b) XOR swizzle: **kept**. Fits s4 at 128x256 / 256x128; 2048³ 71.1 ->
    83.6%, 4096³ 63.6 -> 71.0%.
- Tooling: `aegis-perf paired-ab`; winners keyed by impl, so stale kernel_ids
  are ignored.

**Next, in order.**
- a. **BK = 64 config**, to test the 8-wavefront cp.async hypothesis (A-tile
  LDGSTS take 8 shared wavefronts instead of 4, one per 64 B row?). Cheap: new
  config rows, ncu, tune.
- b. **TMA loads** for the A/B tiles (`cp.async.bulk.tensor`, sm_121 only; PTX
  ISA citation in CLAUDE.md). The shared kernel keeps cp.async for sm_87.
- c. **Stream-K or persistent scheduling** for 2048³ wave quantization (2.67
  waves today).
- d. **Phase C: dedicated GEMV / split-K for decode** (M <= 16). This is the
  path to 100%; mma tiles won't get there. 1x896x4864 and 16x896x4864 sit at
  73–75%.
- e. **Power-controls probe** (deferred). Needs a passwordless sudoers entry
  for `nvidia-smi` and `cpupower`, which is not in place.
- f. **Locked-clock pointer-chase re-run (headless, nightly or text console).**
  Run `e3d94274b31d` was disturbed by Firefox and is invalidated; the arch
  file still uses the unlocked run `a825f4e557a3`.

## 2026-10-15 — Phase B a re-run (prepared, then run by hand)

The Phase B a paired-ab result (commit 4220a8c) is marked **suspect**: the
idle guard read 2.9% before it (limit 2%), and paired-ab had no blocking
guard then. paired-ab now has one and stores every result in the perf DB
(`paired_results`). The re-run is prepared but not run: the guard read 3.0%
(max 6%, GPU in P0) with Xorg, gnome-shell and Firefox on the GPU and the
remote-desktop daemon up.

Builds, in the worktree `~/Documents/projects/aegis-phaseBa-rerun` (detached
at b0bedd5, the Phase A kernel code the original A/B used):
- A, `bin-A/bench_gemm`: unmodified. Its `mma_pipe_128x256x32_2x4_s3`
  instruction stream is identical to the saved Phase A binary
  (`aegis-phaseA-bin/bench_gemm`): 192 HMMA, 171 NOP, 188 registers.
- B, `build/bench_gemm`: the fragment double-buffering patch, recovered
  verbatim from the session that ran it (`phaseBa-dbuf.patch` in the worktree):
  192 HMMA, 171 NOP, 206 registers, the same as the original B.

Command (from the Core repo root, idle GPU, remote desktop off):

    W=~/Documents/projects/aegis-phaseBa-rerun
    rust/target/release/aegis-perf paired-ab --a $W/bin-A/bench_gemm --b $W/build/bench_gemm \
      --impl mma_pipe_128x256x32_2x4_s3 --shape 4096 4096 4096 --pairs 30 --reps 10 \
      --sha-a b0bedd58fbf9 --sha-b b0bedd58fbf9-dirty
    rust/target/release/aegis-perf paired-ab --a $W/bin-A/bench_gemm --b $W/build/bench_gemm \
      --impl mma_pipe_128x128x32_2x4_s2 --shape 2048 2048 2048 --pairs 30 --reps 20 \
      --sha-a b0bedd58fbf9 --sha-b b0bedd58fbf9-dirty

(`-dirty` = b0bedd5 plus `phaseBa-dbuf.patch`.) The double-buffering verdict
(no NOP reduction, +18 registers) rests on SASS and stands; only the timing
ratios await the re-run.

**Re-run done (2026-10-15, by hand, the commands above).** The guard passed
both times; results are stored in the perf DB `paired_results`:

| id | Shape | impl | idle | B/A | 95% CI | Call |
|---|---|---|---|---|---|---|
| 1 | 4096³ | mma_pipe_128x256x32_2x4_s3 | 0.0% | 0.9971 | [0.9929, 1.0117] | tie |
| 2 | 2048³ | mma_pipe_128x128x32_2x4_s2 | 1.5% | 1.0074 | [1.0049, 1.0118] | B 0.7% slower |

Same as the suspect run (4096³ 0.999 tie; 2048³ 1.007 [1.002, 1.014]). The
SUSPECT mark on 4220a8c is cleared, and the rejection of fragment double
buffering stands: a tie at 4096³, 0.7% slower at 2048³, and 18 more
registers (188 -> 206).

## 2026-10-15 — Clock locking works on GB10; locked-clock pointer chase

`sudo nvidia-smi -lgc 2418,2418` succeeds on GB10, although `nvidia-smi -q`
lists "Supported Clocks: N/A". Without root, NVML's SetGpuLockedClocks
answers "Insufficient Permissions", so the probes record the lock as not
held. Policy from now on: latency probes run with clocks locked at 2418 MHz
(the default applications clock), via `microbench/run_latency_locked.sh 2418`.

Locked run `e3d94274b31d` (`run_latency_locked.sh 2418`, idle 0.0% NVML,
109 rows; the 145 rows of `a825f4e557a3` became `replaced`). SM clock
measured 2416 / 2381 MHz before / after, mean 2398.16, within the 2%
tolerance, so `clocks_locked = 1`. Values as `emit-arch` derives them (cycles,
median across points):

| | Locked `e3d94274b31d` | Unlocked `a825f4e557a3` (committed sm_121.json) |
|---|---|---|
| L2 latency, 256 KiB – 8 MiB, 12 points | 490.0 cycles (range 336.0 – 1067.8) | 335.4 cycles (335.4 – 335.4) |
| DRAM latency, >= 32 MiB, 10 points | 1088.7 cycles (range 932.5 – 1401.5) | 956.6 cycles (952.1 – 964.8) |
| SM clock | 2398.16 MHz, locked | 2406.78 MHz, unlocked |
| L1 hit (<= 64 KiB) | 37.0 cycles | 37.0 cycles |

**The locked run is not adopted: it was disturbed.** Clock locking cannot
spread an L2 hit from 336 to 1068 cycles. The L2 plateau was flat at 335.4
in the unlocked run and in the earlier runs (see "Pointer chase rerun"). In
the locked run, p10 and median also disagree within single points (1 MiB
cross_page: p10 367, median 428), which points to interference during the
run. The journal shows the cause: Firefox restarted at 18:37:39 and opened
the GPU (`pci id for fd 17: 10de:2e12`). That was after `nvidia-smi -lgc`
(18:37:33) and the idle check, and about 10 s before the run row's
`started_at`, 18:37:50. The idle guard samples only before the run, so it
could not see this. L1 points (37.0 cycles) are identical in both runs.

The `sm_121.json` that `emit-arch` wrote from this run is left uncommitted.
The committed one, from `a825f4e557a3`, stays. Next: re-run with
Firefox and the desktop session idle (over SSH), then emit-arch and commit
the file. To fall back meanwhile, invalidate the run:
`aegis-perf invalidate --run e3d94274b31d --probe ptr_chase_latency --reason "..."`
(emit-arch then uses the newest replaced run, `a825f4e557a3`).

**Done (2026-10-15):** `e3d94274b31d` is invalidated (108 ptr_chase rows,
reason "Firefox started on the GPU mid-run"), and the 144 rows of
`a825f4e557a3` it had replaced are valid again. `emit-arch --arch 121` now
selects `a825f4e557a3`. Its output equals the committed `sm_121.json` except
for the `generated_by` timestamp, so the file is unchanged. The stash
holding the disturbed file was dropped. The re-run is queued (item f).

## 2026-10-15 — Perf DB schema_version; aegis-perf 0.2.0

The perf DB gets a `schema_version` table (one row). The current schema is
v2: microbench status and clock columns, the runs idle reading, and
`paired_results`. A DB without the table is v1. Every aegis-perf reader
checks it before any query: `connect_ro` refuses any other version, and
`connect` migrates older DBs and refuses newer ones without touching them.
Both fail with "DB schema vN, binary expects vM" instead of an SQL error.
The live DB is stamped v2 (backup:
`results/aegis_perf.db.bak-2026-10-14-pre-schema-version`). Binaries built
before this change do not check the version; on a v2 DB they fail on the
missing `superseded` column.

## 2026-10-15 — Idle guard during the run; nightly job; schema v3

**Idle guard during the run.** `bench_gemm --idle-watch --root <pid>` keeps
sampling while a sweep, tune, paired-ab or microbench run executes. Every
2 s it sums the SM utilization that NVML reports per process
(`nvmlDeviceGetProcessUtilization`) over GPU processes outside the run's own
process tree. Each window above the limit (2%) is logged on stderr. Any such
window marks the run `disturbed = 1`, along with every row of it
(measurements, paired, microbench, paired_results), plus a detail line naming
the busiest process. Disturbed rows are excluded like `--allow-busy` rows.
On GB10 the per-process numbers and the device total disagree in both
directions. A cuBLAS 4096³ loop reads 92% on the device but 16–20% on its own
process. The idle desktop reads ~1% on the device, while the per-process sum
is ~5% (Xorg 3.9%). So with a desktop session up, most runs will be marked
disturbed.

The first version filtered samples by wall-clock time and failed in the
pre-commit ctest. NVML on GB10 ignores `lastSeenTimeStamp`, and its sample
timestamps are not a clock: they jumped from 3.4e9 to 5.6e10 within seconds,
and one read near 2^64. Wall-clock filtering had worked only by accident. The
watcher now polls every 250 ms, ignores timestamps, and averages each process
over a 2 s window's polls; it passed 4 of 4 runs, alone and next to two fuzz
processes. ctest
`idle_watch_paths` checks that the run's own load is excluded while a watcher
rooted at pid 1 counts it, and that a forced limit marks every window.

**Perf DB schema v3** (the `disturbed` columns; additive). The live DB was
migrated after a backup (`results/aegis_perf.db.bak-2026-10-14-pre-v3`); all
older rows are `disturbed = NULL` (not watched). aegis-perf is 0.3.0. Not
tagged: aegis-perf-v0.2.0 binaries refuse this DB with
"DB schema v3, binary expects v2".

**Nightly job** (`tools/nightly.sh`, user timer `aegis-nightly.timer` at 03:00,
installed with `tools/nightly.sh --install`). Steps:
1. Take the GPU lock and run the preflight and the idle guard. Busy or locked:
   one line in `results/nightly.log`, exit 0.
2. 50,000-case fuzz, seeded with the date.
3. Regression sweep of `shapes/default.txt`: the impls of the committed
   winners header plus cublas and cublaslt_best, with `--no-winners`, ingested.
4. `aegis-perf regress` compares each shape's % of cublaslt_best with the
   newest earlier clean measurement; a relative drop over 3% is flagged.
5. Back up the DB to `~/backups/aegis_perf-<date>.db`, keeping the 14 newest.
6. Write one summary line.

The nightly job never touches the winners table, the winners headers or the
arch files. The user manager has `Linger=no`, so the timer fires only while
the user is logged in.

**First run (manual, 2026-10-14 20:15, `systemctl --user start
aegis-nightly.service`):** the guard read 0.9%.
- **Fuzz: 1 failure in 50,000** (1003 s, seed 20261014). Case 24632:
  `mma_pipe_128x128x32_2x4_s3`, the dispatched winner at 2048³.
  - The case: 1951x624x3159, CR (column-major A, row-major B), lda=1956
    ldb=625 ldc=635, alpha=-0.0127, beta=-1, B offset 2 (misaligned).
  - Element (871, 532): got 1.00684, cuBLASLt reference 1.00977, allowed
    0.00265 (about 10% over).
  - It reproduces with `build/aegis_fuzz --seed 20261014 --case 24632
    --poison` (0.4 s).
  - `--cpu-ref-only` generates a different case for that number (not
    misaligned), so it doesn't tell whether our kernel or the cuBLASLt
    reference is off. **Open.**
- **Sweep: skipped.** By the time the fuzz finished, the GPU was busy (2.8%
  mean, max 8%) and the sweep's own idle guard refused. `nightly.sh` now logs
  that as "sweep skipped: GPU busy" instead of FAIL. `aegis-perf regress` was
  checked on existing tune runs instead (`41cc8ffb03fd`: 14/14 shapes, min
  71.0%; it flags 1x896x4864 77.8 -> 74.9% against an older tune run).
- The backup and the summary line work: `~/backups/aegis_perf-2026-10-14.db`,
  integrity ok.

## 2026-10-15 — Fuzz case 24632: a cuBLASLt reference error, not ours

The first nightly fuzz failed case 24632 (seed 20261014) against the cuBLASLt
reference. The case:
- Kernel: `mma_pipe_128x128x32_2x4_s3`, the dispatched winner at 2048³.
- Shape 1951x624x3159; A column-major, B row-major (CR); lda 1956, ldb 625,
  ldc 635; B pointer offset by 2 elements (misaligned); batch 1.
- alpha = -0.0126576, beta = -1. Data: A in ±0.25, B in ±1, C0 in ±1.
- K-scaled tolerance (c = 2): 0.001324 = 0.000347 (the alpha*A*B term) +
  0.000977 (eps * |beta| * max|C0|). Against cuBLASLt it is doubled, 0.002648.

**`--cpu-ref-only` now generates the identical case.** The generator no longer
depends on the reference mode: it dropped the 2^24 MAC cap, and aegis::matmul
stays in the target list; that mode skips its cases instead of removing it,
which had shifted every target index. `--cpu-ref-only` now checks every case
against FP64, whatever its size. The FP64 CPU reference transposes B once and
splits rows across up to 16 threads, with the same per-element summation
order, so results are unchanged: 3.85e9 MACs in 0.4 s. ctest
`fuzz_modes_same_cases` diffs the case lists of both modes.

**FP64 evidence** (`aegis_fuzz --seed 20261014 --case 24632 --poison
--explain`; ulp = FP16 ulp at the reference, 2^-10 near 1.0):

| Element | FP64 | Ours (err) | cuBLASLt (err) |
|---|---|---|---|
| row 871 col 532 (the failure) | 1.00725778 | 1.00683594 (0.43 ulp, 0.32x tol) | 1.00976562 (**2.57 ulp, 1.89x tol**) |
| ours, worst element (row 360 col 122) | 1.04345747 | 0.50 ulp (0.37x tol) | 0.50 ulp |
| mean abs error, 1,217,424 elements | | 8.4e-5 | 1.9e-4 |

**Verdict: a reference bug.** Our result is correctly rounded everywhere: the
worst element is 0.50 ulp, the rounding limit. cuBLASLt's result is 2.57 ulp
off FP64 at the failing element, beyond the 1x tolerance on its own. The
"failure" was the difference between the two. At that element, alpha*A*B is
small (0.011) and beta*C0 is near 1 (0.996). The tolerance is right for this
case class (ours used 0.37 of it) and is unchanged.

**Fix.** A case that fails against cuBLASLt is now re-checked against the FP64
CPU reference at **1x** tolerance, which is stricter than the 2x that
cuBLASLt gets, and FP64 decides. Each overrule is printed ("note case ...
cuBLASLt reference error") and counted in the summary. ctest
`fuzz_lt_reference_error` pins case 24632. The `--corrupt` self-test is
unaffected (it skips the re-check): 200/200 still caught.

**50,000-case rerun** (seed 20261014, `--poison`, under `tools/gpu_job.sh`):
**0 failures**. 37,333 cases used the FP64 reference and 12,667 cuBLASLt;
exactly one cuBLASLt failure was overruled by FP64 (case 24632). 995 s.

Note for `tools/sanitize.sh` (synccheck with `--cpu-ref-only --max-dim 512`):
cases are no longer capped at 2^24 MACs, so they reach up to 512³ x 4
batches; expect a longer synccheck run.

## 2026-10-15 — In-run watcher: device-level foreign share, compute contexts

The first watcher summed foreign per-process SM samples. On GB10 those are on
another scale than the device total, so an idle desktop read ~5% summed
(Xorg 3.9%) while the device read ~1%. That would have marked nearly every
row disturbed.

**Why "device minus ours" alone does not work.** Per-process samples lag.
Measured with a cuBLAS 4096³ warm load, polled every 250 ms: the device read
96% throughout, while the load's own per-process value ramped 59 -> 82%
over 3 s. Device minus ours would count the lag as foreign use (~37% at the
start of every run process; a sweep starts a new process per measurement).

**Rule now.**
- The watcher takes a 2 s baseline before the run's GPU work (callers wait
  for its ready line): device D0 and foreign per-process sum F0, giving the
  scale k = D0 / F0 (D0 floored at 0.5%, k at most 1, k = 1 when F0 = 0).
- Per 2 s window: foreign = min(device - ours, k * foreign per-process). Device
  minus ours is the device-level use not attributed to the run. The scaled
  term caps it: a desktop idling as at baseline counts as its baseline
  device-level share (~1%), not 5%.
- Any foreign process holding a compute context (NVML
  `nvmlDeviceGetComputeRunningProcesses`) flags the window, whatever its
  utilization.
- Over: foreign > 2%, or a foreign compute context.

**Tests.**
- ctest `idle_rule` (new) checks the arithmetic on these measured readings.
  The idle desktop (D0 1, F0 5, k 0.2) gives 1.0%: clean, where the old rule
  read 5%. The same desktop during our 96% load (ours 59) gives 1.0%: clean.
  Foreign per-process 40, during our load or not, gives 8%: disturbed. Also
  the k edge cases.
- ctest `idle_watch_paths` (kept, extended):
  - Our own load is excluded, and its compute context is not foreign.
  - A watcher rooted at pid 1 sees that load's compute context as foreign and
    reports the run disturbed.
  - The ready line comes first.
  - A forced limit marks every window, and a missing --root is refused.
  - It runs RUN_SERIAL: in the hook's `ctest -j4`, a parallel fuzz test was
    (correctly) seen as a foreign compute context.
- Live: the desktop was fully idle when I checked (no per-process samples at
  all), so the ~5%-summed case could not be reproduced live. Two 8 s
  watches read 0.0%, clean.
