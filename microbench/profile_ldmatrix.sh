#!/usr/bin/env bash
# Profile the ldmatrix x4 probe at 8 and 16 warps/SM with Nsight Compute:
# shared-memory bank conflicts / wavefronts, LSU pipe utilization, occupancy,
# warp stall reasons, per-SM cycle spread. Needs GPU performance-counter
# access (NVreg_RestrictProfilingToAdminUsers=0, or root).
#
#   microbench/profile_ldmatrix.sh [RUNS]     (default 5 runs per warp count)
#
# --clock-control none: ncu would otherwise lock clocks to base, which could
# hide a clock-related effect. The first launch of each run is the warm-up and
# is skipped. ncu replays the kernel over several passes, so the probe's own
# B/clk line in a CSV may come from a different pass than the counters.
# Output: artifacts/microbench/ldmatrix_ncu_w{8,16}_r<i>.csv (--csv --page raw,
# preceded by the probe's JSON lines).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
OUT="$ROOT/artifacts/microbench"
RUNS="${1:-5}"
mkdir -p "$OUT"
METRICS=(
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
  smsp__inst_executed_op_ldsm.sum
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
  l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__cycles_elapsed.avg
  sm__cycles_elapsed.max
  sm__cycles_elapsed.min
  sm__cycles_active.avg
  smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__issue_active.avg.pct_of_peak_sustained_active
)
for w in 8 16; do
  for r in $(seq 1 "$RUNS"); do
    "$NCU" --clock-control none -k mb_ldmatrix_x4 --launch-skip 1 --launch-count 1 \
      --metrics "$(IFS=,; echo "${METRICS[*]}")" --csv --page raw \
      "$ROOT/build/microbench/mb_ldmatrix" --warps "$w" > "$OUT/ldmatrix_ncu_w${w}_r$r.csv"
    echo "wrote $OUT/ldmatrix_ncu_w${w}_r$r.csv"
  done
done
chmod a+r "$OUT"/ldmatrix_ncu_w*.csv
