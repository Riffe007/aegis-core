#!/usr/bin/env bash
# Profile the ldmatrix x4 probe at 8 and 16 warps/SM with Nsight Compute:
# shared-memory bank conflicts / wavefronts, LSU pipe utilization, occupancy,
# warp stall reasons. Needs GPU performance-counter access (root, or
# NVreg_RestrictProfilingToAdminUsers=0):
#
#   sudo microbench/profile_ldmatrix.sh
#
# Output: artifacts/microbench/ldmatrix_ncu_w{8,16}.csv (ncu --csv --page raw).
# The first launch of each run is the warm-up, so it is skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
OUT="$ROOT/artifacts/microbench"
mkdir -p "$OUT"
METRICS=(
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
  smsp__inst_executed_op_ldsm.sum
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
  l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__cycles_elapsed.avg
  smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__issue_active.avg.pct_of_peak_sustained_active
)
for w in 8 16; do
  "$NCU" -k mb_ldmatrix_x4 --launch-skip 1 --launch-count 1 \
    --metrics "$(IFS=,; echo "${METRICS[*]}")" --csv --page raw \
    "$ROOT/build/microbench/mb_ldmatrix" --warps "$w" > "$OUT/ldmatrix_ncu_w$w.csv"
  echo "wrote $OUT/ldmatrix_ncu_w$w.csv"
done
chmod a+r "$OUT"/ldmatrix_ncu_w*.csv
