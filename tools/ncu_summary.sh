#!/usr/bin/env bash
# Key metrics from tools/ncu_gemm.sh reports, one column per report (the last
# profiled kernel in each: for cublaslt_best, the chosen algorithm).
#
#   tools/ncu_summary.sh artifacts/ncu/*.ncu-rep
set -euo pipefail
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
METRICS=(
  gpu__time_duration.sum
  launch__grid_size launch__block_size launch__registers_per_thread
  launch__shared_mem_per_block_dynamic launch__shared_mem_per_block_static
  launch__occupancy_limit_registers launch__occupancy_limit_shared_mem launch__occupancy_limit_warps
  launch__waves_per_multiprocessor
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_active
  sm__issue_active.avg.pct_of_peak_sustained_elapsed
  smsp__inst_executed.sum
  l1tex__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  lts__t_sectors.sum lts__t_sector_hit_rate.pct
  lts__t_sectors.avg.pct_of_peak_sustained_elapsed
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
  smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
  smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
  smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
  smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
)
printf "%-70s" "metric"
for rep in "$@"; do printf " %16s" "$(basename "$rep" .ncu-rep | sed 's/mma_pipe_//; s/cublaslt_best/lt_best/' | cut -c1-16)"; done
echo
declare -A val
for rep in "$@"; do
  csv="$("$NCU" --import "$rep" --page raw --csv 2>/dev/null)"
  for m in "${METRICS[@]} Kernel Name"; do :; done
  name=$(awk -v FPAT='("[^"]*")|([^,]*)' 'NR==1{for(i=1;i<=NF;i++) if($i=="\"Kernel Name\"") c=i} NR>2{v=$c} END{gsub(/"/,"",v); print v}' <<<"$csv")
  val["$rep|kernel"]="$(cut -c1-16 <<<"$name")"
  for m in "${METRICS[@]}"; do
    val["$rep|$m"]=$(awk -v FPAT='("[^"]*")|([^,]*)' -v m="\"$m\"" 'NR==1{for(i=1;i<=NF;i++) if($i==m) c=i} NR>2 && c{v=$c} END{gsub(/"/,"",v); print (v==""?"-":v)}' <<<"$csv")
  done
done
printf "%-70s" "kernel"; for rep in "$@"; do printf " %16s" "${val[$rep|kernel]}"; done; echo
for m in "${METRICS[@]}"; do
  printf "%-70s" "$m"
  for rep in "$@"; do printf " %16s" "${val[$rep|$m]}"; done
  echo
done
