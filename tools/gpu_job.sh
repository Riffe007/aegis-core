#!/usr/bin/env bash
# Run one long GPU job so that a runaway cannot take the machine down
# (Benchmark policy).
#
#   tools/gpu_job.sh CMD [ARGS...]        host-memory cap: AEGIS_MEM_MAX (default 24G)
#
# GB10 memory is unified: GPU allocations (the driver's buffers, cudaMalloc,
# compute-sanitizer's tracking state) come out of host RAM. On 2026-10-14 a
# compute-sanitizer run drove the machine into a global OOM and the kernel
# killed the interactive session (claude, oom_score_adj 200). Two guards:
#   1. oom_score_adj = 1000 (choom): in any OOM the job is the first victim,
#      and its GPU memory is freed when it dies. This is the guard that covers
#      GPU allocations.
#   2. a transient systemd --user scope with MemoryMax=AEGIS_MEM_MAX and no
#      swap: caps the job's ordinary host memory. It does NOT cap cudaMalloc /
#      driver allocations, which GB10's driver does not charge to the job's
#      cgroup (checked: a 3 GiB cudaMalloc + memset succeeds under MemoryMax=1G).
# Exit status: the job's.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: gpu_job.sh CMD [ARGS...]" >&2; exit 2; }
CAP="${AEGIS_MEM_MAX:-24G}"
for tool in systemd-run choom; do
  command -v "$tool" >/dev/null || { echo "gpu_job.sh: $tool not found; refusing to run an unguarded GPU job" >&2; exit 2; }
done
# The GPU lock (rule 15): taken here unless the caller already holds it.
if [[ -z "${AEGIS_GPU_LOCK_HELD:-}" ]]; then
  exec "$(dirname "$0")/gpu_lock.sh" "$0" "$@"
fi
exec systemd-run --user --scope --quiet --collect -p MemoryMax="$CAP" -p MemorySwapMax=0 -- \
  choom -n 1000 -- "$@"
