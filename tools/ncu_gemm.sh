#!/usr/bin/env bash
# Nsight Compute profile of one GEMM launch (full section set), for diagnosis.
#
#   tools/ncu_gemm.sh IMPL M N K            -> artifacts/ncu/<impl>_<M>x<N>x<K>.ncu-rep
#
# Runs bench_gemm --protocol 1 --reps 1 --warmup 0 (one untimed run, one timed
# launch) under ncu. Our kernels: the timed launch is profiled. cublaslt_best:
# bench_gemm runs every heuristic algorithm, then re-runs the fastest one last;
# every non-harness kernel is profiled and the LAST one is the chosen
# algorithm (tools/ncu_gemm.sh prints its name). ncu defaults apply: clocks
# locked to base (--clock-control base) and caches flushed before each replay
# (cold). Needs GPU counter access (RmProfilingAdminOnly: 0).
set -euo pipefail
IMPL="${1:?impl}" M="${2:?M}" N="${3:?N}" K="${4:?K}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
OUT="$ROOT/artifacts/ncu"
mkdir -p "$OUT"
REP="$OUT/${IMPL}_${M}x${N}x${K}"
EXTRA=()
if [[ "$IMPL" == cublaslt_best ]]; then
  # LT_RANK=R profiles the algorithm at heuristic rank R only: selecting the
  # fastest under ncu (serialized replays, base clocks) can pick a different one.
  [[ -n "${LT_RANK:-}" ]] && EXTRA=(--lt-rank "$LT_RANK")
  FILTER=(-k 'regex:^(?!.*spin_kernel).*$')
else
  FILTER=(-k 'regex:(gemm_.*_kernel|gemv_.*kernel)' --launch-skip 1 --launch-count 1)
fi
"$ROOT/tools/profiler_session.sh" "$NCU" --set full "${FILTER[@]}" -f -o "$REP" \
  "$ROOT/build/bench_gemm" --impl "$IMPL" --m "$M" --n "$N" --k "$K" --protocol 1 --reps 1 --warmup 0 "${EXTRA[@]}" >/dev/null
echo "$REP.ncu-rep"
