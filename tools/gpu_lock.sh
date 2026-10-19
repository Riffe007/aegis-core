#!/usr/bin/env bash
# Run one GPU job under the exclusive GPU lock (CLAUDE.md rule 15).
#
#   tools/gpu_lock.sh [--wait SECS] CMD [ARGS...]
#
# Every GPU job (tune, sweep, paired-ab, microbench, sanitizers, nightly fuzz,
# GPU ctests) holds an exclusive flock on results/gpu.lock (AEGIS_GPU_LOCK
# overrides) for its whole run, so two jobs never share the GPU and a
# benchmark is never contaminated by another session's tests. The holder is
# written into the lock file (time, pid, command). Children see
# AEGIS_GPU_LOCK_HELD=<pid> and pass straight through (nested wrappers such as
# tools/gpu_job.sh, or aegis-perf, do not lock again).
#
# Profiler commands (ncu, nsys, ncu_gemm.sh or profile_*.sh anywhere in CMD)
# run through tools/profiler_session.sh, which logs the session window in
# results/profiler_sessions.log for the nightly kernel-log guard.
#
# Lock busy: exit 75 (EX_TEMPFAIL) after printing the holder; --wait SECS
# blocks up to SECS first. Otherwise the job's exit status.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="${AEGIS_GPU_LOCK:-$ROOT/results/gpu.lock}"
WAIT=""
if [[ "${1:-}" == "--wait" ]]; then WAIT="$2"; shift 2; fi
[[ $# -ge 1 ]] || { echo "usage: gpu_lock.sh [--wait SECS] CMD [ARGS...]" >&2; exit 2; }
# A profiler run (ncu, nsys, or a wrapper of them) logs its session window
# (tools/profiler_session.sh) for the nightly kernel-log guard, nested or not.
is_profiler=0
for a in "$@"; do
  case "$(basename -- "$a")" in ncu|nsys|ncu_gemm.sh|profile_*.sh) is_profiler=1 ;; esac
done
RUN=("$@")
((is_profiler)) && RUN=("$ROOT/tools/profiler_session.sh" "$@")
if [[ -n "${AEGIS_GPU_LOCK_HELD:-}" ]]; then exec "${RUN[@]}"; fi

mkdir -p "$(dirname "$LOCK")"
exec 9>>"$LOCK"
if [[ -n "$WAIT" ]]; then flock -x -w "$WAIT" 9; else flock -x -n 9; fi || {
  echo "gpu_lock: the GPU lock ($LOCK) is held by: $(head -c 300 "$LOCK" 2>/dev/null || echo unknown)" >&2
  exit 75
}
truncate -s 0 "$LOCK"
printf '%s pid %d (%s): %s\n' "$(date -Is)" $$ "$(hostname)" "$*" >&9
export AEGIS_GPU_LOCK_HELD=$$
"${RUN[@]}"
rc=$?
truncate -s 0 "$LOCK"
exit $rc
