#!/usr/bin/env bash
# ctest fixture (gpu_lock_guard): every ctest labelled "gpu" requires it, so GPU
# tests run only while this ctest run holds the GPU lock (CLAUDE.md rule 15).
# Passes when AEGIS_GPU_LOCK_HELD is set and the lock file is in fact locked.
LOCK="${AEGIS_GPU_LOCK:-$(cd "$(dirname "$0")/.." && pwd)/results/gpu.lock}"
if [[ -n "${AEGIS_GPU_LOCK_HELD:-}" ]] && [[ -f "$LOCK" ]] && ! flock -n -s "$LOCK" true; then
  exit 0
fi
echo "GPU ctests need the GPU lock: run  tools/gpu_lock.sh ctest --test-dir build ...  (or ctest -LE gpu for the non-GPU tests)" >&2
exit 1
