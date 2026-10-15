#!/usr/bin/env bash
# Run every microbench probe, collect JSON lines, keep each probe's SASS proof,
# and ingest into the perf DB microbench table.
#
#   microbench/run_all.sh            # from the repo root, after cmake --build build
#
# Output: results/microbench-<host>-<ts>.jsonl, artifacts/microbench/<probe>.sass.txt
# Exits nonzero if any probe fails, including a failed SASS proof (exit 3).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The whole run holds the GPU lock (rule 15).
[[ -n "${AEGIS_GPU_LOCK_HELD:-}" ]] || exec "$ROOT/tools/gpu_lock.sh" "$0" "$@"
BIN="$ROOT/build/microbench"
HOST="$(hostname)"
OUT="$ROOT/results/microbench-$HOST-$(date +%Y%m%d-%H%M%S).jsonl"
SASS="$ROOT/artifacts/microbench"
MANIFEST="$ROOT/results/manifest-$HOST.json"
mkdir -p "$ROOT/results" "$SASS"

# Blocking idle guard (Benchmark policy): refuse a busy GPU unless ALLOW_BUSY=1;
# the reading is stored on the run row.
IDLE_ARGS=()
[[ "${ALLOW_BUSY:-0}" == 1 ]] && IDLE_ARGS+=(--allow-busy)
IDLE_JSON="$("$ROOT/build/bench_gemm" --idle-check "${IDLE_ARGS[@]}")" || {
  echo "run_all.sh: idle guard refused (GPU busy); stop other GPU work or set ALLOW_BUSY=1 (rows are marked)" >&2
  exit 3
}
INGEST_IDLE=(--idle-source "$(jq -r .source <<<"$IDLE_JSON")")
[[ "$(jq -r .ok <<<"$IDLE_JSON")" == true ]] && INGEST_IDLE+=(--idle-util-pct "$(jq -r .mean_pct <<<"$IDLE_JSON")")
[[ "$(jq -r .busy <<<"$IDLE_JSON")" == true ]] && INGEST_IDLE+=(--allow-busy)

# After the guard: aegis_manifest measures DRAM bandwidth.
[[ -f "$MANIFEST" ]] || "$ROOT/build/aegis_manifest" > "$MANIFEST"

# The idle guard keeps sampling during the probes (rows marked disturbed if
# other processes use the GPU above the limit mid-run).
source "$ROOT/tools/idle_watch.sh"
idle_watch_start "$ROOT/build/bench_gemm" "$OUT.watch.json"

status=0
for probe in ptr_chase mma_latency mma_throughput ldmatrix smem_banks cp_async dram_bw launch_overhead; do
  exe="$BIN/mb_$probe"
  [[ -x "$exe" ]] || { echo "missing $exe; build first" >&2; exit 1; }
  echo "== $probe" >&2
  if ! "$exe" >> "$OUT" 2> "$SASS/$probe.sass.txt"; then
    echo "   FAILED (see $SASS/$probe.sass.txt)" >&2
    status=1
  fi
  grep -E "^    check|FAILED" "$SASS/$probe.sass.txt" >&2 || true
done
idle_watch_stop

PERF="$ROOT/rust/target/release/aegis-perf"
[[ -x "$PERF" ]] || { echo "missing $PERF; cargo build --release --manifest-path rust/Cargo.toml" >&2; exit 1; }
(cd "$ROOT" && "$PERF" ingest-microbench "$OUT" --manifest "$MANIFEST" "${INGEST_IDLE[@]}" --watch "$OUT.watch.json")
echo "results: $OUT" >&2
exit $status
