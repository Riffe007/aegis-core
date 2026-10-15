#!/usr/bin/env bash
# Locked-clock re-run of the L2 / DRAM pointer-chase latency probe.
#
#   sudo nvidia-smi -lgc MHZ,MHZ        # lock the GPU clocks (needs root)
#   microbench/run_latency_locked.sh MHZ
#   sudo nvidia-smi -rgc                # restore the default clocks
#
# Runs as the normal user (only the clock lock needs root). Takes the GPU lock
# (rule 15) and runs the blocking idle guard. mb_ptr_chase verifies the measured
# SM clock against MHZ before and after (clock_lock.cuh); if any latency row
# was not at locked clocks, nothing is ingested (exit 4) and the JSONL is kept
# for inspection. Otherwise the rows are ingested (the older ptr_chase run
# becomes "replaced") and the next steps are printed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -n "${AEGIS_GPU_LOCK_HELD:-}" ]] || exec "$ROOT/tools/gpu_lock.sh" "$0" "$@"
[[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]] || { echo "usage: run_latency_locked.sh MHZ   (after sudo nvidia-smi -lgc MHZ,MHZ)" >&2; exit 2; }
MHZ="$1"
HOST="$(hostname)"
OUT="$ROOT/results/ptr_chase-locked-$HOST-$(date +%Y%m%d-%H%M%S).jsonl"
SASS="$ROOT/artifacts/microbench"
MANIFEST="$ROOT/results/manifest-$HOST.json"
PERF="$ROOT/rust/target/release/aegis-perf"
mkdir -p "$ROOT/results" "$SASS"
[[ -x "$PERF" ]] || { echo "missing $PERF; cargo build --release --manifest-path rust/Cargo.toml" >&2; exit 1; }

IDLE_ARGS=()
[[ "${ALLOW_BUSY:-0}" == 1 ]] && IDLE_ARGS+=(--allow-busy)
IDLE_JSON="$("$ROOT/build/bench_gemm" --idle-check "${IDLE_ARGS[@]}")" || {
  echo "run_latency_locked.sh: idle guard refused (GPU busy)" >&2
  exit 3
}
INGEST_IDLE=(--idle-source "$(jq -r .source <<<"$IDLE_JSON")")
[[ "$(jq -r .ok <<<"$IDLE_JSON")" == true ]] && INGEST_IDLE+=(--idle-util-pct "$(jq -r .mean_pct <<<"$IDLE_JSON")")
[[ "$(jq -r .busy <<<"$IDLE_JSON")" == true ]] && INGEST_IDLE+=(--allow-busy)
[[ -f "$MANIFEST" ]] || "$ROOT/build/aegis_manifest" > "$MANIFEST"
source "$ROOT/tools/idle_watch.sh"
idle_watch_start "$ROOT/build/bench_gemm" "$OUT.watch.json"  # the idle guard during the probe

AEGIS_CLOCKS_LOCKED_MHZ="$MHZ" "$ROOT/build/microbench/mb_ptr_chase" > "$OUT" 2> "$SASS/ptr_chase.sass.txt" || {
  echo "mb_ptr_chase failed (see $SASS/ptr_chase.sass.txt)" >&2
  exit 1
}
idle_watch_stop
unlocked=$(jq -s '[.[] | select(.probe == "ptr_chase_latency" and .clocks_locked != true)] | length' "$OUT")
if [[ "$unlocked" -ne 0 ]]; then
  echo "clocks were not locked at $MHZ MHz for $unlocked rows: $(jq -rs '.[0].clock_lock' "$OUT")" >&2
  echo "nothing ingested; rows kept in $OUT" >&2
  exit 4
fi
(cd "$ROOT" && "$PERF" ingest-microbench "$OUT" --manifest "$MANIFEST" "${INGEST_IDLE[@]}" --watch "$OUT.watch.json")
echo "ingested $OUT (clocks locked at $MHZ MHz). Next: sudo nvidia-smi -rgc; then" >&2
echo "  rust/target/release/aegis-perf emit-arch --arch 121   # regenerates include/aegis/arch/sm_121.json" >&2
