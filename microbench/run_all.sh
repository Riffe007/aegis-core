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
BIN="$ROOT/build/microbench"
HOST="$(hostname)"
OUT="$ROOT/results/microbench-$HOST-$(date +%Y%m%d-%H%M%S).jsonl"
SASS="$ROOT/artifacts/microbench"
MANIFEST="$ROOT/results/manifest-$HOST.json"
mkdir -p "$ROOT/results" "$SASS"
[[ -f "$MANIFEST" ]] || "$ROOT/build/aegis_manifest" > "$MANIFEST"

status=0
for probe in ptr_chase mma_latency mma_throughput ldmatrix smem_banks cp_async dram_bw; do
  exe="$BIN/mb_$probe"
  [[ -x "$exe" ]] || { echo "missing $exe; build first" >&2; exit 1; }
  echo "== $probe" >&2
  if ! "$exe" >> "$OUT" 2> "$SASS/$probe.sass.txt"; then
    echo "   FAILED (see $SASS/$probe.sass.txt)" >&2
    status=1
  fi
  grep -E "^    check|FAILED" "$SASS/$probe.sass.txt" >&2 || true
done

PERF="$ROOT/rust/target/release/aegis-perf"
[[ -x "$PERF" ]] || { echo "missing $PERF; cargo build --release --manifest-path rust/Cargo.toml" >&2; exit 1; }
(cd "$ROOT" && "$PERF" ingest-microbench "$OUT" --manifest "$MANIFEST")
echo "results: $OUT" >&2
exit $status
