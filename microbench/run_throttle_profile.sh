#!/usr/bin/env bash
# Record a time-to-throttle profile (aegis-hw step 7, sustained mode).
#
#   microbench/run_throttle_profile.sh [SECONDS]      (default 300)
#
# Takes the GPU lock (rule 15), runs the blocking idle guard and the in-run
# idle watch, then mb_throttle_profile: SECONDS of back-to-back aegis::matmul
# 4096^3 with telemetry every 100 ms. The probe rows go into the perf DB
# (microbench table, probe throttle_profile); the samples stay in
# results/throttle-<host>-<ts>.samples.jsonl; the profile (aegis-throttle/1) is
# written to include/aegis/arch/throttle/sm_<arch>.json (commit it; generated
# hardware-model file, CLAUDE.md Decisions) unless the run was disturbed.
# Reporting and sizing only: clocks and power limits are never touched.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -n "${AEGIS_GPU_LOCK_HELD:-}" ]] || exec "$ROOT/tools/gpu_lock.sh" "$0" "$@"
SECONDS_RUN="${1:-300}"
HOST="$(hostname)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/results/throttle-$HOST-$STAMP.jsonl"
SAMPLES="$ROOT/results/throttle-$HOST-$STAMP.samples.jsonl"
PROFILE="$ROOT/results/throttle-$HOST-$STAMP.profile.json"
MANIFEST="$ROOT/results/manifest-$HOST.json"
PERF="$ROOT/rust/target/release/aegis-perf"
[[ -x "$PERF" ]] || { echo "missing $PERF" >&2; exit 1; }

# Build identity (Benchmark policy): a dirty or unknown build is refused before
# the load runs unless ALLOW_DIRTY=1 (its rows are marked allow_dirty).
INGEST_BUILD=()
if [[ "${ALLOW_DIRTY:-0}" == 1 ]]; then
  INGEST_BUILD+=(--allow-dirty)
elif [[ "$(jq -r '.build_dirty == false and .build_sha != null' <<<"$("$ROOT/build/bench_gemm" --build-info 2>/dev/null || echo '{}')")" != true ]]; then
  echo "$(basename "$0"): refusing to run: build/ is from a dirty or unknown tree (bench_gemm --build-info); commit and rebuild, or set ALLOW_DIRTY=1 (rows are marked)" >&2
  exit 5
fi
IDLE_JSON="$("$ROOT/build/bench_gemm" --idle-check)" || { echo "run_throttle_profile.sh: idle guard refused (GPU busy)" >&2; exit 3; }
[[ -f "$MANIFEST" ]] || "$ROOT/build/aegis_manifest" > "$MANIFEST"
source "$ROOT/tools/idle_watch.sh"
idle_watch_start "$ROOT/build/bench_gemm" "$OUT.watch.json"
"$ROOT/build/microbench/mb_throttle_profile" --seconds "$SECONDS_RUN" --samples "$SAMPLES" --profile "$PROFILE" > "$OUT"
idle_watch_stop
"$PERF" ingest-microbench "$OUT" --manifest "$MANIFEST" --idle-source "$(jq -r .source <<<"$IDLE_JSON")" \
  --idle-util-pct "$(jq -r .mean_pct <<<"$IDLE_JSON")" "${INGEST_BUILD[@]}" --watch "$OUT.watch.json"
if [[ "$(tail -1 "$OUT.watch.json" | jq -r '.disturbed // false')" == true ]]; then
  echo "run_throttle_profile.sh: run disturbed; the profile is not written to include/aegis/arch/throttle" >&2
  exit 4
fi
ARCH="$(jq -r .sm_arch "$MANIFEST")"
DEST="$ROOT/include/aegis/arch/throttle/sm_$ARCH.json"
mkdir -p "$(dirname "$DEST")"
jq --arg gen "microbench/run_throttle_profile.sh on $HOST, $(date -u '+%Y-%m-%d %H:%M UTC')" \
   --arg src "$(basename "$OUT")" '. + {generated_by: $gen, source: $src}' "$PROFILE" > "$DEST"
echo "profile: $DEST" >&2
cat "$DEST"
