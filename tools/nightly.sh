#!/usr/bin/env bash
# Nightly job (systemd user timer aegis-nightly.timer, 03:00; tools/systemd/).
#
#   tools/nightly.sh             run now (what the timer runs)
#   tools/nightly.sh --install   install and enable the user timer
#
# 1. Takes the GPU lock (rule 15) and runs the preflight and the idle guard
#    (Benchmark policy). Lock held, GPU busy or preflight failed: one line in
#    results/nightly.log saying why, exit 0 (a skipped night is not an error).
# 2. aegis_fuzz, 50,000 cases (seed = the date, so every night differs),
#    under tools/gpu_job.sh and the in-process GPU budget (rule 13).
# 3. Regression sweep of the tracked shape set (shapes/default.txt): the
#    impls the committed winners header dispatches, plus cublaslt_best and
#    cublas, through aegis-perf sweep --no-winners (idle guard before, idle
#    watch during; ingested into the perf DB; the winners table, winners
#    headers and arch files are not touched). aegis-perf regress compares each
#    shape's % of cublaslt_best with the newest earlier clean measurement.
# 4. Backs up the perf DB to ~/backups/aegis_perf-<date>.db (sqlite .backup),
#    keeping the 14 newest.
# 5. One summary line in results/nightly.log; everything else in
#    results/nightly/<date>.log.
# Exit status: 0 on a skipped or clean night, 1 if a step failed (fuzz
# failures, sweep correctness failures, ingest or backup errors).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--install" ]]; then
  dir="$HOME/.config/systemd/user"
  mkdir -p "$dir"
  mkdir -p "$ROOT/results/nightly"
  cp tools/systemd/aegis-nightly.service tools/systemd/aegis-nightly.timer "$dir/"
  date "+%F %T" > "$ROOT/results/nightly/.last-start"  # kernel-log errors count from here
  systemctl --user daemon-reload
  systemctl --user enable --now aegis-nightly.timer
  systemctl --user list-timers aegis-nightly.timer --no-pager
  exit 0
fi

DAY="$(date +%F)"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="$ROOT/results/nightly.log"
mkdir -p "$ROOT/results/nightly"
DETAIL="$ROOT/results/nightly/$STAMP.log"
PERF="$ROOT/rust/target/release/aegis-perf"
BENCH="$ROOT/build/bench_gemm"
FUZZ="$ROOT/build/aegis_fuzz"
HEADER="$ROOT/include/aegis/winners/sm_121.h"
BACKUPS="$HOME/backups"
KEEP=14

summary() { echo "$(date '+%F %T') $*" >> "$SUMMARY"; }
exec >>"$DETAIL" 2>&1
echo "== nightly $STAMP, $(git -C "$ROOT" describe --always --dirty 2>/dev/null) on $(git -C "$ROOT" branch --show-current 2>/dev/null)"

# 1. GPU lock (the same flock as tools/gpu_lock.sh), never waited for.
LOCK="${AEGIS_GPU_LOCK:-$ROOT/results/gpu.lock}"
if [[ -z "${AEGIS_GPU_LOCK_HELD:-}" ]]; then
  exec 9>>"$LOCK"
  if ! flock -x -n 9; then
    summary "skipped: GPU lock held by: $(head -c 200 "$LOCK" 2>/dev/null | tr '\n' ' ')"
    exit 0
  fi
  truncate -s 0 "$LOCK"
  printf '%s pid %d (%s): %s\n' "$(date -Is)" $$ "$(hostname)" "tools/nightly.sh" >&9
  export AEGIS_GPU_LOCK_HELD=$$
  trap 'truncate -s 0 "$LOCK"' EXIT
fi

# Preflight (Benchmark policy "Before any run").
for f in "$PERF" "$BENCH" "$FUZZ" "$HEADER"; do
  [[ -e "$f" ]] || { summary "skipped: missing $f"; exit 0; }
done
nvidia-smi -L >/dev/null 2>&1 || { summary "skipped: nvidia-smi does not answer"; exit 0; }
orphans="$(pgrep -a -x 'bench_gemm|aegis_fuzz|aegis_tune|compute-sanitiz.*' | head -3 | tr '\n' ';')"
[[ -z "$orphans" ]] || { summary "skipped: GPU processes already running: $orphans"; exit 0; }
# New Xid / NVRM lines since the previous nightly start (or the install).
since="$(cat "$ROOT/results/nightly/.last-start" 2>/dev/null || echo '-24h')"
date "+%F %T" > "$ROOT/results/nightly/.last-start"
xid="$(journalctl -k --since "$since" --no-pager -q 2>/dev/null | grep -E 'Xid|NVRM' | tail -1)"
[[ -z "$xid" ]] || { summary "skipped: new GPU errors in the kernel log since $since: $xid"; exit 0; }

# Idle guard: busy -> skip, logging the reading.
if ! idle="$("$BENCH" --idle-check 2>"$DETAIL.idle")"; then
  cat "$DETAIL.idle"
  summary "skipped: GPU busy: $(grep -m1 'idle guard:' "$DETAIL.idle" | sed 's/^idle guard: //')"
  rm -f "$DETAIL.idle"
  exit 0
fi
cat "$DETAIL.idle"; rm -f "$DETAIL.idle"
idle_pct="$(jq -r .mean_pct <<<"$idle")"
failed=0

# 2. Fuzz.
seed="$(date +%Y%m%d)"
echo "== aegis_fuzz --cases 50000 --seed $seed --poison"
"$ROOT/tools/gpu_job.sh" "$FUZZ" --cases 50000 --seed "$seed" --poison > "$ROOT/results/nightly/fuzz-$STAMP.txt"
fuzz_rc=$?
fuzz_line="$(grep '^aegis_fuzz:' "$ROOT/results/nightly/fuzz-$STAMP.txt" | tail -1)"
echo "$fuzz_line (exit $fuzz_rc)"
fuzz_fail="$(sed -n 's/^aegis_fuzz: [0-9]* cases, \([0-9]*\) failures.*/\1/p' <<<"$fuzz_line")"
if [[ $fuzz_rc -ne 0 || -z "$fuzz_fail" ]]; then
  fuzz_txt="fuzz FAILED (exit $fuzz_rc, ${fuzz_fail:-?} failures, seed $seed)"
  failed=1
else
  fuzz_txt="fuzz 50000 cases, $fuzz_fail failures (seed $seed)"
  [[ "$fuzz_fail" == 0 ]] || failed=1
fi

# 3. Regression sweep: the dispatched impls of the committed header + baselines.
mapfile -t impls < <(sed -n 's/^ *{[0-9]*, *[0-9]*, *[0-9]*, *"\([^"]*\)"[,}].*/\1/p' "$HEADER" | sort -u)
echo "== aegis-perf sweep --shapes shapes/default.txt --no-winners --impls cublas cublaslt_best ${impls[*]}"
sweep_out="$("$PERF" sweep --shapes shapes/default.txt --no-winners --impls cublas cublaslt_best "${impls[@]}" 2>&1)"
sweep_rc=$?
echo "$sweep_out"
run_id="$(sed -n 's/^results: .* (run \([0-9a-f]*\))$/\1/p' <<<"$sweep_out" | tail -1)"
if [[ -n "$run_id" ]]; then
  regress_out="$("$PERF" regress --run "$run_id" --header "$HEADER" 2>&1)" || failed=1
  echo "$regress_out"
  sweep_txt="sweep $(tail -1 <<<"$regress_out")"
  [[ $sweep_rc -eq 0 ]] || { sweep_txt+=" (sweep exit $sweep_rc: correctness failures)"; failed=1; }
elif grep -q 'idle guard refused' <<<"$sweep_out"; then
  # The GPU went busy after the fuzz: skipped, not failed (same rule as step 1).
  sweep_txt="sweep skipped: GPU busy: $(grep -m1 '^idle guard: GPU utilization' <<<"$sweep_out" | sed 's/^idle guard: //')"
else
  sweep_txt="sweep FAILED (exit $sweep_rc, nothing ingested): $(grep -m1 -iE 'error' <<<"$sweep_out")"
  failed=1
fi

# 4. Backup (consistent copy of a live DB), keep the 14 newest.
mkdir -p "$BACKUPS"
backup="$BACKUPS/aegis_perf-$DAY.db"
if sqlite3 "$ROOT/results/aegis_perf.db" ".backup '$backup'" &&
   [[ "$(sqlite3 "$backup" 'PRAGMA integrity_check')" == ok ]]; then
  ls -1t "$BACKUPS"/aegis_perf-*.db | tail -n +$((KEEP + 1)) | xargs -r rm -f --
  backup_txt="backup $backup ($(ls -1 "$BACKUPS"/aegis_perf-*.db | wc -l) kept)"
else
  backup_txt="backup FAILED ($backup)"
  failed=1
fi

# 5. One line.
summary "$( ((failed)) && echo FAIL || echo ok ): idle ${idle_pct}%; $fuzz_txt; $sweep_txt; $backup_txt; log $DETAIL"
exit $failed
