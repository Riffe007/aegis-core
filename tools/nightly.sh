#!/usr/bin/env bash
# Nightly job (systemd user timer aegis-nightly.timer, 03:00; tools/systemd/).
#
#   tools/nightly.sh             run now (what the timer runs)
#   tools/nightly.sh --install   install and enable the user timer
#   tools/nightly.sh --preflight-only   lock + kernel-log guard, then stop (status=PREFLIGHT_OK); for tests
#   tools/nightly.sh --build-only       ... + build main HEAD in the nightly worktree, check its identity (status=BUILD_OK)
#
# Paths (tests override them): AEGIS_NIGHTLY_SUMMARY (results/nightly.log),
# AEGIS_NIGHTLY_DIR (results/nightly), AEGIS_KERNEL_LOG (a short-unix kernel
# log file instead of journalctl), AEGIS_PROFILER_SESSIONS.
# AEGIS_NIGHTLY_TREE (../aegis-nightly) and AEGIS_NIGHTLY_BRANCH (main): what is
# built and measured; AEGIS_NIGHTLY_CMAKE_ARGS: extra configure arguments.
#
# 1. Takes the GPU lock (rule 15) and runs the preflight and the idle guard
#    (Benchmark policy). Lock held, GPU busy or preflight failed: one line in
#    results/nightly.log, "status=SKIPPED reason=...", exit 0 (a skipped night
#    is not an error, but it is never silent). Kernel log: any new Xid / NVRM
#    line blocks, except the profiler's known messages inside a logged
#    profiler session (tools/kernel_log_guard.sh).
# 1b. Builds the current HEAD of main, clean, in its own worktree (never the
#    main working tree's build/), and checks that bench_gemm --build-info names
#    exactly that commit with a clean tree; a failed build or a mismatch is a
#    FAIL night. Every step below runs that build, with that commit's shapes
#    and winners header; its runs are stored as kind "nightly".
# 2. aegis_fuzz, 50,000 cases (seed = the date, so every night differs),
#    under tools/gpu_job.sh and the in-process GPU budget (rule 13).
# 3. Regression sweep of the tracked shape set (shapes/default.txt): the
#    impls the committed winners header dispatches, plus cublaslt_best and
#    cublas, through aegis-perf sweep --no-winners (idle guard before, idle
#    watch during; ingested into the perf DB; the winners table, winners
#    headers and arch files are not touched). aegis-perf regress compares each
#    shape's % of cublaslt_best with the newest earlier clean measurement: a
#    drop over 3% is a "watch"; a regression only when two consecutive
#    nightlies or a paired-ab confirm it.
# 4. Backs up the perf DB to ~/backups/aegis_perf-<date>.db (sqlite .backup),
#    keeping the 14 newest.
# 5. One summary line in results/nightly.log (status=OK | FAIL | SKIPPED);
#    everything else in results/nightly/<date>.log.
# Exit status: 0 on a skipped or clean night, 1 if a step failed (fuzz
# failures, sweep correctness failures, ingest or backup errors).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NDIR="${AEGIS_NIGHTLY_DIR:-$ROOT/results/nightly}"
PREFLIGHT_ONLY=0
[[ "${1:-}" == "--preflight-only" ]] && PREFLIGHT_ONLY=1
BUILD_ONLY=0
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=1
if [[ "${1:-}" == "--install" ]]; then
  dir="$HOME/.config/systemd/user"
  mkdir -p "$dir"
  mkdir -p "$NDIR"
  cp tools/systemd/aegis-nightly.service tools/systemd/aegis-nightly.timer "$dir/"
  date "+%F %T" > "$NDIR/.last-start"  # kernel-log errors count from here
  systemctl --user daemon-reload
  systemctl --user enable --now aegis-nightly.timer
  systemctl --user list-timers aegis-nightly.timer --no-pager
  exit 0
fi

DAY="$(date +%F)"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${AEGIS_NIGHTLY_SUMMARY:-$ROOT/results/nightly.log}"
mkdir -p "$NDIR"
DETAIL="$NDIR/$STAMP.log"
BACKUPS="$HOME/backups"
KEEP=14

summary() { echo "$(date '+%F %T') $*" >> "$SUMMARY"; }
# A skipped night: loud in the summary and in the detail log, exit 0.
skip() {
  summary "status=SKIPPED reason=\"$*\""
  echo "== NIGHTLY SKIPPED: $*"
  exit 0
}
exec >>"$DETAIL" 2>&1
echo "== nightly $STAMP, $(git -C "$ROOT" describe --always --dirty 2>/dev/null) on $(git -C "$ROOT" branch --show-current 2>/dev/null)"

# 1. GPU lock (the same flock as tools/gpu_lock.sh), never waited for.
LOCK="${AEGIS_GPU_LOCK:-$ROOT/results/gpu.lock}"
if [[ -z "${AEGIS_GPU_LOCK_HELD:-}" ]]; then
  exec 9>>"$LOCK"
  if ! flock -x -n 9; then
    skip "GPU lock held by: $(head -c 200 "$LOCK" 2>/dev/null | tr '\n' ' ')"
  fi
  truncate -s 0 "$LOCK"
  printf '%s pid %d (%s): %s\n' "$(date -Is)" $$ "$(hostname)" "tools/nightly.sh" >&9
  export AEGIS_GPU_LOCK_HELD=$$
  trap 'truncate -s 0 "$LOCK"' EXIT
fi

# New Xid / NVRM lines since the previous nightly start (or the install); the
# profiler's own messages inside logged sessions do not count.
since="$(cat "$NDIR/.last-start" 2>/dev/null || echo '-24h')"
date "+%F %T" > "$NDIR/.last-start"
klog=(--since "$since")
[[ -n "${AEGIS_KERNEL_LOG:-}" ]] && klog=(--log "$AEGIS_KERNEL_LOG")
if ! guard="$("$ROOT/tools/kernel_log_guard.sh" "${klog[@]}" 2>&1)"; then
  skip "kernel log since $since: $guard"
fi
echo "kernel log since $since: $guard"
if ((PREFLIGHT_ONLY)); then
  summary "status=PREFLIGHT_OK kernel log: $guard"
  exit 0
fi

# Build: the current HEAD of main (AEGIS_NIGHTLY_BRANCH), clean, in the nightly's
# own worktree (AEGIS_NIGHTLY_TREE, default ../aegis-nightly). Never the main
# working tree's build/: a session may have left it at another commit, or built
# it from uncommitted work. Everything below runs from that tree: its bench_gemm,
# aegis_fuzz and aegis-perf, its shapes, its committed winners header.
NBRANCH="${AEGIS_NIGHTLY_BRANCH:-main}"
NTREE="${AEGIS_NIGHTLY_TREE:-$(dirname "$ROOT")/aegis-nightly}"
export PATH="$PATH:/usr/local/cuda/bin:$HOME/.cargo/bin"
head_sha="$(git -C "$ROOT" rev-parse --verify -q "$NBRANCH^{commit}")" || skip "no branch $NBRANCH"
if [[ ! -e "$NTREE/.git" ]]; then
  git -C "$ROOT" worktree add -q --detach "$NTREE" "$head_sha" || skip "cannot create the nightly worktree $NTREE"
fi
[[ -z "$(git -C "$NTREE" status --porcelain --untracked-files=no)" ]] ||
  skip "the nightly worktree $NTREE has local changes; it is the nightly's own: git -C $NTREE checkout -f --detach $NBRANCH"
git -C "$NTREE" checkout -q --detach "$head_sha" || skip "cannot check out $head_sha in $NTREE"
echo "== build $NBRANCH HEAD $head_sha in $NTREE (log $DETAIL.build)"
# shellcheck disable=SC2086  # AEGIS_NIGHTLY_CMAKE_ARGS: extra configure arguments, word-split on purpose
if ! { cmake -S "$NTREE" -B "$NTREE/build" -DCMAKE_BUILD_TYPE=Release ${AEGIS_NIGHTLY_CMAKE_ARGS:-} &&
       cmake --build "$NTREE/build" -j &&
       cargo build --release --manifest-path "$NTREE/rust/Cargo.toml" -p aegis-perf; } >"$DETAIL.build" 2>&1; then
  tail -20 "$DETAIL.build"
  summary "status=FAIL build of $NBRANCH $head_sha failed; log $DETAIL.build"
  exit 1
fi
PERF="$NTREE/rust/target/release/aegis-perf"
BENCH="$NTREE/build/bench_gemm"
FUZZ="$NTREE/build/aegis_fuzz"
HEADER="$NTREE/include/aegis/winners/sm_121.h"
# The binaries must name exactly that commit, clean (src/bench/build_info.h).
build_info="$("$BENCH" --build-info 2>/dev/null)"
if [[ "$(jq -r .build_sha <<<"$build_info" 2>/dev/null)" != "$head_sha" ||
      "$(jq -r .build_dirty <<<"$build_info" 2>/dev/null)" != false ]]; then
  summary "status=FAIL build identity: bench_gemm says ${build_info:-nothing}, expected $head_sha clean; log $DETAIL"
  echo "== build identity mismatch: $build_info"
  exit 1
fi
echo "== build: $build_info"
if ((BUILD_ONLY)); then
  summary "status=BUILD_OK $NBRANCH $head_sha clean in $NTREE"
  exit 0
fi
export AEGIS_PERFDB="${AEGIS_PERFDB:-$ROOT/results/aegis_perf.db}"
cd "$NTREE"

# Preflight (Benchmark policy "Before any run").
for f in "$PERF" "$BENCH" "$FUZZ" "$HEADER"; do
  [[ -e "$f" ]] || skip "missing $f"
done
nvidia-smi -L >/dev/null 2>&1 || skip "nvidia-smi does not answer"
orphans="$(pgrep -a -x 'bench_gemm|aegis_fuzz|aegis_tune|compute-sanitiz.*' | head -3 | tr '\n' ';')"
[[ -z "$orphans" ]] || skip "GPU processes already running: $orphans"

# Idle guard: busy -> skip, logging the reading.
if ! idle="$("$BENCH" --idle-check 2>"$DETAIL.idle")"; then
  cat "$DETAIL.idle"
  reason="GPU busy: $(grep -m1 'idle guard:' "$DETAIL.idle" | sed 's/^idle guard: //')"
  rm -f "$DETAIL.idle"
  skip "$reason"
fi
cat "$DETAIL.idle"; rm -f "$DETAIL.idle"
idle_pct="$(jq -r .mean_pct <<<"$idle")"
failed=0

# 2. Fuzz.
seed="$(date +%Y%m%d)"
echo "== aegis_fuzz --cases 50000 --seed $seed --poison"
"$NTREE/tools/gpu_job.sh" "$FUZZ" --cases 50000 --seed "$seed" --poison > "$NDIR/fuzz-$STAMP.txt"
fuzz_rc=$?
fuzz_line="$(grep '^aegis_fuzz:' "$NDIR/fuzz-$STAMP.txt" | tail -1)"
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
echo "== aegis-perf sweep --nightly --shapes shapes/default.txt --no-winners --impls cublas cublaslt_best aegis_matmul ${impls[*]}"
sweep_out="$("$PERF" sweep --nightly --build-dir "$NTREE/build" --results-dir "$ROOT/results" \
  --shapes shapes/default.txt --no-winners --impls cublas cublaslt_best aegis_matmul "${impls[@]}" 2>&1)"
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
if sqlite3 "$AEGIS_PERFDB" ".backup '$backup'" &&
   [[ "$(sqlite3 "$backup" 'PRAGMA integrity_check')" == ok ]]; then
  ls -1t "$BACKUPS"/aegis_perf-*.db | tail -n +$((KEEP + 1)) | xargs -r rm -f --
  backup_txt="backup $backup ($(ls -1 "$BACKUPS"/aegis_perf-*.db | wc -l) kept)"
else
  backup_txt="backup FAILED ($backup)"
  failed=1
fi

# 5. One line.
summary "status=$( ((failed)) && echo FAIL || echo OK ) build ${head_sha:0:12} ($NBRANCH, clean); idle ${idle_pct}%; $fuzz_txt; $sweep_txt; $backup_txt; log $DETAIL"
exit $failed
