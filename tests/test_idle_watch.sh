#!/usr/bin/env bash
# ctest idle_watch_paths: the idle guard during a run (bench_gemm --idle-watch).
#   tests/test_idle_watch.sh BENCH_GEMM
# 1. A GPU load inside the run's process tree is ours: a watcher rooted at this
#    shell excludes its samples and does not count its compute context; a
#    watcher rooted at pid 1 (nothing is ours) counts both and reports the run
#    disturbed by a foreign compute context (AEGIS_IDLE_WATCH_TRACE=1 lists it).
# 2. The watcher prints a ready line after its baseline, before any window.
# 3. AEGIS_IDLE_MAX_PCT=-1 puts every window over the limit: disturbed.
# 4. --idle-watch without --root is refused.
set -uo pipefail
BENCH="$1"
fail() { echo "FAIL: $*" >&2; exit 1; }
field() { tail -1 <<<"$2" | sed -n "s/.*\"$1\":\([^,}]*\).*/\1/p"; }
d="$(mktemp -d)"; trap 'rm -rf "$d"' EXIT

"$BENCH" --impl cublas --m 4096 --n 4096 --k 4096 --reps 3000 --warmup 1 --cache warm >/dev/null 2>&1 &
load_pid=$!
sleep 2
( (sleep 6) | AEGIS_IDLE_WATCH_TRACE=1 "$BENCH" --idle-watch --root $$ > "$d/ours" 2> "$d/ours.trace" ) &
ours_pid=$!
(sleep 6) | AEGIS_IDLE_WATCH_TRACE=1 "$BENCH" --idle-watch --root 1 > "$d/all" 2> "$d/all.trace"
wait $ours_pid
wait $load_pid || fail "load (bench_gemm) failed"
ours="$(cat "$d/ours")"; all="$(cat "$d/all")"
echo "root=\$\$: $(tail -1 <<<"$ours")"
echo "root=1:  $(tail -1 <<<"$all")"
head -1 <<<"$ours" | grep -q '"ready":true' || fail "no ready line first: $ours"
[[ "$(field ok "$ours")" == true ]] || fail "no report: $ours"

grep -q "pid $load_pid .*sm .*counted" "$d/all.trace" || fail "watcher rooted at pid 1 did not count the load (pid $load_pid)"
grep -q "pid $load_pid holds a compute context (foreign)" "$d/all.trace" || fail "watcher rooted at pid 1 missed the load's compute context"
[[ "$(field foreign_compute "$all")" == true && "$(field disturbed "$all")" == true ]] ||
  fail "a foreign compute context did not make the run disturbed: $all"

grep -q "pid $load_pid .*counted" "$d/ours.trace" && fail "watcher rooted at this shell counted the run's own load"
grep -q "pid $load_pid holds a compute context" "$d/ours.trace" && fail "the run's own compute context counted as foreign"
grep -q "pid $load_pid .*excluded" "$d/ours.trace" || fail "watcher rooted at this shell never saw the load"
[[ "$(field foreign_compute "$ours")" == false ]] || fail "foreign compute context reported for our own load: $ours"

out="$( (sleep 4) | AEGIS_IDLE_MAX_PCT=-1 "$BENCH" --idle-watch --root $$ 2>/dev/null)"
[[ "$(field disturbed "$out")" == true ]] || fail "limit -1 not disturbed: $out"
[[ "$(field windows "$out")" == "$(field windows_over "$out")" ]] || fail "limit -1: not every window over: $out"

"$BENCH" --idle-watch </dev/null >/dev/null 2>&1; rc=$?
[[ $rc -eq 2 ]] || fail "--idle-watch without --root exited $rc, want 2"
echo "idle watch paths ok"
