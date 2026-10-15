#!/usr/bin/env bash
# ctest idle_watch_paths: the idle guard during a run (bench_gemm --idle-watch).
#   tests/test_idle_watch.sh BENCH_GEMM
# 1. A GPU load inside the run's process tree is not "other" use: a watcher
#    rooted at this shell excludes its samples, a watcher rooted at pid 1
#    (nothing excluded) counts them (AEGIS_IDLE_WATCH_TRACE=1 lists both).
# 2. AEGIS_IDLE_MAX_PCT=-1 puts every window over the limit: disturbed.
# 3. --idle-watch without --root is refused.
set -uo pipefail
BENCH="$1"
fail() { echo "FAIL: $*" >&2; exit 1; }
field() { sed -n "s/.*\"$1\":\([^,}]*\).*/\1/p" <<<"$2"; }
d="$(mktemp -d)"; trap 'rm -rf "$d"' EXIT

"$BENCH" --impl cublas --m 4096 --n 4096 --k 4096 --reps 3000 --warmup 1 --cache warm >/dev/null 2>&1 &
load_pid=$!
sleep 2
( (sleep 4) | AEGIS_IDLE_WATCH_TRACE=1 "$BENCH" --idle-watch --root $$ > "$d/ours" 2> "$d/ours.trace" ) &
ours_pid=$!
(sleep 4) | AEGIS_IDLE_WATCH_TRACE=1 "$BENCH" --idle-watch --root 1 > "$d/all" 2> "$d/all.trace"
wait $ours_pid
wait $load_pid || fail "load (bench_gemm) failed"
grep "pid $load_pid " "$d/ours.trace" | head -2
grep -q "^idle watch trace: bench_gemm pid $load_pid .*counted" "$d/all.trace" ||
  fail "watcher rooted at pid 1 did not count the load (pid $load_pid): $(cat "$d/all")"
grep -q "pid $load_pid .*counted" "$d/ours.trace" && fail "watcher rooted at this shell counted the run's own load"
grep -q "pid $load_pid .*excluded" "$d/ours.trace" || fail "watcher rooted at this shell never saw the load"
[[ "$(field ok "$(cat "$d/ours")")" == true ]] || fail "no report: $(cat "$d/ours")"

out="$( (sleep 3) | AEGIS_IDLE_MAX_PCT=-1 "$BENCH" --idle-watch --root $$ 2>/dev/null)"
[[ "$(field disturbed "$out")" == true ]] || fail "limit -1 not disturbed: $out"
[[ "$(field windows "$out")" == "$(field windows_over "$out")" ]] || fail "limit -1: not every window over: $out"

"$BENCH" --idle-watch </dev/null >/dev/null 2>&1; rc=$?
[[ $rc -eq 2 ]] || fail "--idle-watch without --root exited $rc, want 2"
echo "idle watch paths ok"
